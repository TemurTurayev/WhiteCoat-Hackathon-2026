"""
DGX Spark Training Script — Segmentation Boost
Trains with Copy-Paste Augmentation + SAM Optimizer + EMA
Goal: Improve IoU from 0.8832 baseline

Usage:
  python train_dgx.py --mode copypaste   # Copy-Paste augmentation
  python train_dgx.py --mode sam          # SAM optimizer
  python train_dgx.py --mode ema          # EMA weights
  python train_dgx.py --mode all          # All combined
"""
import argparse
import copy
import os
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import segmentation_models_pytorch as smp

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, copy_paste_pool=None):
        self.transform = transform
        self.copy_paste_pool = copy_paste_pool
        self.imgs = []
        self.masks = []
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() in {".png", ".jpg"}:
                mp = mask_dir / f"{p.stem}.png"
                if not mp.exists():
                    mp = mask_dir / f"{p.stem}.jpg"
                if mp.exists():
                    self.imgs.append(str(p))
                    self.masks.append(str(mp))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], 0)
        mask = (mask > 127).astype(np.float32)

        # Copy-Paste augmentation
        if self.copy_paste_pool is not None and random.random() < 0.5:
            src_idx = random.randint(0, len(self.copy_paste_pool) - 1)
            src_img_path, src_mask_path = self.copy_paste_pool[src_idx]
            src_img = cv2.cvtColor(cv2.imread(src_img_path), cv2.COLOR_BGR2RGB)
            src_mask = cv2.imread(src_mask_path, 0)
            src_mask = (src_mask > 127).astype(np.float32)

            # Resize source to match target
            h, w = img.shape[:2]
            src_img = cv2.resize(src_img, (w, h))
            src_mask = cv2.resize(src_mask, (w, h))

            # Paste: where source mask is 1, use source image
            paste_mask = (src_mask > 0.5).astype(np.float32)
            paste_mask_3d = paste_mask[:, :, np.newaxis]
            img = (img * (1 - paste_mask_3d) + src_img * paste_mask_3d).astype(np.uint8)
            mask = np.maximum(mask, paste_mask)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        return img, mask.unsqueeze(0) if mask.dim() == 2 else mask


class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("Use first_step/second_step instead")


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_iou(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum((1, 2, 3))
    union = pred_bin.sum((1, 2, 3)) + target.sum((1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def get_transforms(image_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
                               border_mode=0, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                     val_shift_limit=15, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ], p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(alpha=50, sigma=50 * 0.05, p=0.2),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def train_epoch(model, loader, criterion, optimizer, device, scaler, use_sam=False, ema=None):
    model.train()
    total_loss = 0
    total_iou = 0
    n = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if use_sam:
            # SAM: no GradScaler (130GB VRAM, no need for mixed precision)
            # First forward-backward
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.first_step()
            optimizer.base_optimizer.zero_grad()

            # Second forward-backward
            outputs2 = model(images)
            loss2 = criterion(outputs2, masks)
            loss2.backward()
            optimizer.second_step()
            optimizer.base_optimizer.zero_grad()

            total_loss += loss2.item() * images.size(0)
            total_iou += compute_iou(outputs2, masks) * images.size(0)
        else:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total_iou += compute_iou(outputs, masks) * images.size(0)

        if ema is not None:
            ema.update(model)

        n += images.size(0)

    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    n = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)
        total_iou += compute_iou(outputs, masks) * images.size(0)
        n += images.size(0)
    return total_loss / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["copypaste", "sam", "ema", "all"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", default="data/Segmentation")
    parser.add_argument("--pretrained_model", default="all_models/best_seg_mega512_lovasz.pth")
    parser.add_argument("--output_dir", default="all_models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_copypaste = args.mode in ("copypaste", "all")
    use_sam = args.mode in ("sam", "all")
    use_ema = args.mode in ("ema", "all")

    print(f"\nMode: {args.mode}")
    print(f"  Copy-Paste: {use_copypaste}")
    print(f"  SAM: {use_sam}")
    print(f"  EMA: {use_ema}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.img_size}")
    print(f"  Batch size: {args.batch_size}")

    # Data
    train_img_dir = f"{args.data_dir}/training/images"
    train_mask_dir = f"{args.data_dir}/training/masks"
    val_img_dir = f"{args.data_dir}/validation/images"
    val_mask_dir = f"{args.data_dir}/validation/masks"

    # Copy-paste pool
    cp_pool = None
    if use_copypaste:
        cp_pool = []
        for p in sorted(Path(train_img_dir).glob("*")):
            if p.suffix.lower() in {".png", ".jpg"}:
                mp = Path(train_mask_dir) / f"{p.stem}.png"
                if not mp.exists():
                    mp = Path(train_mask_dir) / f"{p.stem}.jpg"
                if mp.exists():
                    cp_pool.append((str(p), str(mp)))
        print(f"  Copy-Paste pool: {len(cp_pool)} images")

    train_tf = get_transforms(args.img_size, is_train=True)
    val_tf = get_transforms(args.img_size, is_train=False)

    train_ds = SegDataset(train_img_dir, train_mask_dir, train_tf, copy_paste_pool=cp_pool)
    val_ds = SegDataset(val_img_dir, val_mask_dir, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)}")

    # Model -- fine-tune from best checkpoint
    model = smp.UnetPlusPlus(
        encoder_name="tu-tf_efficientnetv2_s",
        encoder_weights=None, classes=1, activation=None
    )
    if Path(args.pretrained_model).exists():
        sd = torch.load(args.pretrained_model, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained: {args.pretrained_model}")
    else:
        print("WARNING: No pretrained model found, training from scratch!")
    model = model.to(device)

    # Loss: Dice + BCE
    bce = nn.BCEWithLogitsLoss()

    def criterion(pred, target):
        return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

    # Optimizer
    if use_sam:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

    scaler = torch.amp.GradScaler("cuda")

    # EMA
    ema_tracker = EMA(model, decay=0.999) if use_ema else None

    # Training
    best_iou = 0
    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"dgx_{args.mode}_{args.img_size}"
    best_path = f"{args.output_dir}/best_{output_name}.pth"

    print(f"\n{'='*60}")
    print(f"TRAINING START -- {output_name}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_sam=use_sam, ema=ema_tracker
        )
        scheduler.step()

        # Validate with EMA weights if available
        if ema_tracker is not None:
            ema_tracker.apply(model)

        val_loss, val_iou = validate(model, val_loader, criterion, device)

        if ema_tracker is not None:
            ema_tracker.restore(model)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"] if not use_sam else optimizer.base_optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            if ema_tracker is not None:
                ema_tracker.apply(model)
                torch.save(model.state_dict(), best_path)
                ema_tracker.restore(model)
            else:
                torch.save(model.state_dict(), best_path)
            print(f"  -> NEW BEST: IoU={val_iou:.4f} saved to {best_path}")

    # Also save EMA model at end
    if ema_tracker is not None:
        ema_tracker.apply(model)
        ema_path = f"{args.output_dir}/ema_{output_name}.pth"
        torch.save(model.state_dict(), ema_path)
        print(f"\nEMA model saved to {ema_path}")
        ema_tracker.restore(model)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE -- Best Val IoU: {best_iou:.4f}")
    print(f"Saved to: {best_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
