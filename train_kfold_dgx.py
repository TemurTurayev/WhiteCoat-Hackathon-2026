"""
DGX K-Fold Training + Model Soup for Segmentation
Best strategy from our ISIC soup breakthrough (0.8748 -> now aiming higher)
"""
import argparse
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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, copy_paste_pool=None):
        self.imgs = img_paths
        self.masks = mask_paths
        self.transform = transform
        self.copy_paste_pool = copy_paste_pool

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], 0)
        mask = (mask > 127).astype(np.float32)

        if self.copy_paste_pool is not None and random.random() < 0.5:
            src_idx = random.randint(0, len(self.copy_paste_pool) - 1)
            src_img = cv2.cvtColor(cv2.imread(self.copy_paste_pool[src_idx][0]), cv2.COLOR_BGR2RGB)
            src_mask = cv2.imread(self.copy_paste_pool[src_idx][1], 0)
            src_mask = (src_mask > 127).astype(np.float32)
            h, w = img.shape[:2]
            src_img = cv2.resize(src_img, (w, h))
            src_mask = cv2.resize(src_mask, (w, h))
            paste = (src_mask > 0.5).astype(np.float32)
            paste3d = paste[:, :, np.newaxis]
            img = (img * (1 - paste3d) + src_img * paste3d).astype(np.uint8)
            mask = np.maximum(mask, paste)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        return img, mask.unsqueeze(0) if mask.dim() == 2 else mask


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pf = pred.view(-1)
    tf = target.view(-1)
    inter = (pf * tf).sum()
    return 1 - (2.0 * inter + smooth) / (pf.sum() + tf.sum() + smooth)


def compute_iou(pred, target, thr=0.5):
    pb = (torch.sigmoid(pred) > thr).float()
    inter = (pb * target).sum((1, 2, 3))
    union = pb.sum((1, 2, 3)) + target.sum((1, 2, 3)) - inter
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


def get_transforms(sz, train=True):
    if train:
        return A.Compose([
            A.Resize(sz, sz),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, border_mode=0, p=0.5),
            A.OneOf([
                A.HueSaturationValue(10, 20, 15, p=1), A.RandomBrightnessContrast(0.2, 0.2, p=1),
            ], p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.2), A.GaussNoise(p=0.2),
            A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
        ])
    return A.Compose([A.Resize(sz, sz), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    tl, ti, n = 0, 0, 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.amp.autocast("cuda"):
            out = model(imgs)
            loss = criterion(out, masks)
        tl += loss.item() * imgs.size(0)
        ti += compute_iou(out, masks) * imgs.size(0)
        n += imgs.size(0)
    return tl / n, ti / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", default="data/Segmentation")
    parser.add_argument("--pretrained", default="all_models/best_seg_mega512_lovasz.pth")
    parser.add_argument("--output_dir", default="all_models/dgx_kfold")
    parser.add_argument("--use_copypaste", action="store_true")
    parser.add_argument("--start_fold", type=int, default=0, help="Skip folds before this (0-indexed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect train data + 300 of 400 val (hold out 100 for testing)
    all_imgs, all_masks = [], []
    holdout_imgs, holdout_masks = [], []

    # All training data
    train_img_dir = Path(args.data_dir) / "training" / "images"
    train_mask_dir = Path(args.data_dir) / "training" / "masks"
    for p in sorted(train_img_dir.glob("*")):
        if p.suffix.lower() in {".png", ".jpg"}:
            mp = train_mask_dir / f"{p.stem}.png"
            if not mp.exists():
                mp = train_mask_dir / f"{p.stem}.jpg"
            if mp.exists():
                all_imgs.append(str(p))
                all_masks.append(str(mp))

    # Val: first 300 for training, last 100 for holdout testing
    val_img_dir = Path(args.data_dir) / "validation" / "images"
    val_mask_dir = Path(args.data_dir) / "validation" / "masks"
    val_pairs = []
    for p in sorted(val_img_dir.glob("*")):
        if p.suffix.lower() in {".png", ".jpg"}:
            mp = val_mask_dir / f"{p.stem}.png"
            if not mp.exists():
                mp = val_mask_dir / f"{p.stem}.jpg"
            if mp.exists():
                val_pairs.append((str(p), str(mp)))

    # Use first 300 val for training, last 100 for holdout
    for img_p, mask_p in val_pairs[:300]:
        all_imgs.append(img_p)
        all_masks.append(mask_p)
    for img_p, mask_p in val_pairs[300:]:
        holdout_imgs.append(img_p)
        holdout_masks.append(mask_p)

    print(f"Training pool: {len(all_imgs)} | Holdout test: {len(holdout_imgs)}")

    cp_pool = list(zip(all_imgs, all_masks)) if args.use_copypaste else None
    os.makedirs(args.output_dir, exist_ok=True)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_ious = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_imgs)):
        if fold < args.start_fold:
            print(f"\nSkipping fold {fold+1} (already done)")
            continue
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.n_folds}")
        print(f"{'='*60}")

        train_imgs = [all_imgs[i] for i in train_idx]
        train_masks = [all_masks[i] for i in train_idx]
        val_imgs = [all_imgs[i] for i in val_idx]
        val_masks = [all_masks[i] for i in val_idx]

        train_ds = SegDataset(train_imgs, train_masks, get_transforms(args.img_size, True), cp_pool)
        val_ds = SegDataset(val_imgs, val_masks, get_transforms(args.img_size, False))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

        # Fresh model from pretrained
        model = smp.UnetPlusPlus(encoder_name="tu-tf_efficientnetv2_s", encoder_weights=None, classes=1, activation=None)
        if Path(args.pretrained).exists():
            sd = torch.load(args.pretrained, map_location="cpu")
            model.load_state_dict(sd, strict=False)
        model = model.to(device)

        bce = nn.BCEWithLogitsLoss()
        def criterion(p, t):
            return 0.5 * bce(p, t) + 0.5 * dice_loss(p, t)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda")
        ema = EMA(model, decay=0.999)

        best_iou = 0
        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            tl, ti, n = 0, 0, 0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    out = model(imgs)
                    loss = criterion(out, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                tl += loss.item() * imgs.size(0)
                ti += compute_iou(out, masks) * imgs.size(0)
                n += imgs.size(0)
            scheduler.step()

            ema.apply(model)
            vl, vi = validate(model, val_loader, criterion, device)
            ema.restore(model)

            elapsed = time.time() - t0
            print(f"E{epoch+1:02d}/{args.epochs} | TrL:{tl/n:.4f} IoU:{ti/n:.4f} | VL:{vl:.4f} IoU:{vi:.4f} | {elapsed:.0f}s", end="")

            if vi > best_iou:
                best_iou = vi
                ema.apply(model)
                torch.save(model.state_dict(), f"{args.output_dir}/fold{fold}.pth")
                ema.restore(model)
                print(f" *BEST*")
            else:
                print()

        fold_ious.append(best_iou)
        print(f"Fold {fold+1} best IoU: {best_iou:.4f}")

    # Create Model Soup
    print(f"\n{'='*60}")
    print(f"CREATING MODEL SOUP")
    print(f"{'='*60}")
    print(f"Fold IoUs: {[f'{x:.4f}' for x in fold_ious]}")
    print(f"Mean: {np.mean(fold_ious):.4f}")

    soup_sd = None
    for fold in range(args.n_folds):
        sd = torch.load(f"{args.output_dir}/fold{fold}.pth", map_location="cpu")
        if soup_sd is None:
            soup_sd = {k: v.float() for k, v in sd.items()}
        else:
            for k in soup_sd:
                soup_sd[k] += sd[k].float()

    for k in soup_sd:
        soup_sd[k] /= args.n_folds

    soup_path = f"{args.output_dir}/soup_{args.n_folds}fold.pth"
    torch.save(soup_sd, soup_path)
    print(f"Soup saved to: {soup_path}")

    # Evaluate soup on holdout 100
    if holdout_imgs:
        print(f"\n{'='*60}")
        print(f"EVALUATING ON HOLDOUT ({len(holdout_imgs)} images)")
        print(f"{'='*60}")
        soup_model = smp.UnetPlusPlus(encoder_name="tu-tf_efficientnetv2_s", encoder_weights=None, classes=1, activation=None)
        soup_model.load_state_dict(soup_sd)
        soup_model = soup_model.to(device)
        holdout_ds = SegDataset(holdout_imgs, holdout_masks, get_transforms(args.img_size, False))
        holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
        bce = nn.BCEWithLogitsLoss()
        def crit(p, t):
            return 0.5 * bce(p, t) + 0.5 * dice_loss(p, t)
        hl, hi = validate(soup_model, holdout_loader, crit, device)
        print(f"Soup Holdout IoU: {hi:.4f} (Loss: {hl:.4f})")

        # Also test individual folds on holdout
        for fold in range(args.n_folds):
            fm = smp.UnetPlusPlus(encoder_name="tu-tf_efficientnetv2_s", encoder_weights=None, classes=1, activation=None)
            fm.load_state_dict(torch.load(f"{args.output_dir}/fold{fold}.pth", map_location="cpu"))
            fm = fm.to(device)
            fl, fi = validate(fm, holdout_loader, crit, device)
            print(f"  Fold {fold} Holdout IoU: {fi:.4f}")

    print("DONE!")


if __name__ == "__main__":
    main()
