"""
ФИНАЛЬНЫЙ тренировочный скрипт — Segmentation (Problem B)
Team: WhiteCoat.dev

Запуск:
    python train_segmentation.py --arch UnetPlusPlus --encoder efficientnet-b4 \
        --data_dir data/Segmentation

Что делает:
1. Загружает images + masks из training/ и validation/
2. Augmentations для биопсии (с масками)
3. Dice + BCE комбинированный loss
4. Сохраняет лучшую модель по val IoU
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent))
from utils.common import set_seed, get_device, ensure_dirs, AverageMeter, EarlyStopping
from utils.augmentations import get_segmentation_transforms
from utils.losses import TwoPhaseSegmentationLoss


class BiopsySegDataset(Dataset):
    """Dataset для сегментации биопсии. images/ + masks/ с одинаковыми именами."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        ])
        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Ищем маску с тем же stem но возможно другим расширением
        mask_path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = self.mask_dir / (img_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {img_path.name}")

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask = mask.unsqueeze(0).float()

        return image, mask


class DiceBCELoss(nn.Module):
    """Комбинированный Dice + BCE loss для бинарной сегментации."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def compute_iou(pred, target, threshold=0.5):
    """Compute mean IoU для батча."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(2, 3))
    union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        iou_meter.update(compute_iou(outputs, masks), images.size(0))

    return loss_meter.avg, iou_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss_meter.update(loss.item(), images.size(0))
        iou_meter.update(compute_iou(outputs, masks), images.size(0))

    return loss_meter.avg, iou_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Train Segmentation — WhiteCoat.dev")
    parser.add_argument("--arch", default="UnetPlusPlus", help="SMP architecture")
    parser.add_argument("--encoder", default="efficientnet-b4", help="Encoder backbone")
    parser.add_argument("--data_dir", required=True, help="Path to Segmentation/ folder")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--lovasz_epoch", type=int, default=20,
                        help="Epoch to add Lovász Loss (0=disabled)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dirs(args.checkpoint_dir)

    data_dir = Path(args.data_dir)

    print(f"\n{'='*60}")
    print(f"Architecture: {args.arch} + {args.encoder}")
    print(f"Image size: {args.image_size} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*60}\n")

    train_transform = get_segmentation_transforms(args.image_size, is_train=True)
    val_transform = get_segmentation_transforms(args.image_size, is_train=False)

    train_dataset = BiopsySegDataset(
        str(data_dir / "training" / "images"),
        str(data_dir / "training" / "masks"),
        transform=train_transform,
    )
    val_dataset = BiopsySegDataset(
        str(data_dir / "validation" / "images"),
        str(data_dir / "validation" / "masks"),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # === МОДЕЛЬ ===
    arch_registry = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "Linknet": smp.Linknet,
    }

    model_class = arch_registry[args.arch]
    model = model_class(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    if args.lovasz_epoch > 0:
        criterion = TwoPhaseSegmentationLoss(switch_epoch=args.lovasz_epoch)
        print(f"Using Two-Phase Loss: Dice+BCE → +Lovász at epoch {args.lovasz_epoch}")
    else:
        criterion = DiceBCELoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight)
        print("Using Dice+BCE Loss")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    best_iou = 0.0
    arch_short = f"{args.arch}_{args.encoder}".replace("-", "_")
    checkpoint_path = Path(args.checkpoint_dir) / f"best_{arch_short}.pth"

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        # Update Two-Phase Loss epoch
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)

        start = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss:{train_loss:.4f} IoU:{train_iou:.4f} | "
            f"Val Loss:{val_loss:.4f} IoU:{val_iou:.4f} | "
            f"LR:{lr:.6f} | {elapsed:.0f}s"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_iou": val_iou,
                "config": {
                    "model_name": args.arch,
                    "encoder_name": args.encoder,
                    "num_classes": 1,
                    "image_size": args.image_size,
                },
            }, checkpoint_path)
            print(f"  -> NEW BEST! IoU: {val_iou:.4f} saved to {checkpoint_path}")

        if early_stop(val_iou):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\n{'='*60}")
    print(f"DONE! Best Val IoU: {best_iou:.4f}")
    print(f"Model saved: {checkpoint_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
