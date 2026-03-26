"""
K-Fold Cross-Validation + Ensemble — Segmentation (Problem B)
Team: WhiteCoat.dev

Запуск полного пайплайна:
    python train_segmentation_kfold.py --arch UnetPlusPlus --encoder efficientnet-b4 \
        --data_dir data/Segmentation --n_folds 5

Запуск одного фолда:
    python train_segmentation_kfold.py --arch UnetPlusPlus --encoder efficientnet-b4 \
        --data_dir data/Segmentation --n_folds 5 --fold 0

Что делает:
1. KFold на training данных (1800 images)
2. Для каждого фолда: train + validate на fold-val + external val (400 images)
3. Сохраняет OOF logits для threshold/weight optimization
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent))
from utils.common import set_seed, get_device, ensure_dirs, AverageMeter, EarlyStopping
from utils.augmentations import get_segmentation_transforms


class BiopsySegDataset(Dataset):
    """Dataset для сегментации биопсии."""

    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
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

    @classmethod
    def from_folder(cls, image_dir: str, mask_dir: str, transform=None):
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)

        image_files = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        ])

        image_paths = []
        mask_paths = []
        for img_path in image_files:
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = mask_dir / (img_path.stem + ext)
                if candidate.exists():
                    image_paths.append(img_path)
                    mask_paths.append(candidate)
                    break

        print(f"Found {len(image_paths)} image-mask pairs in {image_dir}")
        return cls(image_paths, mask_paths, transform)


class DiceBCELoss(nn.Module):
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
def validate_seg(model, loader, criterion, device):
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


@torch.no_grad()
def predict_logits(model, loader, device):
    """Get raw logits (before sigmoid) for all samples."""
    model.eval()
    all_logits = []
    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        all_logits.append(outputs.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def sigmoid_np(x):
    """Numerically stable sigmoid for numpy arrays."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def train_single_fold(
    fold_idx, train_indices, val_indices,
    all_image_paths, all_mask_paths,
    ext_val_loader, args, device,
):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{args.n_folds} | {args.arch} + {args.encoder}")
    print(f"{'='*60}")

    train_transform = get_segmentation_transforms(args.image_size, is_train=True)
    val_transform = get_segmentation_transforms(args.image_size, is_train=False)

    train_dataset = BiopsySegDataset(
        [all_image_paths[i] for i in train_indices],
        [all_mask_paths[i] for i in train_indices],
        transform=train_transform,
    )
    fold_val_dataset = BiopsySegDataset(
        [all_image_paths[i] for i in val_indices],
        [all_mask_paths[i] for i in val_indices],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    fold_val_loader = DataLoader(
        fold_val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} | Fold-Val: {len(fold_val_dataset)}")

    # --- Model ---
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

    criterion = DiceBCELoss(dice_weight=args.dice_weight, bce_weight=args.bce_weight)
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
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"kfold_{arch_short}_fold{fold_idx}.pth"
    )

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        fold_val_loss, fold_val_iou = validate_seg(
            model, fold_val_loader, criterion, device
        )
        ext_val_loss, ext_val_iou = validate_seg(
            model, ext_val_loader, criterion, device
        )
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"  Ep {epoch+1:02d}/{args.epochs} | "
            f"Tr IoU:{train_iou:.4f} | "
            f"FoldVal IoU:{fold_val_iou:.4f} | "
            f"ExtVal IoU:{ext_val_iou:.4f} | "
            f"{elapsed:.0f}s"
        )

        if fold_val_iou > best_iou:
            best_iou = fold_val_iou
            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "fold_val_iou": fold_val_iou,
                    "ext_val_iou": ext_val_iou,
                    "config": {
                        "arch": args.arch,
                        "encoder": args.encoder,
                        "image_size": args.image_size,
                    },
                },
                checkpoint_path,
            )
            print(f"    -> BEST fold {fold_idx} IoU: {fold_val_iou:.4f} (ext: {ext_val_iou:.4f})")

        if early_stop(fold_val_iou):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # --- OOF predictions ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    oof_logits = predict_logits(model, fold_val_loader, device)
    ext_logits = predict_logits(model, ext_val_loader, device)

    print(f"Fold {fold_idx} DONE | Best IoU: {best_iou:.4f}")
    return best_iou, oof_logits, ext_logits, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="K-Fold Segmentation — WhiteCoat.dev"
    )
    parser.add_argument("--arch", default="UnetPlusPlus", help="SMP architecture")
    parser.add_argument("--encoder", default="efficientnet-b4", help="Encoder")
    parser.add_argument("--data_dir", required=True, help="Path to Segmentation/")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=-1, help="-1 = all folds")
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dirs(args.checkpoint_dir)

    data_dir = Path(args.data_dir)

    # --- Load training data paths ---
    train_data = BiopsySegDataset.from_folder(
        str(data_dir / "training" / "images"),
        str(data_dir / "training" / "masks"),
        transform=None,
    )
    all_image_paths = train_data.image_paths
    all_mask_paths = train_data.mask_paths
    n_samples = len(all_image_paths)
    print(f"Training samples: {n_samples}")

    # --- External validation set (always evaluated) ---
    val_transform = get_segmentation_transforms(args.image_size, is_train=False)
    ext_val_dataset = BiopsySegDataset.from_folder(
        str(data_dir / "validation" / "images"),
        str(data_dir / "validation" / "masks"),
        transform=val_transform,
    )
    ext_val_loader = DataLoader(
        ext_val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"External validation: {len(ext_val_dataset)}")

    # --- KFold ---
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(kf.split(np.arange(n_samples)))

    fold_ious = []
    checkpoint_paths = []

    if args.fold >= 0:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(args.n_folds))

    ext_val_logits_all = []

    for fold_idx in folds_to_train:
        train_idx, val_idx = splits[fold_idx]

        best_iou, oof_logits, ext_logits, ckpt_path = train_single_fold(
            fold_idx=fold_idx,
            train_indices=train_idx.tolist(),
            val_indices=val_idx.tolist(),
            all_image_paths=all_image_paths,
            all_mask_paths=all_mask_paths,
            ext_val_loader=ext_val_loader,
            args=args,
            device=device,
        )

        fold_ious.append(best_iou)
        checkpoint_paths.append(str(ckpt_path))
        ext_val_logits_all.append(ext_logits)

    # --- Save ext_val logits for ensemble ---
    arch_short = f"{args.arch}_{args.encoder}".replace("-", "_")
    oof_dir = Path(args.checkpoint_dir) / "oof"
    ensure_dirs(str(oof_dir))

    if ext_val_logits_all:
        ext_logits_stacked = np.stack(ext_val_logits_all, axis=0)
        logits_path = oof_dir / f"ext_logits_{arch_short}.npz"
        np.savez(logits_path, logits=ext_logits_stacked, fold_ious=np.array(fold_ious))
        print(f"\nExternal val logits saved: {logits_path}")
        print(f"Shape: {ext_logits_stacked.shape} (folds, samples, 1, H, W)")

    # --- Summary ---
    if len(folds_to_train) == args.n_folds:
        print(f"\n{'='*60}")
        print(f"K-FOLD SUMMARY: {args.arch} + {args.encoder}")
        print(f"{'='*60}")
        for i, iou in enumerate(fold_ious):
            print(f"  Fold {i}: {iou:.4f}")
        print(f"  Mean:  {np.mean(fold_ious):.4f} +/- {np.std(fold_ious):.4f}")

        # Ensemble: average logits across folds, then threshold
        avg_logits = ext_logits_stacked.mean(axis=0)
        avg_probs = sigmoid_np(avg_logits)

        # Compute ensemble IoU on ext_val
        ext_masks = []
        for images, masks in ext_val_loader:
            ext_masks.append(masks.numpy())
        ext_masks = np.concatenate(ext_masks, axis=0)

        for threshold in [0.3, 0.4, 0.5, 0.6]:
            pred_binary = (avg_probs > threshold).astype(np.float32)
            intersection = (pred_binary * ext_masks).sum(axis=(1, 2, 3))
            union = pred_binary.sum(axis=(1, 2, 3)) + ext_masks.sum(axis=(1, 2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            print(f"  Ensemble @{threshold}: Mean IoU = {iou.mean():.4f}")

    # --- Save metadata ---
    meta = {
        "arch": args.arch,
        "encoder": args.encoder,
        "n_folds": args.n_folds,
        "fold_ious": fold_ious,
        "checkpoint_paths": checkpoint_paths,
    }
    meta_path = Path(args.checkpoint_dir) / f"kfold_meta_{arch_short}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
