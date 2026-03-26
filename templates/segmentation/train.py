"""
Тренировка модели сегментации (Problem B — 40 баллов!).

Запуск:
    python segmentation/train.py --config config.yaml --image_dir data/images --mask_dir data/masks

Что делает:
1. Загружает пары image+mask с augmentation
2. Fine-tune pretrained U-Net++ / DeepLabV3+
3. Оптимизирует Dice + BCE loss
4. Сохраняет лучшую модель по Dice Score
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from utils.common import (
    set_seed, get_device, load_config, ensure_dirs,
    AverageMeter, EarlyStopping
)
from utils.augmentations import get_segmentation_transforms
from segmentation.dataset import MedicalSegmentationDataset
from segmentation.model import (
    create_segmentation_model, DiceBCELoss,
    compute_iou, compute_dice
)


def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler=None
) -> tuple[float, float, float]:
    """Одна эпоха обучения сегментации."""
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(compute_dice(outputs.detach(), masks), batch_size)
        iou_meter.update(compute_iou(outputs.detach(), masks), batch_size)

    return loss_meter.avg, dice_meter.avg, iou_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float, float]:
    """Валидация модели сегментации."""
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(compute_dice(outputs, masks), batch_size)
        iou_meter.update(compute_iou(outputs, masks), batch_size)

    return loss_meter.avg, dice_meter.avg, iou_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image_dir", required=True, help="Images directory")
    parser.add_argument("--mask_dir", required=True, help="Masks directory")
    parser.add_argument("--csv_path", default=None, help="CSV with paths (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["segmentation"]
    set_seed(config["general"]["seed"])
    device = get_device()
    ensure_dirs(config["general"]["output_dir"], config["general"]["checkpoint_dir"])

    # === ДАННЫЕ ===
    train_transform = get_segmentation_transforms(cfg["image_size"], is_train=True)
    val_transform = get_segmentation_transforms(cfg["image_size"], is_train=False)

    if args.csv_path:
        full_dataset = MedicalSegmentationDataset.from_csv(
            args.csv_path, num_classes=cfg["num_classes"], transform=None
        )
    else:
        full_dataset = MedicalSegmentationDataset.from_dirs(
            args.image_dir, args.mask_dir,
            num_classes=cfg["num_classes"], transform=None
        )

    # Split 80/20
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=config["general"]["seed"],
    )

    train_dataset = MedicalSegmentationDataset(
        [full_dataset.image_paths[i] for i in train_idx],
        [full_dataset.mask_paths[i] for i in train_idx],
        num_classes=cfg["num_classes"],
        transform=train_transform,
    )
    val_dataset = MedicalSegmentationDataset(
        [full_dataset.image_paths[i] for i in val_idx],
        [full_dataset.mask_paths[i] for i in val_idx],
        num_classes=cfg["num_classes"],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=config["general"]["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=config["general"]["num_workers"],
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # === МОДЕЛЬ ===
    model = create_segmentation_model(
        model_name=cfg["model_name"],
        encoder_name=cfg["encoder_name"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
    ).to(device)

    # === ОБУЧЕНИЕ ===
    criterion = DiceBCELoss(
        dice_weight=cfg["dice_weight"],
        bce_weight=cfg["bce_weight"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    scaler = GradScaler() if config["general"]["mixed_precision"] and device.type == "cuda" else None
    early_stop = EarlyStopping(patience=cfg["early_stopping_patience"], mode="max")

    best_dice = 0.0
    checkpoint_path = Path(config["general"]["checkpoint_dir"]) / "best_segmentation.pth"

    print(f"\nStarting training for {cfg['epochs']} epochs...")
    print(f"Model: {cfg['model_name']} + {cfg['encoder_name']}")
    print(f"LR: {cfg['learning_rate']} | BS: {cfg['batch_size']} | Image: {cfg['image_size']}px")
    print("-" * 70)

    for epoch in range(cfg["epochs"]):
        start = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
                "config": cfg,
            }, checkpoint_path)
            print(f"  -> Saved best model (Dice: {val_dice:.4f}, IoU: {val_iou:.4f})")

        if early_stop(val_dice):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest Val Dice: {best_dice:.4f}")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
