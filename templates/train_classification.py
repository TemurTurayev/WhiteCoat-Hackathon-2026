"""
ФИНАЛЬНЫЙ тренировочный скрипт — Classification (Problem A)
Team: WhiteCoat.dev

Запуск одной модели:
    python train_classification.py --model tf_efficientnetv2_s --data_dir data/classification

Запуск с кастомными параметрами:
    python train_classification.py --model convnext_tiny --data_dir data/classification \
        --image_size 224 --batch_size 64 --epochs 40 --lr 0.0003

Что делает:
1. Загружает данные из папок (train/0, train/1, ..., train/11)
2. Делит train на train/val (80/20, stratified)
3. Применяет augmentations для биопсии
4. Weighted CrossEntropy для несбалансированных классов
5. Fine-tune pretrained модели
6. Сохраняет лучшую модель по val accuracy
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from utils.common import set_seed, get_device, ensure_dirs, AverageMeter, EarlyStopping
from utils.augmentations import get_classification_transforms
from utils.losses import FocalLoss
from classification.dataset import MedicalClassificationDataset


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Вычисляет веса классов обратно пропорционально частоте.
    Класс 8 (331 images) получит больший вес, чем класс 7 (2136 images).
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    weights_tensor = torch.FloatTensor(weights)
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return loss_meter.avg, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return loss_meter.avg, accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Train Classification — WhiteCoat.dev")
    parser.add_argument("--model", default="tf_efficientnetv2_s", help="timm model name")
    parser.add_argument("--data_dir", required=True, help="Path to classification/ folder")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--no_weighted_loss", action="store_true")
    parser.add_argument("--focal_loss", action="store_true", default=True,
                        help="Use Focal Loss instead of CrossEntropy (recommended)")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dirs(args.checkpoint_dir)

    num_classes = 12
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"

    # === ДАННЫЕ ===
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.image_size} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"{'='*60}\n")

    train_transform = get_classification_transforms(args.image_size, is_train=True)
    val_transform = get_classification_transforms(args.image_size, is_train=False)

    full_dataset = MedicalClassificationDataset.from_folder(str(train_dir), transform=None)

    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=args.seed,
        stratify=full_dataset.labels,
    )

    train_dataset = MedicalClassificationDataset(
        [full_dataset.image_paths[i] for i in train_idx],
        [full_dataset.labels[i] for i in train_idx],
        transform=train_transform,
    )
    val_dataset = MedicalClassificationDataset(
        [full_dataset.image_paths[i] for i in val_idx],
        [full_dataset.labels[i] for i in val_idx],
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
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # === LOSS ===
    class_weights = None
    if not args.no_weighted_loss:
        class_weights = compute_class_weights(train_dataset.labels, num_classes).to(device)
        print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    if args.focal_loss:
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            alpha=class_weights,
            label_smoothing=args.label_smoothing,
        )
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    elif class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # === OPTIMIZER + SCHEDULER ===
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    early_stop = EarlyStopping(patience=args.patience, mode="max")

    best_acc = 0.0
    model_short = args.model.replace("/", "_")
    checkpoint_path = Path(args.checkpoint_dir) / f"best_{model_short}.pth"

    print(f"\nTraining {args.model} for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} | "
            f"Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} F1:{val_f1:.4f} | "
            f"LR:{lr:.6f} | {elapsed:.0f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": {
                    "model_name": args.model,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                },
            }, checkpoint_path)
            print(f"  -> NEW BEST! Acc: {val_acc:.4f} saved to {checkpoint_path}")

        if early_stop(val_acc):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\n{'='*60}")
    print(f"DONE! Best Val Accuracy: {best_acc:.4f}")
    print(f"Model saved: {checkpoint_path}")
    print(f"{'='*60}")

    # Финальный classification report
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.numpy())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    main()
