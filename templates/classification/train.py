"""
Тренировка модели классификации.

Запуск:
    python classification/train.py --config config.yaml --data_dir data/

Что делает:
1. Загружает данные с augmentation
2. Fine-tune pretrained модели
3. Сохраняет лучшую модель по валидации
4. Логирует метрики
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from utils.common import (
    set_seed, get_device, load_config, ensure_dirs,
    AverageMeter, EarlyStopping
)
from utils.augmentations import get_classification_transforms
from classification.dataset import MedicalClassificationDataset
from classification.model import create_classification_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> tuple[float, float]:
    """Одна эпоха обучения."""
    model.train()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return loss_meter.avg, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Валидация модели."""
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
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--data_dir", required=True, help="Data directory")
    parser.add_argument("--csv_path", default=None, help="CSV with labels (optional)")
    args = parser.parse_args()

    # Загрузка конфига
    config = load_config(args.config)
    cfg = config["classification"]
    set_seed(config["general"]["seed"])
    device = get_device()
    ensure_dirs(config["general"]["output_dir"], config["general"]["checkpoint_dir"])

    # === ДАННЫЕ ===
    train_transform = get_classification_transforms(cfg["image_size"], is_train=True)
    val_transform = get_classification_transforms(cfg["image_size"], is_train=False)

    if args.csv_path:
        full_dataset = MedicalClassificationDataset.from_csv(
            args.csv_path, root_dir=args.data_dir, transform=None
        )
        # Делим 80/20
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=0.2,
            random_state=config["general"]["seed"],
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
    else:
        train_dataset = MedicalClassificationDataset.from_folder(
            f"{args.data_dir}/train", transform=train_transform
        )
        val_dataset = MedicalClassificationDataset.from_folder(
            f"{args.data_dir}/val", transform=val_transform
        )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=config["general"]["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"] * 2,
        shuffle=False, num_workers=config["general"]["num_workers"],
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # === МОДЕЛЬ ===
    model = create_classification_model(
        model_name=cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
        dropout=cfg["dropout"],
    ).to(device)

    # === ОБУЧЕНИЕ ===
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

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

    best_f1 = 0.0
    checkpoint_path = Path(config["general"]["checkpoint_dir"]) / "best_classification.pth"

    print(f"\nStarting training for {cfg['epochs']} epochs...")
    print(f"Model: {cfg['model_name']} | LR: {cfg['learning_rate']} | BS: {cfg['batch_size']}")
    print("-" * 60)

    for epoch in range(cfg["epochs"]):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {lr:.6f} | Time: {elapsed:.1f}s"
        )

        # Сохраняем лучшую модель
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "config": cfg,
            }, checkpoint_path)
            print(f"  -> Saved best model (F1: {val_f1:.4f})")

        if early_stop(val_f1):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest Val F1: {best_f1:.4f}")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
