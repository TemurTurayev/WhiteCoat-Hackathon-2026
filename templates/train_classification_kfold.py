"""
K-Fold Cross-Validation + Ensemble — Classification (Problem A)
Team: WhiteCoat.dev

Запуск полного пайплайна:
    python train_classification_kfold.py --model tf_efficientnetv2_s --data_dir data/classification --n_folds 5

Запуск одного фолда (для параллельного обучения):
    python train_classification_kfold.py --model tf_efficientnetv2_s --data_dir data/classification \
        --n_folds 5 --fold 0

Что делает:
1. StratifiedKFold разбивает train на K фолдов (сохраняет баланс классов)
2. Для каждого фолда: train модель, сохранить checkpoint + OOF predictions
3. OOF predictions потом используются для оптимизации весов ансамбля
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))
from utils.common import set_seed, get_device, ensure_dirs, AverageMeter, EarlyStopping
from utils.augmentations import get_classification_transforms
from classification.dataset import MedicalClassificationDataset


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weights.append(total / (num_classes * count))
    weights_tensor = torch.FloatTensor(weights)
    return weights_tensor / weights_tensor.mean()


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
    all_softmax = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))

        softmax_out = torch.softmax(outputs, dim=1).cpu().numpy()
        all_softmax.append(softmax_out)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    all_softmax = np.concatenate(all_softmax, axis=0)
    return loss_meter.avg, accuracy, f1, all_softmax


@torch.no_grad()
def predict_softmax(model, loader, device):
    """Get softmax predictions for all samples in loader."""
    model.eval()
    all_softmax = []
    for images, _ in loader:
        images = images.to(device)
        outputs = model(images)
        softmax_out = torch.softmax(outputs, dim=1).cpu().numpy()
        all_softmax.append(softmax_out)
    return np.concatenate(all_softmax, axis=0)


def train_single_fold(
    fold_idx,
    train_indices,
    val_indices,
    full_dataset,
    args,
    device,
    num_classes=12,
):
    """Train one fold. Returns best val accuracy and OOF predictions."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{args.n_folds} | Model: {args.model}")
    print(f"{'='*60}")

    train_transform = get_classification_transforms(args.image_size, is_train=True)
    val_transform = get_classification_transforms(args.image_size, is_train=False)

    train_dataset = MedicalClassificationDataset(
        [full_dataset.image_paths[i] for i in train_indices],
        [full_dataset.labels[i] for i in train_indices],
        transform=train_transform,
    )
    val_dataset = MedicalClassificationDataset(
        [full_dataset.image_paths[i] for i in val_indices],
        [full_dataset.labels[i] for i in val_indices],
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Model ---
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=args.dropout,
    ).to(device)

    # --- Loss ---
    class_weights = compute_class_weights(train_dataset.labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    # --- Optimizer ---
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
    checkpoint_path = (
        Path(args.checkpoint_dir) / f"kfold_{model_short}_fold{fold_idx}.pth"
    )

    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_f1, _ = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"  Epoch {epoch+1:02d}/{args.epochs} | "
            f"TrLoss:{train_loss:.4f} TrAcc:{train_acc:.4f} | "
            f"VlLoss:{val_loss:.4f} VlAcc:{val_acc:.4f} F1:{val_f1:.4f} | "
            f"{elapsed:.0f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "fold": fold_idx,
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
                },
                checkpoint_path,
            )
            print(f"    -> BEST fold {fold_idx} Acc: {val_acc:.4f}")

        if early_stop(val_acc):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # --- Collect OOF predictions from best model ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    oof_softmax = predict_softmax(model, val_loader, device)

    print(f"Fold {fold_idx} DONE | Best Acc: {best_acc:.4f}")
    return best_acc, oof_softmax, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="K-Fold Classification — WhiteCoat.dev"
    )
    parser.add_argument("--model", default="tf_efficientnetv2_s", help="timm model")
    parser.add_argument("--data_dir", required=True, help="Path to classification/")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--fold", type=int, default=-1,
        help="Train only this fold (0-indexed). -1 = all folds",
    )
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dirs(args.checkpoint_dir)

    num_classes = 12
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"

    # --- Load full dataset ---
    full_dataset = MedicalClassificationDataset.from_folder(
        str(train_dir), transform=None
    )
    labels = np.array(full_dataset.labels)
    print(f"Total samples: {len(labels)} | Classes: {num_classes}")
    print(f"Class distribution: {Counter(labels.tolist())}")

    # --- StratifiedKFold ---
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )
    splits = list(skf.split(np.arange(len(labels)), labels))

    # --- OOF storage ---
    oof_predictions = np.zeros((len(labels), num_classes), dtype=np.float32)
    fold_accuracies = []
    checkpoint_paths = []

    # --- Determine which folds to train ---
    if args.fold >= 0:
        folds_to_train = [args.fold]
    else:
        folds_to_train = list(range(args.n_folds))

    for fold_idx in folds_to_train:
        train_idx, val_idx = splits[fold_idx]

        best_acc, oof_softmax, ckpt_path = train_single_fold(
            fold_idx=fold_idx,
            train_indices=train_idx.tolist(),
            val_indices=val_idx.tolist(),
            full_dataset=full_dataset,
            args=args,
            device=device,
            num_classes=num_classes,
        )

        oof_predictions[val_idx] = oof_softmax
        fold_accuracies.append(best_acc)
        checkpoint_paths.append(str(ckpt_path))

    # --- Save OOF predictions ---
    model_short = args.model.replace("/", "_")
    oof_dir = Path(args.checkpoint_dir) / "oof"
    ensure_dirs(str(oof_dir))

    oof_path = oof_dir / f"oof_{model_short}.npz"
    np.savez(
        oof_path,
        predictions=oof_predictions,
        labels=labels,
        fold_accuracies=np.array(fold_accuracies),
    )
    print(f"\nOOF predictions saved to {oof_path}")

    # --- Summary ---
    if len(folds_to_train) == args.n_folds:
        oof_preds = np.argmax(oof_predictions, axis=1)
        overall_acc = accuracy_score(labels, oof_preds)
        overall_f1 = f1_score(labels, oof_preds, average="macro")

        print(f"\n{'='*60}")
        print(f"K-FOLD SUMMARY for {args.model}")
        print(f"{'='*60}")
        for i, acc in enumerate(fold_accuracies):
            print(f"  Fold {i}: {acc:.4f}")
        print(f"  Mean:  {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
        print(f"  OOF Accuracy: {overall_acc:.4f}")
        print(f"  OOF Macro F1: {overall_f1:.4f}")
        print(f"\nClassification Report (OOF):")
        print(classification_report(labels, oof_preds, digits=4))

    # --- Save metadata ---
    meta = {
        "model": args.model,
        "n_folds": args.n_folds,
        "fold_accuracies": fold_accuracies,
        "checkpoint_paths": checkpoint_paths,
        "oof_path": str(oof_path),
    }
    meta_path = Path(args.checkpoint_dir) / f"kfold_meta_{model_short}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
