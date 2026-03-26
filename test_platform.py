"""
WhiteCoat.dev — Платформа тестирования моделей
Запуск: python test_platform.py

Возможности:
- Тест одного изображения или папки
- GradCAM визуализация (где модель смотрит)
- Confusion matrix на валидационных данных
- Per-class accuracy breakdown
- Сравнение моделей бок-о-бок
- Confidence distribution анализ
- Сегментация overlay визуализация
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("Warning: pytorch-grad-cam not installed. GradCAM disabled.")
    print("Install: pip install pytorch-grad-cam")

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================
# CONFIG
# ============================================================
# CRITICAL: Model was trained with alphabetical folder sorting!
# Folders sorted as strings: ['0','1','10','11','2','3','4','5','6','7','8','9']
# So model_idx 0 = folder '0', model_idx 2 = folder '10', etc.
FOLDER_ORDER = ['0', '1', '10', '11', '2', '3', '4', '5', '6', '7', '8', '9']
MODEL_IDX_TO_REAL = {i: int(FOLDER_ORDER[i]) for i in range(12)}
REAL_TO_MODEL_IDX = {v: k for k, v in MODEL_IDX_TO_REAL.items()}

CLASS_NAMES = {
    0: "Actinic Keratosis",
    1: "Basal Cell Carcinoma",
    2: "Dermatofibroma",
    3: "Hemangioma",
    4: "Intraepithelial Ca.",
    5: "Lentigo",
    6: "Melanoma",
    7: "Melanocytic Nevus",
    8: "Pyogenic Granuloma",
    9: "Seborrheic Keratosis",
    10: "Squamous Cell Ca.",
    11: "Wart",
}

CLASS_TYPES = {
    0: "pre-malignant", 1: "malignant", 2: "benign", 3: "benign",
    4: "malignant", 5: "benign", 6: "malignant", 7: "benign",
    8: "benign", 9: "benign", 10: "malignant", 11: "benign",
}

DANGEROUS_PAIRS = [
    (6, 7, "Melanoma -> Nevus (FATAL)"),
    (6, 9, "Melanoma -> Seb.Keratosis"),
    (10, 11, "SCC -> Wart"),
    (1, 2, "BCC -> Dermatofibroma"),
    (8, 6, "Pyogenic Gran. -> Melanoma"),
]

IMAGE_SIZE = 224
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ============================================================
# TRANSFORMS
# ============================================================
def get_val_transform(size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(size=IMAGE_SIZE):
    norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = A.Resize(size, size)
    return [
        A.Compose([resize, norm, ToTensorV2()]),
        A.Compose([resize, A.HorizontalFlip(p=1.0), norm, ToTensorV2()]),
        A.Compose([resize, A.VerticalFlip(p=1.0), norm, ToTensorV2()]),
        A.Compose([resize, A.RandomRotate90(p=1.0), norm, ToTensorV2()]),
    ]


# ============================================================
# DATASET
# ============================================================
class SkinDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        label = self.labels[idx] if self.labels is not None else -1
        return img, label


# ============================================================
# MODEL LOADING
# ============================================================
def load_classification_model(checkpoint_path, model_name=None):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if model_name is None:
        model_name = ckpt.get("config", {}).get("model_name", "tf_efficientnetv2_s")

    num_classes = ckpt.get("config", {}).get("num_classes", 12)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.train(False)

    val_acc = ckpt.get("val_acc", ckpt.get("val_f1", "N/A"))
    print(f"Loaded {model_name} | Val metric: {val_acc}")
    return model, model_name


def load_segmentation_model(checkpoint_path):
    if not HAS_SMP:
        print("segmentation_models_pytorch not installed")
        return None, None

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    arch = cfg.get("arch", "UnetPlusPlus")
    encoder = cfg.get("encoder", "efficientnet-b4")

    model_cls = getattr(smp, arch)
    model = model_cls(encoder_name=encoder, encoder_weights=None, classes=1, activation=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.train(False)

    val_iou = ckpt.get("val_iou", "N/A")
    print(f"Loaded {arch}/{encoder} | Val IoU: {val_iou}")
    return model, f"{arch}/{encoder}"


# ============================================================
# GRADCAM
# ============================================================
def get_target_layer(model, model_name):
    if not HAS_GRADCAM:
        return None

    if "efficientnet" in model_name:
        if hasattr(model, "blocks"):
            return [model.blocks[-1]]
        elif hasattr(model, "features"):
            return [model.features[-1]]
    elif "convnext" in model_name:
        return [model.stages[-1].blocks[-1]]
    elif "swin" in model_name:
        return [model.layers[-1].blocks[-1].norm1]
    elif "resnet" in model_name:
        return [model.layer4[-1]]

    for name in ["features", "blocks", "layer4", "stages"]:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Sequential):
                return [layer[-1]]
    return None


def generate_gradcam(model, model_name, image_path, target_class=None, output_path=None):
    if not HAS_GRADCAM:
        print("GradCAM not available. Install: pip install pytorch-grad-cam")
        return None

    target_layers = get_target_layer(model, model_name)
    if target_layers is None:
        print(f"Cannot find target layer for {model_name}")
        return None

    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_float = img_resized.astype(np.float32) / 255.0

    transform = get_val_transform()
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    # Map model index to real class
    real_pred_class = MODEL_IDX_TO_REAL.get(pred_class, pred_class)

    if target_class is None:
        target_class = pred_class  # keep model idx for GradCAM

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_resized)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    im = axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("GradCAM Heatmap", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(cam_image)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    class_name = CLASS_NAMES.get(real_pred_class, f"Class {real_pred_class}")
    class_type = CLASS_TYPES.get(real_pred_class, "unknown")
    color = "red" if class_type == "malignant" else "orange" if class_type == "pre-malignant" else "green"

    fig.suptitle(
        f"Prediction: {class_name} ({confidence:.1%}) -- {class_type.upper()}",
        fontsize=16, fontweight="bold", color=color
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved GradCAM to {output_path}")
    plt.close()

    top5_model_idxs = torch.argsort(probs, descending=True)[:5].tolist()
    top5 = []
    for mi in top5_model_idxs:
        ri = MODEL_IDX_TO_REAL.get(mi, mi)
        top5.append((ri, CLASS_NAMES.get(ri, f"Class {ri}"), probs[mi].item()))

    return {
        "pred_class": real_pred_class,
        "class_name": class_name,
        "confidence": confidence,
        "class_type": class_type,
        "top5": top5,
    }


# ============================================================
# BATCH EVALUATION
# ============================================================
def evaluate_on_folder(model, data_dir, output_dir):
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    if not classes:
        print(f"No class folders found in {data_dir}")
        return

    image_paths, labels = [], []
    for cls_name in classes:
        cls_dir = data_path / cls_name
        real_cls = int(cls_name)
        model_idx = REAL_TO_MODEL_IDX.get(real_cls, real_cls)
        for img_file in cls_dir.iterdir():
            if img_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                image_paths.append(str(img_file))
                labels.append(model_idx)  # use model's internal class idx

    dataset = SkinDataset(image_paths, labels, transform=get_val_transform())
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, lbls in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"\nOverall Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    target_names = [f"{i}: {CLASS_NAMES.get(i, f'Class {i}')}" for i in sorted(set(all_labels))]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(f"\n{report}")
    with open(output_path / "classification_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}\n\n{report}")

    plot_confusion_matrix(all_labels, all_preds, output_path / "confusion_matrix.png")
    plot_per_class_accuracy(all_labels, all_preds, output_path / "per_class_accuracy.png")
    plot_confidence_distribution(all_preds, all_labels, all_probs, output_path / "confidence_dist.png")
    plot_dangerous_misclassifications(all_labels, all_preds, output_path / "dangerous_errors.png")

    print(f"\nAll visualizations saved to {output_path}/")
    return acc, f1


# ============================================================
# VISUALIZATIONS
# ============================================================
def plot_confusion_matrix(y_true, y_pred, save_path):
    present_classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=[CLASS_NAMES.get(c, str(c)) for c in present_classes],
        yticklabels=[CLASS_NAMES.get(c, str(c)) for c in present_classes],
        ax=ax, vmin=0, vmax=1
    )

    for true_cls, pred_cls, desc in DANGEROUS_PAIRS:
        if true_cls in present_classes and pred_cls in present_classes:
            ti = present_classes.index(true_cls)
            pi = present_classes.index(pred_cls)
            val = cm_norm[ti, pi]
            if val > 0.01:
                ax.add_patch(plt.Rectangle((pi, ti), 1, 1, fill=False,
                             edgecolor="red", linewidth=3))

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (normalized) -- Red = Dangerous Errors", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_accuracy(y_true, y_pred, save_path):
    present_classes = sorted(set(y_true))
    accuracies = []
    names = []
    colors = []

    for cls in present_classes:
        mask = y_true == cls
        acc = (y_pred[mask] == cls).mean()
        accuracies.append(acc)
        names.append(f"{CLASS_NAMES.get(cls, str(cls))}\n(n={mask.sum()})")
        ct = CLASS_TYPES.get(cls, "benign")
        colors.append("#FF4444" if ct == "malignant" else "#FFA500" if ct == "pre-malignant" else "#44AA44")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(present_classes)), accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(present_classes)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Class Accuracy (Red=Malignant, Orange=Pre-malignant, Green=Benign)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(accuracies), color="blue", linestyle="--", label=f"Mean: {np.mean(accuracies):.3f}")
    ax.legend()

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confidence_distribution(preds, labels, probs, save_path):
    correct_mask = preds == labels
    max_conf = probs[np.arange(len(preds)), preds]
    correct_conf = max_conf[correct_mask]
    wrong_conf = max_conf[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(correct_conf, bins=30, alpha=0.7, color="green", label=f"Correct (n={len(correct_conf)})")
    axes[0].hist(wrong_conf, bins=30, alpha=0.7, color="red", label=f"Wrong (n={len(wrong_conf)})")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution: Correct vs Wrong")
    axes[0].legend()

    bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_confs = []

    for i in range(len(bins) - 1):
        mask = (max_conf >= bins[i]) & (max_conf < bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct_mask[mask].mean())
            bin_confs.append(max_conf[mask].mean())
        else:
            bin_accs.append(0)
            bin_confs.append((bins[i] + bins[i + 1]) / 2)

    axes[1].bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="steelblue", edgecolor="black")
    axes[1].plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    axes[1].set_xlabel("Mean Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Reliability Diagram (Calibration)")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_dangerous_misclassifications(y_true, y_pred, save_path):
    errors = []
    for true_cls, pred_cls, desc in DANGEROUS_PAIRS:
        mask = (y_true == true_cls) & (y_pred == pred_cls)
        count = mask.sum()
        total = (y_true == true_cls).sum()
        rate = count / total if total > 0 else 0
        errors.append({"pair": desc, "count": count, "total": total, "rate": rate})

    df = pd.DataFrame(errors)
    df = df.sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#FF4444" if r > 0.05 else "#FFA500" if r > 0.01 else "#44AA44" for r in df["rate"]]
    bars = ax.barh(df["pair"], df["rate"], color=colors, edgecolor="black", linewidth=0.5)

    for bar, count, total in zip(bars, df["count"], df["total"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{count}/{total}", va="center", fontsize=10)

    ax.set_xlabel("Misclassification Rate")
    ax.set_title("Clinically Dangerous Misclassifications")
    ax.set_xlim(0, max(df["rate"].max() * 1.3, 0.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# SEGMENTATION VISUALIZATION
# ============================================================
def visualize_segmentation(model, image_path, mask_path=None, output_path=None):
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()

    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    img_resized = cv2.resize(img_rgb, (256, 256))

    n_cols = 4 if mask_path else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(img_resized)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Probability Map")
    axes[1].axis("off")

    overlay = img_resized.copy()
    overlay[pred_binary == 1] = [255, 0, 0]
    blended = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)
    axes[2].imshow(blended)
    axes[2].set_title("Segmentation Overlay")
    axes[2].axis("off")

    if mask_path:
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (256, 256))
        gt_binary = (gt_mask > 127).astype(np.uint8)

        comparison = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
        comparison[gt_binary == 1] = [0, 255, 0]
        comparison[pred_binary == 1] = [255, 0, 0]
        comparison[(gt_binary == 1) & (pred_binary == 1)] = [255, 255, 0]

        intersection = (gt_binary & pred_binary).sum()
        union = (gt_binary | pred_binary).sum()
        iou = intersection / union if union > 0 else 0

        axes[3].imshow(comparison)
        axes[3].set_title(f"GT(green) vs Pred(red) | IoU: {iou:.3f}")
        axes[3].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MULTI-MODEL COMPARISON
# ============================================================
def compare_models(model_paths, data_dir, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for mpath in model_paths:
        mpath = Path(mpath)
        print(f"\n{'='*60}")
        print(f"Evaluating: {mpath.name}")
        model, model_name = load_classification_model(mpath)
        acc, f1 = evaluate_on_folder(model, data_dir, output_path / mpath.stem)
        results.append({"model": mpath.stem, "accuracy": acc, "f1": f1})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("MODEL COMPARISON:")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    ax.bar([i - 0.15 for i in x], df["accuracy"], 0.3, label="Accuracy", color="steelblue")
    ax.bar([i + 0.15 for i in x], df["f1"], 0.3, label="Macro F1", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=150)
    plt.close()

    df.to_csv(output_path / "model_comparison.csv", index=False)
    return df


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="WhiteCoat.dev Testing Platform")
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("predict", help="Predict single image with GradCAM")
    p1.add_argument("--model", required=True, help="Model checkpoint path")
    p1.add_argument("--image", required=True, help="Image path")
    p1.add_argument("--output", default="outputs/gradcam.png")

    p2 = sub.add_parser("evaluate", help="Evaluate on validation folder")
    p2.add_argument("--model", required=True, help="Model checkpoint path")
    p2.add_argument("--data", required=True, help="Data folder (with class subfolders)")
    p2.add_argument("--output", default="outputs/evaluation")

    p3 = sub.add_parser("compare", help="Compare multiple models")
    p3.add_argument("--models", nargs="+", required=True, help="Model checkpoint paths")
    p3.add_argument("--data", required=True, help="Data folder")
    p3.add_argument("--output", default="outputs/comparison")

    p4 = sub.add_parser("gradcam-batch", help="GradCAM for multiple images")
    p4.add_argument("--model", required=True, help="Model checkpoint path")
    p4.add_argument("--images", required=True, help="Folder with images")
    p4.add_argument("--output", default="outputs/gradcam_batch")
    p4.add_argument("--max", type=int, default=20, help="Max images to process")

    p5 = sub.add_parser("segment", help="Visualize segmentation")
    p5.add_argument("--model", required=True, help="Segmentation model checkpoint")
    p5.add_argument("--image", required=True, help="Image path")
    p5.add_argument("--mask", default=None, help="Ground truth mask (optional)")
    p5.add_argument("--output", default="outputs/segmentation.png")

    args = parser.parse_args()

    if args.command == "predict":
        model, model_name = load_classification_model(args.model)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result = generate_gradcam(model, model_name, args.image, output_path=args.output)
        if result:
            print(f"\nPrediction: {result['class_name']} ({result['confidence']:.1%})")
            print(f"Type: {result['class_type'].upper()}")
            print("Top-5:")
            for cls_id, name, prob in result["top5"]:
                print(f"  {name}: {prob:.1%}")

    elif args.command == "evaluate":
        model, model_name = load_classification_model(args.model)
        evaluate_on_folder(model, args.data, args.output)

    elif args.command == "compare":
        compare_models(args.models, args.data, args.output)

    elif args.command == "gradcam-batch":
        model, model_name = load_classification_model(args.model)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        img_dir = Path(args.images)
        images = sorted(img_dir.glob("*.png"))[:args.max]

        for img_path in tqdm(images, desc="Generating GradCAM"):
            out_path = output_dir / f"gradcam_{img_path.stem}.png"
            generate_gradcam(model, model_name, img_path, output_path=out_path)

    elif args.command == "segment":
        model, model_name = load_segmentation_model(args.model)
        if model:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            visualize_segmentation(model, args.image, args.mask, args.output)

    else:
        parser.print_help()
        print("\n--- Quick Examples ---")
        print("python test_platform.py predict --model all_models/best_tf_efficientnetv2_s.pth --image data/classification/test/100359.png")
        print("python test_platform.py evaluate --model all_models/best_tf_efficientnetv2_s.pth --data data/classification/train")
        print("python test_platform.py gradcam-batch --model all_models/best_tf_efficientnetv2_s.pth --images data/classification/test --max 10")
        print("python test_platform.py compare --models all_models/best_tf_efficientnetv2_s.pth all_models/best_swin_tiny_patch4_window7_224.pth --data data/classification/train")
        print("python test_platform.py segment --model all_models/best_UnetPlusPlus_efficientnet_b4.pth --image data/Segmentation/testing/images/10.png")


if __name__ == "__main__":
    main()
