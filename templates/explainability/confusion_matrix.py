"""
Clinical Confusion Matrix with Dangerous Misclassification Highlighting
Team: WhiteCoat.dev | Hackathon 2026

Creates a publication-quality confusion matrix where clinically dangerous
misclassifications (e.g., melanoma classified as benign nevus) are highlighted
in red to show the model understands safety-critical boundaries.

Usage:
    python -m explainability.confusion_matrix \
        --checkpoint checkpoints/best_tf_efficientnetv2_s.pth \
        --data_dir data/classification/train \
        --output outputs/confusion_matrix.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# Dangerous confusion pairs (domain knowledge)
# ---------------------------------------------------------------------------


def _identify_dangerous_pairs(
    class_names: list[str],
) -> list[tuple[int, int]]:
    """Auto-identify dangerous confusion pairs based on class names.

    Heuristic: any malignant/cancer class confused with benign/nevus is dangerous.
    """
    dangerous = []
    malignant_keywords = ["melanoma", "carcinoma", "malignant", "cancer", "scc", "bcc"]
    benign_keywords = ["nevus", "benign", "normal", "seborrheic", "dermatofibroma"]

    malignant_idx = []
    benign_idx = []

    for i, name in enumerate(class_names):
        name_lower = name.lower()
        if any(kw in name_lower for kw in malignant_keywords):
            malignant_idx.append(i)
        if any(kw in name_lower for kw in benign_keywords):
            benign_idx.append(i)

    for m in malignant_idx:
        for b in benign_idx:
            dangerous.append((m, b))

    return dangerous


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    output_path: str = "outputs/confusion_matrix.png",
    normalize: bool = True,
    dangerous_pairs: Optional[list[tuple[int, int]]] = None,
    title: str = "Confusion Matrix",
    dpi: int = 200,
    figsize: tuple[int, int] = (14, 12),
) -> None:
    """Generate a publication-quality confusion matrix with clinical annotations."""
    matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm_display / row_sums
    else:
        cm_display = cm.astype(float)

    if dangerous_pairs is None and class_names is not None:
        dangerous_pairs = _identify_dangerous_pairs(class_names)

    fig, ax = plt.subplots(figsize=figsize)

    annot_array = np.empty_like(cm_display, dtype=object)
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            if normalize:
                annot_array[i, j] = f"{cm_display[i, j]:.1%}\n({cm[i, j]})"
            else:
                annot_array[i, j] = f"{int(cm[i, j])}"

    mask = cm_display == 0
    sns.heatmap(
        cm_display,
        annot=annot_array,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        mask=mask,
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
        vmin=0,
        vmax=1.0 if normalize else cm.max(),
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    if dangerous_pairs:
        for true_idx, pred_idx in dangerous_pairs:
            if true_idx < cm.shape[0] and pred_idx < cm.shape[1]:
                if cm[true_idx, pred_idx] > 0:
                    ax.add_patch(plt.Rectangle(
                        (pred_idx, true_idx), 1, 1,
                        fill=False, edgecolor="red", linewidth=3,
                    ))

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    if dangerous_pairs:
        from matplotlib.patches import Rectangle
        legend_elements = [
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="red",
                      linewidth=3, label="Clinically Dangerous Confusion")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    output_path: str = "outputs/per_class_metrics.png",
    dpi: int = 200,
) -> None:
    """Bar chart of precision, recall, F1 per class."""
    matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    precisions = []
    recalls = []
    f1s = []
    labels_used = []

    for i, name in enumerate(class_names):
        key = str(i)
        if key in report:
            precisions.append(report[key]["precision"])
            recalls.append(report[key]["recall"])
            f1s.append(report[key]["f1-score"])
            labels_used.append(name)

    x = np.arange(len(labels_used))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#3498db",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, recalls, width, label="Recall", color="#e74c3c",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width, f1s, width, label="F1-Score", color="#2ecc71",
           edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_used, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def generate_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on entire dataset, return (y_true, y_pred)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Confusion Matrix | WhiteCoat.dev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="outputs/confusion_matrix.png")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--per_class", action="store_true",
                        help="Also generate per-class metrics bar chart")
    args = parser.parse_args()

    from explainability.gradcam import load_model
    from utils.augmentations import get_classification_transforms
    from classification.dataset import MedicalClassificationDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, num_classes, image_size = load_model(args.checkpoint, device)

    transform = get_classification_transforms(image_size, is_train=False)
    dataset = MedicalClassificationDataset.from_folder(args.data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4
    )

    print("Running inference...")
    y_true, y_pred = generate_predictions(model, loader, device)

    print(f"Accuracy: {(y_true == y_pred).mean():.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    plot_confusion_matrix(
        y_true, y_pred,
        output_path=args.output,
        normalize=not args.no_normalize,
    )
    print(f"Saved: {args.output}")

    if args.per_class:
        pc_path = args.output.replace(".png", "_per_class.png")
        plot_per_class_metrics(y_true, y_pred, output_path=pc_path)
        print(f"Saved: {pc_path}")


if __name__ == "__main__":
    main()
