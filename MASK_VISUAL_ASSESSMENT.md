# Mask Visual Assessment Guide for Skin Lesion Segmentation

> For research and demonstration purposes only. Not for clinical use.

## Table of Contents
1. [Visual Comparison Tools](#1-visual-comparison-tools)
2. [Common Failure Modes](#2-common-failure-modes-in-skin-lesion-segmentation)
3. [Per-Image Quality Scoring](#3-per-image-quality-scoring)
4. [Ablation Study Visualization](#4-ablation-study-visualization)
5. [Presentation-Ready Visualizations](#5-presentation-ready-visualizations)

---

## Dependencies

```python
# All code in this document requires:
# pip install opencv-python numpy matplotlib scikit-image scipy pandas seaborn
# pip install torch segmentation_models_pytorch albumentations
```

---

## 1. Visual Comparison Tools

### 1.1 Side-by-Side: Original | Predicted | Ground Truth | Overlay

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def side_by_side_comparison(
    image_path: str,
    pred_mask_path: str,
    gt_mask_path: str,
    save_path: str | None = None,
    figsize: tuple = (20, 5),
) -> None:
    """
    4-panel comparison: original | predicted mask | ground truth | overlay.

    Args:
        image_path: Path to the original image.
        pred_mask_path: Path to the predicted binary mask (0 or 255).
        gt_mask_path: Path to the ground truth binary mask.
        save_path: If provided, saves the figure to this path.
        figsize: Figure size tuple.
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pred = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize masks to match image if needed
    h, w = image.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt.shape != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binarize
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)

    # Create overlay: green = prediction boundary on original
    overlay = image.copy()
    pred_contours, _ = cv2.findContours(
        pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gt_contours, _ = cv2.findContours(
        gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, pred_contours, -1, (0, 255, 0), 2)   # green = pred
    cv2.drawContours(overlay, gt_contours, -1, (255, 0, 0), 2)     # red = GT

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    titles = ["Original", "Predicted Mask", "Ground Truth", "Overlay (G=pred, R=GT)"]
    images = [image, pred, gt, overlay]

    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### 1.2 Color-Coded Error Map: TP=Green, FP=Red, FN=Blue

This is the single most informative visualization for segmentation quality.
It shows exactly where the model succeeds and fails.

```python
def error_map_visualization(
    image_path: str,
    pred_mask_path: str,
    gt_mask_path: str,
    alpha: float = 0.5,
    save_path: str | None = None,
) -> dict:
    """
    Color-coded error overlay on original image.

    Colors:
        Green  = True Positive  (correctly segmented lesion)
        Red    = False Positive (model says lesion, actually background)
        Blue   = False Negative (model missed this part of lesion)

    Returns:
        Dictionary with pixel counts for TP, FP, FN, TN and IoU.
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pred = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt.shape != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)

    # Compute confusion regions
    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
    fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin

    # Create color overlay
    color_overlay = np.zeros_like(image)
    color_overlay[tp == 1] = [0, 255, 0]    # Green = TP
    color_overlay[fp == 1] = [255, 0, 0]    # Red = FP
    color_overlay[fn == 1] = [0, 0, 255]    # Blue = FN

    # Blend with original
    blended = image.copy()
    mask_any = (tp | fp | fn).astype(bool)
    blended[mask_any] = (
        (1 - alpha) * image[mask_any] + alpha * color_overlay[mask_any]
    ).astype(np.uint8)

    # Compute metrics
    tp_count = int(tp.sum())
    fp_count = int(fp.sum())
    fn_count = int(fn.sum())
    tn_count = int(h * w - tp_count - fp_count - fn_count)
    iou = tp_count / max(tp_count + fp_count + fn_count, 1)
    dice = 2 * tp_count / max(2 * tp_count + fp_count + fn_count, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title(
        f"Error Map (IoU={iou:.3f}, Dice={dice:.3f})",
        fontweight="bold",
    )
    axes[1].axis("off")

    # Legend
    legend_img = np.zeros((100, 300, 3), dtype=np.uint8)
    legend_img[10:30, 10:40] = [0, 255, 0]
    legend_img[40:60, 10:40] = [255, 0, 0]
    legend_img[70:90, 10:40] = [0, 0, 255]
    axes[2].imshow(legend_img)
    axes[2].text(50, 20, f"True Positive  ({tp_count:,} px)", color="white",
                 fontsize=11, va="center")
    axes[2].text(50, 50, f"False Positive ({fp_count:,} px)", color="white",
                 fontsize=11, va="center")
    axes[2].text(50, 80, f"False Negative ({fn_count:,} px)", color="white",
                 fontsize=11, va="center")
    axes[2].set_title("Legend", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return {
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
        "iou": iou,
        "dice": dice,
    }
```

### 1.3 Boundary Visualization: Predicted Contour on Original

Boundary quality is critical for dermatology -- clinicians assess lesion borders
(the "B" in the ABCDE rule for melanoma). Showing contour accuracy demonstrates
clinical relevance.

```python
def boundary_visualization(
    image_path: str,
    pred_mask_path: str,
    gt_mask_path: str | None = None,
    contour_thickness: int = 2,
    save_path: str | None = None,
) -> None:
    """
    Draw predicted (and optionally GT) contours on original image.
    Predicted = green solid, GT = red dashed (approximated with dots).
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pred = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bin = (pred > 127).astype(np.uint8)
    overlay = image.copy()

    # Predicted contour (green)
    contours_pred, _ = cv2.findContours(
        pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours_pred, -1, (0, 255, 0), contour_thickness)

    if gt_mask_path:
        gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 127).astype(np.uint8)
        contours_gt, _ = cv2.findContours(
            gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours_gt, -1, (255, 0, 0), contour_thickness)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    title = "Boundary: Green=Predicted"
    if gt_mask_path:
        title += ", Red=GT"
    axes[1].set_title(title, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### 1.4 Difference Heatmap Between Predicted and GT

```python
from scipy.ndimage import distance_transform_edt


def difference_heatmap(
    pred_mask_path: str,
    gt_mask_path: str,
    image_path: str | None = None,
    save_path: str | None = None,
) -> None:
    """
    Heatmap showing distance-weighted disagreement between pred and GT.
    Brighter regions = further from correct boundary.

    This highlights not just THAT the model was wrong, but HOW FAR off it was.
    Useful for distinguishing minor boundary errors from completely missed regions.
    """
    pred = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = pred.shape[:2]
    if gt.shape != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bin = (pred > 127).astype(np.float32)
    gt_bin = (gt > 127).astype(np.float32)

    # Simple difference
    diff = np.abs(pred_bin - gt_bin)

    # Distance-weighted difference: how far each error pixel is from the
    # nearest correct boundary
    if diff.sum() > 0:
        gt_boundary = cv2.Canny((gt_bin * 255).astype(np.uint8), 100, 200)
        dist_from_boundary = distance_transform_edt(gt_boundary == 0)
        weighted_diff = diff * np.clip(dist_from_boundary, 0, 50) / 50.0
    else:
        weighted_diff = diff

    ncols = 3 if image_path else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    idx = 0
    if image_path:
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        axes[idx].imshow(image)
        axes[idx].set_title("Original", fontweight="bold")
        axes[idx].axis("off")
        idx += 1

    im1 = axes[idx].imshow(diff, cmap="Reds")
    axes[idx].set_title("Binary Difference", fontweight="bold")
    axes[idx].axis("off")
    plt.colorbar(im1, ax=axes[idx], fraction=0.046)

    idx += 1
    im2 = axes[idx].imshow(weighted_diff, cmap="hot")
    axes[idx].set_title("Distance-Weighted Error", fontweight="bold")
    axes[idx].axis("off")
    plt.colorbar(im2, ax=axes[idx], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### 1.5 Batch Visualization: Grid of Multiple Samples

```python
def batch_visualization_grid(
    image_paths: list[str],
    pred_mask_paths: list[str],
    gt_mask_paths: list[str],
    n_samples: int = 6,
    save_path: str | None = None,
) -> None:
    """
    Grid view: rows = samples, columns = [Original, Pred, GT, Error Map].
    Use this for presentation slides showing model quality at a glance.
    """
    n = min(n_samples, len(image_paths))

    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Predicted", "Ground Truth", "Error Map"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14, fontweight="bold")

    for i in range(n):
        image = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)
        pred = cv2.imread(pred_mask_paths[i], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_mask_paths[i], cv2.IMREAD_GRAYSCALE)

        h, w = image.shape[:2]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        if gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        pred_bin = (pred > 127).astype(np.uint8)
        gt_bin = (gt > 127).astype(np.uint8)

        # Error map
        tp = pred_bin & gt_bin
        fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
        fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin
        error_overlay = image.copy()
        error_color = np.zeros_like(image)
        error_color[tp == 1] = [0, 255, 0]
        error_color[fp == 1] = [255, 0, 0]
        error_color[fn == 1] = [0, 0, 255]
        mask_any = (tp | fp | fn).astype(bool)
        error_overlay[mask_any] = (
            0.5 * image[mask_any] + 0.5 * error_color[mask_any]
        ).astype(np.uint8)

        iou = int(tp.sum()) / max(int(tp.sum()) + int(fp.sum()) + int(fn.sum()), 1)

        axes[i, 0].imshow(image)
        axes[i, 1].imshow(pred, cmap="gray")
        axes[i, 2].imshow(gt, cmap="gray")
        axes[i, 3].imshow(error_overlay)
        axes[i, 3].text(
            5, 20, f"IoU={iou:.3f}", color="white", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

---

## 2. Common Failure Modes in Skin Lesion Segmentation

### Failure Mode Detector

```python
def detect_failure_modes(
    image_path: str,
    pred_mask_path: str,
    gt_mask_path: str,
) -> dict:
    """
    Analyze a prediction to identify likely failure modes.
    Returns a dictionary of detected issues and severity scores.

    Failure modes checked:
    1. Hair artifacts: high-frequency edges outside lesion boundary
    2. Low contrast: small gradient magnitude at lesion boundary
    3. Over-segmentation: FP ratio is high
    4. Under-segmentation: FN ratio is high
    5. Multiple components: more than one connected component in prediction
    6. Boundary irregularity: high ratio of perimeter to area (could be noisy)
    7. Ruler/ink markings: linear structures in FP regions
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pred = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt.shape != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)

    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
    fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin

    tp_sum = int(tp.sum())
    fp_sum = int(fp.sum())
    fn_sum = int(fn.sum())
    gt_sum = int(gt_bin.sum())
    pred_sum = int(pred_bin.sum())

    issues = {}

    # 1. Over-segmentation check
    if pred_sum > 0:
        fp_ratio = fp_sum / pred_sum
        if fp_ratio > 0.3:
            issues["over_segmentation"] = {
                "severity": "high" if fp_ratio > 0.5 else "medium",
                "fp_ratio": round(fp_ratio, 3),
                "description": "Model predicts lesion where there is none",
            }

    # 2. Under-segmentation check
    if gt_sum > 0:
        fn_ratio = fn_sum / gt_sum
        if fn_ratio > 0.3:
            issues["under_segmentation"] = {
                "severity": "high" if fn_ratio > 0.5 else "medium",
                "fn_ratio": round(fn_ratio, 3),
                "description": "Model misses part of the lesion",
            }

    # 3. Multiple connected components in prediction
    num_components, labels = cv2.connectedComponents(pred_bin)
    if num_components > 2:  # background + 1 lesion expected
        issues["multiple_components"] = {
            "severity": "medium",
            "count": num_components - 1,
            "description": "Multiple disconnected predicted regions",
        }

    # 4. Low contrast at GT boundary
    gt_boundary = cv2.dilate(gt_bin, np.ones((5, 5))) - cv2.erode(
        gt_bin, np.ones((5, 5))
    )
    if gt_boundary.sum() > 0:
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        boundary_gradient = np.abs(gradient[gt_boundary > 0]).mean()
        if boundary_gradient < 15:
            issues["low_contrast"] = {
                "severity": "high" if boundary_gradient < 8 else "medium",
                "mean_gradient": round(float(boundary_gradient), 2),
                "description": "Low contrast between lesion and surrounding skin",
            }

    # 5. Hair artifacts: high-frequency content in FP regions
    if fp_sum > 100:
        edges = cv2.Canny(gray, 50, 150)
        hair_like_edges = edges & (fp > 0).astype(np.uint8)
        edge_density = hair_like_edges.sum() / max(fp_sum, 1)
        if edge_density > 0.05:
            issues["hair_artifacts"] = {
                "severity": "medium",
                "edge_density": round(float(edge_density), 4),
                "description": "Hair or artifact edges may cause false boundaries",
            }

    # 6. Boundary irregularity
    contours, _ = cv2.findContours(
        pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours and pred_sum > 100:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        area = cv2.contourArea(largest)
        if area > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < 0.2:
                issues["irregular_boundary"] = {
                    "severity": "medium",
                    "circularity": round(float(circularity), 3),
                    "description": "Predicted boundary is very irregular/noisy",
                }

    # 7. Empty prediction
    if pred_sum == 0 and gt_sum > 0:
        issues["empty_prediction"] = {
            "severity": "critical",
            "description": "Model produced an empty mask for an image with a lesion",
        }

    return issues


def visualize_failure_mode(
    image_path: str,
    pred_mask_path: str,
    gt_mask_path: str,
    save_path: str | None = None,
) -> None:
    """
    Combined visualization: error map + detected failure modes as text annotations.
    """
    issues = detect_failure_modes(image_path, pred_mask_path, gt_mask_path)

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pred = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]
    if pred.shape != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt.shape != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)

    # Build error overlay
    tp = pred_bin & gt_bin
    fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
    fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin

    error_overlay = image.copy()
    error_color = np.zeros_like(image)
    error_color[tp == 1] = [0, 255, 0]
    error_color[fp == 1] = [255, 0, 0]
    error_color[fn == 1] = [0, 0, 255]
    mask_any = (tp | fp | fn).astype(bool)
    error_overlay[mask_any] = (
        0.5 * image[mask_any] + 0.5 * error_color[mask_any]
    ).astype(np.uint8)

    iou = int(tp.sum()) / max(int(tp.sum()) + int(fp.sum()) + int(fn.sum()), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(error_overlay)
    axes[0].set_title(f"Error Map (IoU={iou:.3f})", fontweight="bold")
    axes[0].axis("off")

    # Issues panel
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    axes[1].set_title("Detected Issues", fontweight="bold", fontsize=14)

    if not issues:
        axes[1].text(0.1, 0.5, "No issues detected", fontsize=14, color="green")
    else:
        y = 0.9
        severity_colors = {"critical": "darkred", "high": "red", "medium": "orange"}
        for name, info in issues.items():
            color = severity_colors.get(info["severity"], "gray")
            axes[1].text(
                0.05, y,
                f"[{info['severity'].upper()}] {name}",
                fontsize=13, color=color, fontweight="bold",
            )
            axes[1].text(
                0.05, y - 0.06,
                info["description"],
                fontsize=11, color="black",
            )
            y -= 0.15

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### Typical Skin Lesion Failure Categories

| Failure Mode | Visual Clue | Cause | Mitigation |
|---|---|---|---|
| **Hair artifacts** | Thin linear FP regions radiating from lesion | Dark hairs create edge-like features | Hair removal preprocessing (DullRazor), morphological opening, CLAHE |
| **Low contrast** | Large FN at lesion periphery | Lesion blends into skin tone | CLAHE preprocessing, multi-scale input, attention mechanisms |
| **Irregular borders** | Jagged predicted boundary, noisy edges | Melanoma with diffuse borders | Larger receptive field, CRF post-processing, boundary-aware loss |
| **Multiple lesions** | Several disconnected predicted regions | Two lesions in frame, or satellites | Keep-largest-component post-processing, instance segmentation |
| **Dark skin** | Under-segmentation (large FN) | Training data bias toward lighter skin | Augment with color jittering, balance training skin tones |
| **Ruler/ink marks** | Linear FP structures, often at image edges | Measurement tools included in photo | Artifact-aware augmentation, edge masking |
| **Over-segmentation** | Large red (FP) halo around lesion | Low threshold, model too sensitive | Raise threshold, morphological erosion, CRF |
| **Under-segmentation** | Large blue (FN) inside lesion | High threshold, uneven coloring | Lower threshold, multi-scale features |

---

## 3. Per-Image Quality Scoring

### 3.1 Compute IoU for Every Image and Sort

```python
import pandas as pd


def compute_per_image_iou(
    image_dir: str,
    pred_dir: str,
    gt_dir: str,
) -> pd.DataFrame:
    """
    Compute IoU, Dice, and pixel counts for every image.
    Returns a DataFrame sorted by IoU (worst first) for easy triage.

    Args:
        image_dir: Directory with original images.
        pred_dir: Directory with predicted masks.
        gt_dir: Directory with ground truth masks.
    """
    image_dir = Path(image_dir)
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    results = []
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    gt_files = sorted([
        f for f in gt_dir.iterdir() if f.suffix.lower() in extensions
    ])

    for gt_path in gt_files:
        stem = gt_path.stem
        pred_path = pred_dir / f"{stem}.png"

        if not pred_path.exists():
            # Try other extensions
            found = False
            for ext in extensions:
                candidate = pred_dir / f"{stem}{ext}"
                if candidate.exists():
                    pred_path = candidate
                    found = True
                    break
            if not found:
                results.append({
                    "image_id": stem,
                    "iou": 0.0,
                    "dice": 0.0,
                    "tp": 0, "fp": 0, "fn": 0,
                    "pred_coverage": 0.0,
                    "gt_coverage": 0.0,
                    "status": "MISSING_PRED",
                })
                continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred, (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (pred > 127).astype(np.uint8)

        tp = int((pred_bin & gt_bin).sum())
        fp = int((pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)).sum())
        fn = int(((~pred_bin.astype(bool)).astype(np.uint8) & gt_bin).sum())

        iou = tp / max(tp + fp + fn, 1)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        total_pixels = gt.shape[0] * gt.shape[1]

        results.append({
            "image_id": stem,
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pred_coverage": round(pred_bin.sum() / total_pixels, 4),
            "gt_coverage": round(gt_bin.sum() / total_pixels, 4),
            "status": "OK",
        })

    df = pd.DataFrame(results)
    df = df.sort_values("iou", ascending=True).reset_index(drop=True)
    return df


def print_quality_summary(df: pd.DataFrame) -> None:
    """Print a summary report from the per-image IoU DataFrame."""
    total = len(df)
    mean_iou = df["iou"].mean()
    mean_dice = df["dice"].mean()

    # Quality bins
    excellent = (df["iou"] >= 0.9).sum()
    good = ((df["iou"] >= 0.7) & (df["iou"] < 0.9)).sum()
    mediocre = ((df["iou"] >= 0.5) & (df["iou"] < 0.7)).sum()
    poor = ((df["iou"] >= 0.2) & (df["iou"] < 0.5)).sum()
    terrible = (df["iou"] < 0.2).sum()
    missing = (df["status"] == "MISSING_PRED").sum()

    print("=" * 60)
    print("SEGMENTATION QUALITY REPORT")
    print("=" * 60)
    print(f"Total images:    {total}")
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(f"Mean Dice:       {mean_dice:.4f}")
    print(f"Median IoU:      {df['iou'].median():.4f}")
    print(f"Std IoU:         {df['iou'].std():.4f}")
    print("-" * 60)
    print(f"Excellent (>=0.9): {excellent:4d}  ({excellent/total*100:5.1f}%)")
    print(f"Good    (0.7-0.9): {good:4d}  ({good/total*100:5.1f}%)")
    print(f"Mediocre(0.5-0.7): {mediocre:4d}  ({mediocre/total*100:5.1f}%)")
    print(f"Poor    (0.2-0.5): {poor:4d}  ({poor/total*100:5.1f}%)")
    print(f"Terrible  (<0.2):  {terrible:4d}  ({terrible/total*100:5.1f}%)")
    print(f"Missing pred:      {missing:4d}  ({missing/total*100:5.1f}%)")
    print("-" * 60)
    print("\nWorst 10 images:")
    worst = df.head(10)[["image_id", "iou", "dice", "status"]]
    print(worst.to_string(index=False))
    print("=" * 60)
```

### 3.2 IoU Distribution Plot

```python
import seaborn as sns


def plot_iou_distribution(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Histogram + KDE of per-image IoU scores.
    Red dashed line = mean, blue dashed line = median.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histogram
    sns.histplot(df["iou"], bins=50, kde=True, ax=axes[0], color="steelblue")
    axes[0].axvline(df["iou"].mean(), color="red", linestyle="--",
                    label=f"Mean={df['iou'].mean():.3f}")
    axes[0].axvline(df["iou"].median(), color="blue", linestyle="--",
                    label=f"Median={df['iou'].median():.3f}")
    axes[0].set_xlabel("IoU", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("IoU Distribution", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)

    # Sorted IoU plot (useful for identifying a "cliff" of bad predictions)
    sorted_iou = df["iou"].sort_values(ascending=False).values
    axes[1].plot(range(len(sorted_iou)), sorted_iou, linewidth=1.5)
    axes[1].axhline(0.5, color="red", linestyle=":", alpha=0.7, label="IoU=0.5")
    axes[1].fill_between(
        range(len(sorted_iou)), sorted_iou,
        alpha=0.3, color="steelblue",
    )
    axes[1].set_xlabel("Image Index (sorted)", fontsize=12)
    axes[1].set_ylabel("IoU", fontsize=12)
    axes[1].set_title("Sorted IoU Curve", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### 3.3 Automatic Flagging for Manual Review

```python
def flag_for_review(
    df: pd.DataFrame,
    iou_threshold: float = 0.5,
    coverage_ratio_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Flag images that likely need manual review or model retraining.

    Criteria:
    1. IoU below threshold (obvious bad predictions)
    2. Empty predictions when GT is non-empty
    3. Prediction coverage vastly different from GT coverage
    4. Missing predictions

    Returns a filtered DataFrame with a 'flag_reason' column.
    """
    flags = []

    for _, row in df.iterrows():
        reasons = []

        if row["status"] == "MISSING_PRED":
            reasons.append("missing_prediction")
        elif row["iou"] < iou_threshold:
            reasons.append(f"low_iou_{row['iou']:.3f}")

        if row["gt_coverage"] > 0.01 and row["pred_coverage"] < 0.001:
            reasons.append("empty_prediction")

        if row["gt_coverage"] > 0 and row["pred_coverage"] > 0:
            ratio = row["pred_coverage"] / row["gt_coverage"]
            if ratio > coverage_ratio_threshold:
                reasons.append(f"over_segmented_{ratio:.1f}x")
            elif ratio < 1.0 / coverage_ratio_threshold:
                reasons.append(f"under_segmented_{ratio:.2f}x")

        if reasons:
            flags.append({**row, "flag_reason": " | ".join(reasons)})

    flagged = pd.DataFrame(flags)
    print(f"Flagged {len(flagged)} / {len(df)} images for review")
    return flagged
```

---

## 4. Ablation Study Visualization

### 4.1 Show Effect of Each Post-Processing Step

```python
def visualize_postprocessing_ablation(
    image_path: str,
    prob_mask: np.ndarray,
    gt_mask_path: str,
    threshold: float = 0.5,
    save_path: str | None = None,
) -> None:
    """
    Show the effect of each post-processing step side by side.

    Pipeline stages:
    1. Raw probability map
    2. Binary threshold
    3. Morphological opening (remove small FP)
    4. Keep largest connected component
    5. Morphological closing (fill holes)

    Each stage shows the mask AND its IoU vs GT.

    Args:
        prob_mask: Float array (H, W) with values in [0, 1] from model output.
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]

    if prob_mask.shape != (h, w):
        prob_mask = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    if gt.shape != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    gt_bin = (gt > 127).astype(np.uint8)

    def compute_iou(pred_binary: np.ndarray) -> float:
        tp = int((pred_binary & gt_bin).sum())
        fp = int((pred_binary & (~gt_bin.astype(bool)).astype(np.uint8)).sum())
        fn = int(((~pred_binary.astype(bool)).astype(np.uint8) & gt_bin).sum())
        return tp / max(tp + fp + fn, 1)

    # Stage 1: Raw probabilities (shown as heatmap)
    stage1_name = "Raw Probability"
    stage1_vis = prob_mask

    # Stage 2: Binary threshold
    binary = (prob_mask > threshold).astype(np.uint8)
    stage2_iou = compute_iou(binary)

    # Stage 3: Morphological opening (remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    stage3_iou = compute_iou(opened)

    # Stage 4: Keep largest connected component
    num_components, labels = cv2.connectedComponents(opened)
    if num_components > 2:
        # Find the largest non-background component
        sizes = [
            (labels == i).sum() for i in range(1, num_components)
        ]
        largest_label = np.argmax(sizes) + 1
        largest_cc = (labels == largest_label).astype(np.uint8)
    else:
        largest_cc = opened
    stage4_iou = compute_iou(largest_cc)

    # Stage 5: Morphological closing (fill holes)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(largest_cc, cv2.MORPH_CLOSE, kernel_close)
    stage5_iou = compute_iou(closed)

    # Visualization
    stages = [
        ("Probability Map", prob_mask, None, "hot"),
        (f"Threshold ({threshold})", binary * 255, stage2_iou, "gray"),
        ("Morph Opening", opened * 255, stage3_iou, "gray"),
        ("Largest Component", largest_cc * 255, stage4_iou, "gray"),
        ("Morph Closing", closed * 255, stage5_iou, "gray"),
        ("Ground Truth", gt, None, "gray"),
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))

    for i, (title, vis, iou, cmap) in enumerate(stages):
        axes[i].imshow(vis, cmap=cmap)
        label = title
        if iou is not None:
            label += f"\nIoU={iou:.3f}"
        axes[i].set_title(label, fontsize=10, fontweight="bold")
        axes[i].axis("off")

    plt.suptitle(
        "Post-Processing Ablation",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### 4.2 Threshold Sweep Visualization

```python
def threshold_sweep_visualization(
    prob_mask: np.ndarray,
    gt_mask_path: str,
    image_path: str | None = None,
    thresholds: list[float] | None = None,
    save_path: str | None = None,
) -> float:
    """
    Show how different thresholds affect the binary mask and IoU.
    Returns the optimal threshold.

    This is extremely important for hackathon submissions -- a threshold
    change from 0.5 to 0.45 can improve mean IoU by 1-3 points.
    """
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = gt.shape[:2]

    if prob_mask.shape != (h, w):
        prob_mask = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    gt_bin = (gt > 127).astype(np.uint8)

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]

    ious = []
    for t in thresholds:
        pred_bin = (prob_mask > t).astype(np.uint8)
        tp = int((pred_bin & gt_bin).sum())
        fp = int((pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)).sum())
        fn = int(((~pred_bin.astype(bool)).astype(np.uint8) & gt_bin).sum())
        iou = tp / max(tp + fp + fn, 1)
        ious.append(iou)

    best_idx = np.argmax(ious)
    best_threshold = thresholds[best_idx]
    best_iou = ious[best_idx]

    # Select a few thresholds to show visually
    show_thresholds = [0.2, 0.4, best_threshold, 0.6, 0.8]
    show_thresholds = sorted(set(show_thresholds))

    ncols = len(show_thresholds) + 1  # +1 for the IoU curve
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    for i, t in enumerate(show_thresholds):
        pred_bin = (prob_mask > t).astype(np.uint8)
        tp = int((pred_bin & gt_bin).sum())
        fp = int((pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)).sum())
        fn = int(((~pred_bin.astype(bool)).astype(np.uint8) & gt_bin).sum())
        iou = tp / max(tp + fp + fn, 1)
        axes[i].imshow(pred_bin * 255, cmap="gray")
        color = "green" if t == best_threshold else "black"
        axes[i].set_title(f"t={t:.2f}\nIoU={iou:.3f}", fontsize=10,
                          fontweight="bold", color=color)
        axes[i].axis("off")

    # IoU vs threshold curve
    axes[-1].plot(thresholds, ious, "bo-", linewidth=2, markersize=6)
    axes[-1].axvline(best_threshold, color="green", linestyle="--",
                     label=f"Best t={best_threshold:.2f}")
    axes[-1].set_xlabel("Threshold")
    axes[-1].set_ylabel("IoU")
    axes[-1].set_title("IoU vs Threshold", fontsize=10, fontweight="bold")
    axes[-1].legend(fontsize=9)
    axes[-1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Threshold Sweep (Best IoU={best_iou:.3f} at t={best_threshold:.2f})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return best_threshold
```

### 4.3 Resolution Effect Visualization

```python
def resolution_effect_visualization(
    image_path: str,
    model,
    transform_fn,
    gt_mask_path: str,
    device: str = "cpu",
    resolutions: list[int] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Show how input resolution affects segmentation quality.
    Tests the same image at multiple resolutions.

    Args:
        model: Loaded PyTorch segmentation model (in eval mode).
        transform_fn: Callable(image_size) -> albumentations transform.
        resolutions: List of image sizes to test.
    """
    import torch

    if resolutions is None:
        resolutions = [128, 256, 384, 512, 640, 768]

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    gt_bin = (gt > 127).astype(np.uint8)

    ious = []
    masks = []

    for res in resolutions:
        transform = transform_fn(res)
        augmented = transform(image=image)
        tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = output.squeeze().cpu().numpy()

        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_bin = (prob > 0.5).astype(np.uint8)

        tp = int((pred_bin & gt_bin).sum())
        fp = int((pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)).sum())
        fn = int(((~pred_bin.astype(bool)).astype(np.uint8) & gt_bin).sum())
        iou = tp / max(tp + fp + fn, 1)

        ious.append(iou)
        masks.append(pred_bin * 255)

    fig, axes = plt.subplots(1, len(resolutions) + 1, figsize=(4 * (len(resolutions) + 1), 4))

    for i, (res, mask, iou) in enumerate(zip(resolutions, masks, ious)):
        axes[i].imshow(mask, cmap="gray")
        axes[i].set_title(f"{res}x{res}\nIoU={iou:.3f}", fontsize=10, fontweight="bold")
        axes[i].axis("off")

    axes[-1].plot(resolutions, ious, "ro-", linewidth=2, markersize=8)
    axes[-1].set_xlabel("Resolution")
    axes[-1].set_ylabel("IoU")
    axes[-1].set_title("Resolution vs IoU", fontsize=10, fontweight="bold")
    axes[-1].grid(True, alpha=0.3)

    plt.suptitle("Effect of Input Resolution on Segmentation Quality",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### 4.4 Probability Heatmap Overlay

```python
def probability_heatmap_overlay(
    image_path: str,
    prob_mask: np.ndarray,
    gt_mask_path: str | None = None,
    threshold: float = 0.5,
    save_path: str | None = None,
) -> None:
    """
    Overlay the model's probability map (not binary) on the original image.

    This reveals:
    - Where the model is confident vs uncertain
    - Boundary regions where probability drops gradually (good) vs sharply (potentially noisy)
    - Internal areas of low confidence (potential holes in prediction)

    Much more informative than binary masks for understanding model behavior.
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    if prob_mask.shape != (h, w):
        prob_mask = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    ncols = 4 if gt_mask_path else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    # Probability heatmap alone
    im = axes[1].imshow(prob_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Probability Map", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Probability overlaid on original
    heatmap_colored = plt.cm.jet(prob_mask)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    # Only overlay where probability > 0.1
    mask_significant = prob_mask > 0.1
    overlay = image.copy()
    overlay[mask_significant] = (
        0.4 * image[mask_significant] + 0.6 * heatmap_colored[mask_significant]
    ).astype(np.uint8)

    # Draw threshold contour
    binary_at_threshold = (prob_mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_at_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (contour at t={threshold})", fontweight="bold")
    axes[2].axis("off")

    if gt_mask_path:
        gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        axes[3].imshow(gt, cmap="gray")
        axes[3].set_title("Ground Truth", fontweight="bold")
        axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

---

## 5. Presentation-Ready Visualizations

### 5.1 What Judges Want to See

**For a segmentation task in a hackathon, judges evaluate:**

1. **Overall metrics** -- Mean IoU, Mean Dice, shown prominently
2. **Distribution of quality** -- histogram showing most images are high IoU
3. **Good examples** -- cherry-picked cases with IoU > 0.9 (shows the model works)
4. **Bad examples** -- cherry-picked cases with low IoU (shows self-awareness and honesty)
5. **Error analysis** -- why the model fails (with color-coded error maps)
6. **Ablation** -- what each design decision contributed
7. **Post-processing pipeline** -- before/after for each step

**Honesty about failures impresses judges more than hiding them.**

### 5.2 Summary Slide Figure

```python
def create_summary_figure(
    df: pd.DataFrame,
    image_dir: str,
    pred_dir: str,
    gt_dir: str,
    n_good: int = 3,
    n_bad: int = 3,
    save_path: str | None = None,
) -> None:
    """
    Create a single publication-quality figure for presentation slides.

    Layout:
    - Top row: best examples with error maps
    - Bottom row: worst examples with error maps
    - Right panel: IoU distribution

    This is the single most important figure for your presentation.
    """
    image_dir = Path(image_dir)
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    df_sorted = df.sort_values("iou", ascending=False)
    best_ids = df_sorted.head(n_good)["image_id"].tolist()
    worst_ids = df_sorted.tail(n_bad)["image_id"].tolist()

    total_rows = n_good + n_bad
    fig = plt.figure(figsize=(24, 4 * total_rows + 2))
    gs = fig.add_gridspec(
        total_rows + 1, 5,
        width_ratios=[1, 1, 1, 1, 1.5],
        hspace=0.3, wspace=0.2,
    )

    def find_image(stem, directory, extensions=(".png", ".jpg", ".jpeg", ".bmp")):
        for ext in extensions:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)
        return None

    def plot_sample(row_idx, image_id, label_prefix):
        img_path = find_image(image_id, image_dir)
        pred_path = find_image(image_id, pred_dir)
        gt_path = find_image(image_id, gt_dir)

        if not all([img_path, pred_path, gt_path]):
            return

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        pred_raw = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        h, w = image.shape[:2]
        if pred_raw.shape != (h, w):
            pred_raw = cv2.resize(pred_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        if gt_raw.shape != (h, w):
            gt_raw = cv2.resize(gt_raw, (w, h), interpolation=cv2.INTER_NEAREST)

        pred_bin = (pred_raw > 127).astype(np.uint8)
        gt_bin = (gt_raw > 127).astype(np.uint8)

        tp = pred_bin & gt_bin
        fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
        fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin

        error_overlay = image.copy()
        error_color = np.zeros_like(image)
        error_color[tp == 1] = [0, 255, 0]
        error_color[fp == 1] = [255, 0, 0]
        error_color[fn == 1] = [0, 0, 255]
        mask_any = (tp | fp | fn).astype(bool)
        error_overlay[mask_any] = (
            0.5 * image[mask_any] + 0.5 * error_color[mask_any]
        ).astype(np.uint8)

        iou = int(tp.sum()) / max(int(tp.sum()) + int(fp.sum()) + int(fn.sum()), 1)

        panels = [image, pred_raw, gt_raw, error_overlay]
        titles_row = [
            f"{label_prefix}: {image_id}",
            "Predicted",
            "Ground Truth",
            f"Error (IoU={iou:.3f})",
        ]

        for col, (panel, title) in enumerate(zip(panels, titles_row)):
            ax = fig.add_subplot(gs[row_idx, col])
            if panel.ndim == 2:
                ax.imshow(panel, cmap="gray")
            else:
                ax.imshow(panel)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

    # Plot best examples
    for i, image_id in enumerate(best_ids):
        plot_sample(i, image_id, "BEST")

    # Plot worst examples
    for i, image_id in enumerate(worst_ids):
        plot_sample(n_good + i, image_id, "WORST")

    # IoU distribution on the right
    ax_dist = fig.add_subplot(gs[:total_rows, 4])
    sns.histplot(df["iou"], bins=40, kde=True, ax=ax_dist, color="steelblue")
    ax_dist.axvline(df["iou"].mean(), color="red", linestyle="--",
                    label=f"Mean IoU={df['iou'].mean():.3f}")
    ax_dist.set_xlabel("IoU", fontsize=12)
    ax_dist.set_ylabel("Count", fontsize=12)
    ax_dist.set_title("IoU Distribution", fontsize=13, fontweight="bold")
    ax_dist.legend(fontsize=10)

    # Legend row
    ax_legend = fig.add_subplot(gs[total_rows, :4])
    ax_legend.axis("off")
    ax_legend.text(
        0.5, 0.5,
        "Error Map Legend:  GREEN = True Positive  |  RED = False Positive  |  BLUE = False Negative",
        ha="center", va="center", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.suptitle(
        f"Segmentation Results Summary  |  Mean IoU = {df['iou'].mean():.4f}  |  Mean Dice = {df['dice'].mean():.4f}",
        fontsize=16, fontweight="bold", y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

### 5.3 Before/After Comparison for Specific Improvements

```python
def before_after_comparison(
    image_path: str,
    gt_mask_path: str,
    mask_before_path: str,
    mask_after_path: str,
    label_before: str = "Before",
    label_after: str = "After",
    save_path: str | None = None,
) -> None:
    """
    Show improvement from one model/step to another.
    Useful for showing the effect of:
    - Better architecture
    - Post-processing
    - More data augmentation
    - Fine-tuning on hard cases
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    before = cv2.imread(mask_before_path, cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(mask_after_path, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape[:2]
    for mask_arr in [gt, before, after]:
        if mask_arr.shape != (h, w):
            mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)

    gt_bin = (gt > 127).astype(np.uint8)
    before_bin = (before > 127).astype(np.uint8)
    after_bin = (after > 127).astype(np.uint8)

    def make_error_overlay(pred_bin):
        tp = pred_bin & gt_bin
        fp = pred_bin & (~gt_bin.astype(bool)).astype(np.uint8)
        fn = (~pred_bin.astype(bool)).astype(np.uint8) & gt_bin
        overlay = image.copy()
        color = np.zeros_like(image)
        color[tp == 1] = [0, 255, 0]
        color[fp == 1] = [255, 0, 0]
        color[fn == 1] = [0, 0, 255]
        mask_any = (tp | fp | fn).astype(bool)
        overlay[mask_any] = (
            0.5 * image[mask_any] + 0.5 * color[mask_any]
        ).astype(np.uint8)
        iou = int(tp.sum()) / max(int(tp.sum()) + int(fp.sum()) + int(fn.sum()), 1)
        return overlay, iou

    overlay_before, iou_before = make_error_overlay(before_bin)
    overlay_after, iou_after = make_error_overlay(after_bin)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay_before)
    axes[1].set_title(f"{label_before} (IoU={iou_before:.3f})",
                      fontweight="bold", color="red")
    axes[1].axis("off")

    axes[2].imshow(overlay_after)
    axes[2].set_title(f"{label_after} (IoU={iou_after:.3f})",
                      fontweight="bold", color="green")
    axes[2].axis("off")

    axes[3].imshow(gt, cmap="gray")
    axes[3].set_title("Ground Truth", fontweight="bold")
    axes[3].axis("off")

    improvement = iou_after - iou_before
    sign = "+" if improvement > 0 else ""
    plt.suptitle(
        f"Improvement: {sign}{improvement:.3f} IoU",
        fontsize=15, fontweight="bold",
        color="green" if improvement > 0 else "red",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
```

---

## Quick-Start: Run Everything on Validation Set

```python
"""
Complete pipeline to assess segmentation quality on the validation set.
Run this after training to get all visualizations at once.

Paths assume the standard Hackathon directory structure:
    data/Segmentation/validation/images/
    data/Segmentation/validation/masks/
    results/  (your predicted masks)
"""
from pathlib import Path

# === CONFIGURE THESE PATHS ===
BASE_DIR = Path("/Users/temur/Desktop/Claude/Hackathon")
VAL_IMAGE_DIR = str(BASE_DIR / "data/Segmentation/validation/images")
VAL_GT_DIR = str(BASE_DIR / "data/Segmentation/validation/masks")
PRED_DIR = str(BASE_DIR / "results/segmentation_val_preds")  # your predictions
OUTPUT_DIR = str(BASE_DIR / "results/visualizations")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Step 1: Compute per-image IoU
df = compute_per_image_iou(VAL_IMAGE_DIR, PRED_DIR, VAL_GT_DIR)
print_quality_summary(df)
df.to_csv(f"{OUTPUT_DIR}/per_image_iou.csv", index=False)

# Step 2: IoU distribution plot
plot_iou_distribution(df, save_path=f"{OUTPUT_DIR}/iou_distribution.png")

# Step 3: Flag bad predictions
flagged = flag_for_review(df, iou_threshold=0.5)
if len(flagged) > 0:
    flagged.to_csv(f"{OUTPUT_DIR}/flagged_for_review.csv", index=False)

# Step 4: Summary figure (best + worst examples)
create_summary_figure(
    df, VAL_IMAGE_DIR, PRED_DIR, VAL_GT_DIR,
    n_good=3, n_bad=3,
    save_path=f"{OUTPUT_DIR}/summary_figure.png",
)

# Step 5: Detailed error analysis on worst 5 images
worst_5 = df.head(5)["image_id"].tolist()
for image_id in worst_5:
    img_path = f"{VAL_IMAGE_DIR}/{image_id}.png"
    pred_path = f"{PRED_DIR}/{image_id}.png"
    gt_path = f"{VAL_GT_DIR}/{image_id}.png"

    # Check existence and try .jpg if .png not found
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        if Path(f"{VAL_IMAGE_DIR}/{image_id}{ext}").exists():
            img_path = f"{VAL_IMAGE_DIR}/{image_id}{ext}"
            break
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        if Path(f"{VAL_GT_DIR}/{image_id}{ext}").exists():
            gt_path = f"{VAL_GT_DIR}/{image_id}{ext}"
            break

    visualize_failure_mode(
        img_path, pred_path, gt_path,
        save_path=f"{OUTPUT_DIR}/failure_{image_id}.png",
    )

print(f"\nAll visualizations saved to {OUTPUT_DIR}/")
```

---

## Tips for the Presentation

1. **Lead with your best metric.** If Mean IoU = 0.85, put that number in 48pt font on the results slide.

2. **Show the IoU distribution histogram.** A tight distribution around 0.85 is more impressive than a mean of 0.87 with a long tail of failures.

3. **Include exactly 2-3 good examples and 2-3 bad examples.** For each bad example, explain WHY it failed (hair artifacts, low contrast, etc.) and what you tried to fix it.

4. **Show your post-processing pipeline visually.** The ablation figure (raw probability -> threshold -> morphology -> final) demonstrates engineering rigor.

5. **Show the probability heatmap, not just binary.** This proves you understand your model's confidence and can make nuanced decisions.

6. **Use consistent color coding throughout all slides:** green=TP, red=FP, blue=FN. State the legend once, keep it everywhere.

7. **Mention the threshold.** Saying "we optimized the threshold to 0.47 based on validation IoU" shows attention to detail.

8. **End with the disclaimer:** "For research and demonstration purposes only. Not for clinical use."
