# Inference-Time Segmentation IoU Optimization

**Baseline**: U-Net++ EfficientNetV2-S, 512px, 4x TTA, threshold 0.40, IoU **0.8952**
**Goal**: Maximize IoU without retraining any models.

---

## Table of Contents

1. [Full 8x TTA (replacing current 4x)](#1-full-8x-tta)
2. [Multi-Scale Inference](#2-multi-scale-inference)
3. [Threshold Optimization](#3-threshold-optimization)
4. [Dense CRF Boundary Refinement](#4-dense-crf-boundary-refinement)
5. [Morphological Post-Processing (tuned)](#5-morphological-post-processing)
6. [Connected Component Analysis](#6-connected-component-analysis)
7. [Probability Map Smoothing](#7-probability-map-smoothing)
8. [Model Soup / Weight Averaging](#8-model-soup--weight-averaging)
9. [Resize Strategy (logits-first)](#9-resize-strategy)
10. [Pixel-Level Voting Ensemble](#10-pixel-level-voting-ensemble)
11. [Complete Pipeline](#11-complete-pipeline)
12. [Expected Gains Summary](#12-expected-gains-summary)

---

## 1. Full 8x TTA

Current code uses 4 transforms (original, hflip, vflip, hvflip). Skin lesions have no canonical orientation, so 90/180/270 degree rotations are free gains.

**Expected gain: +0.003-0.008 IoU**

```python
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_tta_transforms(image_size: int) -> list:
    """
    8 TTA transforms: all elements of the D4 dihedral symmetry group.
    Each entry: (rot_k, flip_h, flip_v)
    """
    base_resize_norm = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])

    # D4 group: 4 rotations x 2 (with/without reflection) = 8
    transforms_d4 = [
        (0, False, False),  # identity
        (1, False, False),  # rot90
        (2, False, False),  # rot180
        (3, False, False),  # rot270
        (0, True,  False),  # hflip
        (1, True,  False),  # rot90 + hflip
        (0, False, True),   # vflip
        (1, False, True),   # rot90 + vflip
    ]
    return transforms_d4, base_resize_norm


def apply_tta(image: np.ndarray, rot_k: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    img = image.copy()
    if rot_k > 0:
        img = np.rot90(img, k=rot_k).copy()
    if flip_h:
        img = np.flip(img, axis=1).copy()
    if flip_v:
        img = np.flip(img, axis=0).copy()
    return img


def invert_tta(prob: np.ndarray, rot_k: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    p = prob.copy()
    # Inverse order: undo flip, then undo rotation
    if flip_v:
        p = np.flip(p, axis=0).copy()
    if flip_h:
        p = np.flip(p, axis=1).copy()
    if rot_k > 0:
        p = np.rot90(p, k=4 - rot_k).copy()
    return p


def predict_with_8x_tta(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run model with 8x TTA (all D4 symmetries).
    Returns probability map at model resolution (image_size x image_size).
    """
    transforms_d4, base_transform = build_tta_transforms(image_size)
    accumulator = np.zeros((image_size, image_size), dtype=np.float64)

    with torch.no_grad():
        for rot_k, flip_h, flip_v in transforms_d4:
            augmented_img = apply_tta(image, rot_k, flip_h, flip_v)
            tensor = base_transform(image=augmented_img)["image"].unsqueeze(0).to(device)
            logits = model(tensor)
            prob = torch.sigmoid(logits).cpu().numpy().squeeze()
            prob_inv = invert_tta(prob, rot_k, flip_h, flip_v)
            accumulator += prob_inv

    return (accumulator / len(transforms_d4)).astype(np.float32)
```

---

## 2. Multi-Scale Inference

Run the model at multiple resolutions and average the probability maps. Larger scales capture fine boundary detail; smaller scales give better global context.

**Expected gain: +0.005-0.015 IoU**

```python
def predict_single_scale(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    """Single scale, no TTA."""
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).cpu().numpy().squeeze()
    return prob


def predict_multiscale(
    model: torch.nn.Module,
    image: np.ndarray,
    scales: list,
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    """
    Multi-scale inference with optional TTA at each scale.

    Args:
        scales: List of resolutions, e.g. [384, 512, 640, 768]

    Returns:
        Probability map at the LARGEST scale resolution.
    """
    target_size = max(scales)
    accumulator = np.zeros((target_size, target_size), dtype=np.float64)
    total_weight = 0.0

    # Weight larger scales slightly more (finer detail)
    scale_weights = {s: (s / max(scales)) ** 0.5 for s in scales}

    for scale in scales:
        if use_tta:
            prob = predict_with_8x_tta(model, image, scale, device)
        else:
            prob = predict_single_scale(model, image, scale, device)

        # Resize prob map to target_size
        if scale != target_size:
            prob = cv2.resize(prob, (target_size, target_size),
                              interpolation=cv2.INTER_LINEAR)

        w = scale_weights[scale]
        accumulator += prob * w
        total_weight += w

    return (accumulator / total_weight).astype(np.float32)


# Recommended scales for 512-trained model:
# Conservative (fast):  [384, 512, 640]
# Aggressive (slow):    [320, 384, 448, 512, 576, 640, 768]
# Best balance:         [384, 512, 640, 768]
```

---

## 3. Threshold Optimization

A global fixed threshold (0.40 or 0.50) is rarely optimal. Search on the validation set.

**Expected gain: +0.005-0.015 IoU**

```python
from pathlib import Path


def find_optimal_threshold(
    model: torch.nn.Module,
    val_image_dir: str,
    val_mask_dir: str,
    device: torch.device,
    image_size: int = 512,
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Search for the threshold that maximizes mean IoU on validation set.
    Returns: (best_threshold, best_iou)
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.80, 0.01)

    val_images = sorted(Path(val_image_dir).glob("*.png"))
    val_masks_dir = Path(val_mask_dir)

    # Collect all probability maps
    all_probs = []
    all_gts = []

    for img_path in val_images:
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        oh, ow = image.shape[:2]

        prob = predict_with_8x_tta(model, image, image_size, device)
        prob = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)

        mask_path = val_masks_dir / img_path.name
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.float32)

        all_probs.append(prob)
        all_gts.append(gt)

    # Search thresholds
    best_thresh = 0.5
    best_iou = 0.0

    for t in thresholds:
        ious = []
        for prob, gt in zip(all_probs, all_gts):
            pred = (prob > t).astype(np.float32)
            intersection = (pred * gt).sum()
            union = pred.sum() + gt.sum() - intersection
            iou = intersection / max(union, 1e-8)
            ious.append(iou)
        mean_iou = np.mean(ious)
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_thresh = t

    print(f"Optimal threshold: {best_thresh:.2f} -> IoU: {best_iou:.4f}")
    return best_thresh, best_iou


def per_image_adaptive_threshold(
    prob: np.ndarray,
    low: float = 0.25,
    high: float = 0.65,
    min_coverage: float = 0.01,
    max_coverage: float = 0.70,
) -> np.ndarray:
    """
    Adaptive threshold per image using Otsu on the probability map.
    For skin lesion segmentation, typical coverage is 5-50% of image.
    """
    prob_uint8 = (prob * 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(
        prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    t = np.clip(otsu_thresh / 255.0, low, high)

    mask = (prob > t).astype(np.float32)
    coverage = mask.mean()

    if coverage < min_coverage:
        t = max(low, t - 0.10)
    elif coverage > max_coverage:
        t = min(high, t + 0.10)

    return (prob > t).astype(np.uint8)
```

---

## 4. Dense CRF Boundary Refinement

CRF uses color/spatial information from the original image to refine boundaries. Very effective for skin lesion segmentation where lesion boundaries follow color gradients.

**Expected gain: +0.005-0.020 IoU**

```python
# pip install pydensecrf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def apply_dense_crf(
    image: np.ndarray,
    prob: np.ndarray,
    sxy_gauss: int = 3,
    compat_gauss: float = 3.0,
    sxy_bilateral: int = 80,
    srgb_bilateral: int = 13,
    compat_bilateral: float = 10.0,
    n_iterations: int = 5,
) -> np.ndarray:
    """
    Apply Dense CRF to refine segmentation probability map.

    Args:
        image: Original RGB image (H, W, 3) uint8
        prob: Probability map (H, W) float32 in [0, 1]
        sxy_gauss: Spatial std for Gaussian pairwise (smoothness)
        compat_gauss: Compatibility for Gaussian term
        sxy_bilateral: Spatial std for bilateral pairwise
        srgb_bilateral: Color std for bilateral pairwise
        compat_bilateral: Compatibility for bilateral term
        n_iterations: CRF inference iterations

    Returns:
        Refined binary mask (H, W) uint8 {0, 1}

    Tuning for skin lesions:
        sxy_bilateral=60-100: larger = more spatial smoothing
        srgb_bilateral=10-20: smaller = stronger color boundary following
    """
    h, w = prob.shape

    # 2-class probability: [background, foreground]
    prob_2class = np.stack([1.0 - prob, prob], axis=0).astype(np.float32)
    unary = unary_from_softmax(prob_2class)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)

    # Gaussian pairwise (smoothness)
    d.addPairwiseGaussian(
        sxy=sxy_gauss,
        compat=compat_gauss,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # Bilateral pairwise (appearance / boundary following)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=image.copy(order='C'),
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(n_iterations)
    result = np.argmax(np.array(Q).reshape((2, h, w)), axis=0)

    return result.astype(np.uint8)
```

---

## 5. Morphological Post-Processing

Tuned kernel sizes for skin lesion segmentation. Order matters: fill holes, close, open.

**Expected gain: +0.002-0.005 IoU**

```python
from scipy import ndimage


def morphological_postprocess(
    binary_mask: np.ndarray,
    close_kernel_size: int = 7,
    open_kernel_size: int = 3,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Pipeline: fill holes -> close -> open.

    close_kernel_size: 5-9 for skin lesions (fills boundary gaps)
    open_kernel_size: 3-5 (removes small noise)
    """
    mask = binary_mask.copy()

    if mask.sum() == 0:
        return mask

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


def smooth_boundary(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth jagged edges via Gaussian blur + re-threshold."""
    mask_float = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_float, (0, 0), sigma)
    return (blurred > 0.5).astype(np.uint8)
```

---

## 6. Connected Component Analysis

Skin lesion images typically contain ONE lesion. Keep only the largest component.

**Expected gain: +0.002-0.005 IoU**

```python
def keep_largest_component(
    binary_mask: np.ndarray,
    min_area_ratio: float = 0.001,
) -> np.ndarray:
    """Keep only the largest connected component."""
    if binary_mask.sum() == 0:
        return binary_mask

    labeled, num_features = ndimage.label(binary_mask)

    if num_features <= 1:
        return binary_mask

    component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_idx = np.argmax(component_sizes) + 1
    result = (labeled == largest_idx).astype(np.uint8)

    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    if component_sizes[largest_idx - 1] < total_pixels * min_area_ratio:
        return np.zeros_like(binary_mask)

    return result


def smart_component_selection(
    binary_mask: np.ndarray,
    prob_map: np.ndarray,
    confidence_threshold: float = 0.6,
) -> np.ndarray:
    """Keep components where average probability exceeds threshold."""
    if binary_mask.sum() == 0:
        return binary_mask

    labeled, num_features = ndimage.label(binary_mask)

    if num_features <= 1:
        return binary_mask

    result = np.zeros_like(binary_mask)

    for i in range(1, num_features + 1):
        component = (labeled == i)
        avg_prob = prob_map[component].mean()
        if avg_prob >= confidence_threshold:
            result[component] = 1

    if result.sum() == 0:
        return keep_largest_component(binary_mask)

    return result
```

---

## 7. Probability Map Smoothing

Gaussian smoothing on probabilities before thresholding reduces noise.

**Expected gain: +0.001-0.003 IoU**

```python
def smooth_probability_map(
    prob: np.ndarray,
    sigma: float = 1.5,
    method: str = "gaussian",
) -> np.ndarray:
    """
    Smooth probability map before thresholding.
    sigma 1.0-2.5 works for skin lesions.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(prob, (0, 0), sigma)
    elif method == "bilateral":
        prob_uint8 = (prob * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(prob_uint8, d=9, sigmaColor=75, sigmaSpace=75)
        return smoothed.astype(np.float32) / 255.0
    return prob
```

---

## 8. Model Soup / Weight Averaging

Average weights from multiple checkpoints without retraining.

**Expected gain: +0.005-0.015 IoU (if multiple checkpoints available)**

```python
import copy


def average_model_weights(
    model_template: torch.nn.Module,
    checkpoint_paths: list,
    device: torch.device,
) -> torch.nn.Module:
    """
    Model soup: average weights from multiple checkpoints.
    All checkpoints must be the same architecture.
    """
    avg_state_dict = None

    for path in checkpoint_paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        if avg_state_dict is None:
            avg_state_dict = copy.deepcopy(state)
        else:
            for key in avg_state_dict:
                avg_state_dict[key] = avg_state_dict[key] + state[key]

    n = len(checkpoint_paths)
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / n

    model = copy.deepcopy(model_template)
    model.load_state_dict(avg_state_dict, strict=False)
    model.to(device)

    return model
```

---

## 9. Resize Strategy

CRITICAL: Always resize the probability map (float), NOT the binary mask. Resizing binary masks introduces jagged staircase edges that hurt IoU.

**Expected gain: +0.002-0.005 IoU (if currently doing it wrong)**

```python
def correct_resize_pipeline(
    prob_at_model_res: np.ndarray,
    original_h: int,
    original_w: int,
    threshold: float = 0.40,
) -> tuple:
    """
    CORRECT: resize float probs FIRST, then threshold at full resolution.
    """
    prob_full = cv2.resize(
        prob_at_model_res,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR,
    )
    binary = (prob_full > threshold).astype(np.uint8)
    return binary, prob_full

# Your current segment.py already does this correctly (line 71).
```

---

## 10. Pixel-Level Voting Ensemble

If you have multiple models, average their probability maps at the pixel level.

**Expected gain: +0.010-0.025 IoU (with 2-3 diverse models)**

```python
import segmentation_models_pytorch as smp


def ensemble_predict(
    models: list,
    image: np.ndarray,
    device: torch.device,
    scales: list = None,
    use_tta: bool = True,
    model_weights: list = None,
) -> np.ndarray:
    """
    Pixel-level ensemble across multiple models.

    Args:
        models: List of (model, native_image_size) tuples
        image: Original RGB image (H, W, 3)
        model_weights: Weight per model (None = equal)

    Returns:
        Ensembled probability map at original image resolution
    """
    oh, ow = image.shape[:2]
    accumulator = np.zeros((oh, ow), dtype=np.float64)
    total_weight = 0.0

    if model_weights is None:
        model_weights = [1.0] * len(models)

    for (model, native_size), weight in zip(models, model_weights):
        if scales is not None:
            prob = predict_multiscale(model, image, scales, device, use_tta)
            target_size = max(scales)
        elif use_tta:
            prob = predict_with_8x_tta(model, image, native_size, device)
            target_size = native_size
        else:
            prob = predict_single_scale(model, image, native_size, device)
            target_size = native_size

        prob_full = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
        accumulator += prob_full * weight
        total_weight += weight

    return (accumulator / total_weight).astype(np.float32)


def load_all_models(checkpoint_configs: list, device: torch.device) -> tuple:
    """
    Load multiple models.

    checkpoint_configs example:
    [
        {
            "path": "checkpoints/best_tf_efficientnetv2_s.pth",
            "arch": "UnetPlusPlus",
            "encoder": "tu-tf_efficientnetv2_s",
            "image_size": 512,
            "weight": 1.0,
        },
    ]
    """
    models = []
    weights = []

    for cfg in checkpoint_configs:
        arch_class = getattr(smp, cfg["arch"])
        model = arch_class(
            encoder_name=cfg["encoder"],
            encoder_weights=None,
            classes=1,
            activation=None,
        )
        state = torch.load(cfg["path"], map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval().to(device)

        models.append((model, cfg["image_size"]))
        weights.append(cfg.get("weight", 1.0))

    return models, weights
```

---

## 11. Complete Pipeline

Full inference pipeline combining all techniques in correct order.

```python
"""
Complete optimized inference pipeline.

Usage:
    python optimized_inference.py \
        --input_dir data/Segmentation/testing/images/ \
        --output_dir submission/WhiteCoat.dev/ \
        --checkpoint checkpoints/best_tf_efficientnetv2_s.pth \
        --threshold 0.40 \
        --use_crf \
        --multiscale
"""

import argparse
import time
from pathlib import Path
from scipy import ndimage

# =================== CONFIG ===================
DEFAULT_CONFIG = {
    "scales": [384, 512, 640],
    "use_tta": True,
    "use_crf": True,
    "use_morphology": True,
    "use_smooth": True,
    "smooth_sigma": 1.0,
    "close_kernel": 7,
    "open_kernel": 3,
    "keep_largest_only": True,
    "threshold": 0.40,
}


def optimized_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    config: dict = None,
) -> np.ndarray:
    """
    Full optimized inference pipeline for a single image.

    Pipeline order:
    1. Multi-scale + 8x TTA -> averaged probability map
    2. Resize probabilities to original resolution (bilinear)
    3. Gaussian smooth on probabilities
    4. Dense CRF refinement (optional)
    5. Threshold to binary
    6. Fill holes
    7. Keep largest component
    8. Morphological close + open
    9. Output binary mask (0 or 255)
    """
    if config is None:
        config = DEFAULT_CONFIG

    oh, ow = image.shape[:2]

    # Step 1: Multi-scale + TTA inference
    if config.get("scales"):
        prob = predict_multiscale(
            model, image, config["scales"], device,
            use_tta=config.get("use_tta", True),
        )
        target_size = max(config["scales"])
    elif config.get("use_tta", True):
        prob = predict_with_8x_tta(model, image, 512, device)
    else:
        prob = predict_single_scale(model, image, 512, device)

    # Step 2: Resize to original resolution
    prob_full = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)

    # Step 3: Smooth probability map
    if config.get("use_smooth", True):
        sigma = config.get("smooth_sigma", 1.0)
        prob_full = cv2.GaussianBlur(prob_full, (0, 0), sigma)

    # Step 4: Dense CRF (or simple threshold)
    if config.get("use_crf", False):
        try:
            binary = apply_dense_crf(image, prob_full)
        except ImportError:
            print("pydensecrf not installed, skipping CRF")
            binary = (prob_full > config["threshold"]).astype(np.uint8)
    else:
        binary = (prob_full > config["threshold"]).astype(np.uint8)

    # Step 5: Fill holes
    if binary.sum() > 0:
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

    # Step 6: Keep largest component
    if config.get("keep_largest_only", True):
        binary = keep_largest_component(binary)

    # Step 7: Morphological refinement
    if config.get("use_morphology", True):
        binary = morphological_postprocess(
            binary,
            close_kernel_size=config.get("close_kernel", 7),
            open_kernel_size=config.get("open_kernel", 3),
        )

    return binary * 255


def run_optimized_inference(
    checkpoint_path: str,
    input_dir: str,
    output_dir: str,
    arch: str = "UnetPlusPlus",
    encoder: str = "tu-tf_efficientnetv2_s",
    config: dict = None,
):
    """Run the full optimized pipeline on all test images."""
    if config is None:
        config = DEFAULT_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = getattr(smp, arch)(
        encoder_name=encoder, encoder_weights=None, classes=1, activation=None
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")),
        key=lambda p: p.stem,
    )
    print(f"Processing {len(images)} images...")

    start = time.time()

    for i, img_path in enumerate(images):
        try:
            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            mask = optimized_predict(model, image, device, config)
            cv2.imwrite(str(output_path / f"{img_path.stem}.png"), mask)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                per_img = elapsed / (i + 1)
                remaining = per_img * (len(images) - i - 1)
                print(f"  [{i+1}/{len(images)}] "
                      f"{per_img:.1f}s/img, ~{remaining:.0f}s remaining")
        except Exception as e:
            print(f"Error {img_path.name}: {e}")
            oh, ow = 256, 256
            try:
                img_tmp = cv2.imread(str(img_path))
                if img_tmp is not None:
                    oh, ow = img_tmp.shape[:2]
            except Exception:
                pass
            cv2.imwrite(str(output_path / f"{img_path.stem}.png"),
                        np.zeros((oh, ow), dtype=np.uint8))

    total = time.time() - start
    print(f"Done: {len(images)} masks in {total:.1f}s ({total/len(images):.1f}s/img)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="submission/WhiteCoat.dev/")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", default="UnetPlusPlus")
    parser.add_argument("--encoder", default="tu-tf_efficientnetv2_s")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--no_tta", action="store_true")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["threshold"] = args.threshold
    if not args.multiscale:
        config["scales"] = None
    if args.no_tta:
        config["use_tta"] = False
    config["use_crf"] = args.use_crf

    run_optimized_inference(
        args.checkpoint, args.input_dir, args.output_dir,
        args.arch, args.encoder, config,
    )
```

---

## 12. Expected Gains Summary

Starting from baseline IoU = **0.8952** (4x TTA, threshold 0.40, basic morphology).

| # | Technique | Estimated Gain | Cumulative IoU | Notes |
|---|-----------|---------------|----------------|-------|
| 1 | 8x TTA (from 4x) | +0.003-0.008 | 0.898-0.903 | Almost free, just 2x slower |
| 2 | Multi-scale (3 scales) | +0.005-0.015 | 0.903-0.918 | 3x slower per scale added |
| 3 | Threshold optimization | +0.005-0.015 | 0.908-0.933 | Requires val set, zero cost at test time |
| 4 | Dense CRF | +0.005-0.020 | 0.913-0.953 | Best single technique for boundaries |
| 5 | Morphology (tuned) | +0.002-0.005 | 0.915-0.958 | Marginal if CRF is used |
| 6 | Connected components | +0.002-0.005 | 0.917-0.963 | Removes false positive islands |
| 7 | Probability smoothing | +0.001-0.003 | 0.918-0.966 | Diminishing returns with CRF |
| 8 | Model soup | +0.005-0.015 | 0.923-0.981 | Only if multiple checkpoints exist |
| 9 | Resize strategy | +0.000 | -- | Already correct in your code |
| 10 | Multi-model ensemble | +0.010-0.025 | 0.933-1.000 | Biggest gain, needs different architectures |

### Realistic Estimates (single model, single checkpoint)

| Scenario | Expected IoU |
|----------|-------------|
| Conservative: 8x TTA + threshold opt + morphology | 0.905-0.915 |
| Moderate: + multi-scale [384,512,640] + CRF | 0.915-0.930 |
| Aggressive: + 5 scales + tuned CRF + all tricks | 0.920-0.940 |

### Recommended Priority (highest ROI first)

1. **Threshold optimization on val set** -- free, potentially +0.015
2. **8x TTA** -- 2x inference time, easy +0.005
3. **Multi-scale [384, 512, 640]** -- 3x slower, +0.010
4. **Dense CRF** -- needs `pip install pydensecrf`, +0.010
5. **Everything else** -- diminishing returns

### Time Budget (200 test images)

| Configuration | Per Image (CPU) | Total 200 imgs (CPU) | Total (GPU) |
|---------------|----------------|---------------------|-------------|
| Current (4x TTA, 512) | ~2s | ~7 min | ~1 min |
| 8x TTA, 512 | ~4s | ~13 min | ~2 min |
| 8x TTA, 3 scales | ~12s | ~40 min | ~5 min |
| 8x TTA, 3 scales + CRF | ~14s | ~47 min | ~7 min |
| Full pipeline, 5 scales | ~25s | ~83 min | ~12 min |
