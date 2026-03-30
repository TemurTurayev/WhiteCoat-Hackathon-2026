# Binary Mask Optimization for Skin Lesion Segmentation

> Target: Improve IoU from ~0.81 to 0.90+
> Context: 200 test images, PNG binary masks (0 or 255), same size as input

---

## Table of Contents
1. [Post-Processing Pipeline (Optimal Order)](#1-post-processing-pipeline)
2. [Adaptive Thresholding](#2-adaptive-thresholding)
3. [Multi-Scale Inference](#3-multi-scale-inference)
4. [CRF Refinement](#4-crf-refinement)
5. [Boundary-Aware Techniques](#5-boundary-aware-techniques)
6. [Proper Mask Resizing](#6-proper-mask-resizing)
7. [Ensemble Mask Merging](#7-ensemble-mask-merging)
8. [Quality Metrics Beyond IoU](#8-quality-metrics-beyond-iou)
9. [Complete Production Pipeline](#9-complete-production-pipeline)

---

## 1. Post-Processing Pipeline

### Optimal Order of Operations

```
Raw logits -> Sigmoid -> [Multi-scale average] -> [CRF] -> Threshold -> Morphological ops -> Connected components -> Hole filling -> Boundary refinement -> Final binary mask
```

**Key principle**: Work on probability maps (float32) as long as possible. Binarize LATE.

### 1.1 Morphological Operations

```python
import cv2
import numpy as np

def morphological_cleanup(binary_mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean a binary mask.
    Input: binary mask (0 or 255), uint8
    Output: cleaned binary mask (0 or 255), uint8
    """
    # Step 1: Opening -- remove small noise/artifacts (erosion then dilation)
    # Kernel size 3x3 for small noise, 5x5 for larger artifacts
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Step 2: Closing -- fill small holes and gaps (dilation then erosion)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    return mask
```

**Kernel size guidelines for skin lesions**:
- Small noise removal (opening): 3x3 to 5x5 elliptical
- Gap filling (closing): 5x5 to 11x11 elliptical
- For larger images (1024+), scale kernel sizes proportionally
- Always use `MORPH_ELLIPSE` for skin lesions (round structures)
- Iterations: 1-2 for opening, 1-3 for closing

### 1.2 Connected Component Analysis

```python
def keep_largest_components(binary_mask: np.ndarray,
                            min_area_ratio: float = 0.01,
                            max_components: int = 3) -> np.ndarray:
    """
    Keep only the largest connected components.

    Args:
        binary_mask: Binary mask (0 or 255), uint8
        min_area_ratio: Minimum area as fraction of total image area
        max_components: Maximum number of components to keep
    Returns:
        Cleaned binary mask (0 or 255), uint8
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels <= 1:  # Only background
        return binary_mask

    # Get areas of each component (skip background at index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    total_area = binary_mask.shape[0] * binary_mask.shape[1]
    min_area = total_area * min_area_ratio

    # Sort components by area (descending)
    sorted_indices = np.argsort(areas)[::-1]

    result = np.zeros_like(binary_mask)
    kept = 0
    for idx in sorted_indices:
        if kept >= max_components:
            break
        if areas[idx] < min_area:
            break
        # +1 because we skipped background
        result[labels == (idx + 1)] = 255
        kept += 1

    return result
```

**When to keep only largest vs multiple**:
- **Single lesion images** (most common): keep only the largest component
- **Multi-lesion or satellite lesions**: keep top 2-3 components above min area
- **Rule of thumb**: if 2nd largest component is < 10% of largest, discard it

### 1.3 Hole Filling

```python
def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside the segmented region using contours.
    Input/Output: binary mask (0 or 255), uint8
    """
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(binary_mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def fill_holes_floodfill(binary_mask: np.ndarray) -> np.ndarray:
    """
    Alternative: Flood fill from corners to find internal holes.
    More robust for complex shapes.
    """
    h, w = binary_mask.shape
    # Create a slightly larger image for flood fill
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood[1:-1, 1:-1] = binary_mask.copy()

    # Flood fill from (0,0) -- fills the background
    cv2.floodFill(flood, None, (0, 0), 255)

    # Invert: holes become white
    flood_inv = cv2.bitwise_not(flood[1:-1, 1:-1])

    # Combine: original mask OR filled holes
    return cv2.bitwise_or(binary_mask, flood_inv)


def fill_holes_scipy(binary_mask: np.ndarray) -> np.ndarray:
    """
    scipy-based hole filling. Often the most reliable.
    """
    from scipy import ndimage
    bool_mask = binary_mask > 0
    filled = ndimage.binary_fill_holes(bool_mask)
    return (filled.astype(np.uint8) * 255)
```

### 1.4 Boundary Refinement (Gaussian Smoothing + Re-threshold)

```python
def smooth_boundary(binary_mask: np.ndarray,
                    sigma: float = 2.0,
                    threshold: float = 0.5) -> np.ndarray:
    """
    Smooth jagged boundaries by blurring and re-thresholding.
    Produces cleaner, more natural contours.
    """
    float_mask = binary_mask.astype(np.float32) / 255.0
    ksize = int(sigma * 6) | 1  # Ensure odd kernel size
    smoothed = cv2.GaussianBlur(float_mask, (ksize, ksize), sigma)
    result = (smoothed >= threshold).astype(np.uint8) * 255
    return result
```

---

## 2. Adaptive Thresholding

### 2.1 Per-Image Threshold Optimization (with validation set)

```python
def find_optimal_threshold(prob_maps: list,
                           gt_masks: list,
                           thresholds: np.ndarray = None) -> float:
    """
    Find the single best threshold across all validation images.
    Use this threshold for test inference.

    Args:
        prob_maps: List of probability maps (float32, 0-1)
        gt_masks: List of ground truth binary masks
        thresholds: Thresholds to search over
    Returns:
        Optimal threshold value
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.01)

    best_iou = 0
    best_thresh = 0.5

    for thresh in thresholds:
        ious = []
        for prob, gt in zip(prob_maps, gt_masks):
            pred = (prob >= thresh).astype(np.uint8)
            gt_bin = (gt > 0).astype(np.uint8)

            intersection = np.logical_and(pred, gt_bin).sum()
            union = np.logical_or(pred, gt_bin).sum()
            iou = intersection / (union + 1e-8)
            ious.append(iou)

        mean_iou = np.mean(ious)
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_thresh = thresh

    print(f"Optimal threshold: {best_thresh:.3f}, Mean IoU: {best_iou:.4f}")
    return best_thresh
```

### 2.2 Otsu's Method on Probability Maps

```python
def otsu_threshold_prob_map(prob_map: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's method to a probability map for per-image adaptive threshold.
    Works well when the probability map has a bimodal distribution.
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    thresh_val, binary = cv2.threshold(
        prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu threshold: {thresh_val / 255:.3f}")
    return binary
```

### 2.3 Histogram-Based Threshold Selection

```python
def histogram_threshold(prob_map: np.ndarray,
                        n_bins: int = 256) -> np.ndarray:
    """
    Find threshold using valley detection in the probability histogram.
    Good for bimodal distributions typical of segmentation outputs.
    """
    from scipy.signal import argrelextrema
    from scipy.ndimage import gaussian_filter1d

    hist, bin_edges = np.histogram(prob_map.flatten(), bins=n_bins, range=(0, 1))
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=5)

    # Find valleys (local minima)
    valleys = argrelextrema(hist_smooth, np.less, order=10)[0]

    if len(valleys) == 0:
        return (prob_map >= 0.5).astype(np.uint8) * 255

    # Choose the valley closest to 0.5 (middle)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valley_values = bin_centers[valleys]
    best_idx = np.argmin(np.abs(valley_values - 0.5))
    threshold = valley_values[best_idx]

    return (prob_map >= threshold).astype(np.uint8) * 255
```

### 2.4 When to Use Global vs Per-Image Threshold

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Global fixed** (0.5) | Baseline, well-calibrated models | Simple, consistent | Suboptimal for edge cases |
| **Global optimized** | Have validation set | Better than 0.5, stable | Single value for all images |
| **Per-image Otsu** | Variable probability distributions | Adapts to each image | Can fail on unimodal distributions |
| **Per-image histogram** | Complex distributions | Robust valley detection | Slower, requires tuning |

**Recommendation**: Start with global threshold optimized on validation set. Try per-image Otsu and compare.

---

## 3. Multi-Scale Inference

### 3.1 Multi-Scale TTA

```python
import torch
import torch.nn.functional as F

def multi_scale_inference(model, image: torch.Tensor,
                          scales: list = None,
                          flip: bool = True) -> np.ndarray:
    """
    Predict at multiple scales and average probability maps.

    Args:
        model: Segmentation model (mode set to evaluation)
        image: Input tensor [1, C, H, W]
        scales: Scale factors to use
        flip: Whether to also do horizontal/vertical flips
    Returns:
        Averaged probability map (H, W), float32, range [0, 1]
    """
    if scales is None:
        scales = [0.75, 1.0, 1.25]

    _, _, orig_h, orig_w = image.shape
    prob_sum = torch.zeros(1, 1, orig_h, orig_w, device=image.device)
    count = 0

    for scale in scales:
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        scaled = F.interpolate(image, size=(new_h, new_w),
                               mode='bilinear', align_corners=False)

        augmented_inputs = [scaled]
        if flip:
            augmented_inputs.append(torch.flip(scaled, dims=[3]))  # H-flip
            augmented_inputs.append(torch.flip(scaled, dims=[2]))  # V-flip

        for i, aug_input in enumerate(augmented_inputs):
            with torch.no_grad():
                logits = model(aug_input)
                prob = torch.sigmoid(logits)

            # Reverse augmentation
            if i == 1:
                prob = torch.flip(prob, dims=[3])
            elif i == 2:
                prob = torch.flip(prob, dims=[2])

            # Resize probability map back to original size
            # CRITICAL: Use bilinear interpolation on probability maps
            prob_orig_size = F.interpolate(
                prob, size=(orig_h, orig_w),
                mode='bilinear', align_corners=False
            )

            prob_sum += prob_orig_size
            count += 1

    avg_prob = prob_sum / count
    return avg_prob.squeeze().cpu().numpy()
```

### 3.2 Full 8x TTA (All Flips + Rotations)

```python
def tta_8x(model, image: torch.Tensor) -> np.ndarray:
    """
    8x TTA: original + 3 rotations x 2 (with/without h-flip).
    Skin lesions have no canonical orientation, so all 8 are valid.

    Returns: averaged probability map (H, W), float32
    """
    _, _, H, W = image.shape
    prob_sum = np.zeros((H, W), dtype=np.float64)

    for k in range(4):  # 0, 90, 180, 270 degrees
        for do_flip in [False, True]:
            x = torch.rot90(image, k, dims=[2, 3])
            if do_flip:
                x = torch.flip(x, dims=[3])

            with torch.no_grad():
                logits = model(x)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

            if do_flip:
                prob = np.flip(prob, axis=1).copy()
            prob = np.rot90(prob, -k).copy()
            prob_sum += prob

    return (prob_sum / 8.0).astype(np.float32)
```

### 3.3 Combined Multi-Scale + TTA

```python
def multi_scale_tta(model, image: torch.Tensor,
                    scales: list = None) -> np.ndarray:
    """
    Combine multi-scale inference with 8x TTA.
    Total predictions: len(scales) * 8. Expensive but best results.
    For 200 test images it is very manageable.
    """
    if scales is None:
        scales = [0.75, 1.0, 1.25]

    _, _, H, W = image.shape
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = 0

    for scale in scales:
        new_h, new_w = int(H * scale), int(W * scale)
        scaled = F.interpolate(image, size=(new_h, new_w),
                               mode='bilinear', align_corners=False)

        # Run 8x TTA at this scale
        prob_at_scale = tta_8x(model, scaled)  # (new_h, new_w)

        # Resize probability map back to original size using bilinear
        prob_resized = cv2.resize(prob_at_scale, (W, H),
                                  interpolation=cv2.INTER_LINEAR)

        prob_sum += prob_resized
        count += 1

    return (prob_sum / count).astype(np.float32)
```

### 3.4 Does Multi-Scale Improve Boundary Quality?

**Yes, demonstrably**:
- Higher scales (1.25x, 1.5x) capture fine boundary details
- Lower scales (0.5x, 0.75x) provide better global context, reducing false positives
- Averaging across scales smooths out scale-dependent artifacts
- Typical improvement: +1-3% IoU over single-scale inference
- Diminishing returns beyond 3-4 scales

**Recommended scale sets**:
- Quick: [1.0] with 8x TTA (8 predictions)
- Balanced: [0.75, 1.0, 1.25] with 8x TTA (24 predictions)
- Maximum: [0.5, 0.75, 1.0, 1.25, 1.5] with 8x TTA (40 predictions)

---

## 4. CRF (Conditional Random Fields) Refinement

### 4.1 Installation

```bash
pip install pydensecrf
# or for Python 3.10+:
pip install pydensecrf2
```

### 4.2 DenseCRF for Binary Skin Lesion Masks

```python
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(image_rgb: np.ndarray,
              prob_map: np.ndarray,
              n_iters: int = 5,
              sxy_gaussian: int = 3,
              compat_gaussian: int = 3,
              sxy_bilateral: int = 80,
              srgb_bilateral: int = 13,
              compat_bilateral: int = 10) -> np.ndarray:
    """
    Apply DenseCRF post-processing to refine a probability map.

    Args:
        image_rgb: Original RGB image (H, W, 3), uint8
        prob_map: Probability map (H, W), float32, range [0, 1]
        n_iters: Number of mean-field iterations (5-10)
        sxy_gaussian: Spatial std for appearance-independent term
        compat_gaussian: Compatibility for Gaussian kernel
        sxy_bilateral: Spatial std for bilateral term
        srgb_bilateral: Color std for bilateral term
        compat_bilateral: Compatibility for bilateral kernel
    Returns:
        Refined probability map (H, W), float32
    """
    h, w = prob_map.shape
    d = dcrf.DenseCRF2D(w, h, 2)  # width, height, num_classes

    # Unary potentials from probability map
    prob_fg = np.clip(prob_map.copy(), 1e-6, 1.0 - 1e-6)
    prob_bg = 1.0 - prob_fg
    probs = np.stack([prob_bg, prob_fg], axis=0)  # (2, H, W)

    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Pairwise: appearance-independent (Gaussian) -- smoothness
    d.addPairwiseGaussian(
        sxy=sxy_gaussian,
        compat=compat_gaussian,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Pairwise: appearance-dependent (Bilateral) -- edge-aware
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=image_rgb.copy(order='C'),
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Inference
    Q = d.inference(n_iters)
    Q = np.array(Q).reshape((2, h, w))
    return Q[1]
```

### 4.3 CRF Parameter Tuning Guide

```python
def tune_crf_params(images: list, prob_maps: list,
                    gt_masks: list) -> dict:
    """Grid search for optimal CRF parameters on validation set."""
    param_grid = {
        'sxy_gaussian': [1, 3, 5],
        'compat_gaussian': [1, 3, 5],
        'sxy_bilateral': [40, 80, 120],
        'srgb_bilateral': [5, 13, 20],
        'compat_bilateral': [5, 10, 15],
    }

    best_iou = 0
    best_params = {}

    for sxy_g in param_grid['sxy_gaussian']:
        for sxy_b in param_grid['sxy_bilateral']:
            for srgb_b in param_grid['srgb_bilateral']:
                ious = []
                for img, prob, gt in zip(images, prob_maps, gt_masks):
                    refined = apply_crf(
                        img, prob,
                        sxy_gaussian=sxy_g,
                        sxy_bilateral=sxy_b,
                        srgb_bilateral=srgb_b
                    )
                    pred = (refined >= 0.5).astype(np.uint8)
                    gt_bin = (gt > 0).astype(np.uint8)

                    intersection = np.logical_and(pred, gt_bin).sum()
                    union = np.logical_or(pred, gt_bin).sum()
                    ious.append(intersection / (union + 1e-8))

                mean_iou = np.mean(ious)
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    best_params = {
                        'sxy_gaussian': sxy_g,
                        'sxy_bilateral': sxy_b,
                        'srgb_bilateral': srgb_b
                    }

    print(f"Best CRF params: {best_params}, IoU: {best_iou:.4f}")
    return best_params
```

### 4.4 Is CRF Worth the Complexity?

| Aspect | Assessment |
|--------|-----------|
| **IoU improvement** | +1-3% when model is already decent |
| **Boundary quality** | Significant improvement, edges align with image boundaries |
| **Speed** | ~0.1-0.5s per image (acceptable for 200 images) |
| **Complexity** | Moderate -- requires original RGB image + tuning |
| **When it helps most** | Fuzzy boundaries, hair artifacts, when model is uncertain at edges |
| **When it hurts** | Already sharp predictions, very small lesions |

**Verdict**: Worth trying. Test on validation set. If it improves IoU by >0.5%, keep it.

---

## 5. Boundary-Aware Techniques

### 5.1 Boundary Loss During Training

```python
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    """
    Compute signed distance map from binary mask boundary.
    Positive inside, negative outside.
    """
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    dist_inside = distance_transform_edt(mask_bool)
    dist_outside = distance_transform_edt(~mask_bool)
    return (dist_inside - dist_outside).astype(np.float32)


class BoundaryLoss(nn.Module):
    """
    Boundary loss from Kervadec et al. (2019).
    Minimizes the distance between predicted and GT boundaries.

    Use in combination with Dice/BCE loss:
        total_loss = dice_loss + alpha * boundary_loss
    where alpha increases from 0 to 1 during training.
    """
    def forward(self, logits: torch.Tensor,
                dist_maps: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        loss = (probs * dist_maps).mean()
        return loss


class CombinedBoundaryLoss(nn.Module):
    """Dice + BCE + Boundary loss with scheduled alpha."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryLoss()
        self.alpha = 0.0  # Increases during training

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(2, 3))
        union_val = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union_val + smooth)
        return 1 - dice.mean()

    def forward(self, logits, targets, dist_maps):
        bce = self.bce(logits, targets)
        dice = self.dice_loss(logits, targets)
        boundary = self.boundary(logits, dist_maps)

        region_loss = 0.5 * bce + 0.5 * dice
        return (1 - self.alpha) * region_loss + self.alpha * boundary

    def update_alpha(self, epoch, max_epochs):
        """Linearly increase alpha from 0 to 1 over training."""
        self.alpha = min(1.0, epoch / max_epochs)
```

### 5.2 Boundary Refinement Post-Processing

```python
def refine_boundary_with_edges(image_rgb: np.ndarray,
                                binary_mask: np.ndarray,
                                sigma: float = 1.0) -> np.ndarray:
    """
    Refine mask boundary using image edge information.
    Snaps mask boundary to nearest strong edge in the image.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Get mask boundary
    mask_boundary = cv2.morphologyEx(
        binary_mask, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # Where mask boundary does not align with edges, smooth it
    misaligned = cv2.bitwise_and(mask_boundary, cv2.bitwise_not(edges_dilated))

    smoothed = smooth_boundary(binary_mask, sigma=sigma)

    result = binary_mask.copy()
    misaligned_dilated = cv2.dilate(misaligned, kernel, iterations=2)
    result[misaligned_dilated > 0] = smoothed[misaligned_dilated > 0]

    return result
```

### 5.3 Active Contour (Snake) Refinement

```python
from skimage.segmentation import active_contour
from skimage.filters import gaussian

def snake_refinement(image_rgb: np.ndarray,
                     binary_mask: np.ndarray,
                     alpha: float = 0.01,
                     beta: float = 0.1,
                     gamma: float = 0.01) -> np.ndarray:
    """
    Refine mask boundary using active contours (snakes).

    alpha: Elasticity (higher = smoother snake)
    beta: Rigidity (higher = less curvature)
    gamma: Time step
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    gray_smooth = gaussian(gray, sigma=2)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return binary_mask

    largest = max(contours, key=cv2.contourArea)
    init_snake = largest.squeeze()

    if len(init_snake) < 10:
        return binary_mask

    init_points = np.column_stack([init_snake[:, 1], init_snake[:, 0]])

    snake = active_contour(
        gray_smooth,
        init_points,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_iterations=250
    )

    refined_mask = np.zeros_like(binary_mask)
    snake_cv = snake[:, [1, 0]].astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(refined_mask, [snake_cv], -1, 255, cv2.FILLED)

    return refined_mask
```

**Note on active contours**: They can be slow and sensitive to parameters. CRF is generally more robust for this use case.

---

## 6. Proper Mask Resizing

### 6.1 The Critical Rule: Resize PROBABILITY MAP, Then Threshold

```python
# === CORRECT ORDER ===
# 1. Get probability map at model resolution
prob_map = torch.sigmoid(model(image_tensor))  # e.g., (1, 1, 384, 384)

# 2. Resize probability map to ORIGINAL image size using BILINEAR
prob_full = F.interpolate(
    prob_map,
    size=(orig_h, orig_w),
    mode='bilinear',
    align_corners=False
)

# 3. THEN threshold
binary_mask = (prob_full.squeeze().cpu().numpy() >= threshold)
final_mask = (binary_mask.astype(np.uint8) * 255)


# === WRONG ORDER ===
# 1. Threshold at model resolution
binary_small = (prob_map >= 0.5)  # Binary at 384x384

# 2. Resize binary mask -- introduces interpolation artifacts!
binary_full = F.interpolate(binary_small.float(), size=(orig_h, orig_w),
                             mode='nearest')  # Blocky edges!
# or even worse:
binary_full = F.interpolate(binary_small.float(), size=(orig_h, orig_w),
                             mode='bilinear')  # Creates non-binary values!
```

### 6.2 Interpolation Methods for Different Data Types

| Data Type | Interpolation | Why |
|-----------|--------------|-----|
| **Probability maps** (float) | `bilinear` or `bicubic` | Smooth gradients, preserves boundary info |
| **Binary masks** (0/255) | `nearest` | Keeps values binary, no interpolation artifacts |
| **Label maps** (0,1,2,...) | `nearest` | Prevents creating invalid class values |
| **RGB images** | `bilinear` or `bicubic` | Natural-looking resize |

### 6.3 Complete Resize Pipeline

```python
def resize_prob_and_threshold(prob_map: np.ndarray,
                               target_h: int, target_w: int,
                               threshold: float = 0.5) -> np.ndarray:
    """
    Properly resize a probability map and convert to binary mask.

    Args:
        prob_map: Model output probability map (float32)
        target_h, target_w: Original image dimensions
        threshold: Binarization threshold
    Returns:
        Binary mask (0 or 255), uint8, exact target dimensions
    """
    # Step 1: Resize probability map with bilinear interpolation
    prob_resized = cv2.resize(
        prob_map,
        (target_w, target_h),  # cv2.resize takes (width, height)
        interpolation=cv2.INTER_LINEAR
    )

    # Step 2: Threshold to binary
    binary = (prob_resized >= threshold).astype(np.uint8) * 255

    # Step 3: Verify exact dimensions
    assert binary.shape == (target_h, target_w), \
        f"Shape mismatch: {binary.shape} vs ({target_h}, {target_w})"

    return binary
```

### 6.4 Anti-Aliasing Considerations

```python
def resize_prob_map(prob_map: np.ndarray,
                    target_h: int, target_w: int) -> np.ndarray:
    """Resize with proper anti-aliasing based on scale direction."""
    src_h, src_w = prob_map.shape[:2]

    if target_h < src_h or target_w < src_w:
        # Downscaling: use INTER_AREA to avoid aliasing
        method = cv2.INTER_AREA
    else:
        # Upscaling: use INTER_LINEAR or INTER_CUBIC
        method = cv2.INTER_LINEAR

    return cv2.resize(prob_map, (target_w, target_h), interpolation=method)
```

---

## 7. Ensemble Mask Merging

### 7.1 Average Logits (Best Practice)

```python
def ensemble_logits(models: list, image: torch.Tensor) -> np.ndarray:
    """
    Average raw logits from multiple models, THEN apply sigmoid.
    Better than averaging probabilities because logits are unbounded.
    """
    logit_sum = None

    for model in models:
        with torch.no_grad():
            logits = model(image)

        if logit_sum is None:
            logit_sum = logits
        else:
            logit_sum = logit_sum + logits

    avg_logits = logit_sum / len(models)
    prob = torch.sigmoid(avg_logits)
    return prob.squeeze().cpu().numpy()
```

### 7.2 Weighted Averaging Based on Per-Model IoU

```python
def weighted_ensemble(models: list, model_ious: list,
                      image: torch.Tensor) -> np.ndarray:
    """
    Weight each model contribution by its validation IoU.
    """
    weights = np.array(model_ious)
    weights = weights / weights.sum()

    prob_sum = None

    for model, w in zip(models, weights):
        with torch.no_grad():
            logits = model(image)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        if prob_sum is None:
            prob_sum = prob * w
        else:
            prob_sum = prob_sum + prob * w

    return prob_sum.astype(np.float32)
```

### 7.3 Soft Voting vs Hard Voting

```python
def soft_vote(prob_maps: list, threshold: float = 0.5) -> np.ndarray:
    """
    Soft voting: average probability maps, then threshold.
    Better for boundary regions where models disagree slightly.
    """
    avg_prob = np.mean(prob_maps, axis=0)
    return (avg_prob >= threshold).astype(np.uint8) * 255


def hard_vote(prob_maps: list, threshold: float = 0.5) -> np.ndarray:
    """
    Hard voting: threshold each model, then majority vote.
    More robust when models make independent errors.
    """
    binary_masks = [(p >= threshold).astype(np.uint8) for p in prob_maps]
    vote_sum = np.sum(binary_masks, axis=0)
    majority = len(prob_maps) / 2.0
    return (vote_sum >= majority).astype(np.uint8) * 255
```

**Soft voting is generally better** for segmentation because it preserves boundary uncertainty.

### 7.4 STAPLE Algorithm for Mask Fusion

```python
import SimpleITK as sitk

def staple_fusion(binary_masks: list) -> np.ndarray:
    """
    STAPLE: Simultaneous Truth and Performance Level Estimation.
    Treats each model as an "expert" and estimates the true segmentation
    while simultaneously estimating each expert sensitivity/specificity.

    Better than simple averaging when models have different error patterns.
    """
    sitk_masks = []
    for mask in binary_masks:
        mask_bin = (mask > 0).astype(np.uint8)
        sitk_img = sitk.GetImageFromArray(mask_bin)
        sitk_img = sitk.Cast(sitk_img, sitk.sitkUInt8)
        sitk_masks.append(sitk_img)

    staple_filter = sitk.STAPLEImageFilter()
    staple_filter.SetForegroundValue(1)
    result = staple_filter.Execute(sitk_masks)

    prob_map = sitk.GetArrayFromImage(result).astype(np.float32)

    for i in range(len(binary_masks)):
        sens = staple_filter.GetSensitivity(i)
        spec = staple_filter.GetSpecificity(i)
        print(f"  Model {i}: sensitivity={sens:.4f}, specificity={spec:.4f}")

    return prob_map
```

### 7.5 Comparison of Ensemble Methods

| Method | IoU Gain | Speed | When Best |
|--------|----------|-------|-----------|
| **Average logits** | +1-2% | Fast | Default choice, calibrated models |
| **Weighted average** | +1-3% | Fast | When model quality varies |
| **Soft voting** | +1-2% | Fast | Simple, robust baseline |
| **Hard voting** | +0.5-1.5% | Fast | Independent error patterns |
| **STAPLE** | +1-3% | Medium | Different model architectures |

---

## 8. Quality Metrics Beyond IoU

### 8.1 Implementation of Key Metrics

```python
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute comprehensive segmentation metrics.

    Args:
        pred: Predicted binary mask (0 or 1)
        gt: Ground truth binary mask (0 or 1)
    Returns:
        Dictionary of metrics
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    # IoU (Jaccard Index)
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union_val = np.logical_or(pred_bool, gt_bool).sum()
    iou = intersection / (union_val + 1e-8)

    # Dice Coefficient
    dice = (2 * intersection) / (pred_bool.sum() + gt_bool.sum() + 1e-8)

    # Hausdorff Distance
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        hausdorff = float('inf')
        hausdorff_95 = float('inf')
    else:
        pred_boundary = _get_boundary_points(pred_bool)
        gt_boundary = _get_boundary_points(gt_bool)

        d1 = directed_hausdorff(pred_boundary, gt_boundary)[0]
        d2 = directed_hausdorff(gt_boundary, pred_boundary)[0]
        hausdorff = max(d1, d2)
        hausdorff_95 = _hausdorff_95(pred_boundary, gt_boundary)

    # Boundary F1 Score
    boundary_f1 = _boundary_f1(pred_bool, gt_bool, tolerance=2)

    # Surface Dice (Normalized Surface Distance)
    surface_dice = _surface_dice(pred_bool, gt_bool, tolerance=2)

    return {
        'iou': iou,
        'dice': dice,
        'hausdorff': hausdorff,
        'hausdorff_95': hausdorff_95,
        'boundary_f1': boundary_f1,
        'surface_dice': surface_dice,
    }


def _get_boundary_points(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixel coordinates from a binary mask."""
    eroded = cv2.erode(mask.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    boundary = mask.astype(np.uint8) - eroded
    points = np.argwhere(boundary > 0)
    return points


def _hausdorff_95(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """95th percentile Hausdorff distance."""
    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    all_distances = np.concatenate([dist_a_to_b, dist_b_to_a])
    return np.percentile(all_distances, 95)


def _boundary_f1(pred: np.ndarray, gt: np.ndarray,
                 tolerance: int = 2) -> float:
    """
    Boundary F1 score.
    A boundary pixel is "correct" if within `tolerance` pixels
    of a GT boundary pixel.
    """
    pred_boundary = _get_boundary_points(pred)
    gt_boundary = _get_boundary_points(gt)

    if len(pred_boundary) == 0 and len(gt_boundary) == 0:
        return 1.0
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return 0.0

    tree_gt = cKDTree(gt_boundary)
    tree_pred = cKDTree(pred_boundary)

    dist_pred_to_gt, _ = tree_gt.query(pred_boundary)
    precision = (dist_pred_to_gt <= tolerance).mean()

    dist_gt_to_pred, _ = tree_pred.query(gt_boundary)
    recall = (dist_gt_to_pred <= tolerance).mean()

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def _surface_dice(pred: np.ndarray, gt: np.ndarray,
                  tolerance: int = 2) -> float:
    """Normalized Surface Dice (same as boundary F1 for binary case)."""
    return _boundary_f1(pred, gt, tolerance)
```

### 8.2 Which Metric Correlates Best with Visual Quality?

| Metric | What It Measures | Visual Correlation | Sensitivity |
|--------|-----------------|-------------------|-------------|
| **IoU** | Region overlap | Medium | Insensitive to boundary details |
| **Dice** | Region overlap (F1) | Medium | Similar to IoU, more forgiving |
| **Hausdorff** | Worst boundary error | Low (one outlier dominates) | Very sensitive to outliers |
| **Hausdorff 95%** | Robust boundary error | High | Good balance |
| **Boundary F1** | Boundary accuracy | Very High | Best for boundary quality |
| **Surface Dice** | Boundary overlap | Very High | Best overall for visual quality |

**For hackathon optimization**: Focus on IoU (it is the scoring metric), but use Boundary F1 and Hausdorff 95% as diagnostic tools to find images where boundary quality is poor.

---

## 9. Complete Production Pipeline

### 9.1 Full Inference Pipeline

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MaskPredictionPipeline:
    """Complete pipeline for generating optimized binary masks."""

    def __init__(self,
                 models: list,
                 model_weights: list = None,
                 device: str = 'cuda',
                 input_size: int = 384,
                 threshold: float = 0.5,
                 use_tta: bool = True,
                 use_multiscale: bool = True,
                 scales: list = None,
                 use_crf: bool = False,
                 crf_params: dict = None,
                 morph_kernel_open: int = 5,
                 morph_kernel_close: int = 7,
                 min_component_ratio: float = 0.01):

        self.models = models
        self.model_weights = model_weights or [1.0 / len(models)] * len(models)
        self.device = device
        self.input_size = input_size
        self.threshold = threshold
        self.use_tta = use_tta
        self.use_multiscale = use_multiscale
        self.scales = scales or [0.75, 1.0, 1.25]
        self.use_crf = use_crf
        self.crf_params = crf_params or {
            'sxy_gaussian': 3, 'compat_gaussian': 3,
            'sxy_bilateral': 80, 'srgb_bilateral': 13,
            'compat_bilateral': 10, 'n_iters': 5,
        }
        self.morph_kernel_open = morph_kernel_open
        self.morph_kernel_close = morph_kernel_close
        self.min_component_ratio = min_component_ratio

        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        for m in self.models:
            m.to(device)

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image_rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
        augmented = self.transform(image=resized)
        tensor = augmented['image'].unsqueeze(0).to(self.device)
        return tensor

    def _tta_8x(self, model, image: torch.Tensor) -> np.ndarray:
        _, _, H, W = image.shape
        prob_sum = np.zeros((H, W), dtype=np.float64)

        for k in range(4):
            for do_flip in [False, True]:
                x = torch.rot90(image, k, dims=[2, 3])
                if do_flip:
                    x = torch.flip(x, dims=[3])

                with torch.no_grad():
                    logits = model(x)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                if do_flip:
                    prob = np.flip(prob, axis=1).copy()
                prob = np.rot90(prob, -k).copy()
                prob_sum += prob

        return (prob_sum / 8.0).astype(np.float32)

    def _multi_scale_tta(self, model, image: torch.Tensor) -> np.ndarray:
        _, _, H, W = image.shape
        prob_sum = np.zeros((H, W), dtype=np.float64)

        for scale in self.scales:
            new_h, new_w = int(H * scale), int(W * scale)
            scaled = F.interpolate(image, size=(new_h, new_w),
                                   mode='bilinear', align_corners=False)
            prob_at_scale = self._tta_8x(model, scaled)
            prob_resized = cv2.resize(prob_at_scale, (W, H),
                                      interpolation=cv2.INTER_LINEAR)
            prob_sum += prob_resized

        return (prob_sum / len(self.scales)).astype(np.float32)

    def _predict_single_model(self, model, image_tensor: torch.Tensor,
                               orig_h: int, orig_w: int) -> np.ndarray:
        if self.use_multiscale and self.use_tta:
            prob = self._multi_scale_tta(model, image_tensor)
        elif self.use_tta:
            prob = self._tta_8x(model, image_tensor)
        else:
            with torch.no_grad():
                logits = model(image_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        prob_resized = cv2.resize(prob, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)
        return prob_resized

    def _ensemble_predictions(self, image_tensor: torch.Tensor,
                               orig_h: int, orig_w: int) -> np.ndarray:
        prob_sum = np.zeros((orig_h, orig_w), dtype=np.float64)

        for model, weight in zip(self.models, self.model_weights):
            prob = self._predict_single_model(model, image_tensor,
                                              orig_h, orig_w)
            prob_sum += prob * weight

        return prob_sum.astype(np.float32)

    def _apply_crf(self, image_rgb: np.ndarray,
                    prob_map: np.ndarray) -> np.ndarray:
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            print("Warning: pydensecrf not installed, skipping CRF")
            return prob_map

        h, w = prob_map.shape
        d = dcrf.DenseCRF2D(w, h, 2)

        prob_fg = np.clip(prob_map, 1e-6, 1.0 - 1e-6)
        prob_bg = 1.0 - prob_fg
        probs = np.stack([prob_bg, prob_fg], axis=0)
        unary = unary_from_softmax(probs)
        d.setUnaryEnergy(unary)

        d.addPairwiseGaussian(
            sxy=self.crf_params['sxy_gaussian'],
            compat=self.crf_params['compat_gaussian'],
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )

        img_c = np.ascontiguousarray(image_rgb)
        d.addPairwiseBilateral(
            sxy=self.crf_params['sxy_bilateral'],
            srgb=self.crf_params['srgb_bilateral'],
            rgbim=img_c,
            compat=self.crf_params['compat_bilateral'],
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )

        Q = d.inference(self.crf_params['n_iters'])
        Q = np.array(Q).reshape((2, h, w))
        return Q[1].astype(np.float32)

    def _postprocess(self, binary_mask: np.ndarray) -> np.ndarray:
        mask = binary_mask.copy()

        # Step 1: Opening (remove small noise)
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_open, self.morph_kernel_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Step 2: Keep largest connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            total_area = mask.shape[0] * mask.shape[1]
            min_area = total_area * self.min_component_ratio

            result = np.zeros_like(mask)
            largest_idx = np.argmax(areas) + 1
            result[labels == largest_idx] = 255

            for i in range(1, num_labels):
                if i == largest_idx:
                    continue
                if (stats[i, cv2.CC_STAT_AREA] >= min_area and
                        stats[i, cv2.CC_STAT_AREA] >= areas[largest_idx - 1] * 0.1):
                    result[labels == i] = 255
            mask = result

        # Step 3: Fill holes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        mask = filled

        # Step 4: Closing (smooth boundaries, fill small gaps)
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_close, self.morph_kernel_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close,
                                iterations=1)

        # Step 5: Smooth boundary
        float_mask = mask.astype(np.float32) / 255.0
        sigma = 1.5
        ksize = int(sigma * 6) | 1
        smoothed = cv2.GaussianBlur(float_mask, (ksize, ksize), sigma)
        mask = (smoothed >= 0.5).astype(np.uint8) * 255

        return mask

    def predict(self, image_path: str) -> np.ndarray:
        """
        Full prediction pipeline for a single image.
        Returns: Binary mask (0 or 255), uint8, same size as input image
        """
        image_rgb = np.array(Image.open(image_path).convert('RGB'))
        orig_h, orig_w = image_rgb.shape[:2]

        image_tensor = self._preprocess(image_rgb)

        # Ensemble predictions (multi-scale + TTA included)
        prob_map = self._ensemble_predictions(image_tensor, orig_h, orig_w)

        # Optional CRF refinement
        if self.use_crf:
            prob_map = self._apply_crf(image_rgb, prob_map)

        # Threshold
        binary = (prob_map >= self.threshold).astype(np.uint8) * 255

        # Post-process
        final_mask = self._postprocess(binary)

        # Verify dimensions
        assert final_mask.shape == (orig_h, orig_w), \
            f"Mask shape {final_mask.shape} != image shape ({orig_h}, {orig_w})"
        assert set(np.unique(final_mask)).issubset({0, 255}), \
            f"Mask values not binary: {np.unique(final_mask)}"

        return final_mask

    def predict_batch(self, image_dir: str, output_dir: str):
        """Process all test images and save binary masks."""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
        )
        print(f"Processing {len(image_paths)} images...")

        for i, img_path in enumerate(image_paths):
            mask = self.predict(str(img_path))
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), mask)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")

        print(f"Done. Masks saved to {output_dir}")
```

### 9.2 Quick Usage Example

```python
import segmentation_models_pytorch as smp

# Load models
model1 = smp.UnetPlusPlus(encoder_name='efficientnet-b4',
                           in_channels=3, classes=1)
model1.load_state_dict(torch.load('checkpoints/unetpp_effb4.pth'))

model2 = smp.DeepLabV3Plus(encoder_name='resnet101',
                            in_channels=3, classes=1)
model2.load_state_dict(torch.load('checkpoints/deeplabv3p_r101.pth'))

# Create pipeline
pipeline = MaskPredictionPipeline(
    models=[model1, model2],
    model_weights=[0.55, 0.45],  # Weight by validation IoU
    device='cuda',
    input_size=384,
    threshold=0.45,           # Tuned on validation set
    use_tta=True,             # 8x TTA
    use_multiscale=True,      # Multi-scale
    scales=[0.75, 1.0, 1.25],
    use_crf=True,             # CRF refinement
    morph_kernel_open=5,
    morph_kernel_close=7,
    min_component_ratio=0.005,
)

# Run on test set
pipeline.predict_batch(
    image_dir='data/Segmentation/testing/images/',
    output_dir='WhiteCoat.dev/WhiteCoat.dev/'
)
```

### 9.3 Ablation Study Template

```python
def run_ablation(models, val_images, val_masks):
    """Test each component contribution to final IoU."""
    configs = {
        'baseline (no TTA, thresh=0.5)': {
            'use_tta': False, 'use_multiscale': False,
            'threshold': 0.5, 'use_crf': False,
        },
        '+ optimized threshold': {
            'use_tta': False, 'use_multiscale': False,
            'threshold': 0.45, 'use_crf': False,
        },
        '+ TTA 8x': {
            'use_tta': True, 'use_multiscale': False,
            'threshold': 0.45, 'use_crf': False,
        },
        '+ multi-scale': {
            'use_tta': True, 'use_multiscale': True,
            'threshold': 0.45, 'use_crf': False,
        },
        '+ CRF': {
            'use_tta': True, 'use_multiscale': True,
            'threshold': 0.45, 'use_crf': True,
        },
    }

    for name, config in configs.items():
        pipeline = MaskPredictionPipeline(models=models, **config)
        ious = []
        for img_path, gt_mask in zip(val_images, val_masks):
            pred = pipeline.predict(img_path)
            pred_bin = (pred > 0).astype(np.uint8)
            gt_bin = (gt_mask > 0).astype(np.uint8)

            intersection = np.logical_and(pred_bin, gt_bin).sum()
            union_val = np.logical_or(pred_bin, gt_bin).sum()
            ious.append(intersection / (union_val + 1e-8))

        print(f"{name:40s} -> Mean IoU: {np.mean(ious):.4f}")
```

---

## Summary: Expected IoU Gains

| Technique | Expected Gain | Effort | Priority |
|-----------|---------------|--------|----------|
| Optimized threshold (val search) | +1-2% | Low | HIGH |
| TTA 8x (flips + rotations) | +1-3% | Low | HIGH |
| Multi-scale inference | +1-2% | Low | HIGH |
| Model ensemble (2-3 models) | +2-4% | Medium | HIGH |
| Morphological post-processing | +0.5-1% | Low | MEDIUM |
| Hole filling | +0.5-1% | Low | MEDIUM |
| CRF refinement | +0.5-2% | Medium | MEDIUM |
| Boundary smoothing | +0.2-0.5% | Low | MEDIUM |
| STAPLE fusion | +0.5-1% | Medium | LOW |
| Boundary loss (training) | +1-3% | High (retrain) | LOW (if no time) |
| Active contours | +0.1-0.5% | Medium | LOW |

**Cumulative potential**: With all techniques combined, moving from 0.81 to 0.88-0.92 IoU is realistic.

**Priority order for maximum impact with limited time**:
1. Optimize threshold on validation set
2. Enable TTA 8x
3. Multi-scale inference (3 scales)
4. Ensemble 2+ models
5. Morphological cleanup + hole filling
6. CRF (if time permits and validation shows improvement)
