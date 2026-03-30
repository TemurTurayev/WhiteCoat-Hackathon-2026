# Segmentation Optimization: From 0.81 to 0.90+ IoU

## Current Setup
- Dataset: 1800 train + 400 val, binary masks
- Models: U-Net++ (EfficientNet-B4), DeepLabV3+ (ResNet50), U-Net (ResNet34)
- Current best IoU: ~0.81
- Target: 0.90+ IoU

---

## 1. Optimal Threshold Selection

### Problem
Default threshold of 0.5 is almost never optimal. Finding the right threshold can boost IoU by 2-5%.

### Global Optimal Threshold (on Validation Set)

```python
import numpy as np
import torch

def find_optimal_threshold(model, val_loader, device, thresholds=None):
    """Search for the threshold that maximizes IoU on the validation set."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.75, 0.05)

    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            preds = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.append(preds)
            all_masks.append(masks.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    best_iou = 0
    best_thresh = 0.5

    for thresh in thresholds:
        binary_preds = (all_preds > thresh).astype(np.float32)
        intersection = (binary_preds * all_masks).sum()
        union = binary_preds.sum() + all_masks.sum() - intersection
        iou = intersection / (union + 1e-7)

        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    # Fine-grained search around best threshold
    fine_thresholds = np.arange(best_thresh - 0.05, best_thresh + 0.05, 0.01)
    for thresh in fine_thresholds:
        binary_preds = (all_preds > thresh).astype(np.float32)
        intersection = (binary_preds * all_masks).sum()
        union = binary_preds.sum() + all_masks.sum() - intersection
        iou = intersection / (union + 1e-7)

        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    return best_thresh, best_iou
```

### Per-Image Adaptive Threshold (Advanced)

For skin lesions, per-image thresholding can help because lesion contrast varies widely:

```python
def adaptive_threshold_per_image(prob_mask, method='otsu'):
    """Apply adaptive threshold per image using Otsu or percentile."""
    from skimage.filters import threshold_otsu

    if method == 'otsu':
        try:
            thresh = threshold_otsu(prob_mask)
        except ValueError:
            thresh = 0.5
        return (prob_mask > thresh).astype(np.uint8)

    elif method == 'percentile':
        # Use top percentile of probability values
        if prob_mask.max() < 0.1:
            return np.zeros_like(prob_mask, dtype=np.uint8)
        thresh = np.percentile(prob_mask[prob_mask > 0.1], 30)
        return (prob_mask > thresh).astype(np.uint8)
```

### Key Insight
- Start with global threshold search on validation set
- Typical optimal range for skin lesion segmentation: 0.35-0.55
- If distribution of lesion sizes varies a lot, per-image Otsu can help
- Always validate threshold choice on held-out data

---

## 2. Post-Processing Techniques

### Recommended Order of Operations
1. Threshold the probability map
2. Fill holes
3. Remove small components (keep largest connected component)
4. Morphological closing (fill gaps)
5. Morphological opening (remove noise)
6. (Optional) CRF refinement
7. (Optional) Boundary smoothing with Gaussian blur

### Complete Post-Processing Pipeline

```python
import cv2
import numpy as np
from scipy import ndimage

def postprocess_mask(prob_mask, threshold=0.5, min_area_ratio=0.005):
    """
    Full post-processing pipeline for binary segmentation masks.

    Args:
        prob_mask: probability map (H, W), values in [0, 1]
        threshold: binarization threshold
        min_area_ratio: minimum area of component relative to image area
    Returns:
        cleaned binary mask (H, W), values 0 or 255
    """
    h, w = prob_mask.shape

    # Step 1: Threshold
    binary = (prob_mask > threshold).astype(np.uint8)

    # Step 2: Fill holes
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

    # Step 3: Remove small connected components, keep largest
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels > 1:
        # stats[:, 4] is the area of each component
        # Skip label 0 (background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        min_area = h * w * min_area_ratio

        # Keep only components larger than min_area
        cleaned = np.zeros_like(binary)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label_id] = 1
        binary = cleaned

    # Step 4: Morphological closing (fills small gaps)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Step 5: Morphological opening (removes small noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Step 6: Optional boundary smoothing
    binary_smooth = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
    binary = (binary_smooth > 0.5).astype(np.uint8)

    return binary * 255


def keep_largest_component(binary_mask):
    """Keep only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if num_labels <= 1:
        return binary_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    result = np.zeros_like(binary_mask)
    result[labels == largest_label] = 1
    return result
```

### CRF Post-Processing

CRF can improve IoU by 1-3% by enforcing spatial consistency at boundaries:

```python
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(image, prob_mask, n_iterations=5):
    """
    Apply DenseCRF to refine segmentation boundaries.

    Args:
        image: original RGB image (H, W, 3), uint8
        prob_mask: probability map (H, W), values in [0, 1]
        n_iterations: CRF inference iterations
    Returns:
        refined binary mask (H, W)
    """
    h, w = prob_mask.shape

    # Create 2-class probability map
    prob_fg = prob_mask.copy()
    prob_bg = 1.0 - prob_fg
    probs = np.stack([prob_bg, prob_fg], axis=0)  # (2, H, W)

    # Setup CRF
    d = dcrf.DenseCRF2D(w, h, 2)

    # Unary potentials
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Pairwise Gaussian (spatial smoothness)
    d.addPairwiseGaussian(
        sxy=3,       # spatial smoothness
        compat=3,    # compatibility
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Pairwise bilateral (appearance-based, uses color)
    d.addPairwiseBilateral(
        sxy=40,      # spatial smoothness
        srgb=13,     # color smoothness
        rgbim=image,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Inference
    Q = d.inference(n_iterations)
    result = np.argmax(np.array(Q).reshape(2, h, w), axis=0)

    return result.astype(np.uint8)
```

### Installation for CRF
```bash
pip install cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

### Expected Improvement
- Morphological operations alone: +1-2% IoU
- Connected component analysis: +0.5-1% IoU (eliminates false positives)
- CRF: +1-3% IoU (best for boundary refinement)
- Hole filling: +0.5-1% IoU

---

## 3. Loss Functions for Maximizing IoU

### Strategy: Two-Phase Training

Phase 1 (BCE/CE warmup): Train with pixel-wise loss for stable convergence
Phase 2 (IoU-aware loss): Switch to or add Lovasz/Dice loss to directly optimize IoU

### Loss Function Implementations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class LovaszHingeLoss(nn.Module):
    """
    Binary Lovasz hinge loss. Directly optimizes IoU.
    From: https://github.com/bermanmaxim/LovaszSoftmax
    """
    def forward(self, logits, targets):
        return lovasz_hinge(logits, targets)


def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels):
    """Binary Lovasz hinge loss.
    logits: (B, 1, H, W) raw model output (before sigmoid)
    labels: (B, 1, H, W) binary ground truth
    """
    losses = []
    for logit, label in zip(logits.squeeze(1), labels.squeeze(1)):
        logit_flat = logit.view(-1)
        label_flat = label.view(-1)
        signs = 2.0 * label_flat.float() - 1.0
        errors = 1.0 - logit_flat * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = label_flat[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        losses.append(loss)
    return torch.stack(losses).mean()


class ComboLoss(nn.Module):
    """BCE + Dice + Lovasz combo loss.

    Recommended schedule:
    - Epochs 1-15:  bce_w=1.0, dice_w=0.5, lovasz_w=0.0  (warmup)
    - Epochs 16-30: bce_w=0.5, dice_w=0.5, lovasz_w=0.5  (transition)
    - Epochs 31+:   bce_w=0.2, dice_w=0.3, lovasz_w=0.5  (IoU focus)
    """
    def __init__(self, bce_w=0.5, dice_w=0.3, lovasz_w=0.2):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.lovasz_w = lovasz_w
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        loss = self.bce_w * self.bce(logits, targets)
        loss += self.dice_w * self.dice(logits, targets)
        if self.lovasz_w > 0:
            loss += self.lovasz_w * lovasz_hinge(logits, targets)
        return loss


class BoundaryLoss(nn.Module):
    """Boundary loss for fine boundary segmentation.
    Requires precomputed distance maps from ground truth boundaries.
    """
    def forward(self, logits, dist_maps):
        probs = torch.sigmoid(logits)
        # dist_maps: signed distance transform of GT boundary
        # positive inside, negative outside
        return (probs * dist_maps).mean()
```

### Using segmentation_models_pytorch Losses (Recommended)

```python
import segmentation_models_pytorch as smp

# These are already well-implemented and tested:
dice_loss = smp.losses.DiceLoss(mode='binary')
focal_loss = smp.losses.FocalLoss(mode='binary')
lovasz_loss = smp.losses.LovaszLoss(mode='binary')

# Combo: BCE + Dice + Lovasz
class SMPComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.lovasz = smp.losses.LovaszLoss(mode='binary')

    def forward(self, logits, targets):
        return (
            0.4 * self.bce(logits, targets)
            + 0.3 * self.dice(logits, targets)
            + 0.3 * self.lovasz(logits, targets)
        )
```

### Two-Phase Training Schedule

```python
def get_loss_for_epoch(epoch, total_epochs):
    """Dynamic loss scheduling."""
    warmup_epochs = total_epochs // 4

    if epoch < warmup_epochs:
        # Phase 1: BCE + Dice (stable convergence)
        return BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
    else:
        # Phase 2: Add Lovasz for IoU optimization
        return ComboLoss(bce_w=0.3, dice_w=0.3, lovasz_w=0.4)
```

### Key Insight
- BCE alone converges well but does not directly optimize IoU
- Dice loss directly correlates with IoU improvement
- Lovasz hinge is the only loss that is a true surrogate for IoU
- Two-phase training (BCE warmup then Lovasz/Dice) gives best results
- Expected improvement from loss function alone: +2-4% IoU

---

## 4. Test-Time Augmentation (TTA) for Segmentation

### Core Principle
Apply augmentations to input, predict mask, apply INVERSE transform to mask, then average all masks.

### Implementation

```python
import torch
import numpy as np
import albumentations as A

def tta_predict(model, image_tensor, device, n_augments='d4'):
    """
    Test-time augmentation for binary segmentation.

    Args:
        model: trained model
        image_tensor: (1, C, H, W) normalized input
        device: torch device
        n_augments: 'd4' for 8 transforms, 'flips' for 4 transforms
    Returns:
        averaged probability mask (H, W)
    """
    model.eval()

    def predict(x):
        with torch.no_grad():
            logit = model(x.to(device))
            return torch.sigmoid(logit).cpu()

    predictions = []

    # Original
    predictions.append(predict(image_tensor))

    # Horizontal flip
    flipped_h = torch.flip(image_tensor, dims=[3])
    pred_h = predict(flipped_h)
    predictions.append(torch.flip(pred_h, dims=[3]))

    # Vertical flip
    flipped_v = torch.flip(image_tensor, dims=[2])
    pred_v = predict(flipped_v)
    predictions.append(torch.flip(pred_v, dims=[2]))

    # Both flips
    flipped_hv = torch.flip(image_tensor, dims=[2, 3])
    pred_hv = predict(flipped_hv)
    predictions.append(torch.flip(pred_hv, dims=[2, 3]))

    if n_augments == 'd4':
        # 90-degree rotation
        rot90 = torch.rot90(image_tensor, k=1, dims=[2, 3])
        pred_r90 = predict(rot90)
        predictions.append(torch.rot90(pred_r90, k=-1, dims=[2, 3]))

        # 180-degree rotation
        rot180 = torch.rot90(image_tensor, k=2, dims=[2, 3])
        pred_r180 = predict(rot180)
        predictions.append(torch.rot90(pred_r180, k=-2, dims=[2, 3]))

        # 270-degree rotation
        rot270 = torch.rot90(image_tensor, k=3, dims=[2, 3])
        pred_r270 = predict(rot270)
        predictions.append(torch.rot90(pred_r270, k=-3, dims=[2, 3]))

        # Transpose
        transposed = image_tensor.permute(0, 1, 3, 2)
        pred_t = predict(transposed)
        predictions.append(pred_t.permute(0, 1, 3, 2))

    # Average all predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred.squeeze().numpy()


def tta_predict_logits(model, image_tensor, device):
    """
    TTA averaging logits (slightly better than averaging probabilities).
    """
    model.eval()

    def predict_logit(x):
        with torch.no_grad():
            return model(x.to(device)).cpu()

    logits = []

    # Original
    logits.append(predict_logit(image_tensor))

    # Horizontal flip
    flipped_h = torch.flip(image_tensor, dims=[3])
    logit_h = predict_logit(flipped_h)
    logits.append(torch.flip(logit_h, dims=[3]))

    # Vertical flip
    flipped_v = torch.flip(image_tensor, dims=[2])
    logit_v = predict_logit(flipped_v)
    logits.append(torch.flip(logit_v, dims=[2]))

    # Both flips
    flipped_hv = torch.flip(image_tensor, dims=[2, 3])
    logit_hv = predict_logit(flipped_hv)
    logits.append(torch.flip(logit_hv, dims=[2, 3]))

    # Average logits, then sigmoid
    avg_logit = torch.stack(logits).mean(dim=0)
    avg_pred = torch.sigmoid(avg_logit)
    return avg_pred.squeeze().numpy()
```

### Using ttach Library (Simpler)

```python
import ttach as tta

# Define TTA transforms
tta_transforms = tta.Compose([
    tta.HorizontalFlip(),
    tta.VerticalFlip(),
    tta.Rotate90(angles=[0, 90, 180, 270]),
])

# Wrap model
tta_model = tta.SegmentationTTAWrapper(
    model,
    tta_transforms,
    merge_mode='mean'
)

# Use like a normal model
with torch.no_grad():
    tta_output = tta_model(image_tensor.to(device))
```

### How Many TTA Transforms?

| Transforms | Count | Typical IoU Gain | Inference Time |
|-----------|-------|-----------------|----------------|
| None | 1x | baseline | 1x |
| H-flip only | 2x | +0.5-1.0% | 2x |
| H+V flip | 4x | +1.0-1.5% | 4x |
| D4 (flips + rotations) | 8x | +1.5-2.0% | 8x |
| D4 + multi-scale | 16-24x | +2.0-3.0% | 16-24x |

### Key Insight
- For skin lesions, D4 (8x) is ideal because lesions have no canonical orientation
- Average logits (before sigmoid) is marginally better than averaging probabilities
- Diminishing returns beyond D4 for this task
- Expected improvement: +1.5-2.5% IoU

---

## 5. Image Size Impact

### Resolution vs IoU (General Findings)

| Resolution | Relative IoU | Training Time | Memory |
|-----------|-------------|---------------|--------|
| 128x128 | baseline | 1x | 1x |
| 256x256 | +3-5% | 4x | 4x |
| 384x384 | +5-8% | 9x | 9x |
| 512x512 | +6-10% | 16x | 16x |

For skin lesion segmentation on dermoscopic images, 384-512 is the sweet spot.

### Progressive Resizing Strategy

Train at low resolution first, then fine-tune at higher resolution:

```python
def progressive_resize_training(model, train_dataset, val_dataset, device):
    """
    Progressive resizing: start small, increase resolution.
    Trains faster and can improve final IoU.
    """
    phases = [
        {'size': 256, 'epochs': 20, 'lr': 1e-3, 'batch_size': 32},
        {'size': 384, 'epochs': 15, 'lr': 5e-4, 'batch_size': 16},
        {'size': 512, 'epochs': 10, 'lr': 1e-4, 'batch_size': 8},
    ]

    for phase in phases:
        size = phase['size']

        train_transform = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15,
                rotate_limit=45, p=0.5
            ),
            A.OneOf([
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ], p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20, p=0.3
            ),
            A.CoarseDropout(
                max_holes=8, max_height=size//16,
                max_width=size//16, p=0.3
            ),
            A.Normalize(),
        ])

        val_transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(),
        ])

        # Update dataset transforms
        train_dataset.transform = train_transform
        val_dataset.transform = val_transform

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=phase['lr'],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase['epochs']
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=phase['batch_size'],
            shuffle=True,
            num_workers=4
        )

        # Train for this phase...
        print(f"Phase: {size}x{size}, {phase['epochs']} epochs, "
              f"lr={phase['lr']}, bs={phase['batch_size']}")
```

### Multi-Scale Inference

Predict at multiple scales and average:

```python
def multiscale_predict(model, image, device, scales=[0.75, 1.0, 1.25]):
    """Predict at multiple scales and average."""
    model.eval()
    h, w = image.shape[2], image.shape[3]
    predictions = []

    for scale in scales:
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize input
        scaled = F.interpolate(
            image, size=(new_h, new_w),
            mode='bilinear', align_corners=False
        )

        with torch.no_grad():
            pred = model(scaled.to(device))

        # Resize prediction back to original size
        pred_resized = F.interpolate(
            pred, size=(h, w),
            mode='bilinear', align_corners=False
        )
        predictions.append(pred_resized.cpu())

    # Average logits
    avg_pred = torch.stack(predictions).mean(dim=0)
    return torch.sigmoid(avg_pred)
```

### Key Insight
- Moving from 128 to 256 gives the biggest jump (+3-5% IoU)
- 384 is a good balance of quality and speed
- 512 gives marginal improvement over 384 but doubles memory
- Progressive resizing: faster training AND sometimes better final IoU
- Multi-scale inference at test time: +0.5-1.0% IoU additional

---

## 6. Ensemble Strategies for Segmentation

### Method Comparison

| Method | How | Best When |
|--------|-----|-----------|
| Average logits | Mean of raw outputs, then sigmoid | Models output similar ranges |
| Average probabilities | Mean of sigmoid outputs | Simple, robust default |
| Weighted average | Weighted mean by val IoU | Models differ in quality |
| Majority voting | Pixel is 1 if >50% models agree | Many diverse models |
| Stacking | Train meta-learner on model outputs | Large validation set |

### Implementation

```python
import torch
import torch.nn.functional as F

def ensemble_average_logits(models, image, device):
    """Average logits from multiple models (best practice)."""
    all_logits = []

    for model in models:
        model.eval()
        with torch.no_grad():
            logit = model(image.to(device))
            all_logits.append(logit.cpu())

    avg_logit = torch.stack(all_logits).mean(dim=0)
    return torch.sigmoid(avg_logit)


def ensemble_weighted_average(models, weights, image, device):
    """
    Weighted average of probabilities.
    Weights should be proportional to validation IoU.
    """
    all_probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            logit = model(image.to(device))
            prob = torch.sigmoid(logit).cpu()
            all_probs.append(prob)

    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()

    weighted_prob = sum(w * p for w, p in zip(weights, all_probs))
    return weighted_prob


def ensemble_majority_voting(models, image, device, threshold=0.5):
    """Majority voting: pixel is positive if >50% of models agree."""
    votes = []

    for model in models:
        model.eval()
        with torch.no_grad():
            logit = model(image.to(device))
            pred = (torch.sigmoid(logit) > threshold).float().cpu()
            votes.append(pred)

    vote_sum = torch.stack(votes).sum(dim=0)
    majority = (vote_sum > len(models) / 2).float()
    return majority
```

### Multi-Scale Ensemble

```python
def multi_resolution_ensemble(model, image_np, device, sizes=[256, 384, 512]):
    """
    Run same model at multiple resolutions and average.
    Captures both fine detail (high res) and context (low res).
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    h_orig, w_orig = image_np.shape[:2]
    all_preds = []

    for size in sizes:
        transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(),
            ToTensorV2(),
        ])

        augmented = transform(image=image_np)
        img_tensor = augmented['image'].unsqueeze(0)

        with torch.no_grad():
            logit = model(img_tensor.to(device))

        # Resize prediction back to original size
        pred = F.interpolate(
            logit, size=(h_orig, w_orig),
            mode='bilinear', align_corners=False
        )
        all_preds.append(pred.cpu())

    avg_logit = torch.stack(all_preds).mean(dim=0)
    return torch.sigmoid(avg_logit).squeeze().numpy()
```

### Weight Calculation from Validation IoU

```python
def calculate_ensemble_weights(val_ious):
    """
    Calculate weights proportional to validation IoU.

    Example:
        val_ious = [0.82, 0.79, 0.76]  # U-Net++, DeepLabV3+, U-Net
        weights = calculate_ensemble_weights(val_ious)
        # weights ~ [0.346, 0.333, 0.321]
    """
    ious = np.array(val_ious)
    # Softmax-style weighting with temperature
    temperature = 0.1
    exp_ious = np.exp(ious / temperature)
    weights = exp_ious / exp_ious.sum()
    return weights.tolist()
```

### Key Insight
- Average logits > average probabilities > majority voting (in most cases)
- 3 diverse models is the sweet spot for ensemble
- Diversity matters more than individual model quality
- Multi-resolution ensemble from the same model gives free improvement
- Expected improvement from 3-model ensemble: +1-3% IoU

---

## 7. Visual Quality Assessment and Failure Modes

### Metrics Beyond IoU

```python
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import binary_erosion, binary_dilation, disk

def hausdorff_distance(pred, gt):
    """
    Hausdorff distance: measures worst-case boundary error.
    Lower is better.
    """
    pred_boundary = pred ^ binary_erosion(pred, disk(1))
    gt_boundary = gt ^ binary_erosion(gt, disk(1))

    pred_points = np.argwhere(pred_boundary)
    gt_points = np.argwhere(gt_boundary)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')

    d1 = directed_hausdorff(pred_points, gt_points)[0]
    d2 = directed_hausdorff(gt_points, pred_points)[0]
    return max(d1, d2)


def boundary_f1_score(pred, gt, tolerance=2):
    """
    Boundary F1: precision/recall of boundary pixels within tolerance.
    Useful for evaluating boundary quality specifically.
    """
    pred_boundary = pred ^ binary_erosion(pred, disk(1))
    gt_boundary = gt ^ binary_erosion(gt, disk(1))

    # Dilate boundaries by tolerance
    gt_dilated = binary_dilation(gt_boundary, disk(tolerance))
    pred_dilated = binary_dilation(pred_boundary, disk(tolerance))

    # Precision: what fraction of pred boundary is near GT boundary
    if pred_boundary.sum() == 0:
        precision = 0
    else:
        precision = (pred_boundary & gt_dilated).sum() / pred_boundary.sum()

    # Recall: what fraction of GT boundary is near pred boundary
    if gt_boundary.sum() == 0:
        recall = 0
    else:
        recall = (gt_boundary & pred_dilated).sum() / gt_boundary.sum()

    if precision + recall == 0:
        return 0

    f1 = 2 * precision * recall / (precision + recall)
    return f1
```

### Common Failure Modes in Skin Lesion Segmentation

1. **Hair artifacts**: Dense hair crossing the lesion causes false negatives or fragmented masks
   - Fix: Hair removal preprocessing, CoarseDropout augmentation
2. **Low contrast boundaries**: Faint lesion borders lead to under-segmentation
   - Fix: CLAHE preprocessing, boundary-aware loss
3. **Small lesions**: Tiny lesions get missed entirely
   - Fix: Higher resolution, focal loss for hard pixels
4. **Gel bubbles / artifacts**: Circular artifacts cause false positives
   - Fix: Connected component filtering, augmentation with synthetic artifacts
5. **Color variation**: Different skin tones affect contrast
   - Fix: Color constancy normalization, HueSaturationValue augmentation
6. **Ambiguous boundaries**: Gradual transition from lesion to healthy skin
   - Fix: CRF post-processing, boundary loss, multi-scale features

### Visualization for Debugging

```python
import matplotlib.pyplot as plt

def visualize_segmentation_quality(image, gt_mask, pred_mask, prob_map=None):
    """Visualize prediction quality with error highlighting."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')

    # Ground truth
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth')

    # Prediction
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title('Prediction')

    # Overlay: GT boundary on image
    gt_boundary = gt_mask ^ binary_erosion(gt_mask, disk(2))
    overlay = image.copy()
    overlay[gt_boundary > 0] = [0, 255, 0]  # Green for GT
    pred_boundary = pred_mask ^ binary_erosion(pred_mask, disk(2))
    overlay[pred_boundary > 0] = [255, 0, 0]  # Red for prediction
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Boundaries: Green=GT, Red=Pred')

    # Error map
    tp = (pred_mask > 0) & (gt_mask > 0)  # True positive
    fp = (pred_mask > 0) & (gt_mask == 0)  # False positive
    fn = (pred_mask == 0) & (gt_mask > 0)  # False negative
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    error_map[tp] = [0, 255, 0]    # Green: correct
    error_map[fp] = [255, 0, 0]    # Red: false positive
    error_map[fn] = [0, 0, 255]    # Blue: false negative
    axes[1, 1].imshow(error_map)
    axes[1, 1].set_title('Errors: Red=FP, Blue=FN, Green=TP')

    # Probability map
    if prob_map is not None:
        axes[1, 2].imshow(prob_map, cmap='hot')
        axes[1, 2].set_title('Probability Map')
    else:
        axes[1, 2].axis('off')

    # Compute metrics
    iou = tp.sum() / (tp.sum() + fp.sum() + fn.sum() + 1e-7)
    dice = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-7)
    fig.suptitle(f'IoU: {iou:.4f} | Dice: {dice:.4f}', fontsize=14)

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    return fig
```

---

## 8. ISIC Competition Winner Tricks

### Top Techniques from ISIC Segmentation Challenges

#### Architecture Tricks
- **Deep supervision**: Add auxiliary loss at intermediate decoder stages, not just the final output. Forces all decoder levels to learn useful features.
- **Attention gates**: SE blocks and spatial attention between encoder and decoder improve boundary precision.
- **ASPP / multi-scale features**: Atrous Spatial Pyramid Pooling captures objects at multiple scales.

#### Preprocessing Tricks
- **Color constancy (Shades of Gray)**: Normalizes illumination differences across images. Significant improvement on ISIC data.
- **CLAHE**: Contrast-limited adaptive histogram equalization improves visibility of low-contrast boundaries.
- **Hair removal**: DullRazor or morphological black-hat filtering to remove hair artifacts.

```python
def color_constancy(image, power=6, gamma=None):
    """
    Shades of Gray color constancy.
    Normalizes illumination across different dermoscopic images.
    """
    image = image.astype('float32')
    img_power = np.power(image, power)
    rgb_vec = np.power(np.mean(img_power, axis=(0, 1)), 1.0 / power)
    rgb_norm = np.sqrt(np.sum(rgb_vec ** 2))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1.0 / (rgb_vec * np.sqrt(3))

    img_corrected = np.zeros_like(image)
    for i in range(3):
        img_corrected[:, :, i] = image[:, :, i] * rgb_vec[i]

    img_corrected = np.clip(img_corrected, 0, 255).astype('uint8')
    if gamma is not None:
        img_corrected = np.power(
            img_corrected / 255.0, gamma
        ) * 255
        img_corrected = img_corrected.astype('uint8')

    return img_corrected


def remove_hair_blackhat(image, kernel_size=17):
    """
    Remove hair artifacts using morphological black-hat transform.
    Simple but effective for thin dark hairs.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to get hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint to remove hair
    result = cv2.inpaint(image, hair_mask, inpaintRadius=6,
                         flags=cv2.INPAINT_TELEA)
    return result
```

#### Training Tricks
- **Warm restarts (SGDR)**: CosineAnnealingWarmRestarts scheduler finds multiple local minima for snapshot ensembling.
- **Snapshot ensembling**: Save model at each cosine cycle trough, ensemble all snapshots at inference.
- **Mixup / CutMix for segmentation**: Mix images AND their masks together.
- **Heavy augmentation**: Elastic transform, grid distortion, optical distortion are particularly helpful.

```python
def cutmix_segmentation(image1, mask1, image2, mask2, alpha=1.0):
    """CutMix augmentation for segmentation."""
    h, w = image1.shape[:2]
    lam = np.random.beta(alpha, alpha)

    # Random box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(h)
    cx = np.random.randint(w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    mixed_image = image1.copy()
    mixed_mask = mask1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

    return mixed_image, mixed_mask
```

#### Augmentation Tricks Specific to Dermoscopy
- **CoarseDropout**: Simulates hair artifacts and occlusions
- **HueSaturationValue**: Moderate (hue=10, sat=20, val=20) to simulate different skin tones
- **ElasticTransform**: Simulates non-rigid lesion deformation
- **CLAHE as augmentation**: Apply CLAHE with random parameters during training

```python
import albumentations as A

skin_lesion_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.2,
        rotate_limit=45, border_mode=cv2.BORDER_REFLECT, p=0.5
    ),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=6, p=1),
        A.GridDistortion(p=1),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1),
    ], p=0.3),
    A.OneOf([
        A.CLAHE(clip_limit=4, p=1),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=1
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=1),
    ], p=0.5),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=20,
        val_shift_limit=20, p=0.3
    ),
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32,
        fill_value=0, p=0.3
    ),
    A.GaussNoise(var_limit=(5, 25), p=0.2),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

#### Self-Training / Pseudo-Labels
- Train teacher model on labeled data
- Generate pseudo-labels on unlabeled images
- Train student model on both labeled and pseudo-labeled data
- Repeat with noise injection (dropout, augmentation) for robustness

---

## 9. Complete Inference Pipeline (Putting It All Together)

```python
import torch
import numpy as np
import cv2
from scipy import ndimage

def full_inference_pipeline(
    models,
    image_np,
    device,
    input_sizes=[384],
    threshold=None,
    use_tta=True,
    use_crf=False,
    use_postprocess=True,
    model_weights=None,
):
    """
    Complete inference pipeline combining all optimization techniques.

    Args:
        models: list of trained models
        image_np: original image as numpy array (H, W, 3), uint8
        device: torch device
        input_sizes: list of input sizes for multi-scale
        threshold: binarization threshold (None = use 0.5)
        use_tta: whether to use test-time augmentation
        use_crf: whether to apply CRF post-processing
        use_postprocess: whether to apply morphological post-processing
        model_weights: optional weights for ensemble (by val IoU)
    Returns:
        binary mask (H, W), values 0 or 255
    """
    h_orig, w_orig = image_np.shape[:2]
    all_predictions = []

    for model_idx, model in enumerate(models):
        model.eval()

        for size in input_sizes:
            # Resize and normalize
            resized = cv2.resize(image_np, (size, size))
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(
                normalized.transpose(2, 0, 1)
            ).unsqueeze(0).float()

            if use_tta:
                pred = tta_predict_logits(model, tensor, device)
            else:
                with torch.no_grad():
                    logit = model(tensor.to(device))
                    pred = torch.sigmoid(logit).cpu().squeeze().numpy()

            # Resize prediction back to original size
            pred_resized = cv2.resize(pred, (w_orig, h_orig))

            weight = 1.0
            if model_weights is not None:
                weight = model_weights[model_idx]

            all_predictions.append(pred_resized * weight)

    # Weighted average
    if model_weights is not None:
        total_weight = sum(model_weights) * len(input_sizes)
        avg_pred = sum(all_predictions) / total_weight
    else:
        avg_pred = np.mean(all_predictions, axis=0)

    # CRF refinement (before thresholding)
    if use_crf:
        avg_pred = apply_crf(image_np, avg_pred).astype(np.float32)

    # Threshold
    thresh = threshold if threshold is not None else 0.5
    binary = (avg_pred > thresh).astype(np.uint8)

    # Post-processing
    if use_postprocess:
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + np.argmax(areas)
            binary = (labels == largest).astype(np.uint8)

        # Morphological closing then opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary * 255
```

---

## 10. Priority Action Plan (What to Do First)

### Estimated IoU Gains (Cumulative)

| Technique | Expected Gain | Effort | Priority |
|-----------|--------------|--------|----------|
| Train at 384 instead of 256 | +2-4% | Low | 1 |
| Switch to BCE+Dice+Lovasz loss | +2-3% | Low | 2 |
| TTA (D4, 8x) | +1.5-2% | Low | 3 |
| Optimal threshold search | +1-2% | Low | 4 |
| 3-model ensemble (avg logits) | +1-3% | Medium | 5 |
| Post-processing pipeline | +1-2% | Low | 6 |
| Progressive resizing (256->384->512) | +1-2% | Medium | 7 |
| CRF post-processing | +0.5-1% | Medium | 8 |
| Color constancy preprocessing | +0.5-1% | Low | 9 |
| Hair removal preprocessing | +0.5-1% | Low | 10 |
| Multi-scale inference | +0.5-1% | Low | 11 |

### Realistic Path from 0.81 to 0.90+

1. **Quick wins (0.81 -> 0.86)**: Higher resolution (384), better loss function, TTA
2. **Ensemble gains (0.86 -> 0.89)**: 3-model ensemble with multi-scale, optimal threshold
3. **Polish (0.89 -> 0.91)**: Post-processing, CRF, preprocessing improvements

### Critical Notes
- These gains are NOT purely additive -- some overlap
- Diminishing returns as you stack techniques
- Always validate on your held-out val set
- The biggest single improvement is usually resolution + loss function change
- Ensemble + TTA together give the most reliable boost
