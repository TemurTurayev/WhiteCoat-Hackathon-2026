# Test-Time Adaptation (TTA) & Test-Time Training (TTT) for Medical Image Segmentation

## Context
- Model: U-Net++ with EfficientNetV2-S encoder, trained at 512px
- Current validation IoU: 0.8268
- Test images: 200 skin lesion images (128x128)
- Constraint: NO GPU, CPU-only inference, no retraining
- Goal: Improve IoU at inference time

---

## 1. TENT (Test-Time Entropy Minimization)

**Paper**: Wang et al., ICLR 2021 — "Tent: Fully Test-Time Adaptation by Entropy Minimization"
**Code**: https://github.com/DequanWang/tent

### How It Works
1. At test time, switch BatchNorm layers to `train()` mode (updates running stats)
2. Freeze all parameters except BN affine parameters (gamma, beta)
3. For each test batch, compute prediction entropy: `H(y) = -sum(p * log(p))`
4. Backpropagate entropy loss to update only BN gamma/beta
5. Make prediction with updated parameters

### Implementation (PyTorch)
```python
import torch
import torch.nn as nn

def setup_tent(model):
    """Configure model for TENT adaptation."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train()
            module.requires_grad_(True)
        else:
            for param in module.parameters():
                param.requires_grad_(False)
    return model

def tent_entropy_loss(logits):
    """Compute entropy of softmax predictions."""
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy

def adapt_and_predict(model, image, optimizer, n_steps=1):
    """Adapt model on single image, then predict."""
    for _ in range(n_steps):
        logits = model(image)
        loss = tent_entropy_loss(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(image)
    return prediction
```

### Limitations for Our Use Case
- **Single-image instability**: TENT expects batches; with 1 image, BN stats are unreliable
- **Requires backpropagation**: Slow on CPU (~3-5x slower than forward-only inference)
- **Binary segmentation**: Entropy minimization is less effective for binary masks (only 2 classes)
- **Risk of collapse**: Without enough diversity, model can converge to trivial solutions

### Feasibility: LOW-MEDIUM
- CPU cost is significant (backprop needed per image)
- Single-image BN instability is a known problem
- Binary segmentation limits entropy signal

---

## 2. InTEnt (Integrated Entropy Weighting)

**Paper**: Dong et al., CVPR 2024 Workshop — "Medical Image Segmentation with InTEnt"
**Code**: https://github.com/mazurowski-lab/single-image-test-time-adaptation

### How It Works (Designed for Single-Image TTA)
1. For a single test image, interpolate BN statistics between training stats and test-image stats
2. Generate multiple predictions using different interpolation weights alpha in [0, 1]
3. Weight each prediction by inverse entropy (more confident = higher weight)
4. Aggregate weighted predictions into final mask
5. Novel balanced entropy: equally weights foreground and background entropy contributions

### Key Advantage
- Specifically designed for **single-image** medical segmentation
- No backpropagation needed (just multiple forward passes with different BN stats)
- Surpasses leading methods by 2.9% Dice on average across 24 domain splits

### Implementation Sketch
```python
def intent_adapt(model, image, n_interpolations=10):
    """InTEnt: Interpolate BN stats, weight by entropy."""
    original_stats = save_bn_stats(model)
    test_stats = compute_test_bn_stats(model, image)

    predictions = []
    weights = []
    for alpha in torch.linspace(0, 1, n_interpolations):
        set_interpolated_bn_stats(model, original_stats, test_stats, alpha)
        with torch.no_grad():
            logits = model(image)
            probs = torch.sigmoid(logits)
            ent = -(probs * torch.log(probs + 1e-8) +
                    (1 - probs) * torch.log(1 - probs + 1e-8)).mean()
            predictions.append(probs)
            weights.append(1.0 / (ent.item() + 1e-8))

    weights = torch.tensor(weights)
    weights = weights / weights.sum()
    final = sum(w * p for w, p in zip(weights, predictions))
    restore_bn_stats(model, original_stats)
    return (final > 0.5).float()
```

### Feasibility: MEDIUM
- No backprop needed (just N forward passes)
- Designed for single-image medical segmentation
- CPU cost: ~10 forward passes per image (manageable for 200 images)
- Estimated time: ~10-20 min on CPU for 200 images

---

## 3. Test-Time Augmentation (TTA) — The Most Practical Option

### Standard TTA (Already Likely in Use)
Apply geometric transforms, run inference on each, reverse transforms, average predictions.

### Optimal TTA Strategy for Skin Lesions
```python
import torch
import torchvision.transforms.functional as TF

def tta_predict(model, image, threshold=0.5):
    """8x TTA: 4 rotations x 2 flips. Skin lesions have no canonical orientation."""
    predictions = []
    for angle in [0, 90, 180, 270]:
        for flip in [False, True]:
            aug = TF.rotate(image, angle)
            if flip:
                aug = TF.hflip(aug)
            with torch.no_grad():
                pred = torch.sigmoid(model(aug))
            if flip:
                pred = TF.hflip(pred)
            pred = TF.rotate(pred, -angle)
            predictions.append(pred)

    mean_pred = torch.stack(predictions).mean(dim=0)
    return (mean_pred > threshold).float()
```

### Advanced TTA Strategies

**a) Multi-Scale TTA**
```python
def multiscale_tta(model, image, scales=[0.75, 1.0, 1.25]):
    """Run inference at multiple scales, resize back, average."""
    h, w = image.shape[-2:]
    predictions = []
    for scale in scales:
        sh, sw = int(h * scale), int(w * scale)
        scaled = F.interpolate(image, (sh, sw), mode='bilinear', align_corners=False)
        with torch.no_grad():
            pred = torch.sigmoid(model(scaled))
        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

**b) Color/Intensity TTA**
```python
def color_tta(model, image):
    """Augment brightness/contrast at test time."""
    predictions = []
    for brightness in [0.9, 1.0, 1.1]:
        for contrast in [0.9, 1.0, 1.1]:
            aug = TF.adjust_brightness(image, brightness)
            aug = TF.adjust_contrast(aug, contrast)
            with torch.no_grad():
                pred = torch.sigmoid(model(aug))
            predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

**c) Selective TTA (S3-TTA)**
Recent work (2024) shows that not all augmentations help for every image. Select only the augmentations whose predictions are most consistent (lowest variance).

### Merge Strategy Comparison
| Strategy | Description | Best For |
|----------|------------|----------|
| Mean | Average all probabilities | General, smooth boundaries |
| Geometric Mean | exp(mean(log(p))) | Conservative, fewer false positives |
| Max | Take maximum probability | High recall needed |
| Majority Vote | Binarize then vote | Robust to outliers |
| Entropy-Weighted | Weight by prediction confidence | Unreliable augmentations |

**Recommendation**: Mean averaging with threshold tuning (try 0.45-0.55) gives best IoU for binary segmentation.

### Feasibility: HIGH
- No backpropagation, forward-only
- 8x TTA: ~8x inference time (still fast on CPU for 128x128 images)
- Expected improvement: +1-3% IoU
- Combining geometric + multiscale: ~24 forward passes, still feasible

---

## 4. Morphological Post-Processing (Zero-Cost, No Model Change)

### Operations That Improve Segmentation IoU

```python
import cv2
import numpy as np

def refine_mask(mask, kernel_size=3):
    """Apply morphological refinement to binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 1. Closing: fill small holes inside lesion
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. Opening: remove small noise outside lesion
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Fill remaining holes — keep only largest connected component
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(refined)
        cv2.drawContours(filled, [largest], -1, 255, -1)
        refined = filled

    return refined
```

### Feasibility: VERY HIGH
- Negligible compute cost
- No model changes
- Expected improvement: +0.5-1.5% IoU (depends on current mask quality)

---

## 5. CRF (Conditional Random Field) Post-Processing

**Library**: `pydensecrf` or `crfseg` (PyTorch)

### How It Works
1. Takes raw probability map + original RGB image
2. Uses spatial proximity and color similarity to refine boundaries
3. Pixels with similar colors nearby get similar labels

### Implementation
```python
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crf_refine(image_rgb, prob_map, n_iterations=5):
    """
    image_rgb: (H, W, 3) uint8
    prob_map: (H, W) float32 in [0,1] — probability of foreground
    """
    h, w = prob_map.shape
    probs = np.stack([1 - prob_map, prob_map], axis=0)  # (2, H, W)

    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Appearance kernel (color-dependent)
    d.addPairwiseBilateral(
        sxy=10,
        srgb=13,
        rgbim=image_rgb,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Smoothness kernel (spatial only)
    d.addPairwiseGaussian(sxy=3, compat=3)

    Q = d.inference(n_iterations)
    result = np.argmax(np.array(Q).reshape(2, h, w), axis=0)
    return (result * 255).astype(np.uint8)
```

### Feasibility: HIGH
- CPU-friendly (designed for CPU)
- ~0.1-0.5s per 128x128 image
- Expected improvement: +0.5-2% IoU (particularly for boundary refinement)
- Install: `pip install pydensecrf`

---

## 6. Threshold Optimization (Simplest, Often Overlooked)

### The Default 0.5 Threshold Is Rarely Optimal

```python
def find_optimal_threshold(model, val_loader):
    """Search for best binarization threshold on validation set."""
    best_iou = 0
    best_thresh = 0.5

    all_probs = []
    all_masks = []
    for images, masks in val_loader:
        with torch.no_grad():
            probs = torch.sigmoid(model(images))
        all_probs.append(probs)
        all_masks.append(masks)

    all_probs = torch.cat(all_probs)
    all_masks = torch.cat(all_masks)

    for thresh in np.arange(0.30, 0.70, 0.01):
        preds = (all_probs > thresh).float()
        intersection = (preds * all_masks).sum()
        union = preds.sum() + all_masks.sum() - intersection
        iou = (intersection / (union + 1e-8)).item()
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    return best_thresh, best_iou
```

### Feasibility: VERY HIGH
- Requires validation set (we have 400 images)
- Typical improvement: +0.5-2% IoU
- Cost: negligible

---

## 7. Med-TTT and TTT-UNet (Research-Stage, GPU Required)

**Papers**:
- Med-TTT (Oct 2024): Vision backbone with TTT layers for medical segmentation
- TTT-UNet: Hybrid architecture integrating TTT layers into U-Net

### How TTT Layers Work
- Replace standard self-attention with a self-supervised learning objective
- At test time, the model updates internal hidden states using a reconstruction loss on the test input
- Captures long-range dependencies with linear complexity

### Feasibility: LOW (for our case)
- Requires model architecture changes (TTT layers baked in)
- Would need retraining from scratch
- GPU-intensive during inference (hidden state updates)
- Not applicable to our existing U-Net++ model

---

## 8. MedSeg-TTA Benchmark Insights (Dec 2025)

The MedSeg-TTA benchmark compared 20 TTA methods across 7 modalities including dermoscopy.

### Key Findings Relevant to Us
1. **No single method dominates** across all modalities
2. **Output-level regularization** (entropy-based) is robust under heavy noise
3. **Input-level methods** (style normalization) are more stable under mild shifts
4. **Feature-level methods** offer best boundary refinement (relevant for IoU)
5. Methods can **degrade** under large distribution shifts — always validate on held-out data

### Four TTA Paradigms
| Paradigm | Examples | Our Applicability |
|----------|---------|-------------------|
| Input-Level | Style transfer, histogram matching | Medium — images are consistent |
| Feature-Level | BN adaptation, feature alignment | Medium — requires backprop |
| Output-Level | Entropy minimization, pseudo-labels | Medium — binary limits signal |
| Prior Estimation | Shape priors, atlas-based | Low — needs shape atlas |

---

## Recommended Strategy (Priority Order)

Given our constraints (CPU-only, 200 test images at 128x128, existing U-Net++):

### Tier 1: Immediate, High Confidence (implement first)
1. **Threshold Optimization** — Search [0.30, 0.70] on validation set. Expected: +0.5-2% IoU
2. **8x Geometric TTA** — Rotations (0/90/180/270) x flips (H/none), mean merge. Expected: +1-3% IoU
3. **Morphological Post-Processing** — Close holes, remove noise, keep largest component. Expected: +0.5-1.5% IoU

### Tier 2: Medium Effort, Good Potential
4. **CRF Post-Processing** — Refine boundaries using original image colors. Expected: +0.5-2% IoU
5. **Multi-Scale TTA** — Add scales [0.75, 1.0, 1.25] to geometric TTA. Expected: +0.5-1% IoU
6. **Color/Intensity TTA** — Brightness/contrast variations. Expected: +0.3-0.5% IoU

### Tier 3: Higher Effort, Research-Stage
7. **InTEnt** — BN stat interpolation with entropy weighting (10 forward passes per image). Expected: +1-3% IoU
8. **TENT** — Entropy minimization with BN updates (requires backprop). Expected: +0-2% IoU (risky)

### Combined Pipeline
```
Input Image
  -> Multi-Scale TTA (3 scales)
    -> 8x Geometric TTA per scale (24 total forward passes)
      -> Mean aggregation of all 24 predictions
        -> Optimal threshold (tuned on validation)
          -> CRF refinement (optional, if boundary errors dominate)
            -> Morphological cleanup (close holes, remove noise, largest component)
              -> Final binary mask
```

### Estimated Total Improvement: +2-5% IoU (0.8268 -> ~0.85-0.87)

### Time Budget (CPU, 200 images)
| Step | Per Image | Total (200 imgs) |
|------|-----------|-------------------|
| Threshold tuning | - | One-time on val set |
| 8x TTA | ~0.5s | ~100s |
| Multi-scale (3x) | ~0.15s | ~30s |
| CRF | ~0.3s | ~60s |
| Morphology | ~0.01s | ~2s |
| **Combined pipeline** | **~2-3s** | **~8-10 min** |

---

## Key References

1. Wang et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization" (ICLR 2021)
2. Dong et al. "Medical Image Segmentation with InTEnt" (CVPR 2024 Workshop)
3. MedSeg-TTA Benchmark (Dec 2025) — 20 methods, 7 modalities
4. S3-TTA: Scale-Style Selection for Test-Time Augmentation (2024)
5. SicTTA: Single Image Continual TTA (Medical Image Analysis, 2025)
6. Med-TTT: Vision Test-Time Training for Medical Segmentation (Oct 2024)
7. TTT-UNet: Enhancing U-Net with Test-Time Training (2024)
8. TEGDA: Test-Time Evaluation-Guided Dynamic Adaptation (MICCAI 2025)
9. Progressive Test Time Energy Adaptation (ICCV 2025)
10. GraTa: Gradient Alignment for TTA (AAAI 2025)
