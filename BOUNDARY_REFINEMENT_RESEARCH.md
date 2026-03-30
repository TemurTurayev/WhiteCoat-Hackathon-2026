# Boundary Refinement Research for Skin Lesion Segmentation

**Current IoU**: 0.8952 | **Target IoU**: 0.95+
**Current setup**: U-Net++ EfficientNetV2-S, 512px, Dice+BCE+Lovasz, 4x TTA, morphological post-processing

---

## Priority Summary (Ranked by Expected Impact / Effort)

| # | Technique | Expected IoU Gain | Impl. Time | Complexity | Requires Retraining |
|---|-----------|-------------------|------------|------------|---------------------|
| 1 | Dense CRF Post-Processing | +0.01-0.03 | 1-2 hours | Low | No |
| 2 | Multi-Scale TTA (8x full) | +0.005-0.015 | 30 min | Low | No |
| 3 | Boundary Loss (training) | +0.01-0.02 | 2-3 hours | Medium | Yes |
| 4 | Morphological Snake Refinement | +0.005-0.015 | 1-2 hours | Low | No |
| 5 | SegFix-style Boundary Replacement | +0.01-0.03 | 4-6 hours | High | Yes (small net) |
| 6 | Hausdorff Distance Loss | +0.01-0.02 | 2-3 hours | Medium | Yes |
| 7 | Multi-Scale Boundary Refinement | +0.01-0.02 | 3-4 hours | High | Yes |
| 8 | Edge-Aware Networks (BPR) | +0.02-0.04 | 6+ hours | Very High | Yes |

**Recommendation**: Do #1, #2, and #4 first (no retraining, 3-4 hours total). If still below target, add #3/#6 with retraining.

---

## 1. Dense CRF (Conditional Random Fields)

### What It Does
Fully-connected CRF refines segmentation by enforcing spatial consistency: pixels with similar color AND proximity tend to share labels. It sharpens boundaries where the model is uncertain by leveraging the original RGB image as guidance.

### Installation
```bash
pip install pydensecrf
# If build fails on newer Python:
pip install pydensecrf2
```

### Implementation for Skin Lesion Binary Segmentation

```python
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_dense_crf(image_rgb, prob_map, params=None):
    """
    Refine a probability map using Dense CRF.

    Args:
        image_rgb: Original RGB image, shape (H, W, 3), uint8
        prob_map: Probability map from model, shape (H, W), float32 [0,1]
        params: Dict of CRF parameters

    Returns:
        Refined binary mask, shape (H, W), uint8 {0, 255}
    """
    if params is None:
        params = {
            # Gaussian (smoothness) kernel
            'gauss_sxy': 3,        # spatial smoothness (lower = more local)
            'gauss_compat': 3,     # compatibility (penalty for label disagreement)
            # Bilateral (appearance) kernel
            'bi_sxy': 40,          # spatial range for bilateral
            'bi_srgb': 5,          # color range (CRITICAL for skin lesions)
            'bi_compat': 10,       # compatibility
            # Inference
            'n_iters': 10,         # CRF inference iterations
        }

    h, w = prob_map.shape

    # Create 2-class probability: [background, foreground]
    # Clip to avoid log(0)
    prob_fg = np.clip(prob_map, 1e-6, 1 - 1e-6)
    prob_bg = 1.0 - prob_fg
    probs = np.stack([prob_bg, prob_fg], axis=0)  # (2, H, W)

    # Create unary potentials from softmax probabilities
    unary = unary_from_softmax(probs)  # (2, H*W)

    # Create CRF object
    d = dcrf.DenseCRF2D(w, h, 2)  # width, height, n_labels
    d.setUnaryEnergy(unary)

    # Gaussian pairwise: encourages spatial smoothness
    d.addPairwiseGaussian(
        sxy=params['gauss_sxy'],
        compat=params['gauss_compat'],
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Bilateral pairwise: encourages color-consistent boundaries
    # This is the KEY kernel for skin lesion boundary refinement
    d.addPairwiseBilateral(
        sxy=params['bi_sxy'],
        srgb=params['bi_srgb'],
        rgbim=image_rgb.copy(order='C'),  # Must be C-contiguous
        compat=params['bi_compat'],
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Inference
    Q = d.inference(params['n_iters'])
    result = np.argmax(Q, axis=0).reshape(h, w)

    return (result * 255).astype(np.uint8)
```

### Optimal Parameters for Skin Lesions

Skin lesion images have specific characteristics: lesion interior is typically darker/more pigmented, clear color transition at boundary, variable hair artifacts.

```python
# Conservative (less change, safer)
PARAMS_CONSERVATIVE = {
    'gauss_sxy': 3, 'gauss_compat': 3,
    'bi_sxy': 50, 'bi_srgb': 5, 'bi_compat': 5,
    'n_iters': 5,
}

# Moderate (good default for skin lesions)
PARAMS_MODERATE = {
    'gauss_sxy': 3, 'gauss_compat': 3,
    'bi_sxy': 40, 'bi_srgb': 8, 'bi_compat': 10,
    'n_iters': 10,
}

# Aggressive (stronger boundary snapping, risk of erosion)
PARAMS_AGGRESSIVE = {
    'gauss_sxy': 5, 'gauss_compat': 5,
    'bi_sxy': 80, 'bi_srgb': 13, 'bi_compat': 15,
    'n_iters': 15,
}
```

### Parameter Tuning Notes
- **bi_srgb** is the most critical parameter for skin lesions. Lower values (3-5) respect color boundaries more strictly; higher values (10-15) allow smoother transitions but may erode thin structures.
- **bi_sxy** controls spatial reach. For 512px images, 30-80 is typical. For original resolution images, scale proportionally.
- **n_iters**: 5-10 is usually sufficient. More iterations rarely help and cost time.
- **IMPORTANT**: CRF must run on the ORIGINAL resolution image, not the 512px resized version. Resize the probability map to original resolution first, then apply CRF with the original RGB image.

### Integration with Current Inference Pipeline

```python
# In segment.py, after computing avg probability map:
prob = cv2.resize(avg, (ow, oh), interpolation=cv2.INTER_LINEAR)

# Option A: Apply CRF before thresholding
img_for_crf = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
mask_crf = apply_dense_crf(img_for_crf, prob, PARAMS_MODERATE)

# Option B: Apply CRF then existing post-processing
mask = postprocess(prob, args.threshold)  # existing pipeline
# OR combine: use CRF output as refined probability, then postprocess
```

### Expected Impact
- IoU gain: +0.01 to +0.03 (depends on how well boundaries are already predicted)
- Speed: ~0.2-0.5 sec per image at 512px; ~1-3 sec at original resolution
- Risk: Low. CRF rarely hurts if parameters are conservative.

---

## 2. Enhanced Test-Time Augmentation (TTA)

### Current State
The inference pipeline uses 4 TTA transforms (original, hflip, vflip, hvflip).

### Upgrade to 8x TTA (all D4 symmetries)
Skin lesions have no canonical orientation, so all 8 dihedral group transformations are valid:

```python
def get_full_tta_transforms(image_size):
    """All 8 D4 symmetry group transformations."""
    base = [A.Resize(image_size, image_size), A.Normalize(MEAN, STD), ToTensorV2()]

    tta_list = [
        ("orig",   A.Compose(base), lambda p: p),
        ("hf",     A.Compose([A.Resize(image_size,image_size), A.HorizontalFlip(p=1.0)] + base[1:]),
                   lambda p: np.flip(p, axis=1).copy()),
        ("vf",     A.Compose([A.Resize(image_size,image_size), A.VerticalFlip(p=1.0)] + base[1:]),
                   lambda p: np.flip(p, axis=0).copy()),
        ("hvf",    A.Compose([A.Resize(image_size,image_size), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base[1:]),
                   lambda p: np.flip(np.flip(p, axis=0), axis=1).copy()),
        ("r90",    A.Compose([A.Resize(image_size,image_size), A.Rotate(limit=(90,90), p=1.0, border_mode=0)] + base[1:]),
                   lambda p: np.rot90(p, k=-1).copy()),
        ("r180",   A.Compose([A.Resize(image_size,image_size), A.Rotate(limit=(180,180), p=1.0, border_mode=0)] + base[1:]),
                   lambda p: np.rot90(p, k=-2).copy()),
        ("r270",   A.Compose([A.Resize(image_size,image_size), A.Rotate(limit=(270,270), p=1.0, border_mode=0)] + base[1:]),
                   lambda p: np.rot90(p, k=-3).copy()),
        ("r90hf",  A.Compose([A.Resize(image_size,image_size), A.Rotate(limit=(90,90), p=1.0, border_mode=0), A.HorizontalFlip(p=1.0)] + base[1:]),
                   lambda p: np.flip(np.rot90(p, k=-1), axis=1).copy()),
    ]
    return tta_list
```

### Multi-Scale TTA
Run inference at multiple resolutions and average:

```python
SCALES = [384, 512, 640, 768]

def multi_scale_tta(model, img, device):
    """Multi-scale inference with TTA at each scale."""
    h, w = img.shape[:2]
    combined = np.zeros((h, w), dtype=np.float64)
    total_weight = 0

    # Weight higher resolutions more (they capture finer boundaries)
    scale_weights = {384: 0.5, 512: 1.0, 640: 1.2, 768: 1.5}

    for size in SCALES:
        weight = scale_weights[size]
        # Run standard TTA at this scale
        prob_at_scale = run_tta_at_scale(model, img, size, device)  # returns (size, size)
        # Resize to original resolution
        prob_orig = cv2.resize(prob_at_scale, (w, h), interpolation=cv2.INTER_LINEAR)
        combined += prob_orig * weight
        total_weight += weight

    return (combined / total_weight).astype(np.float32)
```

### Expected Impact
- 8x TTA vs 4x: +0.003-0.008 IoU
- Multi-scale TTA: +0.005-0.015 IoU (especially for boundary precision)
- Speed cost: 2x (8x TTA) or 4-8x (multi-scale), but only 200 test images

---

## 3. Boundary Loss (Training-Time)

### Concept
Standard Dice+BCE losses focus on region overlap. Boundary Loss directly penalizes distance between predicted and ground-truth boundaries, measured as surface distance.

### Implementation (from Kervadec et al. MIDL 2019)

```python
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

def compute_distance_map(mask_np):
    """Precompute signed distance map for a binary mask."""
    # mask_np: (H, W) binary numpy array
    pos_dist = distance_transform_edt(mask_np)
    neg_dist = distance_transform_edt(1 - mask_np)
    # Signed distance: negative inside, positive outside
    dist_map = neg_dist - pos_dist
    # Normalize to [-1, 1] range
    max_val = max(pos_dist.max(), neg_dist.max(), 1)
    dist_map = dist_map / max_val
    return dist_map

class BoundaryLoss(nn.Module):
    """
    Boundary Loss: penalizes predictions based on distance to GT boundary.
    Must precompute distance maps in the dataset __getitem__.
    """
    def forward(self, pred_logits, dist_maps):
        """
        pred_logits: (B, 1, H, W) - raw logits from model
        dist_maps: (B, 1, H, W) - precomputed signed distance maps
        """
        pred_probs = torch.sigmoid(pred_logits)
        # Boundary loss = mean of (probability * distance)
        # Minimizing this pushes probability toward 0 where distance > 0 (outside GT)
        # and toward 1 where distance < 0 (inside GT)
        loss = (pred_probs * dist_maps).mean()
        return loss
```

### Training Strategy (Critical: Scheduling)

Boundary Loss should NOT be used alone from the start. It needs warm-up with region-based loss:

```python
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryLoss()

    def forward(self, pred, target, dist_map, alpha):
        """
        alpha: boundary loss weight, scheduled from 0 -> 0.01 over training
        Start: alpha=0 (pure Dice+BCE)
        End: alpha=0.01 (slight boundary emphasis)
        """
        region_loss = 0.5 * self.dice(pred, target) + 0.5 * self.bce(pred, target)
        bd_loss = self.boundary(pred, dist_map)
        return (1 - alpha) * region_loss + alpha * bd_loss

# Schedule alpha: linear warmup
def get_boundary_alpha(epoch, total_epochs, max_alpha=0.01):
    """Start boundary loss after 30% of training, ramp up linearly."""
    warmup_end = int(total_epochs * 0.3)
    if epoch < warmup_end:
        return 0.0
    return max_alpha * (epoch - warmup_end) / (total_epochs - warmup_end)
```

### Dataset Modification
Distance maps must be precomputed:

```python
class SegDatasetWithBoundary(Dataset):
    def __getitem__(self, i):
        # ... load image and mask as before ...
        m_binary = (m > 127).astype(np.float32)
        dist_map = compute_distance_map(m_binary)

        if self.tf:
            r = self.tf(image=im, masks=[m_binary, dist_map])
            im = r['image']
            m_binary = r['masks'][0]
            dist_map = r['masks'][1]

        return im, m_binary.unsqueeze(0), torch.from_numpy(dist_map).unsqueeze(0).float()
```

### Expected Impact
- IoU gain: +0.01-0.02 (specifically improves boundary alignment)
- Requires full retraining (or fine-tuning last 10-15 epochs)
- Risk: Medium. Over-weighting boundary loss can destabilize training.

---

## 4. Morphological Snake / Active Contour Refinement

### Concept
Use the model's prediction as initialization, then evolve the contour to snap to image edges using morphological operations (faster and more stable than PDE-based active contours).

### Implementation with scikit-image

```python
from skimage.segmentation import morphological_geodesic_active_contour as MorphGAC
from skimage.segmentation import morphological_chan_vese as MorphACWE
from skimage.segmentation import inverse_gaussian_gradient
import numpy as np

def refine_with_morphgac(image_rgb, initial_mask, iterations=100, smoothing=1, threshold='auto', balloon=-1):
    """
    Morphological Geodesic Active Contour refinement.
    Best when lesion boundaries are visible (good edge contrast).

    Args:
        image_rgb: (H, W, 3) uint8
        initial_mask: (H, W) binary {0, 1}
        iterations: number of evolution steps
        smoothing: smoothing parameter (1-4 typical)
        balloon: balloon force. Negative = contraction, positive = expansion.
                 Use -1 for skin lesions (slight contraction tendency)
    """
    # Convert to grayscale for edge detection
    gray = np.mean(image_rgb.astype(np.float64) / 255.0, axis=2)

    # Compute inverse gradient image (edge indicator)
    gimage = inverse_gaussian_gradient(gray, alpha=100, sigma=5.0)

    # Run MorphGAC with model prediction as initialization
    refined = MorphGAC(
        gimage,
        iterations=iterations,
        init_level_set=initial_mask.astype(np.float64),
        smoothing=smoothing,
        threshold=threshold,
        balloon=balloon
    )

    return refined.astype(np.uint8)


def refine_with_morphacwe(image_rgb, initial_mask, iterations=50, smoothing=2):
    """
    Morphological Chan-Vese (Active Contours without Edges).
    Better for low-contrast boundaries where gradient-based methods fail.
    Works on region intensity differences rather than edges.
    """
    gray = np.mean(image_rgb.astype(np.float64) / 255.0, axis=2)

    refined = MorphACWE(
        gray,
        iterations=iterations,
        init_level_set=initial_mask.astype(np.float64),
        smoothing=smoothing,
        lambda1=1,  # weight for inside region
        lambda2=1,  # weight for outside region
    )

    return refined.astype(np.uint8)
```

### Parameters for Skin Lesions

```python
# For dermoscopy images (typically good contrast):
MORPHGAC_PARAMS = {
    'iterations': 80,
    'smoothing': 1,
    'threshold': 'auto',
    'balloon': -1,     # slight contraction (lesion masks tend to be slightly over-segmented)
    'alpha': 100,      # inverse_gaussian_gradient alpha
    'sigma': 5.0,      # inverse_gaussian_gradient sigma
}

# For clinical photos (often low contrast):
MORPHACWE_PARAMS = {
    'iterations': 50,
    'smoothing': 2,
    'lambda1': 1,
    'lambda2': 1,
}
```

### Integration Strategy
```python
def refine_mask(image_rgb, prob_map, threshold=0.4):
    """Pipeline: threshold -> morphological cleanup -> active contour refinement"""
    # Step 1: Initial binary mask from model
    binary = (prob_map > threshold).astype(np.uint8)

    # Step 2: Basic cleanup (existing postprocess)
    binary = postprocess_basic(binary)  # fill holes, keep largest, morph open/close

    # Step 3: Active contour refinement (boundary snapping)
    refined = refine_with_morphgac(image_rgb, binary, **MORPHGAC_PARAMS)

    # Step 4: Final cleanup
    refined = postprocess_basic(refined)

    return refined * 255
```

### Expected Impact
- IoU gain: +0.005-0.015
- Speed: 0.5-2 sec per image depending on resolution and iterations
- Risk: Low, but must tune carefully. Too many iterations can shrink small lesions.

---

## 5. SegFix-Style Boundary Replacement (Post-Processing)

### Concept
Empirically, interior pixel predictions are more reliable than boundary pixels. SegFix replaces boundary pixel labels with their nearest interior pixel's label by learning a direction map pointing from each boundary pixel to a reliable interior pixel.

### Simplified Version (No Retraining Required)

```python
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def segfix_simple(prob_map, binary_mask, boundary_width=5):
    """
    Simplified SegFix: replace uncertain boundary predictions with
    nearest confident interior prediction.

    prob_map: (H, W) float probability
    binary_mask: (H, W) uint8 {0, 1}
    boundary_width: pixels from boundary to refine
    """
    # Find boundary region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary_mask, kernel, iterations=boundary_width)
    dilated = cv2.dilate(binary_mask, kernel, iterations=boundary_width)
    boundary_region = dilated - eroded  # 1 in boundary band

    # Identify confident interior pixels (high probability, far from boundary)
    confident_fg = (prob_map > 0.8) & (eroded == 1)
    confident_bg = (prob_map < 0.2) & (dilated == 0)

    # For boundary pixels, find nearest confident pixel's label
    result = binary_mask.copy()

    # Distance transform from confident foreground
    dist_fg = distance_transform_edt(~confident_fg)
    # Distance transform from confident background
    dist_bg = distance_transform_edt(~confident_bg)

    # In boundary region, assign label of nearest confident pixel
    boundary_mask = boundary_region.astype(bool)
    result[boundary_mask] = (dist_fg[boundary_mask] < dist_bg[boundary_mask]).astype(np.uint8)

    return result
```

### Expected Impact
- IoU gain: +0.005-0.015 (specifically at boundaries)
- No retraining needed for simplified version
- Speed: fast (~50ms per image)
- Risk: Very low

---

## 6. Hausdorff Distance Loss (Training-Time)

### Concept
Directly minimizes the Hausdorff distance between predicted and GT boundaries during training. Uses a differentiable approximation based on morphological erosion.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HausdorffDTLoss(nn.Module):
    """
    Hausdorff Distance loss based on distance transforms.
    From: https://github.com/JunMa11/SegLossOdyssey
    """
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img):
        """Compute distance transform for a batch of binary images."""
        field = torch.zeros_like(img)
        for batch in range(img.shape[0]):
            fg = img[batch] > 0.5
            if fg.any():
                bg = ~fg
                fg_dist = self._edt(fg.float())
                bg_dist = self._edt(bg.float())
                field[batch] = fg_dist + bg_dist
        return field

    def _edt(self, binary):
        """Approximate EDT using iterative erosion."""
        from scipy.ndimage import distance_transform_edt
        np_bin = binary.cpu().numpy().squeeze()
        dt = distance_transform_edt(np_bin)
        return torch.from_numpy(dt).to(binary.device).unsqueeze(0)

    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) logits
        target: (B, 1, H, W) binary
        """
        pred_prob = torch.sigmoid(pred)
        pred_dt = self.distance_field(pred_prob)
        target_dt = self.distance_field(target)

        pred_error = (pred_prob - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        loss = (pred_error * distance).mean()
        return loss
```

### Combined Loss with HD

```python
class DiceBCEHDLoss(nn.Module):
    def __init__(self, hd_weight=0.5):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()
        self.hd = HausdorffDTLoss(alpha=2.0)
        self.hd_weight = hd_weight

    def forward(self, pred, target):
        region = 0.5 * self.dice(pred, target) + 0.5 * self.bce(pred, target)
        hd = self.hd(pred, target)
        return region + self.hd_weight * hd
```

### Expected Impact
- IoU gain: +0.01-0.02
- Requires retraining (can fine-tune last 10-15 epochs)
- Speed overhead during training only
- Risk: Medium (HD loss can be noisy; use with region loss)

---

## 7. Multi-Scale Boundary Refinement (Architecture-Level)

### Concept
Add a boundary detection auxiliary head to the segmentation network. The boundary head produces an edge map that guides the main decoder to produce sharper boundaries.

### Implementation Sketch

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class BoundaryAwareUnetPP(nn.Module):
    def __init__(self, encoder_name='tu-tf_efficientnetv2_s'):
        super().__init__()
        self.base = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            classes=1,
            activation=None
        )
        # Add boundary head using the same encoder features
        self.boundary_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),  # adapt input channels
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        seg_out = self.base(x)
        # Extract edge from segmentation for auxiliary supervision
        boundary_out = self.boundary_head(self.base.decoder.blocks[-1].conv1[0].weight)
        return seg_out, boundary_out
```

**Note**: This requires significant architecture changes and retraining. Consider only if other methods are insufficient.

### Expected Impact
- IoU gain: +0.01-0.02
- Requires full retraining with boundary GT (computed from mask edges)
- Complexity: High
- Risk: Medium-High (architecture changes can destabilize)

---

## 8. Edge-Aware Networks: SegFix and BPR

### SegFix (ECCV 2020)
- Model-agnostic post-processing
- Learns a boundary-to-interior direction map
- Replaces boundary predictions with interior predictions
- Code: https://github.com/openseg-group/openseg.pytorch
- Reported: consistent +1-2% mIoU improvement on Cityscapes

### BPR - Boundary Patch Refinement (CVPR 2021)
- Extracts small patches along predicted boundaries
- Refines each patch at higher resolution with a small refinement network
- Code: https://github.com/chenhang98/BPR
- Reported: significant improvement on boundary-aware metrics

### Applicability to Our Case
Both methods were designed for multi-class semantic/instance segmentation on natural images. Adapting them to binary skin lesion segmentation would require:
- Training a boundary direction network (SegFix)
- Training a patch refinement network (BPR)
- Generating boundary GT labels

**Verdict**: Too complex for hackathon timeline unless pre-trained models can be adapted. Use the simplified SegFix approach from Section 5 instead.

---

## Recommended Implementation Order (Hackathon Timeline)

### Phase 1: No Retraining (2-3 hours)

1. **Upgrade to 8x TTA** (30 min)
   - Add rotation-based TTA transforms
   - Expected: +0.003-0.008

2. **Multi-Scale TTA** (30 min)
   - Run at 384, 512, 640 with weighted averaging
   - Expected: +0.005-0.010

3. **Dense CRF** (1-2 hours)
   - Implement CRF post-processing on original resolution
   - Grid search parameters on validation set
   - Expected: +0.01-0.03

4. **Morphological Snake Refinement** (1 hour)
   - Add MorphGAC as optional post-processing step
   - Expected: +0.005-0.015

5. **Simple SegFix** (30 min)
   - Distance-transform based boundary replacement
   - Expected: +0.005-0.010

**Total expected gain from Phase 1: +0.02-0.05 IoU (0.8952 -> ~0.92-0.94)**

### Phase 2: With Retraining (3-4 hours, if needed)

6. **Add Boundary Loss to existing training**
   - Fine-tune best model for 10-15 epochs with Dice+BCE+BoundaryLoss
   - Expected: +0.01-0.02

7. **Threshold Optimization**
   - Grid search threshold on validation set (0.30-0.55 range)
   - Expected: +0.005-0.010

### Validation Strategy
- Always validate on the 400-image validation set
- Compare per-image IoU distributions (not just mean)
- Watch for regression on small lesions (boundary refinement can shrink them)
- Keep backup of current best predictions

---

## Key References

- Dense CRF: Krahenbuhl & Koltun, "Efficient Inference in Fully Connected CRFs" (NeurIPS 2011)
- pydensecrf: https://github.com/lucasb-eyer/pydensecrf
- Boundary Loss: Kervadec et al., "Boundary loss for highly unbalanced segmentation" (MIDL 2019)
- SegLossOdyssey: https://github.com/JunMa11/SegLossOdyssey
- SegFix: Yuan et al., "SegFix: Model-Agnostic Boundary Refinement" (ECCV 2020)
- BPR: Tang et al., "Look Closer to Segment Better" (CVPR 2021)
- Morphological Snakes: scikit-image `morphological_geodesic_active_contour`
- Hausdorff DT Loss: Karimi & Salcudean, "Reducing the Hausdorff Distance in Medical Image Segmentation" (2019)
