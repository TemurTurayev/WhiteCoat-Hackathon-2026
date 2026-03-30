# Ensemble & Model Fusion Strategies for Segmentation IoU 0.82 -> 0.90+

## Current Model Zoo

| Model | Architecture | Resolution | IoU |
|-------|-------------|-----------|-----|
| Lovasz MEGA 512 | U-Net++ EfficientNetV2-S | 512px | 0.8268 |
| ISIC pretrained fold-1..5 | U-Net++ EfficientNetV2-S | 384px | ~0.808 |
| U-Net++ B4 | U-Net++ EfficientNet-B4 | 256px | ~0.81 |
| DeepLabV3+ R50 | DeepLabV3+ ResNet50 | 256px | ~0.80 |
| U-Net R34 | U-Net ResNet34 | 256px | ~0.80 |

---

## Strategy 1: Pixel-Level Probability Averaging (Soft Voting)

**The single most important technique.** Average probability maps (logits or sigmoid outputs) across models, then threshold.

### Implementation

```python
import torch
import numpy as np

def ensemble_predict(models, image, device, tta=True):
    """Average probability maps from multiple models."""
    all_probs = []

    for model_info in models:
        model = model_info['model']
        weight = model_info['weight']
        resize = model_info['resize']  # model's native resolution

        # Resize image to model's expected input
        img_resized = F.interpolate(image, size=resize, mode='bilinear', align_corners=False)

        with torch.no_grad():
            logit = model(img_resized)

        # Resize prediction back to original resolution
        logit = F.interpolate(logit, size=image.shape[2:], mode='bilinear', align_corners=False)
        prob = torch.sigmoid(logit)

        all_probs.append(prob * weight)

    # Weighted average
    avg_prob = torch.stack(all_probs).sum(dim=0) / sum(m['weight'] for m in models)
    return avg_prob
```

### Why It Works
- Each model makes different errors at different pixel locations
- Averaging reduces variance and smooths out individual model noise
- Probability-level averaging preserves uncertainty information better than hard voting
- **Expected IoU gain: +2-5% over best single model**

### Key Decision: Average Logits vs Probabilities
- **Logits (pre-sigmoid)**: Slightly better in practice, preserves more information about confidence
- **Probabilities (post-sigmoid)**: Easier to interpret, works well when models have different calibration
- **Recommendation**: Try both, validate on held-out set. For our case with different architectures, probabilities are safer.

---

## Strategy 2: Optimal Ensemble Weight Search

Not all models contribute equally. Find optimal per-model weights using validation set.

### Method A: Grid Search (Simple)

```python
import itertools
import numpy as np

def grid_search_weights(models_preds, gt_masks, step=0.1):
    """Brute-force search for optimal ensemble weights."""
    n_models = len(models_preds)
    best_iou = 0
    best_weights = None

    # Generate weight combinations that sum to 1
    weight_range = np.arange(0, 1 + step, step)

    for weights in itertools.product(weight_range, repeat=n_models):
        if abs(sum(weights) - 1.0) > 0.01:
            continue
        if max(weights) < 0.1:  # At least one model must contribute
            continue

        # Compute weighted average
        ensemble_prob = sum(w * p for w, p in zip(weights, models_preds))
        ensemble_mask = (ensemble_prob > 0.5).float()

        iou = compute_mean_iou(ensemble_mask, gt_masks)
        if iou > best_iou:
            best_iou = iou
            best_weights = weights

    return best_weights, best_iou
```

### Method B: Greedy Soup (from Model Soups paper)

```python
def greedy_ensemble(models_preds, gt_masks):
    """
    Greedy soup: start with best model, add others only if they improve IoU.
    From 'Model Soups' (Wortsman et al., ICML 2022).
    """
    # Sort by individual IoU (best first)
    individual_ious = [(i, compute_mean_iou(p > 0.5, gt_masks)) for i, p in enumerate(models_preds)]
    individual_ious.sort(key=lambda x: x[1], reverse=True)

    # Start with best model
    best_idx = individual_ious[0][0]
    soup = [best_idx]
    current_avg = models_preds[best_idx]
    current_iou = individual_ious[0][1]

    # Try adding each remaining model
    for idx, _ in individual_ious[1:]:
        candidate_avg = (current_avg * len(soup) + models_preds[idx]) / (len(soup) + 1)
        candidate_mask = (candidate_avg > 0.5).float()
        candidate_iou = compute_mean_iou(candidate_mask, gt_masks)

        if candidate_iou > current_iou:
            soup.append(idx)
            current_avg = candidate_avg
            current_iou = candidate_iou
            print(f"  Added model {idx}, IoU now: {current_iou:.4f}")
        else:
            print(f"  Skipped model {idx} (would decrease IoU)")

    return soup, current_iou
```

### Recommended Initial Weights (Based on Model Quality)

```python
weights = {
    'lovasz_mega_512': 0.35,      # Best single model
    'isic_fold_ensemble': 0.25,   # 5-fold mean (diversity)
    'unetpp_b4_256': 0.20,        # Different encoder
    'deeplabv3_r50_256': 0.10,    # Different architecture
    'unet_r34_256': 0.10,         # Different architecture
}
```

Then optimize these weights on the 400-image validation set.

---

## Strategy 3: Multi-Resolution Ensembling

Different resolutions capture different information. This is especially valuable for skin lesion segmentation where lesions vary in size.

### Rationale
- **512px models**: Better fine-grained boundary detail, better for small features
- **384px models**: Good balance of context and detail
- **256px models**: More global context, better for large lesions, faster

### Implementation

```python
def multi_resolution_ensemble(model_configs, image_full_res):
    """
    Each model processes at its native resolution.
    All predictions are resized to full resolution before averaging.
    """
    H, W = image_full_res.shape[2:]
    all_probs = []

    for cfg in model_configs:
        # Resize to model's native resolution
        img = F.interpolate(image_full_res, size=(cfg['res'], cfg['res']),
                           mode='bilinear', align_corners=False)

        with torch.no_grad():
            logit = cfg['model'](img)

        # Resize prediction back to FULL resolution
        prob = torch.sigmoid(
            F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)
        )
        all_probs.append(prob * cfg['weight'])

    return sum(all_probs) / sum(cfg['weight'] for cfg in model_configs)
```

### Key Insight
- Models at 256px and 512px make fundamentally different errors
- 256px models tend to over-segment (smooth boundaries)
- 512px models tend to capture fine detail but may miss global context
- Combining them is more valuable than combining same-resolution models

---

## Strategy 4: Test-Time Augmentation (TTA) with D4 Symmetry

Skin lesions have NO canonical orientation -- TTA is extremely effective here.

### D4 Group (8 Transformations)

```python
import ttach  # pip install ttach

# D4 = 4 rotations x 2 flips = 8 augmentations
transforms = ttach.Compose([
    ttach.HorizontalFlip(),
    ttach.VerticalFlip(),
    ttach.Rotate90(angles=[0, 90, 180, 270]),
])

# Usage with ttach library
tta_model = ttach.SegmentationTTAWrapper(
    model, transforms, merge_mode='mean'
)
```

### Manual Implementation (More Control)

```python
def tta_predict(model, image, merge='mean'):
    """8x TTA with D4 symmetry group."""
    preds = []

    for k in range(4):  # 0, 90, 180, 270 degree rotations
        img_rot = torch.rot90(image, k, [2, 3])

        for flip in [False, True]:
            img = torch.flip(img_rot, [3]) if flip else img_rot

            with torch.no_grad():
                logit = model(img)

            # Reverse augmentation
            if flip:
                logit = torch.flip(logit, [3])
            logit = torch.rot90(logit, -k, [2, 3])

            preds.append(torch.sigmoid(logit))

    if merge == 'mean':
        return torch.stack(preds).mean(dim=0)
    elif merge == 'max':
        return torch.stack(preds).max(dim=0)[0]
```

### Expected Gains
- 8x TTA on single model: **+1-3% IoU**
- 8x TTA on ensemble: **+0.5-1.5% IoU** (diminishing returns but still worth it)

---

## Strategy 5: STAPLE Algorithm for Label Fusion

STAPLE (Simultaneous Truth and Performance Level Estimation) is a probabilistic algorithm specifically designed for combining multiple segmentations in medical imaging.

### How It Works
1. Takes N binary segmentation masks as input
2. Iteratively estimates:
   - The "true" segmentation (E-step)
   - Each rater's sensitivity and specificity (M-step)
3. Returns a probabilistic fusion that weights each model by its estimated accuracy

### Implementation

```python
import SimpleITK as sitk
import numpy as np

def staple_fusion(binary_masks_list):
    """
    STAPLE label fusion using SimpleITK.

    Args:
        binary_masks_list: list of numpy arrays, each H x W, values 0 or 1

    Returns:
        fused_mask: probability map (threshold at 0.5 for binary)
    """
    sitk_masks = [sitk.GetImageFromArray(m.astype(np.uint8)) for m in binary_masks_list]

    staple_filter = sitk.STAPLEImageFilter()
    staple_filter.SetForegroundValue(1)

    result = staple_filter.Execute(sitk_masks)
    fused_prob = sitk.GetArrayFromImage(result)

    # Get estimated sensitivity/specificity per model
    for i in range(len(binary_masks_list)):
        sens = staple_filter.GetSensitivity(i)
        spec = staple_filter.GetSpecificity(i)
        print(f"Model {i}: sensitivity={sens:.4f}, specificity={spec:.4f}")

    return fused_prob
```

### Limitations for Our Case
- STAPLE works on **binary masks**, not probability maps -- loses soft information
- Tends to underestimate boundaries (shrinks edges via majority voting)
- Better suited when you have many (5+) models with similar quality
- **Recommendation**: Use as a comparison baseline, but probability averaging will likely beat it

---

## Strategy 6: Model Soups (Weight Averaging)

Instead of ensembling predictions, average the model weights themselves. Zero additional inference cost.

### When It Works
- Models must share the **same architecture** and be fine-tuned from the **same pretrained checkpoint**
- Works best when models are fine-tuned with different hyperparameters (LR, augmentation, loss)
- The 5 ISIC pretrained folds are perfect candidates

### Implementation

```python
def model_soup(model_paths, architecture_fn):
    """
    Average weights of multiple models (must share architecture).
    From Wortsman et al., 'Model Soups', ICML 2022.
    """
    state_dicts = [torch.load(p, map_location='cpu')['model_state_dict'] for p in model_paths]

    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)

    model = architecture_fn()
    model.load_state_dict(avg_state)
    return model

def greedy_model_soup(model_paths, architecture_fn, val_loader):
    """
    Greedy soup: only include models that improve validation IoU.
    """
    # Sort by individual val IoU
    scored = []
    for path in model_paths:
        model = architecture_fn()
        model.load_state_dict(torch.load(path)['model_state_dict'])
        iou = evaluate(model, val_loader)
        scored.append((path, iou))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build soup greedily
    soup_paths = [scored[0][0]]
    best_iou = scored[0][1]

    for path, _ in scored[1:]:
        candidate = model_soup(soup_paths + [path], architecture_fn)
        candidate_iou = evaluate(candidate, val_loader)
        if candidate_iou > best_iou:
            soup_paths.append(path)
            best_iou = candidate_iou

    return model_soup(soup_paths, architecture_fn), best_iou
```

### Application to Our Models
- **5 ISIC pretrained folds** (same arch, same pretrain): Perfect for model soup -> single model with ~0.81-0.82 IoU, no extra inference cost
- **Lovasz MEGA + ISIC folds** (same arch, different training): May work if loss landscapes are connected
- **Cannot soup**: U-Net++ B4 with DeepLabV3+ R50 (different architectures)

---

## Strategy 7: Optimal Threshold Search

Default threshold of 0.5 is rarely optimal. Search for the best threshold on validation data.

```python
def find_optimal_threshold(probs, gt_masks, thresholds=None):
    """Search for threshold that maximizes mean IoU."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.70, 0.01)

    best_iou = 0
    best_thresh = 0.5

    for t in thresholds:
        masks = (probs > t).astype(np.float32)
        iou = compute_mean_iou(masks, gt_masks)
        if iou > best_iou:
            best_iou = iou
            best_thresh = t

    return best_thresh, best_iou

# Can also do per-image adaptive thresholding:
def adaptive_threshold(prob_map, base_thresh=0.5):
    """Otsu-like adaptive threshold per image."""
    from skimage.filters import threshold_otsu
    try:
        t = threshold_otsu(prob_map)
        # Blend with base threshold for stability
        return 0.7 * t + 0.3 * base_thresh
    except ValueError:
        return base_thresh
```

### Expected Gain: +0.5-1.5% IoU
- Often the optimal threshold is 0.45-0.48 (slightly below 0.5)
- This is free performance -- always do this

---

## Strategy 8: CRF Post-Processing for Boundary Refinement

Dense CRF can sharpen boundaries and improve IoU by 1-3%.

```python
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crf_refine(image_rgb, prob_map, n_iters=5):
    """
    Apply Dense CRF to refine segmentation boundaries.

    Args:
        image_rgb: H x W x 3 uint8 image
        prob_map: H x W float probability map [0, 1]
    """
    H, W = prob_map.shape

    # Create 2-class probability
    probs = np.stack([1 - prob_map, prob_map], axis=0)

    d = dcrf.DenseCRF2D(W, H, 2)

    # Unary potential
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Pairwise potentials
    # Appearance kernel (uses image color)
    d.addPairwiseBilateral(
        sxy=50, srgb=13,
        rgbim=image_rgb.astype(np.uint8),
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Smoothness kernel
    d.addPairwiseGaussian(sxy=3, compat=3)

    # Inference
    Q = d.inference(n_iters)
    result = np.array(Q).reshape(2, H, W)

    return result[1]  # Foreground probability

# IMPORTANT: Tune sxy, srgb, compat on validation set!
# For skin lesions, typical good values:
#   sxy=30-80, srgb=5-20, compat=5-15
```

### When to Use
- After ensemble averaging, before final thresholding
- Most helpful when boundaries are fuzzy or jagged
- **Caution**: Can hurt performance if lesion boundaries are naturally smooth -- validate first

---

## Strategy 9: Stacking / Meta-Learner Fusion

Train a small model to learn the optimal combination of base model predictions.

```python
def prepare_stacking_features(models, images, masks):
    """
    Create feature matrix for stacking: each pixel gets N probability values
    (one per model) as features, and the true mask as target.
    """
    all_features = []
    all_targets = []

    for img, mask in zip(images, masks):
        model_probs = []
        for model in models:
            with torch.no_grad():
                prob = torch.sigmoid(model(img.unsqueeze(0)))
            model_probs.append(prob.squeeze().cpu().numpy())

        # Stack: shape (N_models, H, W) -> flatten to (H*W, N_models)
        features = np.stack(model_probs, axis=-1).reshape(-1, len(models))
        targets = mask.cpu().numpy().flatten()

        all_features.append(features)
        all_targets.append(targets)

    return np.vstack(all_features), np.concatenate(all_targets)

# Train a simple logistic regression or small MLP as meta-learner
from sklearn.linear_model import LogisticRegression

X_train, y_train = prepare_stacking_features(models, val_images, val_masks)
meta_model = LogisticRegression(C=1.0).fit(X_train, y_train)
```

### Pros and Cons
- **Pro**: Can learn complex non-linear combinations
- **Pro**: Can learn that some models are better in certain image regions
- **Con**: Risk of overfitting on small validation set (400 images)
- **Recommendation**: Use only if simpler methods plateau; prefer logistic regression over deep meta-learners

---

## Recommended Pipeline (Priority Order)

### Phase 1: Quick Wins (Expected: 0.82 -> 0.85-0.87)

1. **Model soup the 5 ISIC folds** into a single model (~0.81-0.82, free)
2. **8x TTA on Lovasz MEGA 512** alone -> ~0.84-0.85
3. **Simple average** of Lovasz MEGA (with TTA) + ISIC soup (with TTA) -> ~0.85-0.86
4. **Optimal threshold search** on validation set -> +0.5-1%

### Phase 2: Careful Optimization (Expected: 0.86 -> 0.88-0.89)

5. **Add diversity models** (DeepLabV3+, U-Net R34) to ensemble with low weights
6. **Grid search weights** on validation set
7. **Multi-resolution benefit**: 512px + 384px + 256px predictions all resized to original, weighted average
8. **CRF post-processing** on ensemble output (tune on validation)

### Phase 3: Squeeze Last Drops (Expected: 0.89 -> 0.90+)

9. **Per-image adaptive thresholding**
10. **Greedy ensemble selection** (remove models that hurt)
11. **Stacking meta-learner** if validation set is large enough
12. **STAPLE comparison** as sanity check

---

## Quick Implementation Script

```python
"""
Full ensemble pipeline for segmentation.
Run on 400-image validation set to find optimal config,
then apply to 200-image test set.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Step 1: Load all models
models = {
    'lovasz_mega_512': load_model('lovasz_mega_512.pth', arch='unetpp_effv2s', res=512),
    'isic_soup': load_soup_model(['fold1.pth', ..., 'fold5.pth'], arch='unetpp_effv2s', res=384),
    'unetpp_b4': load_model('unetpp_b4.pth', arch='unetpp_b4', res=256),
    'deeplab_r50': load_model('deeplab_r50.pth', arch='deeplabv3p_r50', res=256),
    'unet_r34': load_model('unet_r34.pth', arch='unet_r34', res=256),
}

# Step 2: Generate TTA predictions for all models on validation set
val_predictions = {}
for name, (model, res) in models.items():
    val_predictions[name] = predict_with_tta(model, val_loader, res, tta='d4')

# Step 3: Find optimal weights
weights, best_iou = grid_search_weights(
    list(val_predictions.values()),
    val_gt_masks,
    step=0.05
)
print(f"Optimal weights: {dict(zip(models.keys(), weights))}")
print(f"Validation IoU: {best_iou:.4f}")

# Step 4: Find optimal threshold
ensemble_probs = weighted_average(val_predictions, weights)
best_thresh, thresh_iou = find_optimal_threshold(ensemble_probs, val_gt_masks)
print(f"Optimal threshold: {best_thresh:.3f}, IoU: {thresh_iou:.4f}")

# Step 5: Optional CRF refinement
if USE_CRF:
    refined_probs = [crf_refine(img, prob) for img, prob in zip(val_images, ensemble_probs)]
    crf_iou = compute_mean_iou(np.array(refined_probs) > best_thresh, val_gt_masks)
    print(f"CRF IoU: {crf_iou:.4f} (delta: {crf_iou - thresh_iou:+.4f})")

# Step 6: Apply to test set with best config
test_predictions = generate_test_predictions(models, weights, best_thresh, use_crf=USE_CRF)
save_binary_masks(test_predictions, output_dir='WhiteCoat.dev/')
```

---

## Empirical Expectations

| Strategy | Expected IoU Gain | Compute Cost |
|----------|-------------------|-------------|
| 8x TTA on best model | +1.5-3.0% | 8x inference |
| Probability averaging (2 models) | +2-3% | 2x inference |
| Probability averaging (5 models) | +3-5% | 5x inference |
| Model soup (5 folds) | +0.5-1.5% | 0x extra |
| Optimal threshold | +0.5-1.5% | Negligible |
| Weight optimization | +0.5-1.0% | Negligible |
| CRF post-processing | +0.5-2.0% | ~2x per image |
| Multi-resolution fusion | +1-2% | Already paid |
| Stacking meta-learner | +0.5-1.0% | Negligible |

**Cumulative realistic target**: 0.8268 (best single) -> 0.88-0.91 (full pipeline)

---

## References

- Wortsman et al., "Model Soups", ICML 2022 -- https://arxiv.org/abs/2203.05482
- Warfield et al., STAPLE algorithm -- SimpleITK implementation
- qubvel/ttach -- https://github.com/qubvel/ttach
- Albumentations TTA -- https://albumentations.ai/docs/4-advanced-guides/test-time-augmentation/
- pydensecrf -- Dense CRF post-processing
- segmentation_models_pytorch -- https://github.com/qubvel/segmentation_models.pytorch
