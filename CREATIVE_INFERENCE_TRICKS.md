# Creative Inference-Time Tricks for Segmentation Quality

**Context**: Skin lesion segmentation, U-Net++ baseline IoU 0.8268, images 128x128 upscaled to 512x512, Apple MPS (no CUDA GPU).

---

## Summary Table

| # | Technique | Expected IoU Gain | Impl. Time | Needs GPU? | Risk |
|---|-----------|-------------------|------------|------------|------|
| 1 | Super-Resolution Preprocessing | +0.005-0.020 | 2-3 hrs | MPS ok | Medium |
| 2 | Color Constancy (Shades of Gray) | +0.005-0.015 | 30 min | No (CPU) | Low |
| 3 | CLAHE + Histogram Equalization | +0.003-0.010 | 20 min | No (CPU) | Low |
| 4 | MC Dropout Self-Ensemble | +0.005-0.015 | 1-2 hrs | MPS ok | Low |
| 5 | Morphological Active Contours | +0.005-0.020 | 1-2 hrs | No (CPU) | Medium |
| 6 | DenseCRF Post-Processing | +0.010-0.025 | 1-2 hrs | No (CPU) | Medium |
| 7 | Multi-Scale Inference | +0.005-0.015 | 1 hr | MPS ok | Low |
| 8 | Boundary-Aware Refinement | +0.003-0.010 | 1 hr | No (CPU) | Low |
| 9 | Uncertainty-Guided Region Fix | +0.005-0.015 | 2 hrs | MPS ok | Medium |
| 10 | Patch-Based Voting | +0.003-0.010 | 1.5 hrs | MPS ok | Low |

**Realistic combined improvement (stacking 3-4 best)**: +0.015-0.040 IoU (reaching ~0.84-0.87)

---

## 1. Super-Resolution Before Segmentation

### Concept
Instead of naive bicubic upscaling from 128x128 to 512x512, use a lightweight SR model as an intermediate step: 128 -> 256 (SR) -> 512 (bicubic), or 128 -> 512 (SR directly). SR recovers high-frequency edge details that bicubic smooths away, giving the segmentation model sharper boundary cues.

### Why It Works for Skin Lesions
- Skin lesion boundaries are often subtle gradients between lesion and healthy skin
- Bicubic interpolation creates smooth, blurry transitions at boundaries
- SR models trained on natural images still recover edge sharpness that helps segmentation
- A 2025 PMC study found consistent improvements in boundary-sensitive metrics

### Implementation

```python
import torch
from PIL import Image
import numpy as np

# Option A: Real-ESRGAN (best quality, slower on CPU/MPS)
# pip install realesrgan basicsr
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4)  # lighter 6-block version
upsampler = RealESRGANer(scale=4, model_path='RealESRGAN_x4plus.pth',
                          model=model, tile=128, half=False)  # tile for memory
sr_img, _ = upsampler.enhance(img_128, outscale=4)  # 128->512

# Option B: EDSR-baseline (lightest, best for CPU/MPS)
# Via OpenCV DNN super-resolution
import cv2
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")
sr.setModel("edsr", 4)
sr_img = sr.upsample(img_128)  # 128->512

# Option C: Hybrid approach (recommended for speed)
# SR 128->256, then bicubic 256->512
sr.readModel("EDSR_x2.pb")
sr.setModel("edsr", 2)
sr_256 = sr.upsample(img_128)   # 128->256 with SR
sr_512 = cv2.resize(sr_256, (512, 512), interpolation=cv2.INTER_CUBIC)
```

### Practical Considerations
- **EDSR-baseline x2** is the best speed/quality tradeoff for CPU/MPS (~0.1-0.3s per image)
- **Real-ESRGAN** with the compact 6-block model works on MPS (~1-2s per image with tiling)
- The hybrid 128->256(SR)->512(bicubic) approach is often optimal: SR has most impact at 2x
- Must use SR on BOTH training and test images for consistency, OR only on test if model was trained on bicubic-upscaled images (test-time enhancement)

### Expected Impact
- **IoU improvement**: +0.005-0.020 (depends on how boundary-sensitive the current model is)
- **Best case**: when model was trained on bicubic-upscaled and test uses SR = sharper boundaries
- **Risk**: if model was trained on blurry bicubic, giving it SR images could cause domain shift

### References
- [Super-Resolution in Biomedical Imaging (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12027580/)
- [Medical image super-resolution survey (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0010482525006961)

---

## 2. Color Constancy: Shades of Gray

### Concept
Dermoscopy images suffer from inconsistent illumination across devices and clinics. The Shades of Gray algorithm normalizes the color distribution by estimating the scene illuminant using the Minkowski p-norm, then dividing out the estimated light color. This removes device-specific color bias.

### Why It Matters for Skin Lesions
- Different dermoscopes produce different color casts (warm vs cool)
- Training data may mix multiple devices; test images may come from unseen devices
- Color normalization creates a canonical color space, reducing domain gap at test time
- Research has shown Shades of Gray improves dermoscopy classification sensitivity significantly

### Implementation

```python
import numpy as np
import cv2

def shades_of_gray(img, power=6):
    """
    Shades of Gray color constancy.
    power=1: Gray World, power=inf: Max-RGB, power=6: recommended for dermoscopy
    """
    img = img.astype(np.float32)
    # Compute Minkowski p-norm per channel
    norm = np.power(np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power)
    # Normalize to gray world assumption
    norm = norm / np.max(norm)
    # Avoid division by zero
    norm = np.maximum(norm, 1e-6)
    result = img / norm[np.newaxis, np.newaxis, :]
    return np.clip(result, 0, 255).astype(np.uint8)

# Apply to test images
img = cv2.imread("test_image.png")
img_normalized = shades_of_gray(img, power=6)
```

### Practical Considerations
- **power=6** is the standard for dermoscopy (Barata et al. 2014)
- Apply to BOTH train and test for consistency; or apply to test only if model is robust
- Zero computational overhead (pure NumPy, <1ms per image)
- Can be combined with CLAHE for double benefit
- If training data was already color-normalized, applying to test data maintains consistency

### Expected Impact
- **IoU improvement**: +0.005-0.015
- Biggest gains when test images have different color profiles than training
- Almost zero risk: if images are already normalized, effect is minimal

### References
- [Color constancy for dermoscopy classification (PubMed)](https://pubmed.ncbi.nlm.nih.gov/25073179/)
- [Color constancy effect on skin lesion segmentation (ResearchGate)](https://www.researchgate.net/publication/331794206_The_effect_of_color_constancy_algorithms_on_semantic_segmentation_of_skin_lesions)
- [Shades of Gray GitHub implementation](https://github.com/nickshawn/Shades_of_Gray-color_constancy_transformation)

---

## 3. CLAHE and Histogram Equalization

### Concept
Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the luminance channel to enhance local contrast, making lesion boundaries more visible to the segmentation model.

### Implementation

```python
import cv2
import numpy as np

def enhance_for_segmentation(img):
    """CLAHE on L-channel of LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# Multi-channel enhancement variant
def clahe_all_channels(img, clip_limit=2.0):
    """Apply CLAHE to each channel independently in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v)
    s_enhanced = clahe.apply(s)
    enhanced = cv2.merge([h, s_enhanced, v_enhanced])
    return cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
```

### Practical Considerations
- clipLimit=2.0 is safe; higher values (3.0-4.0) risk amplifying noise
- LAB-space CLAHE is preferred over RGB (preserves color relationships)
- Essentially free computation (<1ms per image)
- Combine with Shades of Gray: first color constancy, then CLAHE

### Expected Impact
- **IoU improvement**: +0.003-0.010
- Helps most on low-contrast images where lesion-skin boundary is subtle

---

## 4. MC Dropout Self-Ensemble (Test-Time Dropout)

### Concept
Enable dropout layers during inference and run the model N times. Average the N probability maps to get a smoother, more robust prediction. The variance across runs gives an uncertainty map identifying regions where the model is unsure.

### Why It Works
- Approximates Bayesian inference: each dropout pattern samples a different sub-network
- Averaging multiple stochastic predictions reduces noise and smooths boundaries
- The uncertainty map can be used for downstream refinement (see Section 9)
- 2025 research shows MC-Frequency Dropout improves calibration for medical segmentation

### Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np

def enable_mc_dropout(model):
    """Enable dropout layers during mode that is normally non-stochastic."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()  # keep dropout active

def mc_dropout_inference(model, image, n_forward=20, threshold=0.5):
    """
    Run MC Dropout inference.
    Returns: mean prediction, uncertainty map, binary mask
    """
    model.eval()  # set to non-training mode first
    enable_mc_dropout(model)  # then re-enable dropout specifically

    predictions = []
    with torch.no_grad():
        for _ in range(n_forward):
            logits = model(image)
            prob = torch.sigmoid(logits)
            predictions.append(prob)

    # Stack and compute statistics
    preds = torch.stack(predictions)          # (N, B, 1, H, W)
    mean_pred = preds.mean(dim=0)             # (B, 1, H, W)
    uncertainty = preds.std(dim=0)            # (B, 1, H, W)

    # Binary mask from mean prediction
    mask = (mean_pred > threshold).float()

    return mean_pred, uncertainty, mask
```

### Practical Considerations
- **N=10-20 forward passes** is the sweet spot (diminishing returns beyond 20)
- If model has NO dropout layers, you need to add Dropout2d(p=0.1) to decoder blocks
- 20 passes = 20x inference time; at 512x512 on MPS this is ~2-4 seconds per image
- For 200 test images: ~5-15 minutes total (acceptable)
- The uncertainty map is valuable input for technique 9 (uncertainty-guided refinement)

### Expected Impact
- **IoU improvement**: +0.005-0.015 from averaging alone
- The uncertainty map enables further +0.005-0.010 via guided refinement
- Low risk: averaging never hurts, worst case is same performance

### References
- [MC Dropout for Uncertainty Estimation (EmergentMind)](https://www.emergentmind.com/topics/monte-carlo-dropout-mc-dropout)
- [MC-Frequency Dropout for Segmentation (arXiv 2025)](https://arxiv.org/abs/2501.11258)
- [MC Dropout in Brain Tumor Segmentation (arXiv 2025)](https://arxiv.org/html/2510.15541v1)

---

## 5. Morphological Active Contour Refinement

### Concept
Use the neural network prediction as initialization for a morphological active contour (morphological snakes) that evolves the boundary to better fit the image edges. This refines jagged or imprecise boundaries into smooth, image-aligned contours.

### Two Approaches

**MorphGAC (Geodesic Active Contours)**: Evolves contours toward image edges. Best when the lesion boundary has visible gradient in the image. Needs an edge indicator (preprocessed image).

**MorphACWE (Chan-Vese / Active Contours Without Edges)**: Segments based on intensity differences between inside and outside regions. Works even without clear edges. Better for skin lesions where boundary is subtle.

### Implementation

```python
import numpy as np
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour
from skimage.segmentation import inverse_gaussian_gradient
from skimage.color import rgb2gray
from skimage.filters import gaussian

def refine_with_morphacwe(image_rgb, initial_mask, iterations=100, smoothing=2):
    """
    Refine segmentation mask using Chan-Vese active contour.
    image_rgb: (H, W, 3) uint8
    initial_mask: (H, W) binary {0, 1}
    """
    gray = rgb2gray(image_rgb)

    refined = morphological_chan_vese(
        gray,
        num_iter=iterations,
        init_level_set=initial_mask.astype(float),
        smoothing=smoothing,
        lambda1=1.0,   # weight for inside region
        lambda2=1.0,   # weight for outside region
    )
    return refined.astype(np.uint8)

def refine_with_morphgac(image_rgb, initial_mask, iterations=100, smoothing=1):
    """
    Refine using Geodesic Active Contour (edge-based).
    """
    gray = rgb2gray(image_rgb)
    # Create edge indicator function
    gimage = inverse_gaussian_gradient(gray, alpha=100, sigma=5.0)

    refined = morphological_geodesic_active_contour(
        gimage,
        num_iter=iterations,
        init_level_set=initial_mask.astype(float),
        smoothing=smoothing,
        threshold=0.69,  # balloon force threshold
        balloon=-1,      # -1 = shrink, +1 = expand
    )
    return refined.astype(np.uint8)

def refine_mask_adaptive(image, mask, method='acwe'):
    """
    Smart refinement: only refine boundary region.
    Keep interior confident pixels, only evolve near edges.
    """
    from skimage.morphology import dilation, erosion, disk

    # Create boundary band (dilated - eroded)
    dilated = dilation(mask, disk(10))
    eroded = erosion(mask, disk(10))
    boundary_band = dilated.astype(int) - eroded.astype(int)

    # Run active contour on full image
    if method == 'acwe':
        refined = refine_with_morphacwe(image, mask)
    else:
        refined = refine_with_morphgac(image, mask)

    # Blend: keep original in confident regions, use refined at boundaries
    result = mask.copy()
    result[boundary_band > 0] = refined[boundary_band > 0]

    return result
```

### Practical Considerations
- **MorphACWE** (Chan-Vese) is recommended for skin lesions: no edge preprocessing needed
- **iterations=50-150**: too few = no refinement, too many = contour wanders
- **smoothing=1-3**: controls contour regularity (higher = smoother boundary)
- Pure CPU, fast: ~0.1-0.5s per image at 512x512
- The "adaptive" approach (only refine boundary band) is safer than full re-segmentation
- Risk: active contour may shrink or expand beyond correct boundary. Limit iterations.

### Expected Impact
- **IoU improvement**: +0.005-0.020
- Biggest gains on images with irregular or jagged prediction boundaries
- MorphACWE works well when lesion is darker than surrounding skin (high contrast)

### References
- [Morphological Snakes (scikit-image docs)](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_morphsnakes.html)
- [skimage.segmentation API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)

---

## 6. DenseCRF Post-Processing

### Concept
Apply a fully connected Conditional Random Field (DenseCRF) as post-processing. The CRF considers both the unary potentials (model prediction) and pairwise potentials (spatial proximity + color similarity), encouraging spatially coherent predictions that align with image edges.

### Why It Works for Skin Lesions
- Neural networks predict each pixel somewhat independently; CRF enforces spatial coherence
- The bilateral kernel in DenseCRF uses both spatial distance AND color similarity
- This aligns prediction boundaries with actual color edges in the image
- Research on melanoma segmentation shows CRF + TTA combined gives consistent improvements

### Implementation

```python
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_densecrf(image_rgb, prob_map, n_iterations=5,
                   sxy_gaussian=3, compat_gaussian=3,
                   sxy_bilateral=80, srgb_bilateral=13, compat_bilateral=10):
    """
    Apply DenseCRF post-processing.

    image_rgb: (H, W, 3) uint8
    prob_map: (H, W) float32, probability of foreground [0, 1]
    """
    h, w = prob_map.shape

    # Create 2-class probability (background, foreground)
    prob_2class = np.stack([1.0 - prob_map, prob_map], axis=0)  # (2, H, W)

    # Unary potentials from softmax probabilities
    unary = unary_from_softmax(prob_2class)

    # Setup CRF
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)

    # Gaussian pairwise: encourages nearby pixels to have same label
    d.addPairwiseGaussian(
        sxy=sxy_gaussian,
        compat=compat_gaussian,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Bilateral pairwise: spatial + color proximity
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=image_rgb.copy(order='C'),
        compat=compat_bilateral,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )

    # Inference
    Q = d.inference(n_iterations)
    result = np.argmax(np.array(Q).reshape((2, h, w)), axis=0)

    return result.astype(np.uint8)

# Optimized parameters for dermoscopy (from literature)
def crf_for_dermoscopy(image, prob_map):
    return apply_densecrf(
        image, prob_map,
        n_iterations=10,
        sxy_gaussian=3,
        compat_gaussian=3,
        sxy_bilateral=50,      # skin lesions: larger spatial range
        srgb_bilateral=10,     # moderate color sensitivity
        compat_bilateral=10
    )
```

### Practical Considerations
- Install: `pip install pydensecrf` (or from GitHub source)
- **Key parameters**: `sxy_bilateral` (spatial range) and `srgb_bilateral` (color sensitivity)
- Higher `srgb_bilateral` = more color-sensitive (good for clear lesion-skin boundary)
- Lower `srgb_bilateral` = more smoothing (good for fuzzy boundaries)
- Tune on validation set: try sxy_bilateral in [30, 50, 80], srgb_bilateral in [5, 10, 15]
- Speed: ~0.05-0.2s per image (CPU), very fast
- Needs the soft probability map from model, not the binary mask

### Expected Impact
- **IoU improvement**: +0.010-0.025
- This is one of the most reliable post-processing methods in the literature
- Published melanoma segmentation research confirms consistent gains
- Best when model predictions have correct overall shape but noisy boundaries

### References
- [Melanoma segmentation with TTA and CRF (Nature Scientific Reports)](https://www.nature.com/articles/s41598-022-07885-y)
- [End-to-end segmentation with posterior CRF (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S136184152100356X)

---

## 7. Multi-Scale Inference

### Concept
Run the segmentation model at multiple input scales (e.g., 384, 512, 640), resize predictions back to the original size, and average. Different scales capture different levels of detail.

### Implementation

```python
import torch
import torch.nn.functional as F

def multi_scale_inference(model, image_tensor, scales=[0.75, 1.0, 1.25],
                          device='mps'):
    """
    image_tensor: (1, 3, H, W) normalized
    """
    model.eval()
    _, _, H, W = image_tensor.shape
    predictions = []

    with torch.no_grad():
        for scale in scales:
            new_h = int(H * scale)
            new_w = int(W * scale)

            # Resize input
            scaled = F.interpolate(image_tensor, size=(new_h, new_w),
                                   mode='bilinear', align_corners=False)

            # Predict
            logits = model(scaled.to(device))
            prob = torch.sigmoid(logits)

            # Resize prediction back to original size
            prob_resized = F.interpolate(prob, size=(H, W),
                                         mode='bilinear', align_corners=False)
            predictions.append(prob_resized.cpu())

    # Average all scales
    mean_pred = torch.stack(predictions).mean(dim=0)
    return mean_pred

# Combine with TTA (flips + scales)
def full_tta_multiscale(model, image, scales=[0.75, 1.0, 1.25], device='mps'):
    """Complete TTA: flips x rotations x scales."""
    all_preds = []

    # All 8 flip/rotation augmentations
    augmentations = [
        lambda x: x,                                         # original
        lambda x: torch.flip(x, [-1]),                       # h-flip
        lambda x: torch.flip(x, [-2]),                       # v-flip
        lambda x: torch.flip(x, [-1, -2]),                   # both flips
        lambda x: torch.rot90(x, 1, [-2, -1]),               # 90
        lambda x: torch.rot90(x, 2, [-2, -1]),               # 180
        lambda x: torch.rot90(x, 3, [-2, -1]),               # 270
        lambda x: torch.flip(torch.rot90(x, 1, [-2, -1]), [-1]),  # 90+flip
    ]

    inverse_augmentations = [
        lambda x: x,
        lambda x: torch.flip(x, [-1]),
        lambda x: torch.flip(x, [-2]),
        lambda x: torch.flip(x, [-1, -2]),
        lambda x: torch.rot90(x, 3, [-2, -1]),
        lambda x: torch.rot90(x, 2, [-2, -1]),
        lambda x: torch.rot90(x, 1, [-2, -1]),
        lambda x: torch.flip(torch.rot90(x, 3, [-2, -1]), [-1]),
    ]

    for aug, inv_aug in zip(augmentations, inverse_augmentations):
        augmented = aug(image)
        ms_pred = multi_scale_inference(model, augmented, scales, device)
        restored = inv_aug(ms_pred)
        all_preds.append(restored)

    return torch.stack(all_preds).mean(dim=0)
```

### Practical Considerations
- 3 scales x 8 augmentations = 24 forward passes per image
- On MPS at 512x512: ~10-20 seconds per image, ~30-60 min for 200 images
- Scales [0.75, 1.0, 1.25] are standard; [0.5, 0.75, 1.0, 1.25, 1.5] for maximum quality
- Larger scales may cause MPS memory issues; use [0.75, 1.0, 1.25] to be safe
- Diminishing returns: adding more scales beyond 3-5 gives minimal improvement

### Expected Impact
- **IoU improvement**: +0.005-0.015 over single-scale TTA
- If you already have 8x flip/rotation TTA, adding scales gives incremental gain

---

## 8. Boundary-Aware Morphological Refinement

### Concept
A simple but effective CPU-only pipeline: detect the prediction boundary, apply targeted morphological operations to clean it up, and use the original image edges to snap the boundary to the nearest real edge.

### Implementation

```python
import numpy as np
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import sobel
from skimage.color import rgb2gray

def boundary_aware_refinement(image_rgb, mask, min_object_size=500,
                               min_hole_size=500, edge_snap_radius=5):
    """
    Multi-step morphological refinement.
    """
    mask = mask.astype(bool)

    # Step 1: Remove small spurious objects
    mask = remove_small_objects(mask, min_size=min_object_size)

    # Step 2: Fill small holes
    mask = remove_small_holes(mask, area_threshold=min_hole_size)

    # Step 3: Smooth boundary with morphological opening then closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Step 4: Final smoothing with Gaussian blur on boundary
    mask_float = mask_uint8.astype(np.float32) / 255.0
    mask_smooth = cv2.GaussianBlur(mask_float, (3, 3), 0)
    result = (mask_smooth > 0.5).astype(np.uint8)

    return result

def clean_prediction(mask, image=None):
    """Quick cleanup pipeline."""
    mask = mask.astype(np.uint8)

    # Keep only largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # Find largest non-background component
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # Fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, -1)

    return mask
```

### Expected Impact
- **IoU improvement**: +0.003-0.010
- Very safe: only cleans up obviously wrong predictions
- "Keep largest component" alone often fixes +0.003-0.005 on noisy predictions

---

## 9. Uncertainty-Guided Region Refinement

### Concept
Use the uncertainty map from MC Dropout (Section 4) to identify boundary regions where the model is unsure. Then apply targeted refinement ONLY in those uncertain regions, keeping confident regions untouched.

### Implementation

```python
import numpy as np
import torch
from skimage.segmentation import morphological_chan_vese
from skimage.color import rgb2gray

def uncertainty_guided_refinement(image_rgb, mean_pred, uncertainty,
                                   uncertainty_threshold=0.15):
    """
    Refine only high-uncertainty regions.

    mean_pred: (H, W) float, mean probability from MC Dropout
    uncertainty: (H, W) float, std from MC Dropout
    """
    h, w = mean_pred.shape

    # Step 1: Identify uncertain regions
    uncertain_mask = uncertainty > uncertainty_threshold

    # Step 2: Create confident baseline
    confident_fg = (mean_pred > 0.7) & ~uncertain_mask
    confident_bg = (mean_pred < 0.3) & ~uncertain_mask

    # Step 3: For uncertain regions, use active contour locally
    gray = rgb2gray(image_rgb)

    # Initialize with mean prediction
    init_mask = (mean_pred > 0.5).astype(float)

    # Run Chan-Vese
    refined = morphological_chan_vese(
        gray, num_iter=80,
        init_level_set=init_mask,
        smoothing=2
    )

    # Step 4: Blend - keep confident predictions, use refined for uncertain
    result = init_mask.copy()
    result[confident_fg] = 1.0
    result[confident_bg] = 0.0
    result[uncertain_mask] = refined[uncertain_mask]

    return result.astype(np.uint8)

def adaptive_threshold_from_uncertainty(mean_pred, uncertainty):
    """
    Use uncertainty to set per-pixel threshold instead of global 0.5.
    High uncertainty -> require higher confidence to predict foreground.
    """
    # Base threshold 0.5, increase in uncertain regions
    threshold = 0.5 + 0.2 * (uncertainty / uncertainty.max())
    mask = (mean_pred > threshold).astype(np.uint8)
    return mask
```

### Practical Considerations
- Requires MC Dropout output (Section 4) as prerequisite
- uncertainty_threshold: tune on validation set (0.1-0.2 typical)
- The adaptive threshold approach is simpler and often equally effective
- Can combine with DenseCRF: run CRF only in uncertain regions for speed

### Expected Impact
- **IoU improvement**: +0.005-0.015 on top of MC Dropout
- Biggest gains on images where the model is partially confused about boundary location

### References
- [TEGDA: Test-time Evaluation-Guided Dynamic Adaptation (MICCAI 2025)](https://papers.miccai.org/miccai-2025/0906-Paper2263.html)
- [TRUST: Uncertainty-Guided SSM Traverses (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/117665)

---

## 10. Patch-Based Voting (Puzzle-Mix TTA)

### Concept
Instead of segmenting the whole 512x512 image, extract overlapping patches (e.g., 256x256 with stride 128), segment each patch, and average overlapping predictions. Boundary regions get predictions from multiple patches, reducing edge artifacts.

### Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np

def patch_based_inference(model, image_tensor, patch_size=256, stride=128,
                          device='mps'):
    """
    Sliding window inference with overlap averaging.
    image_tensor: (1, 3, H, W)
    """
    _, _, H, W = image_tensor.shape
    pred_sum = torch.zeros(1, 1, H, W)
    count = torch.zeros(1, 1, H, W)

    model.eval()
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
                # Resize patch to model input size if needed
                patch_resized = F.interpolate(patch, size=(512, 512),
                                               mode='bilinear')

                logits = model(patch_resized.to(device))
                prob = torch.sigmoid(logits).cpu()

                # Resize back to patch size
                prob_patch = F.interpolate(prob, size=(patch_size, patch_size),
                                            mode='bilinear')

                pred_sum[:, :, y:y+patch_size, x:x+patch_size] += prob_patch
                count[:, :, y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping regions
    result = pred_sum / count.clamp(min=1)
    return result
```

### Expected Impact
- **IoU improvement**: +0.003-0.010
- Most useful when model struggles with images that have lesions at unusual positions

---

## 11. Stain Normalization (Bonus: Dermoscopy-Specific)

### Concept
Normalize the color distribution of test images to match a reference "template" image from the training set using Reinhard color normalization in LAB space.

### Implementation

```python
import numpy as np
import cv2

def reinhard_normalize(source, target):
    """
    Reinhard color normalization: match mean and std of target.
    Works in LAB color space.
    """
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    s_mean, s_std = source_lab.mean(axis=(0, 1)), source_lab.std(axis=(0, 1))
    t_mean, t_std = target_lab.mean(axis=(0, 1)), target_lab.std(axis=(0, 1))

    result = (source_lab - s_mean) * (t_std / (s_std + 1e-6)) + t_mean
    result = np.clip(result, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
```

### Expected Impact
- **IoU improvement**: +0.003-0.010
- Zero GPU requirement, <1ms per image

---

## 12. Prediction Averaging Across Model Checkpoints

### Concept
If you saved checkpoints from different training epochs or folds, average their predictions. Different checkpoints capture different aspects of the data distribution.

### Implementation

```python
def checkpoint_ensemble(models, image, device='mps'):
    """Average predictions from multiple model checkpoints."""
    all_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(image.to(device))
            prob = torch.sigmoid(logits).cpu()
            all_preds.append(prob)

    return torch.stack(all_preds).mean(dim=0)

# Use checkpoints from: best_val, last_epoch, epoch_N-5, epoch_N-10
# Or from different folds if using K-fold CV
```

### Expected Impact
- **IoU improvement**: +0.005-0.020 (depends on checkpoint diversity)
- No additional training needed if checkpoints were saved during training

---

## Recommended Pipeline (Priority Order)

For maximum IoU improvement with limited time:

### Quick Wins (30 min, +0.01-0.02 expected)
1. **Shades of Gray** color constancy on all test images
2. **CLAHE** enhancement on L-channel
3. **Keep largest component** + fill holes cleanup

### Medium Investment (2-3 hrs, +0.02-0.04 cumulative)
4. **DenseCRF** post-processing (tune on validation)
5. **Multi-scale TTA** (add scales [0.75, 1.25] to existing 8x TTA)
6. **Morphological Active Contour** refinement (Chan-Vese)

### Full Pipeline (4-6 hrs, +0.03-0.05 cumulative)
7. **MC Dropout** self-ensemble (if model has/can accept dropout)
8. **Uncertainty-guided refinement** using MC Dropout uncertainty
9. **Super-Resolution** preprocessing (EDSR-baseline x2)
10. **Checkpoint ensemble** (if multiple checkpoints available)

### Recommended Execution Order

```
Test Image
    |
    v
[Shades of Gray] -> [CLAHE] -> [Optional: SR upscale]
    |
    v
[Model Inference: multi-scale + TTA + MC Dropout]
    |
    v
[Average predictions] -> soft probability map
    |
    v
[DenseCRF] -> refined probability map
    |
    v
[Threshold 0.5] -> binary mask
    |
    v
[Morphological cleanup: largest component, fill holes]
    |
    v
[Optional: Active Contour on uncertain boundaries]
    |
    v
Final Mask
```

---

## Risk Assessment

| Technique | Can Hurt IoU? | When It Hurts |
|-----------|---------------|---------------|
| Color Constancy | Rarely | If training data was already normalized differently |
| CLAHE | Rarely | If images are already high contrast |
| Super-Resolution | Sometimes | Domain shift if model trained on bicubic |
| MC Dropout | Almost never | Only adds noise if N is too small (<5) |
| DenseCRF | Sometimes | If parameters are badly tuned (always tune on val) |
| Active Contour | Sometimes | If iterations too high, contour wanders |
| Multi-Scale | Almost never | Slight memory pressure at large scales |
| Morphological cleanup | Almost never | If min_size threshold removes valid small lesions |

**Golden rule**: Always validate any new technique on the validation set before applying to test.

---

## Key References

- [Super-Resolution in Biomedical Imaging (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12027580/)
- [Melanoma Segmentation with TTA and CRF (Nature 2022)](https://www.nature.com/articles/s41598-022-07885-y)
- [TEGDA: Test-time Dynamic Adaptation (MICCAI 2025)](https://papers.miccai.org/miccai-2025/0906-Paper2263.html)
- [MC-Frequency Dropout for Segmentation (arXiv 2025)](https://arxiv.org/abs/2501.11258)
- [Morphological Snakes (scikit-image)](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_morphsnakes.html)
- [Color Constancy for Dermoscopy (PubMed)](https://pubmed.ncbi.nlm.nih.gov/25073179/)
- [SicTTA: Single Image Test Time Adaptation (2025)](https://www.sciencedirect.com/science/article/abs/pii/S1361841525004050)
- [Test-Time Generative Augmentation (2025)](https://www.sciencedirect.com/science/article/abs/pii/S1361841525004487)
