# Advanced Techniques for Medical Image Hackathon
## WhiteCoat.dev -- Biopsy Classification & Segmentation

---

## 1. ENSEMBLE STRATEGIES

### 1A. Classification Ensemble (EfficientNetV2-S + ConvNeXt-Tiny + Swin-Tiny)

**Why this trio works:** EfficientNetV2-S excels at efficient feature extraction, ConvNeXt-Tiny captures local convolutional features robustly, and Swin-Tiny provides hierarchical global attention. Their error patterns are complementary -- recent research (ViSwNeXtNet, 2025) combining exactly ConvNeXt-Tiny + Swin-Tiny + ViT achieved 94-99% accuracy on histopathology datasets.

#### Method A: Weighted Soft-Voting (RECOMMENDED -- Easy, +1-3% accuracy)

Average the softmax probability outputs with learned weights:

```python
import numpy as np
from scipy.optimize import minimize

def weighted_ensemble_predict(models, x, weights):
    """Weighted average of softmax predictions."""
    preds = [model.predict_proba(x) for model in models]
    weighted = sum(w * p for w, p in zip(weights, preds))
    return weighted / sum(weights)

def optimize_weights(val_preds_list, val_labels, n_models=3):
    """Find optimal weights using Nelder-Mead on validation accuracy."""
    def neg_accuracy(weights):
        weights = np.abs(weights)
        weighted = sum(w * p for w, p in zip(weights, val_preds_list))
        preds = np.argmax(weighted, axis=1)
        return -np.mean(preds == val_labels)

    result = minimize(neg_accuracy, x0=np.ones(n_models) / n_models,
                      method='Nelder-Mead')
    return np.abs(result.x) / np.sum(np.abs(result.x))

# Typical optimal weights for this combo:
# EfficientNetV2-S: 0.30, ConvNeXt-Tiny: 0.35, Swin-Tiny: 0.35
```

**Expected improvement:** +1-3% accuracy over best single model.
**Complexity:** Easy. Just save predictions from each model and combine.

#### Method B: Stacking with Meta-Learner (Medium, +2-4% accuracy)

Train a small logistic regression on top of the three models' outputs:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Step 1: Get out-of-fold predictions from each model using K-Fold
# Step 2: Stack them as features for meta-learner
stacked_features = np.concatenate([
    efn_val_preds,      # shape (N, 12)
    convnext_val_preds,  # shape (N, 12)
    swin_val_preds       # shape (N, 12)
], axis=1)  # shape (N, 36)

# Step 3: Train meta-learner
meta_model = LogisticRegression(C=1.0, max_iter=1000)
meta_model.fit(stacked_features, val_labels)

# Step 4: For test data, stack the same way and predict
test_stacked = np.concatenate([efn_test, convnext_test, swin_test], axis=1)
final_preds = meta_model.predict(test_stacked)
```

**Expected improvement:** +2-4% accuracy.
**Complexity:** Medium. Requires K-Fold out-of-fold predictions to avoid overfitting.
**Worth it for hackathon:** YES if you have time.

#### Method C: Majority Voting (Easy, +0.5-1.5% accuracy)

```python
from scipy.stats import mode

hard_preds = np.stack([
    np.argmax(efn_preds, axis=1),
    np.argmax(convnext_preds, axis=1),
    np.argmax(swin_preds, axis=1)
])
final_preds = mode(hard_preds, axis=0).mode[0]
```

**Expected improvement:** +0.5-1.5%. Less powerful than soft voting.
**Verdict:** Use soft voting or stacking instead.

### 1B. Segmentation Ensemble (2 models)

For combining 2 segmentation models (e.g., U-Net + DeepLabV3+):

```python
def ensemble_segmentation(model1_logits, model2_logits, w1=0.5, w2=0.5):
    """Average logits before sigmoid for better calibration."""
    combined = w1 * model1_logits + w2 * model2_logits
    return torch.sigmoid(combined)

# Optimize threshold on validation set
from sklearn.metrics import jaccard_score

def find_best_threshold(pred_probs, true_masks, thresholds=np.arange(0.3, 0.7, 0.01)):
    best_iou, best_t = 0, 0.5
    for t in thresholds:
        binary = (pred_probs > t).astype(int)
        iou = jaccard_score(true_masks.flatten(), binary.flatten())
        if iou > best_iou:
            best_iou, best_t = iou, t
    return best_t, best_iou
```

**Expected improvement:** +1-3% IoU over single model.
**Key insight:** Average LOGITS (before sigmoid), not probabilities. This preserves calibration.

---

## 2. TEST-TIME AUGMENTATION (TTA)

### 2A. TTA for Classification (Easy, +0.5-2% accuracy)

**Best TTA transforms for histopathology:**
- Horizontal flip (most important)
- Vertical flip (tissue has no canonical orientation)
- 90/180/270 degree rotations (tissue is rotationally invariant)
- Small color jitter (simulates stain variation)

```python
import torch
import torchvision.transforms.functional as TF

def tta_classify(model, image, n_classes=12):
    """8-fold TTA: identity + 3 rotations + flips."""
    model.eval()
    preds = []

    transforms_list = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.rotate(x, 90),
        lambda x: TF.rotate(x, 180),
        lambda x: TF.rotate(x, 270),
        lambda x: TF.vflip(TF.hflip(x)),
        lambda x: TF.hflip(TF.rotate(x, 90)),
    ]

    with torch.no_grad():
        for t in transforms_list:
            augmented = t(image)
            pred = torch.softmax(model(augmented.unsqueeze(0)), dim=1)
            preds.append(pred)

    return torch.stack(preds).mean(dim=0)
```

**Expected improvement:** +0.5-2% accuracy. Consistent gains on histopathology.
**Complexity:** Easy. Only affects inference, no retraining needed.
**Cost:** 8x slower inference (acceptable for hackathon submission).

### 2B. TTA for Segmentation (Easy, +1-3% IoU)

```python
def tta_segment(model, image):
    """TTA for segmentation with proper inverse transforms."""
    model.eval()
    preds = []

    with torch.no_grad():
        # Original
        preds.append(model(image.unsqueeze(0)))

        # Horizontal flip -> predict -> flip back
        flipped_h = TF.hflip(image)
        pred_h = model(flipped_h.unsqueeze(0))
        preds.append(TF.hflip(pred_h.squeeze(0)).unsqueeze(0))

        # Vertical flip -> predict -> flip back
        flipped_v = TF.vflip(image)
        pred_v = model(flipped_v.unsqueeze(0))
        preds.append(TF.vflip(pred_v.squeeze(0)).unsqueeze(0))

        # 90 degree rotation -> predict -> rotate back
        rot90 = TF.rotate(image, 90)
        pred_r = model(rot90.unsqueeze(0))
        preds.append(TF.rotate(pred_r.squeeze(0), -90).unsqueeze(0))

    avg_pred = torch.stack(preds).mean(dim=0)
    return avg_pred

# CRITICAL: You must apply the INVERSE transform to the predicted mask
# before averaging. This is the most common mistake with segmentation TTA.
```

**Expected improvement:** +1-3% IoU. Particularly effective at boundaries.

---

## 3. ADVANCED TRAINING TRICKS

### 3A. CutMix (RECOMMENDED for histopathology -- Medium, +1-3% accuracy)

**Why CutMix over Mixup for histopathology:** CutMix preserves local spatial patterns (tissue structures) while Mixup creates unrealistic blended images.

```python
import numpy as np
import torch

def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut and paste patches between training images."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# In training loop:
# inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)
# loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
```

**Expected improvement:** +1-3% accuracy.
**Alpha:** Use alpha=1.0. Apply CutMix with probability 0.5 per batch.

### 3B. Mixup (Easy, +0.5-1.5%)

```python
def mixup_data(x, y, alpha=0.2):
    """Mixup: linear interpolation of image pairs."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

**For histopathology:** Use alpha=0.2 (gentle mixing).

### 3C. Progressive Resizing (Medium, +2-5% accuracy)

Train on smaller images first, then fine-tune on larger:

```python
# Phase 1: Train on 64x64 for quick convergence (15-20 epochs)
train_transform_small = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# Train model... save checkpoint

# Phase 2: Fine-tune on 100x100 (original size) for 10-15 epochs
train_transform_large = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# Load Phase 1 weights, reduce learning rate by 10x, fine-tune
```

**Expected improvement:** +2-5% accuracy. Also trains 2-3x faster in initial phase.
**Worth it:** YES. One of the best time-to-improvement ratios.

### 3D. Stochastic Weight Averaging -- SWA (Easy, +0.5-1.5%)

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# After normal training for ~80% of epochs:
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.001)

for epoch in range(swa_start, total_epochs):
    train_one_epoch(model, train_loader, optimizer)
    swa_model.update_parameters(model)
    swa_scheduler.step()

# CRITICAL: Update batch normalization statistics
torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

**Expected improvement:** +0.5-1.5%. Finds wider optima that generalize better.
**Timing:** Start SWA at 75% of total training budget.

### 3E. Label Smoothing (Trivial, +0.5-1%)

```python
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimal values for medical imaging:
# 0.05-0.10 for clean labels (recommended start: 0.1)
# 0.15-0.20 for noisy labels
```

**Expected improvement:** +0.5-1%. More importantly, improves ensemble calibration.

### 3F. Focal Loss for Imbalanced Classes (Medium, +1-2% on minority classes)

```python
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma  # focusing parameter
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Calculate class weights from training set
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
alpha = torch.FloatTensor(weights).to(device)

criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.1)
```

**Gamma:** Start with 2.0. Higher gamma = more focus on hard examples.
**When to use:** Check class distribution first. If imbalanced (>3:1 ratio), use Focal Loss.

### 3G. Learning Rate Strategies

**OneCycleLR (RECOMMENDED):**

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=30,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,        # 30% warmup
    div_factor=25,        # start_lr = max_lr / 25
    final_div_factor=1e4  # end_lr = start_lr / 10000
)
# Call scheduler.step() after EVERY BATCH, not every epoch
```

**CosineAnnealingWarmRestarts (for SWA combo):**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,     # restart every 10 epochs
    T_mult=2,   # double restart period each time
    eta_min=1e-6
)
```

**Best combo for hackathon:** OneCycleLR for 80% of training, then switch to SWA.

### 3H. Knowledge Distillation (Hard, +1-2%)

Train your ensemble first, then distill into a single model:

```python
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=4.0, alpha=0.7):
    """Combine hard label loss with soft teacher loss."""
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss

# Teacher = your ensemble (freeze it)
# Student = single best model (retrain it)
# Temperature: 3-5 for medical images
# Alpha: 0.7 (more weight on teacher knowledge)
```

**Worth it for hackathon:** ONLY if you have time left. Ensemble submission is usually better.

---

## 4. POST-PROCESSING FOR SEGMENTATION

### 4A. Threshold Optimization (Trivial, +1-5% IoU)

The default threshold of 0.5 is almost never optimal.

```python
import numpy as np
from sklearn.metrics import jaccard_score

def optimize_threshold(val_preds, val_masks):
    """Search for best threshold on validation set."""
    best_iou = 0
    best_threshold = 0.5

    for threshold in np.arange(0.25, 0.75, 0.005):
        binary_preds = (val_preds > threshold).astype(np.uint8)
        iou = jaccard_score(
            val_masks.flatten(),
            binary_preds.flatten(),
            average='binary'
        )
        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold

    return best_threshold, best_iou

# Often the optimal threshold is 0.35-0.45 for medical segmentation
```

**Expected improvement:** +1-5% IoU. This is FREE performance.
**Must-do:** YES. Takes 5 minutes and can give significant gains.

### 4B. Morphological Post-Processing (Easy, +0.5-2% IoU)

```python
import cv2
import numpy as np

def postprocess_mask(binary_mask, min_area=50):
    """Clean up segmentation mask with morphological operations."""
    mask = binary_mask.astype(np.uint8)

    # Step 1: Remove small noise (opening = erosion + dilation)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Step 2: Fill small holes (closing = dilation + erosion)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Step 3: Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0

    return mask
```

**Expected improvement:** +0.5-2% IoU.
**Kernel sizes:** Start with 3x3 for opening, 5x5 for closing. Tune on validation.
**min_area:** Start with 50 pixels for 128x128 images.

### 4C. CRF Post-Processing (Hard, +1-3% IoU)

Conditional Random Fields refine boundaries using image intensity:

```python
# pip install pydensecrf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crf_postprocess(image, prob_map, n_iters=5):
    """Apply dense CRF to refine segmentation boundaries."""
    h, w = prob_map.shape
    prob_2class = np.stack([1 - prob_map, prob_map], axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(prob_2class)
    d.setUnaryEnergy(unary)

    d.addPairwiseBilateral(
        sxy=10, srgb=13,
        rgbim=image.astype(np.uint8),
        compat=4
    )
    d.addPairwiseGaussian(sxy=3, compat=3)

    Q = d.inference(n_iters)
    result = np.argmax(Q, axis=0).reshape((h, w))
    return result
```

**Expected improvement:** +1-3% IoU, especially at boundaries.
**Worth it:** Maybe. Only if model already has decent predictions but rough boundaries.

### 4D. Combined Post-Processing Pipeline

```python
def full_postprocess(raw_logits, image, threshold=None, val_preds=None, val_masks=None):
    """Complete post-processing pipeline."""
    if threshold is None and val_preds is not None:
        threshold, _ = optimize_threshold(val_preds, val_masks)
    elif threshold is None:
        threshold = 0.45

    prob_map = torch.sigmoid(raw_logits).cpu().numpy()
    binary = (prob_map > threshold).astype(np.uint8)
    cleaned = postprocess_mask(binary, min_area=30)
    return cleaned
```

---

## 5. MODEL INTERPRETABILITY (Presentation Bonus)

### 5A. GradCAM for Classification (Easy, HIGH IMPACT for judges)

```python
# pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_gradcam_visualization(model, input_tensor, original_image, target_layer):
    """Generate GradCAM heatmap overlay."""
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0))
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(
        original_image / 255.0,
        grayscale_cam,
        use_rgb=True
    )
    return visualization

# For each model architecture:
# EfficientNetV2-S: target_layer = model.features[-1]
# ConvNeXt-Tiny:    target_layer = model.features[-1][-1]
# Swin-Tiny:        target_layer = model.features[-1][-1].norm1

# Library: pip install grad-cam (pytorch-grad-cam)
```

**Presentation tip:** Show GradCAM for correct AND incorrect predictions. For incorrect ones, explain what the model focused on and why it was confused.

### 5B. Attention Map Visualization for Swin Transformer (Medium)

```python
def visualize_swin_attention(model, image):
    """Extract and visualize attention maps from Swin Transformer."""
    attention_maps = []

    def hook_fn(module, input, output):
        attention_maps.append(output.detach())

    hooks = []
    for name, module in model.named_modules():
        if 'attn' in name and hasattr(module, 'softmax'):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(image.unsqueeze(0))

    for h in hooks:
        h.remove()

    return attention_maps
```

### 5C. Presentation Strategy for Judges

Create a visual dashboard showing:

1. **Per-class GradCAM grid:** One row per class, showing 3 examples with heatmap overlays
2. **Confusion matrix with visual examples:** Where errors cluster, show the actual images
3. **Ensemble disagreement visualization:** Show cases where models disagree and the ensemble resolves it
4. **Segmentation overlay comparison:** Original | Ground truth | Model 1 | Model 2 | Ensemble

```python
import matplotlib.pyplot as plt

def create_presentation_figure(images, masks_gt, masks_pred, gradcams):
    """Create a professional comparison figure for judges."""
    fig, axes = plt.subplots(len(images), 4, figsize=(16, 4*len(images)))
    titles = ['Original', 'Ground Truth', 'Prediction', 'GradCAM']

    for i, (img, gt, pred, cam) in enumerate(
        zip(images, masks_gt, masks_pred, gradcams)
    ):
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 3].imshow(cam)

        for j, title in enumerate(titles):
            axes[i, j].set_title(title)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('presentation_figure.png', dpi=150, bbox_inches='tight')
```

**Impact:** This is what separates winning teams. Judges remember visual evidence.

---

## 6. PSEUDO-LABELING / SEMI-SUPERVISED

### 6A. Legality Assessment

**Your hackathon rules say:** "no using test images for training, no manual labeling."

**Pseudo-labeling on test data is RISKY and likely violates rules.** DO NOT use test data pseudo-labels.

**SAFE alternative -- Cross-validation pseudo-labeling on TRAIN data:**

```python
from sklearn.model_selection import KFold

def generate_cross_pseudo_labels(model_class, train_data, n_folds=5):
    """Generate pseudo-labels using cross-validation on training data only."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    pseudo_labels = np.zeros((len(train_data), 12))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        model = model_class()
        # Train on train_idx subset
        # Predict on val_idx subset (these are soft pseudo labels)
        pseudo_labels[val_idx] = model.predict_proba(train_data[val_idx])

    return pseudo_labels

# Use soft pseudo-labels as auxiliary training signal:
# loss = 0.7 * CE(output, hard_label) + 0.3 * KL(output, pseudo_soft_label)
```

### 6B. Noisy Student Approach (SAFE version)

Use your trained model to generate augmented training data with soft labels:

```python
def noisy_student_augment(teacher_model, train_images, train_labels):
    """Create augmented copies with teacher soft labels."""
    augmented_images = []
    soft_labels = []

    strong_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomErasing(p=0.3)
    ])

    teacher_model.eval()
    with torch.no_grad():
        for img, label in zip(train_images, train_labels):
            for _ in range(2):
                aug_img = strong_aug(img)
                soft_label = torch.softmax(
                    teacher_model(aug_img.unsqueeze(0)), dim=1
                )
                augmented_images.append(aug_img)
                soft_labels.append(soft_label)

    return augmented_images, soft_labels
```

**Expected improvement:** +0.5-1% accuracy.
**Worth it:** Only if you have spare time. Modest gain for the effort.

---

## 7. DATA-SPECIFIC TRICKS FOR HISTOPATHOLOGY

### 7A. Stain Normalization (RECOMMENDED -- Medium, +1-3%)

Histopathology images vary in staining intensity. Normalization makes the model robust:

```python
# pip install staintools
import staintools

def normalize_stain_macenko(image, reference_image):
    """Macenko stain normalization -- fast and effective."""
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(reference_image)
    normalized = normalizer.transform(image)
    return normalized

# Choose reference image: pick the median-stained image from training set
# Apply to ALL images (train + test) before training

# Alternative: Vahadane method (slower but better structure preservation)
# normalizer = staintools.StainNormalizer(method='vahadane')
```

**Expected improvement:** +1-3% accuracy AND makes model more robust.
**Best approach for hackathon:** Normalize all images ONCE and save to disk.
**Reference image selection:** Pick an image that looks average in staining.

### 7B. Color Augmentation for H&E Staining (Easy, +1-2%)

```python
import torchvision.transforms as T

histopath_augmentation = T.Compose([
    # Standard geometric (tissue is rotationally invariant)
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=90),

    # H&E-specific color augmentation
    T.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.04          # SMALL hue shift (H&E is pink-purple)
    ),

    # Simulate slight blur (out-of-focus regions)
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),

    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
```

**Key insight for H&E:** Keep hue jitter SMALL (0.03-0.05). Large hue shifts create unrealistic images.

### 7C. Advanced: HED Color Space Augmentation (Medium, +1-2%)

Augment in Hematoxylin-Eosin-DAB color space for more realistic variations:

```python
from skimage.color import rgb2hed, hed2rgb
import numpy as np

def augment_hed(image, alpha_range=(-0.2, 0.2), beta_range=(-0.05, 0.05)):
    """Augment in HED color space for histopathology-specific variation."""
    hed_image = rgb2hed(image)

    for channel in range(3):
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        hed_image[:, :, channel] = hed_image[:, :, channel] * (1 + alpha) + beta

    augmented = hed2rgb(hed_image)
    augmented = np.clip(augmented, 0, 1)
    return (augmented * 255).astype(np.uint8)
```

### 7D. Optimal Image Sizes

**Task 1 (Classification, 100x100 original):**
- Train at 100x100 first
- Try upsampling to 128x128 or 224x224 (pretrained model native size)
- 224x224 lets you use ImageNet pretrained weights most effectively
- Progressive resizing: 64 -> 100 -> 128

**Task 2 (Segmentation, 128x128 original):**
- 128x128 is already good for segmentation
- Try 256x256 if GPU memory allows (more spatial detail)
- U-Net architectures work well at power-of-2 sizes

---

## PRIORITY RANKING: What to Implement First

Given a 36-hour hackathon, here is the order of implementation:

### TIER 1: Do These First (highest return on time investment)

| Technique | Task | Time | Expected Gain |
|-----------|------|------|---------------|
| Label smoothing (0.1) | Classification | 5 min | +0.5-1% |
| Threshold optimization | Segmentation | 10 min | +1-5% IoU |
| TTA (8-fold flips+rotations) | Both | 30 min | +0.5-3% |
| H&E color augmentation | Both | 15 min | +1-2% |
| OneCycleLR scheduler | Both | 10 min | +0.5-1% |

### TIER 2: Do These Next (good return)

| Technique | Task | Time | Expected Gain |
|-----------|------|------|---------------|
| Weighted soft-voting ensemble | Classification | 1 hr | +1-3% |
| Segmentation ensemble (avg logits) | Segmentation | 1 hr | +1-3% IoU |
| CutMix augmentation | Classification | 30 min | +1-3% |
| Morphological post-processing | Segmentation | 30 min | +0.5-2% IoU |
| Progressive resizing | Both | 1 hr | +2-5% |
| SWA (last 5-10 epochs) | Both | 20 min | +0.5-1.5% |

### TIER 3: If Time Permits

| Technique | Task | Time | Expected Gain |
|-----------|------|------|---------------|
| Stain normalization (Macenko) | Both | 2 hr | +1-3% |
| Stacking meta-learner | Classification | 2 hr | +2-4% |
| Focal Loss (if imbalanced) | Classification | 30 min | +1-2% |
| GradCAM visualizations | Presentation | 1 hr | Judge bonus |
| CRF post-processing | Segmentation | 2 hr | +1-3% IoU |
| HED augmentation | Both | 1 hr | +1-2% |

### TIER 4: Luxury (only if everything else is done)

| Technique | Task | Time | Expected Gain |
|-----------|------|------|---------------|
| Knowledge distillation | Classification | 3 hr | +1-2% |
| Pseudo-labeling (safe version) | Both | 2 hr | +0.5-1% |
| Attention visualization | Presentation | 2 hr | Judge bonus |

---

## QUICK REFERENCE: Library Requirements

```
pip install torch torchvision
pip install timm                    # pretrained models
pip install segmentation-models-pytorch  # segmentation architectures
pip install albumentations          # advanced augmentations
pip install grad-cam                # GradCAM visualization
pip install staintools              # stain normalization
pip install pydensecrf              # CRF post-processing
pip install scikit-image            # HED color space
pip install scipy                   # weight optimization
pip install matplotlib              # visualization
```

---

## CUMULATIVE GAIN ESTIMATES

**Classification (Task 1):**
- Baseline single model: ~85%
- + Label smoothing + OneCycleLR: ~87%
- + CutMix + H&E augmentation: ~89%
- + Progressive resizing + SWA: ~91%
- + 3-model ensemble + TTA: ~93-95%

**Segmentation (Task 2):**
- Baseline single model: ~75% IoU
- + Threshold optimization: ~78% IoU
- + Morphological post-processing: ~79% IoU
- + 2-model ensemble + TTA: ~82-84% IoU
- + Stain normalization + CRF: ~85-87% IoU

Note: These are rough estimates. Actual gains depend on your specific dataset and baseline model quality. Gains are NOT additive -- they compound with diminishing returns.
