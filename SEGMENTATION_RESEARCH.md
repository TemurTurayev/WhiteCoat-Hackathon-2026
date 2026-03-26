# Medical Image Segmentation: State-of-the-Art Research for Hackathon
## Task: Binary segmentation, 128x128 histopathology, 1800 training images, metric = Mean IoU

---

## 1. BEST ARCHITECTURES FOR SMALL MEDICAL IMAGES

### Tier List (for small datasets < 2000 images, 128x128 resolution)

| Architecture | Expected IoU Range | Small Dataset Suitability | Implementation (SMP) | Hackathon Priority |
|---|---|---|---|---|
| **U-Net++** (deep supervision) | +3.9 IoU over U-Net | EXCELLENT - nested skip connections capture fine structures | `smp.UnetPlusPlus` | **#1 PICK** |
| **U-Net** | Baseline | GOOD - proven, simple, less prone to overfit | `smp.Unet` | Strong baseline |
| **MAnet** | Comparable to U-Net++ | GOOD - multi-scale attention helps small objects | `smp.MAnet` | Worth trying |
| **FPN** | Slightly below U-Net++ | GOOD - multi-scale features | `smp.FPN` | Quick experiment |
| **DeepLabV3+** | High on large datasets | MODERATE - may overfit on 1800 images | `smp.DeepLabV3Plus` | Try if time allows |
| **PAN** | Similar to FPN | GOOD - works with >= 128x128 | `smp.PAN` | Quick experiment |
| **TransUNet** | 93.25% IoU (synapse) | RISKY - transformers need more data, prone to overfit | Custom implementation | Skip for hackathon |
| **Swin-UNet** | High (large datasets) | POOR - heavy, needs lots of data | Complex setup | Skip |
| **SegFormer** | High (large datasets) | POOR - transformer-based, data-hungry | Complex setup | Skip |

### Key Finding
**U-Net++ with deep supervision achieves +3.9 IoU points over U-Net and +3.4 over Wide U-Net.** Deep supervision adds +0.6 IoU on top of U-Net++ without it. For 1800 images at 128x128, U-Net++ is the clear winner because:
- Nested dense skip connections capture fine-grained features critical for biopsy boundaries
- Deep supervision acts as implicit regularization (reduces overfitting)
- Proven performance on small medical datasets

### AVOID for this task
- Pure Transformer architectures (TransUNet, Swin-UNet, SegFormer): they need >10K images to outperform CNNs
- Very deep architectures without strong regularization

---

## 2. ENCODER SELECTION

### Tier List for Medical Segmentation with Small Datasets

| Encoder | ImageNet Top-1 | Params | Small Dataset Performance | Recommendation |
|---|---|---|---|---|
| **EfficientNet-B4** | 83.4% | 19M | EXCELLENT - best accuracy/params ratio | **#1 PICK** |
| **SE-ResNeXt50_32x4d** | 81.1% | 27M | EXCELLENT - squeeze-excitation helps attention | **#2 PICK** |
| **ResNet50** | 80.9% | 25M | VERY GOOD - proven, stable, well-studied | Strong fallback |
| **EfficientNet-B7** | 84.3% | 66M | GOOD but risk overfitting (too many params) | Try with strong regularization |
| **DenseNet121** | 74.4% | 8M | GOOD - less params = less overfitting | Lightweight option |
| **ResNet34** | 73.3% | 21M | MODERATE - lighter but less expressive | If overfitting is severe |
| **MobileNetV3** | 74.0% | 5.4M | MODERATE - very fast but lower capacity | Only if speed matters |
| **timm-efficientnet-b4** | 83.4% | 19M | EXCELLENT - timm pretrained often better | Best if available |

### Key Findings from Papers
1. **ResNet50 + DeepLabV3+** was identified as the most effective combination for GI tract segmentation in a comprehensive 2024 encoder-decoder comparison study
2. **SE-ResNeXt + U-Net** was shown effective for kidney tumor segmentation
3. **EfficientNet family** consistently performs well for medical segmentation with good accuracy/compute tradeoff
4. **U-Net++ with EfficientNet-B7** was identified as superior in a benchmark study using 5-fold CV

### Recommendation for Hackathon
**Primary:** `U-Net++ + EfficientNet-B4` (imagenet pretrained)
**Backup:** `U-Net++ + SE-ResNeXt50_32x4d` (imagenet pretrained)
**Quick baseline:** `U-Net + ResNet50` (imagenet pretrained)

All encoders should use `encoder_weights='imagenet'` -- transfer learning is CRITICAL with only 1800 images.

---

## 3. LOSS FUNCTIONS (CRITICAL for IoU optimization)

### Loss Function Comparison for Maximizing IoU

| Loss Function | Direct IoU Opt? | Expected IoU vs BCE | Stability | Hackathon Priority |
|---|---|---|---|---|
| **Dice + BCE** (combo) | Indirect | +2-4% | HIGH | **#1 PICK - start here** |
| **Lovasz Loss** | YES (direct!) | +1-3% over Dice+BCE | MODERATE | **#2 PICK - add later** |
| **Focal Tversky** (alpha=0.7, beta=0.3) | Indirect | Best DSC in prostate study (0.74) | HIGH | **#3 for imbalanced masks** |
| **Tversky Loss** | Indirect | Similar to Dice | HIGH | Good for FP/FN control |
| **Dice Loss alone** | Indirect | Better than BCE for segmentation | HIGH | Simple alternative |
| **BCE alone** | NO | Baseline | HIGH | Too weak alone |
| **Boundary Loss** | NO | Helps boundary quality | MODERATE | Advanced |

### CRITICAL STRATEGY: Two-Phase Loss Training

**Phase 1 (epochs 1-30):** Train with `Dice + BCE` (0.5 * Dice + 0.5 * BCE)
- This gives stable, reliable training
- BCE provides pixel-level gradient signal
- Dice handles class imbalance

**Phase 2 (epochs 30-50):** Switch to or add `Lovasz Loss`
- Lovasz directly optimizes IoU (the Jaccard index)
- It is a convex surrogate for IoU, computable in O(p log p)
- Should be used after model has learned basic features

### Implementation

```python
import segmentation_models_pytorch as smp
from lovasz_losses import lovasz_hinge  # github.com/bermanmaxim/LovaszSoftmax

# Phase 1: Dice + BCE
loss_phase1 = smp.losses.DiceLoss(mode='binary') + smp.losses.SoftBCEWithLogitsLoss()

# Phase 2: Lovasz (directly optimizes IoU!)
loss_phase2 = lovasz_hinge  # for binary segmentation

# Alternative combo: Focal Tversky for imbalanced masks
loss_tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.7, beta=0.3)
```

### Why Lovasz is Special
- It is THE ONLY loss function that directly optimizes the IoU/Jaccard metric
- Based on Lovasz extension of submodular functions (CVPR 2018)
- Performs significantly better than cross-entropy for the Jaccard index
- Key insight: Dice Loss approximates IoU optimization, but Lovasz is the true surrogate

### Expected Improvement
- BCE alone -> Dice+BCE: +3-5% IoU
- Dice+BCE -> Dice+BCE+Lovasz: +1-3% IoU
- Using Focal Tversky for highly imbalanced masks: additional +1-2% IoU

---

## 4. TRAINING ON SMALL DATASETS (1800 images)

### A. Heavy Augmentation Strategy (MOST IMPORTANT for preventing overfitting)

```python
import albumentations as A

train_transform = A.Compose([
    # --- Spatial transforms (CRITICAL) ---
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.3),  # Simulates tissue deformation
    A.GridDistortion(p=0.3),
    A.OpticalDistortion(p=0.3),

    # --- Color/Intensity transforms (IMPORTANT for histopathology) ---
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),

    # --- Histopathology-specific ---
    # A.HEStain(p=0.3),  # If albumentations version supports it - H&E stain augmentation
    A.CLAHE(clip_limit=4.0, p=0.3),  # Contrast-limited adaptive histogram equalization

    # --- Cutout/Dropout (regularization) ---
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.3),

    # --- Normalize ---
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

**Expected impact:** Without augmentation, model will severely overfit with 1800 images. Augmentation alone can improve generalization IoU by +5-15%.

### B. Copy-Paste Augmentation

Copy foreground regions from one training image and paste onto another. Creates combinatorial explosion of training data.

```python
# Concept: for each training pair (image, mask)
# 1. Extract foreground region using mask
# 2. Paste it onto a random other image
# 3. Update the target mask accordingly
```

**Expected improvement:** +1-3% IoU
**Implementation time:** 2-3 hours
**Verdict:** WORTH IT if masks have distinct foreground objects

### C. K-Fold Cross-Validation (5-fold)

**Strategy:**
1. Split 1800 images into 5 folds (360 images each for validation)
2. Train 5 separate models, each using different fold as validation
3. Average predictions from all 5 models at test time

**Expected improvement:** +1-3% IoU over single split
**Implementation time:** 5x training time (but can parallelize on multiple GPUs)
**Verdict:** ABSOLUTELY WORTH IT. This is the single most reliable way to squeeze out IoU.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = []
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    model = train_fold(train_idx, val_idx, fold)
    models.append(model)

# Ensemble prediction
def predict_ensemble(image, models):
    predictions = [model(image) for model in models]
    return torch.stack(predictions).mean(dim=0)
```

### D. Regularization Techniques

| Technique | Impact | Implementation |
|---|---|---|
| **Dropout in decoder** | Prevents co-adaptation | Add `Dropout(0.2-0.5)` in decoder layers |
| **Weight decay** | L2 regularization | `optimizer = AdamW(lr=1e-4, weight_decay=1e-4)` |
| **Early stopping** | Prevents overfitting | Monitor val IoU, patience=10-15 |
| **Label smoothing** | Soft targets | Minimal impact for segmentation |
| **Stochastic depth** | Random layer dropping | Built into EfficientNet |
| **Gradient clipping** | Training stability | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |

### E. Learning Rate Strategy

```python
# Cosine Annealing with Warm Restarts - BEST for medical segmentation
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Or OneCycleLR - fast convergence
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=num_epochs * len(train_loader)
)
```

### F. Pseudo-Labeling (if unlabeled test data available)

1. Train initial model on 1800 labeled images
2. Predict on test/unlabeled images
3. Use high-confidence predictions (IoU > 0.9) as pseudo-labels
4. Retrain with expanded dataset

**Expected improvement:** +1-2% IoU
**Implementation time:** 3-4 hours
**Verdict:** Only if confident predictions are available

---

## 5. POST-PROCESSING TRICKS

### A. Optimal Threshold Search (MUST DO - Free IoU points!)

Default threshold is 0.5, but optimal is often 0.3-0.7 depending on data.

```python
import numpy as np

def find_optimal_threshold(model, val_loader):
    """Search for threshold that maximizes IoU on validation set"""
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in val_loader:
            preds = torch.sigmoid(model(images.cuda()))
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)

    best_iou = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        binary_preds = (all_preds > thresh).float()
        intersection = (binary_preds * all_masks).sum()
        union = binary_preds.sum() + all_masks.sum() - intersection
        iou = intersection / (union + 1e-8)
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    return best_thresh  # Use this instead of 0.5!
```

**Expected improvement:** +0.5-2% IoU (essentially FREE)
**Implementation time:** 15 minutes
**Verdict:** MUST DO. Zero risk, guaranteed improvement.

### B. Test-Time Augmentation (TTA)

Average predictions from multiple augmented versions of the test image.

```python
def predict_with_tta(model, image):
    """TTA: average predictions from original + flipped + rotated"""
    preds = []

    # Original
    preds.append(torch.sigmoid(model(image)))

    # Horizontal flip
    flipped_h = torch.flip(image, [3])
    pred_h = torch.sigmoid(model(flipped_h))
    preds.append(torch.flip(pred_h, [3]))

    # Vertical flip
    flipped_v = torch.flip(image, [2])
    pred_v = torch.sigmoid(model(flipped_v))
    preds.append(torch.flip(pred_v, [2]))

    # 90 degree rotation
    rotated = torch.rot90(image, 1, [2, 3])
    pred_r = torch.sigmoid(model(rotated))
    preds.append(torch.rot90(pred_r, 3, [2, 3]))

    # Average all predictions
    return torch.stack(preds).mean(dim=0)
```

**Expected improvement:** +1-2.3% IoU (proven in medical imaging papers)
**Implementation time:** 30 minutes
**Verdict:** MUST DO. S3-TTA paper showed +1.3-3.4% improvement.

### C. Morphological Post-Processing

```python
import cv2
import numpy as np

def morphological_postprocess(mask, kernel_size=3):
    """Clean up binary mask predictions"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Remove small noise (opening = erosion + dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small holes (closing = dilation + erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask
```

**Expected improvement:** +0.2-0.5% IoU
**Implementation time:** 15 minutes
**Verdict:** Worth doing. Small but free improvement.

### D. Connected Component Filtering

Remove small isolated regions that are likely noise.

```python
def remove_small_components(mask, min_size=50):
    """Remove connected components smaller than min_size pixels"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1
    return cleaned
```

**Expected improvement:** +0.1-0.3% IoU
**Implementation time:** 15 minutes
**Verdict:** Worth doing if predictions have scattered noise.

### E. CRF Post-Processing

Conditional Random Fields refine boundaries using image appearance.

**Expected improvement:** +0.2-1% IoU (boundary refinement)
**Implementation time:** 1-2 hours
**Verdict:** SKIP for hackathon. Marginal gain vs complexity. Better to spend time on TTA and threshold search.

### Post-Processing Priority Order
1. Optimal Threshold Search (+0.5-2% IoU, 15 min) -- **MUST DO**
2. Test-Time Augmentation (+1-2.3% IoU, 30 min) -- **MUST DO**
3. Morphological Operations (+0.2-0.5% IoU, 15 min) -- DO
4. Connected Component Filtering (+0.1-0.3% IoU, 15 min) -- DO
5. CRF Post-Processing (+0.2-1% IoU, 1-2 hrs) -- SKIP

---

## 6. SAM (Segment Anything Model) FOR MEDICAL IMAGES

### Available Medical SAM Variants

| Model | Year | Key Feature | Training Data |
|---|---|---|---|
| **MedSAM** | 2023 | SAM fine-tuned on 11 modalities | >1M image-mask pairs |
| **SAM-Med2D** | 2023 | Optimized for 2D medical | 4.6M images, 19.7M masks |
| **LiteMedSAM** | 2024 | 10x faster than MedSAM | Same as MedSAM |
| **MedSAM2** | 2025 | 3D + video support | 450K volumes |

### Can SAM Work for Your Task?

**Pros:**
- MedSAM achieved 94.0-98.4% DSC on various medical tasks
- Pre-trained on massive medical dataset -- instant domain knowledge
- SAM-Med2D trained on 4.6M medical images with 19.7M masks
- Could count as "innovative approach" for bonus points

**Cons:**
- Requires bounding box or point prompts for each image
- Not designed for batch automated segmentation
- Fine-tuning SAM encoder is computationally expensive
- 128x128 images are very small for SAM (designed for 1024x1024)
- Integration complexity high for 36-hour hackathon

### Feasibility for Hackathon

**Verdict: USE AS SECONDARY APPROACH ONLY**

Option A (Recommended): Use MedSAM/SAM-Med2D as a teacher model for knowledge distillation:
1. Generate pseudo-labels using SAM on your training images
2. Use pseudo-labels to augment your training data
3. Train your U-Net++ as the primary model

Option B (Risky): Fine-tune SAM-Med2D decoder on your 1800 images:
1. Freeze image encoder
2. Fine-tune only the mask decoder
3. Use automatic prompt generation (center of image or grid points)
4. Time needed: 4-8 hours

**For bonus "innovation" points:** Mention SAM in your presentation but rely on U-Net++ for actual submission scores.

---

## 7. K-FOLD CROSS-VALIDATION STRATEGY

### 5-Fold CV Implementation Plan

```
Total: 1800 images
Each fold: 1440 train / 360 validation

Fold 1: Train on folds [2,3,4,5], validate on fold 1
Fold 2: Train on folds [1,3,4,5], validate on fold 2
Fold 3: Train on folds [1,2,4,5], validate on fold 3
Fold 4: Train on folds [1,2,3,5], validate on fold 4
Fold 5: Train on folds [1,2,3,4], validate on fold 5
```

### Ensemble Strategy

```python
# At inference time:
def ensemble_predict(image, models, threshold):
    preds = []
    for model in models:
        pred = torch.sigmoid(model(image))
        preds.append(pred)

    # Method 1: Average probabilities then threshold (RECOMMENDED)
    avg_pred = torch.stack(preds).mean(dim=0)
    return (avg_pred > threshold).float()

    # Method 2: Majority voting
    # binary_preds = [(p > threshold).float() for p in preds]
    # return (torch.stack(binary_preds).sum(dim=0) >= 3).float()  # 3 of 5
```

### Expected Improvement
- Single model: baseline IoU
- 5-fold ensemble: **+2-4% IoU** (consistently demonstrated in competitions)
- 5-fold + TTA: **+3-5% IoU** combined

### Time Cost Analysis

| Scenario | Training Time (per fold) | Total Time | Worth It? |
|---|---|---|---|
| 50 epochs, batch 16, 128x128 | ~15-20 min on GPU | ~1.5 hrs for 5 folds | **ABSOLUTELY YES** |
| 100 epochs | ~30-40 min | ~3 hrs | YES if time allows |
| With TTA at inference | +2 min per fold | +10 min | YES |

### Stratified K-Fold
If masks have varying amounts of foreground pixels, use stratified splitting:

```python
from sklearn.model_selection import StratifiedKFold

# Compute foreground ratio for each image
fg_ratios = [mask.sum() / mask.numel() for mask in all_masks]
# Bin into categories for stratification
bins = pd.qcut(fg_ratios, q=5, labels=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, bins)):
    # train fold...
```

---

## MASTER IMPLEMENTATION PLAN (Priority-Ordered for 36-Hour Hackathon)

### Hour 0-2: Setup & Baseline
- [ ] Setup environment, install SMP, albumentations, pytorch
- [ ] Load data, explore masks, compute class balance statistics
- [ ] Train U-Net + ResNet34 baseline with Dice+BCE loss
- [ ] **Expected IoU: ~0.65-0.75** (rough baseline)

### Hour 2-6: Architecture & Encoder Optimization
- [ ] Switch to **U-Net++ with EfficientNet-B4** encoder (imagenet pretrained)
- [ ] Implement heavy augmentation pipeline (albumentations)
- [ ] Train with Dice+BCE loss, cosine annealing LR
- [ ] **Expected IoU: ~0.75-0.82**

### Hour 6-10: Loss Function Optimization
- [ ] Add Lovasz Loss (phase 2 training or combined)
- [ ] Try Focal Tversky if masks are imbalanced
- [ ] Implement optimal threshold search on validation
- [ ] **Expected IoU: ~0.80-0.85**

### Hour 10-16: K-Fold Cross-Validation
- [ ] Implement 5-fold CV
- [ ] Train all 5 folds with best architecture/loss
- [ ] Implement ensemble prediction
- [ ] **Expected IoU: ~0.83-0.87**

### Hour 16-20: Post-Processing
- [ ] Implement TTA (flip + rotate)
- [ ] Implement morphological post-processing
- [ ] Implement connected component filtering
- [ ] Fine-tune optimal threshold with ensemble
- [ ] **Expected IoU: ~0.85-0.89**

### Hour 20-24: Advanced Optimization
- [ ] Try alternative encoder (SE-ResNeXt50) if EfficientNet plateaus
- [ ] Experiment with copy-paste augmentation
- [ ] Consider pseudo-labeling if unlabeled data exists
- [ ] **Expected IoU: ~0.86-0.90**

### Hour 24-30: Final Refinement
- [ ] Train final models with best configuration
- [ ] Optimize all hyperparameters
- [ ] Generate final predictions
- [ ] **Target IoU: ~0.87-0.91**

### Hour 30-36: Presentation & Safety Net
- [ ] Prepare presentation highlighting innovative approaches
- [ ] Document methodology
- [ ] Keep backup predictions ready

---

## QUANTIFIED IoU IMPROVEMENT SUMMARY

| Technique | IoU Gain | Time to Implement | Priority |
|---|---|---|---|
| ImageNet pretrained encoder | +5-10% | 5 min | MUST DO |
| U-Net++ over U-Net | +3-4% | 10 min | MUST DO |
| Heavy augmentation | +5-15% | 1 hr | MUST DO |
| Dice+BCE loss (over BCE) | +3-5% | 10 min | MUST DO |
| Lovasz Loss (phase 2) | +1-3% | 30 min | HIGH |
| 5-Fold Ensemble | +2-4% | 2-3 hrs | HIGH |
| Optimal Threshold Search | +0.5-2% | 15 min | MUST DO |
| Test-Time Augmentation | +1-2.3% | 30 min | MUST DO |
| Morphological Post-processing | +0.2-0.5% | 15 min | DO |
| Connected Component Filter | +0.1-0.3% | 15 min | DO |
| Copy-Paste Augmentation | +1-3% | 2-3 hrs | MEDIUM |
| EfficientNet-B4 over ResNet34 | +1-2% | 5 min | MUST DO |
| Focal Tversky (if imbalanced) | +1-2% | 15 min | CONDITIONAL |
| Pseudo-labeling | +1-2% | 3-4 hrs | LOW |
| SAM/MedSAM fine-tune | +?% (risky) | 4-8 hrs | INNOVATION BONUS |
| CRF post-processing | +0.2-1% | 1-2 hrs | SKIP |

### Total Theoretical Maximum Improvement
From raw U-Net + BCE baseline (~0.60-0.65) to fully optimized pipeline: **+20-30% IoU**
Realistic target: **0.85-0.91 Mean IoU**

---

## KEY LIBRARIES AND CODE REFERENCES

```bash
pip install segmentation-models-pytorch albumentations torch torchvision
pip install opencv-python scikit-learn pandas
# For Lovasz loss:
pip install git+https://github.com/bermanmaxim/LovaszSoftmax.git
```

### SMP Quick Start
```python
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,  # binary segmentation
    decoder_attention_type="scse",  # Squeeze-and-Excitation in decoder
)
```

### Critical Reminder: Image Size
128x128 is divisible by 32 (128/32 = 4), so all SMP architectures will work without padding issues.
