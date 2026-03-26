# Skin Lesion Classification & Segmentation Research
## For WhiteCoat.dev Hackathon — March 2026

**KEY FINDING**: Our dataset is DERMATOLOGY / SKIN LESION images (clinical/dermoscopic photographs), NOT histopathology. 100x100px RGB, 12 classes.

---

## 1. BEST MODELS FOR SKIN LESION CLASSIFICATION (2024-2025)

### ISIC 2019 Challenge Winners
- **1st Place (both tasks)**: Ensemble of **EfficientNets + SENet + ResNeXt WSL**
  - Team DAISYLab (Nils Gessert et al.)
  - GitHub: https://github.com/ngessert/isic2019
  - Paper: https://doi.org/10.1016/j.mex.2020.100864
  - Key: multi-resolution cropping, patient metadata, loss balancing

### Architecture Rankings (2024-2025 literature)

| Architecture | ISIC Performance | Notes |
|---|---|---|
| **ConvNeXt-Large** | 97.2% accuracy | Best single model in recent ensemble study |
| **ConvNeXt ensemble** | 97.7% accuracy | State-of-the-art ensemble |
| **GlobalSkinNet (ViT+CNN)** | 98% accuracy on ISIC-2019 | Hybrid transformer approach |
| **DermViT** | 96.29% MAUC (ISIC-2018) | Outperforms ResNet, ViT, SwinT, ConvNeXt |
| **Swin Transformer** | Superior to all CNN variants | Best transformer for dermoscopy |
| **EfficientNet-B4/B7** | ~93-95% | Solid baseline, good cost/performance |
| **SkinEHDLF (ConvNeXt+EfficientNetV2+Swin)** | Top tier | Adaptive attention fusion |

### Recommendation for Our Hackathon (100x100px, 12 classes)
Given our small image size (100x100), avoid huge models. Best choices:
1. **EfficientNet-B4** (already chosen, good call) — proven on ISIC
2. **ConvNeXt-Tiny/Small** — modern CNN, may outperform EfficientNet
3. **EfficientNet-B4 + ConvNeXt-Small ensemble** — if time allows

### Best Published Accuracy on ISIC 2019 (8 classes)
- Single model: ~93-95% (EfficientNet-B7 or ConvNeXt-Large)
- Ensemble: 97.7% (ConvNeXt ensemble)
- Hybrid: 98% (GlobalSkinNet)

### For 12+ Class Datasets
Limited published work on 12-class skin lesion datasets. Our hackathon dataset appears to be custom. Strategy: pretrain on ISIC 2019 (8 classes), then fine-tune on our 12 classes.

---

## 2. ISIC DATASET DETAILS

### ISIC 2019
- **Images**: 25,331 dermoscopic images
- **Classes (8)**: MEL (melanoma), NV (melanocytic nevus), BCC (basal cell carcinoma), AK (actinic keratosis), BKL (benign keratosis), DF (dermatofibroma), VASC (vascular lesion), SCC (squamous cell carcinoma)
- **Source**: Aggregate of BCN_20000 + HAM10000 + MSK Dataset
- **License**: CC-BY-NC 4.0
- **Download size**: ~9GB (full images)
- **Official page**: https://challenge.isic-archive.com/landing/2019/
- **Kaggle links**:
  - https://www.kaggle.com/datasets/andrewmvd/isic-2019
  - https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification
- **HuggingFace**: https://huggingface.co/datasets/flwrlabs/fed-isic2019

### ISIC 2018 (Segmentation)
- **Task 1 (Segmentation)**: 2,594 training images + binary masks, 100 validation, 1,000 test
- **Kaggle**: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation
- **Official**: https://challenge.isic-archive.com/landing/2018/

### HAM10000
- **Images**: 10,015 dermoscopic images
- **Classes (7)**: Same as ISIC 2019 minus SCC
- **Part of ISIC 2019** (subset)

### Download Commands
```bash
# Install Kaggle CLI
pip install kaggle

# Make sure ~/.kaggle/kaggle.json exists with your API key

# ISIC 2019 Classification
kaggle datasets download -d andrewmvd/isic-2019
# or
kaggle datasets download -d salviohexia/isic-2019-skin-lesion-images-for-classification

# ISIC 2018 Segmentation
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation

# Unzip
unzip isic-2019.zip -d isic2019/
unzip isic2018-challenge-task1-data-segmentation.zip -d isic2018_seg/
```

### Transfer Learning Pipeline: ImageNet -> ISIC -> Hackathon

```python
import timm
import torch
import torch.nn as nn

# STEP 1: Load ImageNet-pretrained model
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=8)

# STEP 2: Fine-tune on ISIC 2019 (8 classes)
# Train for ~20-30 epochs on ISIC 2019 with:
#   - lr=1e-4 for backbone, 1e-3 for head
#   - weighted CrossEntropy for class imbalance
#   - resize images to 224x224 or 380x380
# Save as: isic2019_efficientnet_b4.pth

# STEP 3: Load ISIC-pretrained, replace head for 12 classes
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=8)
model.load_state_dict(torch.load('isic2019_efficientnet_b4.pth'))
model.classifier = nn.Linear(model.classifier.in_features, 12)

# STEP 4: Fine-tune on hackathon data (12 classes)
# Lower learning rate (1e-5 for backbone, 1e-4 for head)
# Use all augmentations
# Train for 30-50 epochs
```

---

## 3. SKIN LESION SPECIFIC AUGMENTATIONS

### What WORKS for Dermoscopic Images

**Geometric (essential)**:
- RandomRotate90 — dermoscopic images have no canonical orientation
- HorizontalFlip, VerticalFlip — lesions are orientation-invariant
- ShiftScaleRotate — small shifts and scales
- ElasticTransform — subtle, mimics skin deformation

**Color (critical for dermoscopy)**:
- HueSaturationValue — accounts for skin tone variation
- ColorJitter / RandomBrightnessContrast — lighting differences
- **Color constancy preprocessing** — Shades of Gray algorithm improved sensitivity from 71% to 79.7% and specificity from 55.2% to 76%
- RGBShift — slight color shifts for robustness

**Dermoscopy-specific**:
- **Hair artifact simulation** — draw random thin dark lines across image
- **Microscope circle cropping** — random circular crop to simulate dermoscope field of view
- Coarse dropout / cutout — mimics occlusion by bubbles, rulers, markers

**Advanced**:
- Mixup (alpha=0.2-0.4) — proven effective for skin lesion classification
- CutMix — region mixing between classes
- **GAN-based augmentation** — 93.12% accuracy (vs lower without), generates realistic synthetic lesions

### What HURTS Performance
- **Excessive geometric distortion** — too much elastic transform destroys lesion morphology
- **Heavy blur** — dermoscopic features are fine-grained, blur destroys diagnostic info
- **Extreme color shifts** — unrealistic colors confuse the model
- **GridDistortion at high magnitude** — distorts lesion border patterns
- **CLAHE at high clip** — can amplify noise in dermoscopic images

### Recommended Augmentation Pipeline

```python
import albumentations as A

train_transform = A.Compose([
    A.Resize(224, 224),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.15,
        rotate_limit=45, p=0.5
    ),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
    ], p=0.3),
    A.OneOf([
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30,
            val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.5),
    ], p=0.5),
    A.CoarseDropout(
        max_holes=8, max_height=16, max_width=16,
        min_holes=1, p=0.3
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    A.pytorch.transforms.ToTensorV2(),
])
```

### Hair Artifact Simulation (Custom)

```python
import cv2
import numpy as np

def add_hair_artifacts(image, num_hairs=5, p=0.3):
    """Simulate hair artifacts on dermoscopic images."""
    if np.random.random() > p:
        return image
    result = image.copy()
    h, w = result.shape[:2]
    for _ in range(np.random.randint(1, num_hairs + 1)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        thickness = np.random.randint(1, 3)
        color = tuple(np.random.randint(0, 50, 3).tolist())  # dark hair
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result
```

---

## 4. SKIN LESION SEGMENTATION

### Best Approaches (ISIC 2018 Challenge)
- **ISIC 2018 Winner**: Dice=0.898, Jaccard=0.838, Accuracy=0.942
- **Attention Squeeze U-Net**: Dice=0.9035 (beat ISIC 2017 winner)
- **U-Net + VGG16 encoder**: 97.59% accuracy, 89.12% Jaccard, 94.24% Dice
- **Fusion models**: Dice=0.92 on ISIC 2018

### Architecture Recommendations

| Model | Dice Score | Notes |
|---|---|---|
| **U-Net++ (EfficientNet-B4 encoder)** | ~0.89-0.91 | Our current choice, excellent |
| **Attention U-Net** | ~0.90 | Attention on skip connections |
| **TransUNet (hybrid)** | ~0.90-0.92 | Transformer + U-Net |
| **MUCM-Net (Mamba-based)** | 0.89 | Lightweight, new approach |
| **DeepLabV3+** | ~0.88-0.90 | Good alternative |

### Our setup (U-Net++ with EfficientNet-B4) is already competitive.

### Optimal Image Size for Skin Lesion Segmentation
- **256x256**: Good balance of speed and accuracy
- **384x384**: Better for detailed boundaries
- **512x512**: Best accuracy but slower, used by challenge winners
- For our 100x100 input: upscale to **256x256** minimum

### Post-Processing for Skin Lesion Masks

```python
import cv2
import numpy as np
from scipy import ndimage

def postprocess_mask(mask, threshold=0.5):
    """Post-process skin lesion segmentation mask."""
    # 1. Threshold
    binary = (mask > threshold).astype(np.uint8)

    # 2. Remove small components (noise)
    labeled, num_features = ndimage.label(binary)
    if num_features > 1:
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        binary = (labeled == largest).astype(np.uint8)

    # 3. Fill holes
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8)

    # 4. Smooth boundaries with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. Optional: Gaussian blur + re-threshold for smoother edges
    smoothed = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)
    binary = (smoothed > 0.5).astype(np.uint8)

    return binary * 255  # Return as 0/255 PNG
```

---

## 5. DEALING WITH CLASS IMBALANCE IN SKIN LESIONS

### Our Dataset Imbalance
```
Class 7:  2136 (largest) — 6.5x of smallest
Class 8:   331 (smallest)
Ratio: 6.5:1
```

### What Works Best (ISIC Competition Winners)

**Tier 1 — Most effective:**
1. **Focal Loss** — improved accuracy from 74% to 89% in published study
   - Best technique for overfitting from imbalance
   - Focuses on hard/misclassified examples
2. **Weighted CrossEntropy** — weights inversely proportional to class frequency
3. **Oversampling minority classes** — simple and effective

**Tier 2 — Advanced:**
4. **Augmented data + Focal Loss** — 98.85% accuracy, 95.52% precision
5. **SMOTE + Tomek Links** — hybrid: oversample minority + clean boundaries
6. **Class-balanced sampling** — each batch has equal class representation

### Recommended Strategy

```python
import torch
import torch.nn as nn
import numpy as np

# Class counts from our dataset
class_counts = [571, 974, 1043, 750, 814, 441, 545, 2136, 331, 1111, 899, 1796]
total = sum(class_counts)

# Option 1: Weighted CrossEntropy
weights = [total / (12 * c) for c in class_counts]
weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)

# Option 2: Focal Loss (RECOMMENDED)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets)
            focal_loss = at * focal_loss
        return focal_loss.mean()

# Use with class weights
focal_criterion = FocalLoss(alpha=weights, gamma=2.0)

# Option 3: Weighted Random Sampler for balanced batches
from torch.utils.data import WeightedRandomSampler

sample_weights = [1.0 / class_counts[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(all_labels))
```

### Combined Strategy (Best for Competition)
Use ALL of these together:
1. WeightedRandomSampler for balanced batches
2. Focal Loss (gamma=2.0) with class weights
3. Heavy augmentation on minority classes
4. Test-Time Augmentation (TTA) with 4-8 flips/rotations

---

## 6. PRETRAINED MODELS ON SKIN/DERMOSCOPY

### PanDerm — Best Dermoscopy Foundation Model (2025)
- **What**: Vision foundation model pretrained on **2 million dermatological images**
- **Modalities**: Dermoscopy, clinical photos, TBP, dermatopathology
- **Performance**: Outperforms clinicians in early melanoma detection
- **GitHub**: https://github.com/SiyuanYan1/PanDerm
- **Models**:
  - `PanDerm_Large_LP` (ViT-Large) — best accuracy
  - `PanDerm_Base_LP` (ViT-Base) — smaller, faster
- **Checkpoints**:
  - `panderm_ll_data6_checkpoint-499.pth` (Large)
  - `panderm_bb_data6_checkpoint-499.pth` (Base)
- **HuggingFace**: https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256

### Using PanDerm for Our Task

```python
# Clone PanDerm repo
# git clone https://github.com/SiyuanYan1/PanDerm.git

# Load PanDerm pretrained encoder
import timm

# PanDerm is based on ViT architecture
# After downloading checkpoint:
model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=12)
checkpoint = torch.load('panderm_ll_data6_checkpoint-499.pth')
# Load pretrained weights (may need key mapping)
model.load_state_dict(checkpoint, strict=False)
```

### Derm1M — Million-Scale Vision-Language Dataset
- **GitHub**: https://github.com/SiyuanYan1/Derm1M
- 1M+ dermoscopy images with text descriptions
- ICCV 2025 Highlight paper

### Transfer Learning Performance: ImageNet -> Dermoscopy
Published results show:
- ImageNet pretrained EfficientNet-B4 on ISIC 2019: ~90-93% balanced accuracy
- ImageNet pretrained Swin Transformer on ISIC 2019: ~93-95%
- Dermoscopy-specific pretraining (PanDerm) on downstream tasks: outperforms ImageNet by 2-5%

### Practical Recommendation for Hackathon (Time-Constrained)
1. **Fastest path**: Use `timm` EfficientNet-B4 (ImageNet pretrained) directly on our data
2. **Better if time**: Download ISIC 2019, fine-tune on it first, then our data
3. **Best if possible**: Use PanDerm checkpoint as starting point

---

## 7. EXACT DOWNLOAD INSTRUCTIONS

### ISIC 2019 (Classification Pretraining)

**Option A: Kaggle API**
```bash
pip install kaggle
# Put your kaggle.json in ~/.kaggle/

# Dataset 1 (by andrewmvd, ~2.6GB compressed)
kaggle datasets download -d andrewmvd/isic-2019
unzip isic-2019.zip -d isic2019_data/

# Dataset 2 (by salviohexia, pre-organized into folders)
kaggle datasets download -d salviohexia/isic-2019-skin-lesion-images-for-classification
unzip isic-2019-skin-lesion-images-for-classification.zip -d isic2019_folders/
```

**Option B: HuggingFace**
```python
from datasets import load_dataset
dataset = load_dataset("flwrlabs/fed-isic2019")
```

**Option C: Official ISIC Archive**
- URL: https://challenge.isic-archive.com/data/
- Direct download of training images + ground truth CSV

### ISIC 2018 Segmentation Masks

```bash
# Kaggle (2,594 images + masks for training)
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation
unzip isic2018-challenge-task1-data-segmentation.zip -d isic2018_seg/
```

### Pre-organized Dataset on HuggingFace (Resized)

```python
# ISIC 2019 resized to 224x224
from datasets import load_dataset
dataset = load_dataset("MKZuziak/ISIC_2019_224")
```

### PanDerm Model Weights

```bash
# Clone PanDerm repository
git clone https://github.com/SiyuanYan1/PanDerm.git

# Download checkpoints (check their README for exact Google Drive / HF links)
# PanDerm Large: panderm_ll_data6_checkpoint-499.pth
# PanDerm Base: panderm_bb_data6_checkpoint-499.pth
```

### ISIC 2019 1st Place Code

```bash
git clone https://github.com/ngessert/isic2019.git
# Contains full training pipeline for the winning solution
```

---

## 8. ACTION PLAN FOR HACKATHON

### Classification (Priority)
1. **Immediate**: Update augmentations to dermoscopy-specific (add hair artifacts, color constancy)
2. **If time**: Download ISIC 2019 from Kaggle, do intermediate fine-tuning
3. **Try**: ConvNeXt-Small as alternative to EfficientNet-B4
4. **Ensemble**: Average predictions from EfficientNet-B4 + ConvNeXt-Small
5. **Loss**: Switch from weighted CE to Focal Loss (gamma=2.0) with class weights
6. **TTA**: 8-fold TTA (4 rotations x 2 flips) at inference

### Segmentation (Already Good Setup)
1. Keep U-Net++ with EfficientNet-B4 encoder
2. Add post-processing: largest component, fill holes, smooth boundaries
3. Consider upscaling inputs to 256x256 or 384x384
4. Add TTA for segmentation too (flip + average masks)

### Time Allocation (36 hours)
- Hours 0-4: Update augmentations, implement focal loss, hair artifacts
- Hours 4-12: Train classification models (EfficientNet-B4, try ConvNeXt)
- Hours 12-20: Train segmentation, add post-processing
- Hours 20-28: Ensemble, TTA, generate predictions
- Hours 28-36: UI, presentation, submission polish
