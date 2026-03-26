# PanDerm & ISIC Pretraining Research
## Deep Research for WhiteCoat.dev Hackathon

---

## 1. PanDerm Foundation Model

### Architecture

| Variant | Architecture | embed_dim | depth | num_heads | Params |
|---------|-------------|-----------|-------|-----------|--------|
| PanDerm_Large | ViT-L/16 | 1024 | 24 | 16 | ~307M |
| PanDerm_Base | ViT-B/16 | 768 | 12 | 12 | ~86M |

- **Input size**: 224x224 (patch size 16x16)
- **Pretraining**: Self-supervised on 2M+ skin disease images across 4 imaging modalities from 11 institutions
- **Checkpoint names**: `panderm_ll_data6_checkpoint-499.pth` (Large), `panderm_bb_data6_checkpoint-499.pth` (Base)
- **License**: CC-BY-NC-ND 4.0 (non-commercial academic only -- fine for hackathon)

### Download Links

```
# PanDerm Base (recommended for hackathon -- faster, still excellent)
https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view

# PanDerm Large (better accuracy, slower training)
https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view

# DermLIP_PanDerm (HuggingFace, newest variant)
https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256
```

### How to Download via CLI

```bash
# Install gdown for Google Drive downloads
pip install gdown

# PanDerm Base (~350MB)
gdown 17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB -O panderm_base_checkpoint.pth

# PanDerm Large (~1.2GB)
gdown 1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE -O panderm_large_checkpoint.pth
```

### Dependencies

```bash
conda create -n panderm python=3.10 -y
conda activate panderm
pip install torch==2.4.1 torchvision==0.19.1
pip install timm==0.9.16
pip install pandas scikit-learn opencv-python albumentations tqdm wandb
```

### Loading PanDerm for 12-Class Classification

```python
import torch
import torch.nn as nn
import timm

def load_panderm_base(checkpoint_path, num_classes=12):
    """Load PanDerm Base as a 12-class classifier."""
    # Create ViT-B/16 architecture matching PanDerm
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
    )

    # Load PanDerm weights
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # PanDerm checkpoints have "encoder." prefix -- strip it
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("encoder.", "")
        cleaned[new_key] = v

    # Load with strict=False (head dimensions will differ)
    msg = model.load_state_dict(cleaned, strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    return model


def load_panderm_large(checkpoint_path, num_classes=12):
    """Load PanDerm Large as a 12-class classifier."""
    model = timm.create_model(
        'vit_large_patch16_224',
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.2,
    )

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cleaned = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(cleaned, strict=False)
    print(f"Missing keys: {msg.missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")

    return model


# Usage:
model = load_panderm_base("panderm_base_checkpoint.pth", num_classes=12)
model = model.cuda()
```

**Important note**: The `strict=False` is necessary because PanDerm's pretrained checkpoint has a different classification head dimension. The head (final linear layer) will be randomly initialized for your 12 classes -- that is expected and correct.

### Using PanDerm as Feature Extractor

```python
def load_panderm_feature_extractor(checkpoint_path, variant='base'):
    """Load PanDerm without classification head for feature extraction."""
    if variant == 'base':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        feat_dim = 768
    else:
        model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0)
        feat_dim = 1024

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cleaned = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    model.set_grad_checkpointing(False)

    return model, feat_dim


# Extract features then train a simple classifier
model, feat_dim = load_panderm_feature_extractor("panderm_base_checkpoint.pth")
model = model.cuda()

# feat_dim = 768 for Base, 1024 for Large
classifier = nn.Linear(feat_dim, 12).cuda()
```

### Data Preprocessing for PanDerm

```python
from torchvision import transforms

# ImageNet normalization (PanDerm uses this)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

### Fine-Tuning Hyperparameters (from PanDerm paper)

```python
# PanDerm recommended fine-tuning config:
config = {
    'batch_size': 128,       # reduce to 32-64 if GPU memory limited
    'lr': 5e-4,
    'epochs': 50,            # for hackathon: 15-25 epochs is enough
    'layer_decay': 0.65,     # lower LR for earlier layers
    'drop_path': 0.2,        # for Large; 0.1 for Base
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'label_smoothing': 0.1,
    'use_weighted_sampling': True,  # critical for imbalanced data
    'use_tta': True,         # test-time augmentation
}
```

### Published PanDerm Benchmarks

PanDerm achieves state-of-the-art on multiple derm benchmarks:
- Outperforms ImageNet-pretrained ViTs by 5-15% on small datasets
- Matches or exceeds performance with only 10% of labeled data
- Evaluated on: HAM10000, BCN20000, DDI, Derm7pt, Dermnet, HIBA, MSKCC, PAD-UFES, PATCH16

---

## 2. ISIC 2019 Intermediate Fine-Tuning

### The 3-Stage Pipeline

```
Stage 1: ImageNet pretraining (already done in timm models)
    |
Stage 2: ISIC 2019 fine-tuning (25,331 derm images, 8 classes)
    |
Stage 3: Your data fine-tuning (11,400 images, 12 classes)
```

### ISIC 2019 Classes (8 classes)

| Class | Disease | Count |
|-------|---------|-------|
| MEL | Melanoma | 4,522 |
| NV | Melanocytic nevus | 12,875 |
| BCC | Basal cell carcinoma | 3,323 |
| AK | Actinic keratosis | 867 |
| BKL | Benign keratosis | 2,624 |
| DF | Dermatofibroma | 239 |
| VASC | Vascular lesion | 253 |
| SCC | Squamous cell carcinoma | 628 |

### Mapping ISIC 8 Classes to Your 12 Classes

Since your class names are hidden (0-11), you cannot map directly. Instead:

**Strategy: Use ISIC 2019 as intermediate pretraining, NOT class mapping.**

```python
# Stage 2: Fine-tune on ISIC 2019 (8 classes)
model = timm.create_model('efficientnetv2_s', pretrained=True, num_classes=8)
# ... train on ISIC 2019 for 15-20 epochs ...

# Stage 3: Replace head, fine-tune on your data (12 classes)
model.classifier = nn.Linear(model.classifier.in_features, 12)
# ... train on your 11,400 images ...
```

### Complete 3-Stage Code

```python
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import numpy as np

# ============================================================
# STAGE 2: Fine-tune on ISIC 2019
# ============================================================

def train_on_isic2019(isic_train_dir, isic_val_dir, epochs=15):
    """
    Fine-tune ImageNet-pretrained model on ISIC 2019.

    Directory structure expected:
    isic_train_dir/
        MEL/
        NV/
        BCC/
        AK/
        BKL/
        DF/
        VASC/
        SCC/
    """
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(isic_train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(isic_val_dir, transform=transform_val)

    # Weighted sampling for class imbalance
    class_counts = np.bincount([t for _, t in train_dataset.samples])
    weights = 1.0 / class_counts
    sample_weights = weights[train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Create model with 8 ISIC classes
    model = timm.create_model('efficientnetv2_s', pretrained=True, num_classes=8)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "isic2019_pretrained.pth")

    return model


# ============================================================
# STAGE 3: Fine-tune on your 12-class data
# ============================================================

def finetune_on_our_data(model_or_path, train_dir, val_dir, epochs=30):
    """
    Fine-tune ISIC-pretrained model on your 12-class dataset.
    """
    if isinstance(model_or_path, str):
        model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=8)
        model.load_state_dict(torch.load(model_or_path, map_location='cpu'))

    # Replace 8-class head with 12-class head
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 12)
    model = model.cuda()

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    # Weighted sampling
    class_counts = np.bincount([t for _, t in train_dataset.samples])
    weights = 1.0 / class_counts
    sample_weights = weights[train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_classification_model.pth")

    return model
```

### Should You Fine-Tune All Layers or Just the Head?

**For ISIC 2019 stage**: Fine-tune ALL layers. The domain shift from ImageNet to dermoscopy is large, so the entire network benefits from domain adaptation. Use a smaller LR (1e-4) with layer-wise LR decay.

**For your data stage**: Fine-tune ALL layers with a small LR (3e-4). Your dataset has 11,400 images which is enough for full fine-tuning. Freezing early layers gives minimal benefit with this much data.

### How Many Epochs on ISIC?

- **ISIC 2019 stage**: 10-15 epochs is sufficient. The model converges fast since it starts from ImageNet weights. Going beyond 20 epochs risks overfitting to ISIC class distribution.
- **Your data stage**: 20-30 epochs with early stopping.

---

## 3. PanDerm vs ISIC Pretraining: Which to Use?

### Decision Matrix

| Factor | PanDerm | ISIC 2019 Pretrain |
|--------|---------|-------------------|
| Setup time | 5 min (download + load) | 1-2 hours (download + train) |
| Expected boost | +5-15% over ImageNet | +3-8% over ImageNet |
| GPU time | 0 (pretrained) | 1-2 hours on single GPU |
| Complexity | Low | Medium |
| Risk | Low | Low-Medium |

### RECOMMENDATION FOR HACKATHON

**Use PanDerm Base directly.** Here is why:

1. PanDerm was pretrained on 2M+ skin images -- far more domain knowledge than ISIC 2019's 25K
2. Zero additional training time for the pretraining stage
3. PanDerm Base (ViT-B/16, 86M params) fits comfortably on a single GPU
4. The timm integration means you can swap it into your existing pipeline easily

**If you have spare time (>6 hours left)**: Also try ISIC 2019 intermediate fine-tuning with EfficientNetV2-S for an ensemble member.

---

## 4. ISIC 2018 Segmentation Pretraining for U-Net++

### Can You Pretrain U-Net++ Encoder on ISIC 2018?

Yes. ISIC 2018 Task 1 has 2,594 dermoscopic images with segmentation masks. Pretraining your U-Net++ encoder on this data before fine-tuning on your 1,800 images is a solid strategy.

### Expected Benefit

- **Without ISIC 2018 pretraining**: ~85-88% mean IoU (typical for 1,800 training images)
- **With ISIC 2018 pretraining**: ~88-91% mean IoU (+2-4% improvement)
- The benefit is moderate because your 1,800 images is already a reasonable amount

### ISIC 2018 Segmentation Pretraining Code

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================
# ISIC 2018 Segmentation Dataset
# ============================================================

class ISICSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # ISIC 2018 masks have _segmentation suffix
        mask_name = img_name.replace('.jpg', '_segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].unsqueeze(0)

        return image, mask


# ============================================================
# Stage 1: Pretrain on ISIC 2018
# ============================================================

def pretrain_on_isic2018(isic_image_dir, isic_mask_dir, epochs=20):
    """Pretrain U-Net++ encoder on ISIC 2018 segmentation data."""

    transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    dataset = ISICSegDataset(isic_image_dir, isic_mask_dir, transform=transform)

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Create U-Net++ with EfficientNet-B4 encoder
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Combined Dice + BCE loss
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()

    best_iou = 0
    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = 0.5 * dice_loss(outputs, masks) + 0.5 * bce_loss(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.cuda(), masks.cuda()
                outputs = torch.sigmoid(model(images))
                preds = (outputs > 0.5).float()
                intersection = (preds * masks).sum(dim=(2, 3))
                union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
                iou = (intersection + 1e-7) / (union + 1e-7)
                ious.extend(iou.cpu().numpy().flatten())

        mean_iou = np.mean(ious)
        scheduler.step()
        print(f"ISIC 2018 Epoch {epoch+1}/{epochs}, Mean IoU: {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "isic2018_unetpp_pretrained.pth")

    return model


# ============================================================
# Stage 2: Fine-tune on your 1,800 images
# ============================================================

def finetune_segmentation(pretrained_path, train_image_dir, train_mask_dir,
                          val_image_dir, val_mask_dir, epochs=40):
    """Fine-tune ISIC-pretrained U-Net++ on your data."""

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,  # don't load ImageNet, we have better
        in_channels=3,
        classes=1,
        activation=None,
    )

    # Load ISIC 2018 pretrained weights
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.cuda()

    # Use your existing training pipeline from here...
    # (same transforms, loss, optimizer as your current segmentation code)

    return model
```

---

## 5. Dataset Downloads

### ISIC 2019 Classification (25,331 images, 8 classes)

**Kaggle** (recommended -- fastest):
```bash
# Install Kaggle CLI
pip install kaggle

# Set up API key: ~/.kaggle/kaggle.json

# Download Option 1: Pre-organized into folders (~9GB)
kaggle datasets download -d salviohexia/isic-2019-skin-lesion-images-for-classification
unzip isic-2019-skin-lesion-images-for-classification.zip -d isic2019/

# Download Option 2: Raw with CSV labels (~9GB)
kaggle datasets download -d andrewmvd/isic-2019
unzip isic-2019.zip -d isic2019_raw/
```

**Direct from ISIC Archive**:
```
https://challenge.isic-archive.com/data/#2019
```

| Component | Size |
|-----------|------|
| Training images (25,331 JPEG) | ~9 GB |
| Ground truth CSV | <1 MB |
| Total | ~9 GB |

**Download time**: ~15-30 min on 50 Mbps connection.

### ISIC 2018 Segmentation (2,594 images + masks)

**Kaggle**:
```bash
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation
unzip isic2018-challenge-task1-data-segmentation.zip -d isic2018_seg/
```

| Component | Size |
|-----------|------|
| Training images (2,594 JPEG) | ~2.5 GB |
| Training masks (2,594 PNG) | ~100 MB |
| Total | ~2.6 GB |

**Download time**: ~5-10 min on 50 Mbps.

### HAM10000

```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d ham10000/
```

| Component | Size |
|-----------|------|
| 10,015 images | ~2.8 GB |
| Metadata CSV | <1 MB |

### Fitzpatrick17k

```bash
# Clone the repo for download script and metadata
git clone https://github.com/mattgroh/fitzpatrick17k.git
cd fitzpatrick17k
# Images need to be downloaded from original atlas sources
# The repo provides URLs and scripts
```

| Component | Details |
|-----------|---------|
| 16,577 clinical photos | ~4 GB |
| 114 skin conditions | Clinical (not dermoscopic) |
| Fitzpatrick skin types I-VI | Diversity-focused |

---

## 6. Dataset Overlap and Combination

### HAM10000 vs ISIC 2019

**HAM10000 is a SUBSET of ISIC 2019.** All 10,015 HAM10000 images are included in ISIC 2019. Do NOT combine them -- you would have duplicates.

ISIC 2019 = HAM10000 (10,015) + BCN20000 (~12,000) + MSK dataset (~3,300)

### Fitzpatrick17k: Worth Adding?

**Maybe, but with caveats:**
- Fitzpatrick17k has **clinical photos** (camera photos, not dermoscopy)
- Your hackathon data appears to include both clinical and dermoscopic
- Adding Fitzpatrick17k could help if your data has clinical photos
- BUT: different class taxonomy (114 conditions vs your 12), so you would use it for encoder pretraining only
- **Verdict for hackathon**: Skip it. Too complex to integrate in 36 hours. PanDerm already covers this.

---

## 7. Time Estimates for Hackathon

### Single GPU Training Times (approximate)

| Task | GPU | Time |
|------|-----|------|
| Download ISIC 2019 (9GB) | N/A | 15-30 min |
| Fine-tune EfficientNetV2-S on ISIC 2019 (15 epochs) | T4/A100 | 1-2 hours |
| Fine-tune PanDerm Base on your data (25 epochs) | T4 | 2-3 hours |
| Fine-tune PanDerm Base on your data (25 epochs) | A100 | 45-90 min |
| Download ISIC 2018 seg (2.6GB) | N/A | 5-10 min |
| Pretrain U-Net++ on ISIC 2018 (20 epochs) | T4 | 1-1.5 hours |
| Fine-tune U-Net++ on your 1,800 images (40 epochs) | T4 | 1-2 hours |
| Fine-tune U-Net++ on your 1,800 images (40 epochs) | A100 | 30-60 min |

### Is ISIC Pretraining Worth It in 36 Hours?

**Classification (PanDerm approach)**: ABSOLUTELY YES.
- PanDerm download: 5 minutes
- No additional pretraining needed
- Expected boost: +5-15% over ImageNet-only ViT
- Time cost: near zero

**Classification (ISIC 2019 intermediate)**: YES if you have time.
- Total time: 2-3 hours (download + train)
- Expected boost: +3-8% over ImageNet-only
- Use this as a second ensemble member

**Segmentation (ISIC 2018 pretraining)**: MAYBE.
- Total time: 2-3 hours (download + train)
- Expected boost: +2-4% mean IoU
- Worth it if segmentation score is close and you need the edge
- Lower priority than getting classification right

### Expected Accuracy Improvements

| Approach | Expected Accuracy |
|----------|------------------|
| EfficientNetV2-S (ImageNet only) | 82-87% |
| EfficientNetV2-S + ISIC 2019 pretrain | 85-90% |
| PanDerm Base (direct fine-tune) | 88-93% |
| PanDerm Base + EfficientNetV2-S ensemble | 90-95% |
| Add TTA to ensemble | +1-2% |

---

## 8. Recommended Hackathon Pipeline

### Priority Order (do in this sequence)

```
Hour 0-1:   Download PanDerm Base checkpoint (5 min)
            Set up data loaders for your 12-class data
            Start PanDerm Base fine-tuning (25 epochs)

Hour 1-3:   While PanDerm trains, set up segmentation pipeline
            Start U-Net++ training on your 1,800 images (ImageNet encoder)

Hour 3-5:   PanDerm classification done -- check results
            Start EfficientNetV2-S on your data (separate model)
            Download ISIC 2018 segmentation data (background)

Hour 5-8:   EfficientNetV2-S done -- check results
            Build classification ensemble (PanDerm + EffNetV2)
            Start U-Net++ with ISIC 2018 pretrained encoder (if time)

Hour 8-12:  Optimize ensemble weights
            Add TTA (test-time augmentation)
            Generate final predictions

Hour 12+:   Buffer for debugging, presentation prep
```

### The Single Most Important Thing

If you can only do ONE thing from this research: **Use PanDerm Base as your classification backbone.** It is a free 5-15% accuracy boost that takes 5 minutes to set up.

---

## 9. Quick-Start: PanDerm for Your Exact Pipeline

```python
"""
Minimal script to integrate PanDerm into your hackathon pipeline.
Drop-in replacement for any timm model.
"""

import torch
import timm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# ============================================================
# 1. Download checkpoint first:
#    pip install gdown
#    gdown 17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB -O panderm_base.pth
# ============================================================

# 2. Load PanDerm Base
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=12)
state_dict = torch.load('panderm_base.pth', map_location='cpu', weights_only=True)
cleaned = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
model.load_state_dict(cleaned, strict=False)
model = model.cuda()

# 3. Data transforms (PanDerm recommended)
train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 4. Create datasets
train_ds = datasets.ImageFolder('data/classification/train', transform=train_tf)
# For validation, split off 10% of training data

# 5. Weighted sampler for imbalanced classes
targets = np.array(train_ds.targets)
class_counts = np.bincount(targets)
weights = 1.0 / class_counts
sample_weights = weights[targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)

# 6. Train
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(25):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}/25 done")

torch.save(model.state_dict(), "panderm_12class_best.pth")
```

---

## Sources

- PanDerm GitHub: https://github.com/SiyuanYan1/PanDerm
- PanDerm paper (Nature Medicine): https://www.nature.com/articles/s41591-025-03747-y
- PanDerm arXiv: https://arxiv.org/abs/2410.15038
- ISIC 2019 Kaggle: https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification
- ISIC 2018 Segmentation Kaggle: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation
- HAM10000 Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- ISIC Archive: https://challenge.isic-archive.com/data/
- Fitzpatrick17k: https://github.com/mattgroh/fitzpatrick17k
- timm library: https://github.com/huggingface/pytorch-image-models
