# ISIC Challenge Winner Segmentation Techniques

Research compilation for WhiteCoat.dev hackathon segmentation task.

---

## 1. ISIC 2017 Task 1 Winner: Yuan (Jaccard 0.765)

**Paper**: "Automatic skin lesion segmentation with fully convolutional-deconvolutional networks" (arXiv:1703.05165)

### Architecture
- **29-layer Fully Convolutional-Deconvolutional Network (CDNN)**, ~5.5M parameters
- **Encoder**: 5 convolutional blocks (conv-1 through conv-5) with max-pooling, filters 16 -> 512
- **Decoder**: Deconvolutional + upsampling layers (decv-1 through decv-5), filters 256 -> 16
- All kernels 3x3 (except conv-3-2 and decv-3-1 at 4x4), stride=1
- Dropout (p=0.5) before conv-4-1 and decv-5-1
- Batch normalization on all conv/deconv outputs

### Input
- **Resolution**: 192x256 (3:4 aspect ratio)
- **7 input channels**: RGB (3) + HSV (3) + CIELAB L-channel (1), all rescaled to [0,1]
- Multiple color spaces were critical for performance

### Loss Function
- **Jaccard distance loss** (directly optimizes the competition metric):
```
L_dJ = 1 - sum(t * p) / (sum(t^2) + sum(p^2) - sum(t * p))
```
- Handles class imbalance without rebalancing

### Training
- Optimizer: Adam, LR=0.003
- Batch size: 16
- Epochs: 500
- Augmentation: random flips, shifts, rotations, scaling + random contrast normalization per channel (on-the-fly)

### Post-Processing
- **Dual-threshold method**: threshold 0.8 identifies tumor center (largest connected component centroid), then threshold 0.5 with morphological filling; final mask = region embracing the center
- **Ensemble**: 6 CDNNs combined via bagging

### Scores
- Validation Jaccard: 0.784 (150 images)
- **Test Jaccard: 0.765** (600 images) -- 1st place

### Other Top ISIC 2017 Methods
- **RECOD Titans**: 4 models with different configs, validation 0.780-0.783, averaged output 0.793
- **Berseth**: Deep convolutional networks for segmentation + classification

---

## 2. ISIC 2018 Task 1 Winner: MT / Meitu (Jaccard 0.802)

**Paper**: "A Detection and Segmentation Architecture for Skin Lesion Segmentation on Dermoscopy Images" (arXiv:1809.03917)

### Architecture (Two-Stage)

**Stage 1 -- Detection**: Mask R-CNN detection branch for lesion localization and bounding box prediction (with segmentation branch supervision)

**Stage 2 -- Segmentation**: Encoder-decoder inspired by DeepLab/PSPNet/DenseASPP
- **Encoder**: Extended ResNet-101 with 3 cascading blocks
- **ASPP module**: Modified with dilated convolutions (rates 3, 6, 12), standard convolutions (sizes 3, 5, 7), and pooling layers (sizes 5, 9, 13, 17)
- Crops from detection bounding box fed into segmentation network

### Input
- **Resolution**: 512x512 (after crop from detection bbox)
- **8 input channels**: RGB (3) + HSV SV-channels (2) + CIELAB lab-channels (3), scaled to [0,1]

### Loss Function
- **Dice loss**:
```
L = -sum(p * g) / (sum(p) + sum(g) - sum(p * g))
```

### Training
- Optimizer: Adam, LR=0.001
- LR decay: 92% per epoch
- Batch size: 8
- Early stopping on overfitting
- Hardware: 2x NVIDIA 1080 Ti

### Augmentation
- Rotation, color jitter, flip, crop, shear
- Random crop between 81%-121% of bounding box

### Post-Processing
- **4-image TTA ensemble**: original + rotated 90/180 + flipped; final mask = average of 4 results

### Scores
- Validation Jaccard: 0.846
- **Test Jaccard: 0.802** -- 1st place

### ISIC 2018 Task 1 Leaderboard (Top 5)
| Rank | Team | Jaccard | Method |
|------|------|---------|--------|
| 1 | MT (Meitu) | 0.802 | MaskRCNN + segmentation |
| 2 | Holidayburned | 0.799 | Ensemble with CRF |
| 3 | imsight | 0.799 | DCNN segmentation |
| 4 | Tencent Youtu Lab | 0.798 | Adversarial learning |
| 5 | NMN_team | 0.796 | Ensemble + dual threshold |

### Key Improvements 2017 -> 2018
1. Two-stage detection+segmentation vs single-stage FCN
2. Higher input resolution (512x512 vs 192x256)
3. ResNet-101 backbone vs shallow 29-layer network
4. ASPP multi-scale feature extraction
5. Detection-guided cropping reduces background noise
6. Scores improved from 0.765 to 0.802 Jaccard

---

## 3. Hair Artifact Removal

### Black-Hat Morphological Filter + Inpainting Pipeline

Hair strands introduce additional edges that confuse automatic segmentation. The standard pipeline:

1. Convert to grayscale
2. Apply black-hat morphological operation (detects dark thin structures on light background)
3. Threshold to create binary hair mask
4. Inpaint detected regions using Telea or Navier-Stokes method

### Implementation

```python
import cv2
import numpy as np

def remove_hair(image, kernel_size=17, threshold=10):
    """
    Remove hair artifacts from dermoscopy images.

    Args:
        image: BGR image (numpy array)
        kernel_size: size of morphological kernel (odd number, larger = detects thicker hair)
        threshold: binary threshold for hair mask

    Returns:
        clean_image: image with hair removed
        hair_mask: binary mask of detected hair
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Black-hat morphological operation
    # Detects dark thin structures (hair) on lighter background
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (kernel_size, kernel_size)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create binary hair mask
    _, hair_mask = cv2.threshold(
        blackhat, threshold, 255, cv2.THRESH_BINARY
    )

    # Optional: dilate mask slightly to cover hair edges
    dilate_kernel = np.ones((3, 3), np.uint8)
    hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

    # Remove small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hair_mask, connectivity=8
    )
    min_area = 50  # minimum area to keep
    clean_mask = np.zeros_like(hair_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    # Inpaint using Telea method (fast marching)
    clean_image = cv2.inpaint(
        image, clean_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA
    )

    return clean_image, clean_mask


def remove_hair_batch(images):
    """Process a batch of images."""
    results = []
    for img in images:
        clean, _ = remove_hair(img)
        results.append(clean)
    return results
```

### Does Hair Removal Improve IoU?

- Studies show hair removal maintains high SSIM (structural similarity) while removing artifacts
- Improvement is most significant on images with heavy hair coverage
- For deep learning models with sufficient training data, the benefit is moderate (1-3% IoU improvement)
- Most effective when combined with other preprocessing
- Modern deep nets can learn to ignore hair, but explicit removal still helps on edge cases

### Recommendation for Hackathon
- Apply hair removal as preprocessing BEFORE augmentation
- Use kernel_size=17 for typical dermoscopy images
- Also apply to test images during inference
- Consider using it selectively (only on images where hair is detected)

---

## 4. Color Constancy Preprocessing

### Shades of Gray Algorithm

Normalizes illumination differences across dermoscopy images captured by different devices.

**Formula**: The illuminant estimate is based on the Minkowski p-norm:
```
e = (mean(I^p))^(1/p)
```
Where p=6 is the standard "Shades of Gray" parameter.

### Implementation

```python
import numpy as np
import cv2

def shades_of_gray(image, power=6):
    """
    Apply Shades of Gray color constancy.

    Args:
        image: BGR image (numpy array, uint8)
        power: Minkowski norm power (default 6 for Shades of Gray)

    Returns:
        corrected: color-corrected image (uint8)
    """
    img = image.astype(np.float32)

    # Compute p-norm for each channel
    # e_c = (mean(I_c^p))^(1/p)
    illuminant = np.zeros(3)
    for c in range(3):
        channel = img[:, :, c]
        illuminant[c] = np.power(
            np.mean(np.power(channel, power)), 1.0 / power
        )

    # Normalize illuminant
    illuminant_norm = illuminant / np.linalg.norm(illuminant)

    # Scale factor: target is uniform gray illuminant
    # Each channel scaled by (1/sqrt(3)) / illuminant_norm[c]
    target = 1.0 / np.sqrt(3.0)
    scale = target / (illuminant_norm + 1e-8)

    # Apply correction
    corrected = img.copy()
    for c in range(3):
        corrected[:, :, c] = img[:, :, c] * scale[c]

    # Clip and convert back
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


def shades_of_gray_fast(image, power=6):
    """Vectorized version for speed."""
    img = image.astype(np.float64)
    illuminant = np.power(
        np.mean(np.power(img, power), axis=(0, 1)), 1.0 / power
    )
    illuminant_norm = illuminant / (np.linalg.norm(illuminant) + 1e-8)
    scale = (1.0 / np.sqrt(3.0)) / (illuminant_norm + 1e-8)
    corrected = np.clip(img * scale[np.newaxis, np.newaxis, :], 0, 255)
    return corrected.astype(np.uint8)
```

### Does It Help for Segmentation?

- Color constancy improved classification sensitivity from 71.0% to 79.7%
- For segmentation specifically, the effect is smaller but positive (~0.5-1.5% IoU)
- Most helpful when training data comes from multiple imaging devices
- Shades of Gray (p=6) is recommended over Gray World (p=1) or max-RGB (p=inf)
- Less aggressive than Gray World, preserves more color information

### Recommendation
- Apply Shades of Gray (p=6) as preprocessing before training
- Apply same preprocessing at inference time
- If dataset is from a single device, benefit may be minimal

---

## 5. Augmentations for Segmentation Quality

### Elastic Deformation

Simulates natural tissue deformation. Critical for medical segmentation because organs/tissues deform during scans.

```python
import albumentations as A

elastic_transform = A.ElasticTransform(
    alpha=120,          # displacement magnitude
    sigma=120 * 0.05,   # smoothness of displacement
    p=0.3
)
```

**How it works**: Generates coarse displacement grid with random vectors, smooths with Gaussian kernel, scales by magnitude factor, then applies displacement to every pixel via spline interpolation.

### Grid Distortion

Divides image into grid cells and randomly shifts grid points.

```python
grid_distortion = A.GridDistortion(
    num_steps=5,        # grid cells per side
    distort_limit=0.3,  # max displacement as fraction of cell size
    p=0.3
)
```

### Optical Distortion

Simulates lens distortion (barrel/pincushion).

```python
optical_distortion = A.OpticalDistortion(
    distort_limit=0.05,
    shift_limit=0.05,
    p=0.3
)
```

### CutMix for Segmentation

In segmentation, CutMix must mix BOTH images AND masks consistently.

```python
import numpy as np
import torch

def cutmix_segmentation(image1, mask1, image2, mask2, alpha=1.0):
    """
    CutMix augmentation for segmentation tasks.
    Mixes both images and masks using the same rectangular region.

    Args:
        image1, image2: [C, H, W] tensors
        mask1, mask2: [H, W] tensors
        alpha: Beta distribution parameter

    Returns:
        mixed_image, mixed_mask
    """
    lam = np.random.beta(alpha, alpha)
    _, h, w = image1.shape

    # Generate random bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(h)
    cx = np.random.randint(w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    # Mix images
    mixed_image = image1.clone()
    mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]

    # Mix masks (same region -- hard label, not interpolated)
    mixed_mask = mask1.clone()
    mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

    return mixed_image, mixed_mask
```

**Key point for segmentation**: Masks must use HARD labels (copy-paste), not soft blending. The boundary between the two pasted regions is a hard transition.

### Recommended Full Augmentation Pipeline

```python
import albumentations as A

train_transform = A.Compose([
    # Geometric (applied to both image + mask)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5
    ),
    A.ElasticTransform(alpha=120, sigma=6, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),

    # Color (image only, not mask)
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.3
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32,
        fill_value=0, p=0.2
    ),

    # Normalize
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 6. Deep Supervision

### Concept

Intermediate mask predictions at multiple decoder scales, with auxiliary losses at each. Forces early decoder layers to produce meaningful segmentation features rather than relying only on the final output.

### Why It Works for Boundaries
- Low-level decoder features capture fine boundary details
- Without deep supervision, gradients may not reach early layers effectively
- Auxiliary losses at multiple scales create a multi-scale learning signal
- Particularly effective for fuzzy boundaries common in skin lesions

### Implementation with SMP (segmentation_models_pytorch)

SMP does not natively support deep supervision, but you can implement it by modifying the decoder:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DeepSupervisedUNetPP(nn.Module):
    """
    U-Net++ with deep supervision using SMP.
    Produces predictions at multiple decoder scales.
    """
    def __init__(
        self,
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        # Get decoder channel sizes for auxiliary heads
        decoder_channels = self.model.decoder.blocks

        # Auxiliary prediction heads at intermediate scales
        # These predict masks at 1/2, 1/4, 1/8 resolution
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(ch, classes, kernel_size=1)
            for ch in [64, 128, 256]  # adjust per encoder
        ])

    def forward(self, x):
        # Main prediction
        main_out = self.model(x)

        if self.training:
            # Return main + auxiliary predictions
            return main_out, aux_outputs
        return main_out


class DeepSupervisionLoss(nn.Module):
    """
    Combined loss with deep supervision.
    Main output gets full weight, auxiliary outputs get decaying weights.
    """
    def __init__(self, base_loss_fn, aux_weights=None):
        super().__init__()
        self.base_loss = base_loss_fn
        self.aux_weights = aux_weights or [0.4, 0.2, 0.1]

    def forward(self, predictions, target):
        if isinstance(predictions, tuple):
            main_pred = predictions[0]
            aux_preds = predictions[1:]
        else:
            return self.base_loss(predictions, target)

        # Main loss
        total_loss = self.base_loss(main_pred, target)

        # Auxiliary losses at different scales
        for aux_pred, weight in zip(aux_preds, self.aux_weights):
            # Resize target to match auxiliary prediction size
            aux_target = F.interpolate(
                target.unsqueeze(1).float(),
                size=aux_pred.shape[2:],
                mode='nearest'
            ).squeeze(1)
            total_loss = total_loss + weight * self.base_loss(
                aux_pred, aux_target
            )

        return total_loss


# Simpler approach: use SMP's built-in auxiliary head for classification
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    aux_params={
        "classes": 1,
        "pooling": "avg",
        "dropout": 0.5,
    }
)
# model returns (mask, classification_label) -- classification auxiliary task
```

---

## 7. Snapshot Ensemble

### Concept

Train ONE model with cyclic learning rate (cosine annealing). Save checkpoints at each LR minimum. At inference, ensemble predictions from all saved snapshots. Get M models for the cost of training 1.

### How It Works
1. Set total epochs = M * T (M snapshots, T epochs per cycle)
2. Each cycle: LR goes from max -> min via cosine annealing
3. At end of each cycle (LR at minimum), model has converged to a local minimum
4. Save checkpoint
5. LR resets to max, model escapes local minimum and finds a new one
6. At inference: average predictions from all M checkpoints

### Implementation

```python
import torch
import math
import os


class SnapshotEnsembleTrainer:
    """
    Train one model, get M snapshots for free.
    Uses cosine annealing with warm restarts.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        n_snapshots=5,
        epochs_per_cycle=30,
        max_lr=1e-3,
        min_lr=1e-6,
        save_dir="snapshots",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.n_snapshots = n_snapshots
        self.epochs_per_cycle = epochs_per_cycle
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=max_lr
        )
        self.total_epochs = n_snapshots * epochs_per_cycle

    def cosine_annealing_lr(self, epoch):
        """Compute LR for current epoch within cycle."""
        cycle_epoch = epoch % self.epochs_per_cycle
        cos_value = math.cos(
            math.pi * cycle_epoch / self.epochs_per_cycle
        )
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + cos_value
        )
        return lr

    def train(self):
        snapshots = []

        for epoch in range(self.total_epochs):
            # Update learning rate
            lr = self.cosine_annealing_lr(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Train one epoch
            self.model.train()
            for images, masks in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            # Save snapshot at end of each cycle
            cycle_num = epoch // self.epochs_per_cycle
            is_cycle_end = (epoch + 1) % self.epochs_per_cycle == 0

            if is_cycle_end:
                path = os.path.join(
                    self.save_dir, f"snapshot_{cycle_num}.pth"
                )
                torch.save(self.model.state_dict(), path)
                snapshots.append(path)
                print(
                    f"Snapshot {cycle_num} saved at epoch {epoch}, "
                    f"LR={lr:.2e}"
                )

        return snapshots


def snapshot_ensemble_predict(model, snapshot_paths, test_loader, device):
    """
    Average predictions from all snapshots.

    Args:
        model: model architecture (same as trained)
        snapshot_paths: list of checkpoint file paths
        test_loader: DataLoader for test images
        device: torch device

    Returns:
        averaged predictions
    """
    all_predictions = []

    for path in snapshot_paths:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        predictions = []
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                outputs = torch.sigmoid(model(images))
                predictions.append(outputs.cpu())

        all_predictions.append(torch.cat(predictions, dim=0))

    # Average all snapshot predictions
    ensemble_pred = torch.stack(all_predictions).mean(dim=0)
    return ensemble_pred
```

### Using PyTorch Built-in Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# T_0 = epochs per cycle, T_mult = 1 (constant cycle length)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=30,      # epochs per cycle
    T_mult=1,    # keep cycle length constant
    eta_min=1e-6 # minimum LR
)

# In training loop:
for epoch in range(total_epochs):
    train_one_epoch(model, optimizer, train_loader)
    scheduler.step()

    # Save at cycle boundaries
    if (epoch + 1) % 30 == 0:
        torch.save(
            model.state_dict(),
            f"snapshot_epoch{epoch}.pth"
        )
```

### Recommended Settings for Hackathon

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_snapshots | 3-5 | More snapshots = better ensemble, but diminishing returns |
| epochs_per_cycle | 20-30 | Must be enough to converge |
| max_lr | 1e-3 | Standard for Adam |
| min_lr | 1e-6 | Near zero for convergence |

---

## Summary: What to Implement for Maximum IoU

### Priority Order (effort vs impact)

| Priority | Technique | Expected IoU Gain | Effort |
|----------|-----------|-------------------|--------|
| 1 | Strong encoder (EfficientNet-B4 / ResNet-101) | +5-10% | Low (use SMP) |
| 2 | Dice + BCE + Lovasz combined loss | +2-4% | Low |
| 3 | TTA (flips + rotations, 8x) | +1-3% | Low |
| 4 | Snapshot ensemble (3-5 checkpoints) | +1-2% | Low |
| 5 | Deep augmentations (elastic, grid, optical) | +1-2% | Low |
| 6 | Hair removal preprocessing | +0.5-1.5% | Medium |
| 7 | Multi-color-space input (RGB+HSV+LAB) | +1-2% | Medium |
| 8 | Deep supervision / auxiliary losses | +0.5-1% | Medium |
| 9 | Two-stage detect+segment | +1-2% | High |
| 10 | Color constancy (Shades of Gray) | +0.5-1% | Low |
| 11 | CutMix for segmentation | +0.5-1% | Low |
| 12 | CRF post-processing | +0.5-1% | Medium |

### Key Takeaways from Winners

1. **Multi-color-space inputs** (RGB + HSV + CIELAB) were used by BOTH 2017 and 2018 winners
2. **Jaccard/Dice-based losses** directly optimize the competition metric
3. **Ensemble approaches** were universal among top performers
4. **Post-processing matters**: dual-threshold, morphological operations, CRF
5. **Higher resolution inputs** (512x512) significantly improved 2018 over 2017
6. **Detection-guided cropping** (2018 winner) reduces background noise
7. **TTA with rotations/flips** is free IoU improvement for skin lesion images (no canonical orientation)
