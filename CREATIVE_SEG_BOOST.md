# Creative Segmentation Boost: 0.895 -> 0.95+ IoU

## Current Baseline
- **Best IoU**: ~0.895 (validation)
- **Architecture**: U-Net++ with EfficientNetV2-S encoder, 512px
- **Training**: Phase 1 (Dice+BCE, 40ep) -> Phase 2 (Lovasz fine-tune, 15ep)
- **Data**: 2000 train + 200 val (merged training+validation splits)
- **Augmentation**: Standard geometric + color (HFlip, VFlip, Rotate90, ShiftScaleRotate, HSV, CLAHE, Elastic)

---

## Technique Ranking: Impact vs Implementation Time

| # | Technique | Expected Boost | Time | Difficulty | PRIORITY |
|---|-----------|---------------|------|------------|----------|
| 1 | Copy-Paste Augmentation | +2-4% | 30 min | Easy | **DO FIRST** |
| 2 | Deep Supervision | +1-3% | 45 min | Medium | **DO SECOND** |
| 3 | Progressive Training (128->256->512) | +1-2% | 60 min | Easy | **DO THIRD** |
| 4 | Self-Training / Pseudo-Label Refinement | +1-3% | 60 min | Medium | **DO FOURTH** |
| 5 | Test-Time Augmentation (8x full) | +1-2% | 20 min | Easy | **ALREADY HAVE** |
| 6 | Multi-Scale Inference | +1-2% | 30 min | Easy | COMBINE W/ TTA |
| 7 | Attention (CBAM/SE) | +0.5-1.5% | 45 min | Medium | IF TIME |
| 8 | Knowledge Distillation | +1-2% | 90 min | Hard | IF TIME |
| 9 | Stochastic Depth/DropPath | +0.5-1% | 20 min | Easy | IF TIME |
| 10 | Style Transfer Augmentation | +0.5-1% | 90 min | Hard | SKIP |
| 11 | Multi-Task (classify+segment) | +0.5-1% | 120 min | Hard | SKIP |
| 12 | Noisy Student | +1-2% | 120 min | Hard | SKIP |

---

## Top 4 Techniques (Implement These)

### 1. Copy-Paste Augmentation (30 min, +2-4% IoU)

**Why it works**: Skin lesion datasets have limited diversity. Copy-Paste takes a lesion from one image and pastes it onto the skin background of another, creating novel training examples that force the model to learn boundary precision rather than memorizing backgrounds. This was a key technique in COCO segmentation competitions.

**Why it is especially good for skin lesions**:
- Lesion boundaries are well-defined (mask tells us exactly where)
- Skin backgrounds are relatively homogeneous
- Creates more boundary diversity, which is exactly what IoU measures

```python
import cv2
import numpy as np
import random

class CopyPasteAugmentation:
    """
    Copy a lesion from a donor image and paste it onto the current image.
    Applied BEFORE standard augmentations.
    """
    def __init__(self, all_images, all_masks, p=0.5):
        self.images = all_images  # list of image paths
        self.masks = all_masks    # list of mask paths
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        # Pick a random donor
        idx = random.randint(0, len(self.images) - 1)
        donor_img = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        donor_mask = cv2.imread(self.masks[idx], 0)
        donor_mask = (donor_mask > 127).astype(np.uint8)

        # Resize donor to match target
        h, w = image.shape[:2]
        donor_img = cv2.resize(donor_img, (w, h))
        donor_mask = cv2.resize(donor_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Random scale the lesion (0.5x to 1.5x)
        scale = random.uniform(0.5, 1.5)
        if scale != 1.0:
            sh, sw = int(h * scale), int(w * scale)
            donor_img = cv2.resize(donor_img, (sw, sh))
            donor_mask = cv2.resize(donor_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            # Center crop or pad back to (h, w)
            donor_img, donor_mask = _crop_or_pad(donor_img, donor_mask, h, w)

        # Alpha blend at boundaries for smooth transition (Gaussian blur on mask edge)
        alpha = cv2.GaussianBlur(donor_mask.astype(np.float32), (15, 15), 5)
        alpha = np.clip(alpha, 0, 1)
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        # Composite
        result_img = (image * (1 - alpha_3ch) + donor_img * alpha_3ch).astype(np.uint8)
        result_mask = np.maximum(mask, donor_mask.astype(np.float32))

        return result_img, result_mask


def _crop_or_pad(img, mask, target_h, target_w):
    """Center crop or zero-pad to target size."""
    h, w = img.shape[:2]
    if h > target_h:
        start = (h - target_h) // 2
        img = img[start:start+target_h]
        mask = mask[start:start+target_h]
    if w > target_w:
        start = (w - target_w) // 2
        img = img[:, start:start+target_w]
        mask = mask[:, start:start+target_w]
    h, w = img.shape[:2]
    if h < target_h or w < target_w:
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
    return img[:target_h, :target_w], mask[:target_h, :target_w]
```

**Integration into training**: Apply in `__getitem__` before albumentations transforms.

---

### 2. Deep Supervision with Auxiliary Losses (45 min, +1-3% IoU)

**Why it works**: U-Net++ already has nested skip connections at multiple resolutions. Deep supervision adds loss at EACH decoder level, not just the final output. This forces intermediate features to produce meaningful segmentation, improving gradient flow and preventing the model from relying only on the deepest features.

**Why it is high-impact for your case**: You are already using U-Net++, which natively supports deep supervision in `segmentation_models_pytorch`. The simplest version requires no model changes at all.

```python
import torch.nn.functional as F

def deep_supervision_loss(model, images, masks, dice_fn, bce_fn):
    """Apply loss at multiple resolutions of the same prediction."""
    pred = model(images)  # Full resolution prediction

    # Full resolution loss (weight 1.0)
    loss_full = 0.5 * dice_fn(pred, masks) + 0.5 * bce_fn(pred, masks)

    # Downsampled losses force model to get large-scale structure right
    for scale, weight in [(0.5, 0.3), (0.25, 0.1)]:
        h, w = int(masks.shape[2] * scale), int(masks.shape[3] * scale)
        pred_down = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
        mask_down = F.interpolate(masks, size=(h, w), mode='nearest')
        loss_down = 0.5 * dice_fn(pred_down, mask_down) + 0.5 * bce_fn(pred_down, mask_down)
        loss_full = loss_full + weight * loss_down

    return loss_full
```

**Simplest practical approach**: Multi-scale loss computation (no model changes needed). Compute your existing Dice+BCE at 512, 256, and 128 resolutions and sum them with decreasing weights (1.0, 0.3, 0.1).

---

### 3. Progressive Training: 128 -> 256 -> 512 (60 min, +1-2% IoU)

**Why it works**: Starting training at low resolution forces the model to learn coarse, global structure first (overall lesion shape, location). Then fine-tuning at higher resolution teaches boundary precision. This is a form of curriculum learning proven effective in segmentation competitions.

**Why it is great for your scenario**:
- Your images are originally small (~128px), upscaled to 512
- Low-res training is FAST (4-8x faster per epoch)
- You get better convergence at high resolution because weights are pre-conditioned

```python
progressive_schedule = [
    {"size": 128, "epochs": 15, "lr": 3e-4, "batch_size": 64},
    {"size": 256, "epochs": 15, "lr": 1e-4, "batch_size": 32},
    {"size": 512, "epochs": 20, "lr": 5e-5, "batch_size": 8},
    {"size": 512, "epochs": 10, "lr": 2e-5, "batch_size": 8, "loss": "lovasz"},
]

# At each phase transition:
# 1. Load best checkpoint from previous phase
# 2. Update DataLoader with new resolution
# 3. Reset optimizer (but keep model weights)
# 4. Use cosine annealing within each phase

for phase in progressive_schedule:
    # Rebuild transforms with new size
    ttf = A.Compose([
        A.Resize(phase["size"], phase["size"]),
        # ... same augmentations ...
    ])
    # Rebuild DataLoader with new batch size
    # Reset optimizer with new LR
    # Train for phase["epochs"]
```

**Key insight**: The 128px phase takes only ~2 minutes for 15 epochs. You spend most time on 512px anyway, but the low-res pretraining gives better initialization.

---

### 4. Self-Training / Pseudo-Label Refinement on TRAINING Data (60 min, +1-3% IoU)

**Why it works**: Some training masks may have annotation noise (rough boundaries, missing small regions). Self-training uses the model's own confident predictions to *correct* noisy ground truth labels. This is NOT using test data -- it is cleaning your training data.

**Mechanism**:
1. Train model normally to ~0.895 IoU
2. Run inference on the TRAINING set
3. For pixels where the model is very confident (sigmoid > 0.95 or < 0.05), trust the model over the original label
4. Blend the cleaned labels with original labels
5. Retrain (or fine-tune) on the improved labels

```python
import torch
import torch.nn.functional as F
import numpy as np

def generate_refined_masks(model, train_loader, device, conf_threshold=0.95):
    """
    Use model predictions to refine training masks.
    Only override ground truth where model is VERY confident.
    """
    model.eval()
    refined_masks = {}

    with torch.no_grad():
        for batch_idx, (images, masks, paths) in enumerate(train_loader):
            images = images.to(device)
            probs = torch.sigmoid(model(images))  # [B, 1, H, W]

            for i in range(len(images)):
                prob = probs[i, 0].cpu().numpy()  # [H, W]
                gt = masks[i, 0].numpy()           # [H, W]

                # Create confidence mask
                high_conf_pos = prob > conf_threshold
                high_conf_neg = prob < (1 - conf_threshold)
                uncertain = ~high_conf_pos & ~high_conf_neg

                # Refined mask: trust model where confident, keep GT where uncertain
                refined = np.zeros_like(gt)
                refined[high_conf_pos] = 1.0
                refined[high_conf_neg] = 0.0
                refined[uncertain] = gt[uncertain]

                refined_masks[paths[i]] = refined

    return refined_masks


def soft_label_training_loss(pred, refined_masks, original_masks, alpha=0.7):
    """
    Train with soft blend of refined and original masks.
    alpha=0.7 means 70% refined + 30% original.
    """
    targets = alpha * refined_masks + (1 - alpha) * original_masks
    # Use BCE (not Dice) for soft labels -- Dice needs binary targets
    loss = F.binary_cross_entropy_with_logits(pred, targets)
    return loss
```

**Important**: This is 100% legal -- you are using the model to fix your OWN training labels, not looking at test data. Many top Kaggle solutions use this technique.

**Expected pipeline**:
1. Train normally -> 0.895 IoU
2. Run inference on train set with confidence >= 0.95
3. Replace ~5-10% of noisy pixels with model predictions
4. Fine-tune for 10 epochs on refined labels
5. Expected: +1-3% IoU (especially on images with noisy GT boundaries)

---

## Quick Wins (Combine with Top 4)

### 5. Multi-Scale Inference (30 min, +1-2% IoU)

Run inference at 384, 512, and 640 resolutions, resize all predictions to 512, average probabilities, then threshold. Captures both global and fine-grained details.

```python
def multi_scale_inference(model, image, scales=[0.75, 1.0, 1.25], base_size=512):
    """Average predictions across multiple input scales."""
    predictions = []
    for scale in scales:
        size = int(base_size * scale)
        resized = F.interpolate(image, size=(size, size), mode='bilinear', align_corners=False)
        pred = torch.sigmoid(model(resized))
        pred = F.interpolate(pred, size=(base_size, base_size), mode='bilinear', align_corners=False)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

### 6. Stochastic Depth / DropPath (20 min, +0.5-1%)

Already built into `timm` encoders. Forces the model not to rely on any single path. Increase `drop_path_rate` in the encoder for stronger regularization during training.

---

## Techniques to SKIP (low ROI for 2-3 hours)

### Style Transfer Augmentation
- Requires training a style transfer network or using CycleGAN
- Implementation is non-trivial (2+ hours minimum)
- Marginal gain since color augmentation already covers skin tone variation
- **Verdict**: Skip unless you have a pretrained model ready

### Multi-Task Learning (Classify + Segment)
- Requires redesigning the model architecture
- Need to balance classification and segmentation losses
- Risk of degrading segmentation while trying to improve classification
- **Verdict**: Too much engineering for uncertain gain

### Noisy Student (Full Pipeline)
- Requires multiple training rounds with increasing noise
- Very compute-intensive (2-3x training time)
- The simplified version (self-training above) captures 80% of the benefit
- **Verdict**: Use the self-training variant instead (technique #4)

### Knowledge Distillation
- Need multiple trained teacher models first
- Training the student is another full training cycle
- Complex hyperparameter tuning (temperature, loss weight)
- **Verdict**: Worth it only if you have 3+ pretrained models ready to serve as teachers

---

## Recommended Implementation Order (2.5 hours)

```
[0:00-0:30] Copy-Paste Augmentation
  - Implement CopyPaste class
  - Integrate into Dataset.__getitem__
  - Start training with Phase 1

[0:30-1:00] Deep Supervision (multi-scale loss)
  - Add loss computation at 3 resolutions
  - Modify training loop
  - Apply to ongoing training

[1:00-1:30] Progressive Training
  - Set up 4-phase schedule
  - Implement phase transitions
  - Start fresh training with 128->256->512->Lovasz

[1:30-2:00] Self-Training Label Refinement
  - Run inference on training set with best model
  - Generate refined masks (conf > 0.95)
  - Fine-tune on refined labels for 10 epochs

[2:00-2:30] Multi-Scale Inference + TTA Combination
  - Implement 3-scale inference
  - Combine with existing 8x TTA
  - Tune optimal threshold on combined predictions
```

## Expected Cumulative Improvement

```
Baseline:                              0.895
+ Copy-Paste Augmentation:             0.910-0.920
+ Deep Supervision:                    0.920-0.935
+ Progressive Training:                0.930-0.945
+ Self-Training Refinement:            0.940-0.955
+ Multi-Scale Inference + TTA:         0.950-0.960
```

These estimates assume gains compound, but with diminishing returns. Realistically, combining all five should push you to the **0.94-0.96** range.

---

## Key Insight

The single highest-impact change that most teams miss: **Copy-Paste Augmentation**. It is trivially simple but generates training diversity that no amount of geometric/color augmentation can match. In the COCO segmentation challenge, it was worth +2-3% mAP, which translates to even more in binary segmentation where the task is simpler.

The second most underrated technique: **Self-Training on training data**. Most teams never question their ground truth labels, but annotation noise is real, especially near lesion boundaries. A model trained to 0.895 IoU already "knows" many boundaries better than the original annotator drew them.
