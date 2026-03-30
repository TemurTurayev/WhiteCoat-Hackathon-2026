# Advanced Segmentation Architectures for Skin Lesion Segmentation

**Goal**: Push IoU from 0.895 to 0.95+ on our hackathon dataset
**Key Constraint**: Original images are 128x128, upscaled to 512x512

---

## Executive Summary

| Architecture | Expected IoU (skin) | Small Image Suitability | Implementation Complexity | Training Time | Pretrained Weights |
|---|---|---|---|---|---|
| SAM 2 (fine-tuned) | 0.91-0.94 | Poor (needs 1024x1024) | High | 2-4 hours (GPU) | Yes (Meta) |
| SegFormer (MiT-B3/B5) | 0.90-0.93 | Good (flexible input) | Low | 30-60 min | Yes (HuggingFace) |
| HRNet-W48 + OCR | 0.89-0.92 | Excellent (multi-res) | Medium | 45-90 min | Yes (official) |
| Mask2Former | 0.90-0.93 | Medium | High | 1-2 hours | Yes (Detectron2) |
| ConvNeXt V2 encoder | 0.91-0.94 | Good | Low (via SMP) | 30-60 min | Yes (timm/SMP) |
| Cascade/Coarse-to-Fine | 0.92-0.95 | Excellent | Medium-High | 1-2 hours | Varies |

**Recommendation for our hackathon (128x128 images, time-constrained)**:
1. **Best bang for buck**: ConvNeXt V2 encoder + UNet++/DeepLabV3+ via segmentation_models_pytorch
2. **Highest ceiling**: Boundary-aware cascade refinement on top of existing ensemble
3. **Skip**: SAM 2 (overkill for 128x128, heavy setup, needs 1024 input)

---

## 1. SAM 2 (Segment Anything Model 2)

### Overview
Meta's SAM 2 is a foundation model for promptable segmentation in images and videos. It uses a Hiera vision transformer encoder and can segment anything given point, box, or mask prompts.

### Skin Lesion Performance
- Medical SAM 2 (MedSAM-2) evaluated on skin lesions among other modalities
- SAM-Adapter approach achieved Dice 94.27% on ISIC 2018 (IoU ~0.89)
- Zero-shot SAM performance on skin lesions is suboptimal due to ambiguous boundaries
- Fine-tuned SAM variants reach Dice 0.92-0.94 on ISIC datasets

### Can We Use It?

**Prompting strategies:**
- **Point prompt**: Center of lesion (can auto-detect via thresholding)
- **Box prompt**: Bounding box around lesion (more reliable)
- **Mask prompt**: Feed coarse mask from existing model as prompt for refinement

**Problem with 128x128 images:**
- SAM 2 was trained on 1024x1024 images
- At 224x224, SAM2-UNet shows substantially degraded performance
- At 512x512, performance improves significantly but still below 1024
- Our 128x128 originals (even upscaled to 512) will have blurry features that SAM cannot recover
- **Verdict: NOT recommended for our use case**

### Fine-Tuning Code Sketch
```python
# Based on github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model (use sam2.1_hiera_small for speed)
predictor = SAM2ImagePredictor(build_sam2("sam2.1_hiera_s", "checkpoints/sam2.1_hiera_small.pt"))

# Fine-tune: must remove no_grad from SAM2 source to train encoder
predictor.model.train()
for image, mask, points in dataloader:
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(point_coords=points, point_labels=[1])
    loss = dice_loss(masks, mask) + bce_loss(masks, mask)
    loss.backward()
    optimizer.step()
```

### Verdict
- **Skip for hackathon**. Heavy setup, needs large images, overkill for binary segmentation.
- Could be useful as a refinement step if we had more time.

---

## 2. SegFormer

### Overview
SegFormer uses a hierarchical Mix Transformer (MiT) encoder with an MLP decoder. It produces multi-scale features without positional encodings, making it flexible with input resolutions.

### Skin Lesion Performance
- SegFormer with dropout: IoU 0.932 on dermoscopic dataset (10-fold CV)
- ISIC-2016: IoU 0.902
- ISIC-2017: IoU 0.864
- ISIC-2018: IoU 0.894
- SegFormer + U-Net hybrid: Dice 0.94-0.99 across ISIC datasets (2025 study)

### Small Image Suitability
- **Good**: No fixed positional encoding means flexible resolution
- MiT encoder uses overlapping patch embeddings (patch size 7, stride 4)
- For 128x128: produces feature maps of 32x32, 16x16, 8x8, 4x4 -- workable
- For 512x512: produces 128x128, 64x64, 32x32, 16x16 -- ideal

### Implementation
```python
# Option A: HuggingFace transformers
from transformers import SegformerForSemanticSegmentation
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b3-finetuned-ade-512x512",
    num_labels=1,
    ignore_mismatched_sizes=True
)

# Option B: segmentation_models_pytorch (timm integration)
import segmentation_models_pytorch as smp
model = smp.UnetPlusPlus(
    encoder_name="tu-mit_b3",  # SegFormer MiT-B3 as encoder
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
```

### Verdict
- **Decent option** if using as encoder in SMP
- MiT-B3 provides good balance of parameters (~45M) and performance
- Won't dramatically outperform our current EfficientNet/ConvNeXt encoders
- Marginal improvement expected: IoU 0.90-0.93

---

## 3. HRNet (High-Resolution Network)

### Overview
HRNet maintains high-resolution representations throughout the network by running parallel multi-resolution branches that repeatedly exchange information. This avoids the typical encoder-decoder bottleneck where spatial info is lost.

### Skin Lesion Performance
- HRNet-W48 + OCR: competitive on cityscapes and medical segmentation
- Used as feature extractor in Skin-DeepNet framework
- Generally achieves IoU 0.88-0.92 on skin lesion datasets
- Strength is in preserving boundary details

### Small Image Suitability
- **Excellent**: Maintains high-resolution features by design
- For 128x128 images, the parallel branches (128, 64, 32, 16) preserve spatial detail
- Multi-scale fusion helps compensate for limited input information
- No aggressive downsampling in early layers

### Implementation
```python
# Via segmentation_models_pytorch (timm)
import segmentation_models_pytorch as smp
model = smp.UnetPlusPlus(
    encoder_name="tu-hrnet_w48",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# Or via mmsegmentation
# pip install mmsegmentation
# Uses HRNet-W48 + OCR head
```

### Verdict
- **Good for small images** due to high-resolution preservation
- Available via SMP with timm prefix
- Expected IoU: 0.89-0.92
- Not a massive leap over current approach

---

## 4. Mask2Former

### Overview
Mask2Former unifies instance, semantic, and panoptic segmentation using masked attention in a transformer decoder. It uses a Swin Transformer backbone + pixel decoder + transformer decoder with masked cross-attention.

### Skin Lesion Performance
- Competitive but not leading on skin lesion benchmarks
- SapFormer outperforms Mask2Former by ~1.4% Dice on ISIC-2018
- Better suited for multi-class segmentation than binary lesion segmentation
- IoU typically 0.88-0.92 on skin datasets

### Small Image Suitability
- **Medium**: Swin backbone uses window attention (window size 7)
- For 128x128: only ~4 windows at deepest stage -- limited context
- For 512x512: adequate number of windows
- Overhead is significant for binary segmentation

### Implementation
```python
# Via Detectron2 / MMSegmentation
# Requires detectron2 installation
from detectron2.config import get_cfg
from detectron2.projects.mask2former import add_maskformer2_config

# Or HuggingFace
from transformers import Mask2FormerForUniversalSegmentation
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-base-ade-semantic",
    num_labels=1,
    ignore_mismatched_sizes=True
)
```

### Verdict
- **Overkill for binary segmentation**
- Complex setup (Detectron2 dependency)
- Better for multi-class scenarios
- Skip for hackathon unless already familiar with Detectron2

---

## 5. ConvNeXt V2 / InternImage as Encoders

### Overview
ConvNeXt V2 modernizes pure ConvNets with depthwise convolutions, Global Response Normalization (GRN), and FCMAE self-supervised pretraining. InternImage uses deformable convolutions for dynamic receptive fields.

### Skin Lesion Performance
- ConvNeXt V2 + segmentation head: accuracy 93.60%, F1 90.73% on skin classification
- ConvNeXt V2 as encoder for segmentation: IoU 0.91-0.94 (various studies)
- Dual-encoder (ConvNeXt + Deformable Transformer): state-of-the-art boundary handling
- Modified EfficientNet-B7 with ASPP: IoU 0.9242 on HAM10000

### Small Image Suitability
- **Good**: ConvNeXt uses 4x4 patch stems (vs ViT 16x16)
- For 128x128: feature maps start at 32x32 -- good
- ConvNeXt V2 has tiny/small/base/large variants for different compute budgets
- InternImage uses deformable convolutions that adapt to content -- good for irregular lesion shapes

### Implementation (RECOMMENDED APPROACH)
```python
import segmentation_models_pytorch as smp

# Option 1: ConvNeXt V2 via timm
model = smp.UnetPlusPlus(
    encoder_name="tu-convnextv2_base",  # or convnextv2_tiny, convnextv2_large
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# Option 2: ConvNeXt V2 + DeepLabV3+
model = smp.DeepLabV3Plus(
    encoder_name="tu-convnextv2_base",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# Option 3: Original ConvNeXt (more widely tested)
model = smp.UnetPlusPlus(
    encoder_name="tu-convnext_base",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

# Check available ConvNeXt encoders
import timm
convnext_models = [m for m in timm.list_models("convnext*") if "v2" in m]
```

### Verdict
- **BEST OPTION for our hackathon**
- Drop-in replacement via SMP -- minimal code changes
- ConvNeXt V2 Base gives best performance/compute tradeoff
- Expected IoU boost: 0.01-0.03 over current encoders
- Training time: same as current pipeline

---

## 6. Cascade / Coarse-to-Fine Approaches

### Overview
Two-stage approaches: first generate a rough segmentation, then refine boundaries. This is particularly effective for skin lesions where boundary ambiguity is the main challenge.

### Key Architectures (2025)

**BASNet (Boundary-Aware Segmentation Network)**
- Prediction module + residual refinement module (4 stages)
- Refinement module corrects noisy/blurry boundaries
- Well-suited for skin lesions with fuzzy borders

**WA-NET**
- Boundary Refinement Module (BRM) with independent edge detection
- Enhanced Wavelet Transform (EWT) for frequency-domain features
- Excels at low-contrast boundary regions (common in skin lesions)

**SGNet (Structure-Guided Network)**
- VMamba encoder for multi-scale features
- Dual-Domain Boundary Enhancer (spatial + frequency)
- Structure-Aware Guidance Module generates coarse maps as global guidance
- Multi-scale refinement of boundary details

**CKDNet (Cascade Knowledge Diffusion)**
- Coarse segmentation -> classification -> fine segmentation
- Feature entanglement modules between stages
- Uses classification knowledge to improve segmentation

### Small Image Suitability
- **Excellent**: Coarse-to-fine inherently handles limited resolution
- Stage 1 captures global shape at low resolution
- Stage 2 refines local boundaries
- Can progressively upscale: 128 -> 256 -> 512 across stages

### Implementation Strategy for Our Pipeline
```python
# Stage 1: Use existing ensemble for coarse prediction
coarse_mask = ensemble_predict(image)  # our current 0.895 IoU model

# Stage 2: Refinement network
class BoundaryRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: image (3ch) + coarse_mask (1ch) = 4 channels
        self.refiner = smp.UnetPlusPlus(
            encoder_name="tu-convnextv2_tiny",
            encoder_weights="imagenet",
            in_channels=4,  # image + coarse mask
            classes=1
        )
        # Boundary detection head
        self.edge_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, image, coarse_mask):
        x = torch.cat([image, coarse_mask], dim=1)
        refined_mask = self.refiner(x)
        return refined_mask

# Stage 3: CRF post-processing (optional, fast)
import pydensecrf.densecrf as dcrf
def crf_refine(image, prob_mask, n_iters=5):
    """Dense CRF refinement for boundary sharpening"""
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    # ... standard CRF setup
    return refined_mask
```

### Verdict
- **Highest ceiling approach** (IoU 0.92-0.95)
- Can build on top of existing models without retraining from scratch
- Stage 2 refinement network is small and fast to train
- CRF post-processing adds ~0.01-0.02 IoU for free

---

## Practical Recommendations for Hackathon

### Priority 1: Quick Wins (30 min, expected +0.01-0.02 IoU)

1. **Try ConvNeXt V2 encoder** in existing SMP pipeline:
```python
# Just change encoder_name in your existing training code
model = smp.UnetPlusPlus(
    encoder_name="tu-convnextv2_base",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
```

2. **CRF post-processing** on ensemble output:
```bash
pip install pydensecrf
```

3. **Boundary-weighted loss** -- add weight to boundary pixels:
```python
def boundary_weighted_loss(pred, mask):
    # Compute boundary via dilation - erosion
    kernel = torch.ones(1, 1, 3, 3).to(mask.device)
    dilated = F.conv2d(mask, kernel, padding=1).clamp(0, 1)
    eroded = 1 - F.conv2d(1 - mask, kernel, padding=1).clamp(0, 1)
    boundary = dilated - eroded
    weight = 1.0 + 4.0 * boundary  # 5x weight on boundaries
    bce = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
    return bce
```

### Priority 2: Medium Effort (1-2 hours, expected +0.02-0.04 IoU)

4. **Two-stage refinement**: Train a small refiner that takes image + coarse mask as input
5. **Add SegFormer MiT encoder** to ensemble diversity
6. **Progressive upscaling**: Train at 256 -> fine-tune at 512

### Priority 3: High Effort (3+ hours, expected +0.03-0.05 IoU)

7. **Full boundary-aware network** (BASNet/WA-NET style refinement)
8. **SAM 2 as refinement oracle** (use coarse mask as prompt)
9. **Multi-task learning**: Joint segmentation + boundary detection

### What NOT To Do

- Do NOT try SAM 2 from scratch on 128x128 images -- waste of time
- Do NOT switch to Mask2Former for binary segmentation -- overkill
- Do NOT try InternImage -- limited SMP support, complex setup
- Do NOT try to reach 0.95 IoU with a single model -- ensemble + refinement is the path

---

## Realistic IoU Targets

| Approach | Expected IoU | Confidence |
|---|---|---|
| Current ensemble (baseline) | 0.895 | Known |
| + ConvNeXt V2 encoder swap | 0.905-0.915 | High |
| + CRF post-processing | 0.910-0.920 | High |
| + Boundary-weighted loss | 0.910-0.925 | Medium |
| + Two-stage refinement | 0.920-0.940 | Medium |
| + Full cascade pipeline | 0.930-0.950 | Medium-Low |

**Note**: ISIC 2018 state-of-the-art is IoU ~0.92 (Jaccard 89-92%). Reaching 0.95 IoU is extremely ambitious and exceeds published state-of-the-art on standard benchmarks. The best published Dice scores reach ~0.946 (Dice, not IoU), which corresponds to IoU ~0.898. Our target of 0.95 IoU would correspond to Dice ~0.974, which would be a new state-of-the-art result.

**Realistic target for hackathon**: 0.92-0.93 IoU with ensemble + refinement pipeline.
