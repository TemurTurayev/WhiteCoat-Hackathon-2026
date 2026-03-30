# Competition Solutions: 0.90+ IoU Skin Lesion Segmentation

## Table of Contents
1. [ISIC Challenge Official Leaderboards](#isic-challenge-official-leaderboards)
2. [Top Competition Solutions with Architecture Details](#top-competition-solutions)
3. [Recent SOTA Papers (2024-2026)](#recent-sota-papers-2024-2026)
4. [Asan Dataset / Han et al.](#asan-dataset)
5. [Theoretical Maximum IoU & Annotation Noise Ceiling](#theoretical-maximum)
6. [Practical Takeaways for Our Hackathon](#practical-takeaways)

---

## 1. ISIC Challenge Official Leaderboards <a name="isic-challenge-official-leaderboards"></a>

### ISIC 2017 Task 1 — Lesion Segmentation

| Rank | Team/Author | Thresholded Jaccard | Method |
|------|-------------|--------------------:|--------|
| 1 | Yading Yuan | 0.784 | Fully Convolutional-Deconvolutional Network |
| 2 | Matt Berseth | ~0.765 | Deep CNN encoder-decoder |

**Winner details (Yuan et al.):**
- Architecture: 19-layer deep fully convolutional-deconvolutional network, 29 layers total, 5,042,589 params
- Multi-color-space input (RGB + additional color channels)
- End-to-end training, no prior knowledge required
- Jaccard distance as loss function
- Paper: arXiv:1703.05165

### ISIC 2018 Task 1 — Lesion Boundary Segmentation (Official Leaderboard)

| Rank | Team | Jaccard Index | Method |
|------|------|-------------:|--------|
| 1 | MT (Meitu/Chengyao Qian) | 0.802 | MaskRCNN + segmentation ensemble |
| 2 | Holidayburned | 0.799 | Ensemble with CRF v3 |
| 3 | imsight | 0.799 | Automatic Skin Lesion Seg by DCNN |
| 4 | Tencent Youtu Lab | 0.798 | Skin Lesion Seg with Adversarial Learning |
| 5 | NMN_team | 0.796 | Ensemble ALL (Th=0.80, Tl=0.65) |
| 6 | GPM-UC3M | 0.788 | SR FCN Init 2 |
| 7 | Andrey Sorokin | 0.779 | Mask-RCNN with SGD |
| 8 | DC | 0.777 | MaskRCNN |
| 9 | Weill Cornell Medicine | 0.773 | Deep UNet |
| 10 | Opsins | 0.771 | Transfer learning CNN segmentation |

**1st Place (MT/Meitu) details:**
- Two-stage approach: detection + segmentation
- Backbone: Extended ResNet-101 with three cascading blocks
- Modified ASPP (Atrous Spatial Pyramid Pooling) with dense ASPP, standard conv, pooling
- MaskRCNN segmentation branch for supervision
- Ensemble post-processing
- Paper: arXiv:1809.03917

**2nd Place (Holidayburned) details:**
- Multi-model ensemble
- Conditional Random Field (CRF) post-processing
- Threshold optimization

**Key observation:** Top ISIC 2018 challenge Jaccard was ~0.80. Note: challenge test set is harder than typical validation splits due to diverse/difficult cases.

---

## 2. Top Competition Solutions with Architecture Details <a name="top-competition-solutions"></a>

### Solution A: Yuan et al. (ISIC 2017 Winner)
- **Architecture:** Custom 19-layer FCDN (Fully Convolutional-Deconvolutional Network)
- **Encoder:** Custom deep CNN (not pretrained ImageNet)
- **Resolution:** 192x256
- **Loss:** Jaccard distance loss
- **Training:** End-to-end, multi-color-space input
- **Post-processing:** Thresholding
- **Score:** Jaccard 0.784 (ISIC 2017 test)
- **Code:** Not publicly released

### Solution B: MT/Meitu (ISIC 2018 Winner)
- **Architecture:** Two-stage: Mask R-CNN detection + segmentation
- **Encoder:** Extended ResNet-101
- **Key module:** Modified ASPP (dense ASPP + conv + pooling)
- **Loss:** Multi-task (detection + segmentation)
- **Post-processing:** Ensemble of multiple runs + post-processing pipeline
- **Score:** Jaccard 0.802 (ISIC 2018 test)
- **Paper:** arXiv:1809.03917

### Solution C: Holidayburned (ISIC 2018 2nd Place)
- **Architecture:** Ensemble of multiple segmentation models
- **Post-processing:** Conditional Random Field (CRF)
- **Score:** Jaccard 0.799 (ISIC 2018 test)

### Solution D: Tencent Youtu Lab (ISIC 2018 4th Place)
- **Architecture:** Adversarial learning for skin lesion segmentation
- **Approach:** GAN-based training with discriminator for boundary refinement
- **Score:** Jaccard 0.798 (ISIC 2018 test)

### Solution E: EfficientUNet++ (PH2 benchmark)
- **Architecture:** U-Net++ with EfficientNet-B7 encoder
- **Score:** Dice 0.93, IoU 0.96 on PH2 dataset
- **Note:** PH2 is a small (200 images), clean dataset -- scores are inflated vs. ISIC test sets

### Solution F: Ensemble of SegNet + DeepLabV3 + U-Net
- **Architecture:** Three-model ensemble
- **Post-processing:** Weighted averaging + thresholding at 0.5
- **Score:** Dice 0.93, IoU 0.90 (ISIC 2017)
- **Paper:** Referenced in enhanced ensemble models study

---

## 3. Recent SOTA Papers (2024-2026) <a name="recent-sota-papers-2024-2026"></a>

### 3a. SkinFormNet (2026) -- Highest Reported Dice
- **Architecture:** SegFormer feature extraction + U-Net + attention mechanism
- **Classification:** GlobalSkinNet (Global Contextual Vision Transformer)
- **Results by dataset:**

| Dataset | Dice | IoU |
|---------|-----:|----:|
| PH2 | 0.97 | ~0.94 |
| ISIC-2016 | 0.98 | ~0.96 |
| ISIC-2017 | 0.94 | ~0.89 |
| ISIC-2018 | 0.99 | ~0.98 |
| HAM10000 | 0.96 | ~0.92 |

- **Paper:** Scientific Reports (2026), "Advanced hybrid transformer CNN framework"
- **CAUTION:** These numbers are unusually high (0.99 Dice on ISIC-2018). The paper uses the ISIC-2018 *training/val split*, NOT the official challenge test set. Direct comparison to leaderboard scores (Jaccard 0.80) is misleading.

### 3b. Deep_SOTA_Net-B7 / Dual-Branch Framework (2025)
- **Architecture:** Modified EfficientNet-B7 encoder + ASPP + transformer blocks
- **Decoder:** Attention gates + Squeeze-and-Excitation blocks
- **Results on HAM10000:** Dice 0.9568, IoU 0.9242, Accuracy 0.9708
- **Paper:** Scientific Reports (2025), "A deep learning-based dual-branch framework"

### 3c. SAM-ViT Framework (2025)
- **Architecture:** SAM-Adapter fine-tuning + ViT classifier with cross-attention fusion
- **Approach:** Lesion-specific cropping from segmentation + classification
- **Results:**
  - ISIC 2018: Dice 0.9427
  - PH2: Dice 0.9562, IoU 0.9291
- **Paper:** Diagnostics (2025), "Foundation-Model-Driven Skin Lesion Segmentation"

### 3d. SkinSAM (2023-2024)
- **Architecture:** SAM (ViT-B, ViT-L, ViT-H) fine-tuned on HAM10000
- **Best model:** ViT_b_finetuned
- **Results:** Mean pixel accuracy 0.945, Mean Dice 0.8879, Mean IoU 0.7843
- **Note:** Full SAM fine-tuning on medical data still underperforms specialized architectures
- **Paper:** arXiv:2304.13973

### 3e. MPBA-Net / CNN-Transformer Fusion (2025)
- **Architecture:** CNN + Transformer parallel branches with boundary-aware loss
- **Results:**
  - ISIC-2016: Dice 0.9272
  - ISIC-2017: Dice 0.8910
  - ISIC-2018: Dice 0.9067
  - PH2: Dice 0.9483
- **Paper:** ScienceDirect (2025), "Lesion boundary detection for skin lesion segmentation"

### 3f. MAFF-Net (2025)
- **Architecture:** SAM ViT-H (Hiera) frozen encoder + lightweight adapters + frequency-guided fusion
- **Approach:** Mixed multi-scale perception adaptation, frozen SAM backbone
- **Results:** Leading performance on ISIC series
- **Paper:** ScienceDirect (2025)

### 3g. Melanoma segmentation with TTA + CRF (2022)
- **Architecture:** Deep learning model + TTA + CRF post-processing
- **Pipeline:** Hair removal (morphological + inpainting) -> segmentation -> TTA -> CRF
- **Results:** Significant improvement from CRF and TTA over baseline
- **Paper:** Scientific Reports (2022)

---

## 4. Asan Dataset / Han et al. <a name="asan-dataset"></a>

Han et al. (Asan Medical Center, Seoul) created the Asan dataset:
- **Task:** Classification (not segmentation) of clinical skin photos
- **Architecture:** ResNet-152, fine-tuned on 19,398 training images
- **Classes:** 12 skin diseases (basal cell carcinoma, SCC, melanoma, etc.)
- **Results:** AUC 0.96 for BCC, 0.96 for melanoma, 0.83 for SCC, 0.82 for intraepithelial carcinoma
- **Dataset:** Open-access (Asan + Hallym Dataset)
- **Note:** This dataset is for CLASSIFICATION, not segmentation. No segmentation masks are provided.
- **Relevance:** The 12-class setup is similar to our hackathon Task 1.

---

## 5. Theoretical Maximum IoU & Annotation Noise Ceiling <a name="theoretical-maximum"></a>

### Inter-Annotator Agreement Data (IMA++ Dataset, 2025)

The IMA++ dataset (arXiv:2508.09381, arXiv:2512.21472) is the largest multi-annotator skin lesion segmentation dataset:
- **2,394 dermoscopic images** segmented by **15 unique annotators** -> 5,111 masks total

**Measured inter-annotator agreement (pairwise Dice):**

| Lesion Type | Mean Dice +/- SD |
|-------------|------------------|
| Benign | 0.791 +/- 0.215 |
| Malignant | 0.753 +/- 0.227 |
| Overall | ~0.77 +/- 0.22 |

**Distribution details:**
- 818 out of 2,394 images (34%) have mean Dice above 0.90
- 344 out of 2,394 images (14%) have mean Dice above 0.95
- 23 images have Dice of 0.0 (complete disagreement)
- Statistically significant (p<0.001) association between IAA and malignancy

### What This Means for Theoretical Maximum

1. **Human-level ceiling on ISIC-style data is ~0.85-0.90 Dice (0.75-0.82 IoU)** when measured against a single ground truth annotation. This matches the ISIC 2018 leaderboard top score of Jaccard 0.802.

2. **Per-image variability is enormous.** On "easy" benign lesions with clear borders, Dice > 0.95 is achievable. On ambiguous malignant lesions, even experts disagree by 0.20+ Dice.

3. **Claims of 0.95+ Dice on ISIC-2018 in papers** almost always use the TRAINING split (2,594 images) with a random train/val split, NOT the official challenge test set (1,000 hard images). The official test set contains deliberately difficult cases.

4. **PH2 dataset (200 images)** has cleaner annotations and simpler cases. 0.95+ Dice on PH2 is achievable and realistic.

5. **HAM10000 (10,015 images)** uses the ISIC-2018 training data. Reports of 0.95+ Dice on HAM10000 are therefore on the "easier" subset.

### Summary: Is 0.95 IoU Realistic?

| Scenario | Realistic Target |
|----------|-----------------|
| ISIC official challenge test set (hard) | Jaccard 0.80-0.82 = WORLD CLASS |
| ISIC training data with random split | Dice 0.90-0.93, IoU 0.82-0.87 |
| PH2 (small, clean dataset) | Dice 0.95-0.97, IoU 0.90-0.95 |
| HAM10000 random split | Dice 0.92-0.96, IoU 0.85-0.93 |
| Hackathon data (unknown difficulty) | Dice 0.88-0.93, IoU 0.80-0.88 likely achievable |
| Annotation noise ceiling | Mean Dice ~0.77 between experts |

**Bottom line:** 0.95 IoU (= ~0.975 Dice) is NOT realistic on challenging test sets. It is only seen on small/clean datasets (PH2) or when models are evaluated on easy train/val splits. On a hackathon test set with 200 images of mixed difficulty, a realistic world-class target is **IoU 0.85-0.90** (Dice 0.91-0.95).

---

## 6. Practical Takeaways for Our Hackathon <a name="practical-takeaways"></a>

### What the Winners Actually Did (Distilled)

1. **Encoder:** ResNet-101 or EfficientNet-B7 (pretrained ImageNet) -- both work
2. **Architecture:** U-Net++ or DeepLabV3+ or Mask R-CNN segmentation head
3. **Multi-scale:** ASPP or pyramid pooling for multi-scale features
4. **Loss:** Combination of BCE + Dice loss (or Jaccard loss). Top solutions add boundary-aware terms
5. **Ensemble:** 3-5 models with different architectures or folds, average predictions
6. **TTA:** Horizontal flip + vertical flip + rotations (90/180/270) -- 4-8x augmentation at inference
7. **Post-processing:**
   - CRF (Conditional Random Field) -- +1-2% Jaccard
   - Morphological operations (opening/closing to clean small artifacts)
   - Connected component analysis (keep largest component)
   - Threshold optimization on validation set
8. **Resolution:** 384x384 or 512x512 (higher helps but costs more)
9. **Augmentation:** Moderate color jitter, flips, rotations, elastic transforms, CLAHE
10. **Hair removal:** Morphological black-hat filter + inpainting (DermHair approach)

### Priority Actions for Maximum IoU Gain

| Priority | Action | Expected Gain |
|----------|--------|---------------|
| 1 | Train U-Net++ with EfficientNet-B4/B7 encoder at 384+ resolution | Baseline |
| 2 | Add Dice+BCE combined loss | +2-3% over BCE alone |
| 3 | Add TTA (8x: flips + rotations) | +1-3% |
| 4 | Ensemble 2-3 models (different encoders or architectures) | +1-3% |
| 5 | CRF post-processing | +0.5-2% |
| 6 | Morphological cleanup (remove small components, fill holes) | +0.5-1% |
| 7 | Threshold optimization on val set | +0.5-1% |
| 8 | Higher resolution (512x512) | +0.5-1% |
| 9 | Lovasz-Softmax loss addition | +0.5-1% |
| 10 | Hair removal preprocessing | +0.5-1% on hairy images |

### Recommended Ensemble Configuration

```
Model 1: U-Net++    / EfficientNet-B4  / 384x384 / Dice+BCE loss
Model 2: DeepLabV3+ / ResNet-101       / 384x384 / Dice+BCE+Lovasz loss
Model 3: U-Net++    / SE-ResNeXt-50    / 512x512 / Dice+Focal loss

Inference: Average predictions from all 3 models x 8 TTA -> threshold -> morphological cleanup
```

### Post-Processing Pipeline

```python
# 1. Model ensemble prediction (soft probabilities)
pred = (model1_tta + model2_tta + model3_tta) / 3.0

# 2. Optimal threshold (tuned on validation)
binary = (pred > optimal_threshold).astype(np.uint8)

# 3. Morphological cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # fill holes
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove noise

# 4. Keep largest connected component only
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest = max(contours, key=cv2.contourArea)
    binary = np.zeros_like(binary)
    cv2.drawContours(binary, [largest], -1, 1, -1)

# 5. Optional: CRF refinement (adds ~1% but slow)
# from pydensecrf import densecrf
# binary = apply_crf(image, pred, binary)

# 6. Scale to 0/255 for submission
mask = binary * 255
```

---

## References

- ISIC 2018 Leaderboard: https://challenge.isic-archive.com/leaderboards/2018/
- ISIC 2017 Challenge: https://challenge.isic-archive.com/landing/2017/
- Yuan (2017) FCDN: arXiv:1703.05165
- MT/Meitu (2018) Two-stage: arXiv:1809.03917
- ISIC 2018 Challenge Paper: arXiv:1902.03368
- IMA++ Inter-Annotator Study: arXiv:2508.09381
- IMA++ Dataset Paper: arXiv:2512.21472
- SkinSAM: arXiv:2304.13973
- FAT-Net: https://www.sciencedirect.com/science/article/abs/pii/S1361841521003728
- SkinFormNet (2026): Scientific Reports, "Advanced hybrid transformer CNN framework"
- Deep_SOTA_Net-B7 (2025): Scientific Reports, "A deep learning-based dual-branch framework"
- SAM-ViT (2025): Diagnostics 16(3):468
- TTA+CRF for melanoma: Scientific Reports (2022)
- Han et al. Asan Dataset: Open-access, classification only (not segmentation)

---

*Compiled: 2026-03-27 for WhiteCoat.dev Hackathon*
