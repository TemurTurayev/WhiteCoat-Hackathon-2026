# Asan Medical Center Skin Lesion Paper: Deep Analysis

## Paper: Han SS et al. (2018) "Classification of the Clinical Images for Benign and Malignant Cutaneous Tumors Using a Deep Learning Algorithm"
**Journal**: J Invest Dermatol. 2018;138(7):1529-1538
**DOI**: 10.1016/j.jid.2018.01.028 | PubMed: 29428356

---

## CRITICAL FINDING: YOUR HACKATHON DATASET IS ALMOST CERTAINLY THE ASAN DATASET

Evidence:
- Asan dataset: **12,209 images**, 12 classes → Your dataset: **11,411 train + 1,276 test = 12,687** (close match, slight differences from split/filtering)
- Asan: **12 identical disease classes** (see below)
- Asan test set from CAN repo: **1,276 images** → Your test set: **exactly 1,276 images**
- Images resized to ~101x101 thumbnails (Asan thumbnails are publicly available at this size on Figshare)

The 1,276 test images matching exactly confirms this is the Asan dataset.

---

## 1. Models and Accuracy (YOUR BASELINE TO BEAT)

### Original Han et al. 2018 Paper:
- **Architecture**: Microsoft ResNet-152, pretrained on ImageNet, fine-tuned
- **Training data**: 19,398 images total (Asan training portion + MED-NODE + atlas site images)
- **Additional trick**: 248 additional "distractor" classes added to reduce false positives and improve middle-layer representations. Total training involved classifying 260 classes (12 target + 248 distractor), but distractor classes did NOT outgrow the 12 main classes in sample count
- **Augmentation factor**: ~20-40x using zooming and rotation
- **Total augmented training images**: ~855,370
- **Training**: 2 epochs, batch size 6, learning rate 0.0001, no decay for 2 epochs, early stopping
- **Framework**: Caffe (not PyTorch/TensorFlow)

### Key AUC Results on Asan Test Set:

| Disease | AUC (Han 2018) |
|---------|---------------|
| Basal Cell Carcinoma | 0.96 +/- 0.01 |
| Squamous Cell Carcinoma | 0.83 +/- 0.01 |
| Intraepithelial Carcinoma | 0.82 +/- 0.02 |
| Melanoma | 0.96 +/- 0.00 |
| Actinic Keratosis | 0.83 |
| Seborrheic Keratosis | 0.89 |
| Melanocytic Nevus | 0.94 |
| Lentigo | 0.95 |
| Hemangioma | 0.83 |
| Dermatofibroma | 0.90 |
| Pyogenic Granuloma | 0.97 |
| Wart | 0.94 |

**Overall accuracy**: ~81% on Asan test set (reported in review papers)
**Average AUC**: ~0.91, average Sensitivity: 0.864, average Specificity: 0.855

### Mendes & da Silva 2018 (Replication/Extension - arxiv:1812.02316):
Used same ResNet-152 architecture, improved results on same 12-class problem:

| Disease | Han et al. AUC | Mendes AUC |
|---------|---------------|------------|
| Actinic Keratosis | 0.83 | **0.96** |
| Basal Cell Carcinoma | 0.90 | **0.91** |
| Dermatofibroma | 0.90 | **0.90** |
| Hemangioma | 0.83 | **0.99** |
| Intraepithelial Carcinoma | 0.83 | **0.99** |
| Lentigo | 0.95 | **0.95** |
| Malignant Melanoma | 0.88 (Edinburgh) / 0.96 (Asan) | **0.96** |
| Melanocytic Nevus | 0.94 | **0.95** |
| Pyogenic Granuloma | 0.97 | **0.99** |
| Seborrheic Keratosis | 0.89 | **0.90** |
| Squamous Cell Carcinoma | 0.91 | **0.95** |
| Wart | 0.94 | **0.89** |

**Overall accuracy (Mendes)**: 78% (11/12 classes above 80%, Actinic Keratosis was hardest)

---

## 2. Preprocessing

### Han 2018 (Original):
- Clinical photographs (not dermoscopic)
- Images resized to ResNet-152 standard input (224x224 pixels)
- No explicit segmentation/cropping of lesion pre-classification
- Standard ImageNet normalization

### Mendes 2018 (Replication):
- Test/train split: 10-20% for test, rest for training
- Training set further split: 80% train / 20% validation (stratified by class)
- Images stored in LMDB format for fast I/O
- No lesion-specific preprocessing (whole clinical photo used)

---

## 3. Augmentations

### Han 2018:
- Zooming and rotation
- Augmentation factor: 20-40x per image
- Total augmented dataset: ~855,370 images

### Mendes 2018 (More Detailed - Used Augmentor Python Library):

| Transformation | Probability |
|---------------|------------|
| Rotation | 0.5 |
| Random Zoom | 0.4 |
| Flip Horizontally | 0.7 |
| Flip Vertically | 0.5 |
| Random Distortion | 0.8 |
| Lightning (brightness) Variance | 0.5 |

- Augmentation factor: 29x per class
- Each augmented image has random combination of above transforms
- Total training: 88,090 train + 22,023 validation + 956 test = 111,069 images

---

## 4. Train/Test Split

### Han 2018:
- Training: Asan training portion + MED-NODE (170 images) + atlas images
- Testing: Asan test portion + Hallym dataset + Edinburgh dataset (1,300 images)
- Exact per-class split not fully public, but Asan total is ~12,209

### Mendes 2018 (per-class, after 29x augmentation):

| Lesion Type | Train | Validation | Test |
|------------|-------|-----------|------|
| Actinic Keratosis | 712 | 186 | 8 |
| Basal Cell Carcinoma | 30,067 | 7,517 | 324 |
| Dermatofibroma | 1,067 | 267 | 12 |
| Hemangioma | 1,601 | 400 | 18 |
| Intraepithelial Carcinoma | 1,299 | 325 | 14 |
| Lentigo | 1,137 | 284 | 13 |
| Malignant Melanoma | 6,218 | 1,554 | 68 |
| Melanocytic Nevus | 17,632 | 4,408 | 191 |
| Pyogenic Granuloma | 371 | 93 | 5 |
| Seborrheic Keratosis | 19,256 | 4,814 | 208 |
| Squamous Cell Carcinoma | 1,462 | 365 | 16 |
| Wart | 7,238 | 1,810 | 79 |
| **TOTAL** | **88,090** | **22,023** | **956** |

Note: The Mendes split used MED-NODE + Edinburgh + Atlas data, not the original Asan images.

---

## 5. Per-Class Accuracy: Hardest Classes

### Hardest to classify (lowest AUC):
1. **Intraepithelial Carcinoma** - AUC 0.82 (Han) - often confused with SCC and AK
2. **Squamous Cell Carcinoma** - AUC 0.83 (Han) - confused with intraepithelial carcinoma
3. **Actinic Keratosis** - AUC 0.83 (Han) - confused with SCC (it's a pre-cancerous lesion)
4. **Hemangioma** - AUC 0.83 (Han) - confused with pyogenic granuloma (both vascular)

### Easiest to classify (highest AUC):
1. **Melanoma** - AUC 0.96 (distinctive features)
2. **Basal Cell Carcinoma** - AUC 0.96 (distinctive morphology)
3. **Pyogenic Granuloma** - AUC 0.97 (Mendes: 0.99)
4. **Lentigo** - AUC 0.95

### Key Confusion Patterns:
- SCC vs. Intraepithelial Carcinoma (biologically related - IC is early-stage SCC)
- Actinic Keratosis vs. SCC (AK is precursor to SCC)
- Hemangioma vs. Pyogenic Granuloma (both red vascular lesions)
- Melanoma vs. Melanocytic Nevus (critical diagnostic distinction)

---

## 6. Ensemble Methods

### Han 2018:
- **Single model** (ResNet-152) in the original paper, NOT an ensemble
- However, the trick of adding 248 distractor classes effectively acted as a regularizer

### Han 2020 Follow-up (PLOS Medicine):
- Used **SENet and SE-ResNeXt-50** as disease classifiers
- Combined with **Faster R-CNN** for lesion detection
- 3-stage pipeline: blob detection → fine image selection → disease classification
- 178 disease classes, trained on **1,106,886 cropped images**
- This IS closer to an ensemble/pipeline approach

### ModelDerm (Current Production System by same authors):
- **Ensemble of CNNs** classifying 184 skin diseases
- Build2024 version used during 2024-2025
- Planet-wide validated (published npj Digital Medicine 2025)

---

## 7. Human Dermatologist Accuracy

### Han 2018:
- 16 dermatologists tested on 480 images (Asan + Edinburgh)
- Algorithm performance was **"comparable to that of 16 dermatologists"**
- No exact dermatologist accuracy numbers published in abstract

### Han 2020 (Detailed Comparison):
- **65 attending physicians** in real-world practice:
  - Top-1 diagnosis sensitivity: 70.2%, specificity: 95.6%
  - Top-3 diagnosis sensitivity: 88.1%, specificity: 83.8%
- **44 dermatologists** (image-only reader test):
  - Algorithm comparable (p=0.607 sensitivity, p=0.097 specificity)
- **Algorithm** on Severance dataset:
  - Binary (malignant/benign) AUC: 0.863
  - High-specificity mode: Sensitivity 62.7%, Specificity 90.0%
  - High-sensitivity mode: Sensitivity 79.1%, Specificity 76.9%
- **Multiclass (32 diseases)**:
  - Algorithm Top-1: 42.6%, Top-3: 61.9%
  - Attending physicians Top-1: 65.4%, Top-3: 74.7%
  - (Physicians outperformed AI on multiclass due to metadata access)

---

## 8. Follow-up Papers and Improved Results

### By Same Group (Han / whria78):

1. **Han et al. 2020** - "Augmented Intelligence Dermatology" - J Invest Dermatol
   - 220,680 training images, 174 disorders
   - Validated on Edinburgh (1,300 images) and SNU (2,201 images; 134 disorders)

2. **Han et al. 2020** - "Assessment of deep neural networks" - PLOS Medicine
   - SENet + SE-ResNeXt-50 + Faster R-CNN
   - 1,106,886 training images, 178 classes
   - Mean AUC 0.931 across 32 classes on Severance dataset

3. **Han et al. 2023** - CAN dataset paper - JAMA Dermatology
   - EfficientNet-Lite0 trained on CAN5600 (5,619 annotated images)
   - Plus GAN-synthesized images (StyleGAN2-ADA)
   - Achieved higher/equivalent AUC to models trained on pathology-confirmed datasets

4. **Han et al. 2025** - "Planet-wide performance" - npj Digital Medicine
   - ModelDerm Build2024
   - Korea skin cancer sensitivity: 78.2% (NIA), specificity: 88.0%
   - Top-1 accuracy for 70 diseases: 43.3%, Top-3: 66.6%

### By Other Groups (Improvements):

5. **Mendes & da Silva 2018** (arxiv:1812.02316)
   - Same ResNet-152 on same 12 classes
   - Improved most classes, overall AUC better than Han original
   - Added GradCAM interpretability

6. **Recent State-of-the-Art (2024-2025)**:
   - Ensemble of Swin Transformer + ViT + EfficientNetB4: **98.5% accuracy** (on CSMUH dataset, not Asan)
   - Hybrid EfficientNet + Swin Transformer models
   - Multi-scale attention + ensemble: **95.05%** on ISIC2018

---

## 9. Image Resolution

### Original Asan Dataset:
- Original clinical photographs: **high resolution** (exact px not consistently published, likely 1000+ px)
- Publicly available thumbnails: **~100x100 px** (this is what your hackathon uses!)
- The Figshare thumbnails: https://figshare.com/articles/figure/Asan_and_Hallym_Dataset_Thumbnails_/5406136

### Input to Models:
- Han 2018: Resized to **224x224** (standard ResNet-152 input)
- They likely had access to full-resolution originals at Asan Medical Center
- Your hackathon images at ~101x101 are the **thumbnail versions**, NOT the originals

### IMPLICATION FOR HACKATHON:
- You are working with heavily downsampled thumbnails (~101x101)
- The original paper used 224x224 input (from higher-res originals)
- This means published AUC scores were achieved on BETTER quality images
- Your accuracy will likely be LOWER than published results due to resolution loss
- Upsampling from 101 to 224 will introduce artifacts but is standard practice

---

## 10. Dataset Availability

### Public Access:
- **Thumbnails (~100x100)**: Publicly available on Figshare
  - URL: https://figshare.com/articles/figure/Asan_and_Hallym_Dataset_Thumbnails_/5406136
- **Test subset (1,276 Asan + 152 Hallym images)**: Available in CAN repo
  - URL: https://figshare.com/articles/software/Caffemodel_files_and_Python_Examples/5406223
- **CAN dataset** (melanoma/nevus only): https://github.com/whria78/can (MIT license)
  - Includes Asan test folder under [DATA/asan]

### Restricted Access:
- **Full-resolution Asan images**: NOT publicly available
  - Require IRB approval and direct request to Seung Seog Han (whria78@gmail.com)
  - Privacy regulations restrict full dataset sharing
- **Severance dataset**: Thumbnails + demographics available; full images require IRB

### The Dataset IS Private to Asan Medical Center:
- The training portion of Asan dataset was never fully released publicly
- Only thumbnails and test subsets are available
- This is why the hackathon uses ~101x101 thumbnail versions

---

## Kaggle Competitions & GitHub Repos

### Kaggle:
- **No specific Kaggle competition** uses the Asan dataset directly
- ISIC challenges (2017-2024) use dermoscopic images (different from clinical Asan images)
- ISIC 2024 had 900K+ images, but these are dermoscopic, not clinical

### GitHub Repositories:
1. **whria78/can** - Official CAN dataset repo (MIT license)
   - https://github.com/whria78/can
   - Includes EfficientNet-Lite0 training code (`train.py`)
   - Asan test data included
   - Training params: batch 64, 30 epochs, lr 0.001, RAdam optimizer

2. **whria78/modelderm_rcnn_api** - Model Dermatology with Region-based CNN API
   - https://github.com/whria78/modelderm_rcnn_api

3. **whria78/skinimagecrawler** - Tool to crawl skin images from web
   - https://github.com/whria78/skinimagecrawler

4. **No specific repo** replicating Han 2018 results on Asan 12-class with modern architectures

---

## PRACTICAL RECOMMENDATIONS FOR HACKATHON

### What Accuracy is "Good" (to tell judges):
- **Han 2018 baseline**: ~81% overall accuracy, ~0.91 average AUC
- **Mendes 2018 replication**: ~78% accuracy (but higher per-class AUCs on many classes)
- **With thumbnail resolution** (~101x101): Expect 5-15% lower than published results
- **Realistic target**: 70-75% accuracy would be competitive given thumbnail quality
- **Impressive target**: 80%+ accuracy would match/beat the original paper (on degraded images)
- **If you get 85%+**: This would be genuinely impressive and publication-worthy

### Tricks That Work for This Specific Data:
1. **Add distractor classes** during training (Han's key innovation - 248 extra classes)
2. **Heavy augmentation** (29-40x factor): rotation, zoom, flip, distortion, brightness
3. **Transfer learning** from ImageNet is essential (ResNet-152 or EfficientNet)
4. **Class imbalance handling**: Oversample rare classes (Actinic Keratosis, Pyogenic Granuloma)
5. **Focus on hard pairs**: SCC vs IC, AK vs SCC, Hemangioma vs PG
6. **Ensemble models** (combine CNN + Transformer for best results)
7. **Test-time augmentation** (predict on multiple augmented versions, average)
8. **GradCAM/interpretability** adds value for judges (medical AI needs explainability)

### Model Recommendations for 24h Hackathon:
1. **EfficientNet-B0/B3** - Best efficiency/accuracy tradeoff (used by Han's group in CAN paper)
2. **ResNet-152** - Original baseline, proven on this exact dataset
3. **Swin Transformer** - State-of-the-art for medical imaging
4. **Ensemble of 2-3 models** - Best if you have time

### Resolution Strategy:
- Upsample 101x101 → 224x224 (bilinear/bicubic interpolation) for standard models
- Or use 128x128 if using custom architecture
- Super-resolution preprocessing could help (ESRGAN) but may not be worth the time cost

---

## Key Sources

- PubMed: https://pubmed.ncbi.nlm.nih.gov/29428356/
- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0022202X18301118
- CAN Dataset: https://github.com/whria78/can
- Asan Thumbnails: https://figshare.com/articles/figure/Asan_and_Hallym_Dataset_Thumbnails_/5406136
- Han 2020 PLOS Med: https://pmc.ncbi.nlm.nih.gov/articles/PMC7688128/
- Mendes 2018 (replication): https://arxiv.org/abs/1812.02316
- ModelDerm Algorithm: https://modelderm.com/index.php/algorithm/index.html
- Han 2025 npj Digital Med: https://www.nature.com/articles/s41746-025-01980-w
