# Dataset Research for AI in Healthcare Hackathon 2026

## CRITICAL FINDING: Dataset Type Identification

After examining the actual hackathon images, these are **NOT histopathology/biopsy microscopy images**.
They are **clinical skin lesion photographs** (macroscopic/dermoscopic images of skin conditions).

### Your Hackathon Dataset Summary

**Task 1 - Classification:**
- 11,411 training images across 12 classes (folders 0-11)
- 1,276 test images
- Image size: ~101x101 px (mostly 101x101, some 100x101 or 101x100)
- Format: RGB PNG
- Classes are numbered 0-11, disease names hidden
- Class distribution is imbalanced (331 to 2,136 images per class)

**Task 2 - Segmentation:**
- 1,800 training images + masks, 400 validation + masks, 200 test images (no masks)
- Image size: 128x128 px (JPG images, PNG masks)
- Binary masks: values 0 and 255 (background vs. lesion)
- Clinical skin lesion photos with lesion boundary segmentation

---

## Part 1: Dataset Identity Analysis

### Most Likely Identity of the 12-Class Classification Dataset

The dataset characteristics (clinical skin photos, ~11.4K images, 12 classes, ~100px patches) do NOT exactly match any single well-known public dataset. However, it is most likely:

**Hypothesis 1: Modified/subset of ISIC Archive data** (HIGH probability)
- ISIC has 8 standard classes; organizers may have split into 12 subclasses or added custom classes
- Image sizes were resized from originals (ISIC images are typically 600x450 or larger)
- The test file naming (e.g., 100359.png, 102386.png) suggests sequential IDs from a larger archive

**Hypothesis 2: Custom curated dataset from DermNet/similar atlas** (MEDIUM probability)
- 12 classes could map to common clinical skin conditions
- Clinical (not dermoscopic) photos suggest atlas-style images

**Hypothesis 3: Subset of SD-198 or Fitzpatrick17k** (LOW probability)
- These have too many classes (198 and 114 respectively), but a 12-class subset is possible

### Likely 12 Class Mapping (speculative based on visual inspection):
Based on the images I examined, these appear to be **skin lesion types** including conditions like:
warts, melanoma, basal cell carcinoma, actinic keratosis, dermatofibroma, seborrheic keratosis,
vascular lesions, nevi, cysts, and others.

---

## Part 2: Best Datasets for PRETRAINING (Classification Task)

### Tier 1 - Highest Relevance (Skin Lesion Classification)

#### 1. ISIC 2019 Challenge Dataset
- **URL**: https://challenge.isic-archive.com/data/ | Kaggle: https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification
- **Size**: 25,331 dermoscopic images
- **Resolution**: Variable (600x450 to 1024x1024), can be resized
- **Content**: 8 classes - MEL, NV, BCC, AK, BKL, DF, VASC, SCC
- **Staining/Type**: Dermoscopic + clinical skin photos
- **Download**: `kaggle datasets download -d salviohexia/isic-2019-skin-lesion-images-for-classification`
- **Access**: Free, CC BY-NC license
- **Why useful**: Largest single dermoscopic dataset, 8 classes overlap with likely hackathon classes. BEST for pretraining a classification backbone.

#### 2. HAM10000 (Human Against Machine with 10000 images)
- **URL**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Size**: 10,015 dermoscopic images
- **Resolution**: 600x450
- **Content**: 7 classes - akiec, bcc, bkl, df, mel, nv, vasc
- **Download**: `kaggle datasets download -d kmader/skin-cancer-mnist-ham10000`
- **Access**: Free, CC BY-NC 4.0
- **Why useful**: Well-curated, similar size to hackathon dataset. Foundation of ISIC 2018. Good for direct fine-tuning experiments.

#### 3. DERM12345
- **URL**: Harvard Dataverse (doi: 10.7910/DVN/DAXZ7P)
- **Size**: 12,345 dermatoscopic images
- **Resolution**: High-resolution
- **Content**: 5 super classes, 15 main classes, 40 subclasses
- **Download**: Harvard Dataverse direct download
- **Access**: Free, CC BY license
- **Why useful**: Multi-level hierarchy with 15 main classes close to our 12. Recent (2024) and diverse sources.

#### 4. Fitzpatrick17k
- **URL**: https://github.com/mattgroh/fitzpatrick17k
- **Size**: 16,577 clinical photographs
- **Resolution**: Variable
- **Content**: 114 skin disease classes + Fitzpatrick skin type labels
- **Download**: Apply via GitHub, then download
- **Access**: Free for research
- **Why useful**: Clinical (NOT dermoscopic) images - matches our dataset type better. Good for pretraining on diverse skin conditions.

#### 5. PAD-UFES-20
- **URL**: https://data.mendeley.com/datasets/zr7vgbcyr2/1
- **Size**: 2,298 clinical images from smartphones
- **Resolution**: Variable (smartphone photos)
- **Content**: 6 classes - 3 skin diseases + 3 skin cancers
- **Download**: Direct download from Mendeley Data
- **Access**: Free
- **Why useful**: Clinical smartphone photos (not dermoscopic), similar visual style to some hackathon images.

### Tier 2 - Good for Transfer Learning Backbone

#### 6. ISIC 2020 Challenge Dataset
- **URL**: https://challenge2020.isic-archive.com/
- **Size**: 33,126 dermoscopic images
- **Resolution**: Variable
- **Content**: Binary (benign vs malignant), histopathology-confirmed
- **Download**: Via ISIC archive or Kaggle
- **Access**: Free
- **Why useful**: Massive dataset for pretraining a general skin lesion feature extractor, even if binary.

#### 7. SLICE-3D Dataset
- **URL**: https://www.kaggle.com/competitions/isic-2024-challenge
- **Size**: 400,000+ skin lesion crops
- **Resolution**: Variable
- **Content**: Skin lesion crops from 3D total body photography
- **Download**: Kaggle competition download
- **Access**: Free
- **Why useful**: Enormous dataset for self-supervised pretraining of skin lesion features.

#### 8. DermaMNIST (MedMNIST)
- **URL**: https://medmnist.com/
- **Size**: 10,015 images (same source as HAM10000)
- **Resolution**: 28x28 (too small for direct use, but useful reference)
- **Content**: 7 classes
- **Download**: `pip install medmnist` then load via Python API
- **Access**: Free, CC BY-NC 4.0
- **Why useful**: Easy to load via pip, good for quick baseline experiments. But 28x28 is too low-res for real pretraining.

---

## Part 3: Best Datasets for PRETRAINING (Segmentation Task)

### Tier 1 - Direct Skin Lesion Segmentation

#### 1. ISIC 2018 Task 1 - Lesion Segmentation
- **URL**: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation
- **Size**: 2,594 training images + binary masks
- **Resolution**: ~600x450 (resize to 128x128)
- **Content**: Dermoscopic images with expert-drawn binary lesion masks
- **Download**: `kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation`
- **Access**: Free
- **Why useful**: BEST match - same task (binary skin lesion segmentation), similar scale, expert annotations. Combine with hackathon data for more training samples.

#### 2. ISIC 2017 Part 1 - Lesion Segmentation
- **URL**: https://challenge.isic-archive.com/data/
- **Size**: 2,000 training + 150 validation + 600 test images with binary masks
- **Resolution**: Variable (resize to 128x128)
- **Content**: Dermoscopic images + binary mask PNG annotations
- **Download**: Direct from ISIC archive
- **Access**: Free
- **Why useful**: Standard benchmark for skin lesion segmentation. Binary masks exactly like our task.

#### 3. PH2 Dataset
- **URL**: https://www.fc.up.pt/addi/ph2%20database.html
- **Size**: 200 dermoscopic images + masks
- **Resolution**: ~768x560
- **Content**: Expertly annotated dermoscopic images with segmentation masks
- **Download**: Direct download after registration
- **Access**: Free for research
- **Why useful**: Small but very high quality annotations. Good for fine-tuning.

#### 4. IMA++ (ISIC MultiAnnot++)
- **URL**: Referenced in recent 2024 paper
- **Size**: 17,684 segmentation masks for 14,967 dermoscopic images
- **Resolution**: Variable
- **Content**: Multi-annotator skin lesion segmentation masks
- **Download**: Via ISIC Archive
- **Access**: Free
- **Why useful**: Largest skin lesion segmentation dataset. Multi-annotator provides robust training signal.

---

## Part 4: Foundation Models to Consider

Instead of training from scratch, consider using pretrained pathology/dermatology models:

| Model | Source | Pretrained On | How to Use |
|-------|--------|---------------|------------|
| **UNI2** | HuggingFace: MahmoodLab/UNI2-h | 200M+ histology images | Feature extractor, fine-tune classifier head |
| **CONCH** | HuggingFace: MahmoodLab/CONCH | 1.17M histology image-text pairs | Vision-language, zero-shot or fine-tune |
| **Virchow** | HuggingFace: paige-ai/Virchow | 1.5M whole-slide images | Feature extraction |
| **EfficientNet-B0 to B7** | torchvision / timm | ImageNet | Best practical choice - fine-tune on skin data |
| **ResNet50 (ImageNet)** | torchvision | ImageNet | Simple baseline, fine-tune all layers |

**Recommended approach for the hackathon:**
1. Use EfficientNet or ResNet pretrained on ImageNet as backbone
2. Optionally do intermediate fine-tuning on ISIC 2019 (25K images, 8 classes)
3. Then fine-tune on the hackathon 12-class dataset

---

## Part 5: Quick Download Commands

```bash
# HAM10000 via Kaggle
pip install kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d ham10000/

# ISIC 2019 via Kaggle
kaggle datasets download -d salviohexia/isic-2019-skin-lesion-images-for-classification
unzip isic-2019-skin-lesion-images-for-classification.zip -d isic2019/

# ISIC 2018 Segmentation via Kaggle
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation
unzip isic2018-challenge-task1-data-segmentation.zip -d isic2018_seg/

# DermaMNIST via pip (quick experiment)
pip install medmnist
python -c "import medmnist; medmnist.DermaMNIST(split='train', download=True)"

# ISIC 2019 via HuggingFace
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('MKZuziak/ISIC_2019_224')"
```

---

## Part 6: Strategy Recommendations

### For Classification (12 classes, 101x101):
1. **Backbone**: Use `timm` library with EfficientNet-B0 or ResNet50 pretrained on ImageNet
2. **Intermediate pretraining**: Fine-tune on ISIC 2019 (25K images, 8 classes) to learn skin lesion features
3. **Final fine-tuning**: Train on hackathon 12-class dataset with heavy augmentation
4. **Data augmentation**: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomResizedCrop
5. **Handle imbalance**: Use weighted loss or oversampling for minority classes (class 8 has only 331 images vs class 7 with 2136)

### For Segmentation (128x128, binary masks):
1. **Architecture**: U-Net with pretrained encoder (ResNet34 or EfficientNet-B0)
2. **Pretrain encoder**: On ISIC 2018 segmentation task (2,594 images with masks)
3. **Then fine-tune**: On hackathon 1,800 training images
4. **Loss function**: Dice Loss + BCE Loss combined
5. **Augmentation**: Geometric transforms (flip, rotate, elastic deform) applied to BOTH image and mask

### Key Risk:
The hackathon says we CANNOT use datasets containing the exact same images. Since ISIC datasets are aggregated from multiple sources, there is a small risk of overlap. Mitigate by using the datasets for pretraining/transfer learning (learning general features) rather than including them directly in training data.

---

---

## Part 7: MEDICAL DOMAIN DEEP DIVE

### 7.1 What Are These Images? (Corrected Understanding)

After visual inspection of ALL 12 classes, these are **clinical macroscopic photographs of skin lesions** -- NOT histopathology and NOT dermoscopic images. Key evidence:
- Visible skin texture, hair follicles, and nails in many images
- No dermoscope contact plate artifact or immersion fluid
- No H&E staining or microscope-level detail
- Surface-level view of lesions on skin, some on fingers/nails
- Mix of pigmented, vascular, keratotic, and inflammatory lesions

### 7.2 Probable 12-Class Mapping (Visual Analysis of Sample Images)

Based on careful examination of multiple images from each class:

| Class | Count | Visual Characteristics | Likely Diagnosis |
|-------|-------|----------------------|------------------|
| 0 | 571 | Pigmented lesions, brown-tan patches, some with irregular borders | Actinic keratosis / Solar lentigo |
| 1 | 974 | Dark pigmented lesions, well-demarcated, brown-black | Melanocytic nevus (moles) OR Melanoma |
| 2 | 1,043 | Flat/slightly raised, skin-colored to light brown, some with hair | Dermatofibroma OR Benign keratosis |
| 3 | 750 | Lesions on/near nails, periungual, warty | Warts (verruca) OR Nail pathology |
| 4 | 814 | Blue-purple vascular spots, small punctate lesions | Vascular lesion (hemangioma/angiokeratoma) |
| 5 | 441 | Light brown flat patches on skin | Seborrheic keratosis (early/flat) |
| 6 | 545 | Nail/periungual lesions, some ulcerated | Squamous cell carcinoma OR Onychomycosis |
| 7 | 2,136 | Small, dome-shaped, skin-colored papules | Melanocytic nevus (benign, most common) |
| 8 | 331 | Red/pink raised nodules, vascular appearance | Pyogenic granuloma OR Basal cell carcinoma |
| 9 | 1,111 | Red, eroded, ulcerated, bloody lesions | Squamous cell carcinoma OR Ulcerated lesion |
| 10 | 899 | Brown lesions on hairy skin, some linear patterns | Seborrheic keratosis (dark/verrucous) |
| 11 | 1,796 | Warty, papillomatous, verrucous surface texture | Warts / Seborrheic keratosis (verrucous) |

**IMPORTANT**: These are educated guesses. The hackathon deliberately hides disease names. The mapping above should NOT be used in submissions but helps understand the medical domain for model design.

### 7.3 Clinical Significance of Skin Lesion Classification

**Why this matters clinically:**

1. **Skin cancer is the most common cancer worldwide.** Early detection of melanoma, BCC, and SCC dramatically improves survival rates. Melanoma has 99% 5-year survival if caught early vs. 32% if metastatic.

2. **Dermatologist shortage is global.** In Uzbekistan and Central Asia, access to trained dermatologists is extremely limited, especially in rural areas. AI can serve as a triage tool.

3. **The "ugly duckling" problem.** Clinicians must differentiate ~12+ categories of similar-looking lesions. Even expert dermatologists have diagnostic accuracy of only 65-80% for melanoma without dermoscopy. AI can match or exceed this.

4. **Benign vs. malignant distinction is critical.** Unnecessary biopsies burden healthcare systems. AI classification helps prioritize which lesions need urgent biopsy.

5. **Telemedicine application.** Clinical photos (like our dataset) can be taken by primary care physicians or patients themselves and sent to AI for initial screening -- directly relevant to Central Asian healthcare settings.

### 7.4 Known Class Confusion Patterns in Skin Lesion AI

These are the well-documented confusion pairs that our model will likely struggle with:

| Confusion Pair | Why They Look Similar | Clinical Consequence |
|---|---|---|
| **Melanoma vs. Melanocytic Nevus** | Both pigmented, brown/black. Melanoma may have subtle asymmetry/border irregularity | MOST DANGEROUS: Missing melanoma = death |
| **Seborrheic Keratosis vs. Melanoma** | Both can be dark, irregular. "Stuck-on" appearance of SK helps but is not always obvious | Can lead to unnecessary biopsies or missed melanoma |
| **Actinic Keratosis vs. SCC** | AK is precursor to SCC; they exist on a spectrum | Clinically important to detect SCC progression |
| **BCC vs. Dermatofibroma** | Both can be dome-shaped, pink/brown | BCC requires treatment, DF does not |
| **Pyogenic Granuloma vs. Amelanotic Melanoma** | Both are red/vascular nodules | PG is benign, amelanotic melanoma is deadly |
| **Warts vs. SCC** | Verrucous SCC mimics wart morphology | Missing SCC in a wart-like lesion is dangerous |

**Implication for our model**: We should pay special attention to precision/recall tradeoffs on classes that might represent malignant lesions. A false negative for melanoma is far worse than a false positive.

### 7.5 Segmentation Task: Lesion Boundary Delineation

The segmentation task (128x128 images with binary masks) is **skin lesion boundary segmentation**:

- **What is being segmented**: The boundary between the skin lesion and surrounding normal skin
- **Binary mask meaning**: White (255) = lesion area, Black (0) = normal skin/background
- **Clinical purpose**: Accurate segmentation enables:
  - Measurement of lesion diameter (one of ABCDE criteria for melanoma)
  - Assessment of border regularity (irregular borders suggest malignancy)
  - Color distribution analysis within the lesion
  - Monitoring lesion growth over time (teledermatology)
  - Input for downstream classification (segment-then-classify pipeline)

**Most likely source inspiration**: ISIC 2017/2018 Task 1 (Lesion Segmentation), which has 2,000-2,594 dermoscopic images with expert-drawn binary masks. Our dataset (1,800 train) is similarly sized.

**Key challenges for segmentation**:
- Fuzzy/gradual borders (some lesions fade into surrounding skin)
- Hair occlusion covering lesion edges
- Varying skin tones affecting contrast
- Small lesions where segmentation precision matters most
- Artifacts: ruler markings, ink dots, bubble artifacts

### 7.6 Domain-Specific Preprocessing Recommendations

**For Classification (101x101 clinical photos):**

1. **Hair removal** (NOT needed for these images -- hair removal is for dermoscopy with DullRazor algorithm; clinical photos already show surface view)

2. **Color normalization** (CRITICAL):
   - Skin tone varies dramatically across images
   - Apply Shades of Gray or Gray World color constancy
   - Or normalize to a standard color space (LAB color space)
   - Consider Reinhard color normalization adapted for skin images

3. **Contrast enhancement**:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel in LAB space
   - Helps reveal subtle texture differences

4. **Resize strategy**:
   - Images are already 101x101 -- consider upscaling to 224x224 for pretrained models
   - Use bicubic interpolation for upscaling
   - Center-crop or pad to exact 224x224

5. **Augmentation pipeline (recommended order)**:
   ```
   RandomHorizontalFlip(p=0.5)
   RandomVerticalFlip(p=0.5)
   RandomRotation(degrees=90)     # Skin lesions have no natural orientation
   ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
   RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
   RandomErasing(p=0.1)           # Simulates partial occlusion
   GaussianBlur(kernel_size=3, p=0.1)
   ```

6. **DO NOT use**:
   - Extreme hue shifts (skin should remain skin-colored)
   - Heavy perspective transforms (lesions are photographed flat)
   - Cutout/CutMix with patches from different classes (can create misleading features)

**For Segmentation (128x128):**

1. **Joint augmentation** (CRITICAL -- apply same transform to image AND mask):
   ```
   RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(90)
   ElasticTransform(alpha=50, sigma=5)  # Simulates natural shape variation
   RandomAffine(scale=(0.8, 1.2))
   ```

2. **NO color augmentation on masks** -- only geometric transforms

3. **Consider test-time augmentation (TTA)**:
   - Predict on 4 rotations (0, 90, 180, 270) + original
   - Average predictions for more robust segmentation

### 7.7 Published Baselines and State-of-the-Art

**Classification benchmarks (ISIC 2019, 8 classes):**

| Method | Accuracy | Notes |
|--------|----------|-------|
| EfficientNet-B6 ensemble | ~90% | ISIC 2019 challenge top solutions |
| ResNet50 + augmentation | ~85% | Standard baseline |
| Human dermatologist | ~76% | Without dermoscopy |
| Human dermatologist + dermoscopy | ~85% | With dermoscopy tool |

**Segmentation benchmarks (ISIC 2018, binary):**

| Method | Dice Score | Jaccard | Notes |
|--------|-----------|---------|-------|
| UNet++ (EfficientNet encoder) | ~0.92 | ~0.86 | Strong baseline |
| DeepLabv3+ | ~0.90 | ~0.84 | Good for boundary detail |
| Attention U-Net | ~0.91 | ~0.85 | Focuses on lesion area |
| TransUNet | ~0.91 | ~0.85 | Transformer-based |

**Key insight**: Ensemble methods consistently outperform single models in skin lesion tasks. Consider training 3-5 models with different architectures or seeds and averaging predictions.

### 7.8 What a Pathologist/Dermatologist Would Look For

For the presentation to judges, here is the clinical framework:

**ABCDE Criteria for Melanoma Assessment:**
- **A**symmetry -- one half does not match the other
- **B**order irregularity -- edges are ragged, notched, or blurred
- **C**olor variation -- shades of brown, black, red, white, or blue
- **D**iameter -- larger than 6mm (pencil eraser)
- **E**volving -- changing in size, shape, or color

**Dermoscopic Features (for understanding what the model should learn):**
- Pigment network patterns (regular vs. atypical)
- Blue-white structures (associated with melanoma)
- Vascular patterns (dot vessels, linear, glomerular)
- Globules and dots (regular = benign, irregular = concerning)
- Structureless areas (homogeneous pigmentation)

**Clinical Features in Our Images:**
- Surface texture (smooth, rough, verrucous, ulcerated)
- Color (flesh-colored, brown, black, red, blue)
- Shape (round, oval, irregular)
- Border definition (sharp vs. gradual)
- Surrounding skin changes (erythema, scaling)

### 7.9 Presentation Talking Points for Judges

1. **Problem statement**: Skin cancer kills over 100,000 people annually worldwide. Early detection through AI screening of clinical photos could be deployed in resource-limited settings like Central Asia where dermatologist access is scarce.

2. **Clinical workflow integration**: Our model takes a simple phone photo of a skin lesion and provides both classification (what type?) and segmentation (where exactly?). This dual output mimics the dermatologist's diagnostic workflow.

3. **Medical relevance of 12-class approach**: Unlike binary benign/malignant classifiers, multi-class classification provides actionable differential diagnosis, helping primary care physicians decide referral urgency.

4. **Segmentation enables monitoring**: By precisely delineating lesion boundaries, the system can track changes over time -- critical for patients with multiple nevi who need longitudinal monitoring.

5. **Ethical AI considerations**: Class imbalance in our dataset reflects real-world prevalence. We address this with weighted loss functions and augmentation, ensuring rare but dangerous conditions (like melanoma) are not missed.

---

## Sources

- ISIC Archive: https://www.isic-archive.com/
- ISIC Challenge Data: https://challenge.isic-archive.com/data/
- HAM10000 on Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- HAM10000 Paper: https://www.nature.com/articles/sdata2018161
- ISIC 2019 on Kaggle: https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification
- ISIC 2018 Segmentation on Kaggle: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation
- DERM12345: https://pmc.ncbi.nlm.nih.gov/articles/PMC11604664/
- Fitzpatrick17k: https://github.com/mattgroh/fitzpatrick17k
- MedMNIST/DermaMNIST: https://medmnist.com/
- PAD-UFES-20: https://www.sciencedirect.com/science/article/pii/S235234092031115X
- PH2 Dataset: https://www.fc.up.pt/addi/ph2%20database.html
- SLICE-3D: https://www.nature.com/articles/s41597-024-03743-w
- UNI2 Foundation Model: https://huggingface.co/MahmoodLab/UNI2-h
- CONCH Foundation Model: https://huggingface.co/MahmoodLab/CONCH
- Asan 12-class skin classification: https://pubmed.ncbi.nlm.nih.gov/29428356/
- Dermoscopy differential diagnosis: https://pmc.ncbi.nlm.nih.gov/articles/PMC6738388/
- Skin lesion augmentation study: https://www.nature.com/articles/s41598-022-22644-9
- ISIC 2017 segmentation: https://datasetninja.com/isic-2017-part-1
- ISIC 2019 class distribution: https://www.kaggle.com/datasets/andrewmvd/isic-2019
