# CLAUDE.md — AI in Healthcare Hackathon 2026

## Hackathon Context
- **Event**: AI in Healthcare Hackathon 2026, Central Asian University, Tashkent
- **Team**: WhiteCoat.dev (#37)
- **Deadline**: March 27, 2026 — 22:00 (Tashkent, UTC+5)
- **Top 10 Announcement**: March 28, 01:00
- **Presentations**: March 28, 09:30-14:30

## Tasks

### Task 1 — Classification (30 pts, metric: ACCURACY)
- **Data**: SKIN LESION clinical/dermoscopic photos (NOT histopathology!), 12 classes (0-11), disease names hidden
- **Domain**: Dermatology — likely ISIC-based dataset (melanoma, nevus, BCC, etc.)
- **Train**: ~11,400 images in `data/classification/train/{0..11}/`
- **Test**: 1,276 PNG images in `data/classification/test/`
- **Output**: Excel `WhiteCoat.dev test_ground_truth.xlsx`
  - Columns: `Image_ID | Label`
  - Image_ID = filename without .png extension
  - Label = integer 0-11
  - Must include ALL 1276 images

### Task 2 — Segmentation (40 pts, metric: MEAN IoU)
- **Train**: 1,800 images + masks in `data/Segmentation/training/`
- **Val**: 400 images + masks in `data/Segmentation/validation/`
- **Test**: 200 images in `data/Segmentation/testing/images/`
- **Output**: Folder `WhiteCoat.dev/` with 200 PNG binary masks
  - Format: PNG, binary (0 or 255), same size as input
  - Filename must match input image filename

### Class Distribution (IMBALANCED!)
```
Class 0:   571  | Class 6:   545
Class 1:   974  | Class 7:  2136  ← largest
Class 2:  1043  | Class 8:   331  ← smallest
Class 3:   750  | Class 9:  1111
Class 4:   814  | Class 10:  899
Class 5:   441  | Class 11: 1796
```

## Submission Structure
```
WhiteCoat.dev/
├── WhiteCoat.dev test_ground_truth.xlsx
├── WhiteCoat.dev/          # segmentation masks
│   ├── {id}.png ...
└── models/
    ├── classification/
    │   ├── classify.py
    │   ├── best_model.pth
    │   └── requirements.txt
    └── segmentation/
        ├── segment.py
        ├── best_model.pth
        └── requirements.txt
```

## Scoring (100 total)
- Phase 1 (70 pts, automatic): `accuracy × 30` + `mean_IoU × 40`
- Phase 2 (30 pts, top 10): UI(10) + Error Handling(5) + Innovation(5) + Presentation(10)

## Rules
- ✅ Pretrained models, transfer learning, public libraries, public datasets
- ❌ Using test images for training
- ❌ Manually labeling test images
- ❌ External datasets with same images
- ❌ Sharing predictions/models with other teams

## CRITICAL: Domain = Dermatology (Skin Lesions)
Images are clinical/dermoscopic photographs of skin lesions, NOT microscopy biopsy slides.
Likely based on ISIC Archive. 12 classes = skin conditions (melanoma, nevus, BCC, SCC, etc.)

## Key Technical Decisions
- **Classification**: EfficientNetV2-S + ConvNeXt-Tiny + Swin-Tiny ensemble, weighted CrossEntropy/Focal Loss
- **Segmentation**: U-Net++ + DeepLabV3+ ensemble, Dice+BCE+Lovász loss
- **Augmentation**: Adapted for dermatology (moderate HueSaturation, CLAHE, CoarseDropout for hair artifacts)
- **Pretraining**: ImageNet → optionally ISIC 2019 intermediate fine-tune → hackathon data
- **Framework**: PyTorch + timm + segmentation_models_pytorch + albumentations
- **TTA**: 8x (all flips + rotations) — skin lesions have no canonical orientation

## Project Structure
```
Hackathon/
├── CLAUDE.md               # This file
├── data/                   # Dataset (not in git)
│   ├── classification/
│   └── Segmentation/
├── templates/              # Code templates
│   ├── classification/     # Training pipeline
│   ├── segmentation/       # Training pipeline
│   ├── utils/              # Common utilities
│   ├── ui/                 # Streamlit app
│   ├── inference_a.py      # classify.py template
│   ├── inference_b.py      # segment.py template
│   └── config.yaml         # Hyperparameters
├── checkpoints/            # Saved models
├── results/                # Predictions
└── submission/             # Final submission folder
```

## Presentation Requirements (15 min, English)
1. AI Workflow (pipeline, preprocessing, training)
2. Model Design (architectures, why chosen)
3. Challenges & Solutions
4. Results (metrics, visualizations)
5. Interface Demo
- Must include: "For research and demonstration purposes only. Not for clinical use."
