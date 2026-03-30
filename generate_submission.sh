#!/bin/bash
# WhiteCoat.dev — Generate Submission (run after training)
set -e
cd /root/hackathon
mkdir -p submission/"WhiteCoat.dev"
mkdir -p submission/models/classification
mkdir -p submission/models/segmentation

echo '=== Generating submission ==='

# Classification: ensemble + TTA
python3 templates/inference_a.py \
    --model_path checkpoints/best_tf_efficientnetv2_s.pth \
    --test_dir data/classification/test \
    --output_path "submission/WhiteCoat.dev test_ground_truth.xlsx"

# Segmentation: inference
python3 templates/inference_b.py \
    --model_path checkpoints/best_UnetPlusPlus_efficientnet_b4.pth \
    --test_dir data/Segmentation/testing/images \
    --output_dir submission/"WhiteCoat.dev"/

# Copy scripts and models
cp templates/inference_a.py submission/models/classification/classify.py
cp templates/inference_b.py submission/models/segmentation/segment.py
cp checkpoints/best_tf_efficientnetv2_s.pth submission/models/classification/best_model.pth
cp checkpoints/best_UnetPlusPlus_efficientnet_b4.pth submission/models/segmentation/best_model.pth 2>/dev/null || true

echo 'torch>=2.0.0
timm>=0.9.0
albumentations>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0' > submission/models/classification/requirements.txt

echo 'torch>=2.0.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.0
opencv-python>=4.8.0
numpy>=1.24.0' > submission/models/segmentation/requirements.txt

echo '=== DONE ==='
ls -la submission/
