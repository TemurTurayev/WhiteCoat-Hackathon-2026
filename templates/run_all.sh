#!/bin/bash
# ============================================
# WhiteCoat.dev — Полный pipeline обучения
# Запуск: bash run_all.sh /path/to/data
# ============================================

DATA_DIR="${1:-data}"
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR"

echo "============================================"
echo "  WhiteCoat.dev — AI in Healthcare 2026"
echo "  Data: $DATA_DIR"
echo "============================================"

# ===== CLASSIFICATION: 3 модели =====

echo ""
echo ">>> [1/5] Training EfficientNetV2-S (Classification)..."
python train_classification.py \
    --model tf_efficientnetv2_s \
    --data_dir "$DATA_DIR/classification" \
    --image_size 224 --batch_size 64 --epochs 40 --lr 3e-4 \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo ">>> [2/5] Training ConvNeXt-Tiny (Classification)..."
python train_classification.py \
    --model convnext_tiny \
    --data_dir "$DATA_DIR/classification" \
    --image_size 224 --batch_size 64 --epochs 40 --lr 3e-4 \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo ">>> [3/5] Training Swin-Tiny (Classification)..."
python train_classification.py \
    --model swin_tiny_patch4_window7_224 \
    --data_dir "$DATA_DIR/classification" \
    --image_size 224 --batch_size 64 --epochs 40 --lr 3e-4 \
    --checkpoint_dir "$CHECKPOINT_DIR"

# ===== SEGMENTATION: 2 модели =====

echo ""
echo ">>> [4/5] Training U-Net++ + EfficientNet-B4 (Segmentation)..."
python train_segmentation.py \
    --arch UnetPlusPlus --encoder efficientnet-b4 \
    --data_dir "$DATA_DIR/Segmentation" \
    --image_size 256 --batch_size 32 --epochs 60 --lr 3e-4 \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo ">>> [5/5] Training DeepLabV3+ + ResNet50 (Segmentation)..."
python train_segmentation.py \
    --arch DeepLabV3Plus --encoder resnet50 \
    --data_dir "$DATA_DIR/Segmentation" \
    --image_size 256 --batch_size 32 --epochs 60 --lr 3e-4 \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo ""
echo "============================================"
echo "  ALL TRAINING COMPLETE!"
echo "  Checkpoints saved in: $CHECKPOINT_DIR/"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run inference: python inference_a.py --model_path $CHECKPOINT_DIR/best_*.pth --test_dir $DATA_DIR/classification/test"
echo "  2. Run inference: python inference_b.py --model_path $CHECKPOINT_DIR/best_*.pth --test_dir $DATA_DIR/Segmentation/testing/images"
echo "  3. Prepare submission folder"
