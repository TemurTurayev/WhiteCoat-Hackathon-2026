#!/bin/bash
# ============================================================
# K-Fold Training Orchestrator — WhiteCoat.dev
# ============================================================
#
# Стратегия для 36-часового хакатона:
#
# ПЛАН A (РЕКОМЕНДОВАН — 5-fold, ~22 часа):
#   ./run_kfold_all.sh --plan a --data_dir data
#
# ПЛАН B (Быстрый — 3-fold, ~14 часов):
#   ./run_kfold_all.sh --plan b --data_dir data
#
# ПЛАН C (Минимальный — без K-fold, ~5 часов):
#   ./run_kfold_all.sh --plan c --data_dir data
#
# ============================================================

set -e

PLAN="a"
DATA_DIR="data"
CHECKPOINT_DIR="checkpoints"
N_WORKERS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --plan) PLAN="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$CHECKPOINT_DIR/oof"
mkdir -p logs

echo "============================================================"
echo "WhiteCoat.dev — K-Fold Training Orchestrator"
echo "Plan: $PLAN | Data: $DATA_DIR | Checkpoints: $CHECKPOINT_DIR"
echo "Start time: $(date)"
echo "============================================================"

# ============================================================
# PLAN A: Full 5-Fold (recommended if you have 22+ hours)
# 3 classification models x 5 folds = 15 models (~15 hours)
# 2 segmentation models x 5 folds = 10 models (~7 hours)
# Total: ~22 hours + ensemble optimization
# ============================================================

if [ "$PLAN" = "a" ]; then
    echo ""
    echo ">>> PLAN A: Full 5-Fold Cross-Validation"
    echo ""

    N_FOLDS=5

    # --- Classification ---
    echo "=== CLASSIFICATION (5-fold x 3 models) ==="
    for MODEL in tf_efficientnetv2_s convnext_tiny swin_tiny_patch4_window7_224; do
        echo ""
        echo "--- Training $MODEL ($N_FOLDS folds) ---"
        python train_classification_kfold.py \
            --model "$MODEL" \
            --data_dir "$DATA_DIR/classification" \
            --n_folds $N_FOLDS \
            --image_size 224 \
            --batch_size 64 \
            --epochs 40 \
            --patience 7 \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --num_workers $N_WORKERS \
            2>&1 | tee "logs/cls_${MODEL}_kfold.log"
    done

    # --- Segmentation ---
    echo ""
    echo "=== SEGMENTATION (5-fold x 2 models) ==="
    for ARCH_ENC in "UnetPlusPlus:efficientnet-b4" "DeepLabV3Plus:resnet50"; do
        ARCH="${ARCH_ENC%%:*}"
        ENC="${ARCH_ENC##*:}"
        echo ""
        echo "--- Training $ARCH + $ENC ($N_FOLDS folds) ---"
        python train_segmentation_kfold.py \
            --arch "$ARCH" \
            --encoder "$ENC" \
            --data_dir "$DATA_DIR/Segmentation" \
            --n_folds $N_FOLDS \
            --image_size 256 \
            --batch_size 32 \
            --epochs 60 \
            --patience 10 \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --num_workers $N_WORKERS \
            2>&1 | tee "logs/seg_${ARCH}_${ENC}_kfold.log"
    done
fi

# ============================================================
# PLAN B: 3-Fold (faster, still good ensemble)
# 3 models x 3 folds = 9 classification models (~9 hours)
# 2 models x 3 folds = 6 segmentation models (~5 hours)
# Total: ~14 hours
# ============================================================

if [ "$PLAN" = "b" ]; then
    echo ""
    echo ">>> PLAN B: 3-Fold Cross-Validation (faster)"
    echo ""

    N_FOLDS=3

    echo "=== CLASSIFICATION (3-fold x 3 models) ==="
    for MODEL in tf_efficientnetv2_s convnext_tiny swin_tiny_patch4_window7_224; do
        echo ""
        echo "--- Training $MODEL ($N_FOLDS folds) ---"
        python train_classification_kfold.py \
            --model "$MODEL" \
            --data_dir "$DATA_DIR/classification" \
            --n_folds $N_FOLDS \
            --image_size 224 \
            --batch_size 64 \
            --epochs 35 \
            --patience 6 \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --num_workers $N_WORKERS \
            2>&1 | tee "logs/cls_${MODEL}_kfold3.log"
    done

    echo ""
    echo "=== SEGMENTATION (3-fold x 2 models) ==="
    for ARCH_ENC in "UnetPlusPlus:efficientnet-b4" "DeepLabV3Plus:resnet50"; do
        ARCH="${ARCH_ENC%%:*}"
        ENC="${ARCH_ENC##*:}"
        echo ""
        echo "--- Training $ARCH + $ENC ($N_FOLDS folds) ---"
        python train_segmentation_kfold.py \
            --arch "$ARCH" \
            --encoder "$ENC" \
            --data_dir "$DATA_DIR/Segmentation" \
            --n_folds $N_FOLDS \
            --image_size 256 \
            --batch_size 32 \
            --epochs 50 \
            --patience 8 \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --num_workers $N_WORKERS \
            2>&1 | tee "logs/seg_${ARCH}_${ENC}_kfold3.log"
    done
fi

# ============================================================
# PLAN C: No K-fold (single split, fastest baseline)
# 3 classification + 2 segmentation = 5 models (~5 hours)
# Then simple ensemble of 3+2 single models
# ============================================================

if [ "$PLAN" = "c" ]; then
    echo ""
    echo ">>> PLAN C: Single Split (fastest baseline)"
    echo ""

    echo "=== CLASSIFICATION ==="
    for MODEL in tf_efficientnetv2_s convnext_tiny swin_tiny_patch4_window7_224; do
        echo ""
        echo "--- Training $MODEL (single split) ---"
        python train_classification.py \
            --model "$MODEL" \
            --data_dir "$DATA_DIR/classification" \
            --image_size 224 \
            --batch_size 64 \
            --epochs 40 \
            --patience 7 \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --num_workers $N_WORKERS \
            2>&1 | tee "logs/cls_${MODEL}_single.log"
    done

    echo ""
    echo "=== SEGMENTATION ==="
    for ARCH_ENC in "UnetPlusPlus:efficientnet-b4" "DeepLabV3Plus:resnet50"; do
        ARCH="${ARCH_ENC%%:*}"
        ENC="${ARCH_ENC##*:}"
        echo ""
        echo "--- Training $ARCH + $ENC (single split) ---"
        python train_segmentation.py \
            --arch "$ARCH" \
            --encoder "$ENC" \
            --data_dir "$DATA_DIR/Segmentation" \
            --image_size 256 \
            --batch_size 32 \
            --epochs 60 \
            --patience 10 \
            --checkpoint_dir "$CHECKPOINT_DIR" \
            --num_workers $N_WORKERS \
            2>&1 | tee "logs/seg_${ARCH}_${ENC}_single.log"
    done
fi

# ============================================================
# ENSEMBLE OPTIMIZATION (runs after any plan)
# ============================================================

echo ""
echo "============================================================"
echo "ENSEMBLE OPTIMIZATION"
echo "============================================================"

if [ "$PLAN" != "c" ]; then
    python ensemble_optimizer.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --task both \
        2>&1 | tee logs/ensemble_optimization.log
fi

echo ""
echo "============================================================"
echo "ALL DONE! End time: $(date)"
echo "Checkpoints in: $CHECKPOINT_DIR"
echo "Logs in: logs/"
echo "============================================================"
