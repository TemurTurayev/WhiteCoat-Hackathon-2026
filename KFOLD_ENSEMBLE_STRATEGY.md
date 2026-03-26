# K-Fold & Ensemble Strategy — WhiteCoat.dev Hackathon

## Hardware: NVIDIA DGX Spark
- GPU: GB10 Blackwell, 6144 CUDA cores
- Memory: 128 GB unified LPDDR5x (273 GB/s)
- Compute: ~100 TFLOPS BF16, ~208 TFLOPS FP8
- Training is bandwidth-limited, not compute-limited
- Roughly comparable to RTX 3060-3070 for CNN training throughput

---

## 1. K-Fold Strategy for Classification

### Recommendation: 5-fold Stratified (Plan A) or 3-fold (Plan B)

| Strategy | Models | Training Time | Accuracy Boost |
|----------|--------|---------------|----------------|
| No K-fold (baseline) | 3 | ~5 hours | baseline |
| 3-fold x 3 models | 9 | ~9 hours | +1-2% |
| 5-fold x 3 models | 15 | ~15 hours | +2-3% |

**Why StratifiedKFold is critical:**
- Your dataset has class imbalance (class 8: 331 vs class 7: 2136)
- Regular KFold could put zero samples of rare classes in a fold
- StratifiedKFold guarantees proportional class distribution in every fold

**5-fold is better than 3-fold because:**
- Each fold trains on 80% of data (9,120 images) vs 67% (7,638 images)
- More models = better ensemble diversity
- More reliable OOF accuracy estimate (lower variance)
- BUT: takes 67% more time

**For a 36-hour hackathon: use 5-fold if you start training within first 2 hours.**

### How 15 Models Are Ensembled

Each architecture produces 5 fold-models. For inference on test data:

```
EfficientNetV2-S: fold0, fold1, fold2, fold3, fold4  -> average softmax -> model_A_probs
ConvNeXt-Tiny:    fold0, fold1, fold2, fold3, fold4  -> average softmax -> model_B_probs
Swin-Tiny:        fold0, fold1, fold2, fold3, fold4  -> average softmax -> model_C_probs

Final = w_A * model_A_probs + w_B * model_B_probs + w_C * model_C_probs
```

Weights (w_A, w_B, w_C) are optimized on OOF predictions.

### Disk Space for 15 Checkpoints
- EfficientNetV2-S: ~84 MB per checkpoint x 5 = ~420 MB
- ConvNeXt-Tiny: ~112 MB x 5 = ~560 MB
- Swin-Tiny: ~112 MB x 5 = ~560 MB
- Total: ~1.5 GB (trivial on 128 GB system)

---

## 2. K-Fold for Segmentation

### Strategy: 5-fold on training data + external validation

```
Training: 1,800 images
  -> 5-fold: train on 1,440, validate on 360 per fold
External validation: 400 images (ALWAYS evaluated)
```

**How to use both validation sets:**
- fold-val (360 images): used for model selection (best checkpoint)
- ext-val (400 images): used for reporting and ensemble optimization
- This prevents information leakage: ext-val is never used for training decisions

### Ensembling Segmentation Masks

**Average logits before sigmoid (RECOMMENDED):**
```
logits_fold0 + logits_fold1 + ... + logits_fold4
avg_logits = sum / 5
probability = sigmoid(avg_logits)
mask = probability > threshold
```

**Why logit averaging > mask averaging:**
- Logits preserve uncertainty information (values near 0 = uncertain)
- Averaging in probability space loses this: sigmoid(0.1) = 0.52, sigmoid(-0.1) = 0.48
- Binary mask voting is even worse: destroys all probability information
- Empirically, logit averaging gives +1-3% IoU over majority voting

**Threshold optimization is critical for segmentation:**
- Default 0.5 is rarely optimal
- Search range: 0.2 to 0.8, step 0.05
- Optimize on external validation set
- The ensemble_optimizer.py script does this automatically

---

## 3. Ensemble Weight Optimization

### Method: scipy.optimize.minimize with Nelder-Mead

The ensemble_optimizer.py implements this:
1. Load OOF softmax from each model
2. Define objective: negative accuracy (classification) or negative IoU (segmentation)
3. Optimize weights using Nelder-Mead (gradient-free, works well for 2-5 weights)
4. Use softmax parameterization for weights (ensures positive, sum to 1)
5. Multiple random restarts (20) for robustness

**Expected improvement from weight optimization:**
- Over simple average: +0.3-0.8% accuracy
- Best models get higher weights automatically
- If all models are equally good, weights converge to equal

---

## 4. Time Estimates on DGX Spark

### Classification (per fold)

| Model | Params | Batch | Est. sec/epoch | Epochs (w/ early stop) | Time/fold |
|-------|--------|-------|----------------|----------------------|-----------|
| EfficientNetV2-S | 21M | 64 | ~45-60s | ~25-35 | ~25 min |
| ConvNeXt-Tiny | 29M | 64 | ~50-65s | ~25-35 | ~30 min |
| Swin-Tiny | 29M | 64 | ~55-70s | ~25-35 | ~32 min |

**Assumptions:**
- 9,120 train images per fold (5-fold)
- 224x224 images, mixed precision (AMP)
- DGX Spark ~100 TFLOPS BF16
- ~142 batches/epoch (9120/64)
- Early stopping typically fires at 60-85% of max epochs

### Segmentation (per fold)

| Model | Params | Batch | Est. sec/epoch | Epochs (w/ early stop) | Time/fold |
|-------|--------|-------|----------------|----------------------|-----------|
| UNet++ EfficientNet-B4 | 25M | 32 | ~35-45s | ~35-50 | ~28 min |
| DeepLabV3+ ResNet50 | 40M | 32 | ~40-50s | ~35-50 | ~32 min |

**Assumptions:**
- 1,440 train images per fold (5-fold)
- 256x256, mixed precision
- ~45 batches/epoch (1440/32)

### Total Time Summary

| Plan | Classification | Segmentation | Ensemble | Total |
|------|---------------|-------------|----------|-------|
| A (5-fold) | 5x(25+30+32) = ~7.2h | 5x(28+32) = ~5h | 10 min | ~12.5h |
| B (3-fold) | 3x(25+30+32) = ~4.3h | 3x(28+32) = ~3h | 10 min | ~7.5h |
| C (no fold) | 1x(25+30+32) = ~1.5h | 1x(28+32) = ~1h | 5 min | ~2.5h |

**NOTE:** These are conservative estimates. Actual times could be:
- 30-50% faster if DGX Spark has better CNN throughput than estimated
- 20% slower if data loading becomes a bottleneck
- Early stopping can save 15-40% of total time

**RECOMMENDATION FOR 36 HOURS:**
- Plan A (5-fold) is absolutely feasible in 36 hours
- Leaves ~23 hours for: data exploration, debugging, inference, presentation
- Start training ASAP, let it run overnight

---

## 5. Out-of-Fold (OOF) Predictions

OOF predictions are the key to proper ensemble optimization:

```
Fold 0: train on [1,2,3,4], predict on [0] -> save softmax for samples in fold 0
Fold 1: train on [0,2,3,4], predict on [1] -> save softmax for samples in fold 1
...
Fold 4: train on [0,1,2,3], predict on [4] -> save softmax for samples in fold 4

Result: every sample has a prediction from a model that NEVER saw it during training
```

**Uses of OOF predictions:**
1. **Ensemble weight optimization** — find optimal w_A, w_B, w_C
2. **Error analysis** — which samples are hardest? (all models wrong)
3. **Threshold optimization** — for segmentation
4. **Reliable accuracy estimate** — no information leakage
5. **Stacking** — train level-2 model on OOF features

The train_classification_kfold.py saves OOF as .npz files automatically.

---

## 6. Stacking (Level 2 Model)

### How it works:
```
Input features: [softmax_modelA (12 dims) | softmax_modelB (12 dims) | softmax_modelC (12 dims)]
= 36-dimensional feature vector per sample

Level-2 model: LogisticRegression or XGBoost
Output: final class prediction
```

### Is stacking better than weighted averaging?

**For 12 classes with 3 models:**
- Stacking CAN learn class-specific model preferences
  (e.g., model A is best for class 3, model C is best for class 8)
- Simple averaging treats all classes equally
- Stacking typically gives +0.5-1.5% over simple averaging
- BUT: risk of overfitting with only ~11,400 samples and 36 features

**Recommendation:**
- Try both in ensemble_optimizer.py (it does this automatically)
- If stacking OOF accuracy > weighted average: use stacking
- If they're similar: use weighted average (simpler, less overfitting risk)
- Time to implement: already done in ensemble_optimizer.py (adds ~30 seconds)

---

## 7. Practical Recommendations

### Minimum Viable Ensemble (if time is tight)
1. Train 3 classification models + 2 segmentation models (Plan C, ~2.5h)
2. Simple average of softmax (classification) / logits (segmentation)
3. Expected boost: +1-2% accuracy, +1-3% IoU over single best model

### Recommended Strategy (Plan A)
1. Start 5-fold training immediately after data exploration (~hour 2)
2. Let it run overnight (~12 hours)
3. Morning: run ensemble_optimizer.py (10 minutes)
4. Use optimized weights in inference scripts
5. Expected boost: +2-4% accuracy, +2-5% IoU over single model

### Parallelization on DGX Spark
- DGX Spark has ONE GPU — no multi-GPU parallelism
- Train models sequentially (as in run_kfold_all.sh)
- CAN parallelize data loading (num_workers=4)
- CAN run different folds on different machines if available

### Decision Tree:
```
Time remaining > 24h? -> Plan A (5-fold)
Time remaining > 12h? -> Plan B (3-fold)
Time remaining > 6h?  -> Plan C (no fold, simple ensemble)
Time remaining < 6h?  -> Single best model, no ensemble
```

---

## Files Created

| File | Purpose |
|------|---------|
| `train_classification_kfold.py` | K-fold classification training with OOF |
| `train_segmentation_kfold.py` | K-fold segmentation training with OOF |
| `ensemble_optimizer.py` | Weight optimization + stacking |
| `run_kfold_all.sh` | Orchestrator (Plan A/B/C) |

### Quick Start:
```bash
# On DGX Spark:
cd templates
chmod +x run_kfold_all.sh
./run_kfold_all.sh --plan a --data_dir /path/to/data

# After training completes:
python ensemble_optimizer.py --checkpoint_dir checkpoints --task both
```
