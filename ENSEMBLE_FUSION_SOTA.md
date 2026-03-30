# Ensemble & Model Fusion: State-of-the-Art (2024-2026)

> Research compiled for WhiteCoat.dev Hackathon - AI in Healthcare 2026
> Focus: Classification (15+ models) and Segmentation (10+ models) ensemble strategies

---

## 1. Model Soups (Weight-Space Averaging)

**Core idea**: Average the weights of multiple models fine-tuned from the same pretrained backbone with different hyperparameters. Zero additional inference cost.

### Variants

| Variant | Description | When to Use |
|---------|-------------|-------------|
| **Uniform Soup** | Simple average of all model weights | Quick baseline, all models similar quality |
| **Greedy Soup** | Add model to soup only if it improves val metric | Best general-purpose approach |
| **Learned Soup** | Learn interpolation coefficients per layer | When you have enough val data |
| **Sparse Soup** | Average pruned models with identical connectivity | Memory-constrained deployment |

### Requirements
- All models MUST share the same architecture and be fine-tuned from the same pretrained checkpoint
- Models trained with different hyperparameters (LR, augmentation, dropout, weight decay)
- Works because fine-tuned models lie in the same loss basin

### Implementation (Greedy Soup)
```python
import torch
import copy

def greedy_model_soup(models, val_loader, metric_fn):
    """
    Greedy soup: iteratively add models that improve validation metric.
    models: list of state_dicts sorted by individual val performance (best first)
    """
    best_soup = copy.deepcopy(models[0])
    best_metric = evaluate(best_soup, val_loader, metric_fn)

    for i in range(1, len(models)):
        # Trial: average current soup with candidate
        candidate_soup = {}
        n = i + 1  # number of models if we add this one
        for key in best_soup:
            candidate_soup[key] = (best_soup[key] * i + models[i][key]) / n

        candidate_metric = evaluate(candidate_soup, val_loader, metric_fn)
        if candidate_metric > best_metric:
            best_soup = candidate_soup
            best_metric = candidate_metric
            print(f"Added model {i}, metric: {best_metric:.4f}")
        else:
            print(f"Skipped model {i}")

    return best_soup
```

### 2024-2025 Updates
- **Rethinking Weight-Averaged Model Merging** (2024): Proposed improved greedy averaging that accounts for parameter-level importance, not just uniform averaging
- **Sparse Model Soups** (NeurIPS 2024): Demonstrated that pruned models can be souped if they share the same sparse connectivity pattern
- **Task Arithmetic** (2024): Extend soups to merge task-specific "vectors" (delta weights), enabling multi-task soups

### Applicability to Our Hackathon
- Apply greedy soup to our multiple EfficientNetV2-S checkpoints trained with different hyperparameters
- Apply greedy soup to ConvNeXt-Tiny variants separately
- Cannot soup EfficientNet with ConvNeXt (different architectures)
- For segmentation: soup multiple U-Net++ checkpoints, soup multiple DeepLabV3+ checkpoints separately

---

## 2. Snapshot Ensembles

**Core idea**: Use cosine annealing with warm restarts. Save a checkpoint at each learning rate minimum. Ensemble these snapshots. Train 1 model, get M for free.

### Learning Rate Schedule
```python
import math

def snapshot_cosine_lr(epoch, T, M, lr_max):
    """
    T: total epochs, M: number of cycles (snapshots), lr_max: max learning rate
    """
    cycle_length = T // M
    t = epoch % cycle_length
    lr = (lr_max / 2) * (1 + math.cos(math.pi * t / cycle_length))
    return lr

# Example: 200 epochs, 5 snapshots, save at epochs 40, 80, 120, 160, 200
```

### 2025 Updates: Snap-MAE
- Recent work (2025) integrates snapshot ensembles into Masked Autoencoder pretraining
- Cyclic cosine scheduler with snapshots every 200 epochs during self-supervised pretraining
- Produces multiple pretrained encoders from a single pretraining run
- Tested on pediatric thoracic disease classification and cardiovascular diagnosis

### Practical Tips
- Use 3-5 snapshots (diminishing returns beyond 5)
- Each snapshot captures a different local minimum in the loss landscape
- Snapshots from the same training run have moderate diversity (less than independently trained models)
- Combine with model soups: soup the snapshots instead of averaging predictions

### When to Use
- Limited compute budget (1 GPU, time pressure -- OUR SITUATION)
- Want ensemble benefits without training multiple models from scratch
- Works well with both classification and segmentation

---

## 3. Stacking vs Blending vs Weighted Average

### Comparison Table

| Method | Complexity | Overfitting Risk | Typical Gain | Best For |
|--------|-----------|------------------|--------------|----------|
| **Weighted Average** | Low | Low | +0.5-2% | Hackathons, small val sets |
| **Rank Average** | Low | Very Low | +0.5-1.5% | Poorly calibrated models |
| **Blending** | Medium | Medium | +1-3% | Medium val sets (>1000) |
| **Stacking** | High | High | +2-5% | Large datasets, Kaggle finals |

### Weighted Average (RECOMMENDED for our case)
```python
import numpy as np
from scipy.optimize import minimize

def optimize_weights(predictions, labels, n_models):
    """
    predictions: (n_models, n_samples, n_classes) array of softmax outputs
    labels: (n_samples,) ground truth
    """
    def objective(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()  # normalize
        blended = np.tensordot(weights, predictions, axes=([0], [0]))
        preds = blended.argmax(axis=1)
        accuracy = (preds == labels).mean()
        return -accuracy  # minimize negative accuracy

    # Start with uniform weights
    x0 = np.ones(n_models) / n_models

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-6})

    weights = np.abs(result.x)
    weights = weights / weights.sum()
    return weights

# IMPORTANT: Use cross-validation to prevent overfitting weights!
def cv_optimize_weights(predictions_folds, labels_folds, n_models, n_folds=5):
    """
    Use out-of-fold predictions to optimize weights.
    predictions_folds[fold] = (n_models, n_val_samples, n_classes)
    """
    all_oof_preds = np.concatenate(predictions_folds, axis=1)
    all_oof_labels = np.concatenate(labels_folds)
    return optimize_weights(all_oof_preds, all_oof_labels, n_models)
```

### Rank-Based Ensemble
```python
from scipy.stats import rankdata

def rank_average_ensemble(predictions_list):
    """
    Convert probabilities to ranks, then average ranks.
    Useful when models have different calibration.
    predictions_list: list of (n_samples, n_classes) arrays
    """
    ranked = []
    for preds in predictions_list:
        ranked_preds = np.zeros_like(preds)
        for c in range(preds.shape[1]):
            ranked_preds[:, c] = rankdata(preds[:, c])
        ranked.append(ranked_preds)

    return np.mean(ranked, axis=0)
```

### Stacking (for reference, likely overkill for our val set size)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def stacking_ensemble(oof_predictions, oof_labels, test_predictions):
    """
    oof_predictions: (n_train, n_models * n_classes) -- out-of-fold preds
    test_predictions: (n_test, n_models * n_classes) -- test preds
    """
    meta_model = LogisticRegression(C=1.0, max_iter=1000)
    meta_model.fit(oof_predictions, oof_labels)
    return meta_model.predict_proba(test_predictions)
```

### ISIC 2024 Competition Insights
- 2nd place: Auxiliary losses + self-supervised pretraining, simple weighted ensemble
- 4th place: Tabular + image models with 2:8 ratio weighted ensemble
- Multiple top teams reported that stacking was not more effective than weighted averaging for this domain
- Weighted ensemble of diverse models consistently outperformed complex stacking

---

## 4. Diverse Architecture Ensemble (CNN + Transformer + Hybrid)

### Why Diversity Matters

Ensemble error decomposes as:
```
E_ensemble = E_avg - Diversity
```
Where diversity measures disagreement among models. More diverse models = lower ensemble error, EVEN if individual models are slightly weaker.

### Architecture Diversity Sources

| Architecture | Inductive Bias | Strength | Weakness |
|-------------|---------------|----------|----------|
| **EfficientNetV2** | Strong local (CNN) | Texture, fine detail | Global context |
| **ConvNeXt** | Moderate local (modernized CNN) | Good balance | Jack of all trades |
| **Swin Transformer** | Global attention | Shape, global patterns | Needs more data |
| **EVA-02** | CLIP-pretrained ViT | Robust features | Heavy compute |
| **MaxViT** | Multi-axis attention + Conv | Local + Global | Complex |
| **DeiT-III** | Distilled ViT | Efficient | Moderate performance |

### Practical Diversity Rules (from Kaggle winners)

1. **Architecture diversity > hyperparameter diversity**: EfficientNet + Swin beats 5 EfficientNets
2. **Pretraining diversity**: ImageNet-1k vs ImageNet-21k vs CLIP vs self-supervised
3. **Input resolution diversity**: 224px + 384px + 512px models
4. **Augmentation diversity**: Different augmentation policies per model
5. **Loss function diversity**: CrossEntropy vs Focal Loss vs Label Smoothing

### Measuring Diversity
```python
def prediction_disagreement(pred_a, pred_b):
    """Fraction of samples where two models disagree."""
    return (pred_a.argmax(1) != pred_b.argmax(1)).mean()

def correlation_matrix(predictions_list):
    """Correlation between model probability outputs."""
    n = len(predictions_list)
    flat_preds = [p.flatten() for p in predictions_list]
    corr = np.corrcoef(flat_preds)
    return corr

# Rule of thumb: add a model to ensemble if its average correlation
# with existing models is < 0.95 AND its individual accuracy is
# within 3% of the best model
```

### Our Hackathon Strategy
- EfficientNetV2-S (CNN, local features, texture-focused)
- ConvNeXt-Tiny (modernized CNN, good generalization)
- Swin-Tiny (Transformer, global patterns, shape-focused)
- These three have fundamentally different inductive biases = high diversity

---

## 5. Multi-Resolution Ensemble

**Core idea**: Train the same architecture at different input sizes. Each resolution captures different feature scales.

### Resolution Strategy

| Resolution | Captures | Trade-off |
|-----------|----------|-----------|
| 224x224 | Global structure, coarse features | Fast, less detail |
| 384x384 | Balanced features | Good default |
| 512x512 | Fine-grained details, textures | Slower, more memory |
| 640x640+ | Subtle patterns, small lesions | Very slow |

### Implementation
```python
# Train same architecture at multiple resolutions
configs = [
    {'size': 224, 'batch_size': 64, 'epochs': 30},
    {'size': 384, 'batch_size': 32, 'epochs': 25},
    {'size': 512, 'batch_size': 16, 'epochs': 20},
]

# At inference, each model processes at its native resolution
# Ensemble the softmax outputs
```

### Key Findings from Competitions
- Multi-resolution ensemble consistently adds +1-2% accuracy over single resolution
- For skin lesion classification: 384 + 512 is the sweet spot
- Higher resolutions help for small lesions and subtle texture differences
- Lower resolutions help for shape-based classification
- Different resolutions on the same architecture provide moderate diversity (correlation ~0.85-0.92)

### For Segmentation
- Multi-scale inference is critical: predict at 0.5x, 1.0x, 1.5x of training resolution
- Average the upsampled probability maps
- Especially important for lesions of varying sizes

---

## 6. Rank-Based Ensemble

### When to Use Rank Averaging vs Probability Averaging

| Scenario | Use Rank Average | Use Probability Average |
|----------|-----------------|------------------------|
| Models have different calibration | Yes | No |
| Models trained with different losses | Yes | Maybe |
| Binary classification | Either | Either |
| Multi-class (12 classes like ours) | Careful | Usually better |
| Models from same family | No | Yes |

### Geomean-Weighted Rank Average (Advanced)
```python
def geomean_weighted_rank(predictions_list, weights):
    """
    Geometric mean of weighted rank predictions.
    Used by top Kaggle competitors for better calibration.
    """
    n_models = len(predictions_list)
    n_samples, n_classes = predictions_list[0].shape

    result = np.ones((n_samples, n_classes))
    for preds, w in zip(predictions_list, weights):
        ranked = np.zeros_like(preds)
        for c in range(n_classes):
            ranked[:, c] = rankdata(preds[:, c])
        # Normalize ranks to [0, 1]
        ranked = ranked / n_samples
        result *= ranked ** w

    return result
```

### For Our Case (12-class)
- Rank averaging is less intuitive with many classes
- Probability averaging with temperature scaling per model is usually better
- Use rank averaging only if models are severely miscalibrated

---

## 7. Neural Ensemble Selection (Greedy Forward Selection)

**The key question**: With 15+ models, which subset gives the best ensemble?

### Greedy Forward Selection Algorithm
```python
def greedy_ensemble_selection(models_predictions, labels, max_models=None,
                               metric='accuracy', n_bags=20):
    """
    Greedy forward selection with bagging (Caruana et al. 2004).

    models_predictions: dict {model_name: (n_samples, n_classes)}
    labels: ground truth
    max_models: max ensemble size (None = try all)
    n_bags: number of bootstrap samples for stability
    """
    model_names = list(models_predictions.keys())
    n_models = len(model_names)
    if max_models is None:
        max_models = n_models

    best_ensemble = []
    best_score = 0

    for step in range(max_models):
        best_candidate = None
        best_candidate_score = best_score

        for name in model_names:
            # Trial ensemble: existing + candidate
            trial = best_ensemble + [name]

            # Average predictions of trial ensemble
            trial_preds = np.mean(
                [models_predictions[m] for m in trial], axis=0
            )

            # Evaluate with bagging for stability
            scores = []
            for _ in range(n_bags):
                idx = np.random.choice(len(labels), len(labels), replace=True)
                score = compute_metric(trial_preds[idx], labels[idx], metric)
                scores.append(score)

            avg_score = np.mean(scores)

            if avg_score > best_candidate_score:
                best_candidate = name
                best_candidate_score = avg_score

        if best_candidate is None:
            break  # No improvement possible

        best_ensemble.append(best_candidate)
        best_score = best_candidate_score
        # NOTE: Allow same model to be selected again (with replacement)
        # This effectively gives it higher weight
        print(f"Step {step+1}: Added {best_candidate}, "
              f"score: {best_score:.4f}, size: {len(best_ensemble)}")

    return best_ensemble, best_score
```

### Diversity-Aware Selection (2024 Improvement)
```python
def diversity_aware_selection(models_predictions, labels,
                               alpha=0.7, beta=0.3):
    """
    Select models balancing accuracy and diversity.
    alpha: weight for accuracy improvement
    beta: weight for diversity contribution
    """
    model_names = list(models_predictions.keys())

    # Compute pairwise disagreement matrix
    disagreement = np.zeros((len(model_names), len(model_names)))
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            p1 = models_predictions[m1].argmax(1)
            p2 = models_predictions[m2].argmax(1)
            disagreement[i, j] = (p1 != p2).mean()

    selected = []
    remaining = list(range(len(model_names)))

    # Start with best individual model
    individual_scores = {
        i: (models_predictions[model_names[i]].argmax(1) == labels).mean()
        for i in remaining
    }
    best_first = max(individual_scores, key=individual_scores.get)
    selected.append(best_first)
    remaining.remove(best_first)

    while remaining:
        best_candidate = None
        best_combined_score = -1

        for idx in remaining:
            # Accuracy component
            trial = selected + [idx]
            trial_preds = np.mean(
                [models_predictions[model_names[i]] for i in trial], axis=0
            )
            acc = (trial_preds.argmax(1) == labels).mean()

            # Diversity component: avg disagreement with selected models
            div = np.mean([disagreement[idx, s] for s in selected])

            combined = alpha * acc + beta * div
            if combined > best_combined_score:
                best_combined_score = combined
                best_candidate = idx

        # Check if ensemble actually improves
        trial_preds = np.mean(
            [models_predictions[model_names[i]]
             for i in selected + [best_candidate]], axis=0
        )
        new_acc = (trial_preds.argmax(1) == labels).mean()

        current_preds = np.mean(
            [models_predictions[model_names[i]] for i in selected], axis=0
        )
        current_acc = (current_preds.argmax(1) == labels).mean()

        if new_acc >= current_acc:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return [model_names[i] for i in selected]
```

### Key Insights from NeurIPS 2024
- **Adaptive greedy** outperforms vanilla greedy across classification tasks
- Genetic algorithm-based selection with structural diversity + behavioral diversity + accuracy criterion effectively prunes redundant models
- For 15 models, the optimal subset is typically 5-8 models
- Selection with replacement (allowing a model to appear multiple times) implicitly learns weights

---

## 8. Feature-Level Fusion vs Decision-Level Fusion

### Comparison

| Aspect | Feature-Level (Early) | Decision-Level (Late) |
|--------|----------------------|----------------------|
| **Fusion Point** | Intermediate features | Final predictions |
| **Coupling** | Tight (single forward pass) | Loose (independent models) |
| **Flexibility** | Low (architectures must be compatible) | High (any model combination) |
| **Missing Modality** | Difficult to handle | Easy (just skip) |
| **Interaction Modeling** | Cross-feature interactions | No cross-model interaction |
| **Implementation** | Custom architecture needed | Simple averaging/voting |
| **Training** | End-to-end jointly | Models trained independently |
| **Current Trend (2024)** | Gaining popularity | Still dominant in competitions |

### Feature-Level Fusion Architecture
```python
import torch
import torch.nn as nn

class FeatureFusionClassifier(nn.Module):
    """
    Extract features from multiple backbones, concatenate, classify.
    More powerful but requires joint training.
    """
    def __init__(self, backbone_a, backbone_b, num_classes=12):
        super().__init__()
        self.backbone_a = backbone_a  # e.g., EfficientNet (remove classifier)
        self.backbone_b = backbone_b  # e.g., Swin (remove classifier)

        # Feature dimensions from each backbone
        dim_a = 1280  # EfficientNetV2-S
        dim_b = 768   # Swin-Tiny

        self.fusion = nn.Sequential(
            nn.Linear(dim_a + dim_b, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat_a = self.backbone_a(x)  # (B, 1280)
        feat_b = self.backbone_b(x)  # (B, 768)
        fused = torch.cat([feat_a, feat_b], dim=1)  # (B, 2048)
        return self.fusion(fused)
```

### Decision-Level Fusion (What we are doing)
```python
def decision_level_fusion(model_outputs, weights=None):
    """
    Simple but effective. Each model runs independently.
    model_outputs: list of (n_samples, n_classes) softmax arrays
    """
    if weights is None:
        return np.mean(model_outputs, axis=0)
    else:
        weights = np.array(weights) / sum(weights)
        return np.tensordot(weights, model_outputs, axes=([0], [0]))
```

### 2024 Trend: Intermediate Fusion
- Recent work shows intermediate fusion (fuse at penultimate layer) is gaining traction
- Combines benefits of both: some cross-model interaction + modular design
- For multimodal (image + tabular): intermediate fusion outperforms both early and late

### Recommendation for Our Case
- **Decision-level fusion** for final submission (simpler, more robust, no retraining needed)
- **Feature-level fusion** only if we have time to train a joint model from scratch
- Our 15+ classification models are already trained independently, so decision-level is the practical choice

---

## 9. Practical Guide: Our 15+ Classification Models

### Step-by-Step Ensemble Pipeline

```
Step 1: Generate OOF predictions
    - Run 5-fold CV for each model
    - Save out-of-fold (OOF) predictions: (n_train, 12) per model
    - Save test predictions (with TTA): (1276, 12) per model

Step 2: Evaluate individual models
    - Rank by OOF accuracy
    - Remove models with OOF accuracy < (best_model - 5%)

Step 3: Model Soups (within same architecture)
    - Greedy soup all EfficientNetV2-S variants -> 1 souped model
    - Greedy soup all ConvNeXt variants -> 1 souped model
    - Greedy soup all Swin variants -> 1 souped model

Step 4: Compute diversity matrix
    - Pairwise disagreement between all remaining models
    - Pairwise correlation of probability outputs

Step 5: Greedy Forward Selection
    - Start with best individual model
    - Add models that improve OOF accuracy
    - Allow selection with replacement
    - Typical result: 5-8 models selected

Step 6: Optimize weights
    - Use Nelder-Mead on OOF predictions
    - Constrain weights >= 0, sum to 1
    - Use 5-fold CV of the optimization itself to prevent overfitting

Step 7: Final ensemble
    - Apply optimized weights to test predictions
    - Compare with: uniform average, rank average, greedy selection
    - Pick the method with best OOF score
```

### Weight Optimization Without Overfitting

```python
from sklearn.model_selection import StratifiedKFold

def robust_weight_optimization(oof_predictions, oof_labels, n_models):
    """
    Nested CV to find weights without overfitting.
    oof_predictions: (n_models, n_samples, n_classes)
    """
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_weights = []

    for train_idx, val_idx in outer_cv.split(
        np.zeros(len(oof_labels)), oof_labels
    ):
        train_preds = oof_predictions[:, train_idx, :]
        train_labels = oof_labels[train_idx]

        weights = optimize_weights(train_preds, train_labels, n_models)

        # Validate
        val_preds = oof_predictions[:, val_idx, :]
        val_labels = oof_labels[val_idx]
        blended = np.tensordot(weights, val_preds, axes=([0], [0]))
        acc = (blended.argmax(1) == val_labels).mean()

        all_weights.append(weights)
        print(f"Fold val accuracy: {acc:.4f}, weights: {weights}")

    # Average weights across folds for stability
    final_weights = np.mean(all_weights, axis=0)
    final_weights = final_weights / final_weights.sum()
    return final_weights
```

---

## 10. Practical Guide: Our 10+ Segmentation Models

### Segmentation Ensemble Specifics

Segmentation ensembling differs from classification:
- Output is a 2D probability map, not a vector
- Models may produce different boundary sharpness
- Threshold selection matters (0.5 is not always optimal)

### Ensemble Methods for Segmentation

```python
def segmentation_ensemble(mask_predictions, weights=None, threshold=0.5):
    """
    mask_predictions: list of (H, W) probability maps from different models
    """
    if weights is None:
        avg_mask = np.mean(mask_predictions, axis=0)
    else:
        weights = np.array(weights) / sum(weights)
        avg_mask = np.tensordot(weights, mask_predictions, axes=([0], [0]))

    binary_mask = (avg_mask > threshold).astype(np.uint8) * 255
    return binary_mask

def majority_voting_segmentation(mask_predictions, threshold=0.5):
    """
    Pixel-wise majority voting. More robust to outlier models.
    """
    binary_masks = [(m > threshold).astype(int) for m in mask_predictions]
    votes = np.sum(binary_masks, axis=0)
    majority = (votes > len(mask_predictions) / 2).astype(np.uint8) * 255
    return majority

def optimized_threshold(oof_masks, oof_gt_masks, thresholds=None):
    """
    Find optimal binarization threshold using validation IoU.
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.7, 0.01)

    best_iou = 0
    best_t = 0.5

    for t in thresholds:
        binary = (oof_masks > t).astype(int)
        intersection = (binary * oof_gt_masks).sum()
        union = binary.sum() + oof_gt_masks.sum() - intersection
        iou = intersection / (union + 1e-7)

        if iou > best_iou:
            best_iou = iou
            best_t = t

    return best_t, best_iou
```

### Multi-Scale Test-Time Augmentation for Segmentation
```python
def multi_scale_tta_segmentation(model, image, scales=[0.75, 1.0, 1.25, 1.5]):
    """
    Predict at multiple scales, resize back, average.
    """
    h, w = image.shape[2:]
    accumulated = np.zeros((h, w), dtype=np.float32)

    for scale in scales:
        sh, sw = int(h * scale), int(w * scale)
        scaled = F.interpolate(image, size=(sh, sw), mode='bilinear')

        pred = model(scaled).sigmoid()
        pred = F.interpolate(pred, size=(h, w), mode='bilinear')
        accumulated += pred.squeeze().cpu().numpy()

        # Also flip
        pred_flip = model(scaled.flip(-1)).sigmoid()
        pred_flip = F.interpolate(pred_flip.flip(-1), size=(h, w), mode='bilinear')
        accumulated += pred_flip.squeeze().cpu().numpy()

    return accumulated / (2 * len(scales))
```

---

## 11. Diversity vs Quality: The Key Trade-off

### Empirical Rules from Competition Winners

1. **Never add a model that drops individual accuracy by >3%** just for diversity
2. **Correlation < 0.95** between a new model and existing ensemble members is the sweet spot
3. **Diminishing returns** after 5-7 diverse models for classification
4. **Architecture diversity contributes more** than hyperparameter diversity
5. **Different pretraining sources** (ImageNet-1k, ImageNet-21k, CLIP) provide meaningful diversity

### The Ensemble Equation
```
Ensemble_Error <= Average_Individual_Error - Diversity_Term

Where Diversity = (1/N^2) * sum of pairwise covariances of errors
```

**Practical meaning**: Adding a weak but uncorrelated model can help more than adding a strong but correlated model.

### Quick Diversity Check
```python
def should_add_model(new_preds, existing_ensemble_preds, labels,
                      min_acc_gap=0.03, max_correlation=0.95):
    """
    Quick check: should we add this model to our ensemble?
    """
    # Check individual quality
    new_acc = (new_preds.argmax(1) == labels).mean()
    best_acc = max(
        (p.argmax(1) == labels).mean() for p in existing_ensemble_preds
    )

    if new_acc < best_acc - min_acc_gap:
        return False, "Too weak"

    # Check diversity
    new_flat = new_preds.flatten()
    for existing in existing_ensemble_preds:
        corr = np.corrcoef(new_flat, existing.flatten())[0, 1]
        if corr > max_correlation:
            return False, f"Too correlated ({corr:.3f})"

    # Check if it actually improves ensemble
    current_blend = np.mean(existing_ensemble_preds, axis=0)
    current_acc = (current_blend.argmax(1) == labels).mean()

    new_blend = np.mean(
        list(existing_ensemble_preds) + [new_preds], axis=0
    )
    new_ensemble_acc = (new_blend.argmax(1) == labels).mean()

    if new_ensemble_acc > current_acc:
        return True, f"Improves ensemble by {new_ensemble_acc - current_acc:.4f}"

    return False, "Does not improve ensemble"
```

---

## 12. Summary: Recommended Strategy for Hackathon

### Classification (15+ models -> final prediction)

| Priority | Technique | Expected Gain | Effort |
|----------|-----------|---------------|--------|
| 1 | Greedy Forward Selection | +1-3% | Low |
| 2 | Optimized Weights (Nelder-Mead + CV) | +0.5-1.5% | Low |
| 3 | Model Soups (per architecture) | +0.5-1% | Low |
| 4 | Multi-Resolution Ensemble (384+512) | +1-2% | Medium |
| 5 | Snapshot Ensemble | +0.5-1% | Medium |
| 6 | Rank Average (if calibration issues) | +0-1% | Low |
| 7 | Stacking with LogisticRegression | +0-2% | Medium |

### Segmentation (10+ models -> final masks)

| Priority | Technique | Expected Gain | Effort |
|----------|-----------|---------------|--------|
| 1 | Probability Average of top models | +2-5% IoU | Low |
| 2 | Multi-Scale TTA (0.75x, 1x, 1.25x) | +1-3% IoU | Low |
| 3 | Optimized Threshold | +0.5-1% IoU | Low |
| 4 | Model Soups (same architecture) | +0.5-1% IoU | Low |
| 5 | Majority Voting | +0-1% IoU | Low |
| 6 | Greedy Ensemble Selection | +1-2% IoU | Medium |

### Quick-Win Pipeline (30 minutes)
```bash
# 1. Generate all OOF + test predictions
# 2. Greedy forward selection on OOF
# 3. Nelder-Mead weight optimization with CV
# 4. Apply to test set
# 5. Compare with simple average as sanity check
```

### Maximum Performance Pipeline (2-3 hours)
```bash
# 1. Model soups within each architecture family
# 2. Generate multi-resolution predictions (384, 512)
# 3. Greedy forward selection with diversity awareness
# 4. Nested CV weight optimization
# 5. Compare: weighted avg vs rank avg vs stacking
# 6. Select best based on OOF metric
# 7. For segmentation: optimize threshold per-model before ensembling
```

---

## References

- Wortsman et al. "Model soups: averaging weights of multiple fine-tuned models" (ICML 2022)
- Rethinking Weight-Averaged Model Merging (arXiv 2024, 2411.09263)
- Sparse Model Soups (NeurIPS 2024)
- Huang et al. "Snapshot Ensembles: Train 1, Get M for Free" (ICLR 2017)
- Snap-MAE: Integrating snapshot ensemble learning into masked autoencoders (Scientific Reports 2025)
- Caruana et al. "Ensemble Selection from Libraries of Models" (ICML 2004)
- ISIC 2024 Challenge winning solutions (Kaggle)
- Most Influential Subset Selection (NeurIPS 2024)
- A fuzzy rank-based ensemble of CNN models for classification (Scientific Reports 2021)
- Deep multimodal fusion review (PMC 2023)
- A Unified Theory of Diversity in Ensemble Learning (JMLR 2024)
