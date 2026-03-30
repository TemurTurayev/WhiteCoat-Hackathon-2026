# Data-Centric AI: Boost Model Quality Without Changing Architecture

> "It is now more productive to hold the neural network architecture fixed and instead find ways to improve the data." -- Andrew Ng

## Overview

Data-centric AI is the discipline of systematically engineering the data used to build an AI system. Instead of tweaking architectures, loss functions, or hyperparameters, you improve the DATA itself -- fixing labels, removing noise, ordering examples by difficulty, and augmenting smartly. Research from 2024-2026 consistently shows this yields higher returns per hour invested than model-centric approaches.

---

## 1. Data Quality Over Quantity

### The Core Insight

A model trained on 5,000 clean examples often outperforms one trained on 50,000 noisy examples. For small datasets like ours (11,411 classification images, 2,200 segmentation images), every mislabeled example has outsized impact.

### Practical Steps (Andrew Ng's Framework)

1. **Error analysis first** -- Train a baseline model, examine its worst predictions. Misclassifications often cluster around label errors, not model failures.
2. **Make labels consistent** -- Map labels to a deterministic function. If two dermatologists would disagree, clarify the labeling rule.
3. **Focus on the tail** -- The smallest classes (Class 5: 441, Class 8: 331) are most vulnerable to label noise. A single mislabeled image in Class 8 shifts accuracy by ~0.3%.
4. **Remove noisy examples** -- Toss out examples that confuse the model rather than help it learn.
5. **Document and iterate** -- Keep a log of what you fix and retrain.

### Impact Estimate

Fixing 2-5% mislabeled examples in a medical imaging dataset typically yields +1-3% accuracy improvement -- sometimes more on minority classes.

---

## 2. Confident Learning (cleanlab)

### What It Does

cleanlab implements "confident learning" -- a statistical framework that uses out-of-sample predicted probabilities to estimate the joint distribution of noisy and true labels. It then ranks every example by likelihood of being mislabeled.

### How It Works

1. Train your model with K-fold cross-validation to get out-of-sample `pred_probs` for every training example.
2. cleanlab compares predicted probabilities against given labels.
3. Examples where the model confidently disagrees with the label are flagged.
4. You review and fix (or remove) the flagged examples.

### Code for Classification

```python
import cleanlab
from cleanlab.filter import find_label_issues

# pred_probs: (N, 12) array of out-of-sample predicted probabilities
# labels: (N,) array of integer labels 0-11
# Get pred_probs via K-fold cross-validation on your trained model

ranked_label_issues = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence"
)

# Top-ranked indices are most likely mislabeled
print(f"Found {len(ranked_label_issues)} potential label errors")
print(f"Top 50 most likely errors: {ranked_label_issues[:50]}")
```

### Code for Segmentation

cleanlab v2.5+ supports semantic segmentation:

```python
from cleanlab.segmentation.summary import display_issues, common_label_issues

# pred_probs: (N, H, W, C) predicted class probabilities per pixel
# labels: (N, H, W) integer label masks

issues = common_label_issues(labels, pred_probs)
# Returns per-image quality scores -- low scores = likely bad masks
```

### Performance

Research shows cleanlab can improve model accuracy by up to 15% by identifying and correcting label errors. For dermatology datasets where inter-annotator disagreement is common (especially between similar classes like different types of nevi), this is particularly impactful.

### Time Estimate: 1-2 hours

- 30 min: Generate K-fold cross-validated predictions
- 30 min: Run cleanlab, visualize top-50 suspicious labels
- 30 min: Fix or remove confirmed errors, retrain

---

## 3. Curriculum Learning

### Concept

Train on easy examples first, then gradually introduce harder examples. This mimics how humans learn -- master basics before tackling edge cases.

### Why It Works for Our Data

- Skin lesion classes overlap visually (e.g., melanoma vs. dysplastic nevus)
- Starting with clear, prototypical examples helps the model learn robust features
- Hard/ambiguous examples are introduced only after the model has stable representations

### Implementation Approaches

**Self-paced curriculum (simplest):**
```python
# After initial training, compute per-sample loss
losses = compute_per_sample_loss(model, train_loader)

# Sort by loss (easy = low loss, hard = high loss)
sorted_indices = losses.argsort()

# Epoch 1-5: train on easiest 50%
# Epoch 6-10: train on easiest 75%
# Epoch 11+: train on all data
for epoch in range(num_epochs):
    fraction = min(0.5 + epoch * 0.05, 1.0)
    n_samples = int(len(sorted_indices) * fraction)
    curriculum_indices = sorted_indices[:n_samples]
    train_one_epoch(model, subset(train_data, curriculum_indices))
```

**Anti-curriculum (for noisy data):**
- Counterintuitively, training on HARD examples first can help if the hard examples are actually mislabeled -- the model learns to be robust to noise early.

### When to Use

- When you suspect label noise (hard examples might be mislabeled)
- When classes are highly imbalanced (combine with oversampling of minority classes)
- When training time is limited (converges faster on easy examples)

### Time Estimate: 30-60 minutes

Requires only modifying the data loader, no architecture changes.

---

## 4. Label Smoothing vs. Mixup vs. CutMix

### Comparison for Small Datasets

| Technique | How It Works | Best For | Risk |
|-----------|-------------|----------|------|
| **Label Smoothing** | Softens targets: `y = 0.9 * one_hot + 0.1/K` | Noisy labels, calibration | May hurt if labels are clean |
| **Mixup** | Blends two images and labels: `x = a*x1 + (1-a)*x2` | Small datasets, regularization | Unrealistic blended images |
| **CutMix** | Patches one image onto another, mixes labels proportionally | Localization, small datasets | Can destroy small lesions |

### Recommendation for Skin Lesion Classification

**Use Label Smoothing (0.1) + CutMix together.**

Rationale:
- Label smoothing handles inherent ambiguity between similar dermatological classes
- CutMix forces the model to look at multiple regions (not just the center of the lesion)
- Research shows CutMix outperforms Mixup for localization tasks by +5% on CUB200
- CutMix + Label Smoothing together outperform either alone
- Mixup creates unrealistic blended skin images that may confuse the model

### Implementation

```python
# Label smoothing: just change the loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# CutMix: use timm's built-in implementation
from timm.data.mixup import Mixup

mixup_fn = Mixup(
    mixup_alpha=0.0,     # disable mixup
    cutmix_alpha=1.0,    # enable cutmix
    prob=0.5,            # apply to 50% of batches
    label_smoothing=0.1,
    num_classes=12
)

# In training loop:
for images, targets in train_loader:
    images, targets = mixup_fn(images, targets)
    outputs = model(images)
    loss = criterion(outputs, targets)
```

### Time Estimate: 15 minutes

Just add two lines to existing training code.

---

## 5. Data Pruning

### Concept

Not all training examples help generalization. Some are redundant, some are harmful (mislabeled or ambiguous). Removing them can improve accuracy AND reduce training time.

### Methods

**Forgetting Events (simplest, most practical):**
```python
# Track how many times each example flips between correct/incorrect
# across training epochs
forgetting_counts = torch.zeros(len(train_dataset))

for epoch in range(num_epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct = (preds == targets)

        # Track transitions from correct -> incorrect
        if epoch > 0:
            was_correct = previous_correct[batch_indices]
            forgetting_counts[batch_indices] += (was_correct & ~correct).float()

        previous_correct[batch_indices] = correct
```

**Interpretation:**
- **Never forgotten** (forgetting_count = 0): Easy examples, potentially redundant
- **Frequently forgotten** (high count): Either hard but informative, OR mislabeled
- **Sweet spot**: Keep examples with moderate forgetting counts, investigate extremes

**EL2N Score (faster alternative):**
```python
# Error L2 Norm -- computed from early training
# High EL2N = hard/noisy, Low EL2N = easy/redundant
el2n_scores = []
for images, targets in train_loader:
    probs = F.softmax(model(images), dim=1)
    one_hot = F.one_hot(targets, num_classes=12).float()
    el2n = (probs - one_hot).norm(dim=1)
    el2n_scores.append(el2n)
```

### Research Finding

Pruning 40% of training data on CIFAR-10 halved convergence time with only 1.3% accuracy decrease. For noisy datasets, pruning harmful examples can actually INCREASE accuracy.

### Our Strategy

1. Train baseline for 5-10 epochs
2. Compute forgetting events or EL2N scores
3. Remove top 5% highest-scoring examples (likely mislabeled)
4. Remove bottom 20% easiest examples (redundant, speed up training)
5. Retrain on pruned dataset

### Time Estimate: 45 minutes

Requires one full training run to compute scores, then a retrain.

---

## 6. Active Learning

### Concept

Given a pool of unlabeled data, select the examples that would benefit the model most if labeled. This is less relevant for our hackathon (all data is labeled), but the UNCERTAINTY SCORES from active learning are useful for:

- Identifying ambiguous examples for review
- Prioritizing which classes need more augmentation
- Detecting distribution shift between train and test

### Query Strategies

| Strategy | How | When |
|----------|-----|------|
| **Uncertainty sampling** | Select examples where model is least confident | General purpose |
| **Margin sampling** | Select where top-2 class probabilities are closest | Multi-class problems |
| **Entropy sampling** | Select examples with highest prediction entropy | Many classes |
| **Query by committee** | Select where multiple models disagree | Ensemble available |

### Practical Application for Our Hackathon

Even though we have labels for everything, we can use uncertainty sampling to find "borderline" examples:

```python
# After training, compute entropy on training set
with torch.no_grad():
    probs = F.softmax(model(images), dim=1)
    entropy = -(probs * probs.log()).sum(dim=1)

# High entropy = model is confused = possible label error or hard example
suspicious = entropy.argsort(descending=True)[:100]
```

### Time Estimate: 20 minutes

Reuse existing model predictions.

---

## 7. Andrew Ng's Data-Centric AI Framework -- Practical Checklist

### The Iterative Loop

```
1. Train model (keep architecture fixed)
2. Error analysis (examine worst predictions)
3. Identify data problems (label errors, ambiguity, missing classes)
4. Fix data (relabel, remove, augment)
5. Retrain
6. Repeat until convergence
```

### Ng's Key Principles Applied to Our Dataset

| Principle | Our Application |
|-----------|----------------|
| Fix data, not model | Keep EfficientNetV2/ConvNeXt ensemble, improve labels |
| Consistency > volume | Ensure all 12 classes have consistent labeling criteria |
| Small data needs clean data | With only 331-2136 per class, every label matters |
| Error analysis drives iteration | Focus on confused class pairs (e.g., Class 5 vs Class 0) |
| Multiple labelers reduce noise | Cross-check with ISIC Archive descriptions if available |
| Data augmentation = synthetic data | Smart augmentation (CutMix) acts as data-centric improvement |

---

## Recommended 2-3 Hour Action Plan

### Hour 1: Label Cleaning (Highest ROI)

**Classification (45 min):**
1. Generate K-fold cross-validated predictions from your best classification model
2. Run cleanlab `find_label_issues()` on training data
3. Visualize top 50-100 suspicious images
4. Remove confirmed mislabeled images (expect 2-5% of dataset)
5. Retrain classification model on cleaned data

**Segmentation (15 min):**
1. Run cleanlab segmentation module on training masks
2. Identify images with lowest mask quality scores
3. Remove or downweight worst 5% of masks

### Hour 2: Smart Regularization (Medium ROI)

**Both tasks (30 min):**
1. Add label smoothing (0.1) to both classification and segmentation losses
2. Add CutMix to classification training (prob=0.5, alpha=1.0)
3. Retrain both models

**Classification only (30 min):**
1. Implement simple curriculum learning: train on easy 60% for first 5 epochs, then all data
2. Or: compute EL2N scores, prune bottom 20% easiest + top 5% noisiest

### Hour 3: Verification and Fine-tuning (Lower ROI but important)

1. Run error analysis on validation set predictions
2. Identify remaining confused class pairs
3. Apply targeted augmentation to confused classes
4. Final retrain with all improvements combined
5. Generate submission predictions

### Expected Improvement

| Technique | Classification | Segmentation | Time |
|-----------|---------------|--------------|------|
| cleanlab label cleaning | +1-3% accuracy | +1-2% IoU | 60 min |
| Label smoothing (0.1) | +0.5-1% accuracy | +0.5% IoU | 5 min |
| CutMix augmentation | +0.5-1.5% accuracy | N/A | 10 min |
| Curriculum learning | +0.5-1% accuracy | +0.5% IoU | 30 min |
| Data pruning (top 5% noise) | +0.5-1% accuracy | +0.5-1% IoU | 45 min |
| **Combined** | **+2-5% accuracy** | **+1-3% IoU** | **2-3 hours** |

---

## Quick-Start Code: Full Pipeline

```python
"""
data_centric_pipeline.py -- Apply data-centric improvements in one script
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues


def get_cross_val_predictions(model_class, dataset, n_splits=5):
    """Get out-of-sample predictions via K-fold CV."""
    all_probs = np.zeros((len(dataset), 12))
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
        # Train model on train_idx, predict on val_idx
        # ... (use your existing training code)
        fold_probs = predict_probs(model_class, dataset, train_idx, val_idx)
        all_probs[val_idx] = fold_probs

    return all_probs, labels


def clean_labels(pred_probs, labels, remove_fraction=0.05):
    """Find and remove likely mislabeled examples."""
    issues = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"
    )
    n_remove = int(len(labels) * remove_fraction)
    indices_to_remove = set(issues[:n_remove])
    clean_indices = [i for i in range(len(labels)) if i not in indices_to_remove]

    print(f"Removing {n_remove} likely mislabeled examples out of {len(labels)}")
    return clean_indices


def compute_curriculum_order(model, dataset):
    """Sort examples by difficulty (loss) for curriculum learning."""
    model.train(False)
    losses = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.unsqueeze(0).cuda()
            output = model(img)
            loss = F.cross_entropy(output, torch.tensor([label]).cuda())
            losses.append(loss.item())
    return np.argsort(losses)  # easy to hard


def compute_el2n_scores(model, dataset):
    """Compute Error L2 Norm scores for data pruning."""
    model.train(False)
    scores = []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = img.unsqueeze(0).cuda()
            probs = F.softmax(model(img), dim=1)
            one_hot = F.one_hot(
                torch.tensor([label]), num_classes=12
            ).float().cuda()
            el2n = (probs - one_hot).norm(dim=1).item()
            scores.append(el2n)
    return np.array(scores)
```

---

## Key Takeaways

1. **Label cleaning with cleanlab is the single highest-ROI technique** -- 60 minutes of work for +1-3% accuracy on medical imaging datasets with inherent label ambiguity.

2. **Label smoothing is free performance** -- one line of code, consistent +0.5-1% improvement, especially valuable when classes overlap visually.

3. **CutMix beats Mixup for localization-sensitive tasks** like skin lesion classification where the model needs to focus on specific regions.

4. **Data pruning removes both redundant and harmful examples** -- particularly effective for imbalanced datasets where minority classes are most affected by noise.

5. **Curriculum learning helps convergence** but is less impactful than label cleaning. Use it if you have time remaining.

6. **Combine techniques** -- they are complementary. Clean labels first (cleanlab), then add regularization (label smoothing + CutMix), then curriculum ordering.

---

## References

- cleanlab library: https://github.com/cleanlab/cleanlab
- cleanlab segmentation tutorial: https://docs.cleanlab.ai/v2.7.1/tutorials/segmentation.html
- cleanlab image classification tutorial: https://docs.cleanlab.ai/v2.1.0/tutorials/image.html
- Andrew Ng on data-centric AI: https://spectrum.ieee.org/andrew-ng-data-centric-ai
- MIT data-centric AI course: https://dcai.csail.mit.edu/2024/data-centric-model-centric/
- Dataset pruning research: https://arxiv.org/abs/2205.09329
- CutMix paper (ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf
- Mixup augmentations survey: https://arxiv.org/html/2409.05202v1
- Active learning guide: https://encord.com/blog/active-learning-machine-learning-guide/
- Label smoothing analysis (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/file/9dc5accb1e4f4a9798eae145f2e4869b-Paper-Conference.pdf
