# Latest AI Breakthroughs for Medical Image Classification & Segmentation (2025-2026)

> Research compiled for WhiteCoat.dev Hackathon, March 27, 2026
> Constraint: ~15 hours remaining, ~5 hours GPU time

---

## PRIORITY MATRIX (Start Here)

| Technique | Benefit | Implementation Time | GPU Cost | **VERDICT** |
|-----------|---------|-------------------|----------|-------------|
| EMA | +0.5-1.5% acc | 10 min | Zero extra | **DO IT NOW** |
| SAM Optimizer | +0.5-2% acc | 15 min | 2x slower training | **DO IT** |
| Model Soups | +1-3% acc | 20 min (post-training) | Zero at inference | **DO IT** |
| DINOv2 backbone | +2-5% acc | 1-2 hours | Moderate | **STRONG YES if time** |
| Lion Optimizer | +1-2% acc, -33% VRAM | 15 min | Slightly less | **TRY if retraining** |
| TTA (advanced) | +0.5-1% acc | 30 min | Inference only | **YES for final submission** |
| FlashAttention | Faster training | 5 min (flag toggle) | Saves VRAM | **FREE, enable it** |
| MAE Pretraining | +1-3% on small data | 3-5 hours | Heavy | **NO - not enough time** |
| Mamba/SSM | Promising but new | 4-8 hours | Unknown | **NO - too risky** |
| NAS | Automated arch search | 10+ hours | Massive | **NO - way too slow** |

---

## 1. EMA (Exponential Moving Average) of Model Weights

### What It Is
Maintain a running average of model parameters during training. Instead of using the final checkpoint, use the EMA-smoothed weights for inference. This acts as implicit regularization.

### Why It Works
- Reduces noise from SGD updates, producing smoother loss landscapes
- Improves generalization, robustness to noisy labels, calibration, and transfer learning
- Virtually zero computational overhead

### Key Hyperparameters
- Start with decay = 0.999 (common default)
- For shorter training runs, try 0.99 (updates more aggressively)
- Can increase decay gradually as training converges (0.99 -> 0.999)

### Implementation (PyTorch)
```python
# Option A: timm's built-in ModelEmaV2
from timm.utils import ModelEmaV2
ema_model = ModelEmaV2(model, decay=0.999)

# In training loop:
for batch in dataloader:
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    ema_model.update(model)

# At inference: use ema_model.module
predictions = ema_model.module(test_images)

# Option B: PyTorch native
from torch.optim.swa_utils import AveragedModel
ema_model = AveragedModel(model, multi_avg_fn=...)
```

### Hackathon Applicability: EXCELLENT
- 10 minutes to add, zero GPU overhead
- Works with any architecture (EfficientNet, ConvNeXt, Swin, U-Net++)
- timm already supports it natively
- Expected gain: +0.5-1.5% accuracy / +0.5-1% IoU

### Sources
- [Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits (2024)](https://arxiv.org/abs/2411.18704)
- [timm ModelEMA documentation](https://timm.fast.ai/training_modelEMA)
- [How to Scale Your EMA - Apple ML Research](https://machinelearning.apple.com/research/scale-em)

---

## 2. SAM (Sharpness-Aware Minimization) Optimizer

### What It Is
An optimizer wrapper that seeks parameters in flat loss regions rather than sharp minima. Flat minima generalize better to unseen data. SAM performs two forward-backward passes per step.

### Why It Works
- Models trained with SAM find flatter minima that generalize better
- 2025 study confirmed: SAM is the **only** sharpness-based optimizer that **consistently** improves generalization in medical image analysis
- GCSAM variant (2025) reduces gradient noise and computational overhead
- SAM-FT (2026) specifically designed for medical image segmentation domain generalization

### Key Hyperparameters
- `rho = 0.05` (neighborhood size, default works well)
- Use with SGD or AdamW as the base optimizer
- Learning rate: same as your base optimizer

### Implementation (PyTorch)
```python
# pip install sam-pytorch  OR  copy from github.com/davda54/sam
from sam import SAM

base_optimizer = torch.optim.AdamW
optimizer = SAM(model.parameters(), base_optimizer, lr=1e-4, weight_decay=0.01, rho=0.05)

for images, labels in dataloader:
    # First forward-backward pass
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Second forward-backward pass
    criterion(model(images), labels).backward()
    optimizer.second_step(zero_grad=True)
```

### Hackathon Applicability: HIGH
- 15 min to implement
- Training takes 2x longer (two forward-backward passes) -- budget GPU time accordingly
- Recommended: use SAM only for the final fine-tuning phase (last 10-20 epochs)
- Expected gain: +0.5-2% accuracy, especially on minority classes

### Sources
- [Do Sharpness-Based Optimizers Improve Generalization in Medical Image Analysis? (2025)](https://arxiv.org/abs/2408.04065)
- [GCSAM: Gradient Centralized Sharpness Aware Minimization (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12599882/)
- [SAM-FT for Medical Image Segmentation (2026)](https://link.springer.com/chapter/10.1007/978-981-96-6688-1_7)
- [SAM PyTorch implementation](https://github.com/davda54/sam)

---

## 3. Model Soups (Weight Averaging of Fine-Tuned Models)

### What It Is
Instead of picking the single best model from hyperparameter search, average the weights of multiple fine-tuned models. This creates a "soup" that often outperforms any individual model -- with zero additional inference cost.

### Why It Works
- Fine-tuned models from the same pretrained checkpoint tend to lie in a single low-error basin
- Averaging weights is like ensembling but FREE at inference (same speed, same memory)
- Two recipes: Uniform Soup (average all) and Greedy Soup (add models only if they improve val score)

### Two Strategies
1. **Uniform Soup**: Average all model checkpoints. Simple, fast, often effective.
2. **Greedy Soup**: Sort models by val accuracy. Start with best model. Add next model to the average only if it improves val score. More reliable.

### Implementation (PyTorch)
```python
# Simple uniform soup of checkpoints
def make_soup(model, checkpoint_paths):
    """Average weights from multiple checkpoints."""
    state_dicts = [torch.load(p, map_location='cpu')['model'] for p in checkpoint_paths]
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = torch.mean(torch.stack([sd[key].float() for sd in state_dicts]), dim=0)
    model.load_state_dict(avg_state)
    return model

# Greedy soup
def greedy_soup(model, checkpoint_paths, val_loader, metric_fn):
    """Add checkpoints to soup only if they improve validation metric."""
    best_metric = 0
    soup_state = None
    n_ingredients = 0

    for path in sorted(checkpoint_paths, key=lambda p: get_val_score(p), reverse=True):
        sd = torch.load(path, map_location='cpu')['model']
        if soup_state is None:
            soup_state = {k: v.float() for k, v in sd.items()}
            n_ingredients = 1
        else:
            candidate = {}
            for key in soup_state:
                candidate[key] = (soup_state[key] * n_ingredients + sd[key].float()) / (n_ingredients + 1)
            model.load_state_dict(candidate)
            metric = metric_fn(model, val_loader)
            if metric > best_metric:
                soup_state = candidate
                best_metric = metric
                n_ingredients += 1

    model.load_state_dict(soup_state)
    return model
```

### Hackathon Applicability: EXCELLENT
- Apply AFTER training is done -- no extra GPU time for training
- Average your best classification checkpoints (different epochs, different hyperparams)
- Average your best segmentation checkpoints
- Can also average models from different folds if doing cross-validation
- Expected gain: +1-3% over single best model

### Sources
- [Model Soups: Averaging Weights of Multiple Fine-Tuned Models (Wortsman et al., 2022)](https://arxiv.org/abs/2203.05482)
- [Official implementation](https://github.com/mlfoundations/model-soups)
- [PyTorch AveragedModel](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)

---

## 4. DINOv2 as Feature Extractor

### What It Is
DINOv2 (Meta, 2023-2024) is a self-supervised Vision Transformer trained on 142M curated images. It produces powerful general-purpose visual features without any task-specific labels. You can use it as a frozen or fine-tuned backbone.

### Why It Works for Dermatology
- **2025 skin lesion study**: DINOv2-B achieved state-of-the-art 96.48% accuracy on a 31-class skin disease dataset, ~10% improvement over existing benchmarks
- Sustained 93% accuracy on ISIC 2019 external validation
- Outperformed ViT-B/16, Swin Transformer, and DenseNet121 with faster convergence
- Self-supervised features capture semantic structure without needing labeled dermatology data for pretraining

### Two Usage Modes
1. **Linear Probing** (fastest): Freeze DINOv2, train only a classification head
2. **Fine-tuning** (best performance): Unfreeze last few layers and train end-to-end with low LR

### Implementation
```python
import torch
import torch.nn as nn

# Load DINOv2 backbone
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

# Option 1: Linear probe (fast, 10 min to train)
class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = dinov2
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)

# Option 2: Fine-tune (better, 1-2 hours)
class DINOv2FineTune(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = dinov2
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
# Use lower LR for backbone (1e-5) and higher for head (1e-3)
```

### Hackathon Applicability: HIGH (if retraining classification)
- DINOv2 expects 224x224 or 518x518 images (resize needed from 100-128px)
- Linear probing: 10-20 min, decent results
- Fine-tuning: 1-2 hours, potentially significant gain
- **Risk**: Small input images (100-128px) upscaled to 224px may lose quality advantage
- Expected gain: +2-5% accuracy if it works well with your data

### Sources
- [Skin Lesion Classification Using DINOv2-B and Dermoscopic Imaging (2025)](https://www.ijraset.com/best-journal/skin-lesion-classification-using-dinov2b-and-dermoscopic-imaging-646)
- [Enhancing Skin Disease Classification with Transformer Architectures (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0010482525003580)
- [Explainable Self-Supervised Learning with DINO V2 for Medical Image Diagnosis (2025)](https://www.nature.com/articles/s41598-025-15604-6)
- [DINOv2 for Medical Image Analysis: Experimental Study (2024)](https://arxiv.org/html/2312.02366v3)

---

## 5. Lion Optimizer

### What It Is
Lion (EvoLved Sign Momentum) is an optimizer discovered by Google Brain using genetic algorithms. It uses only sign operations and momentum tracking, avoiding second-moment estimates like AdamW.

### Key Advantages
- **33% less GPU memory** than AdamW (no second moment buffer)
- Reported ~2% higher accuracy on ImageNet with ViT
- Faster convergence on transformer architectures
- Simpler algorithm, fewer hyperparameters

### Critical Hyperparameter Differences from AdamW
- Learning rate: **3-10x smaller** than AdamW (e.g., if AdamW uses 1e-4, try Lion with 1e-5)
- Weight decay: **3-10x larger** than AdamW (e.g., if AdamW uses 0.01, try Lion with 0.1)
- Betas: default (0.9, 0.99) works well

### Implementation
```python
# pip install lion-pytorch
from lion_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=1e-5,           # 3-10x smaller than AdamW
    weight_decay=0.1,  # 3-10x larger than AdamW
    betas=(0.9, 0.99)
)
```

### Hackathon Applicability: MODERATE
- Drop-in replacement for AdamW, 15 min to switch
- Memory savings useful if GPU memory is a constraint
- But: requires tuning LR and weight decay differently
- Best for: if you are retraining from scratch and have time to tune
- Expected gain: +1-2% accuracy, -33% VRAM usage

### Sources
- [Lion vs AdamW: Save 33% GPU Memory](https://blog.sotaaz.com/post/adamw-vs-lion-optimizer-comparison-en)
- [A Refined Lion Optimizer for Deep Learning (2025)](https://www.nature.com/articles/s41598-025-07112-4)
- [lion-pytorch implementation](https://github.com/lucidrains/lion-pytorch)
- [Modern Optimizers: AdamW, Lion, and What Works at Scale](https://medium.com/@spjosyula2005/modern-optimizers-adamw-lion-and-what-actually-works-at-scale-68ffc033713b)

---

## 6. Test-Time Training / Advanced Test-Time Adaptation

### What It Is
Beyond simple TTA (flip/rotate augmentation averaging), TTT adapts the model itself at inference time to each test image. The model updates its own parameters using a self-supervised objective on the test image before making a prediction.

### Recent Methods (2025-2026)
- **SicTTA (2025)**: Continual single-image test-time adaptation for medical segmentation; outperformed 7 state-of-the-art TTA methods
- **TEGDA (MICCAI 2025)**: Dynamic test-time adaptation that adjusts to each institution's domain without retraining
- **Med-TTT**: Vision backbone with TTT layers for linear-complexity adaptive inference
- **TTT-UNet**: U-Net enhanced with test-time training for segmentation

### Implementation Complexity
```python
# Simplified TTT concept (adapt batch norm at test time)
model.train()  # Keep BN layers in train mode
for module in model.modules():
    if not isinstance(module, torch.nn.BatchNorm2d):
        module.requires_grad_(False)

# For each test image:
for test_img in test_loader:
    # Forward pass updates BN running stats
    with torch.no_grad():
        pred = model(test_img)
```

### Hackathon Applicability: MODERATE
- Simple BN adaptation: 30 min, low risk
- Full TTT: 2-4 hours to implement properly, risky for a hackathon
- Standard augmentation-based TTA (8x flips+rotations) is safer and already planned
- Expected gain: +0.5-1% beyond standard TTA

### Sources
- [SicTTA: Single Image Continual Test Time Adaptation (2025)](https://www.sciencedirect.com/science/article/abs/pii/S1361841525004050)
- [TEGDA: Test-time Evaluation-Guided Dynamic Adaptation (MICCAI 2025)](https://papers.miccai.org/miccai-2025/0906-Paper2263.html)
- [MedSeg-TTA: Large Scale Benchmark for TTA in Medical Segmentation](https://arxiv.org/html/2512.02497v1)
- [Med-TTT: Vision Test-Time Training for Medical Segmentation](https://arxiv.org/html/2410.02523v1)

---

## 7. FlashAttention for Vision Transformers

### What It Is
IO-aware exact attention computation that reduces GPU memory reads/writes. FlashAttention-2/3 are drop-in replacements for standard attention in transformers. No accuracy change -- same math, just faster and less memory.

### Recent Advances
- **Flash Window Attention (2025)**: 3x speedup for windowed attention (Swin Transformer), 30% lower end-to-end runtime
- **ELFATT (2025)**: Linear-complexity attention, 4-7x speedup over vanilla softmax attention
- **FlashAttention-3**: Optimized for H100 GPUs with async tensor core operations

### Implementation
```python
# For timm models, FlashAttention is often auto-enabled
# Just ensure: pip install flash-attn
# Or use PyTorch 2.0+ SDPA (scaled dot-product attention):
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # Check if available

# For custom models:
from torch.nn.functional import scaled_dot_product_attention
# This automatically uses FlashAttention when available
```

### Hackathon Applicability: FREE (if using Swin/ViT)
- Already enabled by default in PyTorch 2.0+ with compatible GPUs
- 5 minutes to verify it is active
- No accuracy change, just speed and memory savings
- Most beneficial for: Swin Transformer, ViT backbones
- Less relevant for: CNN-based models (EfficientNet, ConvNeXt)

### Sources
- [FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al.)](https://arxiv.org/abs/2205.14135)
- [ELFATT: Efficient Linear Fast Attention for Vision Transformers (2025)](https://arxiv.org/abs/2501.06098)
- [FlashAttention-3 Paper](https://tridao.me/publications/flash3/flash3.pdf)

---

## 8. Masked Autoencoders (MAE) for Self-Supervised Pretraining

### What It Is
Mask 75% of image patches, train a ViT to reconstruct the missing patches. This self-supervised pretraining learns powerful visual representations without labels.

### Medical Imaging Results
- MAE self-pretraining outperforms ImageNet transfer learning on small medical datasets
- Swin MAE: works with only a few thousand images
- MSMAE (2024): Medical-specific masking strategy that avoids masking lesion regions
- GL-MAE (2025): Global-local masked autoencoders for volumetric medical images

### Hackathon Applicability: LOW (time constraint)
- Full MAE pretraining: 3-5 hours minimum
- Fine-tuning after MAE: another 1-2 hours
- Total: 4-7 hours -- too risky with only 5 hours GPU
- Better alternative: use DINOv2 (already pretrained) or ImageNet pretrained models
- Only consider if: you have a very specific domain shift problem that pretrained models do not handle

### Sources
- [Self Pre-training with MAE for Medical Image Classification and Segmentation](https://ar5iv.labs.arxiv.org/html/2203.05573)
- [Self-supervised Pre-training with Contrastive and MAE for Small Datasets (2023)](https://www.nature.com/articles/s41598-023-46433-0)
- [Swin MAE: Masked Autoencoders for Small Datasets](https://www.sciencedirect.com/science/article/abs/pii/S0010482523005024)
- [Medical Supervised Masked Autoencoder (MSMAE)](https://www.sciencedirect.com/science/article/abs/pii/S1568494624013103)

---

## 9. Mamba / State Space Models for Vision

### What It Is
Mamba is a state space model (SSM) architecture that captures long-range dependencies with linear (not quadratic) complexity. Multiple Mamba-based medical segmentation models emerged in 2025-2026.

### Recent Models
- **SegMamba-V2 (2025)**: Outperformed SOTA on 3D medical segmentation
- **Switch-UMamba (2025)**: Dynamic scanning Mamba UNet, SOTA on medical benchmarks without pretraining
- **CFG-MambaNet (2026)**: Published in npj Digital Medicine, linear complexity for high-res medical images
- **MedMamba (2025)**: Multi-scale deformable attention via SSMs

### Hackathon Applicability: LOW (too risky)
- Novel architecture, limited battle-tested code
- 4-8 hours to adapt to your specific task
- Debugging Mamba issues would eat into remaining time
- Most benefits are for 3D / high-resolution images (not our 100-128px case)
- The linear complexity advantage is negligible at small image sizes
- Stick with proven architectures (EfficientNet, U-Net++, Swin)

### Sources
- [SegMamba-V2: Long-Range Sequential Modeling for 3D Medical Segmentation](https://pubmed.ncbi.nlm.nih.gov/40679879/)
- [Switch-UMamba: Dynamic Scanning Vision Mamba UNet (2025)](https://pubmed.ncbi.nlm.nih.gov/40961581/)
- [CFG-MambaNet (npj Digital Medicine, 2026)](https://www.nature.com/articles/s41746-026-02393-z)
- [Mamba in CV: Paper List](https://github.com/Yangzhangcst/Mamba-in-CV)

---

## 10. Neural Architecture Search (NAS)

### What It Is
Automated search for optimal neural network architecture. Recent methods use zero-cost proxies and evolutionary algorithms to find architectures faster.

### Recent Medical Imaging Results
- BioNAS (2025): Highest average accuracy of 0.848 across medical benchmarks
- ZO-DARTS+ (2025): Matches SOTA accuracy with 3x faster search time
- AEG-cTFNAS (MICCAI 2025): Training-free NAS for UNet-based segmentation

### Hackathon Applicability: NOT RECOMMENDED
- Even "fast" NAS takes 10+ hours of GPU compute
- Requires significant infrastructure setup
- Results are not guaranteed to beat well-tuned standard architectures
- Our model choices (EfficientNetV2-S, ConvNeXt, Swin, U-Net++) are already near-optimal for this scale
- Time better spent on training tricks (EMA, SAM, Model Soups)

### Sources
- [NAS for Biomedical Image Classification: Comparative Study (2025)](https://pubmed.ncbi.nlm.nih.gov/39778345/)
- [A Lightweight NAS Model for Medical Image Classification](https://arxiv.org/abs/2405.03462)
- [Compact Training-Free NAS for Medical Segmentation (MICCAI 2025)](https://link.springer.com/chapter/10.1007/978-3-032-05325-1_11)

---

## RECOMMENDED ACTION PLAN (Priority Order)

Given ~15 hours total, ~5 hours GPU:

### Phase 1: Quick Wins (30 min setup, runs during training)
1. **Enable EMA** in all training scripts (10 min, +0.5-1.5%)
2. **Enable FlashAttention** -- verify PyTorch SDPA is active for Swin models (5 min, free speedup)
3. **Switch to SAM optimizer** for final fine-tuning epochs (15 min, +0.5-2%)

### Phase 2: Post-Training Boosts (30 min, no GPU needed)
4. **Model Soups** -- average your top 3-5 checkpoints per task (20 min, +1-3%)
5. **Advanced TTA** -- if not already doing 8x augmentation TTA, implement it (10 min, +0.5-1%)

### Phase 3: If Time Permits (1-2 hours)
6. **DINOv2 linear probe** -- quick experiment on classification task (1 hour)
7. **Lion optimizer** -- try for one model if AdamW results are disappointing (15 min)

### DO NOT ATTEMPT (for this hackathon)
- MAE pretraining (too slow)
- Mamba architecture (too risky, small images negate benefits)
- Full NAS search (way too slow)
- Full TTT implementation (too complex for remaining time)

---

## EXPECTED CUMULATIVE GAIN

If you apply EMA + SAM + Model Soups + proper TTA:
- Classification: +2-5% accuracy improvement over baseline
- Segmentation: +1-3% mean IoU improvement over baseline
- Implementation time: ~1.5 hours total
- Extra GPU time: ~30% longer training (SAM only)

These are the highest return-on-investment techniques for a hackathon setting.
