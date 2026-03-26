"""
Advanced Loss Functions & Training Utilities for AI in Healthcare Hackathon 2026
Team WhiteCoat.dev

Contains:
1. Focal Loss (12-class skin lesion classification)
2. Lovasz Hinge Loss (binary segmentation)
3. Dice Loss, Tversky Loss, Boundary Loss (segmentation)
4. Combined/Hybrid Loss Strategies
5. CutMix / Mixup Augmentation (standalone + timm)
6. Stochastic Weight Averaging (SWA)
7. Cosine Annealing with Warm Restarts + OneCycleLR comparison
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. FOCAL LOSS — 12-class skin lesion classification
# =============================================================================
# Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
# Skin-lesion evidence:
#   - gamma=2.0 is the standard starting point (Lin et al.)
#   - For ISIC-class datasets, gamma in [2, 3] works best
#     (see "A Deep CNN Transformer Hybrid Model for Skin Lesion
#      Classification Using Focal Loss", PMC 2023)
#   - alpha (per-class) = inverse frequency works well with focal
# =============================================================================


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with optional label smoothing.

    Args:
        gamma: Focusing parameter. Higher -> more focus on hard examples.
               Recommended: 2.0 (standard), up to 3.0 for very hard tasks.
        alpha: Per-class weights tensor of shape (C,). Use inverse frequency.
               If None, all classes weighted equally.
        label_smoothing: Smooth targets toward uniform. 0.0 = no smoothing.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (N, C) raw model output (before softmax).
            targets: (N,) integer class labels in [0, C-1].

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        num_classes = logits.size(1)

        # --- label smoothing ---
        if self.label_smoothing > 0.0:
            with torch.no_grad():
                smooth_targets = torch.full_like(
                    logits, self.label_smoothing / (num_classes - 1)
                )
                smooth_targets.scatter_(
                    1,
                    targets.unsqueeze(1),
                    1.0 - self.label_smoothing,
                )
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # --- focal weight ---
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        focal_weight = (1.0 - probs) ** self.gamma

        # --- alpha weighting ---
        if self.alpha is not None:
            alpha_t = self.alpha.unsqueeze(0).expand_as(logits)
            focal_weight = focal_weight * alpha_t

        # --- focal cross-entropy ---
        loss = -focal_weight * smooth_targets * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(
    samples_per_class: Sequence[int],
    normalize: bool = True,
) -> Tensor:
    """Compute inverse-frequency class weights.

    Args:
        samples_per_class: List/tuple of sample counts per class.
        normalize: If True, weights sum to num_classes.

    Returns:
        Tensor of shape (C,).
    """
    counts = torch.tensor(samples_per_class, dtype=torch.float32)
    weights = 1.0 / counts
    if normalize:
        weights = weights * (len(samples_per_class) / weights.sum())
    return weights


# =============================================================================
# 2. LOVASZ HINGE LOSS — binary segmentation (optimizes IoU directly)
# =============================================================================
# Paper: "The Lovasz-Softmax loss" (Berman et al., CVPR 2018)
# Reference: github.com/bermanmaxim/LovaszSoftmax
#
# Strategy:
#   Phase 1 (epochs 0-19): Dice + BCE for stable convergence
#   Phase 2 (epochs 20-39): Add Lovasz to directly optimize IoU
#   Switch trigger: validation loss plateau OR fixed epoch (simpler)
# =============================================================================


def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    """Compute gradient of the Lovasz extension w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits: Tensor, labels: Tensor) -> Tensor:
    """Binary Lovasz hinge loss, flat version.

    Args:
        logits: (P,) float logits (raw, before sigmoid).
        labels: (P,) binary ground truth {0, 1}.
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """Binary Lovasz Hinge Loss — directly optimizes IoU.

    Args:
        per_image: If True, compute loss per image then average.
                   Recommended True for variable-size batches.
    """

    def __init__(self, per_image: bool = True) -> None:
        super().__init__()
        self.per_image = per_image

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (N, 1, H, W) raw predictions (before sigmoid).
            targets: (N, H, W) or (N, 1, H, W) binary masks {0, 1}.
        """
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        if self.per_image:
            losses = []
            for logit, target in zip(logits, targets):
                loss = _lovasz_hinge_flat(
                    logit.reshape(-1), target.reshape(-1)
                )
                losses.append(loss)
            return torch.stack(losses).mean()

        return _lovasz_hinge_flat(
            logits.reshape(-1), targets.reshape(-1)
        )


# =============================================================================
# 3. DICE LOSS, TVERSKY LOSS, BOUNDARY LOSS
# =============================================================================


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation.

    Args:
        smooth: Smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (N, 1, H, W) raw predictions.
            targets: (N, H, W) or (N, 1, H, W) binary masks.
        """
        probs = torch.sigmoid(logits)
        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1.0 - dice).mean()


class TverskyLoss(nn.Module):
    """Tversky Loss — asymmetric Dice that penalizes FN more than FP.

    For skin lesion segmentation where missing lesion pixels is worse
    than false positives, use alpha=0.7, beta=0.3.

    Args:
        alpha: Weight for false positives.
        beta: Weight for false negatives.
        smooth: Smoothing constant.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1).float()

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky).mean()


class BoundaryLoss(nn.Module):
    """Boundary Loss — penalizes predictions far from true boundary.

    Requires precomputed distance maps (signed distance transform of GT).
    Use scipy.ndimage.distance_transform_edt in your dataset __getitem__.
    """

    def forward(self, logits: Tensor, dist_maps: Tensor) -> Tensor:
        """
        Args:
            logits: (N, 1, H, W) raw predictions.
            dist_maps: (N, H, W) signed distance transform of GT.
                       Negative inside the object, positive outside.
        """
        probs = torch.sigmoid(logits)
        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs.squeeze(1)
        return (probs * dist_maps).mean()


# =============================================================================
# 4. COMBINED / HYBRID LOSS STRATEGIES
# =============================================================================


class DiceBCELoss(nn.Module):
    """Dice + Binary Cross-Entropy — standard stable baseline.

    Args:
        dice_weight: Weight for Dice component.
        bce_weight: Weight for BCE component.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if targets.dim() == 3:
            targets_bce = targets.unsqueeze(1).float()
        else:
            targets_bce = targets.float()
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.bce_weight * self.bce(logits, targets_bce)
        )


class TwoPhaseSegmentationLoss(nn.Module):
    """Two-phase training loss for segmentation.

    Phase 1 (warm-up):  Dice + BCE  — stable convergence
    Phase 2 (IoU push):  Dice + BCE + Lovasz  — direct IoU optimization

    Args:
        switch_epoch: Epoch to transition to Phase 2.
        lovasz_weight: Weight for Lovasz in Phase 2.
        dice_weight: Weight for Dice in both phases.
        bce_weight: Weight for BCE in both phases.
    """

    def __init__(
        self,
        switch_epoch: int = 20,
        lovasz_weight: float = 0.5,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.switch_epoch = switch_epoch
        self.lovasz_weight = lovasz_weight
        self.dice_bce = DiceBCELoss(dice_weight, bce_weight)
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call this at the beginning of each epoch."""
        self.current_epoch = epoch

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = self.dice_bce(logits, targets)
        if self.current_epoch >= self.switch_epoch:
            loss = loss + self.lovasz_weight * self.lovasz(logits, targets)
        return loss


class FullHybridSegmentationLoss(nn.Module):
    """Full hybrid: Dice + BCE + Lovasz + Tversky + optional Boundary.

    Recommended weights (from literature on medical segmentation):
        dice=1.0, bce=1.0, lovasz=0.5, tversky=0.5, boundary=0.0

    Enable boundary_weight > 0 only after warm-up (epoch > ~10).

    Args:
        dice_weight: Weight for Dice.
        bce_weight: Weight for BCE.
        lovasz_weight: Weight for Lovasz.
        tversky_weight: Weight for Tversky.
        boundary_weight: Weight for Boundary loss (0 = disabled).
        tversky_alpha: FP penalty for Tversky.
        tversky_beta: FN penalty for Tversky.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        lovasz_weight: float = 0.5,
        tversky_weight: float = 0.5,
        boundary_weight: float = 0.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary = BoundaryLoss()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        dist_maps: Optional[Tensor] = None,
    ) -> Tensor:
        if targets.dim() == 3:
            targets_bce = targets.unsqueeze(1).float()
        else:
            targets_bce = targets.float()

        loss = (
            self.dice_weight * self.dice(logits, targets)
            + self.bce_weight * self.bce(logits, targets_bce)
            + self.lovasz_weight * self.lovasz(logits, targets)
            + self.tversky_weight * self.tversky(logits, targets)
        )
        if self.boundary_weight > 0.0 and dist_maps is not None:
            loss = loss + self.boundary_weight * self.boundary(logits, dist_maps)
        return loss


# =============================================================================
# 5. CUTMIX / MIXUP AUGMENTATION
# =============================================================================


def rand_bbox(
    height: int, width: int, lam: float
) -> tuple[int, int, int, int]:
    """Generate random bounding box for CutMix.

    Args:
        height: Image height.
        width: Image width.
        lam: Lambda from Beta distribution.

    Returns:
        (y1, y2, x1, x2) bounding box.
    """
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)

    cy = torch.randint(0, height, (1,)).item()
    cx = torch.randint(0, width, (1,)).item()

    y1 = max(0, cy - cut_h // 2)
    y2 = min(height, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(width, cx + cut_w // 2)

    return y1, y2, x1, x2


class CutMixCollator:
    """CutMix collation function for DataLoader.

    Standalone implementation — no external dependencies.
    For 100x100 images, alpha=1.0 and prob=0.5 work well.

    Args:
        alpha: Beta distribution parameter. 1.0 = uniform mixing ratios.
        prob: Probability of applying CutMix to a batch.
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5) -> None:
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self, batch: list[tuple[Tensor, int]]
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        """
        Args:
            batch: List of (image, label) tuples from Dataset.

        Returns:
            images: (N, C, H, W)
            targets_a: (N,) original labels
            targets_b: (N,) shuffled labels
            lam: Effective lambda after bbox adjustment
        """
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])

        if torch.rand(1).item() > self.prob:
            return images, labels, labels, 1.0

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        _, _, h, w = images.shape
        y1, y2, x1, x2 = rand_bbox(h, w, lam)

        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda to match actual area ratio
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)

        return images, labels, labels[index], lam


def cutmix_criterion(
    criterion: nn.Module,
    logits: Tensor,
    targets_a: Tensor,
    targets_b: Tensor,
    lam: float,
) -> Tensor:
    """Compute CutMix loss with mixed labels.

    Args:
        criterion: Base loss function (e.g., FocalLoss or CrossEntropyLoss).
        logits: (N, C) model output.
        targets_a: Original labels.
        targets_b: Shuffled labels.
        lam: Mixing ratio.
    """
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(
        logits, targets_b
    )


# ---- timm CutMix/Mixup (recommended if timm is installed) ----
#
# from timm.data import Mixup
#
# mixup_fn = Mixup(
#     mixup_alpha=0.8,         # Mixup alpha (0 = disabled)
#     cutmix_alpha=1.0,        # CutMix alpha (0 = disabled)
#     cutmix_minmax=None,      # None = use standard square cutout
#     prob=0.5,                # Probability of applying either aug
#     switch_prob=0.5,         # Probability of switching to CutMix vs Mixup
#     mode="batch",            # "batch", "pair", or "elem"
#     label_smoothing=0.1,
#     num_classes=12,
# )
#
# # In training loop:
# for images, labels in train_loader:
#     images, labels = images.to(device), labels.to(device)
#     images, labels = mixup_fn(images, labels)  # labels become soft targets
#     loss = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()


# =============================================================================
# 6. STOCHASTIC WEIGHT AVERAGING (SWA)
# =============================================================================


def setup_swa(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    swa_lr: float = 1e-4,
    anneal_epochs: int = 5,
    anneal_strategy: str = "cos",
) -> tuple:
    """Set up SWA model and scheduler.

    Recommended: start SWA in the last 25% of training.
    For 40 epochs -> swa_start_epoch=30.
    SWA LR should be ~10x lower than peak LR.

    Args:
        model: Base model.
        optimizer: Optimizer in use.
        swa_lr: Constant LR for SWA phase.
        anneal_epochs: Epochs to anneal from current LR to swa_lr.
        anneal_strategy: 'cos' or 'linear'.

    Returns:
        (swa_model, swa_scheduler)
    """
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer,
        swa_lr=swa_lr,
        anneal_epochs=anneal_epochs,
        anneal_strategy=anneal_strategy,
    )
    return swa_model, swa_scheduler


# ---- SWA Training Loop ----
#
# from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
#
# model = YourModel().to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
#
# # Normal scheduler for first phase
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, T_0=10, T_mult=2
# )
#
# # SWA setup
# swa_model = AveragedModel(model)
# swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=5)
# swa_start = 30  # last 25% of 40 epochs
#
# for epoch in range(40):
#     model.train()
#     for images, targets in train_loader:
#         images, targets = images.to(device), targets.to(device)
#         optimizer.zero_grad()
#         loss = criterion(model(images), targets)
#         loss.backward()
#         optimizer.step()
#
#     if epoch >= swa_start:
#         swa_model.update_parameters(model)
#         swa_scheduler.step()
#     else:
#         scheduler.step()
#
# # CRITICAL: Update batch normalization statistics after SWA
# torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
#
# # Save the SWA model
# torch.save(swa_model.module.state_dict(), "model_swa.pth")
#
# # For inference: load weights into base model
# # model.load_state_dict(torch.load("model_swa.pth"))


# =============================================================================
# 7. COSINE ANNEALING WITH WARM RESTARTS
# =============================================================================


def create_cosine_warm_restarts_scheduler(
    optimizer: torch.optim.Optimizer,
    t_0: int = 10,
    t_mult: int = 2,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
    """Create CosineAnnealingWarmRestarts scheduler.

    For 40 epochs:
        T_0=10, T_mult=2 gives restarts at epochs 10, 30 (cycles of 10, 20).
        T_0=10, T_mult=1 gives restarts every 10 epochs.
        T_0=5,  T_mult=2 gives restarts at 5, 15, 35 (cycles of 5, 10, 20).

    Recommended for hackathon: T_0=10, T_mult=2
        -> Cycle 1: epochs 0-9   (fast initial exploration)
        -> Cycle 2: epochs 10-29 (longer refinement)
        -> Remaining: epochs 30-39 (fine-tuning before SWA kicks in)

    Args:
        optimizer: The optimizer.
        t_0: Number of epochs for the first restart.
        t_mult: Multiplier for subsequent restart periods.
        eta_min: Minimum learning rate.
    """
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min
    )


def create_one_cycle_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float = 1e-3,
    total_steps: int = 4000,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """Create OneCycleLR scheduler.

    OneCycleLR vs CosineAnnealingWarmRestarts:
      - OneCycleLR: single cycle, great for known total epochs, often
        gives best single-run results. Preferred when you know exact
        training budget.
      - CosineWarmRestarts: multiple cycles, better for exploration,
        pairs well with SWA (SWA starts after last restart).

    For hackathon (40 epochs, ~100 batches/epoch):
        total_steps = 40 * 100 = 4000

    Args:
        optimizer: The optimizer.
        max_lr: Peak learning rate.
        total_steps: Total training steps (epochs * batches_per_epoch).
        pct_start: Fraction of steps for warm-up.
        div_factor: initial_lr = max_lr / div_factor.
        final_div_factor: final_lr = initial_lr / final_div_factor.
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )


# =============================================================================
# QUICK REFERENCE — Decision Guide
# =============================================================================
#
# CLASSIFICATION (12-class, imbalanced):
#   Loss: FocalLoss(gamma=2.0, alpha=inverse_freq, label_smoothing=0.1)
#   Aug:  timm Mixup (mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5)
#   LR:   CosineAnnealingWarmRestarts(T_0=10, T_mult=2) + SWA last 10 epochs
#   Tip:  Use Focal Loss from epoch 0. No need for warm-up — it is stable.
#         gamma=2.0 is the sweet spot; gamma=3.0 if still underfitting minority.
#
# SEGMENTATION (binary, metric=Mean IoU):
#   Loss: TwoPhaseSegmentationLoss(switch_epoch=20, lovasz_weight=0.5)
#         Phase 1: Dice + BCE (stable convergence)
#         Phase 2: Dice + BCE + Lovasz (IoU optimization)
#   LR:   CosineAnnealingWarmRestarts(T_0=10, T_mult=2) + SWA last 10 epochs
#   Tip:  Do NOT use Lovasz from the start — BCE+Dice gives stable gradients
#         early on. Switch to Lovasz once the model has reasonable predictions.
#
# PERFORMANCE COMPARISON (from literature):
#   CrossEntropy:     Baseline. Works OK but ignores class imbalance.
#   Focal Loss:       +2-5% on minority classes vs CE. Standard for imbalanced.
#   Poly Loss:        Marginal gains over Focal. More complex to tune.
#                     Not worth the complexity for a hackathon.
#   Dice+BCE:         Standard segmentation baseline. Stable.
#   Dice+BCE+Lovasz:  +1-3% Mean IoU over Dice+BCE alone.
#   Focal Tversky:    Best for very small lesions. alpha=0.7, beta=0.3.
#
# CUTMIX vs MIXUP vs CUTOUT:
#   CutMix:  Best for classification — region replacement preserves context.
#   Mixup:   Good complement — global blending. Use both via timm.
#   CutOut:  Weakest of the three. Skip it.
#   For 100x100 images: CutMix+Mixup via timm with alpha=1.0, prob=0.5.
#
# COSINE ANNEALING vs ONECYCLE:
#   CosineWarmRestarts: Better with SWA. Multiple exploration cycles.
#   OneCycleLR:         Better standalone (no SWA). Single aggressive cycle.
#   For hackathon with SWA: use CosineWarmRestarts.
#   Without SWA: use OneCycleLR.
