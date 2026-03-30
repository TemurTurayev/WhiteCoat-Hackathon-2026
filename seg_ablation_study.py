"""
Ablation Study: тестируем каждую технику ОТДЕЛЬНО на held-out 200 val images.
Baseline: Lovász MEGA 512, 4x TTA, threshold 0.45, IoU=0.8952
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import time
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model_path = "all_models/best_seg_mega512_lovasz.pth"
model = smp.UnetPlusPlus(
    encoder_name="tu-tf_efficientnetv2_s",
    encoder_weights=None,
    classes=1,
    activation=None,
).to(device)
state = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
print(f"Loaded: {model_path}")

all_pairs = []
for split in ["training", "validation"]:
    img_dir = Path(f"data/Segmentation/{split}/images")
    mask_dir = Path(f"data/Segmentation/{split}/masks")
    for p in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        mp = mask_dir / f"{p.stem}.png"
        if not mp.exists():
            mp = mask_dir / f"{p.stem}.jpg"
        if mp.exists():
            all_pairs.append((str(p), str(mp)))

random.shuffle(all_pairs)
val_pairs = all_pairs[-200:]
print(f"Held-out validation: {len(val_pairs)} images")


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def load_mask(path):
    m = cv2.imread(path, 0)
    return (m > 127).astype(np.float32)


def preprocess(img, size=512):
    tf = A.Compose([
        A.Resize(size, size),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return tf(image=img)["image"].unsqueeze(0).to(device)


def compute_iou(pred, gt):
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union < 1e-6:
        return 1.0 if gt.sum() < 1e-6 else 0.0
    return float(intersection / union)


def get_logits_8x(img, size=512):
    h, w = img.shape[:2]
    logits_sum = np.zeros((h, w), dtype=np.float64)
    count = 0
    with torch.no_grad():
        for k in range(4):
            for flip in [False, True]:
                aug = np.rot90(img, k).copy()
                if flip:
                    aug = np.flip(aug, axis=1).copy()
                inp = preprocess(aug, size)
                logit = model(inp).squeeze().cpu().numpy()
                if flip:
                    logit = np.flip(logit, axis=1).copy()
                logit = np.rot90(logit, -k).copy()
                logit = cv2.resize(logit, (w, h), interpolation=cv2.INTER_LINEAR)
                logits_sum += logit
                count += 1
    return logits_sum / count


def get_logits_4x(img, size=512):
    h, w = img.shape[:2]
    logits_sum = np.zeros((h, w), dtype=np.float64)
    with torch.no_grad():
        for do_hflip in [False, True]:
            for do_vflip in [False, True]:
                aug = img.copy()
                if do_hflip:
                    aug = np.flip(aug, axis=1).copy()
                if do_vflip:
                    aug = np.flip(aug, axis=0).copy()
                inp = preprocess(aug, size)
                logit = model(inp).squeeze().cpu().numpy()
                if do_vflip:
                    logit = np.flip(logit, axis=0).copy()
                if do_hflip:
                    logit = np.flip(logit, axis=1).copy()
                logit = cv2.resize(logit, (w, h), interpolation=cv2.INTER_LINEAR)
                logits_sum += logit
    return logits_sum / 4.0


def get_logits_multiscale_8x(img, sizes=None):
    if sizes is None:
        sizes = [384, 512, 640]
    h, w = img.shape[:2]
    logits_sum = np.zeros((h, w), dtype=np.float64)
    count = 0
    with torch.no_grad():
        for size in sizes:
            for k in range(4):
                for flip in [False, True]:
                    aug = np.rot90(img, k).copy()
                    if flip:
                        aug = np.flip(aug, axis=1).copy()
                    inp = preprocess(aug, size)
                    logit = model(inp).squeeze().cpu().numpy()
                    if flip:
                        logit = np.flip(logit, axis=1).copy()
                    logit = np.rot90(logit, -k).copy()
                    logit = cv2.resize(logit, (w, h), interpolation=cv2.INTER_LINEAR)
                    logits_sum += logit
                    count += 1
    return logits_sum / count


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


print("\n" + "=" * 60)
print("EXPERIMENT 1: Baseline 4x TTA, t=0.45")
print("=" * 60)
ious_baseline = []
start = time.time()
for i, (ip, mp) in enumerate(val_pairs):
    img = load_image(ip)
    gt = load_mask(mp)
    logits = get_logits_4x(img, 512)
    pred = (sigmoid(logits) > 0.45).astype(np.float32)
    ious_baseline.append(compute_iou(pred, gt))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/200: IoU={np.mean(ious_baseline):.4f}")
print(f"  BASELINE: IoU={np.mean(ious_baseline):.4f} ({time.time()-start:.0f}s)")

print("\n" + "=" * 60)
print("EXPERIMENT 2: 8x TTA, t=0.45")
print("=" * 60)
ious_8x = []
logits_8x_all = []
gt_all = []
start = time.time()
for i, (ip, mp) in enumerate(val_pairs):
    img = load_image(ip)
    gt = load_mask(mp)
    logits = get_logits_8x(img, 512)
    logits_8x_all.append(logits)
    gt_all.append(gt)
    pred = (sigmoid(logits) > 0.45).astype(np.float32)
    ious_8x.append(compute_iou(pred, gt))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/200: IoU={np.mean(ious_8x):.4f}")
print(f"  8x TTA: IoU={np.mean(ious_8x):.4f} ({time.time()-start:.0f}s)")

print("\n" + "=" * 60)
print("EXPERIMENT 3: Threshold Sweep on 8x TTA")
print("=" * 60)
best_t, best_iou = 0.45, 0.0
for t in np.arange(0.30, 0.65, 0.02):
    ious = []
    for logits, gt in zip(logits_8x_all, gt_all):
        pred = (sigmoid(logits) > t).astype(np.float32)
        ious.append(compute_iou(pred, gt))
    mean_iou = np.mean(ious)
    marker = " <<<" if mean_iou > best_iou else ""
    print(f"  t={t:.2f}: IoU={mean_iou:.4f}{marker}")
    if mean_iou > best_iou:
        best_iou = mean_iou
        best_t = t
print(f"  BEST 8x: t={best_t:.2f}, IoU={best_iou:.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT 4: Multi-scale (384+512+640) + 8x TTA")
print("=" * 60)
ious_ms = []
logits_ms_all = []
start = time.time()
for i, (ip, mp) in enumerate(val_pairs):
    img = load_image(ip)
    gt = load_mask(mp)
    logits = get_logits_multiscale_8x(img, [384, 512, 640])
    logits_ms_all.append(logits)
    pred = (sigmoid(logits) > best_t).astype(np.float32)
    ious_ms.append(compute_iou(pred, gt))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/200: IoU={np.mean(ious_ms):.4f} ({time.time()-start:.0f}s)")
print(f"  MULTI-SCALE 8x: IoU={np.mean(ious_ms):.4f} ({time.time()-start:.0f}s)")

print("\nThreshold sweep on multi-scale:")
best_t_ms, best_iou_ms = best_t, 0.0
for t in np.arange(0.30, 0.65, 0.02):
    ious = []
    for logits, gt in zip(logits_ms_all, gt_all):
        pred = (sigmoid(logits) > t).astype(np.float32)
        ious.append(compute_iou(pred, gt))
    mean_iou = np.mean(ious)
    marker = " <<<" if mean_iou > best_iou_ms else ""
    print(f"  t={t:.2f}: IoU={mean_iou:.4f}{marker}")
    if mean_iou > best_iou_ms:
        best_iou_ms = mean_iou
        best_t_ms = t
print(f"  BEST MS: t={best_t_ms:.2f}, IoU={best_iou_ms:.4f}")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
baseline = np.mean(ious_baseline)
print(f"  1. Baseline (4x TTA, t=0.45):          IoU={baseline:.4f}")
print(f"  2. 8x TTA (t=0.45):                    IoU={np.mean(ious_8x):.4f} ({np.mean(ious_8x)-baseline:+.4f})")
print(f"  3. 8x TTA (t={best_t:.2f}):                    IoU={best_iou:.4f} ({best_iou-baseline:+.4f})")
print(f"  4. Multi-scale+8x (t={best_t_ms:.2f}):          IoU={best_iou_ms:.4f} ({best_iou_ms-baseline:+.4f})")
print(f"\nBest: Multi-scale+8x, t={best_t_ms:.2f}, IoU={best_iou_ms:.4f}")
