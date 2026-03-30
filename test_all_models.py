"""
Comprehensive model testing — Classification + Segmentation.
Runs all models, generates results table.
"""
import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import time
import json
from pathlib import Path
from collections import defaultdict

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

CLASS_NAMES = {
    0: "Actinic Keratosis", 1: "BCC", 2: "Dermatofibroma",
    3: "Hemangioma", 4: "Intraepithelial Ca.", 5: "Lentigo",
    6: "Melanoma", 7: "Melanocytic Nevus", 8: "Pyogenic Granuloma",
    9: "Seb. Keratosis", 10: "SCC", 11: "Wart"
}

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def preprocess_cls(img_path, size=224):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)


def preprocess_seg(img_path, size=512):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0), orig_h, orig_w


def load_cls_model(path, model_name, num_classes=12):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    sd = torch.load(str(path), map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    return model


def load_seg_model(path, arch="UnetPlusPlus", encoder="tu-tf_efficientnetv2_s"):
    model = getattr(smp, arch)(encoder_name=encoder, encoder_weights=None, classes=1, activation=None)
    sd = torch.load(str(path), map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    return model


def compute_iou(pred, gt):
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    if union < 1e-6:
        return 1.0 if gt.sum() < 1e-6 else 0.0
    return float(intersection / union)


# ==================== CLASSIFICATION TESTS ====================
print("\n" + "=" * 70)
print("CLASSIFICATION MODEL TESTING")
print("=" * 70)

cls_models_info = [
    ("best_tf_efficientnetv2_s.pth", "tf_efficientnetv2_s"),
    ("best_convnext_tiny.pth", "convnext_tiny"),
    ("best_swin_tiny_patch4_window7_224.pth", "swin_tiny_patch4_window7_224"),
    ("best_efficientnet_b4.pth", "efficientnet_b4"),
    ("best_resnet50.pth", "resnet50"),
]

# Collect val images (last 20% of each class)
val_dir = Path("data/classification/train")
val_images = []
val_labels = []
for class_dir in sorted(val_dir.iterdir()):
    if class_dir.is_dir():
        cls_id = int(class_dir.name)
        imgs = sorted(list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")))
        split = int(len(imgs) * 0.8)
        for img in imgs[split:]:
            val_images.append(str(img))
            val_labels.append(cls_id)

print(f"Validation images: {len(val_images)}")

cls_results = {}

for model_file, model_name in cls_models_info:
    model_path = Path("all_models") / model_file
    if not model_path.exists():
        print(f"  SKIP: {model_file} not found")
        continue

    print(f"\nTesting: {model_name}...")
    try:
        model = load_cls_model(model_path, model_name)
        model.eval()
    except Exception as e:
        print(f"  ERROR loading: {e}")
        continue

    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    inference_times = []

    with torch.no_grad():
        for img_path, label in zip(val_images, val_labels):
            try:
                x = preprocess_cls(img_path).to(DEVICE)
                t0 = time.time()
                out = model(x)
                inference_times.append(time.time() - t0)
                pred = torch.argmax(out, dim=1).item()
                if pred == label:
                    correct += 1
                    per_class_correct[label] += 1
                total += 1
                per_class_total[label] += 1
            except Exception:
                total += 1
                per_class_total[label] += 1

    acc = correct / total if total > 0 else 0
    avg_time = np.mean(inference_times) * 1000 if inference_times else 0

    per_class_acc = {}
    for k in range(12):
        if per_class_total[k] > 0:
            per_class_acc[k] = per_class_correct[k] / per_class_total[k]
        else:
            per_class_acc[k] = 0.0

    cls_results[model_name] = {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_ms": avg_time,
        "per_class": per_class_acc
    }

    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    print(f"  Avg inference: {avg_time:.1f}ms")

    worst = sorted(range(12), key=lambda c: per_class_acc.get(c, 0))
    print(f"  Worst 3: {[(CLASS_NAMES[c], f'{per_class_acc[c]:.2%}') for c in worst[:3]]}")

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ==================== ENSEMBLE ====================
print("\n" + "=" * 70)
print("ENSEMBLE CLASSIFICATION TEST")
print("=" * 70)

ensemble_models = []
for model_file, model_name in cls_models_info:
    model_path = Path("all_models") / model_file
    if model_path.exists():
        try:
            m = load_cls_model(model_path, model_name)
            m.eval()
            ensemble_models.append((model_name, m))
        except Exception:
            pass

if ensemble_models:
    print(f"Ensemble of {len(ensemble_models)} models")
    correct = 0
    total = 0

    with torch.no_grad():
        for img_path, label in zip(val_images, val_labels):
            try:
                x = preprocess_cls(img_path).to(DEVICE)
                avg_probs = torch.zeros(12).to(DEVICE)
                for _, m in ensemble_models:
                    out = m(x)
                    avg_probs += torch.softmax(out, dim=1).squeeze()
                avg_probs /= len(ensemble_models)
                pred = torch.argmax(avg_probs).item()
                if pred == label:
                    correct += 1
                total += 1
            except Exception:
                total += 1

    ens_acc = correct / total if total > 0 else 0
    cls_results["ENSEMBLE"] = {"accuracy": ens_acc, "correct": correct, "total": total, "avg_ms": 0}
    print(f"  Ensemble Accuracy: {ens_acc:.4f} ({correct}/{total})")

    for _, m in ensemble_models:
        del m
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ==================== SEGMENTATION ====================
print("\n" + "=" * 70)
print("SEGMENTATION MODEL TESTING")
print("=" * 70)

seg_models_info = [
    ("best_seg_mega512_lovasz.pth", "UnetPlusPlus", "tu-tf_efficientnetv2_s", 512),
    ("best_UnetPlusPlus_efficientnet_b4.pth", "UnetPlusPlus", "efficientnet-b4", 256),
    ("best_DeepLabV3Plus_resnet50.pth", "DeepLabV3Plus", "resnet50", 256),
    ("best_Unet_resnet34.pth", "Unet", "resnet34", 256),
]

val_img_dir = Path("data/Segmentation/validation/images")
val_mask_dir = Path("data/Segmentation/validation/masks")
seg_val_imgs = sorted(list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpg")))
print(f"Segmentation val images: {len(seg_val_imgs)}")

seg_results = {}

for model_file, arch, encoder, img_size in seg_models_info:
    model_path = Path("all_models") / model_file
    if not model_path.exists():
        print(f"  SKIP: {model_file} not found")
        continue

    print(f"\nTesting: {model_file} ({arch}/{encoder}, {img_size}px)...")
    try:
        model = load_seg_model(model_path, arch, encoder)
        model.eval()
    except Exception as e:
        print(f"  ERROR loading: {e}")
        continue

    ious = []
    dices = []
    inference_times = []
    threshold_ious = {t: [] for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]}

    with torch.no_grad():
        for img_path in seg_val_imgs:
            mask_path = val_mask_dir / img_path.name
            if not mask_path.exists():
                mask_path = val_mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue

            try:
                x, oh, ow = preprocess_seg(img_path, img_size)
                x = x.to(DEVICE)

                t0 = time.time()
                logits = model(x)
                inference_times.append(time.time() - t0)

                prob = torch.sigmoid(logits).cpu().numpy().squeeze()
                prob_resized = cv2.resize(prob, (ow, oh))

                gt = cv2.imread(str(mask_path), 0)
                gt = (gt > 127).astype(np.float32)

                pred = (prob_resized > 0.5).astype(np.float32)
                iou = compute_iou(pred, gt)
                ious.append(iou)

                intersection = (pred * gt).sum()
                dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-6)
                dices.append(dice)

                for t in threshold_ious:
                    p = (prob_resized > t).astype(np.float32)
                    threshold_ious[t].append(compute_iou(p, gt))

            except Exception:
                pass

    mean_iou = np.mean(ious) if ious else 0
    mean_dice = np.mean(dices) if dices else 0
    avg_time = np.mean(inference_times) * 1000 if inference_times else 0

    best_t = 0.5
    best_t_iou = mean_iou
    for t, t_ious in threshold_ious.items():
        t_mean = np.mean(t_ious) if t_ious else 0
        if t_mean > best_t_iou:
            best_t_iou = t_mean
            best_t = t

    seg_results[model_file] = {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "best_threshold": best_t,
        "best_threshold_iou": best_t_iou,
        "avg_ms": avg_time,
        "num_images": len(ious),
    }

    print(f"  Mean IoU (t=0.5): {mean_iou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  Best threshold: {best_t} -> IoU={best_t_iou:.4f}")
    print(f"  Avg inference: {avg_time:.1f}ms")

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ==================== SEG ENSEMBLE ====================
print("\n" + "=" * 70)
print("SEGMENTATION ENSEMBLE TEST")
print("=" * 70)

seg_ensemble = []
for model_file, arch, encoder, img_size in seg_models_info:
    model_path = Path("all_models") / model_file
    if model_path.exists():
        try:
            m = load_seg_model(model_path, arch, encoder)
            m.eval()
            seg_ensemble.append((model_file, m, img_size))
        except Exception:
            pass

if len(seg_ensemble) >= 2:
    print(f"Ensemble of {len(seg_ensemble)} seg models")
    ens_ious = []
    threshold_ious_ens = {t: [] for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]}

    with torch.no_grad():
        for img_path in seg_val_imgs:
            mask_path = val_mask_dir / img_path.name
            if not mask_path.exists():
                mask_path = val_mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue

            try:
                gt = cv2.imread(str(mask_path), 0)
                gt = (gt > 127).astype(np.float32)
                oh, ow = gt.shape

                avg_prob = np.zeros((oh, ow), dtype=np.float32)
                for name, m, sz in seg_ensemble:
                    x, _, _ = preprocess_seg(img_path, sz)
                    x = x.to(DEVICE)
                    logits = m(x)
                    prob = torch.sigmoid(logits).cpu().numpy().squeeze()
                    prob_resized = cv2.resize(prob, (ow, oh))
                    avg_prob += prob_resized
                avg_prob /= len(seg_ensemble)

                pred = (avg_prob > 0.5).astype(np.float32)
                iou = compute_iou(pred, gt)
                ens_ious.append(iou)

                for t in threshold_ious_ens:
                    p = (avg_prob > t).astype(np.float32)
                    threshold_ious_ens[t].append(compute_iou(p, gt))
            except Exception:
                pass

    mean_ens_iou = np.mean(ens_ious) if ens_ious else 0
    best_t_ens = 0.5
    best_t_ens_iou = mean_ens_iou
    for t, v in threshold_ious_ens.items():
        m_val = np.mean(v) if v else 0
        if m_val > best_t_ens_iou:
            best_t_ens_iou = m_val
            best_t_ens = t

    seg_results["ENSEMBLE"] = {
        "mean_iou": mean_ens_iou,
        "best_threshold": best_t_ens,
        "best_threshold_iou": best_t_ens_iou,
    }

    print(f"  Ensemble IoU (t=0.5): {mean_ens_iou:.4f}")
    print(f"  Best threshold: {best_t_ens} -> IoU={best_t_ens_iou:.4f}")

    for _, m, _ in seg_ensemble:
        del m


# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)

print("\n### Classification ###")
print(f"{'Model':<35} {'Accuracy':>10}")
print("-" * 50)
for name, res in sorted(cls_results.items(), key=lambda x: -x[1]["accuracy"]):
    marker = " <-- BEST" if name == max(cls_results, key=lambda k: cls_results[k]["accuracy"]) else ""
    print(f"{name:<35} {res['accuracy']:>9.2%}{marker}")

print("\n### Segmentation ###")
print(f"{'Model':<42} {'IoU@0.5':>8} {'Best t':>7} {'IoU@best':>10}")
print("-" * 70)
for name, res in sorted(seg_results.items(), key=lambda x: -x[1].get("best_threshold_iou", 0)):
    marker = " <-- BEST" if name == max(seg_results, key=lambda k: seg_results[k].get("best_threshold_iou", 0)) else ""
    print(f"{name:<42} {res['mean_iou']:>7.4f} {res['best_threshold']:>7.2f} {res['best_threshold_iou']:>9.4f}{marker}")

# Save
Path("outputs").mkdir(exist_ok=True)
with open("outputs/test_results.json", "w") as f:
    json.dump({"classification": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in cls_results.items()},
               "segmentation": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in seg_results.items()}},
              f, indent=2)

print("\nResults saved to outputs/test_results.json")
