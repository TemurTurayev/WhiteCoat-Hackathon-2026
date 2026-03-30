"""
Тестирование ensemble + TTA + threshold optimization на validation set.
Запуск: python3 test_ensemble_tta.py
"""
import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

CLASS_NAMES = [
    "Actinic Keratosis", "BCC", "Dermatofibroma", "Hemangioma",
    "Intraepithelial Ca", "Lentigo", "Melanoma", "Nevus",
    "Pyogenic Granuloma", "Seb. Keratosis", "SCC", "Wart"
]


class ClassificationDS(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.imgs = []
        self.labels = []
        root = Path(root)
        for cls_dir in sorted(root.iterdir()):
            if cls_dir.is_dir():
                label = int(cls_dir.name)
                for img_path in sorted(cls_dir.glob("*")):
                    if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        self.imgs.append(str(img_path))
                        self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, self.labels[idx]


class SegmentationDS(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.transform = transform
        self.imgs = []
        self.masks = []
        img_dir = Path(img_dir)
        mask_dir = Path(mask_dir)
        for p in sorted(img_dir.glob("*")):
            if p.suffix.lower() in {".png", ".jpg"}:
                mp = mask_dir / f"{p.stem}.png"
                if not mp.exists():
                    mp = mask_dir / f"{p.stem}.jpg"
                if mp.exists():
                    self.imgs.append(str(p))
                    self.masks.append(str(mp))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], 0)
        mask = (mask > 127).astype(np.float32)
        augmented = self.transform(image=img, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0)


def test_classification():
    print("\n" + "=" * 60)
    print("CLASSIFICATION ENSEMBLE + TTA TEST")
    print("=" * 60)

    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    tta_transforms = [
        A.Compose([A.Resize(224, 224), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1.0), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), A.VerticalFlip(p=1.0), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2()]),
    ]

    train_dir = "data/classification/train"
    ds = ClassificationDS(train_dir, val_tf)

    n = len(ds)
    np.random.seed(42)
    idx = np.random.permutation(n)
    val_idx = idx[int(n * 0.8):]
    val_imgs = [ds.imgs[i] for i in val_idx]
    val_labels = [ds.labels[i] for i in val_idx]
    print(f"Validation: {len(val_labels)} images")

    models_info = [
        ("best_tf_efficientnetv2_s.pth", "tf_efficientnetv2_s"),
        ("best_swin_tiny_patch4_window7_224.pth", "swin_tiny_patch4_window7_224"),
        ("best_convnext_tiny.pth", "convnext_tiny"),
    ]

    models = []
    for fname, arch in models_info:
        path = f"all_models/{fname}"
        if not Path(path).exists():
            print(f"  SKIP {fname} (not found)")
            continue
        m = timm.create_model(arch, pretrained=False, num_classes=12, drop_rate=0.3)
        state = torch.load(path, map_location=DEVICE)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        m.load_state_dict(state, strict=False)
        m.to(DEVICE)
        m.requires_grad_(False)
        models.append((fname, m))
        print(f"  Loaded {fname}")

    print("\n--- Single Model Results ---")
    model_probs = {}
    for fname, model in models:
        all_probs = []
        with torch.no_grad():
            for img_path in val_imgs:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                inp = val_tf(image=img)["image"].unsqueeze(0).to(DEVICE)
                out = torch.softmax(model(inp), dim=1).cpu().numpy()[0]
                all_probs.append(out)
        all_probs = np.array(all_probs)
        preds = all_probs.argmax(axis=1)
        acc = accuracy_score(val_labels, preds)
        f1 = f1_score(val_labels, preds, average="macro")
        print(f"  {fname}: Acc={acc:.4f} F1={f1:.4f}")
        model_probs[fname] = all_probs

    print("\n--- Single Model + 4x TTA ---")
    model_tta_probs = {}
    for fname, model in models:
        all_probs = []
        with torch.no_grad():
            for img_path in val_imgs:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                tta_probs = []
                for ttf in tta_transforms:
                    inp = ttf(image=img)["image"].unsqueeze(0).to(DEVICE)
                    out = torch.softmax(model(inp), dim=1).cpu().numpy()[0]
                    tta_probs.append(out)
                all_probs.append(np.mean(tta_probs, axis=0))
        all_probs = np.array(all_probs)
        preds = all_probs.argmax(axis=1)
        acc = accuracy_score(val_labels, preds)
        f1 = f1_score(val_labels, preds, average="macro")
        print(f"  {fname} +TTA: Acc={acc:.4f} F1={f1:.4f}")
        model_tta_probs[fname] = all_probs

    if len(model_probs) >= 2:
        print("\n--- Ensemble (Soft Voting, no TTA) ---")
        avg_probs = np.mean(list(model_probs.values()), axis=0)
        preds = avg_probs.argmax(axis=1)
        acc = accuracy_score(val_labels, preds)
        f1 = f1_score(val_labels, preds, average="macro")
        print(f"  Ensemble {len(model_probs)} models: Acc={acc:.4f} F1={f1:.4f}")

        print("\n--- Ensemble + TTA ---")
        avg_tta = np.mean(list(model_tta_probs.values()), axis=0)
        preds_tta = avg_tta.argmax(axis=1)
        acc_tta = accuracy_score(val_labels, preds_tta)
        f1_tta = f1_score(val_labels, preds_tta, average="macro")
        print(f"  Ensemble {len(model_tta_probs)} models +TTA: Acc={acc_tta:.4f} F1={f1_tta:.4f}")

        print("\n--- Per-Class Accuracy (Ensemble+TTA) ---")
        for cls in range(12):
            mask = np.array(val_labels) == cls
            if mask.sum() > 0:
                cls_acc = (preds_tta[mask] == cls).mean()
                danger = " ⚠️" if cls in [0, 4, 6, 10] else ""
                print(f"  Class {cls:2d} ({CLASS_NAMES[cls]:20s}): {cls_acc:.3f} ({mask.sum():4d} samples){danger}")

    return model_tta_probs, val_labels


def test_segmentation():
    print("\n" + "=" * 60)
    print("SEGMENTATION TEST + THRESHOLD OPTIMIZATION")
    print("=" * 60)

    val_tf = A.Compose([
        A.Resize(512, 512),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    ds = SegmentationDS(
        "data/Segmentation/validation/images",
        "data/Segmentation/validation/masks",
        val_tf
    )
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"Validation: {len(ds)} images")

    seg_models = []

    lovasz_path = "all_models/best_seg_mega512_lovasz.pth"
    if Path(lovasz_path).exists():
        m = smp.UnetPlusPlus(
            encoder_name="tu-tf_efficientnetv2_s",
            encoder_weights=None, classes=1, activation=None
        )
        m.load_state_dict(torch.load(lovasz_path, map_location=DEVICE), strict=False)
        m.to(DEVICE)
        m.requires_grad_(False)
        seg_models.append(("Lovasz MEGA 512", m))
        print(f"  Loaded Lovasz MEGA 512")

    unet_path = "all_models/best_UnetPlusPlus_efficientnet_b4.pth"
    if Path(unet_path).exists():
        m2 = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None, classes=1, activation=None
        )
        m2.load_state_dict(torch.load(unet_path, map_location=DEVICE), strict=False)
        m2.to(DEVICE)
        m2.requires_grad_(False)
        seg_models.append(("UNet++ EffB4", m2))
        print(f"  Loaded UNet++ EffB4")

    deeplab_path = "all_models/best_DeepLabV3Plus_resnet50.pth"
    if Path(deeplab_path).exists():
        m3 = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None, classes=1, activation=None
        )
        m3.load_state_dict(torch.load(deeplab_path, map_location=DEVICE), strict=False)
        m3.to(DEVICE)
        m3.requires_grad_(False)
        seg_models.append(("DeepLabV3+ R50", m3))
        print(f"  Loaded DeepLabV3+ R50")

    print("\n--- Single Model (thr=0.5) ---")
    for name, model in seg_models:
        all_ious = []
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                probs = torch.sigmoid(model(x))
                pred_bin = (probs > 0.5).float()
                ii = (pred_bin * y).sum((1, 2, 3))
                uu = pred_bin.sum((1, 2, 3)) + y.sum((1, 2, 3)) - ii
                ious = ((ii + 1e-6) / (uu + 1e-6)).cpu().numpy()
                all_ious.extend(ious)
        print(f"  {name}: IoU={np.mean(all_ious):.4f}")

    if seg_models:
        print("\n--- Threshold Optimization ---")
        best_model = seg_models[0][1]
        all_probs_list = []
        all_gts = []
        with torch.no_grad():
            for x, y in dl:
                x = x.to(DEVICE)
                probs = torch.sigmoid(best_model(x)).cpu()
                all_probs_list.append(probs)
                all_gts.append(y)
        all_probs_cat = torch.cat(all_probs_list)
        all_gts_cat = torch.cat(all_gts)

        best_thr = 0.5
        best_iou = 0
        results_table = []
        for thr in np.arange(0.30, 0.70, 0.02):
            pred_bin = (all_probs_cat > thr).float()
            ii = (pred_bin * all_gts_cat).sum((1, 2, 3))
            uu = pred_bin.sum((1, 2, 3)) + all_gts_cat.sum((1, 2, 3)) - ii
            iou = ((ii + 1e-6) / (uu + 1e-6)).mean().item()
            results_table.append((thr, iou))
            if iou > best_iou:
                best_iou = iou
                best_thr = thr
        print(f"  Best: thr={best_thr:.2f} IoU={best_iou:.4f}")
        for thr, iou in results_table:
            marker = " <<<" if abs(thr - best_thr) < 0.01 else ""
            print(f"    thr={thr:.2f}: IoU={iou:.4f}{marker}")

        if len(seg_models) >= 2:
            print("\n--- Ensemble (Avg Logits) + Optimal Threshold ---")
            ens_probs_list = []
            with torch.no_grad():
                for x, y in dl:
                    x = x.to(DEVICE)
                    logits_sum = None
                    for _, model in seg_models:
                        logits = model(x)
                        logits_sum = logits if logits_sum is None else logits_sum + logits
                    avg_probs = torch.sigmoid(logits_sum / len(seg_models)).cpu()
                    ens_probs_list.append(avg_probs)
            ens_probs = torch.cat(ens_probs_list)

            best_ens_thr = 0.5
            best_ens_iou = 0
            for thr in np.arange(0.30, 0.70, 0.02):
                pred_bin = (ens_probs > thr).float()
                ii = (pred_bin * all_gts_cat).sum((1, 2, 3))
                uu = pred_bin.sum((1, 2, 3)) + all_gts_cat.sum((1, 2, 3)) - ii
                iou = ((ii + 1e-6) / (uu + 1e-6)).mean().item()
                if iou > best_ens_iou:
                    best_ens_iou = iou
                    best_ens_thr = thr
            print(f"  Ensemble {len(seg_models)} models: thr={best_ens_thr:.2f} IoU={best_ens_iou:.4f}")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    os.chdir("/Users/temur/Desktop/Claude/Hackathon")
    test_classification()
    test_segmentation()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE!")
    print("=" * 60)
