"""
Threshold + ensemble optimization for ISIC Soup model.
Tests on held-out 200 validation images (last 200).
"""
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(path):
    m = smp.UnetPlusPlus(
        encoder_name="tu-tf_efficientnetv2_s",
        encoder_weights=None, classes=1, activation=None
    )
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    m.requires_grad_(False)
    return m.to(DEVICE)


def compute_iou_batch(pred_prob, gt, threshold):
    pred_bin = (pred_prob > threshold).float()
    ii = (pred_bin * gt).sum((1, 2, 3))
    uu = pred_bin.sum((1, 2, 3)) + gt.sum((1, 2, 3)) - ii
    return ((ii + 1e-6) / (uu + 1e-6)).cpu().numpy()


def get_held_out_data():
    """Get last 200 validation images (held-out set)."""
    img_dir = Path("data/Segmentation/validation/images")
    mask_dir = Path("data/Segmentation/validation/masks")

    all_imgs = sorted(img_dir.glob("*"))
    all_imgs = [p for p in all_imgs if p.suffix.lower() in {".png", ".jpg"}]

    # Last 200 = held-out
    held_out = all_imgs[200:]
    print(f"Held-out images: {len(held_out)}")

    tf = A.Compose([
        A.Resize(512, 512),
        A.Normalize(MEAN, STD),
        ToTensorV2()
    ])

    images = []
    masks = []
    for p in held_out:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        mp = mask_dir / f"{p.stem}.png"
        if not mp.exists():
            mp = mask_dir / f"{p.stem}.jpg"
        mask = cv2.imread(str(mp), 0)
        mask = (mask > 127).astype(np.float32)

        aug = tf(image=img, mask=mask)
        images.append(aug["image"])
        m = aug["mask"]
        if isinstance(m, np.ndarray):
            m = torch.from_numpy(m)
        masks.append(m.unsqueeze(0))

    return torch.stack(images), torch.stack(masks)


def predict_with_tta(model, images, batch_size=4):
    """Predict with 4x TTA."""
    n = len(images)
    all_probs = torch.zeros(n, 1, 512, 512)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end].to(DEVICE)

        with torch.no_grad():
            # Original
            p = torch.sigmoid(model(batch)).cpu()

            # HFlip
            p_h = torch.sigmoid(model(torch.flip(batch, [3]))).cpu()
            p_h = torch.flip(p_h, [3])

            # VFlip
            p_v = torch.sigmoid(model(torch.flip(batch, [2]))).cpu()
            p_v = torch.flip(p_v, [2])

            # HVFlip
            p_hv = torch.sigmoid(model(torch.flip(batch, [2, 3]))).cpu()
            p_hv = torch.flip(p_hv, [2, 3])

            all_probs[start:end] = (p + p_h + p_v + p_hv) / 4.0

    return all_probs


def main():
    images, masks = get_held_out_data()
    print(f"Data loaded: {images.shape}, {masks.shape}")

    # Load ISIC Soup
    print("\n--- ISIC Soup (5-fold) ---")
    isic_soup = load_model("all_models/isic_soup_5fold.pth")

    # Load Lovasz MEGA
    print("--- Lovasz MEGA 512 ---")
    lovasz = load_model("all_models/best_seg_mega512_lovasz.pth")

    # === Single models, no TTA ===
    print("\n" + "=" * 60)
    print("SINGLE MODEL, NO TTA, threshold sweep")
    print("=" * 60)

    for name, model in [("ISIC Soup", isic_soup), ("Lovasz MEGA", lovasz)]:
        probs_list = []
        for start in range(0, len(images), 4):
            end = min(start + 4, len(images))
            batch = images[start:end].to(DEVICE)
            with torch.no_grad():
                p = torch.sigmoid(model(batch)).cpu()
            probs_list.append(p)
        probs = torch.cat(probs_list)

        best_thr, best_iou = 0.5, 0
        for thr in np.arange(0.25, 0.65, 0.02):
            ious = compute_iou_batch(probs, masks, thr)
            mean_iou = ious.mean()
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_thr = thr
        print(f"  {name} (no TTA): best thr={best_thr:.2f}, IoU={best_iou:.4f}")

        # Detailed around best
        for thr in np.arange(best_thr - 0.06, best_thr + 0.07, 0.01):
            ious = compute_iou_batch(probs, masks, thr)
            marker = " <<<" if abs(thr - best_thr) < 0.005 else ""
            print(f"    thr={thr:.2f}: IoU={ious.mean():.4f}{marker}")

    # === With 4x TTA ===
    print("\n" + "=" * 60)
    print("WITH 4x TTA, threshold sweep")
    print("=" * 60)

    isic_tta = predict_with_tta(isic_soup, images)
    lovasz_tta = predict_with_tta(lovasz, images)

    for name, probs in [("ISIC Soup +TTA", isic_tta), ("Lovasz MEGA +TTA", lovasz_tta)]:
        best_thr, best_iou = 0.5, 0
        for thr in np.arange(0.25, 0.65, 0.02):
            ious = compute_iou_batch(probs, masks, thr)
            mean_iou = ious.mean()
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_thr = thr
        print(f"  {name}: best thr={best_thr:.2f}, IoU={best_iou:.4f}")

        for thr in np.arange(best_thr - 0.06, best_thr + 0.07, 0.01):
            ious = compute_iou_batch(probs, masks, thr)
            marker = " <<<" if abs(thr - best_thr) < 0.005 else ""
            print(f"    thr={thr:.2f}: IoU={ious.mean():.4f}{marker}")

    # === Ensemble: ISIC Soup + Lovasz ===
    print("\n" + "=" * 60)
    print("ENSEMBLE (ISIC Soup + Lovasz) WITH TTA")
    print("=" * 60)

    for w_isic in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_lov = 1.0 - w_isic
        ens = isic_tta * w_isic + lovasz_tta * w_lov

        best_thr, best_iou = 0.5, 0
        for thr in np.arange(0.25, 0.65, 0.02):
            ious = compute_iou_batch(ens, masks, thr)
            mean_iou = ious.mean()
            if mean_iou > best_iou:
                best_iou = mean_iou
                best_thr = thr
        print(f"  w_isic={w_isic:.1f} w_lov={w_lov:.1f}: best thr={best_thr:.2f}, IoU={best_iou:.4f}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    import os
    os.chdir("/Users/temur/Desktop/Claude/Hackathon")
    main()
