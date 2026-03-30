"""
Final Submission Generator — OPTIMIZED
Classification: 3-model ensemble + 4x TTA
Segmentation: Lovász MEGA 512 + 4x TTA, threshold=0.51, NO post-processing
"""
import torch
import timm
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MEAN_NP = np.array(MEAN)
STD_NP = np.array(STD)

MODEL_TO_REAL = {0:0, 1:1, 2:10, 3:11, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8, 11:9}


# ========== CLASSIFICATION ==========

def generate_classification():
    print("\n" + "=" * 60)
    print("CLASSIFICATION (3-model ensemble + 4x TTA)")
    print("=" * 60)

    models_info = [
        ("all_models/best_tf_efficientnetv2_s.pth", "tf_efficientnetv2_s"),
        ("all_models/best_swin_tiny_patch4_window7_224.pth", "swin_tiny_patch4_window7_224"),
        ("all_models/best_convnext_tiny.pth", "convnext_tiny"),
    ]

    models = []
    for path, name in models_info:
        if not Path(path).exists():
            continue
        model = timm.create_model(name, pretrained=False, num_classes=12)
        sd = torch.load(path, map_location="cpu")
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd, strict=False)
        model.eval().to(DEVICE)
        models.append(model)
        print(f"  Loaded: {name}")

    tta = [
        A.Compose([A.Resize(224,224), A.Normalize(MEAN,STD), ToTensorV2()]),
        A.Compose([A.Resize(224,224), A.HorizontalFlip(p=1.0), A.Normalize(MEAN,STD), ToTensorV2()]),
        A.Compose([A.Resize(224,224), A.VerticalFlip(p=1.0), A.Normalize(MEAN,STD), ToTensorV2()]),
        A.Compose([A.Resize(224,224), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Normalize(MEAN,STD), ToTensorV2()]),
    ]

    test_dir = Path("data/classification/test")
    test_images = sorted(
        list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )
    print(f"  Test images: {len(test_images)}")

    results = []
    for img_path in tqdm(test_images, desc="Classifying"):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        avg_probs = torch.zeros(12).to(DEVICE)
        count = 0
        with torch.no_grad():
            for model in models:
                for tf in tta:
                    x = tf(image=img)["image"].unsqueeze(0).to(DEVICE)
                    avg_probs += torch.softmax(model(x), dim=1).squeeze()
                    count += 1
        avg_probs /= count
        model_pred = torch.argmax(avg_probs).item()
        real_label = MODEL_TO_REAL[model_pred]
        results.append({"Image_ID": img_path.stem, "Label": real_label})

    df = pd.DataFrame(results)
    out_dir = Path("submission/WhiteCoat.dev")
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_path = out_dir / "WhiteCoat.dev test_ground_truth.xlsx"
    df.to_excel(str(excel_path), index=False)
    print(f"  Saved: {excel_path} ({len(df)} rows)")
    print(f"  Distribution: {df['Label'].value_counts().sort_index().to_dict()}")

    for m in models:
        del m
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return df


# ========== SEGMENTATION ==========

THRESHOLD = 0.51  # Optimized on held-out validation set


def generate_segmentation():
    print("\n" + "=" * 60)
    print("SEGMENTATION (Lovasz MEGA 512 + 4x TTA, thr=0.51, raw)")
    print("=" * 60)

    model = smp.UnetPlusPlus(
        encoder_name="tu-tf_efficientnetv2_s",
        encoder_weights=None, classes=1, activation=None
    )
    sd = torch.load("all_models/best_seg_mega512_lovasz.pth", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    print("  Loaded: Lovasz MEGA 512")

    test_dir = Path("data/Segmentation/testing/images")
    test_images = sorted(
        list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")),
        key=lambda p: p.stem
    )
    print(f"  Test images: {len(test_images)}")
    print(f"  Threshold: {THRESHOLD} | Post-processing: OFF (raw predictions)")

    out_dir = Path("submission/WhiteCoat.dev/WhiteCoat.dev")
    out_dir.mkdir(parents=True, exist_ok=True)

    resize_tf = A.Compose([
        A.Resize(512, 512),
        A.Normalize(MEAN, STD),
        ToTensorV2()
    ])

    for img_path in tqdm(test_images, desc="Segmenting"):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]

        x = resize_tf(image=img)["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Original
            p = torch.sigmoid(model(x)).cpu()
            # HFlip
            p_h = torch.sigmoid(model(torch.flip(x, [3]))).cpu()
            p_h = torch.flip(p_h, [3])
            # VFlip
            p_v = torch.sigmoid(model(torch.flip(x, [2]))).cpu()
            p_v = torch.flip(p_v, [2])
            # HVFlip
            p_hv = torch.sigmoid(model(torch.flip(x, [2, 3]))).cpu()
            p_hv = torch.flip(p_hv, [2, 3])

        avg_prob = ((p + p_h + p_v + p_hv) / 4.0).squeeze().numpy()

        # Resize back to original size
        prob_orig = cv2.resize(avg_prob, (ow, oh), interpolation=cv2.INTER_LINEAR)

        # Raw thresholding — no post-processing (validated: post-proc hurts IoU)
        mask = ((prob_orig > THRESHOLD).astype(np.uint8)) * 255
        cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), mask)

    saved = list(out_dir.glob("*.png"))
    print(f"  Saved: {len(saved)} masks")

    sample = cv2.imread(str(saved[0]), 0)
    print(f"  Sample values: {np.unique(sample)} | Size: {sample.shape}")

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# ========== MAIN ==========

if __name__ == "__main__":
    cls_df = generate_classification()
    generate_segmentation()

    print("\n" + "=" * 60)
    print("SUBMISSION VERIFICATION")
    print("=" * 60)
    sub = Path("submission/WhiteCoat.dev")
    excel = sub / "WhiteCoat.dev test_ground_truth.xlsx"
    masks = sub / "WhiteCoat.dev"
    mc = len(list(masks.glob("*.png"))) if masks.exists() else 0
    print(f"  Excel: {'OK' if excel.exists() else 'MISSING'} ({len(cls_df)} rows)")
    print(f"  Masks: {mc}/200")
    status = "READY" if excel.exists() and mc == 200 else "INCOMPLETE"
    print(f"\n  SUBMISSION {status}!")
