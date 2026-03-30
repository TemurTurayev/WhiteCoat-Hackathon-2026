"""
Classification Ensemble Inference — WhiteCoat.dev (#37)
3-model ensemble (EfficientNetV2-S + Swin-Tiny + ConvNeXt-Tiny) x 5 folds x 4 TTA
"""
import torch
import timm
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_TO_REAL = {0: 0, 1: 1, 2: 10, 3: 11, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE = 224

TEST_DIR = "data/classification/test"
OUTPUT = "submission/WhiteCoat.dev test_ground_truth.xlsx"
KFOLD_DIR = "all_models/kfold"

MODELS = [
    {"name": "tf_efficientnetv2_s", "best": "all_models/best_tf_efficientnetv2_s.pth"},
    {"name": "swin_tiny_patch4_window7_224", "best": "all_models/best_swin_tiny_patch4_window7_224.pth"},
    {"name": "convnext_tiny", "best": "all_models/best_convnext_tiny.pth"},
]

TTA_TRANSFORMS = [
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(MEAN, STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.VerticalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
]


def load_model(model_name, weights_path, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=12)
    sd = torch.load(weights_path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device)
    return model


def predict_single(model, img, device):
    avg = torch.zeros(12).to(device)
    with torch.no_grad():
        for tf in TTA_TRANSFORMS:
            x = tf(image=img)["image"].unsqueeze(0).to(device)
            avg += torch.softmax(model(x), dim=1).squeeze()
    avg /= len(TTA_TRANSFORMS)
    return avg


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all models
    all_models = []

    # Best single models
    for m in MODELS:
        if Path(m["best"]).exists():
            model = load_model(m["name"], m["best"], device)
            model.eval()
            all_models.append((m["name"], model))
            print(f"Loaded best: {m['name']}")

    # K-fold models (load only fold 0 for speed on CPU/MPS)
    for m in MODELS:
        kfold_path = Path(KFOLD_DIR) / f"kfold_{m['name']}_fold0.pth"
        if kfold_path.exists():
            model = load_model(m["name"], str(kfold_path), device)
            model.eval()
            all_models.append((f"{m['name']}_fold0", model))
            print(f"Loaded kfold fold0: {m['name']}")

    print(f"\nTotal models in ensemble: {len(all_models)}")

    # Get test images
    test_dir = Path(TEST_DIR)
    images = sorted(
        list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")),
        key=lambda p: p.stem
    )
    print(f"Test images: {len(images)}")

    # Predict
    results = []
    for i, img_path in enumerate(images):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        ensemble_avg = torch.zeros(12).to(device)

        for name, model in all_models:
            ensemble_avg += predict_single(model, img, device)

        ensemble_avg /= len(all_models)
        pred_class = torch.argmax(ensemble_avg).item()
        pred_label = MODEL_TO_REAL[pred_class]
        results.append({"Image_ID": img_path.stem, "Label": pred_label})

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(images)} done")

    # Save
    df = pd.DataFrame(results)
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT, index=False)
    print(f"\nSaved {len(df)} predictions to {OUTPUT}")
    print(f"Label distribution:\n{df['Label'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
