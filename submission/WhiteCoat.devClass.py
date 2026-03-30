"""
WhiteCoat.dev -- Classification Inference
Loads trained ensemble model (WhiteCoat.devClassModel) and generates Excel predictions.

Usage:
    python WhiteCoat.devClass.py --test_dir <path_to_test_images> --model_dir WhiteCoat.devClassModel

Output: WhiteCoat.dev test_ground_truth.xlsx
"""
import argparse
import torch
import timm
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Mapping from model output classes to real label IDs
MODEL_TO_REAL = {0: 0, 1: 1, 2: 10, 3: 11, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# 4x Test-Time Augmentation
TTA_TRANSFORMS = [
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(MEAN, STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.VerticalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()]),
]

# 3-model ensemble architectures
MODEL_CONFIGS = [
    ("tf_efficientnetv2_s", "best_tf_efficientnetv2_s.pth"),
    ("swin_tiny_patch4_window7_224", "best_swin_tiny_patch4_window7_224.pth"),
    ("convnext_tiny", "best_convnext_tiny.pth"),
]


def load_model(arch_name, weights_path, device):
    """Load a timm classification model with saved weights."""
    model = timm.create_model(arch_name, pretrained=False, num_classes=12)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train(False)
    return model


def predict_with_tta(model, image, device):
    """Run 4x TTA and return averaged softmax probabilities."""
    avg_probs = torch.zeros(12).to(device)
    with torch.no_grad():
        for transform in TTA_TRANSFORMS:
            tensor = transform(image=image)["image"].unsqueeze(0).to(device)
            avg_probs += torch.softmax(model(tensor), dim=1).squeeze()
    return avg_probs / len(TTA_TRANSFORMS)


def main():
    parser = argparse.ArgumentParser(description="WhiteCoat.dev Classification")
    parser.add_argument("--test_dir", required=True, help="Path to test images folder")
    parser.add_argument("--model_dir", default="WhiteCoat.devClassModel", help="Path to WhiteCoat.devClassModel folder")
    parser.add_argument("--output", default="WhiteCoat.dev test_ground_truth.xlsx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ensemble models from WhiteCoat.devClassModel/
    models = []
    for arch, filename in MODEL_CONFIGS:
        weights_path = Path(args.model_dir) / filename
        if weights_path.exists():
            model = load_model(arch, str(weights_path), device)
            models.append(model)
            print(f"Loaded: {arch}")
        else:
            print(f"WARNING: {weights_path} not found, skipping")

    if not models:
        raise RuntimeError("No models loaded! Check model_dir path.")

    n_preds = len(models) * len(TTA_TRANSFORMS)
    print(f"Ensemble: {len(models)} models x {len(TTA_TRANSFORMS)} TTA = {n_preds} predictions per image")

    # Find test images
    test_dir = Path(args.test_dir)
    images = sorted(
        list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")),
        key=lambda p: p.stem,
    )
    print(f"Test images: {len(images)}")

    # Predict
    results = []
    for i, img_path in enumerate(images):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        ensemble_probs = torch.zeros(12).to(device)

        for model in models:
            ensemble_probs += predict_with_tta(model, img, device)

        ensemble_probs /= len(models)
        predicted_class = torch.argmax(ensemble_probs).item()
        predicted_label = MODEL_TO_REAL[predicted_class]
        results.append({"Image_ID": img_path.stem, "Label": predicted_label})

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(images)}")

    # Save Excel
    df = pd.DataFrame(results)
    df.to_excel(args.output, index=False)
    print(f"\nSaved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
