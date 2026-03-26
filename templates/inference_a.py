"""
Inference Script — Problem A (Classification)
Team: WhiteCoat.dev (#37)

Usage:
    python classify.py --model_path checkpoints/best_classification.pth \
                       --test_dir data/classification/test/ \
                       --output_path "WhiteCoat.dev test_ground_truth.xlsx"

Output format (Excel):
    Image_ID | Label
    20111    | 2
    20112    | 7
    ...

Requirements:
- Image_ID = filename without extension
- Label = one of 0-11
- All 1276 test images must be included
- Inference only, no retraining
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEAM_NAME = "WhiteCoat.dev"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dcm"}


def load_model(model_path: str, device: torch.device) -> tuple:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = timm.create_model(
        cfg["model_name"],
        pretrained=False,
        num_classes=cfg["num_classes"],
        drop_rate=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    logger.info(
        f"Model loaded: {cfg['model_name']} | "
        f"Classes: {cfg['num_classes']} | "
        f"Val F1: {checkpoint.get('val_f1', 'N/A')}"
    )
    return model, cfg


def get_test_transform(image_size: int = 384):
    """Transform for test images (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def predict_single(
    model: torch.nn.Module,
    image_path: str,
    transform,
    device: torch.device,
) -> tuple[int, float, list[float]]:
    """Predict single image. Returns (class, confidence, probabilities)."""
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Cannot read image: {image_path}, using fallback")
        return 0, 0.0, [0.0]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return predicted_class, confidence, probabilities.tolist()


def run_inference(model_path: str, test_dir: str, output_path: str, device: torch.device):
    """Main inference pipeline."""
    model, cfg = load_model(model_path, device)
    model.eval()
    transform = get_test_transform(cfg.get("image_size", 384))

    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_files = sorted([
        f for f in test_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        raise ValueError(f"No images found in {test_dir}")

    logger.info(f"Found {len(image_files)} test images")

    results = []
    for img_path in tqdm(image_files, desc="Classifying"):
        predicted_class, confidence, probs = predict_single(
            model, img_path, transform, device
        )
        image_id = img_path.stem  # filename without extension
        results.append({
            "Image_ID": image_id,
            "Label": predicted_class,
        })

        if confidence < 0.5:
            logger.warning(f"Low confidence ({confidence:.2f}) for {img_path.name}")

    df = pd.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Excel (.xlsx) — required format
    df.to_excel(output_path, index=False)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total predictions: {len(results)}")

    for cls in sorted(df["Label"].unique()):
        count = (df["Label"] == cls).sum()
        logger.info(f"  Class {cls}: {count} images")

    return df


def main():
    parser = argparse.ArgumentParser(description=f"Inference Problem A - {TEAM_NAME}")
    parser.add_argument("--model_path", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--test_dir", required=True, help="Directory with test images")
    parser.add_argument("--output_path", default=f"{TEAM_NAME} test_ground_truth.xlsx")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Team: {TEAM_NAME} | Device: {device}")

    try:
        run_inference(args.model_path, args.test_dir, args.output_path, device)
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
