"""
Inference Script — Problem B (Segmentation)
Team: WhiteCoat.dev (#37)

Usage:
    python inference_b.py --model_path checkpoints/best_segmentation.pth \
                          --test_dir data/test/ \
                          --output_dir results/WhiteCoat.dev_problem_b/

Requirements from guidelines:
- Load the trained model
- Accept a directory containing test images
- Generate the required output files
- Must perform inference only
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TEAM_NAME = "WhiteCoat.dev"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

MODEL_REGISTRY = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "FPN": smp.FPN,
    "Linknet": smp.Linknet,
}


def load_model(model_path: str, device: torch.device) -> tuple:
    """Load trained segmentation model from checkpoint."""
    logger.info(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model_class = MODEL_REGISTRY[cfg["model_name"]]
    model = model_class(
        encoder_name=cfg["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["num_classes"],
        activation="sigmoid" if cfg["num_classes"] == 1 else None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    logger.info(
        f"Model loaded: {cfg['model_name']} + {cfg['encoder_name']} | "
        f"Classes: {cfg['num_classes']} | "
        f"Val Dice: {checkpoint.get('val_dice', 'N/A')}"
    )
    return model, cfg


def get_test_transform(image_size: int = 512):
    """Transform for test images."""
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
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict segmentation mask for single image.
    Returns: (binary_mask, probability_mask) in original image size.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Cannot read image: {image_path}")
        return np.zeros((256, 256), dtype=np.uint8), np.zeros((256, 256), dtype=np.float32)

    original_h, original_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transform(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob_mask = output.squeeze().cpu().numpy()

    # Resize back to original size
    prob_mask = cv2.resize(prob_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

    # Binarize
    binary_mask = (prob_mask > threshold).astype(np.uint8) * 255

    return binary_mask, prob_mask


def run_inference(
    model_path: str,
    test_dir: str,
    output_dir: str,
    device: torch.device,
    threshold: float = 0.5,
):
    """Main segmentation inference pipeline."""
    model, cfg = load_model(model_path, device)
    model.eval()
    transform = get_test_transform(cfg.get("image_size", 512))

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_positive_pixels = 0
    total_pixels = 0

    for img_path in tqdm(image_files, desc="Segmenting"):
        binary_mask, prob_mask = predict_single(
            model, img_path, transform, device, threshold
        )

        # Save mask with same name
        output_name = img_path.stem + ".png"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), binary_mask)

        positive = (binary_mask > 0).sum()
        total = binary_mask.size
        total_positive_pixels += positive
        total_pixels += total

        if positive == 0:
            logger.warning(f"Empty mask for {img_path.name}")

    avg_coverage = total_positive_pixels / max(total_pixels, 1) * 100
    logger.info(f"Output masks saved to {output_dir}")
    logger.info(f"Total images processed: {len(image_files)}")
    logger.info(f"Average mask coverage: {avg_coverage:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=f"Inference Problem B - {TEAM_NAME}")
    parser.add_argument("--model_path", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--test_dir", required=True, help="Directory with test images")
    parser.add_argument("--output_dir", default=f"results/{TEAM_NAME}_problem_b/")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Team: {TEAM_NAME} | Device: {device}")

    try:
        run_inference(args.model_path, args.test_dir, args.output_dir, device, args.threshold)
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
