"""
WhiteCoat.dev -- Segmentation Inference
Loads trained segmentation model and generates predicted binary masks.

Usage:
    python WhiteCoat.devSegmentation.py --test_dir <path_to_test_images> --model <model.pth>

Output: WhiteCoat.dev/ folder with binary PNG masks
"""
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 512
THRESHOLD = 0.5


def get_tta_transforms(size):
    """4x TTA: original + hflip + vflip + hflip+vflip."""
    base = A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD), ToTensorV2()])
    hflip = A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()])
    vflip = A.Compose([A.Resize(size, size), A.VerticalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()])
    hvflip = A.Compose([A.Resize(size, size), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Normalize(MEAN, STD), ToTensorV2()])
    return [
        (base, None),
        (hflip, "hflip"),
        (vflip, "vflip"),
        (hvflip, "hvflip"),
    ]


def undo_flip(pred, flip_type):
    """Reverse the flip applied during TTA."""
    if flip_type == "hflip":
        return torch.flip(pred, [3])
    elif flip_type == "vflip":
        return torch.flip(pred, [2])
    elif flip_type == "hvflip":
        return torch.flip(pred, [2, 3])
    return pred


def main():
    parser = argparse.ArgumentParser(description="WhiteCoat.dev Segmentation")
    parser.add_argument("--test_dir", required=True, help="Path to test images folder")
    parser.add_argument("--model", default="best_model.pth", help="Path to model weights")
    parser.add_argument("--output_dir", default="WhiteCoat.dev", help="Output masks folder")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = smp.UnetPlusPlus(
        encoder_name="tu-tf_efficientnetv2_s",
        encoder_weights=None,
        classes=1,
        activation=None,
    )
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print(f"Loaded model: {args.model}")

    # TTA transforms
    tta_list = get_tta_transforms(args.img_size)

    # Find test images
    test_dir = Path(args.test_dir)
    images = sorted(
        list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")),
        key=lambda p: p.stem,
    )
    print(f"Test images: {len(images)}")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predict
    for i, img_path in enumerate(images):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        avg_pred = torch.zeros(1, 1, args.img_size, args.img_size).to(device)

        with torch.no_grad():
            for transform, flip_type in tta_list:
                tensor = transform(image=img)["image"].unsqueeze(0).to(device)
                pred = torch.sigmoid(model(tensor))
                pred = undo_flip(pred, flip_type)
                avg_pred += pred

        avg_pred /= len(tta_list)

        # Convert to binary mask at original resolution
        mask = avg_pred.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = ((mask > args.threshold) * 255).astype(np.uint8)

        # Save with same filename as input
        out_path = out_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), binary_mask)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(images)}")

    print(f"\nSaved {len(images)} masks to {args.output_dir}/")


if __name__ == "__main__":
    main()
