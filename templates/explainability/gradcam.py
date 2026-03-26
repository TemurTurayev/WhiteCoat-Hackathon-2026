"""
GradCAM Visualization for Skin Lesion Classification
Team: WhiteCoat.dev | Hackathon 2026

Supports: EfficientNetV2-S, ConvNeXt-Tiny, Swin-Tiny (timm models).
Produces publication-quality heatmap overlays for presentations.

Usage:
    python -m explainability.gradcam \
        --checkpoint checkpoints/best_tf_efficientnetv2_s.pth \
        --image path/to/lesion.png \
        --output outputs/gradcam_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ---------------------------------------------------------------------------
# Target layer lookup — critical: must match the architecture
# ---------------------------------------------------------------------------

TARGET_LAYER_REGISTRY: dict[str, str] = {
    # EfficientNetV2-S: last MBConv block normalization
    "tf_efficientnetv2_s": "blocks.5",
    # ConvNeXt-Tiny: last stage, last block
    "convnext_tiny": "stages.3.blocks.2",
    # Swin-Tiny: last norm of last attention block
    "swin_tiny_patch4_window7_224": "layers.3.blocks.1.norm1",
}


def _resolve_target_layer(model: torch.nn.Module, model_name: str) -> list:
    """Walk the model tree to find the correct target layer."""
    dotpath = TARGET_LAYER_REGISTRY.get(model_name)
    if dotpath is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Add an entry to TARGET_LAYER_REGISTRY. "
            f"Hint: print(model) to find the last conv/norm layer."
        )
    module = model
    for part in dotpath.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return [module]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load a timm model from a WhiteCoat.dev checkpoint.

    Returns (model, model_name, num_classes, image_size).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = timm.create_model(
        cfg["model_name"],
        pretrained=False,
        num_classes=cfg["num_classes"],
        drop_rate=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.set_grad_enabled = True
    return model, cfg["model_name"], cfg["num_classes"], cfg["image_size"]


def preprocess_image(
    image_path: str,
    image_size: int = 224,
) -> tuple[torch.Tensor, np.ndarray]:
    """Read image, resize, normalize for timm.
    Returns (input_tensor [1,3,H,W], rgb_float [H,W,3] in 0..1).
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size))

    rgb_float = rgb.astype(np.float32) / 255.0

    # ImageNet normalization (timm default)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_float - mean) / std

    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, rgb_float


def generate_gradcam(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    rgb_image: np.ndarray,
    target_class: Optional[int] = None,
    method: str = "gradcam++",
    aug_smooth: bool = False,
    eigen_smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate GradCAM heatmap overlay.

    Args:
        model: loaded timm model
        model_name: key into TARGET_LAYER_REGISTRY
        input_tensor: preprocessed [1,3,H,W]
        rgb_image: original image [H,W,3] float32 in 0..1
        target_class: class index (None = predicted class)
        method: "gradcam", "gradcam++", or "eigencam"
        aug_smooth: test-time augmentation smoothing
        eigen_smooth: PCA smoothing of activations

    Returns:
        (overlay_rgb, grayscale_cam, predicted_class)
    """
    target_layers = _resolve_target_layer(model, model_name)

    cam_classes = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigencam": EigenCAM,
    }
    cam_cls = cam_classes.get(method, GradCAMPlusPlus)

    with cam_cls(model=model, target_layers=target_layers) as cam:
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )
        grayscale_cam = grayscale_cam[0, :]

        predicted_class = cam.outputs.argmax(dim=1).item()

    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return overlay, grayscale_cam, predicted_class


def save_publication_figure(
    rgb_image: np.ndarray,
    overlay: np.ndarray,
    grayscale_cam: np.ndarray,
    predicted_class: int,
    class_names: Optional[list[str]] = None,
    output_path: str = "gradcam_output.png",
    title: Optional[str] = None,
    confidence: Optional[float] = None,
    dpi: int = 200,
) -> None:
    """Save a 3-panel publication-quality figure:
    [Original] [Heatmap Only] [Overlay]
    """
    matplotlib.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    label = class_names[predicted_class] if class_names else str(predicted_class)
    conf_str = f" ({confidence:.1%})" if confidence is not None else ""
    suptitle = title or f"Predicted: {label}{conf_str}"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)

    # Panel 1: Original
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Panel 2: Heatmap only
    im = axes[1].imshow(grayscale_cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Activation Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("GradCAM++ Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_multi_method_figure(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    rgb_image: np.ndarray,
    output_path: str = "gradcam_comparison.png",
    class_names: Optional[list[str]] = None,
    dpi: int = 200,
) -> None:
    """Side-by-side comparison of GradCAM, GradCAM++, and EigenCAM."""
    methods = ["gradcam", "gradcam++", "eigencam"]
    results = []
    pred_class = None
    for method in methods:
        overlay, heatmap, pc = generate_gradcam(
            model, model_name, input_tensor, rgb_image, method=method
        )
        results.append((overlay, heatmap))
        pred_class = pc

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    label = class_names[pred_class] if class_names else str(pred_class)
    fig.suptitle(f"XAI Comparison | Predicted: {label}", fontsize=16, fontweight="bold")

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (method, (overlay, _)) in enumerate(zip(methods, results)):
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(method.upper())
        axes[i + 1].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GradCAM | WhiteCoat.dev")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="outputs/gradcam_output.png")
    parser.add_argument("--method", default="gradcam++",
                        choices=["gradcam", "gradcam++", "eigencam"])
    parser.add_argument("--target_class", type=int, default=None)
    parser.add_argument("--compare", action="store_true",
                        help="Generate multi-method comparison figure")
    parser.add_argument("--aug_smooth", action="store_true")
    parser.add_argument("--eigen_smooth", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, num_classes, image_size = load_model(args.checkpoint, device)
    input_tensor, rgb_image = preprocess_image(args.image, image_size)
    input_tensor = input_tensor.to(device)

    if args.compare:
        generate_multi_method_figure(
            model, model_name, input_tensor, rgb_image,
            output_path=args.output,
        )
    else:
        overlay, heatmap, pred_class = generate_gradcam(
            model, model_name, input_tensor, rgb_image,
            target_class=args.target_class,
            method=args.method,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth,
        )

        # Get confidence
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, pred_class].item()

        save_publication_figure(
            rgb_image, overlay, heatmap,
            predicted_class=pred_class,
            confidence=confidence,
            output_path=args.output,
        )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
