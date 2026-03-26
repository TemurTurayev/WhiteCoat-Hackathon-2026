"""
Master script: Generate ALL explainability visualizations for the demo.
Team: WhiteCoat.dev | Hackathon 2026

This script produces every visual needed for a 2-minute demo presentation.
Run it once after training is complete.

Usage:
    python -m explainability.generate_all_visuals \
        --checkpoint checkpoints/best_tf_efficientnetv2_s.pth \
        --sample_image data/classification/train/0/sample.png \
        --sample_mask data/segmentation/train/masks/sample.png \
        --data_dir data/classification/train \
        --output_dir outputs/explainability
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    sys.path.append(str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Generate All XAI Visuals | WhiteCoat.dev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample_image", required=True, help="One representative image")
    parser.add_argument("--sample_mask", default=None, help="Segmentation mask (for ABCDE)")
    parser.add_argument("--data_dir", default=None, help="Dataset dir (for t-SNE, confusion)")
    parser.add_argument("--output_dir", default="outputs/explainability")
    parser.add_argument("--n_mc_passes", type=int, default=30)
    parser.add_argument("--tsne_samples", type=int, default=1500)
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--skip_confusion", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from explainability.gradcam import (
        load_model, preprocess_image,
        generate_gradcam, save_publication_figure,
        generate_multi_method_figure,
    )
    from explainability.mc_dropout import (
        mc_dropout_inference, plot_uncertainty_report, plot_confidence_gauge,
    )

    model, model_name, num_classes, image_size = load_model(args.checkpoint, device)
    input_tensor, rgb_image = preprocess_image(args.sample_image, image_size)
    input_tensor = input_tensor.to(device)

    # ---- 1. GradCAM++ ----
    print("\n[1/6] Generating GradCAM++ heatmap...")
    t0 = time.time()
    overlay, heatmap, pred_class = generate_gradcam(
        model, model_name, input_tensor, rgb_image, method="gradcam++"
    )
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)
        confidence = probs[0, pred_class].item()

    save_publication_figure(
        rgb_image, overlay, heatmap,
        predicted_class=pred_class,
        confidence=confidence,
        output_path=str(output_dir / "01_gradcam.png"),
    )
    print(f"    Done in {time.time()-t0:.1f}s -> 01_gradcam.png")

    # ---- 2. Multi-method comparison ----
    print("\n[2/6] Generating XAI method comparison...")
    t0 = time.time()
    generate_multi_method_figure(
        model, model_name, input_tensor, rgb_image,
        output_path=str(output_dir / "02_xai_comparison.png"),
    )
    print(f"    Done in {time.time()-t0:.1f}s -> 02_xai_comparison.png")

    # ---- 3. MC Dropout Uncertainty ----
    print(f"\n[3/6] Running MC Dropout ({args.n_mc_passes} passes)...")
    t0 = time.time()
    mc_result = mc_dropout_inference(model, input_tensor, n_passes=args.n_mc_passes)
    plot_uncertainty_report(
        mc_result,
        output_path=str(output_dir / "03_uncertainty.png"),
    )
    plot_confidence_gauge(
        mc_result["confidence"],
        mc_result["uncertainty"],
        predicted_label=str(mc_result["predicted_class"]),
        output_path=str(output_dir / "03_confidence_gauge.png"),
    )
    print(f"    Confidence: {mc_result['confidence']:.1%} +/- {mc_result['uncertainty']:.1%}")
    print(f"    Done in {time.time()-t0:.1f}s -> 03_uncertainty.png, 03_confidence_gauge.png")

    # ---- 4. ABCDE Clinical (if mask provided) ----
    if args.sample_mask:
        print("\n[4/6] Computing ABCDE clinical features...")
        t0 = time.time()
        import cv2
        from explainability.abcde_clinical import analyze_abcde, plot_abcde_report

        img_bgr = cv2.imread(args.sample_image)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(args.sample_mask, cv2.IMREAD_GRAYSCALE)

        report = analyze_abcde(img_rgb, mask)
        plot_abcde_report(
            img_rgb, mask, report,
            output_path=str(output_dir / "04_abcde_report.png"),
        )
        print(f"    A={report.asymmetry_score:.2f} B={report.border_irregularity:.2f} "
              f"C={report.color_variation:.2f} D={report.diameter_pixels:.0f}px")
        print(f"    Done in {time.time()-t0:.1f}s -> 04_abcde_report.png")
    else:
        print("\n[4/6] Skipping ABCDE (no --sample_mask provided)")

    # ---- 5. t-SNE Feature Space ----
    if args.data_dir and not args.skip_tsne:
        print(f"\n[5/6] Computing t-SNE ({args.tsne_samples} samples)...")
        t0 = time.time()
        from explainability.feature_space import (
            extract_features, compute_embedding,
            plot_feature_space, plot_feature_space_with_density,
        )
        from utils.augmentations import get_classification_transforms
        from classification.dataset import MedicalClassificationDataset

        transform = get_classification_transforms(image_size, is_train=False)
        dataset = MedicalClassificationDataset.from_folder(args.data_dir, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, num_workers=4
        )

        features, labels = extract_features(
            model, model_name, loader, device, max_samples=args.tsne_samples
        )
        embedding = compute_embedding(features, method="tsne")
        plot_feature_space(
            embedding, labels, method="tsne",
            output_path=str(output_dir / "05_tsne.png"),
        )
        plot_feature_space_with_density(
            embedding, labels, method="tsne",
            output_path=str(output_dir / "05_tsne_density.png"),
        )
        print(f"    Done in {time.time()-t0:.1f}s -> 05_tsne.png, 05_tsne_density.png")
    else:
        print("\n[5/6] Skipping t-SNE")

    # ---- 6. Confusion Matrix ----
    if args.data_dir and not args.skip_confusion:
        print("\n[6/6] Generating confusion matrix...")
        t0 = time.time()
        from explainability.confusion_matrix import (
            generate_predictions, plot_confusion_matrix, plot_per_class_metrics,
        )

        transform = get_classification_transforms(image_size, is_train=False)
        dataset = MedicalClassificationDataset.from_folder(args.data_dir, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=4
        )

        y_true, y_pred = generate_predictions(model, loader, device)
        accuracy = (y_true == y_pred).mean()

        plot_confusion_matrix(
            y_true, y_pred,
            output_path=str(output_dir / "06_confusion_matrix.png"),
        )
        plot_per_class_metrics(
            y_true, y_pred,
            output_path=str(output_dir / "06_per_class_metrics.png"),
        )
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Done in {time.time()-t0:.1f}s -> 06_confusion_matrix.png")
    else:
        print("\n[6/6] Skipping confusion matrix")

    print(f"\n{'='*60}")
    print(f"All visuals saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
