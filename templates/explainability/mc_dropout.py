"""
MC Dropout Uncertainty Quantification for Skin Lesion Classification
Team: WhiteCoat.dev | Hackathon 2026

Monte Carlo Dropout: keep dropout ON during inference, run N forward passes,
measure variance of predictions to quantify model uncertainty.

Usage:
    python -m explainability.mc_dropout \
        --checkpoint checkpoints/best_tf_efficientnetv2_s.pth \
        --image path/to/lesion.png \
        --output outputs/uncertainty.png \
        --n_passes 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F

from explainability.gradcam import load_model, preprocess_image


# ---------------------------------------------------------------------------
# Core MC Dropout
# ---------------------------------------------------------------------------


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Set all Dropout layers to training mode while keeping
    BatchNorm/LayerNorm in inference mode.

    This is the key trick: dropout stays active during inference,
    effectively sampling from an approximate posterior.
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()


def mc_dropout_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 30,
) -> dict:
    """Run N stochastic forward passes with dropout enabled.

    Args:
        model: trained model (dropout layers will be set to train mode)
        input_tensor: preprocessed [1,3,H,W]
        n_passes: number of forward passes (20-50 recommended)

    Returns dict with:
        - mean_probs: [num_classes] mean softmax probabilities
        - std_probs: [num_classes] std of softmax probabilities
        - all_probs: [n_passes, num_classes] all softmax outputs
        - predicted_class: int
        - confidence: float (mean prob of predicted class)
        - uncertainty: float (std of predicted class prob)
        - entropy: float (predictive entropy)
        - mutual_information: float (epistemic uncertainty)
    """
    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    all_probs_np = np.array(all_probs)  # [n_passes, num_classes]
    mean_probs = all_probs_np.mean(axis=0)
    std_probs = all_probs_np.std(axis=0)

    predicted_class = int(mean_probs.argmax())
    confidence = float(mean_probs[predicted_class])
    uncertainty = float(std_probs[predicted_class])

    # Predictive entropy: H[y | x, D] = -sum(mean_p * log(mean_p))
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-10)))

    # Expected entropy: E_w[H[y | x, w]] = -mean(sum(p * log(p)))
    per_pass_entropy = -np.sum(all_probs_np * np.log(all_probs_np + 1e-10), axis=1)
    expected_entropy = float(per_pass_entropy.mean())

    # Mutual information (epistemic uncertainty) = predictive - expected entropy
    mutual_information = entropy - expected_entropy

    return {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "all_probs": all_probs_np,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "entropy": entropy,
        "mutual_information": mutual_information,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_uncertainty_report(
    result: dict,
    output_path: str = "outputs/uncertainty.png",
    class_names: Optional[list[str]] = None,
    dpi: int = 200,
) -> None:
    """Generate a 2-panel uncertainty report:
    Left: bar chart of mean probabilities with error bars
    Right: distribution of predicted class probability across MC passes
    """
    matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    num_classes = len(result["mean_probs"])
    labels = class_names if class_names else [str(i) for i in range(num_classes)]

    pred_cls = result["predicted_class"]
    pred_label = labels[pred_cls]
    conf = result["confidence"]
    unc = result["uncertainty"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"MC Dropout Uncertainty | Predicted: {pred_label} "
        f"({conf:.1%} +/- {unc:.1%})",
        fontsize=14, fontweight="bold",
    )

    # --- Left: bar chart with error bars ---
    ax = axes[0]
    x = np.arange(num_classes)
    colors = ["#e74c3c" if i == pred_cls else "#3498db" for i in range(num_classes)]

    ax.bar(x, result["mean_probs"], yerr=result["std_probs"],
           capsize=3, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Probability")
    ax.set_title("Class Probabilities (with uncertainty)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0 / num_classes, color="gray", linestyle="--", alpha=0.5,
               label="Random chance")
    ax.legend(fontsize=9)

    # --- Right: histogram of predicted class across MC passes ---
    ax2 = axes[1]
    pred_probs = result["all_probs"][:, pred_cls]
    n_passes = len(pred_probs)

    ax2.hist(pred_probs, bins=min(20, n_passes), color="#e74c3c",
             alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axvline(x=conf, color="black", linestyle="--", linewidth=2,
                label=f"Mean: {conf:.3f}")
    ax2.set_xlabel(f"P({pred_label})")
    ax2.set_ylabel("Count (MC passes)")
    ax2.set_title(f"Distribution across {n_passes} MC passes")
    ax2.legend(fontsize=10)

    # Add text box with key metrics
    textstr = (
        f"Predictive Entropy: {result['entropy']:.3f}\n"
        f"Epistemic Uncertainty: {result['mutual_information']:.3f}\n"
        f"N passes: {n_passes}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", bbox=props)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confidence_gauge(
    confidence: float,
    uncertainty: float,
    predicted_label: str,
    output_path: str = "outputs/confidence_gauge.png",
    dpi: int = 200,
) -> None:
    """Simple confidence gauge for demo slides.
    Shows a semicircle gauge from 0% to 100% with the confidence needle.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Determine color based on confidence
    if confidence >= 0.8:
        color = "#27ae60"  # green
        verdict = "HIGH CONFIDENCE"
    elif confidence >= 0.5:
        color = "#f39c12"  # orange
        verdict = "MODERATE CONFIDENCE"
    else:
        color = "#e74c3c"  # red
        verdict = "LOW CONFIDENCE"

    # Draw gauge background
    theta = np.linspace(np.pi, 0, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgray",
            linewidth=20, solid_capstyle="round")

    # Draw filled portion
    filled_angle = np.pi * (1 - confidence)
    theta_filled = np.linspace(np.pi, filled_angle, 100)
    ax.plot(np.cos(theta_filled), np.sin(theta_filled), color=color,
            linewidth=20, solid_capstyle="round")

    # Needle
    needle_angle = np.pi * (1 - confidence)
    ax.annotate("", xy=(np.cos(needle_angle) * 0.7, np.sin(needle_angle) * 0.7),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5))

    ax.text(0, -0.15, f"{confidence:.1%}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=color)
    ax.text(0, -0.35, verdict, ha="center", va="center",
            fontsize=12, fontweight="bold", color=color)
    ax.text(0, -0.5, f"{predicted_label} (+/- {uncertainty:.1%})", ha="center",
            fontsize=10, color="gray")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch uncertainty analysis
# ---------------------------------------------------------------------------


def batch_uncertainty_analysis(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_passes: int = 20,
    class_names: Optional[list[str]] = None,
    output_path: str = "outputs/batch_uncertainty.png",
) -> dict:
    """Run MC Dropout on an entire dataset. Useful for finding:
    - Which samples the model is most uncertain about
    - Class-level uncertainty patterns
    - Correlation between uncertainty and correctness
    """
    all_results = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        for i in range(images.size(0)):
            single = images[i:i+1]
            result = mc_dropout_inference(model, single, n_passes=n_passes)
            all_results.append(result)
            all_labels.append(labels[i].item())

    confidences = np.array([r["confidence"] for r in all_results])
    uncertainties = np.array([r["uncertainty"] for r in all_results])
    predictions = np.array([r["predicted_class"] for r in all_results])
    labels_np = np.array(all_labels)
    correct = predictions == labels_np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(confidences[correct], bins=20, alpha=0.7, label="Correct",
                 color="#27ae60", edgecolor="black")
    axes[0].hist(confidences[~correct], bins=20, alpha=0.7, label="Incorrect",
                 color="#e74c3c", edgecolor="black")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution: Correct vs Incorrect")
    axes[0].legend()

    axes[1].hist(uncertainties[correct], bins=20, alpha=0.7, label="Correct",
                 color="#27ae60", edgecolor="black")
    axes[1].hist(uncertainties[~correct], bins=20, alpha=0.7, label="Incorrect",
                 color="#e74c3c", edgecolor="black")
    axes[1].set_xlabel("Uncertainty (std)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Uncertainty Distribution: Correct vs Incorrect")
    axes[1].legend()

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidences.mean()),
        "mean_uncertainty": float(uncertainties.mean()),
        "correct_confidence": float(confidences[correct].mean()) if correct.any() else 0,
        "incorrect_confidence": float(confidences[~correct].mean()) if (~correct).any() else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MC Dropout Uncertainty | WhiteCoat.dev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/uncertainty.png")
    parser.add_argument("--n_passes", type=int, default=30)
    parser.add_argument("--gauge", action="store_true",
                        help="Also generate confidence gauge")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, num_classes, image_size = load_model(args.checkpoint, device)
    input_tensor, _ = preprocess_image(args.image, image_size)
    input_tensor = input_tensor.to(device)

    result = mc_dropout_inference(model, input_tensor, n_passes=args.n_passes)

    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f} +/- {result['uncertainty']:.4f}")
    print(f"Predictive entropy: {result['entropy']:.4f}")
    print(f"Epistemic uncertainty: {result['mutual_information']:.4f}")

    plot_uncertainty_report(result, output_path=args.output)
    print(f"Saved: {args.output}")

    if args.gauge:
        gauge_path = args.output.replace(".png", "_gauge.png")
        plot_confidence_gauge(
            result["confidence"],
            result["uncertainty"],
            predicted_label=str(result["predicted_class"]),
            output_path=gauge_path,
        )
        print(f"Saved: {gauge_path}")


if __name__ == "__main__":
    main()
