"""
t-SNE / UMAP Feature Space Visualization
Team: WhiteCoat.dev | Hackathon 2026

Extract features from the penultimate layer, run dimensionality reduction,
and visualize class clusters. Shows the model learned meaningful representations.

Usage:
    python -m explainability.feature_space \
        --checkpoint checkpoints/best_tf_efficientnetv2_s.pth \
        --data_dir data/classification/train \
        --output outputs/tsne.png \
        --method tsne
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
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Feature extraction hook
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Hook-based feature extractor for the penultimate layer."""

    def __init__(self, model: torch.nn.Module, model_name: str):
        self.features: Optional[torch.Tensor] = None
        self._hook = None
        self._register(model, model_name)

    def _register(self, model: torch.nn.Module, model_name: str) -> None:
        """Attach a forward hook to the global pooling layer."""
        target = None

        if hasattr(model, "global_pool"):
            target = model.global_pool
        elif hasattr(model, "head") and hasattr(model.head, "global_pool"):
            target = model.head.global_pool
        elif hasattr(model, "avgpool"):
            target = model.avgpool

        if target is None:
            raise RuntimeError(
                f"Cannot find global pool layer for {model_name}. "
                "Set target manually."
            )

        def hook_fn(module, input_val, output):
            if output.dim() > 2:
                output = output.view(output.size(0), -1)
            self.features = output.detach().cpu()

        self._hook = target.register_forward_hook(hook_fn)

    def remove(self) -> None:
        if self._hook is not None:
            self._hook.remove()


def extract_features(
    model: torch.nn.Module,
    model_name: str,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features for all samples.

    Returns (features [N, D], labels [N]).
    """
    extractor = FeatureExtractor(model, model_name)
    model.eval()

    all_features = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if count >= max_samples:
                break
            images = images.to(device)
            _ = model(images)  # triggers hook

            batch_feats = extractor.features.numpy()
            batch_labels = labels.numpy()

            remaining = max_samples - count
            all_features.append(batch_feats[:remaining])
            all_labels.append(batch_labels[:remaining])
            count += min(len(batch_labels), remaining)

    extractor.remove()
    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Dimensionality reduction + plotting
# ---------------------------------------------------------------------------


def compute_embedding(
    features: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE or UMAP on extracted features."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            learning_rate="auto",
            init="pca",
        )
        embedding = reducer.fit_transform(features)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
        )
        embedding = reducer.fit_transform(features)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    return embedding


def plot_feature_space(
    embedding: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    class_names: Optional[list[str]] = None,
    output_path: str = "outputs/feature_space.png",
    title: Optional[str] = None,
    dpi: int = 200,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """Scatter plot of 2D embedding colored by class."""
    matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    if num_classes <= 10:
        cmap = plt.cm.tab10
    elif num_classes <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar

    fig, ax = plt.subplots(figsize=figsize)

    for i, lbl in enumerate(sorted(unique_labels)):
        mask = labels == lbl
        name = class_names[lbl] if class_names else f"Class {lbl}"
        color = cmap(i / max(num_classes - 1, 1))

        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            label=name,
            alpha=0.6,
            s=20,
            edgecolors="none",
        )

    method_upper = method.upper()
    plot_title = title or f"{method_upper} Feature Space Visualization"
    ax.set_title(plot_title, fontsize=16, fontweight="bold")
    ax.set_xlabel(f"{method_upper}-1")
    ax.set_ylabel(f"{method_upper}-2")

    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9, markerscale=2,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_space_with_density(
    embedding: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    class_names: Optional[list[str]] = None,
    output_path: str = "outputs/feature_space_density.png",
    dpi: int = 200,
) -> None:
    """2-panel figure: scatter plot + density contours."""
    matplotlib.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = plt.cm.tab10 if num_classes <= 10 else plt.cm.tab20

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for i, lbl in enumerate(sorted(unique_labels)):
        mask = labels == lbl
        name = class_names[lbl] if class_names else f"Class {lbl}"
        color = cmap(i / max(num_classes - 1, 1))
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=[color], label=name, alpha=0.5, s=15)

    axes[0].set_title(f"{method.upper()} Scatter Plot", fontweight="bold")
    axes[0].legend(fontsize=8, markerscale=2, loc="best")

    axes[1].hexbin(embedding[:, 0], embedding[:, 1], gridsize=40,
                   cmap="YlOrRd", mincnt=1)
    axes[1].set_title("Sample Density", fontweight="bold")
    plt.colorbar(axes[1].collections[0], ax=axes[1], label="Count")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Feature Space Viz | WhiteCoat.dev")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="outputs/feature_space.png")
    parser.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--density", action="store_true",
                        help="Also generate density plot")
    args = parser.parse_args()

    from explainability.gradcam import load_model
    from utils.augmentations import get_classification_transforms
    from classification.dataset import MedicalClassificationDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, num_classes, image_size = load_model(args.checkpoint, device)

    transform = get_classification_transforms(image_size, is_train=False)
    dataset = MedicalClassificationDataset.from_folder(args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    print(f"Extracting features from {min(args.max_samples, len(dataset))} samples...")
    features, labels = extract_features(
        model, model_name, loader, device, max_samples=args.max_samples
    )
    print(f"Feature shape: {features.shape}")

    print(f"Running {args.method.upper()}...")
    embedding = compute_embedding(
        features, method=args.method, perplexity=args.perplexity
    )

    plot_feature_space(
        embedding, labels, method=args.method, output_path=args.output
    )
    print(f"Saved: {args.output}")

    if args.density:
        density_path = args.output.replace(".png", "_density.png")
        plot_feature_space_with_density(
            embedding, labels, method=args.method, output_path=density_path
        )
        print(f"Saved: {density_path}")


if __name__ == "__main__":
    main()
