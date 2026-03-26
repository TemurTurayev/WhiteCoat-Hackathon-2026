"""
Ensemble Weight Optimization + Stacking — Post-Training
Team: WhiteCoat.dev

Запуск после обучения всех K-Fold моделей:
    python ensemble_optimizer.py --checkpoint_dir checkpoints --task classification
    python ensemble_optimizer.py --checkpoint_dir checkpoints --task segmentation

Что делает:
1. Загружает OOF predictions от всех моделей
2. Оптимизирует веса ансамбля (scipy.optimize)
3. Пробует stacking (LogisticRegression / XGBoost)
4. Сравнивает все стратегии и выбирает лучшую
5. Сохраняет оптимальные веса для inference
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CLASSIFICATION ENSEMBLE
# ============================================================

def load_classification_oof(checkpoint_dir: str, model_names: list[str]):
    """Load OOF predictions from all models."""
    oof_dir = Path(checkpoint_dir) / "oof"
    all_predictions = []
    labels = None

    for model_name in model_names:
        model_short = model_name.replace("/", "_")
        oof_path = oof_dir / f"oof_{model_short}.npz"
        if not oof_path.exists():
            print(f"WARNING: {oof_path} not found, skipping {model_name}")
            continue

        data = np.load(oof_path)
        all_predictions.append(data["predictions"])
        if labels is None:
            labels = data["labels"]
        print(f"Loaded OOF for {model_name}: shape={data['predictions'].shape}")
        if "fold_accuracies" in data:
            accs = data["fold_accuracies"]
            print(f"  Fold accuracies: {[f'{a:.4f}' for a in accs]}")
            print(f"  Mean: {accs.mean():.4f}")

    return all_predictions, labels


def classification_simple_average(predictions_list, labels):
    """Simple average of softmax probabilities."""
    avg = np.mean(predictions_list, axis=0)
    preds = np.argmax(avg, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


def classification_weighted_average(predictions_list, labels):
    """Optimize weights via Nelder-Mead to maximize accuracy."""
    n_models = len(predictions_list)
    stacked = np.stack(predictions_list, axis=0)  # (n_models, n_samples, n_classes)

    def neg_accuracy(raw_weights):
        # Softmax to ensure positive weights that sum to 1
        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_w / exp_w.sum()
        weighted_avg = np.tensordot(weights, stacked, axes=([0], [0]))
        preds = np.argmax(weighted_avg, axis=1)
        return -accuracy_score(labels, preds)

    best_result = None
    best_neg_acc = 0.0

    # Multiple random restarts for robustness
    for seed in range(20):
        rng = np.random.RandomState(seed)
        x0 = rng.randn(n_models) * 0.1
        result = minimize(neg_accuracy, x0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-6})
        if best_result is None or result.fun < best_neg_acc:
            best_neg_acc = result.fun
            best_result = result

    exp_w = np.exp(best_result.x - np.max(best_result.x))
    optimal_weights = exp_w / exp_w.sum()

    weighted_avg = np.tensordot(optimal_weights, stacked, axes=([0], [0]))
    preds = np.argmax(weighted_avg, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1, optimal_weights


def classification_majority_voting(predictions_list, labels):
    """Hard majority voting."""
    all_preds = np.stack([np.argmax(p, axis=1) for p in predictions_list], axis=0)
    # (n_models, n_samples) -> majority vote per sample
    n_samples = all_preds.shape[1]
    n_classes = predictions_list[0].shape[1]

    final_preds = []
    for i in range(n_samples):
        votes = np.bincount(all_preds[:, i], minlength=n_classes)
        final_preds.append(np.argmax(votes))
    final_preds = np.array(final_preds)

    acc = accuracy_score(labels, final_preds)
    f1 = f1_score(labels, final_preds, average="macro")
    return acc, f1


def classification_stacking(predictions_list, labels):
    """Level-2 stacking with LogisticRegression on concatenated softmax."""
    # Concatenate softmax outputs: (n_samples, n_models * n_classes)
    stacked_features = np.concatenate(predictions_list, axis=1)

    # 5-fold cross-validation for stacking to avoid overfitting
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(labels), dtype=int)

    for train_idx, val_idx in skf.split(stacked_features, labels):
        meta_model = LogisticRegression(
            C=1.0, max_iter=1000, multi_class="multinomial", solver="lbfgs"
        )
        meta_model.fit(stacked_features[train_idx], labels[train_idx])
        oof_preds[val_idx] = meta_model.predict(stacked_features[val_idx])

    acc = accuracy_score(labels, oof_preds)
    f1 = f1_score(labels, oof_preds, average="macro")
    return acc, f1


def run_classification_ensemble(checkpoint_dir, model_names):
    """Run all classification ensemble strategies and compare."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    predictions_list, labels = load_classification_oof(checkpoint_dir, model_names)
    if not predictions_list:
        print("ERROR: No OOF predictions found!")
        return

    n_models = len(predictions_list)
    print(f"\nModels loaded: {n_models}")
    print(f"Samples: {len(labels)}")

    # --- Individual model OOF accuracies ---
    print("\n--- Individual Model OOF Accuracies ---")
    for i, (name, pred) in enumerate(zip(model_names, predictions_list)):
        preds = np.argmax(pred, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")

    # --- Ensemble strategies ---
    results = {}

    print("\n--- Ensemble Strategies ---")

    # 1. Simple Average
    acc, f1 = classification_simple_average(predictions_list, labels)
    results["simple_average"] = {"accuracy": acc, "f1": f1}
    print(f"  Simple Average:     Acc={acc:.4f} F1={f1:.4f}")

    # 2. Weighted Average
    acc, f1, weights = classification_weighted_average(predictions_list, labels)
    results["weighted_average"] = {
        "accuracy": acc, "f1": f1,
        "weights": {name: float(w) for name, w in zip(model_names, weights)},
    }
    print(f"  Weighted Average:   Acc={acc:.4f} F1={f1:.4f}")
    for name, w in zip(model_names, weights):
        print(f"    {name}: {w:.4f}")

    # 3. Majority Voting
    acc, f1 = classification_majority_voting(predictions_list, labels)
    results["majority_voting"] = {"accuracy": acc, "f1": f1}
    print(f"  Majority Voting:    Acc={acc:.4f} F1={f1:.4f}")

    # 4. Stacking
    acc, f1 = classification_stacking(predictions_list, labels)
    results["stacking_lr"] = {"accuracy": acc, "f1": f1}
    print(f"  Stacking (LR):      Acc={acc:.4f} F1={f1:.4f}")

    # --- Best strategy ---
    best_strategy = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n  BEST: {best_strategy} -> Acc={results[best_strategy]['accuracy']:.4f}")

    # --- Save results ---
    output = {
        "task": "classification",
        "n_models": n_models,
        "model_names": model_names[:n_models],
        "best_strategy": best_strategy,
        "results": results,
    }
    output_path = Path(checkpoint_dir) / "ensemble_results_classification.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


# ============================================================
# SEGMENTATION ENSEMBLE
# ============================================================

def sigmoid_np(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def compute_iou_np(pred_binary, masks):
    """Compute per-sample IoU."""
    intersection = (pred_binary * masks).sum(axis=(1, 2, 3))
    union = pred_binary.sum(axis=(1, 2, 3)) + masks.sum(axis=(1, 2, 3)) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def load_segmentation_logits(checkpoint_dir: str, model_configs: list[dict]):
    """Load external validation logits from all segmentation models."""
    oof_dir = Path(checkpoint_dir) / "oof"
    all_logits = []

    for config in model_configs:
        arch_short = f"{config['arch']}_{config['encoder']}".replace("-", "_")
        logits_path = oof_dir / f"ext_logits_{arch_short}.npz"
        if not logits_path.exists():
            print(f"WARNING: {logits_path} not found")
            continue

        data = np.load(logits_path)
        # Shape: (n_folds, n_samples, 1, H, W)
        logits = data["logits"]
        # Average across folds first
        avg_logits = logits.mean(axis=0)
        all_logits.append(avg_logits)
        print(f"Loaded {config['arch']}+{config['encoder']}: {logits.shape}")

    return all_logits


def segmentation_optimize_weights_and_threshold(logits_list, masks):
    """Joint optimization of model weights and threshold."""
    n_models = len(logits_list)
    stacked = np.stack(logits_list, axis=0)  # (n_models, n_samples, 1, H, W)

    def neg_iou(params):
        raw_weights = params[:n_models]
        threshold = 1.0 / (1.0 + np.exp(-params[n_models]))  # sigmoid for threshold

        exp_w = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_w / exp_w.sum()

        weighted_logits = np.tensordot(weights, stacked, axes=([0], [0]))
        probs = sigmoid_np(weighted_logits)
        pred_binary = (probs > threshold).astype(np.float32)
        ious = compute_iou_np(pred_binary, masks)
        return -ious.mean()

    best_result = None
    best_score = 0.0

    for seed in range(15):
        rng = np.random.RandomState(seed)
        x0 = np.concatenate([
            rng.randn(n_models) * 0.1,
            [0.0],  # threshold=0.5 in logit space
        ])
        result = minimize(neg_iou, x0, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-6})
        if best_result is None or result.fun < best_score:
            best_score = result.fun
            best_result = result

    params = best_result.x
    raw_weights = params[:n_models]
    exp_w = np.exp(raw_weights - np.max(raw_weights))
    optimal_weights = exp_w / exp_w.sum()
    optimal_threshold = 1.0 / (1.0 + np.exp(-params[n_models]))

    return -best_score, optimal_weights, optimal_threshold


def run_segmentation_ensemble(checkpoint_dir, model_configs, ext_val_masks_path=None):
    """Run segmentation ensemble optimization."""
    print("\n" + "=" * 70)
    print("SEGMENTATION ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    logits_list = load_segmentation_logits(checkpoint_dir, model_configs)
    if not logits_list:
        print("ERROR: No logits found!")
        return

    print(f"\nModels loaded: {len(logits_list)}")

    # We need ground truth masks for optimization
    # These should be saved during training or loaded here
    if ext_val_masks_path and Path(ext_val_masks_path).exists():
        masks = np.load(ext_val_masks_path)["masks"]
    else:
        print("\nWARNING: No external validation masks provided.")
        print("Run with --ext_val_masks_path to optimize weights.")
        print("Defaulting to equal weights with threshold search only.")

        # Just show equal-weight results at different thresholds
        avg_logits = np.mean(logits_list, axis=0)
        avg_probs = sigmoid_np(avg_logits)
        print("\nEqual-weight ensemble probabilities computed.")
        print("To evaluate, provide ground truth masks.")

        output = {
            "task": "segmentation",
            "n_models": len(logits_list),
            "best_strategy": "equal_average",
            "optimal_threshold": 0.5,
            "weights": {f"model_{i}": 1.0 / len(logits_list) for i in range(len(logits_list))},
        }
        output_path = Path(checkpoint_dir) / "ensemble_results_segmentation.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        return output

    n_models = len(logits_list)
    results = {}

    # 1. Simple average + threshold sweep
    print("\n--- Simple Average + Threshold Sweep ---")
    avg_logits = np.mean(logits_list, axis=0)
    avg_probs = sigmoid_np(avg_logits)

    best_threshold = 0.5
    best_iou = 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        pred_binary = (avg_probs > t).astype(np.float32)
        iou = compute_iou_np(pred_binary, masks).mean()
        if iou > best_iou:
            best_iou = iou
            best_threshold = t
        print(f"  threshold={t:.2f}: IoU={iou:.4f}")

    results["simple_average"] = {"iou": float(best_iou), "threshold": float(best_threshold)}
    print(f"  BEST: threshold={best_threshold:.2f} IoU={best_iou:.4f}")

    # 2. Weighted average + joint threshold optimization
    if n_models > 1:
        print("\n--- Weighted Average + Joint Optimization ---")
        opt_iou, opt_weights, opt_threshold = segmentation_optimize_weights_and_threshold(
            logits_list, masks
        )
        model_names = [f"{c['arch']}+{c['encoder']}" for c in model_configs[:n_models]]
        results["weighted_average"] = {
            "iou": float(opt_iou),
            "threshold": float(opt_threshold),
            "weights": {name: float(w) for name, w in zip(model_names, opt_weights)},
        }
        print(f"  IoU={opt_iou:.4f} threshold={opt_threshold:.3f}")
        for name, w in zip(model_names, opt_weights):
            print(f"    {name}: {w:.4f}")

    # 3. Majority voting on binary masks
    print("\n--- Majority Voting (binary) ---")
    binary_masks_list = []
    for logits in logits_list:
        probs = sigmoid_np(logits)
        binary_masks_list.append((probs > 0.5).astype(np.float32))
    vote_sum = np.sum(binary_masks_list, axis=0)
    majority = (vote_sum > n_models / 2.0).astype(np.float32)
    maj_iou = compute_iou_np(majority, masks).mean()
    results["majority_voting"] = {"iou": float(maj_iou)}
    print(f"  Majority Voting IoU={maj_iou:.4f}")

    # --- Best strategy ---
    best_strategy = max(results, key=lambda k: results[k]["iou"])
    print(f"\n  BEST: {best_strategy} -> IoU={results[best_strategy]['iou']:.4f}")

    output = {
        "task": "segmentation",
        "n_models": n_models,
        "best_strategy": best_strategy,
        "results": results,
    }
    output_path = Path(checkpoint_dir) / "ensemble_results_segmentation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return output


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Optimizer — WhiteCoat.dev"
    )
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument(
        "--task", choices=["classification", "segmentation", "both"],
        default="both",
    )
    parser.add_argument("--ext_val_masks_path", default=None,
                       help="Path to .npz with ground truth masks for segmentation")
    args = parser.parse_args()

    # Default model names from config
    cls_models = [
        "tf_efficientnetv2_s",
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
    ]
    seg_models = [
        {"arch": "UnetPlusPlus", "encoder": "efficientnet-b4"},
        {"arch": "DeepLabV3Plus", "encoder": "resnet50"},
    ]

    if args.task in ("classification", "both"):
        run_classification_ensemble(args.checkpoint_dir, cls_models)

    if args.task in ("segmentation", "both"):
        run_segmentation_ensemble(
            args.checkpoint_dir, seg_models, args.ext_val_masks_path
        )


if __name__ == "__main__":
    main()
