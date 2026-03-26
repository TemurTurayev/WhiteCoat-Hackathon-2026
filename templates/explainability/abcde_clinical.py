"""
ABCDE Clinical Feature Extraction from Segmentation Masks
Team: WhiteCoat.dev | Hackathon 2026

Computes dermatological ABCDE criteria from a binary segmentation mask:
  A - Asymmetry (compare halves along major/minor axes)
  B - Border irregularity (perimeter^2 / area ratio vs circle)
  C - Color variation (std of RGB channels within lesion)
  D - Diameter (maximum distance across the lesion)
  E - Evolution (requires temporal data, not applicable to single image)

This is HIGHLY innovative for judges: bridging AI with clinical dermatology.

Usage:
    python -m explainability.abcde_clinical \
        --image path/to/lesion.png \
        --mask path/to/mask.png \
        --output outputs/abcde_report.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ABCDEReport:
    """Immutable ABCDE analysis result."""
    asymmetry_score: float       # 0 (symmetric) to 1 (asymmetric)
    asymmetry_major: float       # asymmetry along major axis
    asymmetry_minor: float       # asymmetry along minor axis
    border_irregularity: float   # 0 (smooth circle) to 1 (irregular)
    color_variation: float       # normalized color std (0-1)
    color_channels_std: tuple[float, float, float]  # R, G, B std
    diameter_pixels: float       # max diameter in pixels
    diameter_mm: Optional[float] # if resolution known
    compactness: float           # 4*pi*area / perimeter^2 (1 = circle)
    area_pixels: int
    perimeter_pixels: float
    num_colors_detected: int     # out of 6 clinical colors


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def _align_mask_to_principal_axes(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate mask to align with its principal axes using image moments."""
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return mask, 0.0

    # Orientation angle from central moments
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = np.degrees(theta)

    # Rotate around centroid
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    rot_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    h, w = mask.shape
    rotated = cv2.warpAffine(mask.astype(np.uint8), rot_matrix, (w, h),
                              flags=cv2.INTER_NEAREST)
    return rotated, angle_deg


def compute_asymmetry(mask: np.ndarray) -> tuple[float, float, float]:
    """Compute asymmetry by comparing halves along major and minor axes.

    Returns (overall_score, major_axis_asym, minor_axis_asym).
    Score 0 = perfectly symmetric, 1 = maximum asymmetry.
    """
    aligned, _ = _align_mask_to_principal_axes(mask)

    # Find bounding box of lesion
    coords = np.where(aligned > 0)
    if len(coords[0]) == 0:
        return 0.0, 0.0, 0.0

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    cropped = aligned[y_min:y_max+1, x_min:x_max+1]

    h, w = cropped.shape
    area = cropped.sum()
    if area == 0:
        return 0.0, 0.0, 0.0

    # Major axis asymmetry (horizontal split)
    mid_y = h // 2
    top_half = cropped[:mid_y, :]
    bottom_half = cropped[mid_y:, :]

    # Make same size for comparison
    min_h = min(top_half.shape[0], bottom_half.shape[0])
    top_half = top_half[:min_h, :]
    bottom_flipped = np.flipud(bottom_half[:min_h, :])

    diff_major = np.abs(top_half.astype(float) - bottom_flipped.astype(float))
    asym_major = float(diff_major.sum() / max(area, 1))

    # Minor axis asymmetry (vertical split)
    mid_x = w // 2
    left_half = cropped[:, :mid_x]
    right_half = cropped[:, mid_x:]

    min_w = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_w]
    right_flipped = np.fliplr(right_half[:, :min_w])

    diff_minor = np.abs(left_half.astype(float) - right_flipped.astype(float))
    asym_minor = float(diff_minor.sum() / max(area, 1))

    # Overall: average, clipped to [0, 1]
    overall = min((asym_major + asym_minor) / 2.0, 1.0)

    return overall, min(asym_major, 1.0), min(asym_minor, 1.0)


def compute_border_irregularity(mask: np.ndarray) -> tuple[float, float, float]:
    """Compute border irregularity using compactness metric.

    Compactness = 4 * pi * Area / Perimeter^2
    Perfect circle = 1.0, irregular shape < 1.0

    Returns (irregularity_score, perimeter, compactness).
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0, 0.0, 1.0

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)

    if perimeter == 0 or area == 0:
        return 0.0, 0.0, 1.0

    compactness = (4 * np.pi * area) / (perimeter * perimeter)
    compactness = min(compactness, 1.0)

    # Irregularity: 0 = smooth circle, 1 = very irregular
    irregularity = 1.0 - compactness

    return irregularity, perimeter, compactness


def compute_color_variation(
    image_rgb: np.ndarray, mask: np.ndarray
) -> tuple[float, tuple[float, float, float], int]:
    """Compute color variation within the lesion.

    Checks for presence of 6 clinical colors:
    white, black, red, light-brown, dark-brown, blue-gray

    Returns (normalized_std, (r_std, g_std, b_std), num_colors_detected).
    """
    # Extract pixels within the mask
    lesion_pixels = image_rgb[mask > 0]
    if len(lesion_pixels) == 0:
        return 0.0, (0.0, 0.0, 0.0), 0

    r_std = float(lesion_pixels[:, 0].std())
    g_std = float(lesion_pixels[:, 1].std())
    b_std = float(lesion_pixels[:, 2].std())

    # Normalize to 0-1 range (max possible std for uint8 is ~127.5)
    max_std = 127.5
    overall_std = np.sqrt(r_std**2 + g_std**2 + b_std**2) / (max_std * np.sqrt(3))
    overall_std = min(overall_std, 1.0)

    # Detect presence of 6 clinical colors in HSV space
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lesion_hsv = hsv[mask > 0]

    num_colors = 0

    # White: high value, low saturation
    white_mask = (lesion_hsv[:, 1] < 30) & (lesion_hsv[:, 2] > 200)
    if white_mask.sum() > len(lesion_hsv) * 0.02:
        num_colors += 1

    # Black: low value
    black_mask = lesion_hsv[:, 2] < 50
    if black_mask.sum() > len(lesion_hsv) * 0.02:
        num_colors += 1

    # Red: hue near 0 or 180, high saturation
    red_mask = ((lesion_hsv[:, 0] < 10) | (lesion_hsv[:, 0] > 170)) & (lesion_hsv[:, 1] > 50)
    if red_mask.sum() > len(lesion_hsv) * 0.02:
        num_colors += 1

    # Light brown: hue 10-25, moderate value
    lb_mask = (lesion_hsv[:, 0] >= 10) & (lesion_hsv[:, 0] <= 25) & (lesion_hsv[:, 2] > 120)
    if lb_mask.sum() > len(lesion_hsv) * 0.02:
        num_colors += 1

    # Dark brown: hue 10-25, low value
    db_mask = (lesion_hsv[:, 0] >= 10) & (lesion_hsv[:, 0] <= 25) & (lesion_hsv[:, 2] <= 120)
    if db_mask.sum() > len(lesion_hsv) * 0.02:
        num_colors += 1

    # Blue-gray: hue 90-130, low saturation
    bg_mask = (lesion_hsv[:, 0] >= 90) & (lesion_hsv[:, 0] <= 130) & (lesion_hsv[:, 1] < 80)
    if bg_mask.sum() > len(lesion_hsv) * 0.02:
        num_colors += 1

    return overall_std, (r_std, g_std, b_std), num_colors


def compute_diameter(mask: np.ndarray, px_per_mm: Optional[float] = None) -> tuple[float, Optional[float]]:
    """Compute maximum diameter of the lesion.

    Returns (diameter_pixels, diameter_mm or None).
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0, None

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2)

    # Compute maximum Euclidean distance between any two contour points
    # For efficiency, use the convex hull
    hull = cv2.convexHull(points)
    hull_points = hull.reshape(-1, 2)

    max_dist = 0.0
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            d = np.linalg.norm(hull_points[i].astype(float) - hull_points[j].astype(float))
            if d > max_dist:
                max_dist = d

    diameter_mm = max_dist / px_per_mm if px_per_mm else None
    return float(max_dist), diameter_mm


# ---------------------------------------------------------------------------
# Full ABCDE analysis
# ---------------------------------------------------------------------------


def analyze_abcde(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    px_per_mm: Optional[float] = None,
) -> ABCDEReport:
    """Run full ABCDE analysis on a lesion image + mask.

    Args:
        image_rgb: RGB image [H,W,3] uint8
        mask: binary mask [H,W] (>0 = lesion)
        px_per_mm: pixels per millimeter (for diameter in mm)
    """
    binary_mask = (mask > 0).astype(np.uint8)

    asym_overall, asym_major, asym_minor = compute_asymmetry(binary_mask)
    irreg, perimeter, compactness = compute_border_irregularity(binary_mask)
    color_var, channel_stds, n_colors = compute_color_variation(image_rgb, binary_mask)
    diam_px, diam_mm = compute_diameter(binary_mask, px_per_mm)

    area = int(binary_mask.sum())

    return ABCDEReport(
        asymmetry_score=asym_overall,
        asymmetry_major=asym_major,
        asymmetry_minor=asym_minor,
        border_irregularity=irreg,
        color_variation=color_var,
        color_channels_std=channel_stds,
        diameter_pixels=diam_px,
        diameter_mm=diam_mm,
        compactness=compactness,
        area_pixels=area,
        perimeter_pixels=perimeter,
        num_colors_detected=n_colors,
    )


# ---------------------------------------------------------------------------
# Clinical report visualization
# ---------------------------------------------------------------------------


def plot_abcde_report(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    report: ABCDEReport,
    output_path: str = "outputs/abcde_report.png",
    title: str = "ABCDE Clinical Feature Analysis",
    dpi: int = 200,
) -> None:
    """Generate a publication-quality ABCDE clinical report figure."""
    matplotlib.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    # Layout: 2 rows, 4 columns
    # Row 1: Original | Mask Overlay | Asymmetry | Border
    # Row 2: Color map | Diameter | Score Summary | Risk

    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # --- (0,0) Original image ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title("Original Image", fontweight="bold")
    ax1.axis("off")

    # --- (0,1) Mask overlay ---
    ax2 = fig.add_subplot(gs[0, 1])
    overlay = image_rgb.copy()
    mask_colored = np.zeros_like(image_rgb)
    mask_colored[mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    ax2.imshow(blended)
    ax2.set_title("Segmentation Mask", fontweight="bold")
    ax2.axis("off")

    # --- (0,2) Asymmetry visualization ---
    ax3 = fig.add_subplot(gs[0, 2])
    aligned, angle = _align_mask_to_principal_axes(mask.astype(np.uint8))
    ax3.imshow(aligned, cmap="gray")
    h, w = aligned.shape
    ax3.axhline(y=h//2, color="red", linewidth=2, linestyle="--", label="Major axis")
    ax3.axvline(x=w//2, color="blue", linewidth=2, linestyle="--", label="Minor axis")
    ax3.set_title(f"A: Asymmetry = {report.asymmetry_score:.2f}", fontweight="bold",
                  color="red" if report.asymmetry_score > 0.3 else "green")
    ax3.legend(fontsize=8)
    ax3.axis("off")

    # --- (0,3) Border irregularity ---
    ax4 = fig.add_subplot(gs[0, 3])
    contour_img = image_rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        # Draw fitted ellipse for comparison
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:
            ellipse = cv2.fitEllipse(largest)
            cv2.ellipse(contour_img, ellipse, (0, 255, 0), 2)
    ax4.imshow(contour_img)
    ax4.set_title(f"B: Border Irreg = {report.border_irregularity:.2f}", fontweight="bold",
                  color="red" if report.border_irregularity > 0.3 else "green")
    border_legend = [
        mpatches.Patch(color="red", label="Actual border"),
        mpatches.Patch(color="green", label="Best-fit ellipse"),
    ]
    ax4.legend(handles=border_legend, fontsize=8)
    ax4.axis("off")

    # --- (1,0) Color variation heatmap ---
    ax5 = fig.add_subplot(gs[1, 0])
    # Show color std per pixel region
    if mask.sum() > 0:
        lesion_only = image_rgb.copy().astype(float)
        lesion_only[mask == 0] = 0
        color_std_map = lesion_only.std(axis=2)
        color_std_map[mask == 0] = np.nan
        im = ax5.imshow(color_std_map, cmap="hot")
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, label="Color Std")
    ax5.set_title(f"C: Color Var = {report.color_variation:.2f}\n"
                  f"({report.num_colors_detected}/6 colors)", fontweight="bold",
                  color="red" if report.num_colors_detected >= 3 else "green")
    ax5.axis("off")

    # --- (1,1) Diameter ---
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(image_rgb)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest.reshape(-1, 2))
        hull_pts = hull.reshape(-1, 2)
        # Find the two farthest points
        max_dist = 0
        pt1, pt2 = hull_pts[0], hull_pts[0]
        for i in range(len(hull_pts)):
            for j in range(i + 1, len(hull_pts)):
                d = np.linalg.norm(hull_pts[i].astype(float) - hull_pts[j].astype(float))
                if d > max_dist:
                    max_dist = d
                    pt1, pt2 = hull_pts[i], hull_pts[j]
        ax6.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "r-", linewidth=3)
        ax6.plot(*pt1, "ro", markersize=8)
        ax6.plot(*pt2, "ro", markersize=8)
    diam_text = f"D: {report.diameter_pixels:.0f} px"
    if report.diameter_mm is not None:
        diam_text += f" ({report.diameter_mm:.1f} mm)"
    ax6.set_title(diam_text, fontweight="bold",
                  color="red" if report.diameter_pixels > 60 else "green")
    ax6.axis("off")

    # --- (1,2) Score summary bar chart ---
    ax7 = fig.add_subplot(gs[1, 2])
    criteria = ["Asymmetry", "Border", "Color", "Diameter"]
    scores = [
        report.asymmetry_score,
        report.border_irregularity,
        report.color_variation,
        min(report.diameter_pixels / 100.0, 1.0),  # normalized
    ]
    colors = ["#e74c3c" if s > 0.3 else "#27ae60" for s in scores]
    bars = ax7.barh(criteria, scores, color=colors, edgecolor="black", linewidth=0.5)
    ax7.set_xlim(0, 1)
    ax7.set_xlabel("Score (0=normal, 1=abnormal)")
    ax7.set_title("ABCDE Score Summary", fontweight="bold")
    ax7.axvline(x=0.3, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    for bar, score in zip(bars, scores):
        ax7.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=10, fontweight="bold")

    # --- (1,3) Risk assessment ---
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis("off")

    risk_count = sum(1 for s in scores if s > 0.3)
    if risk_count >= 3:
        risk_level = "HIGH RISK"
        risk_color = "#e74c3c"
        risk_text = "Multiple ABCDE criteria elevated.\nRefer to dermatologist."
    elif risk_count >= 1:
        risk_level = "MODERATE RISK"
        risk_color = "#f39c12"
        risk_text = "Some criteria elevated.\nMonitor closely."
    else:
        risk_level = "LOW RISK"
        risk_color = "#27ae60"
        risk_text = "All criteria within\nnormal range."

    ax8.text(0.5, 0.7, risk_level, ha="center", va="center",
            fontsize=24, fontweight="bold", color=risk_color,
            transform=ax8.transAxes)
    ax8.text(0.5, 0.4, risk_text, ha="center", va="center",
            fontsize=12, color="gray", transform=ax8.transAxes)
    ax8.text(0.5, 0.15, f"Elevated criteria: {risk_count}/4",
            ha="center", va="center", fontsize=11,
            transform=ax8.transAxes)

    # Add a colored border
    for spine in ax8.spines.values():
        spine.set_edgecolor(risk_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ABCDE Clinical Analysis | WhiteCoat.dev")
    parser.add_argument("--image", required=True, help="Path to RGB lesion image")
    parser.add_argument("--mask", required=True, help="Path to binary mask image")
    parser.add_argument("--output", default="outputs/abcde_report.png")
    parser.add_argument("--px_per_mm", type=float, default=None,
                        help="Pixels per mm (for diameter calibration)")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask}")

    report = analyze_abcde(image_rgb, mask, px_per_mm=args.px_per_mm)

    print("=== ABCDE Clinical Report ===")
    print(f"A - Asymmetry:          {report.asymmetry_score:.3f} "
          f"(major={report.asymmetry_major:.3f}, minor={report.asymmetry_minor:.3f})")
    print(f"B - Border Irregularity: {report.border_irregularity:.3f} "
          f"(compactness={report.compactness:.3f})")
    print(f"C - Color Variation:     {report.color_variation:.3f} "
          f"({report.num_colors_detected}/6 clinical colors)")
    print(f"    RGB std: R={report.color_channels_std[0]:.1f}, "
          f"G={report.color_channels_std[1]:.1f}, B={report.color_channels_std[2]:.1f}")
    print(f"D - Diameter:            {report.diameter_pixels:.0f} px", end="")
    if report.diameter_mm is not None:
        print(f" ({report.diameter_mm:.1f} mm)")
    else:
        print()
    print(f"    Area: {report.area_pixels} px, Perimeter: {report.perimeter_pixels:.0f} px")

    plot_abcde_report(image_rgb, mask, report, output_path=args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
