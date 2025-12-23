"""
Visualize the width calculation method in resistance computation.

This script shows how widths are measured between outer and inner curves,
verifying that the measurement direction is perpendicular to the outer curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern import Pattern
from settings import layers_c
from calculate import Calculate


def visualize_width_measurement(pattern_params: dict, mode: str, num_arrows: int = 15):
    """
    Visualize how widths are measured in the resistance calculation.

    Args:
        pattern_params: Pattern parameters
        mode: "straight" or "superelliptic"
        num_arrows: Number of width measurement arrows to show
    """
    # Generate pattern
    params = pattern_params.copy()
    params["pattern_mode"] = mode

    config = {"layer": params}
    pattern_data = Pattern.GetPattern(
        preConfig=None,
        currentConfig=config,
        nextConfig=config,
        side="front",
        layer="mid",
        layerIndex=0,
        patternIndex=4,
        patternCount=9
    )

    # Build assist data to get the actual outer_end_idx
    current_assist = Pattern._buildAssist(params)
    next_assist = Pattern._buildAssist(params)

    # Call _buildShape to get the proper split index
    shape, outer_end_idx, top, bottom = Pattern._buildShape(current_assist, next_assist, "normal")

    # Split shape at outer_end_idx
    outer = shape[0:outer_end_idx]
    inner = shape[outer_end_idx:]

    print(f"Shape total points: {len(shape)}, Outer end idx: {outer_end_idx}")
    print(f"Outer points: {len(outer)}, Inner points: {len(inner)}")

    # Check the direction of outer curve
    # For proper visualization of symmetry, we want it to go from bottom to top (Y increasing)
    if len(outer) > 1:
        print(f"Outer curve Y range (before): {outer[0, 1]:.3f} -> {outer[-1, 1]:.3f}")

        # If Y is decreasing (going from top to bottom), reverse it
        if outer[0, 1] > outer[-1, 1]:
            outer = outer[::-1]
            print(f"Outer curve Y range (after reverse): {outer[0, 1]:.3f} -> {outer[-1, 1]:.3f}")

    # Resample inner curve to match outer curve's point count
    def cumulative_lengths(pts: np.ndarray) -> np.ndarray:
        if len(pts) < 2:
            return np.array([0.0])
        diffs = pts[1:] - pts[:-1]
        seg_lengths = np.linalg.norm(diffs, axis=1)
        return np.concatenate(([0.0], np.cumsum(seg_lengths)))

    def resample_curve(pts: np.ndarray, num_points: int) -> np.ndarray:
        if len(pts) == num_points:
            return pts

        cum_lengths = cumulative_lengths(pts)
        total_length = cum_lengths[-1]

        if total_length < 1e-12:
            return np.tile(pts[0], (num_points, 1))

        target_lengths = np.linspace(0, total_length, num_points)
        resampled = np.zeros((num_points, 2))

        for i, target in enumerate(target_lengths):
            if target <= 0.0:
                resampled[i] = pts[0]
            elif target >= total_length:
                resampled[i] = pts[-1]
            else:
                idx = np.searchsorted(cum_lengths, target, side="right") - 1
                idx = max(0, min(idx, len(pts) - 2))

                seg_len = cum_lengths[idx + 1] - cum_lengths[idx]
                if seg_len < 1e-12:
                    resampled[i] = pts[idx]
                else:
                    t = (target - cum_lengths[idx]) / seg_len
                    resampled[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])

        return resampled

    # Helper function to find actual perpendicular distance from a point to inner curve
    def find_perpendicular_distance(point: np.ndarray, normal: np.ndarray, curve: np.ndarray) -> float:
        """
        Find the actual perpendicular distance from point along normal to curve.
        Returns signed distance t (point + t * normal = intersection).
        Vectorized implementation.
        """
        # Prepare segment data
        seg_starts = curve[:-1]
        seg_ends = curve[1:]
        seg_vecs = seg_ends - seg_starts

        # Calculate determinants for all segments
        dets = normal[1] * seg_vecs[:, 0] - normal[0] * seg_vecs[:, 1]

        # Filter out parallel segments
        valid_mask = np.abs(dets) > 1e-12
        
        if not np.any(valid_mask):
            distances = np.linalg.norm(curve - point, axis=1)
            return np.min(distances)

        # Filter arrays
        dets = dets[valid_mask]
        seg_starts = seg_starts[valid_mask]
        seg_vecs = seg_vecs[valid_mask]

        # Calculate t and s for all valid segments
        dxs = seg_starts[:, 0] - point[0]
        dys = seg_starts[:, 1] - point[1]

        ts = (dys * seg_vecs[:, 0] - dxs * seg_vecs[:, 1]) / dets
        ss = (normal[0] * dys - normal[1] * dxs) / dets

        # Find valid intersections (0 <= s <= 1)
        intersection_mask = (ss >= 0) & (ss <= 1)

        if not np.any(intersection_mask):
            distances = np.linalg.norm(curve - point, axis=1)
            return np.min(distances)

        # Get t with minimum absolute distance
        valid_ts = ts[intersection_mask]
        idx_min = np.argmin(np.abs(valid_ts))
        best_t = valid_ts[idx_min]

        return best_t

    # Calculate cumulative arc length along INNER curve (this is the current flow path!)
    inner_arc_lengths = cumulative_lengths(inner)
    total_length = inner_arc_lengths[-1]

    # Use uniform sampling along arc length (every 0.01mm or adaptively based on total length)
    sample_spacing = 0.01  # mm
    num_samples = max(100, int(total_length / sample_spacing))
    sample_arc_lengths = np.linspace(0, total_length, num_samples)

    # Interpolate points along INNER curve at uniform arc lengths
    def interpolate_at_arc_length(curve, arc_lengths, target_length):
        """Interpolate point on curve at specific arc length."""
        if target_length <= 0:
            return curve[0], 0
        if target_length >= arc_lengths[-1]:
            return curve[-1], len(curve) - 2

        idx = np.searchsorted(arc_lengths, target_length, side='right') - 1
        idx = max(0, min(idx, len(curve) - 2))

        seg_len = arc_lengths[idx + 1] - arc_lengths[idx]
        if seg_len < 1e-12:
            return curve[idx], idx

        t = (target_length - arc_lengths[idx]) / seg_len
        point = curve[idx] + t * (curve[idx + 1] - curve[idx])
        return point, idx

    # Extract widths by sampling along INNER curve
    y_positions = []
    perpendicular_widths_current = []  # METHOD 1: Current (projection to resampled point)
    perpendicular_widths_correct = []  # METHOD 2: Perpendicular distance to curve
    perpendicular_widths_equidist = []  # METHOD 3: Minimum distance (equidistant sampling)

    tol = 1e-6

    for arc_len in sample_arc_lengths:
        # Get point on INNER curve at this arc length
        point, seg_idx = interpolate_at_arc_length(inner, inner_arc_lengths, arc_len)
        y_positions.append(point[1])

        # Find the segment this point belongs to
        if seg_idx >= len(inner) - 1:
            seg_idx = len(inner) - 2

        p1 = inner[seg_idx]
        p2 = inner[seg_idx + 1]
        seg_vec = p2 - p1
        ds = np.linalg.norm(seg_vec)

        if ds < tol:
            # Degenerate segment, use previous values or zero
            perpendicular_widths_current.append(perpendicular_widths_current[-1] if perpendicular_widths_current else 0)
            perpendicular_widths_correct.append(perpendicular_widths_correct[-1] if perpendicular_widths_correct else 0)
            perpendicular_widths_equidist.append(perpendicular_widths_equidist[-1] if perpendicular_widths_equidist else 0)
            continue

        tangent_dir = seg_vec / ds
        # Use RIGHT-hand normal (clockwise 90°) to point towards OUTER curve
        normal_dir = np.array([tangent_dir[1], -tangent_dir[0]])

        # METHOD 1: Not used anymore
        w_current = 0.0

        # METHOD 2: CORRECT - Actual perpendicular distance from INNER to OUTER curve
        w_correct = abs(find_perpendicular_distance(point, normal_dir, outer))

        # METHOD 3: EQUIDISTANT - Minimum Euclidean distance to OUTER curve
        distances = np.linalg.norm(outer - point, axis=1)
        w_equidist = np.min(distances)

        perpendicular_widths_current.append(w_current)
        perpendicular_widths_correct.append(w_correct)
        perpendicular_widths_equidist.append(w_equidist)

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax_pattern = fig.add_subplot(gs[0])
    ax_width = fig.add_subplot(gs[1])

    # ========== TOP PLOT: Pattern with width arrows ==========
    ax_pattern.plot(outer[:, 0], outer[:, 1], 'b-', linewidth=2, label='Outer curve', alpha=0.8)
    ax_pattern.plot(inner[:, 0], inner[:, 1], 'r-', linewidth=2, label='Inner curve', alpha=0.8)

    # Sample along INNER curve to show width arrows
    sample_indices_inner = np.linspace(0, len(sample_arc_lengths) - 1, num_arrows, dtype=int)

    # Draw width measurement arrows from INNER curve sampling points
    for sample_idx in sample_indices_inner:
        if sample_idx >= len(sample_arc_lengths):
            continue

        arc_len = sample_arc_lengths[sample_idx]

        # Get pointA on inner curve
        pointA, seg_idx = interpolate_at_arc_length(inner, inner_arc_lengths, arc_len)

        if seg_idx >= len(inner) - 1:
            seg_idx = len(inner) - 2

        # Get tangent at pointA
        p1 = inner[seg_idx]
        p2 = inner[seg_idx + 1]
        tangent_vec = p2 - p1
        ds = np.linalg.norm(tangent_vec)

        if ds < 1e-6:
            continue

        tangent_dir = tangent_vec / ds
        # Normal pointing towards outer curve
        normal_dir = np.array([tangent_dir[1], -tangent_dir[0]])

        # Find perpendicular width to outer and the actual pointB
        t_val = find_perpendicular_distance(pointA, normal_dir, outer)
        width_l = abs(t_val)

        # Calculate pointB (intersection with outer)
        pointB = pointA + normal_dir * t_val

        # Debug: Check if pointB is actually on outer curve
        if sample_idx == 0:
            dist_to_outer = np.min(np.linalg.norm(outer - pointB, axis=1))
            print(f"Debug sample {sample_idx}:")
            print(f"  pointA: {pointA}")
            print(f"  normal_dir: {normal_dir}")
            print(f"  width_l: {width_l:.4f}")
            print(f"  pointB: {pointB}")
            print(f"  Distance from pointB to outer curve: {dist_to_outer:.6f}")

        # Draw pointA on inner (red)
        ax_pattern.scatter([pointA[0]], [pointA[1]], c='red', s=80, zorder=5, edgecolors='black', linewidths=1)

        # Draw pointB on outer (blue)
        ax_pattern.scatter([pointB[0]], [pointB[1]], c='blue', s=80, zorder=5, edgecolors='black', linewidths=1)

        # Draw line from pointA to pointB
        ax_pattern.plot([pointA[0], pointB[0]], [pointA[1], pointB[1]],
                       'g-', linewidth=2, alpha=0.7,
                       label='Perpendicular width' if sample_idx == sample_indices_inner[0] else '')

        # Draw tangent direction (for reference)
        tangent_scale = 0.3
        ax_pattern.arrow(pointA[0], pointA[1],
                tangent_dir[0] * tangent_scale, tangent_dir[1] * tangent_scale,
                head_width=0.1, head_length=0.08,
                fc='purple', ec='purple', alpha=0.5, linewidth=1.5,
                label='Tangent' if sample_idx == sample_indices_inner[0] else '')

    # Labels and title for pattern plot
    ax_pattern.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax_pattern.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax_pattern.set_title(f'Width Measurement Visualization ({mode} mode)\n' +
                 'Green: Perpendicular width from inner | Purple: Tangent at inner',
                 fontsize=14, fontweight='bold', pad=20)

    ax_pattern.grid(True, alpha=0.3)
    ax_pattern.axis('equal')
    ax_pattern.legend(fontsize=11, loc='best')

    # Add text box with info
    info_text = f'Mode: {mode}\n'
    info_text += f'Outer points: {len(outer)}\n'
    info_text += f'Inner points: {len(inner)}\n'
    info_text += f'Sample arrows: {len(sample_indices_inner)}\n'
    info_text += f'Sampling interval: {sample_spacing:.3f} mm'

    ax_pattern.text(0.02, 0.98, info_text,
            transform=ax_pattern.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== BOTTOM PLOT: Width variation vs Y position - PERPENDICULAR METHOD ONLY ==========
    # Plot PERPENDICULAR method (perpendicular distance) - this is the CORRECT method
    ax_width.plot(y_positions, perpendicular_widths_correct, 'b-', linewidth=3,
                  label='Perpendicular Width', alpha=0.8)
    ax_width.fill_between(y_positions, 0, perpendicular_widths_correct, alpha=0.2, color='blue')

    # Mark average width
    avg_width_correct = np.mean(perpendicular_widths_correct)
    ax_width.axhline(y=avg_width_correct, color='red', linestyle='--', linewidth=2, alpha=0.6,
                     label=f'Average Width: {avg_width_correct:.3f} mm')

    # Mark center line (symmetry axis at h/2)
    if y_positions:
        y_center = (max(y_positions) + min(y_positions)) / 2
        ax_width.axvline(x=y_center, color='green', linestyle=':', linewidth=2, alpha=0.5,
                         label=f'Center (Y={y_center:.2f}mm)')

    # Labels and title for width plot
    ax_width.set_xlabel('Y Position (mm)', fontsize=12, fontweight='bold')
    ax_width.set_ylabel('Perpendicular Width (mm)', fontsize=12, fontweight='bold')
    ax_width.set_title('Perpendicular Width vs Y Position',
                       fontsize=13, fontweight='bold')
    ax_width.grid(True, alpha=0.3)
    ax_width.legend(fontsize=10, loc='best')

    # Add statistics box for perpendicular method
    y_range = max(y_positions) - min(y_positions) if y_positions else 0

    min_width_correct = np.min(perpendicular_widths_correct)
    max_width_correct = np.max(perpendicular_widths_correct)
    range_correct = max_width_correct - min_width_correct

    width_stats = 'Perpendicular Width Statistics:\n'
    width_stats += f'Average: {avg_width_correct:.4f} mm\n'
    width_stats += f'Min: {min_width_correct:.4f} mm\n'
    width_stats += f'Max: {max_width_correct:.4f} mm\n'
    width_stats += f'Range: {range_correct:.4f} mm\n'
    width_stats += f'\nSamples: {len(y_positions)}\n'
    width_stats += f'Height: {y_range:.2f} mm'

    ax_width.text(0.98, 0.97, width_stats,
                  transform=ax_width.transAxes,
                  fontsize=9,
                  verticalalignment='top',
                  horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent.parent.parent / f'width_visualization_{mode}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    return fig, (ax_pattern, ax_width)


def main():
    """Main execution function."""
    print("=" * 70)
    print("Width Measurement Visualization")
    print("=" * 70)

    # Load base configuration
    normal_layer_config = layers_c["layers"][1]
    global_settings = layers_c["global"]

    base_params = normal_layer_config["layer"].copy()
    base_params["pattern_psp"] = global_settings["layer_psp"]
    base_params["pattern_pbw"] = base_params["layer_pbw"]
    base_params["pattern_pbh"] = base_params["layer_pbh"]
    base_params["pattern_ppw"] = base_params["layer_ppw"]
    base_params["pattern_twist"] = False
    base_params["pattern_type"] = "wave"

    layer_pbw = base_params["layer_pbw"]
    base_params["pattern_tp1"] = layer_pbw / 2.0
    base_params["pattern_bp1"] = layer_pbw / 2.0
    base_params["pattern_tnn"] = 2.0
    base_params["pattern_bnn"] = 2.0

    # Test 1: Straight mode
    print("\n" + "=" * 70)
    print("Visualizing STRAIGHT mode")
    print("=" * 70)

    straight_params = base_params.copy()
    straight_params["pattern_tp3"] = 2.5  # Mid-range value
    straight_params["pattern_bp3"] = 2.5

    visualize_width_measurement(straight_params, "straight", num_arrows=20)

    # Test 2: Superellipse mode
    print("\n" + "=" * 70)
    print("Visualizing SUPERELLIPSE mode")
    print("=" * 70)

    superellipse_params = base_params.copy()
    superellipse_params["pattern_tmm"] = 1.2  # Mid-range value
    superellipse_params["pattern_bmm"] = 1.2
    superellipse_params["pattern_tp3"] = 2.0
    superellipse_params["pattern_bp3"] = 2.0
    superellipse_params["pattern_tp2"] = 2.0
    superellipse_params["pattern_bp2"] = 2.0

    visualize_width_measurement(superellipse_params, "superelliptic", num_arrows=20)

    plt.show()

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print("\nLegend:")
    print("  Green arrows: Perpendicular width from inner to outer")
    print("  Purple arrows: Tangent direction at inner curve")
    print("\nThe green arrows show the perpendicular distance measurement")
    print("used in the resistance calculation.")


if __name__ == "__main__":
    main()
