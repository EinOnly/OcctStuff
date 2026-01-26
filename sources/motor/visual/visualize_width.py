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

# Add parent directory to path (sources/motor/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern import Pattern
from settings import layers_c as layers
from parameters import PParams


def visualize_width_measurement(pattern_params: dict, num_arrows: int = 15):
    """
    Visualize how widths are measured in the resistance calculation.

    Args:
        pattern_params: Pattern parameters (must include pattern_mode)
        num_arrows: Number of width measurement arrows to show
    """
    # Use parameters as-is (mode is already set in pattern_params)
    params = pattern_params.copy()
    mode = params.get("pattern_mode", "straight")

    print(f"\n>>> visualize_width_measurement() called with mode: {mode}")
    print(
        f">>> params tmm={params.get('pattern_tmm', 'MISSING')}, tnn={params.get('pattern_tnn', 'MISSING')}"
    )
    print(
        f">>> params bmm={params.get('pattern_bmm', 'MISSING')}, bnn={params.get('pattern_bnn', 'MISSING')}"
    )

    config = {"layer": params}
    pattern_data = Pattern.GetPattern(
        preConfig=None,
        currentConfig=config,
        nextConfig=config,
        side="front",
        layer="mid",
        layerIndex=0,
        patternIndex=4,
        patternCount=9,
    )

    # Build assist data to get the actual outer_end_idx
    current_assist = Pattern._buildAssist(params)
    next_assist = Pattern._buildAssist(params)

    # Call _buildShape to get the proper split index
    shape, outer_end_idx, top, bottom = Pattern._buildShape(
        current_assist, next_assist, "normal"
    )

    # Split shape at outer_end_idx
    # According to pattern.py _buildShape, the order is: [top_outer, bottom_outer, bottom_inner, top_inner]
    # outer_end_idx marks the end of outer path (top_outer + bottom_outer)
    outer = shape[0:outer_end_idx]
    inner = shape[outer_end_idx:]

    print(f"Shape total points: {len(shape)}, Outer end idx: {outer_end_idx}")
    print(f"Outer points: {len(outer)}, Inner points: {len(inner)}")

    # The outer curve already follows the correct order from _buildShape
    # Do NOT reverse it - it goes [top_outer, bottom_outer] naturally
    if len(outer) > 1:
        print(f"Outer curve Y range: {outer[0, 1]:.3f} -> {outer[-1, 1]:.3f}")

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

    # Helper function to find the strictly minimum distance from a point to a polyline
    def get_closest_point_on_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Find the minimum Euclidean distance from point to the polyline segments.
        Returns (min_distance, closest_point_on_polyline).
        Vectorized implementation.
        """
        if len(polyline) < 2:
            dist = np.linalg.norm(polyline[0] - point)
            return dist, polyline[0]

        # Polyline segments
        seg_starts = polyline[:-1]
        seg_ends = polyline[1:]
        seg_vecs = seg_ends - seg_starts
        
        # Point vectors relative to segment starts
        point_vecs = point - seg_starts
        
        # Project point onto lines containing segments
        # t = (point_vec . seg_vec) / (seg_vec . seg_vec)
        seg_lengths_sq = np.sum(seg_vecs**2, axis=1)
        
        # Avoid division by zero for zero-length segments
        valid_segs = seg_lengths_sq > 1e-12
        if not np.any(valid_segs):
            dists = np.linalg.norm(polyline - point, axis=1)
            idx = np.argmin(dists)
            return dists[idx], polyline[idx]
            
        # Filter valid segments
        seg_starts = seg_starts[valid_segs]
        seg_vecs = seg_vecs[valid_segs]
        point_vecs = point_vecs[valid_segs]
        seg_lengths_sq = seg_lengths_sq[valid_segs]
        
        t = np.sum(point_vecs * seg_vecs, axis=1) / seg_lengths_sq
        
        # Clamp t to segment bounds [0, 1]
        t = np.clip(t, 0.0, 1.0)
        
        # Calculate closest points on each infinite line, clamped to segment
        closest_points = seg_starts + t[:, np.newaxis] * seg_vecs
        
        # Calculate distances to these closest points
        dists = np.linalg.norm(closest_points - point, axis=1)
        
        # Find the minimum distance across all segments
        min_idx = np.argmin(dists)
        
        return dists[min_idx], closest_points[min_idx]

    # === RESAMPLING FOR SMOOTHNESS ===
    # Resample OUTER curve to high resolution to avoid discrete polyline artifacts in distance calculation
    # (Measurement is from Inner -> Outer, as requested)
    outer_cum_lengths = cumulative_lengths(outer)
    outer_total_len = outer_cum_lengths[-1]
    # Use very fine resolution (e.g. 5 microns) to approximate the smooth mathematical curve
    outer_res_spacing = 0.005 
    num_outer_samples = max(1000, int(outer_total_len / outer_res_spacing))
    
    outer_high_res = resample_curve(outer, num_outer_samples)
    print(f"Resampling outer curve for measurement: {len(outer)} -> {len(outer_high_res)} points")

    # Calculate cumulative arc length along INNER curve (this is the sampling baseline)
    inner_arc_lengths = cumulative_lengths(inner)
    total_length = inner_arc_lengths[-1]

    # Use CENTER-BASED uniform sampling
    # Sample from center outwards to ensure symmetry and alignment
    sample_spacing = 0.01  # mm
    mid_length = total_length / 2.0
    
    # Generate offsets: 0, 0.01, 0.02, ...
    max_offset = max(mid_length, total_length - mid_length)
    num_side_samples = int(np.ceil(max_offset / sample_spacing))
    
    offsets = np.arange(0, num_side_samples + 1) * sample_spacing
    
    # Create arc lengths: mid, mid+d, mid-d, mid+2d, mid-2d...
    # We want them sorted for plotting, but generated from center
    # Actually, simple set union is easier
    right_side = mid_length + offsets
    left_side = mid_length - offsets
    
    # Combine and filter to valid range [0, total_length]
    # Use a small epsilon to include endpoints if they land exactly on spacing
    combined_samples = np.concatenate([left_side, right_side])
    valid_mask = (combined_samples >= -1e-9) & (combined_samples <= total_length + 1e-9)
    sample_arc_lengths = np.unique(combined_samples[valid_mask]) # unique sorts them too
    
    print(f"Generated {len(sample_arc_lengths)} samples centered at {mid_length:.4f}")

    # Interpolate points along INNER curve at uniform arc lengths
    def interpolate_at_arc_length(curve, arc_lengths, target_length):
        """Interpolate point on curve at specific arc length."""
        if target_length <= 0:
            return curve[0], 0
        if target_length >= arc_lengths[-1]:
            return curve[-1], len(curve) - 2

        idx = np.searchsorted(arc_lengths, target_length, side="right") - 1
        idx = max(0, min(idx, len(curve) - 2))

        seg_len = arc_lengths[idx + 1] - arc_lengths[idx]
        if seg_len < 1e-12:
            return curve[idx], idx

        t = (target_length - arc_lengths[idx]) / seg_len
        point = curve[idx] + t * (curve[idx + 1] - curve[idx])
        return point, idx

    # Extract widths by sampling along INNER curve
    y_positions = []
    perpendicular_widths_correct = []
    
    tol = 1e-6

    for arc_len in sample_arc_lengths:
        # Get point on INNER curve at this arc length
        point, seg_idx = interpolate_at_arc_length(inner, inner_arc_lengths, arc_len)
        y_positions.append(point[1])

        # METHOD: STRICT MINIMUM DISTANCE to outer POLYLINE (using high-res for smoothness)
        # We search FOR the closest point ON outer_high_res
        dist, closest_p = get_closest_point_on_polyline(point, outer_high_res)
        
        perpendicular_widths_correct.append(dist)

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax_pattern = fig.add_subplot(gs[0])
    ax_width = fig.add_subplot(gs[1])

    # ========== TOP PLOT: Pattern with width arrows ==========
    ax_pattern.plot(
        outer[:, 0], outer[:, 1], "b-", linewidth=2, label="Outer curve", alpha=0.8
    )
    
    ax_pattern.plot(
        inner[:, 0], inner[:, 1], "r-", linewidth=2, label="Inner curve", alpha=0.8
    )

    # Sample along INNER curve to show width arrows
    sample_indices = np.linspace(
        0, len(sample_arc_lengths) - 1, num_arrows, dtype=int
    )

    # Draw width measurement arrows from INNER curve sampling points
    for sample_idx in sample_indices:
        if sample_idx >= len(sample_arc_lengths):
            continue

        arc_len = sample_arc_lengths[sample_idx]

        # Get pointA on INNER curve
        pointA, seg_idx = interpolate_at_arc_length(inner, inner_arc_lengths, arc_len)
        
        # Calculate strictly closest point on OUTER curve (high res)
        dist, pointB = get_closest_point_on_polyline(pointA, outer_high_res)
        
        # Draw pointA on inner (red)
        ax_pattern.scatter([pointA[0]], [pointA[1]], c='red', s=60, zorder=5)

        # Draw pointB on outer (blue)
        ax_pattern.scatter([pointB[0]], [pointB[1]], c='blue', s=60, zorder=5)

        # Draw line from pointA to pointB
        ax_pattern.plot([pointA[0], pointB[0]], [pointA[1], pointB[1]],
                       'g-', linewidth=2, alpha=0.7,
                       label='Min Distance' if sample_idx == sample_indices[0] else '')

    # Labels and title for pattern plot
    ax_pattern.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax_pattern.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax_pattern.set_title(f'Width Measurement (Min Distance) ({mode} mode)\n' +
                 'Green: Strictly minimum distance from INNER point to OUTER polyline (Centered Grid)',
                 fontsize=14, fontweight='bold', pad=20)

    ax_pattern.grid(True, alpha=0.3)
    ax_pattern.axis('equal')
    ax_pattern.legend(fontsize=11, loc='best')

    # Add text box with info
    info_text = f'Mode: {mode}\n'
    info_text += f'Outer points: {len(outer)}\n'
    info_text += f'Inner points: {len(inner)}\n'
    info_text += f'Sampling interval: {sample_spacing:.3f} mm'

    ax_pattern.text(0.02, 0.98, info_text,
            transform=ax_pattern.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ========== BOTTOM PLOT: Width variation vs Y position ==========
    ax_width.plot(y_positions, perpendicular_widths_correct, 'b-', linewidth=3,
                  label='Minimum Distance', alpha=0.8)
    ax_width.fill_between(y_positions, 0, perpendicular_widths_correct, alpha=0.2, color='blue')

    # Mark average width
    avg_width = np.mean(perpendicular_widths_correct)
    ax_width.axhline(y=avg_width, color='red', linestyle='--', linewidth=2, alpha=0.6,
                    label=f'Average Width: {avg_width:.3f} mm')

    # Labels and title for width plot
    ax_width.set_xlabel('Y Position (mm)', fontsize=12, fontweight='bold')
    ax_width.set_ylabel('Width (mm)', fontsize=12, fontweight='bold')
    ax_width.set_title('Minimum Distance Width vs Y Position', fontsize=13, fontweight='bold')
    ax_width.grid(True, alpha=0.3)
    ax_width.legend(fontsize=10, loc='best')

    # Add statistics box
    y_range = max(y_positions) - min(y_positions) if y_positions else 0
    min_width = np.min(perpendicular_widths_correct)
    max_width = np.max(perpendicular_widths_correct)
    width_range = max_width - min_width

    stats_text = "Width Statistics:\n"
    stats_text += f"Average: {avg_width:.4f} mm\n"
    stats_text += f"Min: {min_width:.4f} mm\n"
    stats_text += f"Max: {max_width:.4f} mm\n"
    stats_text += f"Range: {width_range:.4f} mm\n"
    
    ax_width.text(0.98, 0.97, stats_text,
            transform=ax_width.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_file = (
        Path(__file__).parent.parent.parent / f"width_visualization_{mode}.png"
    )
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    return fig, (ax_pattern, ax_width)


def main():
    """Main execution function."""
    print("=" * 70)
    print("Width Measurement Visualization")
    print("=" * 70)

    # Load base configuration from settings
    # Use layer 1 (normal) for representative parameters
    normal_layer_config = layers["layers"][1]
    global_settings = layers["global"]
    layer_params = normal_layer_config["layer"]

    print(
        f"Using layer {1}: type={normal_layer_config.get('type', 'normal')}, "
        f"twist={layer_params.get('pattern_twist', False)}"
    )

    # Create PParams instance to apply constraints
    pparams = PParams()

    # Prepare parameters to update in bulk
    # IMPORTANT: Only set dimension parameters (pbw, pbh, ppw) and control parameters (tp0, tp3, etc.)
    # Do NOT set tp1, tp2, bp1, bp2 - these will be computed by constraints
    params_to_update = {
        # Dimensions - Use standard values for clear visualization
        "pattern_psp": 0.05,
        "pattern_pbw": 5.0,
        "pattern_pbh": 7.5,
        "pattern_ppw": 0.5,
        # Mode and flags (mode will be set explicitly before update to ensure constraints work)
        "pattern_type": "wave",
        "pattern_twist": False,
        "pattern_symmetry": True,
        # Control parameters (only these, others are computed)
        "pattern_tp0": layer_params.get("pattern_tp0", 0.0),
        "pattern_tp3": layer_params.get("pattern_tp3", 1.5),
        "pattern_tnn": layer_params.get("pattern_tnn", 2.0),
        "pattern_tmm": layer_params.get("pattern_tmm", 2.0),
        "pattern_bp0": layer_params.get("pattern_bp0", 0.0),
        "pattern_bp3": layer_params.get("pattern_bp3", 1.5),
        "pattern_bnn": layer_params.get("pattern_bnn", 2.0),
        "pattern_bmm": layer_params.get("pattern_bmm", 2.0),
    }

    print(
        f"Config values: pbw={params_to_update['pattern_pbw']}, pbh={params_to_update['pattern_pbh']}, "
        f"tp0={params_to_update['pattern_tp0']}, tp3={params_to_update['pattern_tp3']}"
    )

    # Test 1: Straight mode (use actual config values with constraints)
    print("\n" + "=" * 70)
    print("Visualizing STRAIGHT mode")
    print("=" * 70)

    # Apply straight mode constraints - SET MODE FIRST to ensure consistent state
    pparams.set("pattern_mode", "straight", emit=False)
    pparams.update_bulk(params_to_update, emit=False)
    straight_params = pparams.snapshot()

    print(f"Constrained params (straight):")
    print(
        f"  Top:    tp0={straight_params['pattern_tp0']:.3f}, tp1={straight_params['pattern_tp1']:.3f}, "
        f"tp2={straight_params['pattern_tp2']:.3f}, tp3={straight_params['pattern_tp3']:.3f}"
    )
    print(
        f"  Bottom: bp0={straight_params['pattern_bp0']:.3f}, bp1={straight_params['pattern_bp1']:.3f}, "
        f"bp2={straight_params['pattern_bp2']:.3f}, bp3={straight_params['pattern_bp3']:.3f}"
    )
    print(
        f"  Sum check: tp0+tp1={straight_params['pattern_tp0']+straight_params['pattern_tp1']:.3f} (should be pbw/2={straight_params['pattern_pbw']/2:.3f})"
    )
    print(
        f"  Sum check: tp2+tp3={straight_params['pattern_tp2']+straight_params['pattern_tp3']:.3f} (should be pbh/2={straight_params['pattern_pbh']/2:.3f})"
    )

    visualize_width_measurement(straight_params, num_arrows=20)

    # Test 2: Superellipse mode (use actual config values with constraints)
    print("\n" + "=" * 70)
    print("Visualizing SUPERELLIPSE mode")
    print("=" * 70)

    # Apply superelliptic mode constraints - SET MODE FIRST
    # This prevents constraints from resetting tnn/tmm to 2.0 (straight defaults) when setting those values
    pparams.set("pattern_mode", "superelliptic", emit=False)
    pparams.update_bulk(params_to_update, emit=False)
    superellipse_params = pparams.snapshot()
    print(
        f"  tmm={superellipse_params.get('pattern_tmm', 'NOT SET'):.3f}, "
        f"bmm={superellipse_params.get('pattern_bmm', 'NOT SET'):.3f}, "
        f"tnn={superellipse_params.get('pattern_tnn', 'NOT SET'):.3f}, "
        f"bnn={superellipse_params.get('pattern_bnn', 'NOT SET'):.3f}"
    )
    print(
        f"  Sum check: tp0+tp1={superellipse_params['pattern_tp0']+superellipse_params['pattern_tp1']:.3f} (should be pbw/2={superellipse_params['pattern_pbw']/2:.3f})"
    )
    print(
        f"  Sum check: tp2+tp3={superellipse_params['pattern_tp2']+superellipse_params['pattern_tp3']:.3f} (should be pbh/2={superellipse_params['pattern_pbh']/2:.3f})"
    )
    print(
        f"  Superelliptic check: tp1==tp2? {superellipse_params['pattern_tp1']:.3f}=={superellipse_params['pattern_tp2']:.3f}"
    )

    visualize_width_measurement(superellipse_params, num_arrows=20)

    print("\n" + "=" * 70)
    print("WAITING FOR USER TO CLOSE PLOTS...")
    print("=" * 70)
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
