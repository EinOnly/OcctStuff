"""
Compare resistance between superellipse and straight patterns at equal areas.

This script generates resistance curves for both superellipse and straight patterns
while keeping the convex hull area constant.
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


def find_params_for_target_area(
    base_params: dict,
    mode: str,
    target_area: float,
    tolerance: float = 0.1,
    max_iterations: int = 100
) -> dict:
    """
    Find pattern parameters that achieve a target convex hull area.

    Args:
        base_params: Base pattern parameters
        mode: "straight" or "superelliptic"
        target_area: Target convex hull area in mm²
        tolerance: Acceptable area difference
        max_iterations: Maximum number of iterations

    Returns:
        dict: Pattern parameters that achieve target area, or None if not found
    """
    params = base_params.copy()
    params["pattern_mode"] = mode

    layer_pbh = params["layer_pbh"]
    layer_ppw = params["layer_ppw"]

    if mode == "straight":
        # For straight mode, vary tp3 (and bp3 for symmetry)
        # Note: larger tp3 -> larger area (direct relationship)
        min_val = layer_ppw
        max_val = layer_pbh / 2.0 - layer_ppw

        # Binary search for tp3 value that gives target area
        low, high = min_val, max_val

        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            params["pattern_tp3"] = mid
            params["pattern_bp3"] = mid

            # Generate pattern and check area
            config = {"layer": params}
            pattern = Pattern.GetPattern(
                preConfig=None,
                currentConfig=config,
                nextConfig=config,
                side="front",
                layer="mid",
                layerIndex=0,
                patternIndex=4,
                patternCount=9
            )

            area = pattern["convexhull_area"]

            if abs(area - target_area) < tolerance:
                return params
            elif area < target_area:
                # Area too small, need to increase tp3 (direct relationship)
                low = mid
            else:
                # Area too large, need to decrease tp3
                high = mid

        return None

    elif mode == "superelliptic":
        # For superellipse, vary tmm (and bmm for symmetry)
        # Note: larger mm -> smaller area (inverse relationship)
        min_mm = 0.5
        max_mm = 2.0

        # Binary search for mm value that gives target area
        low, high = min_mm, max_mm

        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            params["pattern_tmm"] = mid
            params["pattern_bmm"] = mid

            # Generate pattern and check area
            config = {"layer": params}
            pattern = Pattern.GetPattern(
                preConfig=None,
                currentConfig=config,
                nextConfig=config,
                side="front",
                layer="mid",
                layerIndex=0,
                patternIndex=4,
                patternCount=9
            )

            area = pattern["convexhull_area"]

            if abs(area - target_area) < tolerance:
                return params
            elif area < target_area:
                # Area too small, need to decrease mm (inverse relationship)
                high = mid
            else:
                # Area too large, need to increase mm
                low = mid

        return None

    return None


def generate_comparison_data(base_params: dict, num_points: int = 20):
    """
    Generate resistance comparison data for straight and superellipse modes
    at equal areas.

    Args:
        base_params: Base pattern parameters
        num_points: Number of data points to generate

    Returns:
        tuple: (areas, straight_resistances, superellipse_resistances)
    """
    layer_pbh = base_params["layer_pbh"]
    layer_ppw = base_params["layer_ppw"]
    layer_pbw = base_params["layer_pbw"]

    # Determine the area range for straight mode
    straight_params = base_params.copy()
    straight_params["pattern_mode"] = "straight"
    straight_params["pattern_tp1"] = layer_pbw / 2.0
    straight_params["pattern_bp1"] = layer_pbw / 2.0

    # Min area: tp3 = ppw (smallest vertical displacement)
    min_tp3 = layer_ppw
    straight_params["pattern_tp3"] = min_tp3
    straight_params["pattern_bp3"] = min_tp3
    config = {"layer": straight_params}
    pattern_min_straight = Pattern.GetPattern(
        preConfig=None,
        currentConfig=config,
        nextConfig=config,
        side="front",
        layer="mid",
        layerIndex=0,
        patternIndex=4,
        patternCount=9
    )
    straight_min_area = pattern_min_straight["convexhull_area"]

    # Max area: tp3 = pbh/2 - ppw (largest vertical displacement)
    max_tp3 = layer_pbh / 2.0 - layer_ppw
    straight_params["pattern_tp3"] = max_tp3
    straight_params["pattern_bp3"] = max_tp3
    config = {"layer": straight_params}
    pattern_max_straight = Pattern.GetPattern(
        preConfig=None,
        currentConfig=config,
        nextConfig=config,
        side="front",
        layer="mid",
        layerIndex=0,
        patternIndex=4,
        patternCount=9
    )
    straight_max_area = pattern_max_straight["convexhull_area"]

    # Determine the area range for superellipse mode
    superellipse_params = base_params.copy()
    superellipse_params["pattern_mode"] = "superelliptic"
    superellipse_params["pattern_tp1"] = layer_pbw / 2.0
    superellipse_params["pattern_bp1"] = layer_pbw / 2.0
    superellipse_params["pattern_tnn"] = 2.0
    superellipse_params["pattern_bnn"] = 2.0
    superellipse_params["pattern_tp3"] = 2.0
    superellipse_params["pattern_bp3"] = 2.0
    superellipse_params["pattern_tp2"] = 2.0
    superellipse_params["pattern_bp2"] = 2.0

    # Min area: mm = 2.0 (smallest for superellipse)
    superellipse_params["pattern_tmm"] = 2.0
    superellipse_params["pattern_bmm"] = 2.0
    config = {"layer": superellipse_params}
    pattern_min_super = Pattern.GetPattern(
        preConfig=None,
        currentConfig=config,
        nextConfig=config,
        side="front",
        layer="mid",
        layerIndex=0,
        patternIndex=4,
        patternCount=9
    )
    super_min_area = pattern_min_super["convexhull_area"]

    # Max area: mm = 0.5 (largest for superellipse)
    superellipse_params["pattern_tmm"] = 0.5
    superellipse_params["pattern_bmm"] = 0.5
    config = {"layer": superellipse_params}
    pattern_max_super = Pattern.GetPattern(
        preConfig=None,
        currentConfig=config,
        nextConfig=config,
        side="front",
        layer="mid",
        layerIndex=0,
        patternIndex=4,
        patternCount=9
    )
    super_max_area = pattern_max_super["convexhull_area"]

    # Find overlapping area range
    min_area = max(straight_min_area, super_min_area)
    max_area = min(straight_max_area, super_max_area)

    print(f"Straight mode area range: {straight_min_area:.4f} to {straight_max_area:.4f} mm²")
    print(f"Superellipse mode area range: {super_min_area:.4f} to {super_max_area:.4f} mm²")
    print(f"Overlapping area range: {min_area:.4f} to {max_area:.4f} mm²")

    # Generate target areas
    target_areas = np.linspace(min_area, max_area, num_points)

    areas = []
    straight_resistances = []
    superellipse_resistances = []

    for target_area in target_areas:
        print(f"\nTarget area: {target_area:.4f} mm²")

        # Find straight parameters for this area
        straight_params = find_params_for_target_area(
            base_params, "straight", target_area
        )

        if straight_params is None:
            print(f"  Could not find straight params for area {target_area:.4f}")
            continue

        # Find superellipse parameters for this area
        superellipse_params = find_params_for_target_area(
            base_params, "superelliptic", target_area
        )

        if superellipse_params is None:
            print(f"  Could not find superellipse params for area {target_area:.4f}")
            continue

        # Generate patterns and collect resistance
        straight_config = {"layer": straight_params}
        straight_pattern = Pattern.GetPattern(
            preConfig=None,
            currentConfig=straight_config,
            nextConfig=straight_config,
            side="front",
            layer="mid",
            layerIndex=0,
            patternIndex=4,
            patternCount=9
        )

        superellipse_config = {"layer": superellipse_params}
        superellipse_pattern = Pattern.GetPattern(
            preConfig=None,
            currentConfig=superellipse_config,
            nextConfig=superellipse_config,
            side="front",
            layer="mid",
            layerIndex=0,
            patternIndex=4,
            patternCount=9
        )

        straight_area = straight_pattern["convexhull_area"]
        superellipse_area = superellipse_pattern["convexhull_area"]
        straight_resistance = straight_pattern["pattern_resistance"] * 1000  # Convert to mΩ
        superellipse_resistance = superellipse_pattern["pattern_resistance"] * 1000

        print(f"  Straight: area={straight_area:.4f}, R={straight_resistance:.4f} mΩ, tp3={straight_params['pattern_tp3']:.4f}")
        print(f"  Superellipse: area={superellipse_area:.4f}, R={superellipse_resistance:.4f} mΩ, m={superellipse_params['pattern_tmm']:.4f}")

        areas.append(target_area)
        straight_resistances.append(straight_resistance)
        superellipse_resistances.append(superellipse_resistance)

    return np.array(areas), np.array(straight_resistances), np.array(superellipse_resistances)


def plot_resistance_comparison(areas, straight_resistances, superellipse_resistances):
    """
    Plot resistance comparison between straight and superellipse modes.

    Args:
        areas: Array of convex hull areas
        straight_resistances: Array of straight mode resistances
        superellipse_resistances: Array of superellipse mode resistances
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both curves
    ax.plot(areas, straight_resistances, 'b-o', linewidth=2, markersize=6, label='Straight', alpha=0.8)
    ax.plot(areas, superellipse_resistances, 'r-s', linewidth=2, markersize=6, label='Superellipse', alpha=0.8)

    # Calculate and plot the difference
    ax2 = ax.twinx()
    resistance_diff = superellipse_resistances - straight_resistances
    percentage_diff = (resistance_diff / straight_resistances) * 100
    ax2.plot(areas, percentage_diff, 'g--', linewidth=1.5, alpha=0.6, label='Difference (%)')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Labels and title
    ax.set_xlabel('Convex Hull Area (mm²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Resistance (mΩ)', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel('Difference (%)', fontsize=12, fontweight='bold', color='green')

    ax.set_title('Resistance Comparison: Superellipse vs Straight at Equal Areas',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)

    # Add summary statistics
    avg_straight = np.mean(straight_resistances)
    avg_superellipse = np.mean(superellipse_resistances)
    avg_diff_pct = np.mean(percentage_diff)

    stats_text = f'Average Resistances:\n'
    stats_text += f'Straight: {avg_straight:.4f} mΩ\n'
    stats_text += f'Superellipse: {avg_superellipse:.4f} mΩ\n'
    stats_text += f'Avg Difference: {avg_diff_pct:.2f}%'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent.parent.parent / 'resistance_comparison_by_area.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Resistance Comparison: Superellipse vs Straight at Equal Areas")
    print("=" * 70)

    # Load base configuration from settings
    normal_layer_config = layers_c["layers"][1]
    global_settings = layers_c["global"]

    # Create base params by merging layer config with global settings
    base_layer_params = normal_layer_config["layer"].copy()

    # Add global settings
    base_layer_params["pattern_psp"] = global_settings["layer_psp"]
    base_layer_params["pattern_mode"] = global_settings["layer_pmd"]

    # Add required mappings
    base_layer_params["pattern_pbw"] = base_layer_params["layer_pbw"]
    base_layer_params["pattern_pbh"] = base_layer_params["layer_pbh"]
    base_layer_params["pattern_ppw"] = base_layer_params["layer_ppw"]
    base_layer_params["pattern_twist"] = False
    base_layer_params["pattern_type"] = "wave"

    # Set common parameters for fair comparison
    layer_pbw = base_layer_params["layer_pbw"]
    base_layer_params["pattern_tp1"] = layer_pbw / 2.0
    base_layer_params["pattern_bp1"] = layer_pbw / 2.0
    base_layer_params["pattern_tnn"] = 2.0
    base_layer_params["pattern_bnn"] = 2.0

    print(f"\nBase parameters:")
    print(f"  Bounding box: {base_layer_params['layer_pbw']:.2f} × {base_layer_params['layer_pbh']:.2f} mm")
    print(f"  Pattern width: {base_layer_params['layer_ppw']:.3f} mm")
    print(f"  Spacing: {base_layer_params['pattern_psp']:.3f} mm")

    # Generate comparison data
    print("\nGenerating comparison data...")
    areas, straight_resistances, superellipse_resistances = generate_comparison_data(
        base_layer_params, num_points=15
    )

    # Plot results
    print("\nPlotting results...")
    plot_resistance_comparison(areas, straight_resistances, superellipse_resistances)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Number of data points: {len(areas)}")
    print(f"Area range: {min(areas):.4f} to {max(areas):.4f} mm²")
    print(f"\nStraight mode:")
    print(f"  Resistance range: {min(straight_resistances):.4f} to {max(straight_resistances):.4f} mΩ")
    print(f"  Average resistance: {np.mean(straight_resistances):.4f} mΩ")
    print(f"\nSuperellipse mode:")
    print(f"  Resistance range: {min(superellipse_resistances):.4f} to {max(superellipse_resistances):.4f} mΩ")
    print(f"  Average resistance: {np.mean(superellipse_resistances):.4f} mΩ")

    resistance_diff = superellipse_resistances - straight_resistances
    percentage_diff = (resistance_diff / straight_resistances) * 100
    print(f"\nDifference (Superellipse - Straight):")
    print(f"  Average: {np.mean(resistance_diff):.4f} mΩ ({np.mean(percentage_diff):.2f}%)")
    print(f"  Range: {min(percentage_diff):.2f}% to {max(percentage_diff):.2f}%")

    if np.mean(percentage_diff) < 0:
        print(f"\nConclusion: Superellipse mode has {abs(np.mean(percentage_diff)):.2f}% LOWER average resistance")
    else:
        print(f"\nConclusion: Superellipse mode has {np.mean(percentage_diff):.2f}% HIGHER average resistance")


if __name__ == "__main__":
    main()
