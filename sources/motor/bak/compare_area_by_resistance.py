"""
Compare convex hull area between superellipse and straight patterns at equal resistance.

This script generates area curves for both superellipse and straight patterns
while keeping the electrical resistance constant.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path (sources/motor/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern import Pattern
from settings import layers_a as layers


def find_params_for_target_resistance(
    base_params: dict,
    mode: str,
    target_resistance: float,
    tolerance: float = 1e-5,  # tighter tolerance for resistance
    max_iterations: int = 100
) -> dict:
    """
    Find pattern parameters that achieve a target resistance.

    Args:
        base_params: Base pattern parameters
        mode: "straight" or "superelliptic"
        target_resistance: Target resistance in Ohms
        tolerance: Acceptable resistance difference
        max_iterations: Maximum number of iterations

    Returns:
        dict: Pattern parameters that achieve target resistance, or None if not found
    """
    params = base_params.copy()
    params["pattern_mode"] = mode

    layer_pbh = params["layer_pbh"]
    layer_ppw = params["layer_ppw"]
    layer_pbw = params.get("layer_pbw", params.get("pattern_pbw", 6.0))

    # Ensure all required pattern parameters are set
    if "pattern_tp1" not in params or params["pattern_tp1"] is None:
        params["pattern_tp1"] = layer_pbw / 2.0
    if "pattern_bp1" not in params or params["pattern_bp1"] is None:
        params["pattern_bp1"] = layer_pbw / 2.0

    # Ensure tp0/bp0 are set (starting point)
    if "pattern_tp0" not in params:
        params["pattern_tp0"] = 0.0
    if "pattern_bp0" not in params:
        params["pattern_bp0"] = 0.0

    # Ensure nn parameters are set
    if "pattern_tnn" not in params:
        params["pattern_tnn"] = 2.0
    if "pattern_bnn" not in params:
        params["pattern_bnn"] = 2.0

    if mode == "straight":
        # For straight mode, vary tp3 (and bp3 for symmetry)
        # Note: larger tp3 -> longer path -> higher resistance (direct relationship)
        min_val = 0.1
        max_val = layer_pbh / 2.0 - layer_ppw

        # Binary search for tp3 value that gives target resistance
        low, high = min_val, max_val

        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            params["pattern_tp3"] = mid
            params["pattern_bp3"] = mid

            # Generate pattern and check resistance
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

            resistance = pattern["pattern_resistance"]

            if abs(resistance - target_resistance) < tolerance:
                return params
            elif resistance < target_resistance:
                # Resistance too low, need to increase tp3 (longer path)
                low = mid
            else:
                # Resistance too high, need to decrease tp3
                high = mid

        return None

    elif mode == "superelliptic":
        # For superellipse, vary tmm (and bmm for symmetry)
        
        min_mm = 0.2
        max_mm = 6.0

        # Check endpoints to determine relationship direction
        # Low mm (0.2)
        params["pattern_tmm"] = min_mm
        params["pattern_bmm"] = min_mm
        config = {"layer": params}
        p_min = Pattern.GetPattern(preConfig=None, currentConfig=config, nextConfig=config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9)
        r_at_min = p_min["pattern_resistance"]

        # High mm (6.0)
        params["pattern_tmm"] = max_mm
        params["pattern_bmm"] = max_mm
        config = {"layer": params}
        p_max = Pattern.GetPattern(preConfig=None, currentConfig=config, nextConfig=config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9)
        r_at_max = p_max["pattern_resistance"]

        increasing = r_at_max > r_at_min

        # Binary search
        low, high = min_mm, max_mm

        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            params["pattern_tmm"] = mid
            params["pattern_bmm"] = mid

            # Generate pattern and check resistance
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

            resistance = pattern["pattern_resistance"]

            if abs(resistance - target_resistance) < tolerance:
                return params
            
            if increasing:
                if resistance < target_resistance:
                    low = mid
                else:
                    high = mid
            else:
                if resistance < target_resistance:
                    high = mid
                else:
                    low = mid

        return None

    return None


def generate_comparison_data(base_params: dict, num_points: int = 20):
    """
    Generate area comparison data for straight and superellipse modes
    at equal resistances.

    Args:
        base_params: Base pattern parameters
        num_points: Number of data points to generate

    Returns:
        tuple: (resistances, straight_areas, superellipse_areas)
    """
    layer_pbh = base_params["layer_pbh"]
    layer_ppw = base_params["layer_ppw"]
    layer_pbw = base_params["layer_pbw"]

    # Determine the resistance range for straight mode
    straight_params = base_params.copy()
    straight_params["pattern_mode"] = "straight"
    straight_params["pattern_tp1"] = layer_pbw / 2.0
    straight_params["pattern_bp1"] = layer_pbw / 2.0

    # Min resistance: tp3 = 0.1
    min_tp3 = 0.1
    straight_params["pattern_tp3"] = min_tp3
    straight_params["pattern_bp3"] = min_tp3
    config = {"layer": straight_params}
    pattern_min_straight = Pattern.GetPattern(
        preConfig=None, currentConfig=config, nextConfig=config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9
    )
    straight_min_r = pattern_min_straight["pattern_resistance"]

    # Max resistance: tp3 = pbh/2 - ppw
    max_tp3 = layer_pbh / 2.0 - layer_ppw
    straight_params["pattern_tp3"] = max_tp3
    straight_params["pattern_bp3"] = max_tp3
    config = {"layer": straight_params}
    pattern_max_straight = Pattern.GetPattern(
        preConfig=None, currentConfig=config, nextConfig=config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9
    )
    straight_max_r = pattern_max_straight["pattern_resistance"]

    # Determine the resistance range for superellipse mode
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

    # Check range with mm from 0.2 to 6.0
    superellipse_params["pattern_tmm"] = 0.2
    superellipse_params["pattern_bmm"] = 0.2
    config = {"layer": superellipse_params}
    p1 = Pattern.GetPattern(preConfig=None, currentConfig=config, nextConfig=config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9)
    r1 = p1["pattern_resistance"]

    superellipse_params["pattern_tmm"] = 6.0
    superellipse_params["pattern_bmm"] = 6.0
    config = {"layer": superellipse_params}
    p2 = Pattern.GetPattern(preConfig=None, currentConfig=config, nextConfig=config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9)
    r2 = p2["pattern_resistance"]

    super_min_r = min(r1, r2)
    super_max_r = max(r1, r2)

    # Find overlapping resistance range
    min_r = max(straight_min_r, super_min_r)
    max_r = min(straight_max_r, super_max_r)

    print(f"Straight mode R range: {straight_min_r*1000:.4f} to {straight_max_r*1000:.4f} mΩ")
    print(f"Superellipse mode R range: {super_min_r*1000:.4f} to {super_max_r*1000:.4f} mΩ")
    print(f"Overlapping R range: {min_r*1000:.4f} to {max_r*1000:.4f} mΩ")

    # Generate target resistances
    # Use full range including the left side as requested
    target_rs = np.linspace(min_r, max_r, num_points + 20)

    resistances = []
    straight_areas = []
    superellipse_areas = []

    for target_r in target_rs:
        print(f"\nTarget R: {target_r*1000:.4f} mΩ")

        # Find straight parameters for this resistance
        straight_params = find_params_for_target_resistance(
            base_params, "straight", target_r
        )

        if straight_params is None:
            print(f"  Could not find straight params for R {target_r*1000:.4f}")
            continue

        # Find superellipse parameters for this resistance
        superellipse_params = find_params_for_target_resistance(
            base_params, "superelliptic", target_r
        )

        if superellipse_params is None:
            print(f"  Could not find superellipse params for R {target_r*1000:.4f}")
            continue

        # Generate patterns and collect area
        straight_config = {"layer": straight_params}
        straight_pattern = Pattern.GetPattern(
            preConfig=None, currentConfig=straight_config, nextConfig=straight_config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9
        )

        superellipse_config = {"layer": superellipse_params}
        superellipse_pattern = Pattern.GetPattern(
            preConfig=None, currentConfig=superellipse_config, nextConfig=superellipse_config, side="front", layer="mid", layerIndex=0, patternIndex=4, patternCount=9
        )

        straight_area = straight_pattern["convexhull_area"]
        superellipse_area = superellipse_pattern["convexhull_area"]
        
        print(f"  Straight: R={straight_pattern['pattern_resistance']*1000:.4f} mΩ, Area={straight_area:.4f} mm², tp3={straight_params['pattern_tp3']:.4f}")
        print(f"  Superellipse: R={superellipse_pattern['pattern_resistance']*1000:.4f} mΩ, Area={superellipse_area:.4f} mm², m={superellipse_params['pattern_tmm']:.4f}")

        resistances.append(target_r * 1000) # Store in mΩ
        straight_areas.append(straight_area)
        superellipse_areas.append(superellipse_area)

    return np.array(resistances), np.array(straight_areas), np.array(superellipse_areas)


def plot_area_comparison(resistances, straight_areas, superellipse_areas):
    """
    Plot area comparison between straight and superellipse modes.

    Args:
        resistances: Array of resistances (mΩ)
        straight_areas: Array of straight mode areas
        superellipse_areas: Array of superellipse mode areas
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both curves
    ax.plot(resistances, straight_areas, 'b-o', linewidth=2, markersize=6, label='Straight', alpha=0.8)
    ax.plot(resistances, superellipse_areas, 'r-s', linewidth=2, markersize=6, label='Superellipse', alpha=0.8)

    # Calculate and plot the difference
    ax2 = ax.twinx()
    area_diff = superellipse_areas - straight_areas
    percentage_diff = (area_diff / straight_areas) * 100
    ax2.plot(resistances, percentage_diff, 'g--', linewidth=1.5, alpha=0.6, label='Difference (%)')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Labels and title
    ax.set_xlabel('Resistance (mΩ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Convex Hull Area (mm²)', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel('Difference (%)', fontsize=12, fontweight='bold', color='green')

    ax.set_title('Area Comparison: Superellipse vs Straight at Equal Resistance',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)

    # Add summary statistics
    avg_straight = np.mean(straight_areas)
    avg_superellipse = np.mean(superellipse_areas)
    avg_diff_pct = np.mean(percentage_diff)

    stats_text = f'Average Areas:\n'
    stats_text += f'Straight: {avg_straight:.4f} mm²\n'
    stats_text += f'Superellipse: {avg_superellipse:.4f} mm²\n'
    stats_text += f'Avg Difference: {avg_diff_pct:.2f}%'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent.parent.parent / 'area_comparison_by_resistance.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("Area Comparison: Superellipse vs Straight at Equal Resistance")
    print("=" * 70)

    # Load base configuration from settings
    # Use first normal layer (index 0 or 1 depending on whether there's a start layer)
    layers_list = layers["layers"]
    normal_layer_config = None
    for layer_config in layers_list:
        if layer_config.get("type") == "normal":
            normal_layer_config = layer_config
            break

    # Fallback to first layer if no normal layer found
    if normal_layer_config is None:
        normal_layer_config = layers_list[0]

    global_settings = layers["global"]

    # Create base params by merging layer config with global settings
    base_layer_params = normal_layer_config["layer"].copy()

    # Add global settings with proper parameter names
    base_layer_params["pattern_psp"] = global_settings["layer_psp"]
    base_layer_params["pattern_mode"] = global_settings["layer_pmd"]
    base_layer_params["layer_psp"] = global_settings["layer_psp"]
    base_layer_params["layer_pmd"] = global_settings["layer_pmd"]
    base_layer_params["layer_ptc"] = global_settings.get("layer_ptc", 0.047)

    # Copy layer-specific parameters to pattern parameters
    base_layer_params["pattern_pbw"] = base_layer_params["layer_pbw"]
    base_layer_params["pattern_pbh"] = base_layer_params["layer_pbh"]
    base_layer_params["pattern_ppw"] = base_layer_params["layer_ppw"]

    # Copy pattern parameters from layer config (they may already exist)
    base_layer_params["pattern_twist"] = base_layer_params.get("pattern_twist", False)
    base_layer_params["pattern_symmetry"] = base_layer_params.get("pattern_symmetry", True)

    # Set pattern type from global settings
    base_layer_params["pattern_type"] = global_settings.get("layer_type", "wave")

    # Set common parameters for fair comparison
    layer_pbw = base_layer_params["layer_pbw"]

    # Use existing tp1/bp1 if present, otherwise calculate from pbw
    if "pattern_tp1" not in base_layer_params:
        base_layer_params["pattern_tp1"] = layer_pbw / 2.0
    if "pattern_bp1" not in base_layer_params:
        base_layer_params["pattern_bp1"] = layer_pbw / 2.0

    # Use existing nn values if present
    if "pattern_tnn" not in base_layer_params:
        base_layer_params["pattern_tnn"] = 2.0
    if "pattern_bnn" not in base_layer_params:
        base_layer_params["pattern_bnn"] = 2.0

    # Ensure tp0/bp0 are set
    if "pattern_tp0" not in base_layer_params:
        base_layer_params["pattern_tp0"] = 0.0
    if "pattern_bp0" not in base_layer_params:
        base_layer_params["pattern_bp0"] = 0.0

    print(f"\nBase parameters:")
    print(f"  Bounding box: {base_layer_params['layer_pbw']:.2f} × {base_layer_params['layer_pbh']:.2f} mm")
    print(f"  Pattern width: {base_layer_params['layer_ppw']:.3f} mm")
    print(f"  Spacing: {base_layer_params['pattern_psp']:.3f} mm")

    # Generate comparison data
    print("\nGenerating comparison data...")
    # Using 30 points as requested in previous turn, but skipping first 5
    resistances, straight_areas, superellipse_areas = generate_comparison_data(
        base_layer_params, num_points=30
    )

    # Plot results
    print("\nPlotting results...")
    plot_area_comparison(resistances, straight_areas, superellipse_areas)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Number of data points: {len(resistances)}")
    print(f"Resistance range: {min(resistances):.4f} to {max(resistances):.4f} mΩ")
    print(f"\nStraight mode:")
    print(f"  Area range: {min(straight_areas):.4f} to {max(straight_areas):.4f} mm²")
    print(f"  Average area: {np.mean(straight_areas):.4f} mm²")
    print(f"\nSuperellipse mode:")
    print(f"  Area range: {min(superellipse_areas):.4f} to {max(superellipse_areas):.4f} mm²")
    print(f"  Average area: {np.mean(superellipse_areas):.4f} mm²")

    area_diff = superellipse_areas - straight_areas
    percentage_diff = (area_diff / straight_areas) * 100
    print(f"\nDifference (Superellipse - Straight):")
    print(f"  Average: {np.mean(area_diff):.4f} mm² ({np.mean(percentage_diff):.2f}%)")
    print(f"  Range: {min(percentage_diff):.2f}% to {max(percentage_diff):.2f}%")

    if np.mean(percentage_diff) < 0:
        print(f"\nConclusion: Superellipse mode has {abs(np.mean(percentage_diff)):.2f}% SMALLER average area")
    else:
        print(f"\nConclusion: Superellipse mode has {np.mean(percentage_diff):.2f}% LARGER average area")


if __name__ == "__main__":
    main()
