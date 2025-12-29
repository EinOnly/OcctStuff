"""
Compare S/R ratio (Area/Resistance) between superellipse and straight patterns.

This script compares the S/R efficiency coefficient for both superellipse and
straight patterns across different area ranges, showing which pattern provides
better area-to-resistance ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path (sources/motor/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pattern import Pattern
from settings import layers_b as layers

# Thread-safe print lock
print_lock = Lock()


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

    # DO NOT modify tp0/bp0/tp1/bp1/nn/mm parameters that already exist in the config
    # Only set pattern_mode to control which mode we're testing

    if mode == "straight":
        # For straight mode, vary tp3 (and bp3 for symmetry)
        # Note: larger tp3 -> larger area (direct relationship)
        min_val = 0.1  # Allow smaller amplitude
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
        min_mm = 0.2  # Expanded range: 0.2 to 6.0
        max_mm = 6.0

        # DO NOT modify tp3/bp3/tp2/bp2/tnn/bnn - use values from config
        # These parameters are already defined in the layer configurations

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


def process_single_target_area(target_area: float, base_params: dict, index: int, total: int):
    """
    Process a single target area to find parameters and calculate S/R ratios.

    Args:
        target_area: Target convex hull area
        base_params: Base pattern parameters
        index: Current index for progress tracking
        total: Total number of areas to process

    Returns:
        dict: Results containing area, resistances, and S/R ratios, or None if failed
    """
    with print_lock:
        print(f"[{index+1}/{total}] Processing target area: {target_area:.4f} mm²")

    # Find straight parameters for this area
    straight_params = find_params_for_target_area(
        base_params, "straight", target_area
    )

    if straight_params is None:
        with print_lock:
            print(f"  [{index+1}/{total}] Could not find straight params for area {target_area:.4f}")
        return None

    # Find superellipse parameters for this area
    superellipse_params = find_params_for_target_area(
        base_params, "superelliptic", target_area
    )

    if superellipse_params is None:
        with print_lock:
            print(f"  [{index+1}/{total}] Could not find superellipse params for area {target_area:.4f}")
        return None

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

    # Calculate S/R ratios (Area/Resistance)
    straight_sr = straight_area / straight_resistance
    superellipse_sr = superellipse_area / superellipse_resistance

    with print_lock:
        print(f"  [{index+1}/{total}] Straight: area={straight_area:.4f}, R={straight_resistance:.4f} mΩ, S/R={straight_sr:.6f}, tp3={straight_params['pattern_tp3']:.4f}")
        print(f"  [{index+1}/{total}] Superellipse: area={superellipse_area:.4f}, R={superellipse_resistance:.4f} mΩ, S/R={superellipse_sr:.6f}, m={superellipse_params['pattern_tmm']:.4f}")

    return {
        'area': target_area,
        'straight_resistance': straight_resistance,
        'superellipse_resistance': superellipse_resistance,
        'straight_sr': straight_sr,
        'superellipse_sr': superellipse_sr
    }


def generate_comparison_data(base_params: dict, num_points: int = 20, max_workers: int = 4):
    """
    Generate S/R ratio comparison data for straight and superellipse modes using multithreading.

    Args:
        base_params: Base pattern parameters
        num_points: Number of data points to generate
        max_workers: Maximum number of parallel threads

    Returns:
        tuple: (areas, straight_sr_ratios, superellipse_sr_ratios,
                straight_resistances, superellipse_resistances)
    """
    layer_pbh = base_params["layer_pbh"]
    layer_ppw = base_params["layer_ppw"]
    layer_pbw = base_params["layer_pbw"]

    # Determine the area range for straight mode
    straight_params = base_params.copy()
    straight_params["pattern_mode"] = "straight"

    # DO NOT set tp1/bp1 - use values from config

    # Min area: tp3 = ppw (smallest vertical displacement)
    min_tp3 = 0.1
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

    # Use safe default values based on bounding box
    # Use existing values if present, otherwise use calculated defaults
    default_tp3 = superellipse_params.get("pattern_tp3", layer_pbh / 2.0 - layer_ppw)
    superellipse_params["pattern_tp3"] = default_tp3
    superellipse_params["pattern_bp3"] = default_tp3

    # For superelliptic mode, tp2/bp2 defaults to 0.0 (as per pattern.py defaults)
    superellipse_params["pattern_tp2"] = superellipse_params.get("pattern_tp2", 0.0)
    superellipse_params["pattern_bp2"] = superellipse_params.get("pattern_bp2", 0.0)

    # Min area: mm = 6.0 (smallest for superellipse)
    superellipse_params["pattern_tmm"] = 6.0
    superellipse_params["pattern_bmm"] = 6.0
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

    # Max area: mm = 0.2 (largest for superellipse)
    superellipse_params["pattern_tmm"] = 0.2
    superellipse_params["pattern_bmm"] = 0.2
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
    # Skip first 5 points to avoid edge cases at minimum area
    target_areas = np.linspace(min_area, max_area, num_points)[5:]

    print(f"\nProcessing {len(target_areas)} target areas using {max_workers} threads...")

    # Use ThreadPoolExecutor for parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_target_area, target_area, base_params, i, len(target_areas)): i
            for i, target_area in enumerate(target_areas)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            result = future.result()
            if result is not None:
                results.append(result)

    # Sort results by area to maintain order
    results.sort(key=lambda x: x['area'])

    # Extract data from results
    areas = np.array([r['area'] for r in results])
    straight_sr_ratios = np.array([r['straight_sr'] for r in results])
    superellipse_sr_ratios = np.array([r['superellipse_sr'] for r in results])
    straight_resistances = np.array([r['straight_resistance'] for r in results])
    superellipse_resistances = np.array([r['superellipse_resistance'] for r in results])

    return (areas, straight_sr_ratios, superellipse_sr_ratios,
            straight_resistances, superellipse_resistances)


def plot_sr_ratio_comparison(areas, straight_sr_ratios, superellipse_sr_ratios,
                             straight_resistances, superellipse_resistances):
    """
    Plot S/R ratio comparison between straight and superellipse modes.

    Args:
        areas: Array of convex hull areas
        straight_sr_ratios: Array of straight mode S/R ratios
        superellipse_sr_ratios: Array of superellipse mode S/R ratios
        straight_resistances: Array of straight mode resistances
        superellipse_resistances: Array of superellipse mode resistances
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both S/R ratio curves
    ax.plot(areas, straight_sr_ratios, 'b-o', linewidth=2, markersize=6, label='Straight S/R', alpha=0.8)
    ax.plot(areas, superellipse_sr_ratios, 'r-s', linewidth=2, markersize=6, label='Superellipse S/R', alpha=0.8)

    # Find and mark maximum S/R values
    straight_max_idx = np.argmax(straight_sr_ratios)
    superellipse_max_idx = np.argmax(superellipse_sr_ratios)

    straight_max_sr = straight_sr_ratios[straight_max_idx]
    superellipse_max_sr = superellipse_sr_ratios[superellipse_max_idx]
    straight_max_area = areas[straight_max_idx]
    superellipse_max_area = areas[superellipse_max_idx]

    # Mark maximum points with larger markers
    ax.plot(straight_max_area, straight_max_sr, 'b*', markersize=15, markeredgewidth=2,
            markeredgecolor='darkblue', label=f'Straight Max: {straight_max_sr:.6f}', zorder=5)
    ax.plot(superellipse_max_area, superellipse_max_sr, 'r*', markersize=15, markeredgewidth=2,
            markeredgecolor='darkred', label=f'Superellipse Max: {superellipse_max_sr:.6f}', zorder=5)

    # Add annotations for max values
    ax.annotate(f'Max: {straight_max_sr:.6f}\n@ {straight_max_area:.2f} mm²',
                xy=(straight_max_area, straight_max_sr),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='blue', lw=1.5),
                fontsize=9, color='darkblue', fontweight='bold')

    ax.annotate(f'Max: {superellipse_max_sr:.6f}\n@ {superellipse_max_area:.2f} mm²',
                xy=(superellipse_max_area, superellipse_max_sr),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=1.5),
                fontsize=9, color='darkred', fontweight='bold')

    # Calculate and plot the S/R ratio improvement percentage
    ax2 = ax.twinx()
    sr_improvement = ((superellipse_sr_ratios - straight_sr_ratios) / straight_sr_ratios) * 100
    ax2.plot(areas, sr_improvement, 'g--', linewidth=1.5, alpha=0.6, label='S/R Improvement (%)')
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Labels and title
    ax.set_xlabel('Convex Hull Area (mm²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('S/R Ratio (mm²/mΩ)', fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel('S/R Improvement (%)', fontsize=12, fontweight='bold', color='green')

    ax.set_title('S/R Ratio Comparison: Superellipse vs Straight at Equal Areas',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)

    # Add summary statistics
    avg_straight_sr = np.mean(straight_sr_ratios)
    avg_superellipse_sr = np.mean(superellipse_sr_ratios)
    avg_sr_improvement = np.mean(sr_improvement)

    stats_text = f'Average S/R Ratios:\n'
    stats_text += f'Straight: {avg_straight_sr:.6f} mm²/mΩ\n'
    stats_text += f'Superellipse: {avg_superellipse_sr:.6f} mm²/mΩ\n'
    stats_text += f'Avg Improvement: {avg_sr_improvement:.2f}%\n\n'
    stats_text += f'Average Resistances:\n'
    stats_text += f'Straight: {np.mean(straight_resistances):.4f} mΩ\n'
    stats_text += f'Superellipse: {np.mean(superellipse_resistances):.4f} mΩ'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_file = Path(__file__).parent.parent.parent / 'sr_ratio_comparison_by_area.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("S/R Ratio Comparison: Superellipse vs Straight at Equal Areas")
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

    # DO NOT set tp1/bp1 or override nn values
    # The original configs already have all necessary pattern parameters defined
    # Adding or overriding them will change the pattern behavior

    print(f"\nBase parameters:")
    print(f"  Bounding box: {base_layer_params['layer_pbw']:.2f} × {base_layer_params['layer_pbh']:.2f} mm")
    print(f"  Pattern width: {base_layer_params['layer_ppw']:.3f} mm")
    print(f"  Spacing: {base_layer_params['pattern_psp']:.3f} mm")

    # Generate comparison data with multithreading
    print("\nGenerating comparison data...")
    areas, straight_sr_ratios, superellipse_sr_ratios, straight_resistances, superellipse_resistances = generate_comparison_data(
        base_layer_params, num_points=30, max_workers=6
    )

    # Plot results
    print("\nPlotting results...")
    plot_sr_ratio_comparison(areas, straight_sr_ratios, superellipse_sr_ratios,
                             straight_resistances, superellipse_resistances)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Number of data points: {len(areas)}")
    print(f"Area range: {min(areas):.4f} to {max(areas):.4f} mm²")

    # Find max S/R values and their corresponding areas
    straight_max_idx = np.argmax(straight_sr_ratios)
    superellipse_max_idx = np.argmax(superellipse_sr_ratios)

    print(f"\nStraight mode:")
    print(f"  S/R ratio range: {min(straight_sr_ratios):.6f} to {max(straight_sr_ratios):.6f} mm²/mΩ")
    print(f"  Average S/R ratio: {np.mean(straight_sr_ratios):.6f} mm²/mΩ")
    print(f"  Maximum S/R ratio: {straight_sr_ratios[straight_max_idx]:.6f} mm²/mΩ @ {areas[straight_max_idx]:.4f} mm²")
    print(f"  Resistance range: {min(straight_resistances):.4f} to {max(straight_resistances):.4f} mΩ")
    print(f"  Average resistance: {np.mean(straight_resistances):.4f} mΩ")

    print(f"\nSuperellipse mode:")
    print(f"  S/R ratio range: {min(superellipse_sr_ratios):.6f} to {max(superellipse_sr_ratios):.6f} mm²/mΩ")
    print(f"  Average S/R ratio: {np.mean(superellipse_sr_ratios):.6f} mm²/mΩ")
    print(f"  Maximum S/R ratio: {superellipse_sr_ratios[superellipse_max_idx]:.6f} mm²/mΩ @ {areas[superellipse_max_idx]:.4f} mm²")
    print(f"  Resistance range: {min(superellipse_resistances):.4f} to {max(superellipse_resistances):.4f} mΩ")
    print(f"  Average resistance: {np.mean(superellipse_resistances):.4f} mΩ")

    sr_improvement = ((superellipse_sr_ratios - straight_sr_ratios) / straight_sr_ratios) * 100
    resistance_diff = superellipse_resistances - straight_resistances
    resistance_pct_diff = (resistance_diff / straight_resistances) * 100

    print(f"\nS/R Ratio Improvement (Superellipse vs Straight):")
    print(f"  Average: {np.mean(sr_improvement):.2f}%")
    print(f"  Range: {min(sr_improvement):.2f}% to {max(sr_improvement):.2f}%")

    print(f"\nResistance Difference (Superellipse - Straight):")
    print(f"  Average: {np.mean(resistance_diff):.4f} mΩ ({np.mean(resistance_pct_diff):.2f}%)")
    print(f"  Range: {min(resistance_pct_diff):.2f}% to {max(resistance_pct_diff):.2f}%")

    # Compare max S/R ratios
    max_sr_diff_pct = ((superellipse_sr_ratios[superellipse_max_idx] - straight_sr_ratios[straight_max_idx]) /
                       straight_sr_ratios[straight_max_idx]) * 100

    print(f"\nMaximum S/R Comparison:")
    print(f"  Straight max: {straight_sr_ratios[straight_max_idx]:.6f} mm²/mΩ")
    print(f"  Superellipse max: {superellipse_sr_ratios[superellipse_max_idx]:.6f} mm²/mΩ")
    print(f"  Difference: {max_sr_diff_pct:+.2f}%")

    if np.mean(sr_improvement) > 0:
        print(f"\nConclusion: Superellipse mode has {np.mean(sr_improvement):.2f}% BETTER S/R ratio (higher efficiency)")
    else:
        print(f"\nConclusion: Superellipse mode has {abs(np.mean(sr_improvement)):.2f}% WORSE S/R ratio (lower efficiency)")


if __name__ == "__main__":
    main()
