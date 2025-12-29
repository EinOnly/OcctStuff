#!/usr/bin/env python3
"""
Performance analysis for motor patterns.
Compares straight mode vs superelliptic mode for a single configuration.
Plots Area vs S/R ratio performance metric as line graphs.

How to use:
1. Change the configuration by modifying line 20:
    - layers_a: 1320 Solution (4 layers, twisted)
    - layers_b: 1120 Solution (3 layers, non-twisted) [DEFAULT]
    - layers_c: 1020 Solution (4 layers, twisted)

2. Adjust sampling density by changing SAMPLE_COUNT below (default: 20)
    - Higher values = smoother curves but slower computation
    - Recommended range: 10-50

3. Run: python performance.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION - Change these to analyze different setups
# ============================================================
from settings import layers_c as selected_layers  # Change to layers_a, layers_b, or layers_c
SAMPLE_COUNT = 20  # Number of parameter samples per layer (10-50 recommended)
# ============================================================

from parameters import PParams
from pattern import Pattern


def analyze_with_mode(config_name, layers_config, mode_name, color, marker, sample_count=20):
    """
    Analyze a configuration with a specific pattern mode by varying parameters.

    Args:
        config_name: Name of the configuration
        layers_config: Layer configuration dict
        mode_name: "straight" or "superelliptic"
        color: Color for plotting
        marker: Marker style for plotting
        sample_count: Number of samples to generate for each layer

    Returns:
        dict with areas, resistances, and S/R ratios
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {config_name} - Mode: {mode_name}")
    print(f"{'='*60}")

    global_settings = layers_config["global"]
    layers_list = layers_config["layers"]

    areas = []
    resistances = []
    sr_ratios = []
    sample_labels = []

    # Only use the first layer (index 0)
    idx = 0
    layer_config = layers_list[idx]
    layer_type = layer_config.get("type", "normal")
    layer_params = layer_config["layer"]

    print(f"\nAnalyzing Layer {idx} ({layer_type}) - all patterns as mid-layer:")

    # Get base dimensions
    pbw = layer_params["layer_pbw"]
    pbh = layer_params["layer_pbh"]
    ppw = layer_params["layer_ppw"]
    psp = global_settings["layer_psp"]

    # Determine parameter to vary based on mode
    if mode_name == "straight":
        # For straight mode: vary tp3/bp3 from ppw to pbh/2 - ppw
        min_val = ppw
        max_val = pbh / 2.0 - ppw
        print(f"  Varying tp3 (and bp3) from {min_val:.3f} to {max_val:.3f} mm with {sample_count} samples")
    else:
        # For superelliptic mode: vary tmm/bmm from 0.5 to 2.0
        min_val = 0.5
        max_val = 1.9
        print(f"  Varying tmm (and bmm) from {min_val:.2f} to {max_val:.2f} with {sample_count} samples")

    # Generate sample values
    sample_values = np.linspace(min_val, max_val, sample_count)

    for sample_idx, sample_val in enumerate(sample_values):
        # Create PParams instance
        pparams = PParams()

        # Prepare base parameters - override mode
        params_to_update = {
            "pattern_mode": mode_name,  # Use specified mode
            "pattern_type": global_settings.get("layer_type", "wave"),
            "pattern_pbw": pbw,
            "pattern_pbh": pbh,
            "pattern_ppw": ppw,
            "pattern_psp": psp,
            "pattern_twist": layer_params.get("pattern_twist", False),
            "pattern_symmetry": layer_params.get("pattern_symmetry", False),
            "pattern_tp0": 0.0,
            "pattern_bp0": 0.0,
            "pattern_tnn": 2.0,
            "pattern_bnn": 2.0,
        }

        # Set the varying parameter
        if mode_name == "straight":
            params_to_update["pattern_tp3"] = sample_val
            params_to_update["pattern_bp3"] = sample_val
        else:  # superelliptic
            # Use the tp3 value from the original configuration (not middle value)
            # This ensures consistent geometry when varying tmm/bmm
            config_tp3 = layer_params.get("pattern_tp3", pbh / 4.0)
            params_to_update["pattern_tp3"] = config_tp3
            params_to_update["pattern_bp3"] = config_tp3
            params_to_update["pattern_tmm"] = sample_val
            params_to_update["pattern_bmm"] = sample_val

        # Apply constraints
        pparams.update_bulk(params_to_update, emit=False)
        constrained_params = pparams.snapshot()

        # Build pattern - always as mid-layer
        config = {"layer": constrained_params}
        pattern = Pattern.GetPattern(
            preConfig=None,
            currentConfig=config,
            nextConfig=config,
            side="front",
            layer="mid",  # Always mid-layer
            layerIndex=idx,
            patternIndex=4,
            patternCount=layer_params.get("layer_pdc", 9)
        )

        # Extract metrics
        area = pattern["convexhull_area"]  # Using convex hull area as envelope
        resistance_ohm = pattern["pattern_resistance"]  # in Ohm
        resistance_mohm = resistance_ohm * 1000  # Convert to mΩ

        # Calculate S/R ratio (mm²/mΩ)
        sr_ratio = area / resistance_mohm if resistance_mohm > 0 else 0

        areas.append(area)
        resistances.append(resistance_mohm)
        sr_ratios.append(sr_ratio)
        sample_labels.append(f"S{sample_idx}")

    print(f"  Generated {sample_count} samples")
    print(f"  Area range: {min(areas):.4f} - {max(areas):.4f} mm²")
    print(f"  S/R range: {min(sr_ratios):.4f} - {max(sr_ratios):.4f} mm²/mΩ")

    return {
        "name": f"{config_name} ({mode_name})",
        "areas": areas,
        "resistances": resistances,
        "sr_ratios": sr_ratios,
        "sample_labels": sample_labels,
        "color": color,
        "marker": marker
    }


def plot_performance(results_list, config_name=""):
    """
    Plot Area vs S/R ratio comparing straight and superelliptic modes.

    Args:
        results_list: List of result dicts from analyze_with_mode
        config_name: Name of the configuration being analyzed
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    max_points = []  # Store max points for each mode

    for results in results_list:
        # Sort points by area to create a continuous curve
        sorted_indices = np.argsort(results["areas"])
        sorted_areas = np.array(results["areas"])[sorted_indices]
        sorted_sr_ratios = np.array(results["sr_ratios"])[sorted_indices]

        # Plot as single continuous line graph with markers
        ax.plot(
            sorted_areas,
            sorted_sr_ratios,
            color=results["color"],
            marker=results["marker"],
            markersize=5,
            linewidth=2,
            alpha=0.8,
            label=results["name"],
            markerfacecolor=results["color"],
            markeredgecolor='white',
            markeredgewidth=0.5
        )

        # Find and mark maximum S/R point
        max_idx = np.argmax(results["sr_ratios"])
        max_area = results["areas"][max_idx]
        max_sr = results["sr_ratios"][max_idx]
        max_points.append({
            "name": results["name"],
            "area": max_area,
            "sr": max_sr,
            "color": results["color"]
        })

        # Mark the maximum point with a star
        ax.plot(max_area, max_sr, marker='*', markersize=20,
                color=results["color"], markeredgecolor='black',
                markeredgewidth=1.5, zorder=5)

        # Add annotation for max point
        ax.annotate(f'Max: {max_sr:.3f}',
                   xy=(max_area, max_sr),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   color=results["color"],
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor=results["color"], alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color=results["color"], lw=1.5))

    # Calculate improvement rate between modes
    # Assuming results_list[0] is straight and results_list[1] is superelliptic
    if len(max_points) == 2:
        straight_max_sr = max_points[0]["sr"]
        superelliptic_max_sr = max_points[1]["sr"]
        improvement_rate = (superelliptic_max_sr - straight_max_sr) / straight_max_sr * 100

        # Add improvement rate text to the plot (bottom-left corner)
        improvement_text = f'Improvement Rate at Max:\n(S/R_superelliptic - S/R_straight) / S/R_straight\n= ({superelliptic_max_sr:.4f} - {straight_max_sr:.4f}) / {straight_max_sr:.4f}\n= {improvement_rate:+.2f}%'
        ax.text(0.02, 0.02, improvement_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow',
                        edgecolor='black', alpha=0.7, linewidth=2),
               family='monospace')

    ax.set_xlabel('Area (mm²)', fontsize=13, fontweight='bold')
    ax.set_ylabel('S/R Ratio (mm²/mΩ)', fontsize=13, fontweight='bold')

    title = f'Performance Analysis: {config_name}\n'
    title += 'Straight vs Superelliptic Mode - Area vs S/R Ratio\n'
    title += '(Higher S/R is better - more area per unit resistance)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Add reference line for overall mean S/R
    all_sr_values = []
    for r in results_list:
        all_sr_values.extend(r["sr_ratios"])
    overall_mean = np.mean(all_sr_values)
    ax.axhline(y=overall_mean,
               color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(ax.get_xlim()[0], overall_mean, f' Mean S/R: {overall_mean:.2f}',
            verticalalignment='bottom', fontsize=9, color='gray')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'performance_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Plot saved to: {output_file}")
    print(f"{'='*60}")

    return fig


def print_summary(results_list):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    max_sr_values = []
    for results in results_list:
        max_sr = max(results['sr_ratios'])
        max_sr_values.append(max_sr)

        print(f"\n{results['name']}:")
        print(f"  Area range: {min(results['areas']):.4f} - {max(results['areas']):.4f} mm²")
        print(f"  Resistance range: {min(results['resistances']):.4f} - {max(results['resistances']):.4f} mΩ")
        print(f"  S/R ratio range: {min(results['sr_ratios']):.4f} - {max(results['sr_ratios']):.4f} mm²/mΩ")
        print(f"  Average S/R ratio: {np.mean(results['sr_ratios']):.4f} mm²/mΩ")
        print(f"  Maximum S/R ratio: {max_sr:.4f} mm²/mΩ")
        print(f"  Total samples: {len(results['sr_ratios'])}")

    # Calculate and print improvement rate
    if len(max_sr_values) == 2:
        straight_max = max_sr_values[0]
        superelliptic_max = max_sr_values[1]
        improvement_rate = (superelliptic_max - straight_max) / straight_max * 100

        print(f"\n{'='*60}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Straight mode max S/R:        {straight_max:.4f} mm²/mΩ")
        print(f"Superelliptic mode max S/R:   {superelliptic_max:.4f} mm²/mΩ")
        print(f"\nImprovement Rate:")
        print(f"  (S/R_super - S/R_straight) / S/R_straight")
        print(f"  = ({superelliptic_max:.4f} - {straight_max:.4f}) / {straight_max:.4f}")
        print(f"  = {improvement_rate:+.2f}%")

    # Find best overall
    all_sr = []
    all_names = []
    for results in results_list:
        for i, sr in enumerate(results['sr_ratios']):
            all_sr.append(sr)
            all_names.append(f"{results['name']} - {results['sample_labels'][i]}")

    best_idx = np.argmax(all_sr)
    worst_idx = np.argmin(all_sr)

    print(f"\n{'='*60}")
    print(f"BEST PERFORMANCE:")
    print(f"  Configuration: {all_names[best_idx]}")
    print(f"  S/R ratio: {all_sr[best_idx]:.4f} mm²/mΩ")
    print(f"  Area: {results_list[0]['areas'][best_idx] if best_idx < len(results_list[0]['areas']) else results_list[1]['areas'][best_idx - len(results_list[0]['areas'])]:.4f} mm²")

    print(f"\nWORST PERFORMANCE:")
    print(f"  Configuration: {all_names[worst_idx]}")
    print(f"  S/R ratio: {all_sr[worst_idx]:.4f} mm²/mΩ")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Determine configuration name based on selected_layers
    if 'layers_a' in str(selected_layers):
        config_name = "1320 Solution"
    elif 'layers_c' in str(selected_layers):
        config_name = "1020 Solution"
    else:
        config_name = "Solution"

    # Analyze with both modes
    print(f"\n{'#'*60}")
    print(f"# Comparing Straight vs Superelliptic Mode")
    print(f"# Configuration: {config_name}")
    print(f"{'#'*60}")

    results_straight = analyze_with_mode(
        config_name,
        selected_layers,
        mode_name="straight",
        color='#ff6b6b',
        marker='o',
        sample_count=SAMPLE_COUNT
    )

    results_superelliptic = analyze_with_mode(
        config_name,
        selected_layers,
        mode_name="superelliptic",
        color='#4ecdc4',
        marker='s',
        sample_count=SAMPLE_COUNT
    )

    results_list = [results_straight, results_superelliptic]

    # Print summary
    print_summary(results_list)

    # Plot performance
    plot_performance(results_list, config_name=config_name)

    # Show plot
    plt.show()
