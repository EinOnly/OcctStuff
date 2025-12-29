#!/usr/bin/env python3
"""Test all three layer configurations with PParams constraints."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import layers_a, layers_b, layers_c
from parameters import PParams
from pattern import Pattern

def test_config(config_name, layers_config):
    """Test a single configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {config_name}")
    print(f"{'='*60}")

    global_settings = layers_config["global"]
    layers_list = layers_config["layers"]

    print(f"Global settings: {global_settings}")
    print(f"Number of layers: {len(layers_list)}")

    for idx, layer_config in enumerate(layers_list):
        layer_type = layer_config.get("type", "normal")
        layer_params = layer_config["layer"]

        print(f"\n--- Layer {idx} ({layer_type}) ---")

        # Create PParams instance
        pparams = PParams()

        # Prepare parameters
        params_to_update = {
            "pattern_mode": global_settings["layer_pmd"],
            "pattern_type": global_settings.get("layer_type", "wave"),
            "pattern_pbw": layer_params["layer_pbw"],
            "pattern_pbh": layer_params["layer_pbh"],
            "pattern_ppw": layer_params["layer_ppw"],
            "pattern_psp": global_settings["layer_psp"],
            "pattern_twist": layer_params.get("pattern_twist", False),
            "pattern_symmetry": layer_params.get("pattern_symmetry", False),
            "pattern_tp0": layer_params.get("pattern_tp0", 0.0),
            "pattern_tp3": layer_params.get("pattern_tp3", 0.0),
            "pattern_tnn": layer_params.get("pattern_tnn", 2.0),
            "pattern_tmm": layer_params.get("pattern_tmm", 2.0),
            "pattern_bp0": layer_params.get("pattern_bp0", 0.0),
            "pattern_bp3": layer_params.get("pattern_bp3", 0.0),
            "pattern_bnn": layer_params.get("pattern_bnn", 2.0),
            "pattern_bmm": layer_params.get("pattern_bmm", 2.0),
        }

        # Apply constraints
        pparams.update_bulk(params_to_update, emit=False)
        constrained_params = pparams.snapshot()

        print(f"Input pbw={layer_params['layer_pbw']:.4f}, pbh={layer_params['layer_pbh']:.4f}, ppw={layer_params['layer_ppw']:.4f}")
        print(f"Constrained tp0={constrained_params['pattern_tp0']:.4f}, tp1={constrained_params['pattern_tp1']:.4f}, tp2={constrained_params['pattern_tp2']:.4f}, tp3={constrained_params['pattern_tp3']:.4f}")
        print(f"Constrained bp0={constrained_params['pattern_bp0']:.4f}, bp1={constrained_params['pattern_bp1']:.4f}, bp2={constrained_params['pattern_bp2']:.4f}, bp3={constrained_params['pattern_bp3']:.4f}")

        # Verify constraints
        pbw = constrained_params['pattern_pbw']
        pbh = constrained_params['pattern_pbh']
        tp0_tp1_sum = constrained_params['pattern_tp0'] + constrained_params['pattern_tp1']
        tp2_tp3_sum = constrained_params['pattern_tp2'] + constrained_params['pattern_tp3']
        bp0_bp1_sum = constrained_params['pattern_bp0'] + constrained_params['pattern_bp1']
        bp2_bp3_sum = constrained_params['pattern_bp2'] + constrained_params['pattern_bp3']

        print(f"Constraint check: tp0+tp1={tp0_tp1_sum:.4f} (should be {pbw/2:.4f})")
        print(f"Constraint check: tp2+tp3={tp2_tp3_sum:.4f} (should be {pbh/2:.4f})")
        print(f"Constraint check: bp0+bp1={bp0_bp1_sum:.4f} (should be {pbw/2:.4f})")
        print(f"Constraint check: bp2+bp3={bp2_bp3_sum:.4f} (should be {pbh/2:.4f})")

        # Try to build a pattern
        try:
            config = {"layer": constrained_params}
            pattern = Pattern.GetPattern(
                preConfig=None,
                currentConfig=config,
                nextConfig=config,
                side="front",
                layer="mid",
                layerIndex=idx,
                patternIndex=4,
                patternCount=layer_params.get("layer_pdc", 9)
            )
            print(f"Pattern built successfully!")
            print(f"  Resistance: {pattern['pattern_resistance']*1000:.3f} mΩ")
            print(f"  Area: {pattern['pattern_area']:.4f} mm²")
            print(f"  Convex hull area: {pattern['convexhull_area']:.4f} mm²")
        except Exception as e:
            print(f"ERROR building pattern: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_config("layers_a (1320 Solution)", layers_a)
    test_config("layers_b (1120 Solution)", layers_b)
    test_config("layers_c (1020 Solution)", layers_c)

    print(f"\n{'='*60}")
    print("All configurations tested!")
    print(f"{'='*60}")
