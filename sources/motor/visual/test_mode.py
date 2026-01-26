#!/usr/bin/env python3
"""Test mode parameter propagation."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import layers_b as layers
from parameters import PParams

# Get layer configuration
normal_layer_config = layers["layers"][0]  # Use layer 0 (non-twist)
layer_params = normal_layer_config["layer"]
global_settings = layers["global"]

# Prepare parameters to update
params_to_update = {
    "pattern_psp": global_settings.get("layer_psp", 0.05),
    "pattern_pbw": layer_params.get("layer_pbw", 5.0),
    "pattern_pbh": layer_params.get("layer_pbh", 7.5),
    "pattern_ppw": layer_params.get("layer_ppw", 0.5),
    "pattern_type": global_settings.get("layer_type", "wave"),
    "pattern_twist": False,  # Force non-twist for visualization clarity
    "pattern_symmetry": layer_params.get("pattern_symmetry", False),
    "pattern_tp0": layer_params.get("pattern_tp0", 0.0),
    "pattern_tp3": layer_params.get("pattern_tp3", 1.5),
    "pattern_tnn": layer_params.get("pattern_tnn", 2.0),
    "pattern_tmm": layer_params.get("pattern_tmm", 1.2),
    "pattern_bp0": layer_params.get("pattern_bp0", 0.0),
    "pattern_bp3": layer_params.get("pattern_bp3", 1.5),
    "pattern_bnn": layer_params.get("pattern_bnn", 2.0),
    "pattern_bmm": layer_params.get("pattern_bmm", 1.2),
}

print("=" * 70)
print("Testing STRAIGHT mode")
print("=" * 70)

pparams = PParams()
pparams.update_bulk({**params_to_update, "pattern_mode": "straight"}, emit=False)
straight_params = pparams.snapshot()

print(f"MODE in snapshot: {straight_params.get('pattern_mode', 'NOT SET')}")
print(f"tmm: {straight_params.get('pattern_tmm', 'NOT SET')}")
print(f"tp1: {straight_params.get('pattern_tp1', 'NOT SET'):.3f}, tp2: {straight_params.get('pattern_tp2', 'NOT SET'):.3f}")

print("\n" + "=" * 70)
print("Testing SUPERELLIPTIC mode")
print("=" * 70)

print("Before update_bulk, params_to_update has:")
update_dict = {**params_to_update, "pattern_mode": "superelliptic"}
for k in ['pattern_mode', 'pattern_tmm', 'pattern_bmm', 'pattern_tnn', 'pattern_bnn']:
    print(f"  {k}: {update_dict.get(k, 'MISSING')}")

pparams.update_bulk(update_dict, emit=False)
superellipse_params = pparams.snapshot()

print(f"MODE in snapshot: {superellipse_params.get('pattern_mode', 'NOT SET')}")
print(f"tmm: {superellipse_params.get('pattern_tmm', 'NOT SET')}")
print(f"bmm: {superellipse_params.get('pattern_bmm', 'NOT SET')}")
print(f"tnn: {superellipse_params.get('pattern_tnn', 'NOT SET')}")
print(f"bnn: {superellipse_params.get('pattern_bnn', 'NOT SET')}")
print(f"tp1: {superellipse_params.get('pattern_tp1', 'NOT SET'):.3f}, tp2: {superellipse_params.get('pattern_tp2', 'NOT SET'):.3f}")
print(f"tp1==tp2? {abs(superellipse_params['pattern_tp1'] - superellipse_params['pattern_tp2']) < 0.001}")

print("\n" + "=" * 70)
print("All snapshot keys:")
print("=" * 70)
for key in sorted(superellipse_params.keys()):
    if key.startswith('pattern_'):
        print(f"  {key}: {superellipse_params[key]}")
