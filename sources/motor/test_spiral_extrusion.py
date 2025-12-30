#!/usr/bin/env python3
"""
Test script to verify spiral extrusion with debug output
"""

import sys
sys.path.insert(0, '.')

from step import StepExporter
from pattern import Pattern
from settings import layers_c as layers

# Create exporter
exporter = StepExporter()

# Get first layer
layer_config = layers["layers"][0]
global_settings = layers["global"]

# Build single pattern for testing
current_config = {"layer": layer_config["layer"]}
current_config["layer"].update(global_settings)

# Get a normal pattern (not the first one which may have special handling)
pattern_data = Pattern.GetPattern(
    preConfig=None,
    currentConfig=current_config,
    nextConfig=current_config,  # Use same config for next
    side="front",
    layer="mid",
    layerIndex=0,
    patternIndex=4,  # Middle pattern
    patternCount=9
)

print("Pattern data keys:", pattern_data.keys())
print("Shape points:", len(pattern_data["shape"]))
if len(pattern_data["shape"]) > 0:
    import numpy as np
    shape_arr = np.array(pattern_data["shape"])
    print(f"Shape X range: [{shape_arr[:, 0].min():.3f}, {shape_arr[:, 0].max():.3f}]")
    print(f"Shape Y range: [{shape_arr[:, 1].min():.3f}, {shape_arr[:, 1].max():.3f}]")
    print(f"First 5 points:\n{shape_arr[:5]}")

# Create multiple patterns with different positions
patterns = []
for i in range(5):
    p = Pattern.GetPattern(
        preConfig=None,
        currentConfig=current_config,
        nextConfig=current_config,
        side="front",
        layer="mid",
        layerIndex=0,
        patternIndex=i,
        patternCount=9
    )
    patterns.append(p)
    print(f"Pattern {i}: {len(p['shape'])} points")

# Create layers data with multiple patterns
layers_data = {
    "front": patterns,
    "back": []
}

# Test spiral wrapping with extrusion
print("\n" + "="*60)
print("Testing spiral wrapping with extrusion...")
print("="*60 + "\n")

compound = exporter.create_spiral_wrapped_compound(
    layers_data,
    radius=6.2055,
    thick=0.1315,
    offset=0.05,
    layer_pbh=layer_config["layer"]["layer_pbh"],
    layer_ppw=layer_config["layer"]["layer_ppw"],
    layer_ptc=0.047,  # Copper thickness
    num_physical_layers=1
)

print(f"\nResult compound type: {type(compound)}")
print(f"Number of shapes created: {len(exporter.current_shapes)}")

if len(exporter.current_shapes) > 0:
    first_shape = exporter.current_shapes[0]["shape"]
    print(f"First shape type: {type(first_shape)}")
