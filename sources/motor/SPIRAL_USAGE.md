# Spiral Pattern Wrapping - Usage Guide

This document describes the new spiral wrapping functionality added to the motor pattern generation system.

## Overview

The spiral wrapping feature allows you to wrap 2D motor patterns onto 3D spiral surfaces, creating a coiled geometry. This is useful for creating motor windings and similar helical structures.

## Implementation Details

### Key Components

1. **Spiral Class** ([step.py:184-427](step.py#L184-L427))
   - Generates spiral curves (inner, center, outer)
   - Computes arc-length parameterization
   - Maps 2D patterns onto 3D spiral surfaces

2. **StepExporter.create_spiral_wrapped_compound** ([step.py:185-306](step.py#L185-L306))
   - Creates complete spiral-wrapped geometry
   - Handles both front and back layer patterns
   - Extrudes patterns with proper thickness

3. **StepViewer.enable_spiral_mode** ([step.py:662-684](step.py#L662-L684))
   - Enables spiral mode in the viewer
   - Configures spiral parameters

## Workflow Steps

The spiral wrapping process follows these steps:

### 1. Generate Spiral Curves

The system creates three concentric spiral curves:
- **spiral_c**: Center line spiral at radius - thick/2
- **spiral_i**: Inner spiral at centerline - offset
- **spiral_o**: Outer spiral at centerline + offset

Parameters:
- `radius`: Outer radius of the spiral (e.g., 6.2055 mm)
- `thick`: Spiral band thickness/pitch (e.g., 0.1315 mm)
- `layer_count`: Number of turns (determined from layer data)
- `offset`: Radial offset from centerline (e.g., 0.05 mm)

### 2. Create Spiral Surfaces

The inner and outer spiral curves are extruded vertically to create surfaces:
- Extrusion height = `layer_pbh + layer_ppw * 2`
- This ensures the surfaces fully wrap the pattern height

### 3. Map Patterns to Spiral

For each pattern in the layer data:
- Front layer patterns → mapped to outer spiral surface
- Back layer patterns → mapped to inner spiral surface

The mapping process:
1. Distribute patterns along the spiral arc length
2. Transform 2D (x, y) coordinates to 3D (x, y, z) on spiral surface
3. Maintain relative pattern geometry

### 4. Create Pattern Solids

Each mapped pattern is extruded with thickness `layer_ptc`:
- Front patterns: extruded radially outward
- Back patterns: extruded radially inward
- Creates solid 3D geometry with proper copper thickness

### 5. Display and Export

- All pattern solids are combined into a compound shape
- Displayed in the 3D viewer with colors from layer data
- Can be exported to STEP format

## Usage Example

```python
from step import StepViewer
from PyQt5.QtWidgets import QApplication

# Create application and viewer
app = QApplication(sys.argv)
viewer = StepViewer()

# Prepare layer data
layers = {
    "front": [
        {"shape": [(0,0), (10,0), (10,5), (0,5)], "color": "#ff0000"},
        # ... more patterns
    ],
    "back": [
        {"shape": [(0,0), (9,0), (9,4), (0,4)], "color": "#0000ff"},
        # ... more patterns
    ]
}

# Set layers
viewer.setLayers(layers)

# Enable spiral mode
viewer.enable_spiral_mode(
    radius=6.2055,      # Outer radius
    thick=0.1315,       # Pitch between turns
    offset=0.05,        # Inner/outer surface offset
    layer_pbh=8.0,      # Pattern base height
    layer_ppw=0.5,      # Pattern width padding
    layer_ptc=0.047     # Pattern thickness (copper)
)

# Build and display
viewer.refresh_view()
viewer.show()

# Save to STEP (via UI button or programmatically)
viewer.save_step_file()
```

## Testing

A test script is provided: [test_spiral.py](test_spiral.py)

Run it with:
```bash
python test_spiral.py
```

This will:
1. Load layer data from [settings.py](settings.py)
2. Create simple rectangular test patterns
3. Wrap them onto a spiral
4. Display in 3D viewer
5. Allow export to STEP format

## Parameters Reference

### Spiral Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `radius` | float | Outer radius of spiral | 6.2055 |
| `thick` | float | Spiral band thickness (pitch) | 0.1315 |
| `offset` | float | Radial offset for inner/outer surfaces | 0.05 |
| `layer_count` | int | Number of spiral turns | 4 |

### Pattern Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `layer_pbh` | float | Pattern base height | 8.0 |
| `layer_ppw` | float | Pattern width padding | 0.5 |
| `layer_ptc` | float | Pattern thickness (copper) | 0.047 |

## Mathematical Details

### Spiral Equation

The spiral centerline follows an Archimedean spiral in polar coordinates:

```
r(θ) = r₀ - b·θ
```

Where:
- `r₀ = radius - thick/2` (starting radius)
- `b = thick / (2π)` (spiral parameter)
- `θ ∈ [0, 2π·layer_count]` (angle range)

### Arc Length Computation

Arc length is computed numerically:

```
s(i) = s(i-1) + √(dx²+ dz²)
```

Where:
- `dx = (dr/dθ)·cos(θ) - r·sin(θ)`
- `dz = (dr/dθ)·sin(θ) + r·cos(θ)`

### Coordinate Transformation

2D pattern point (px, py) maps to 3D point:

```
P₃ᴅ = P_center + offset·n̂ + py·ŷ
```

Where:
- `P_center`: Position on spiral centerline at arc length px
- `n̂`: Normal vector (radial direction)
- `offset`: ±offset for inner/outer surface
- `ŷ`: Vertical unit vector

## Integration with Existing Code

The spiral functionality integrates seamlessly with the existing pattern generation:

1. Generate patterns using existing pattern generation code
2. Store in layer data structure (front/back)
3. Choose spiral mode or flat mode
4. Viewer handles the rest automatically

## Limitations and Notes

1. **Pattern Distortion**: Very large patterns may show distortion when wrapped
2. **Self-Intersection**: Ensure offset is sufficient to prevent surface overlap
3. **Arc Length Mapping**: Pattern X-coordinates are treated as arc length along spiral
4. **Performance**: Large layer counts or complex patterns may take time to process

## Future Enhancements

Potential improvements:
- Adaptive pattern scaling based on spiral curvature
- Variable pitch spirals
- Multiple spiral start points
- Lofted surfaces between patterns
- Intersection trimming for precise pattern boundaries

## References

- Spiral mathematics: [bak/alignment.py](bak/alignment.py)
- OpenCASCADE documentation: https://dev.opencascade.org/
- Pattern settings: [settings.py](settings.py)
