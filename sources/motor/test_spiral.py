#!/usr/bin/env python3
"""
Test script for spiral wrapping functionality.

This script demonstrates how to use the Spiral class to wrap patterns
onto spiral surfaces and export to STEP format.
"""

import sys
from PyQt5.QtWidgets import QApplication
from step import StepViewer

# Import settings to get layer parameters
from settings import layers_a


def create_test_data():
    """
    Create test data for spiral wrapping using settings from layers_a.
    """
    global_settings = layers_a.get("global", {})
    layers = layers_a.get("layers", [])

    # Extract global parameters
    layer_psp = global_settings.get("layer_psp", 0.05)
    layer_ptc = global_settings.get("layer_ptc", 0.047)

    # For this test, we'll use simple rectangular patterns
    # In real usage, these would come from your pattern generation code
    front_patterns = []
    back_patterns = []

    for idx, layer_data in enumerate(layers):
        layer_info = layer_data.get("layer", {})
        color = layer_info.get("color", "#ff0000")

        # Create a small rectangular pattern for demonstration
        # Pattern X coordinate represents arc length along spiral
        # Pattern Y coordinate represents vertical height offset
        # Use small arc-length width to clearly show spiral wrapping

        arc_width = 0.5  # Small width along arc length
        height = 0.3     # Small vertical height

        # Simple small rectangle - X is arc length, Y is vertical
        pattern_shape = [
            (0, -height/2),
            (arc_width, -height/2),
            (arc_width, height/2),
            (0, height/2),
            (0, -height/2)  # Close the shape
        ]

        # Add to front layer
        front_patterns.append({
            "shape": pattern_shape,
            "color": color
        })

        # Create similar pattern for back layer
        back_pattern_shape = [
            (0, -height/2),
            (arc_width, -height/2),
            (arc_width, height/2),
            (0, height/2),
            (0, -height/2)
        ]

        back_patterns.append({
            "shape": back_pattern_shape,
            "color": color
        })

    return {
        "front": front_patterns,
        "back": back_patterns,
        "params": {
            "layer_psp": layer_psp,
            "layer_ptc": layer_ptc,
            "layer_pbh": layers[0]["layer"].get("layer_pbh", 8.0) if layers else 8.0,
            "layer_ppw": layers[0]["layer"].get("layer_ppw", 0.5) if layers else 0.5,
        }
    }


def main():
    """Main test function."""
    print("=" * 60)
    print("Spiral Wrapping Test")
    print("=" * 60)

    # Create Qt application
    app = QApplication(sys.argv)

    # Create viewer
    viewer = StepViewer()
    viewer.setWindowTitle("Spiral Pattern Wrapping - Test")
    viewer.resize(1200, 800)

    # Create test data
    print("Creating test pattern data...")
    test_data = create_test_data()
    params = test_data["params"]

    # Set layers
    viewer.setLayers({
        "front": test_data["front"],
        "back": test_data["back"]
    })

    # Enable spiral mode with parameters
    print("\nEnabling spiral wrapping mode...")
    print(f"  Radius: 6.2055 mm")
    print(f"  Thickness (pitch): 0.1315 mm")
    print(f"  Offset: 0.05 mm")
    print(f"  Layer count: {len(test_data['front'])}")
    print(f"  Pattern thickness: {params['layer_ptc']} mm")

    viewer.enable_spiral_mode(
        radius=6.2055,
        thick=0.1315,
        offset=0.05,
        layer_pbh=params["layer_pbh"],
        layer_ppw=params["layer_ppw"],
        layer_ptc=params["layer_ptc"]
    )

    # Refresh view to build and display
    print("\nBuilding 3D spiral-wrapped geometry...")
    viewer.refresh_view()

    # Also display the spiral curves for reference
    print("\nAdding spiral reference curves...")
    if viewer.exporter.spiral:
        spiral_i, spiral_c, spiral_o = viewer.exporter.spiral.get_spiral_curves()
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

        # Display center spiral in white
        edge_c = BRepBuilderAPI_MakeEdge(spiral_c).Edge()
        viewer.viewer._display.DisplayShape(
            edge_c,
            color=Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB),
            update=False
        )

        # Display inner spiral in cyan
        edge_i = BRepBuilderAPI_MakeEdge(spiral_i).Edge()
        viewer.viewer._display.DisplayShape(
            edge_i,
            color=Quantity_Color(0.0, 1.0, 1.0, Quantity_TOC_RGB),
            update=False
        )

        # Display outer spiral in magenta
        edge_o = BRepBuilderAPI_MakeEdge(spiral_o).Edge()
        viewer.viewer._display.DisplayShape(
            edge_o,
            color=Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB),
            update=False
        )

        viewer.viewer._display.FitAll()
        viewer.viewer._display.Repaint()

    print("\nViewer ready!")
    print("\nControls:")
    print("  - Mouse: Rotate (left), Pan (middle), Zoom (right/wheel)")
    print("  - Key 1: Top view")
    print("  - Key 2: Fit all")
    print("  - +/-: Zoom in/out")
    print("  - Arrow keys: Pan")
    print("  - Use 'Save STEP' to export geometry")
    print("=" * 60)

    # Show viewer
    viewer.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
