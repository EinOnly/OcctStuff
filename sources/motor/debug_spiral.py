#!/usr/bin/env python3
"""
Debug script to verify spiral curve generation.
"""

import sys
from PyQt5.QtWidgets import QApplication
from step import Spiral
from OCC.Display.backend import load_backend
load_backend('pyqt5')
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCC.Core.gp import gp_Pnt
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB


def main():
    """Test spiral generation."""
    print("=" * 60)
    print("Spiral Curve Debug")
    print("=" * 60)

    # Create spiral
    radius = 6.2055
    thick = 0.1315
    layer_count = 4
    offset = 0.05

    print(f"\nCreating spiral with:")
    print(f"  Radius: {radius}")
    print(f"  Thickness: {thick}")
    print(f"  Layers: {layer_count}")
    print(f"  Offset: {offset}")

    spiral = Spiral(radius, thick, layer_count, offset)

    print(f"\nSpiral total arc length: {spiral.total_length:.3f}")
    print(f"  Arc length per layer: {spiral.total_length / layer_count:.3f}")

    # Get spiral curves
    spiral_i, spiral_c, spiral_o = spiral.get_spiral_curves()

    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Display the three spiral curves
    print("\nDisplaying spiral curves...")

    # Center spiral (white)
    edge_c = BRepBuilderAPI_MakeEdge(spiral_c).Edge()
    display.DisplayShape(
        edge_c,
        color=Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB),
        update=False
    )

    # Inner spiral (cyan)
    edge_i = BRepBuilderAPI_MakeEdge(spiral_i).Edge()
    display.DisplayShape(
        edge_i,
        color=Quantity_Color(0.0, 1.0, 1.0, Quantity_TOC_RGB),
        update=False
    )

    # Outer spiral (magenta)
    edge_o = BRepBuilderAPI_MakeEdge(spiral_o).Edge()
    display.DisplayShape(
        edge_o,
        color=Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB),
        update=False
    )

    # Test pattern mapping
    print("\nTesting pattern mapping...")

    # Create a simple rectangular pattern
    test_pattern = [
        (0, -0.2),
        (0.5, -0.2),
        (0.5, 0.2),
        (0, 0.2),
        (0, -0.2)
    ]

    # Map pattern at different arc lengths
    arc_per_layer = spiral.total_length / layer_count

    for layer_idx in range(layer_count):
        arc_offset = layer_idx * arc_per_layer

        # Map to outer surface
        mapped_outer = spiral.map_pattern_to_spiral(
            test_pattern,
            arc_offset=arc_offset,
            y_offset=0.0,
            use_inner=False
        )

        # Map to inner surface
        mapped_inner = spiral.map_pattern_to_spiral(
            test_pattern,
            arc_offset=arc_offset,
            y_offset=0.0,
            use_inner=True
        )

        # Display as points
        for pt in mapped_outer:
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(pt[0], pt[1], pt[2])).Vertex()
            display.DisplayShape(
                vertex,
                color=Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB),  # Red
                update=False
            )

        for pt in mapped_inner:
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(pt[0], pt[1], pt[2])).Vertex()
            display.DisplayShape(
                vertex,
                color=Quantity_Color(0.0, 0.0, 1.0, Quantity_TOC_RGB),  # Blue
                update=False
            )

        print(f"  Layer {layer_idx}: arc_offset={arc_offset:.3f}")
        print(f"    Outer first point: ({mapped_outer[0][0]:.3f}, {mapped_outer[0][1]:.3f}, {mapped_outer[0][2]:.3f})")
        print(f"    Inner first point: ({mapped_inner[0][0]:.3f}, {mapped_inner[0][1]:.3f}, {mapped_inner[0][2]:.3f})")

    display.FitAll()
    display.View_Iso()

    print("\n" + "=" * 60)
    print("Display ready. Close window to exit.")
    print("=" * 60)

    start_display()


if __name__ == "__main__":
    main()
