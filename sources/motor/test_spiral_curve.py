#!/usr/bin/env python3
"""
Simple test to visualize just the spiral curves.
"""

import sys
from PyQt5.QtWidgets import QApplication
from step import Spiral
from OCC.Display.backend import load_backend
load_backend('pyqt5')
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB


def main():
    """Test spiral curve generation."""
    print("=" * 60)
    print("Spiral Curve Visualization")
    print("=" * 60)

    # Parameters from your application
    radius = 6.2055
    thick = 0.1315
    layer_count = 4  # PHYSICAL LAYERS (not pattern count!)
    offset = 0.05

    print(f"\nSpiral parameters:")
    print(f"  Outer radius: {radius} mm")
    print(f"  Pitch (thick): {thick} mm")
    print(f"  Layer count: {layer_count}")
    print(f"  Offset: {offset} mm")

    # Create spiral
    spiral = Spiral(radius, thick, layer_count, offset)

    print(f"\nSpiral properties:")
    print(f"  Total arc length: {spiral.total_length:.3f} mm")
    print(f"  Arc length per layer: {spiral.total_length / layer_count:.3f} mm")

    # Calculate expected radius range
    import numpy as np
    b = thick / (2.0 * np.pi)
    r0 = radius - thick / 2.0
    theta_max = 2.0 * np.pi * layer_count
    r_start = r0
    r_end = r0 - b * theta_max

    print(f"\nRadius range:")
    print(f"  Starting radius: {r_start:.3f} mm")
    print(f"  Ending radius: {r_end:.3f} mm")
    print(f"  Total radius change: {r_start - r_end:.3f} mm")
    print(f"  Expected (layer_count * thick): {layer_count * thick:.3f} mm")

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

    display.FitAll()
    display.View_Top()  # Top view to see spiral clearly

    print("\n" + "=" * 60)
    print("Spiral curves displayed:")
    print("  White = Center spiral")
    print("  Cyan = Inner spiral")
    print("  Magenta = Outer spiral")
    print("\nPress 1 for top view, 2 to fit all")
    print("=" * 60)

    start_display()


if __name__ == "__main__":
    main()
