"""
Main entry point for the motor pattern drawing application.

This module integrates the rectangle generation logic (pattern.py) with 
the visualization framework (visualize.py) to create an interactive application.
"""

from pattern import RectangleGenerator
from visualize import RectangleVisualizer

def main():
    """
    Main function to create and run the rectangle drawing application.
    
    This function demonstrates the clean separation of concerns:
    - RectangleGenerator handles all geometry calculations
    - RectangleVisualizer handles all matplotlib operations
    """
    print("Motor Pattern Drawing Application")
    print("=" * 40)
    print("Starting Interactive Rectangle Drawer...")
    print("Architecture:")
    print("- pattern.py: Rectangle geometry and calculations")
    print("- visualize.py: Matplotlib visualization and UI")
    print("- main.py: Integration and application entry point")
    print()
    print("Use the sliders to adjust parameters:")
    print("- Coef: scaling coefficient for base rectangle")
    print("- W: additional width in the center")  
    print("- H: additional height in the center")
    print("- Blue lines represent coef segments")
    print("- Red lines represent W/H segments")
    print("- Left side: full opacity, Right side: 30% opacity")
    print("=" * 40)
    
    # Create the rectangle generator (pure geometry logic)
    rectangle_gen = RectangleGenerator(coef=1.0, W=0.0, H=0.0)
    
    # Create the visualizer with custom layout parameters
    # height=300: Base height for all windows (pixels equivalent)
    # multiple=2.5: WindowB is 2.5 times the width of other windows
    # spacing=0.08: Spacing between windows (0.0 to 0.2)
    visualizer = RectangleVisualizer(rectangle_gen, height=300, multiple=2.5, spacing=0.08)
    
    # Show the interactive application
    visualizer.show()

if __name__ == "__main__":
    main()
