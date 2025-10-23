from visualizer import Visualizer
from pattern import Pattern
import sys
from PyQt5.QtWidgets import QApplication

def main():
    """
    Main function to create and run the motor pattern visualization application.
    
    This function demonstrates the clean separation of concerns:
    - Pattern: Handles all parameter storage and validation
    - Visualizer: Handles PyQt5 UI and visualization
    - main.py: Integration and application entry point
    """
    print("Motor Pattern Drawing Application with 3D OCCT Viewer")
    print("=" * 50)
    print("Starting Interactive Motor Pattern Visualizer...")
    print("Architecture:")
    print("- pattern.py: Pattern parameters and calculations")
    print("- visualizer.py: PyQt5 visualization and UI")
    print("- step.py: OCCT 3D shape generation and STEP export")
    print("- main.py: Integration and application entry point")
    print()
    print("Use the sliders to adjust parameters:")
    print("- width: Base width of the pattern")
    print("- height: Base height of the pattern")
    print("- vbh / vlw: Straight and corner spans (exponent == 2)")
    print("- corner: Unified corner span (exponent < 2)")
    print("- exponent: Superellipse corner exponent")
    print()
    print("UI Layout:")
    print("Row 1:")
    print("  - Left: Input controls (sliders)")
    print("  - Center-Left: 2D Pattern window")
    print("  - Center-Right: Chart window")
    print("  - Right: Assembly window")
    print("Row 2:")
    print("  - Left: 3D Model controls")
    print("  - Right: OCCT 3D Viewer")
    print()
    print("3D Features:")
    print("- Auto-updates when pattern changes")
    print("- Export to STEP file (0.047mm thickness)")
    print("- Interactive 3D viewing with mouse")
    print("=" * 50)
    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create pattern and visualizer
    pattern = Pattern(width=4.702, height=7.5)
    visualizer = Visualizer(pattern=pattern, height=300, multiple=2.5, spacing=10)
    
    # Show the interactive application
    visualizer.show()
    
    # Run the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
