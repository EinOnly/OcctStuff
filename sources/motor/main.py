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
    print("Motor Pattern Drawing Application")
    print("=" * 40)
    print("Starting Interactive Motor Pattern Visualizer...")
    print("Architecture:")
    print("- pattern.py: Pattern parameters and calculations")
    print("- visualizer.py: PyQt5 visualization and UI")
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
    print("- Left: Input controls (sliders)")
    print("- Center: Pattern window")
    print("- Right: Assembly window")
    print("=" * 40)
    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create pattern and visualizer
    pattern = Pattern(width=5.89, height=7.5)
    visualizer = Visualizer(pattern=pattern, height=300, multiple=2.5, spacing=10)
    
    # Show the interactive application
    visualizer.show()
    
    # Run the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
