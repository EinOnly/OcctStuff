# Rectangle Drawing Program (Refactored Architecture)

This program creates an interactive rectangle with adjustable parameters using a clean separation of concerns architecture.

## Architecture

The application is now split into three main components:

### 📐 `pattern.py` - Geometry Engine
- **Pure geometry calculations** - No visualization dependencies
- **Rectangle data generation** - Segment coordinates, dimensions, parameters
- **Parameter validation** - Input bounds checking and clamping
- **Mathematical operations** - Formula generation, dimension calculations

### 🎨 `visualize.py` - Visualization Framework  
- **Matplotlib operations** - Figure setup, drawing, interaction handling
- **UI controls** - Sliders, text boxes, event handling
- **Visual styling** - Colors, transparency, line styles
- **Real-time updates** - Dynamic redrawing based on data changes

### 🔗 `main.py` - Integration Layer
- **Application entry point** - Coordinates between modules
- **Component initialization** - Sets up generator and visualizer
- **User interface** - Provides unified application experience

## Layout Configuration

The application uses a horizontal three-panel layout with precise dimensions:

### 📐 **Dimension Specifications**
- **Total Height**: 200 pixels
- **Total Width**: 880 pixels (180 + 200 + 500)
- **Figure Size**: 8.8" × 2.0" (at 100 DPI equivalent)

### 🎛️ **Control Panel (Left - 180px wide)**
- **Sliders**: Coef, W, H parameters (0-20 range)
- **Text Boxes**: Precise numerical input for each parameter
- **Compact Layout**: Optimized for narrow width
- **Real-time Sync**: Bidirectional slider ↔ text box updates

### 🖼️ **Window A (Center - 200px wide)**  
- **Rectangle Display**: Main interactive visualization
- **Parameter Info**: Real-time parameter and dimension display
- **Visual Features**: Color-coded segments, transparency effects
- **Grid & Axes**: Coordinate system with labels

### 📋 **Window B (Right - 500px wide)**
- **Reserved Space**: Available for future features
- **Largest Panel**: Maximum space for additional visualizations
- **Grid Ready**: Pre-configured coordinate system

## Features

- **Interactive Rectangle Drawing**: Real-time rectangle visualization with three adjustable parameters
- **Parameter Control**: Three sliders to control `coef`, `W`, and `H` with range 0-20
- **Precise Input**: Text boxes for exact numerical input of parameter values
- **Color Differentiation**: Different colors for different segment types
  - Blue lines: `coef` segments (scaling coefficient)
  - Red lines: `W/H` segments (additional width/height)
- **Vertical Split**: A vertical axis divides the rectangle into left and right halves
- **Transparency Effect**: Right side has reduced opacity (30%) while left side is fully opaque
- **Fine Boundaries**: Thinner line width (2px) for cleaner appearance
- **Real-time Updates**: Rectangle updates immediately as you adjust sliders or input values
- **Parameter Display**: Shows current parameter values and calculated dimensions

## Rectangle Structure

The rectangle is constructed based on the formula:
- **Width**: `(coef + W + coef)` = `2*coef + W`
- **Height**: `(coef + H + coef)` = `2*coef + H`

Where:
- `coef`: Scaling coefficient for the base unit rectangle (applied to both ends)
- `W`: Additional width added to the center of horizontal sides
- `H`: Additional height added to the center of vertical sides

## Usage

### Running the Interactive Program

```bash
# Navigate to the motor directory
cd /Users/ein/EinDev/OcctStuff/sources/motor

# Run the main pattern program
python pattern.py

# Or run through main.py
python main.py
```

### Running Tests

```bash
# Run the test script to verify calculations
python test_pattern.py
```

This will:
1. Test dimension calculations with different parameter values
2. Create a static test rectangle image (`test_rectangle.png`)
3. Verify that the program logic is working correctly

## Controls

- **Coef Slider**: Controls the scaling coefficient (0-20)
- **W Slider**: Controls additional width in the center (0-20)  
- **H Slider**: Controls additional height in the center (0-20)
- **Text Input Boxes**: Enter precise numerical values for each parameter
  - Values are automatically clamped to the valid range (0-20)
  - Invalid inputs are automatically reset to current values
  - Text boxes sync with sliders in real-time

## Output

The program displays:
1. Interactive rectangle visualization with vertical axis division
2. Real-time parameter values with both sliders and text input boxes
3. Calculated dimensions
4. Formula representation
5. Color-coded legend with transparency indicators
6. Left side with full opacity (100%)
7. Right side with reduced transparency (30% opacity)
8. Thinner line boundaries for cleaner visual appearance

## Requirements

- Python 3.10+
- matplotlib
- numpy

## File Structure

- `pattern.py`: Rectangle geometry generator (matplotlib-free)
- `visualize.py`: Interactive matplotlib visualization framework  
- `main.py`: Application integration and entry point
- `test_refactored.py`: Comprehensive test suite for the refactored system
- `test_pattern.py`: Legacy test script (kept for reference)
- `README.md`: This documentation file

## Example Usage

### Basic Usage
```python
# Run the complete interactive application
python main.py
```

### Using Components Separately

```python
# Use just the geometry engine
from pattern import RectangleGenerator

gen = RectangleGenerator(coef=2.0, W=3.0, H=4.0)
dimensions = gen.get_dimensions()  # {'width': 7.0, 'height': 8.0}
rect_data = gen.generate_rectangle()  # Complete data structure

# Use with custom visualization
from visualize import RectangleVisualizer

visualizer = RectangleVisualizer(gen)
visualizer.show()
```

### Feeding Parameters Asynchronously

```python
import asyncio
from pattern import RectangleGenerator
from visualize import RectangleVisualizer

async def drive_parameters(adapter):
    for step in range(6):
        await adapter.submit_async({"coef": 1.0 + step, "W": step * 0.2})
        await asyncio.sleep(0.5)

async def main():
    generator = RectangleGenerator()
    visualizer = RectangleVisualizer(generator, height=250)
    visualizer.show(block=False)           # keep matplotlib responsive
    await drive_parameters(visualizer.async_input)
    await asyncio.sleep(1)                 # allow final redraw

asyncio.run(main())
```

## Example

With parameters `coef=2`, `W=3`, `H=4`:
- Rectangle width: 2*2 + 3 = 7 units
- Rectangle height: 2*2 + 4 = 8 units
- Blue segments represent the `coef` portions (2 units each)
- Red segments represent the `W` and `H` portions (3 and 4 units respectively)
