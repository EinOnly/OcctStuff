# Performance Analysis Tool

## Overview
This tool analyzes and compares the performance of motor patterns between **straight mode** and **superelliptic mode** by generating performance curves.

## Files Modified/Created

### 1. `performance.py` (NEW)
Main performance analysis script that:
- Analyzes patterns by varying key parameters
- Compares straight vs superelliptic modes
- Generates line graphs showing Area vs S/R ratio
- **S/R ratio** = Area / Resistance (higher is better)

### 2. `pattern.py` (UPDATED)
- Modified `if __name__ == "__main__"` section to use `PParams` constraints
- Now properly applies parameter constraints from `parameters.py`
- All three configurations (layers_a, layers_b, layers_c) work correctly

### 3. `test_all_configs.py` (NEW)
Test script to verify all three configurations work with PParams constraints

## How to Use `performance.py`

### Basic Usage
```bash
cd sources/motor
python performance.py
```

### Configuration Options

Edit `performance.py` at lines 28-33:

```python
# ============================================================
# CONFIGURATION - Change these to analyze different setups
# ============================================================
from settings import layers_b as selected_layers  # Change to layers_a, layers_b, or layers_c
SAMPLE_COUNT = 20  # Number of parameter samples per layer (10-50 recommended)
# ============================================================
```

**Available Configurations:**
- `layers_a`: 1320 Solution (4 layers, twisted, superelliptic)
- `layers_b`: 1120 Solution (3 layers, non-twisted, lap type) **[DEFAULT]**
- `layers_c`: 1020 Solution (4 layers, twisted, superelliptic)

**Sample Count:**
- Default: 20 samples per layer
- Lower values (10-15): Faster computation, rougher curves
- Higher values (30-50): Slower computation, smoother curves

### Parameter Sampling

**Straight Mode:**
- Varies `pattern_tp3` (and `pattern_bp3`) from `ppw` to `pbh/2 - ppw`
- Tests different vertical curve positions
- Constraints automatically applied via `PParams`

**Superelliptic Mode:**
- Varies `pattern_tmm` (and `pattern_bmm`) from 0.5 to 2.0
- Tests different curve shapes (from sharp to rounded)
- Uses middle value for `tp3`

## Output

### Console Output
- Detailed analysis progress for each layer
- Parameter ranges being tested
- Summary statistics for each mode
- Best and worst performing configurations

### Graph Output
- File: `performance_analysis.png` (saved in project root)
- **X-axis**: Convex hull area (mm²)
- **Y-axis**: S/R ratio (mm²/mΩ)
- Two line plots: straight (red circles) vs superelliptic (cyan squares)
- Higher S/R ratio = better performance (more area per unit resistance)

## Example Results (1120 Solution, 20 samples)

```
1120 Solution (straight):
  Area range: 26.2310 - 44.0010 mm²
  S/R ratio range: 0.0332 - 3.3848 mm²/mΩ
  Average S/R ratio: 2.5271 mm²/mΩ
  Total samples: 60

1120 Solution (superelliptic):
  Area range: 29.0578 - 40.1725 mm²
  S/R ratio range: 1.9897 - 4.2690 mm²/mΩ
  Average S/R ratio: 3.1288 mm²/mΩ
  Total samples: 60

BEST PERFORMANCE:
  Configuration: 1120 Solution (superelliptic) - L0_19
  S/R ratio: 4.2690 mm²/mΩ
```

## Key Improvements

1. **Constraint Integration**: All parameters now properly constrained via `PParams`
   - Ensures p0 + p1 = pbw/2
   - Ensures p2 + p3 = pbh/2
   - Mode-specific constraints automatically applied

2. **Dense Sampling**: Generates multiple parameter variations per layer
   - Default 20 samples = 60 total points for 3-layer config
   - Configurable sampling density

3. **Visual Comparison**: Clear line graphs showing performance differences
   - Easy to identify optimal parameter ranges
   - Compare modes at a glance

## Technical Notes

- Uses copper foil standard thickness: 0.047 mm
- Resistance calculated along outer conductor path
- Convex hull area used as envelope metric
- All calculations respect layer type (wave vs lap)
- Twist and symmetry settings from configuration preserved
