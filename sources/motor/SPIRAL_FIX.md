# Spiral Wrapping - Latest Fix

## Problem
When enabling Spiral Mode and clicking Refresh View, the console showed repeated warnings:
```
Warning: Failed to create face from wire (IsDone=False)
```

This prevented the spiral-wrapped patterns from being created and displayed.

## Root Cause
The previous implementation tried to use `BRepBuilderAPI_MakeFace(wire)` to create faces from the wrapped pattern wires, then extrude them with `BRepPrimAPI_MakePrism`.

**This failed because:**
- `MakeFace` requires the wire to be **planar** (all points in a single plane)
- Spiral-wrapped pattern wires are **non-planar** 3D curves
- The operation could not complete (IsDone=False)

## Solution
Replaced the face extrusion approach with **lofting between offset wires**:

### Old Approach (Failed)
```python
# Try to create planar face from non-planar wire
face = BRepBuilderAPI_MakeFace(wire).Face()  # ❌ Fails!
prism = BRepPrimAPI_MakePrism(face, extrusion_vec)
```

### New Approach (Fixed)
```python
# Create two wires (base and radially offset)
base_wire = self._make_wire_from_points(pts)

# Offset each point radially to create thickness
offset_pts = []
for pt in pts:
    x, y, z = pt
    r = sqrt(x² + z²)
    nx, nz = x/r, z/r
    x_new = x + thickness * direction * nx
    z_new = z + thickness * direction * nz
    offset_pts.append((x_new, y, z_new))

offset_wire = self._make_wire_from_points(offset_pts)

# Loft between the two wires to create solid
loft = BRepOffsetAPI_ThruSections(True)  # True = make solid
loft.AddWire(base_wire)
loft.AddWire(offset_wire)
loft.Build()
return loft.Shape()
```

### Why This Works
- `BRepOffsetAPI_ThruSections` can loft between **arbitrary 3D wire profiles**
- No planarity requirement
- Creates smooth solid between inner and outer surfaces
- Handles radial thickness correctly for spiral geometry

## Code Changes

### Modified Files
- [step.py](step.py):
  - Modified `_create_wrapped_pattern_solid` (L351-416)
  - Added `_make_wire_from_points` helper (L418-438)
  - Added import: `BRepOffsetAPI_ThruSections` (L36)

### Key Implementation Details

**Pattern Thickness Direction:**
- Front patterns: `outward=True` → offset radially **outward** from spiral
- Back patterns: `outward=False` → offset radially **inward** from spiral

**Radial Offset Calculation:**
```python
# Calculate radial normal in X-Z plane (Y is vertical)
r = sqrt(x² + z²)
nx = x / r  # X component of radial normal
nz = z / r  # Z component of radial normal

# Offset point
x_new = x + thickness * direction * nx
z_new = z + thickness * direction * nz
y_new = y  # Y unchanged (vertical)
```

## Testing

To test the fix:

1. Start the application:
   ```bash
   cd /Users/ein/EinDev/OcctStuff/sources/motor
   python main.py
   ```

2. Generate layers:
   - Click "Generate Layers"

3. Enable spiral mode:
   - Click "Spiral Mode" button
   - Click "Refresh View"

4. **Expected Results:**
   - No "Failed to create face" warnings
   - Spiral-wrapped patterns visible in 3D viewer
   - Console shows successful pattern creation

## Next Steps

If the fix works:
- ✅ Patterns should appear as spiral-wrapped 3D geometry
- ✅ Can export to STEP file using "Save STEP" button
- ✅ Can rotate/zoom to inspect spiral structure

If issues persist:
- Check console for new error messages
- Verify layer data has valid pattern shapes
- Check pattern bounds are reasonable (X span < arc_per_layer)
