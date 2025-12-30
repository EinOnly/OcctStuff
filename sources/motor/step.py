"""
STEP file generator and 3D viewer for motor pattern shapes using OpenCASCADE.
"""

from typing import List, Tuple, Dict, Any, Optional
import os

# PyQt5 imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QMessageBox, QSizePolicy, QProgressDialog
)
from PyQt5.QtCore import Qt, QCoreApplication

# OpenCASCADE imports
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax2, gp_Dir
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Face
from OCC.Core.BRep import BRep_Builder
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge as MakeEdge_FromCurve
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common

# OCCT Display
from OCC.Display.backend import load_backend
load_backend('pyqt5')
from OCC.Display.qtDisplay import qtViewer3d


class StepExporter:
    """Generate 3D OCCT shapes from 2D motor pattern curves and export to STEP."""

    def __init__(self, thickness: float = 0.047):
        """
        Initialize STEP exporter.

        Args:
            thickness: Extrusion thickness in mm (default 0.047 mm for copper foil)
        """
        self.thickness = thickness
        self.current_shapes = []
        self.spiral = None  # Will hold Spiral instance if using spiral mode

    def create_shape_from_curve(self, curve_points: List[Tuple[float, float]],
                                z_offset: float = 0.0) -> TopoDS_Shape:
        """
        Create a 3D extruded shape from a 2D curve.

        Args:
            curve_points: List of (x, y) coordinates defining the closed curve
            z_offset: Z-axis offset for the base of the extrusion (default 0.0)

        Returns:
            TopoDS_Shape: Extruded 3D solid
        """
        if not curve_points or len(curve_points) < 3:
            raise ValueError("Need at least 3 points to create a shape")

        # Ensure curve is closed
        pts = list(curve_points)
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        # Create wire from line segments connecting consecutive points
        wire_maker = BRepBuilderAPI_MakeWire()

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]

            p1 = gp_Pnt(x1, y1, z_offset)
            p2 = gp_Pnt(x2, y2, z_offset)

            # Create edge between consecutive points
            edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            wire_maker.Add(edge)

        if not wire_maker.IsDone():
            raise RuntimeError("Failed to create wire from curve points")

        wire = wire_maker.Wire()

        # Create face from wire
        face = BRepBuilderAPI_MakeFace(wire).Face()

        # Extrude face along Z direction
        extrusion_vec = gp_Vec(0, 0, self.thickness)
        prism = BRepPrimAPI_MakePrism(face, extrusion_vec)

        shape = prism.Shape()
        return shape

    def create_compound_from_layers(self, layers: Dict[str, Any]) -> TopoDS_Compound:
        """
        Create a compound shape from front and back layers.

        Args:
            layers: Dictionary with 'front' and 'back' layer data

        Returns:
            TopoDS_Compound: Compound containing all extruded shapes
        """
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)

        self.current_shapes = []

        # Process front layer at z=0
        front_shapes = layers.get("front", [])
        for shape_data in front_shapes:
            shape_curve = shape_data.get("shape")
            if shape_curve is not None and len(shape_curve) > 0:
                points = [(pt[0], pt[1]) for pt in shape_curve]
                solid = self.create_shape_from_curve(points, z_offset=0.0)
                builder.Add(compound, solid)
                self.current_shapes.append({
                    "shape": solid,
                    "layer": "front",
                    "color": shape_data.get("color", "#de7cfc")
                })

        # Process back layer at z=thickness (offset to avoid overlap)
        back_shapes = layers.get("back", [])
        for shape_data in back_shapes:
            shape_curve = shape_data.get("shape")
            if shape_curve is not None and len(shape_curve) > 0:
                points = [(pt[0], pt[1]) for pt in shape_curve]
                # Offset back layer slightly in Z to avoid z-fighting
                solid = self.create_shape_from_curve(points, z_offset=self.thickness * 1.5)
                builder.Add(compound, solid)
                self.current_shapes.append({
                    "shape": solid,
                    "layer": "back",
                    "color": shape_data.get("color", "#de7cfc")
                })

        return compound

    def save_step(self, filename: str, shape: TopoDS_Shape = None) -> bool:
        """
        Export shape to STEP file.

        Args:
            filename: Output STEP file path
            shape: Shape to export

        Returns:
            bool: True if export successful
        """
        if shape is None:
            raise ValueError("No shape to export")

        # Ensure .step or .stp extension
        if not (filename.lower().endswith('.step') or filename.lower().endswith('.stp')):
            filename += '.step'

        # Create STEP writer
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)

        # Write to file
        status = writer.Write(filename)

        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to write STEP file: {filename}")

        return True

    def create_spiral_wrapped_compound(self, layers: Dict[str, Any],
                                       radius: float = 6.2055,
                                       thick: float = 0.1315,
                                       offset: float = 0.05,
                                       layer_pbh: float = 8.0,
                                       layer_ppw: float = 0.5,
                                       layer_ptc: float = 0.047,
                                       num_physical_layers: int = 4) -> TopoDS_Compound:
        """
        Create spiral-wrapped patterns with proper thickness extrusion.

        This method implements the complete spiral wrapping workflow:
        1. Generate spiral curves (spiral_i, spiral_c, spiral_o)
        2. Create spiral surfaces by extruding spiral_i and spiral_o
        3. Map patterns onto spiral surfaces
        4. Create 3D solids by extruding pattern surfaces along normal direction

        Args:
            layers: Dictionary with 'front' and 'back' layer data
            radius: Outer radius of spiral
            thick: Spiral band thickness (pitch)
            offset: Radial offset from centerline
            layer_pbh: Pattern base height
            layer_ppw: Pattern base width padding
            layer_ptc: Pattern thickness (copper thickness)
            num_physical_layers: Number of physical winding layers (NOT pattern count)

        Returns:
            TopoDS_Compound: Compound containing all wrapped pattern solids
        """
        # Get pattern shapes
        front_shapes = layers.get("front", [])
        back_shapes = layers.get("back", [])
        num_patterns = max(len(front_shapes), len(back_shapes))

        if num_patterns == 0:
            # Return empty compound
            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)
            return compound

        # Create spiral geometry with PHYSICAL layer count (not pattern count)
        self.spiral = Spiral(radius, thick, num_physical_layers, offset)

        print(f"Spiral geometry created:")
        print(f"  Physical layers: {num_physical_layers}")
        print(f"  Patterns per layer: {num_patterns // num_physical_layers}")
        print(f"  Total patterns: {num_patterns}")
        print(f"  Total arc length: {self.spiral.total_length:.3f} mm")
        print(f"  Arc per physical layer: {self.spiral.total_length / num_physical_layers:.3f} mm")

        # Extrusion height for spiral surfaces (wraps pattern vertically)
        extrusion_height = layer_pbh + layer_ppw * 2

        # Build compound
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        self.current_shapes = []

        # Calculate X coordinate range across ALL patterns to preserve relative positions
        all_x_coords = []
        for shape_data in front_shapes:
            shape_curve = shape_data.get("shape")
            if shape_curve is not None and len(shape_curve) > 0:
                all_x_coords.extend([pt[0] for pt in shape_curve])

        if all_x_coords:
            global_x_min = min(all_x_coords)
            global_x_max = max(all_x_coords)
            global_x_span = global_x_max - global_x_min

            # Patterns should preserve their original relative positions
            # No scaling - use X coordinates directly as arc length
            print(f"\nPattern coordinate analysis:")
            print(f"  Global X range: [{global_x_min:.3f}, {global_x_max:.3f}]")
            print(f"  Global X span: {global_x_span:.3f} mm")
            print(f"  Patterns will map to spiral preserving original spacing")
        else:
            global_x_min = 0.0

        # Process front layer patterns (map to outer spiral surface)
        for idx, shape_data in enumerate(front_shapes):
            shape_curve = shape_data.get("shape")
            if shape_curve is None or len(shape_curve) < 3:
                continue

            # Get this pattern's local X bounds
            pattern_x_coords = [pt[0] for pt in shape_curve]
            pattern_x_min = min(pattern_x_coords)
            pattern_x_max = max(pattern_x_coords)
            pattern_x_span = pattern_x_max - pattern_x_min

            # Normalize pattern to start at 0
            pattern_2d_raw = [(pt[0] - pattern_x_min, pt[1]) for pt in shape_curve]

            # Resample pattern with high point density for smooth boundaries
            # Higher point count creates smoother trimmed edges
            pattern_2d_resampled = self._resample_pattern_2d(pattern_2d_raw, target_points=200)

            # Debug: Check pattern bounds
            if idx == 0:  # Only print for first pattern
                x_coords = [pt[0] for pt in pattern_2d_resampled]
                y_coords = [pt[1] for pt in pattern_2d_resampled]
                print(f"\nFront pattern {idx}:")
                print(f"  Original X: [{pattern_x_min:.3f}, {pattern_x_max:.3f}]")
                print(f"  Pattern X span: {pattern_x_span:.3f} mm")
                print(f"  Normalized X range: [{min(x_coords):.3f}, {max(x_coords):.3f}]")
                print(f"  Y range: [{min(y_coords):.3f}, {max(y_coords):.3f}]")
                print(f"  Points: {len(pattern_2d_resampled)}")

            # Determine which physical layer this pattern belongs to
            layer_idx = idx // (num_patterns // num_physical_layers)

            # Calculate position within this physical layer
            patterns_per_layer = num_patterns // num_physical_layers
            pattern_idx_in_layer = idx % patterns_per_layer

            # Distribute patterns evenly within each physical layer
            arc_per_layer = self.spiral.total_length / num_physical_layers
            layer_start = layer_idx * arc_per_layer

            # Position this pattern within its layer
            pattern_position = layer_start + (pattern_idx_in_layer / patterns_per_layer) * arc_per_layer

            if idx == 0:
                print(f"  Layer {layer_idx}, pattern {pattern_idx_in_layer}/{patterns_per_layer}")
                print(f"  Arc position: {pattern_position:.3f} mm")
                print(f"  Arc per layer: {arc_per_layer:.3f} mm")

            # Create 3D shape from wrapped pattern
            # Pass 2D pattern data to create surface-based solid
            try:
                if idx == 0:
                    print(f"  [DEBUG] Creating front pattern with thickness={layer_ptc}")
                solid = self._create_wrapped_pattern_solid_from_surface(
                    pattern_2d_resampled,
                    pattern_position,
                    layer_ptc,
                    use_inner=False,  # Front goes on outer surface
                    debug=(idx == 0)  # Only debug first pattern
                )
                if solid is not None:
                    builder.Add(compound, solid)
                    self.current_shapes.append({
                        "shape": solid,
                        "layer": "front",
                        "color": shape_data.get("color", "#de7cfc")
                    })
                    if idx == 0:  # Debug first pattern only
                        print(f"  ✓ Front pattern {idx} created successfully")
                else:
                    if idx == 0:
                        print(f"  ✗ Front pattern {idx} returned None")
            except Exception as e:
                print(f"Warning: Failed to create front pattern {idx}: {e}")
                import traceback
                traceback.print_exc()

        # Process back layer patterns (map to inner spiral surface)
        for idx, shape_data in enumerate(back_shapes):
            shape_curve = shape_data.get("shape")
            if shape_curve is None or len(shape_curve) < 3:
                continue

            # Get this pattern's local X bounds
            pattern_x_coords = [pt[0] for pt in shape_curve]
            pattern_x_min = min(pattern_x_coords)

            # Normalize pattern to start at 0
            pattern_2d_raw = [(pt[0] - pattern_x_min, pt[1]) for pt in shape_curve]

            # Resample pattern with high point density for smooth boundaries
            pattern_2d_resampled = self._resample_pattern_2d(pattern_2d_raw, target_points=200)

            # Determine which physical layer this pattern belongs to
            layer_idx = idx // (num_patterns // num_physical_layers)

            # Calculate position within this physical layer
            patterns_per_layer = num_patterns // num_physical_layers
            pattern_idx_in_layer = idx % patterns_per_layer

            # Distribute patterns evenly within each physical layer
            arc_per_layer = self.spiral.total_length / num_physical_layers
            layer_start = layer_idx * arc_per_layer

            # Position this pattern within its layer
            pattern_position = layer_start + (pattern_idx_in_layer / patterns_per_layer) * arc_per_layer

            # Create 3D shape from wrapped pattern
            # Pass 2D pattern data to create surface-based solid
            try:
                solid = self._create_wrapped_pattern_solid_from_surface(
                    pattern_2d_resampled,
                    pattern_position,
                    layer_ptc,
                    use_inner=True  # Back goes on inner surface
                )
                if solid is not None:
                    builder.Add(compound, solid)
                    self.current_shapes.append({
                        "shape": solid,
                        "layer": "back",
                        "color": shape_data.get("color", "#de7cfc")
                    })
            except Exception as e:
                print(f"Warning: Failed to create back pattern {idx}: {e}")

        return compound

    def _resample_pattern_2d(self, pattern_2d: List[Tuple[float, float]],
                            target_points: int = 50) -> List[Tuple[float, float]]:
        """
        Resample a 2D pattern to have more uniform point distribution.

        Args:
            pattern_2d: Original pattern points
            target_points: Desired number of points

        Returns:
            Resampled pattern with uniform distribution
        """
        if len(pattern_2d) < 2:
            return pattern_2d

        # Calculate cumulative arc length
        lengths = [0.0]
        for i in range(len(pattern_2d) - 1):
            dx = pattern_2d[i+1][0] - pattern_2d[i][0]
            dy = pattern_2d[i+1][1] - pattern_2d[i][1]
            segment_length = np.sqrt(dx**2 + dy**2)
            lengths.append(lengths[-1] + segment_length)

        total_length = lengths[-1]
        if total_length < 1e-6:
            return pattern_2d

        # Resample at uniform intervals
        resampled = []
        for i in range(target_points):
            target_length = (i / (target_points - 1)) * total_length

            # Find segment
            for j in range(len(lengths) - 1):
                if lengths[j] <= target_length <= lengths[j+1]:
                    # Interpolate within segment
                    t = (target_length - lengths[j]) / (lengths[j+1] - lengths[j])
                    x = pattern_2d[j][0] + t * (pattern_2d[j+1][0] - pattern_2d[j][0])
                    y = pattern_2d[j][1] + t * (pattern_2d[j+1][1] - pattern_2d[j][1])
                    resampled.append((x, y))
                    break

        return resampled

    def _extrude_face_along_normal(self, face: TopoDS_Face, thickness: float,
                                   outward: bool = True, pattern_center_3d: tuple = None,
                                   debug: bool = False) -> TopoDS_Shape:
        """
        Extrude a face along its normal direction to create a solid.

        Args:
            face: The face to extrude
            thickness: Extrusion thickness (always positive)
            outward: If True, extrude outward from surface; if False, extrude inward
            pattern_center_3d: Optional (x, y, z) tuple of pattern center point for radial direction
            debug: Enable debug output

        Returns:
            TopoDS_Shape: Extruded solid
        """
        if debug:
            print(f"  [EXTRUDE] Starting extrusion: thickness={thickness}, outward={outward}")
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
            from OCC.Core.gp import gp_Vec
            from OCC.Core.GProp import GProp_GProps
            from OCC.Core.BRepGProp import brepgprop
            import numpy as np

            # Strategy: Extrude in RADIAL direction (away/toward spiral center)
            # The spiral is centered on the Z-axis, so radial direction is in XY plane

            # Use provided pattern center if available, otherwise use face center of mass
            if pattern_center_3d is not None:
                x_com, y_com, z_com = pattern_center_3d
                if debug:
                    print(f"    Using provided pattern center: ({x_com:.3f}, {y_com:.3f}, {z_com:.3f})")
            else:
                # Get center of mass of the face
                props_mass = GProp_GProps()
                brepgprop.SurfaceProperties(face, props_mass)
                center_of_mass = props_mass.CentreOfMass()
                x_com = center_of_mass.X()
                y_com = center_of_mass.Y()
                z_com = center_of_mass.Z()
                if debug:
                    print(f"    Using face center of mass: ({x_com:.3f}, {y_com:.3f}, {z_com:.3f})")

            # For a spiral surface, the radial direction at any point is
            # perpendicular to the Y-axis (vertical) and points toward/away from Y-axis
            # Spiral is in XZ plane, Y is vertical
            # Radial direction in XZ plane: normalize(x, 0, z)

            # Calculate radial direction (in XZ plane, away from Y-axis)
            radial_length = np.sqrt(x_com**2 + z_com**2)
            if radial_length < 1e-6:
                # Face is on Y-axis, use X direction
                radial_dir = gp_Vec(1.0, 0.0, 0.0)
            else:
                # Normalized radial direction in XZ plane
                radial_dir = gp_Vec(x_com / radial_length, 0.0, z_com / radial_length)

            # Determine extrusion direction
            # For outer layers: extrude outward (away from Y-axis)
            # For inner layers: extrude inward (toward Y-axis)
            sign = 1.0 if outward else -1.0
            extrusion_vec = gp_Vec(
                radial_dir.X() * thickness * sign,
                radial_dir.Y() * thickness * sign,
                radial_dir.Z() * thickness * sign  # Z component is 0
            )

            if debug:
                print(f"    Center of mass: ({x_com:.3f}, {y_com:.3f}, {z_com:.3f})")
                print(f"    Radial direction: ({radial_dir.X():.3f}, {radial_dir.Y():.3f}, {radial_dir.Z():.3f})")
                print(f"    Extrusion vector: ({extrusion_vec.X():.4f}, {extrusion_vec.Y():.4f}, {extrusion_vec.Z():.4f})")

            # Use prism extrusion to create solid
            prism = BRepPrimAPI_MakePrism(face, extrusion_vec)
            if prism.IsDone():
                solid = prism.Shape()
                if debug:
                    print(f"  ✓ Successfully extruded face to solid (thickness={thickness:.4f}, outward={outward})")
                return solid
            else:
                if debug:
                    print(f"  ✗ Prism extrusion failed")
                return face

        except Exception as e:
            if debug:
                print(f"  ✗ Extrusion failed: {e}")
                import traceback
                traceback.print_exc()
            return face

    def _create_wrapped_pattern_solid_from_surface(self, pattern_2d: List[Tuple[float, float]],
                                                  arc_offset: float,
                                                  thickness: float,
                                                  use_inner: bool = False,
                                                  debug: bool = False) -> TopoDS_Shape:
        """
        Create trimmed surface by:
        1. Creating spiral surface patch for pattern region
        2. Mapping pattern boundary to 3D
        3. Creating face from boundary (trimmed surface)
        4. Extruding face along normal to create solid

        Args:
            pattern_2d: 2D pattern coordinates (X, Y) - boundary points
            arc_offset: Arc length offset for this pattern on the spiral
            thickness: Copper thickness for extrusion (layer_ptc)
            use_inner: If True, use inner spiral; otherwise outer spiral

        Returns:
            TopoDS_Shape: Extruded solid or trimmed surface
        """
        if len(pattern_2d) < 3:
            return None

        from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                             BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire)
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
        from OCC.Core.TColgp import TColgp_Array1OfPnt
        from OCC.Core.GeomAbs import GeomAbs_C2
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections

        # Calculate pattern center for extrusion direction
        # Pattern center is at the middle of the pattern in both X and Y
        y_coords = [pt[1] for pt in pattern_2d]
        x_coords = [pt[0] for pt in pattern_2d]
        pattern_y_center = (max(y_coords) + min(y_coords)) / 2.0
        pattern_x_center = (max(x_coords) + min(x_coords)) / 2.0

        # Get 3D position of pattern center for extrusion direction
        arc_pos_center = arc_offset + pattern_x_center
        pt_3d_center = self.spiral.arc_to_xyz(arc_pos_center, use_inner=use_inner)
        if pt_3d_center:
            x_center, y_center, z_center = pt_3d_center
            pattern_center_3d = (x_center, y_center + pattern_y_center, z_center)
        else:
            pattern_center_3d = None

        # Step 1: Map pattern boundary points to 3D
        if debug:
            print(f"  Step 1: Mapping {len(pattern_2d)} boundary points to 3D...")
            if pattern_center_3d:
                print(f"  Pattern center 3D: ({pattern_center_3d[0]:.3f}, {pattern_center_3d[1]:.3f}, {pattern_center_3d[2]:.3f})")

        boundary_3d_pts = []
        for x_2d, y_2d in pattern_2d:
            arc_pos = arc_offset + x_2d
            pt_3d = self.spiral.arc_to_xyz(arc_pos, use_inner=use_inner)
            if pt_3d:
                x, y, z = pt_3d
                boundary_3d_pts.append(gp_Pnt(x, y + y_2d, z))
            else:
                if debug:
                    print(f"  ✗ arc_to_xyz failed for arc_pos={arc_pos:.3f}")
                return None

        if len(boundary_3d_pts) < 3:
            if debug:
                print(f"  ✗ Not enough 3D boundary points: {len(boundary_3d_pts)}")
            return None

        if debug:
            print(f"  ✓ Mapped {len(boundary_3d_pts)} boundary points to 3D")

        # Step 2: Create boundary wire from 3D points
        try:
            boundary_poly = BRepBuilderAPI_MakePolygon()
            for pt in boundary_3d_pts:
                boundary_poly.Add(pt)
            boundary_poly.Close()

            if not boundary_poly.IsDone():
                if debug:
                    print(f"  ✗ Boundary polygon creation failed")
                return None

            boundary_wire = boundary_poly.Wire()
            if debug:
                print(f"  ✓ Created boundary wire")
        except Exception as e:
            if debug:
                print(f"  ✗ Exception creating boundary wire: {e}")
            return None

        # Step 3: Create a LARGER ruled surface that covers the pattern with margin
        # Get pattern bounds
        x_coords = [pt[0] for pt in pattern_2d]
        y_coords = [pt[1] for pt in pattern_2d]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add margin to ensure surface is larger than pattern boundary
        x_margin = (x_max - x_min) * 0.2  # 20% margin on each side
        y_margin = (y_max - y_min) * 0.2
        x_min_surface = max(0.0, x_min - x_margin)  # Don't go below 0
        x_max_surface = x_max + x_margin
        y_min_surface = y_min - y_margin
        y_max_surface = y_max + y_margin

        # Clamp surface to spiral boundaries
        # arc_pos = arc_offset + x_val, and arc_pos must be <= total_length
        max_x_allowed = self.spiral.total_length - arc_offset
        if x_max_surface > max_x_allowed:
            if debug:
                print(f"  ⚠ Clamping x_max_surface from {x_max_surface:.3f} to {max_x_allowed:.3f}")
            x_max_surface = max_x_allowed

        # Ensure we still have valid range after clamping
        if x_min_surface >= x_max_surface:
            if debug:
                print(f"  ✗ Invalid surface X range after clamping: [{x_min_surface:.3f}, {x_max_surface:.3f}]")
            return None

        # Create profile curves at top and bottom of ENLARGED surface
        # Increase for smoother rendering
        n_profiles = 10  # Create more horizontal slices for smoother surface
        profile_wires = []

        for i in range(n_profiles):
            v = i / (n_profiles - 1) if n_profiles > 1 else 0.5
            y_level = y_min_surface + v * (y_max_surface - y_min_surface)

            # Create points along X at this Y level
            # Increase point density for smoother curves
            n_pts = 50
            pts_array = TColgp_Array1OfPnt(1, n_pts)

            for j in range(n_pts):
                u = j / (n_pts - 1)
                x_val = x_min_surface + u * (x_max_surface - x_min_surface)

                arc_pos = arc_offset + x_val
                pt_3d = self.spiral.arc_to_xyz(arc_pos, use_inner=use_inner)
                if pt_3d:
                    x, y, z = pt_3d
                    pts_array.SetValue(j + 1, gp_Pnt(x, y + y_level, z))
                else:
                    if debug:
                        print(f"  ✗ arc_to_xyz failed at profile {i}, point {j}: arc_pos={arc_pos:.3f}")
                        print(f"     arc_offset={arc_offset:.3f}, x_val={x_val:.3f}")
                        print(f"     Spiral total length: {self.spiral.total_length:.3f}")
                        print(f"     x_min_surface={x_min_surface:.3f}, x_max_surface={x_max_surface:.3f}")
                    return None

            # Create B-spline
            try:
                spline = GeomAPI_PointsToBSpline(pts_array, 3, 8, GeomAbs_C2).Curve()
                edge = BRepBuilderAPI_MakeEdge(spline).Edge()
                wire = BRepBuilderAPI_MakeWire(edge).Wire()
                profile_wires.append(wire)
            except Exception as e:
                if debug:
                    print(f"  ✗ Failed to create profile wire {i}: {e}")
                return None

        if len(profile_wires) < 2:
            if debug:
                print(f"  ✗ Not enough profile wires: {len(profile_wires)}")
            return None

        if debug:
            print(f"  Step 3: Creating ruled surface with {len(profile_wires)} profiles...")

        # Create ruled surface
        base_face = None
        try:
            loft = BRepOffsetAPI_ThruSections(False)  # False = surface, not solid
            for idx_wire, wire in enumerate(profile_wires):
                loft.AddWire(wire)

            if debug:
                print(f"  Added {len(profile_wires)} wires to ThruSections, building...")

            loft.Build()

            if loft.IsDone():
                # Extract ONLY the face from the result, not the entire shape with edges
                from OCC.Core.TopExp import TopExp_Explorer
                from OCC.Core.TopAbs import TopAbs_FACE

                ruled_surface = loft.Shape()
                explorer = TopExp_Explorer(ruled_surface, TopAbs_FACE)
                if explorer.More():
                    base_face = explorer.Current()
                    if debug:
                        print(f"  ✓ Created ruled surface face")
                else:
                    if debug:
                        print(f"  ✗ No face found in ruled surface")
            else:
                if debug:
                    print(f"  ✗ ThruSections failed")
        except Exception as e:
            if debug:
                print(f"  ✗ Exception creating ruled surface: {e}")

        # Try to trim the ruled surface with the boundary wire
        if base_face is not None:
            try:
                # Try to create a trimmed face using the boundary wire
                # Strategy: Project boundary wire onto the surface and use it for trimming
                from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
                from OCC.Core.BRep import BRep_Tool
                from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

                # Get the underlying surface
                face_surface = BRep_Tool.Surface(base_face)

                # Project boundary points onto the surface to get exact surface points
                # This ensures the wire lies exactly on the surface
                projected_boundary_pts = []
                projector = GeomAPI_ProjectPointOnSurf()
                max_dist = 0.0

                for pt in boundary_3d_pts:
                    projector.Init(pt, face_surface)
                    if projector.NbPoints() > 0:
                        # Get the actual point on the surface (not the original point)
                        projected_pt = projector.NearestPoint()
                        projected_boundary_pts.append(projected_pt)

                        # Track projection distance to check accuracy
                        dist = projector.LowerDistance()
                        if dist > max_dist:
                            max_dist = dist

                if len(projected_boundary_pts) >= 3:
                    if debug:
                        print(f"  Max projection distance: {max_dist:.6f} mm")
                        print(f"  Projected {len(projected_boundary_pts)} boundary points")

                    # Create a new boundary wire using the projected points
                    # These points are guaranteed to lie on the surface
                    try:
                        projected_poly = BRepBuilderAPI_MakePolygon()
                        for pt in projected_boundary_pts:
                            projected_poly.Add(pt)
                        projected_poly.Close()

                        if projected_poly.IsDone():
                            projected_wire = projected_poly.Wire()

                            if debug:
                                print(f"  Attempting to trim surface with projected wire...")

                            # Method: Use ShapeFix to ensure wire is on surface, then create face
                            try:
                                from OCC.Core.ShapeFix import ShapeFix_Wire

                                # Fix the wire to ensure it lies on the surface
                                fixer = ShapeFix_Wire()
                                fixer.Load(projected_wire)
                                fixer.SetSurface(face_surface)
                                fixer.Perform()
                                fixed_wire = fixer.Wire()

                                if debug:
                                    print(f"  Wire fixed and placed on surface")

                                # Create face from surface with wire as trimming boundary
                                face_maker = BRepBuilderAPI_MakeFace(face_surface, fixed_wire, False)

                                if face_maker.IsDone():
                                    trimmed_face = face_maker.Face()
                                    if debug:
                                        print(f"  ✓ Successfully created trimmed face")
                                        print(f"  [DEBUG] thickness={thickness}, will extrude: {thickness > 0}")

                                    # Extrude face if thickness is specified
                                    if thickness > 0:
                                        if debug:
                                            print(f"  [DEBUG] Calling extrusion function...")
                                        # Inner layers extrude inward, outer layers extrude outward
                                        outward = not use_inner
                                        return self._extrude_face_along_normal(trimmed_face, thickness, outward, pattern_center_3d, debug)
                                    return trimmed_face
                                else:
                                    if debug:
                                        print(f"  ✗ MakeFace failed even with fixed wire")
                            except Exception as e:
                                if debug:
                                    print(f"  ✗ Trimming failed: {e}")

                            # If trimming fails, return the base face (untrimmed but smooth)
                            # Better to have untrimmed smooth surface than triangulated mesh
                            if debug:
                                print(f"  ⚠ Returning untrimmed base face (trimming failed)")

                            # Extrude face if thickness is specified
                            if thickness > 0:
                                outward = not use_inner
                                return self._extrude_face_along_normal(base_face, thickness, outward, pattern_center_3d, debug)
                            return base_face
                    except Exception as e:
                        if debug:
                            print(f"  ✗ Exception creating trimmed face: {e}")

            except Exception:
                pass

        # If we reach here, return the untrimmed base face if available
        # This is better than Delaunay triangulation
        if base_face is not None:
            if debug:
                print(f"  ⚠ Returning untrimmed base face (no trimming attempted)")

            # Extrude face if thickness is specified
            if thickness > 0:
                outward = not use_inner
                return self._extrude_face_along_normal(base_face, thickness, outward, debug)
            return base_face

        # Last resort: Use Delaunay triangulation - creates many triangle faces
        if debug:
            print(f"  ⚠ Using Delaunay triangulation fallback")
        try:
            from scipy.spatial import Delaunay

            points_2d_array = np.array(pattern_2d)
            tri = Delaunay(points_2d_array)

            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)

            for simplex in tri.simplices:
                i1, i2, i3 = simplex
                try:
                    poly = BRepBuilderAPI_MakePolygon()
                    poly.Add(boundary_3d_pts[i1])
                    poly.Add(boundary_3d_pts[i2])
                    poly.Add(boundary_3d_pts[i3])
                    poly.Close()

                    if poly.IsDone():
                        wire = poly.Wire()
                        face_maker = BRepBuilderAPI_MakeFace(wire)
                        if face_maker.IsDone():
                            builder.Add(compound, face_maker.Face())
                except:
                    continue

            return compound

        except ImportError:
            # Final fallback: fan triangulation
            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)

            center = boundary_3d_pts[0]
            for i in range(1, len(boundary_3d_pts) - 1):
                try:
                    poly = BRepBuilderAPI_MakePolygon()
                    poly.Add(center)
                    poly.Add(boundary_3d_pts[i])
                    poly.Add(boundary_3d_pts[i + 1])
                    poly.Close()

                    if poly.IsDone():
                        wire = poly.Wire()
                        face_maker = BRepBuilderAPI_MakeFace(wire)
                        if face_maker.IsDone():
                            builder.Add(compound, face_maker.Face())
                except:
                    continue

            return compound

    def _create_wrapped_pattern_solid(self, points_3d: List[Tuple[float, float, float]],
                                     thickness: float,
                                     outward: bool = True) -> TopoDS_Shape:
        """
        Create a solid by creating faces between base and offset edges.

        Args:
            points_3d: List of (x, y, z) coordinates defining the wrapped pattern path
            thickness: Thickness of extrusion
            outward: If True, offset outward; otherwise inward

        Returns:
            TopoDS_Shape: Shell created from ruled surfaces
        """
        if len(points_3d) < 2:
            return None

        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
        from OCC.Core.GeomFill import GeomFill_BSplineCurves
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
        from OCC.Core.TColgp import TColgp_Array1OfPnt
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TopoDS import TopoDS_Shell, TopoDS_Compound
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing

        pts = list(points_3d)
        direction_factor = 1.0 if outward else -1.0

        # Create offset points
        offset_pts = []
        for pt in pts:
            x, y, z = pt
            r = np.sqrt(x**2 + z**2)
            if r < 1e-6:
                # Point at origin, offset in X direction
                x_new = x + thickness * direction_factor
                z_new = z
            else:
                # Offset radially
                nx = x / r
                nz = z / r
                x_new = x + thickness * direction_factor * nx
                z_new = z + thickness * direction_factor * nz
            offset_pts.append((x_new, y, z_new))

        # Create ruled surfaces between consecutive points
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)

        faces_created = 0

        for i in range(len(pts) - 1):
            try:
                # Create quadrilateral face between 4 points
                p1 = gp_Pnt(pts[i][0], pts[i][1], pts[i][2])
                p2 = gp_Pnt(pts[i+1][0], pts[i+1][1], pts[i+1][2])
                p3 = gp_Pnt(offset_pts[i+1][0], offset_pts[i+1][1], offset_pts[i+1][2])
                p4 = gp_Pnt(offset_pts[i][0], offset_pts[i][1], offset_pts[i][2])

                # Create edges forming a quadrilateral
                from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
                poly = BRepBuilderAPI_MakePolygon()
                poly.Add(p1)
                poly.Add(p2)
                poly.Add(p3)
                poly.Add(p4)
                poly.Close()

                if poly.IsDone():
                    wire = poly.Wire()
                    face_maker = BRepBuilderAPI_MakeFace(wire)
                    if face_maker.IsDone():
                        face = face_maker.Face()
                        builder.Add(compound, face)
                        faces_created += 1

            except Exception as e:
                if faces_created == 0:
                    print(f"    Face creation failed for segment {i}: {e}")
                continue

        if faces_created == 0:
            print(f"    ✗ No faces created from {len(pts)} points")
            return None

        # Return compound of faces (shell)
        return compound

    def _make_wire_from_points(self, points: List[Tuple[float, float, float]]):
        """Helper to create a smooth wire from 3D points using B-spline."""
        if len(points) < 2:
            return None

        # Create B-spline curve through points for smooth geometry
        num_points = len(points)
        point_array = TColgp_Array1OfPnt(1, num_points)

        for i, pt in enumerate(points):
            point_array.SetValue(i + 1, gp_Pnt(pt[0], pt[1], pt[2]))

        try:
            # Build B-spline curve with lower degree for better control
            spline_builder = GeomAPI_PointsToBSpline(point_array, 3, 8, GeomAbs_C2)
            if not spline_builder.IsDone():
                return None

            curve = spline_builder.Curve()
            edge = BRepBuilderAPI_MakeEdge(curve).Edge()
            wire = BRepBuilderAPI_MakeWire(edge).Wire()
            return wire
        except Exception as e:
            print(f"Warning: B-spline wire creation failed: {e}")
            return None

class Spiral:
    """
    Spiral geometry builder for wrapping patterns onto spiral surfaces.

    This class generates spiral curves (outer, center, inner) and provides
    methods to map 2D patterns onto 3D spiral surfaces.
    """

    def __init__(self, radius: float, thick: float, layer_count: int, offset: float,
                 samples_per_turn: int = 2000):
        """
        Initialize spiral geometry.

        Args:
            radius: Outer radius of the spiral
            thick: Spiral band thickness (pitch between turns)
            layer_count: Number of spiral turns
            offset: Radial offset from centerline for inner/outer curves
            samples_per_turn: Resolution for spiral sampling
        """
        self.radius = radius
        self.thick = thick
        self.layer_count = layer_count
        self.offset = offset
        self.samples_per_turn = max(800, int(samples_per_turn))

        # Pre-compute spiral parameters
        self._precompute_spiral()

    def _precompute_spiral(self):
        """Pre-compute spiral curve parameters and arc lengths."""
        # Spiral parameter: pitch = 2πb
        b = self.thick / (2.0 * np.pi)

        # Center line starts at radius - thick/2
        r0 = self.radius - self.thick / 2.0

        # Total angle for all turns
        theta_max = 2.0 * np.pi * self.layer_count

        # Sample points
        samples = max(int(self.samples_per_turn * self.layer_count), 2000)
        self.theta = np.linspace(0.0, theta_max, samples)

        # Radius as function of theta
        self.r = r0 - b * self.theta

        # Derivatives
        self.dr_dtheta = -b * np.ones_like(self.theta)

        # Cartesian coordinates (using x-z plane, y is vertical)
        self.x = self.r * np.cos(self.theta)
        self.z = self.r * np.sin(self.theta)

        # Tangent vector components
        self.dx_dtheta = self.dr_dtheta * np.cos(self.theta) - self.r * np.sin(self.theta)
        self.dz_dtheta = self.dr_dtheta * np.sin(self.theta) + self.r * np.cos(self.theta)

        # Compute arc lengths
        self.arc_lengths = np.zeros_like(self.theta)
        for i in range(1, len(self.theta)):
            dtheta = self.theta[i] - self.theta[i - 1]
            norm = np.hypot(self.dx_dtheta[i - 1], self.dz_dtheta[i - 1])
            ds = norm * dtheta if norm > 0.0 else 0.0
            self.arc_lengths[i] = self.arc_lengths[i - 1] + ds

        self.total_length = float(self.arc_lengths[-1])

    def _build_bspline_curve(self, x_coords: np.ndarray, y_coords: np.ndarray,
                            z_coords: np.ndarray) -> Geom_BSplineCurve:
        """
        Build a B-spline curve from coordinate arrays.

        Args:
            x_coords: X coordinates
            y_coords: Y coordinates
            z_coords: Z coordinates

        Returns:
            Geom_BSplineCurve: The constructed B-spline curve
        """
        num_points = len(x_coords)
        points = TColgp_Array1OfPnt(1, num_points)

        for idx in range(num_points):
            points.SetValue(idx + 1, gp_Pnt(float(x_coords[idx]),
                                           float(y_coords[idx]),
                                           float(z_coords[idx])))

        spline_builder = GeomAPI_PointsToBSpline(points, 3, 8, GeomAbs_C2)
        return spline_builder.Curve()

    def get_spiral_curves(self) -> Tuple[Geom_BSplineCurve, Geom_BSplineCurve, Geom_BSplineCurve]:
        """
        Generate three spiral curves: inner, center, outer.

        Returns:
            Tuple of (spiral_i, spiral_c, spiral_o) as Geom_BSplineCurve objects
        """
        # Center line curve (spiral_c)
        y_center = np.zeros_like(self.x)
        spiral_c = self._build_bspline_curve(self.x, y_center, self.z)

        # Calculate normal vectors (perpendicular to tangent in x-z plane)
        tangent_norm = np.hypot(self.dx_dtheta, self.dz_dtheta) + 1e-12
        tx = self.dx_dtheta / tangent_norm
        tz = self.dz_dtheta / tangent_norm

        # Normal vector (perpendicular to tangent, pointing outward)
        # In 2D (x-z plane): normal = (-tz, tx)
        nx = -tz
        nz = tx

        # Inner spiral (offset inward)
        x_inner = self.x - self.offset * nx
        z_inner = self.z - self.offset * nz
        spiral_i = self._build_bspline_curve(x_inner, y_center, z_inner)

        # Outer spiral (offset outward)
        x_outer = self.x + self.offset * nx
        z_outer = self.z + self.offset * nz
        spiral_o = self._build_bspline_curve(x_outer, y_center, z_outer)

        return spiral_i, spiral_c, spiral_o

    def create_spiral_surfaces(self, height: float) -> Tuple[TopoDS_Face, TopoDS_Face]:
        """
        Create spiral surfaces by extruding inner and outer spiral curves.

        Args:
            height: Extrusion height in Y direction

        Returns:
            Tuple of (surface_inner, surface_outer) as TopoDS_Face objects
        """
        spiral_i, _, spiral_o = self.get_spiral_curves()

        # Create edges from curves
        edge_inner = BRepBuilderAPI_MakeEdge(spiral_i).Edge()
        edge_outer = BRepBuilderAPI_MakeEdge(spiral_o).Edge()

        # Extrude along Y axis
        extrusion_vec = gp_Vec(0, height, 0)

        # Create surfaces by extruding the curves
        surface_inner = BRepPrimAPI_MakePrism(edge_inner, extrusion_vec).Shape()
        surface_outer = BRepPrimAPI_MakePrism(edge_outer, extrusion_vec).Shape()

        return surface_inner, surface_outer

    def evaluate_at_arc_length(self, s: float) -> Dict[str, Any]:
        """
        Evaluate spiral properties at a given arc length.

        Args:
            s: Arc length parameter

        Returns:
            Dictionary with position, tangent, and normal vectors
        """
        s_clamped = max(0.0, min(s, self.total_length))

        # Find index by binary search
        if s_clamped <= 0.0:
            idx = 1
            t = 0.0
        elif s_clamped >= self.total_length:
            idx = len(self.arc_lengths) - 1
            t = 1.0
        else:
            idx = int(np.searchsorted(self.arc_lengths, s_clamped))
            if idx == 0:
                idx = 1
            s0 = self.arc_lengths[idx - 1]
            s1 = self.arc_lengths[idx]
            t = (s_clamped - s0) / (s1 - s0) if abs(s1 - s0) > 1e-12 else 0.0

        # Interpolate position
        x_val = self.x[idx - 1] + t * (self.x[idx] - self.x[idx - 1])
        z_val = self.z[idx - 1] + t * (self.z[idx] - self.z[idx - 1])

        # Interpolate tangent
        dx = self.dx_dtheta[idx - 1] + t * (self.dx_dtheta[idx] - self.dx_dtheta[idx - 1])
        dz = self.dz_dtheta[idx - 1] + t * (self.dz_dtheta[idx] - self.dz_dtheta[idx - 1])

        # Normalize tangent
        tangent_norm = np.hypot(dx, dz)
        if tangent_norm > 1e-12:
            tangent = np.array([dx / tangent_norm, 0.0, dz / tangent_norm])
        else:
            tangent = np.array([0.0, 0.0, 1.0])

        # Normal (perpendicular to tangent in x-z plane)
        normal = np.array([-tangent[2], 0.0, tangent[0]])

        return {
            'x': float(x_val),
            'z': float(z_val),
            'tangent': tangent,
            'normal': normal,
        }

    def map_pattern_to_spiral(self, pattern_points: List[Tuple[float, float]],
                             arc_offset: float = 0.0,
                             y_offset: float = 0.0,
                             use_inner: bool = True) -> List[Tuple[float, float, float]]:
        """
        Map 2D pattern points onto the spiral surface.

        Args:
            pattern_points: List of (x, y) 2D pattern coordinates
            arc_offset: Starting arc length offset along spiral
            y_offset: Vertical offset
            use_inner: If True, map to inner surface; otherwise outer surface

        Returns:
            List of (x, y, z) 3D coordinates on spiral surface
        """
        radial_offset = -self.offset if use_inner else self.offset
        mapped_points = []

        for px, py in pattern_points:
            # px represents arc length along spiral
            # py represents vertical offset
            arc_length = arc_offset + px

            # Evaluate spiral at this arc length
            data = self.evaluate_at_arc_length(arc_length)

            # Base position on centerline
            base = np.array([data['x'], 0.0, data['z']])

            # Apply radial offset (to inner or outer surface)
            radial_vec = data['normal'] * radial_offset

            # Apply vertical offset
            vertical_vec = np.array([0.0, y_offset + py, 0.0])

            # Final position
            final_pos = base + radial_vec + vertical_vec

            mapped_points.append((float(final_pos[0]), float(final_pos[1]), float(final_pos[2])))

        return mapped_points

    def arc_to_xyz(self, arc_length: float, use_inner: bool = False) -> Optional[Tuple[float, float, float]]:
        """
        Convert arc length position to 3D XYZ coordinates on spiral.

        Args:
            arc_length: Arc length parameter along spiral
            use_inner: If True, use inner surface; otherwise outer surface

        Returns:
            Tuple of (x, y, z) coordinates, or None if invalid
        """
        if arc_length < 0.0 or arc_length > self.total_length:
            return None

        data = self.evaluate_at_arc_length(arc_length)
        radial_offset = -self.offset if use_inner else self.offset

        # Base position on centerline
        base = np.array([data['x'], 0.0, data['z']])

        # Apply radial offset
        radial_vec = data['normal'] * radial_offset
        final_pos = base + radial_vec

        return (float(final_pos[0]), float(final_pos[1]), float(final_pos[2]))

class StepViewer(QWidget):
    """3D viewer widget for OCCT shapes with save functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.exporter = StepExporter(thickness=0.047)
        self.current_compound = None
        self.layers_data = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create OCCT viewer
        self.viewer = qtViewer3d(self)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.viewer)

        # Improve rendering quality - set higher tessellation quality
        # This affects how smooth curves and surfaces are displayed
        try:
            # Get the default drawer for the context
            drawer = self.viewer._display.Context.DefaultDrawer()

            # Set deviation coefficient (lower = higher quality)
            # Default is 0.001, we use 0.0001 for 10x better quality
            drawer.SetDeviationCoefficient(0.0001)

            # Set deviation angle (lower = smoother curves)
            # Default is 0.5 radians, we use 0.1 for smoother curves
            drawer.SetDeviationAngle(0.1)

            # Set maximum chord length deviation
            drawer.SetMaximalChordialDeviation(0.01)

            # Update display
            self.viewer._display.Context.UpdateCurrentViewer()
        except Exception as e:
            print(f"Warning: Could not set rendering quality: {e}")

        # Enable keyboard focus for shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

        self.setMinimumSize(400, 400)

    def setLayers(self, layers: Dict[str, Any]):
        """
        Store layer data without immediately building 3D shapes.
        Call refresh_view() separately to generate and display shapes.

        Args:
            layers: Dictionary containing 'front' and 'back' layer shapes
        """
        self.layers_data = layers
        self.use_spiral_mode = False  # Default to flat extrusion

    def enable_spiral_mode(self, radius: float = 6.2055, thick: float = 0.1315,
                          offset: float = 0.05, layer_pbh: float = 8.0,
                          layer_ppw: float = 0.5, layer_ptc: float = 0.047,
                          num_physical_layers: int = 4):
        """
        Enable spiral wrapping mode with specified parameters.

        Args:
            radius: Outer radius of spiral
            thick: Spiral band thickness (pitch between turns)
            offset: Radial offset from centerline for inner/outer surfaces
            layer_pbh: Pattern base height
            layer_ppw: Pattern base width padding
            layer_ptc: Pattern thickness (copper thickness)
            num_physical_layers: Number of physical winding layers
        """
        self.use_spiral_mode = True
        self.spiral_params = {
            'radius': radius,
            'thick': thick,
            'offset': offset,
            'layer_pbh': layer_pbh,
            'layer_ppw': layer_ppw,
            'layer_ptc': layer_ptc,
            'num_physical_layers': num_physical_layers,
        }

    def refresh_view(self):
        """Rebuild the 3D view from current layer data."""
        if not self.layers_data:
            return

        # Calculate total shapes for progress tracking
        total_shapes = len(self.layers_data.get("front", [])) + len(self.layers_data.get("back", []))

        # Create progress dialog
        progress = None
        if total_shapes > 0:
            mode_text = "spiral-wrapped" if self.use_spiral_mode else "flat"
            progress = QProgressDialog(
                f"Building 3D {mode_text} shapes...",
                "Cancel",
                0,
                total_shapes + 2,  # +2 for compound creation and display update
                self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(500)  # Show after 500ms if still processing

        try:
            # Clear previous display
            if progress:
                progress.setLabelText("Clearing previous view...")
                progress.setValue(0)
                QCoreApplication.processEvents()

            self.viewer._display.Context.RemoveAll(True)

            # Create compound from layers
            if progress:
                mode_text = "spiral-wrapped" if self.use_spiral_mode else "flat"
                progress.setLabelText(f"Creating {mode_text} 3D shapes from layers...")
                progress.setValue(1)
                QCoreApplication.processEvents()

            # Choose mode based on flag
            if self.use_spiral_mode and hasattr(self, 'spiral_params'):
                self.current_compound = self.exporter.create_spiral_wrapped_compound(
                    self.layers_data,
                    **self.spiral_params
                )
            else:
                self.current_compound = self.exporter.create_compound_from_layers(self.layers_data)

            # Display each shape with its color
            current_shape_idx = 2
            for shape_info in self.exporter.current_shapes:
                if progress and progress.wasCanceled():
                    break

                shape = shape_info["shape"]
                color_str = shape_info["color"]
                layer_name = shape_info["layer"]

                if progress:
                    progress.setLabelText(f"Displaying {layer_name} layer shapes...")
                    progress.setValue(current_shape_idx)
                    QCoreApplication.processEvents()

                # Parse hex color
                color_str = color_str.lstrip('#')
                r = int(color_str[0:2], 16) / 255.0
                g = int(color_str[2:4], 16) / 255.0
                b = int(color_str[4:6], 16) / 255.0
                color = Quantity_Color(r, g, b, Quantity_TOC_RGB)

                # Display with color (shaded mode without edges)
                ais_shapes = self.viewer._display.DisplayShape(
                    shape,
                    color=color,
                    transparency=0.3 if layer_name == "back" else 0.1,
                    update=False
                )

                # Set to shaded mode without edges (mode 1 = shaded)
                # DisplayShape may return a list of AIS objects
                if ais_shapes:
                    if isinstance(ais_shapes, list):
                        for ais_shape in ais_shapes:
                            if ais_shape:
                                try:
                                    self.viewer._display.Context.SetDisplayMode(ais_shape, 1, False)
                                except:
                                    pass
                    else:
                        try:
                            self.viewer._display.Context.SetDisplayMode(ais_shapes, 1, False)
                        except:
                            pass

                current_shape_idx += 1

            # Fit all and update
            if progress:
                progress.setLabelText("Finalizing view...")
                progress.setValue(total_shapes + 2)
                QCoreApplication.processEvents()

            self.viewer._display.FitAll()
            self.viewer._display.Repaint()

        except Exception as e:
            print(f"Error rebuilding view: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up progress dialog
            if progress:
                progress.close()

    def save_step_file(self):
        """Save the current shapes to a STEP file."""
        if self.current_compound is None:
            QMessageBox.warning(self, "No Data", "No shapes to save. Please generate layers first.")
            return

        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save STEP File",
            "",
            "STEP Files (*.step *.stp);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.exporter.save_step(filename, self.current_compound)
            QMessageBox.information(self, "Success", f"STEP file saved successfully:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save STEP file:\n{str(e)}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for view control."""
        from PyQt5.QtCore import Qt

        key = event.key()

        # 1 - Top view
        if key == Qt.Key_1:
            self.viewer._display.View_Top()
            self.viewer._display.Repaint()

        # 2 - Fit all (center)
        elif key == Qt.Key_2:
            self.viewer._display.FitAll()
            self.viewer._display.Repaint()

        # - (minus) - Zoom out
        elif key == Qt.Key_Minus:
            self.viewer._display.ZoomFactor(0.8)
            self.viewer._display.Repaint()

        # = (equals) - Zoom in
        elif key == Qt.Key_Equal:
            self.viewer._display.ZoomFactor(1.25)
            self.viewer._display.Repaint()

        # Arrow keys - Pan
        elif key == Qt.Key_Up:
            self.viewer._display.Pan(0, 50)
            self.viewer._display.Repaint()

        elif key == Qt.Key_Down:
            self.viewer._display.Pan(0, -50)
            self.viewer._display.Repaint()

        elif key == Qt.Key_Left:
            self.viewer._display.Pan(50, 0)
            self.viewer._display.Repaint()

        elif key == Qt.Key_Right:
            self.viewer._display.Pan(-50, 0)
            self.viewer._display.Repaint()

        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    viewer = StepViewer()
    viewer.show()

    # Test with sample data
    test_layers = {
        "front": [
            {
                "shape": [(0, 0), (10, 0), (10, 5), (5, 8), (0, 5)],
                "color": "#ff6b6b"
            }
        ],
        "back": [
            {
                "shape": [(2, 1), (8, 1), (8, 4), (5, 6), (2, 4)],
                "color": "#4ecdc4"
            }
        ]
    }
    viewer.setLayers(test_layers)

    sys.exit(app.exec_())
