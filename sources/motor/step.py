"""
STEP file generator for motor pattern shapes using OpenCASCADE.
"""

from typing import List, Tuple
import os

# Geometric primitives
from OCC.Core.gp import (
    gp_Pnt,
    gp_Pnt2d,
    gp_Dir,
    gp_Dir2d,
    gp_Vec,
    gp_Ax1,
    gp_Ax2,
    gp_Trsf,
)

# Curve & interpolation tools
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.Geom import Geom_SurfaceOfLinearExtrusion
from OCC.Core.Geom2d import Geom2d_Line

# Shape construction
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeSolid,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism

# Boolean operations
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

# Topology
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Shell, TopoDS_Wire
from OCC.Core.BRep import BRep_Builder

# STEP file export
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Core.GCPnts import GCPnts_AbscissaPoint

from curve import SpiralMapper


class StepExporter:
    """Generate 3D OCCT shapes from 2D motor pattern curves and export to STEP."""
    
    def __init__(self, thickness: float = 0.047):
        """
        Initialize STEP exporter.
        
        Args:
            thickness: Extrusion thickness in mm (default 0.047 mm for copper foil)
        """
        self.thickness = thickness
        self.current_shape = None
        
    def create_shape_from_curve(self, curve_points: List[Tuple[float, float]], z_offset: float = 0.0) -> TopoDS_Shape:
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
        if curve_points[0] != curve_points[-1]:
            curve_points = list(curve_points) + [curve_points[0]]
        
        # Create wire from line segments connecting consecutive points
        wire_maker = BRepBuilderAPI_MakeWire()
        
        for i in range(len(curve_points) - 1):
            x1, y1 = curve_points[i]
            x2, y2 = curve_points[i + 1]
            
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
        self.current_shape = shape
        return shape
    
    def create_symmetric_shape(self, left_curve: List[Tuple[float, float]], 
                               center_x: float) -> TopoDS_Shape:
        """
        Create a symmetric 3D shape by mirroring the left curve along vertical axis.
        
        Args:
            left_curve: Left side curve points (x, y)
            center_x: X coordinate of the symmetry axis (typically width/2)
            
        Returns:
            TopoDS_Shape: Extruded symmetric 3D solid
        """
        if not left_curve or len(left_curve) < 2:
            raise ValueError("Need at least 2 points for left curve")
        
        # Build closed polygon: left curve + mirrored right curve
        closed_curve = []
        
        # Add left curve points (bottom to top)
        for x, y in left_curve:
            closed_curve.append((x, y))
        
        # Add mirrored curve points in reverse (top to bottom)
        for x, y in reversed(left_curve):
            mirrored_x = 2 * center_x - x
            closed_curve.append((mirrored_x, y))
        
        # Create shape from closed curve
        return self.create_shape_from_curve(closed_curve)
    
    def save_step(self, filename: str, shape: TopoDS_Shape = None) -> bool:
        """
        Export shape to STEP file.
        
        Args:
            filename: Output STEP file path
            shape: Shape to export (uses current_shape if None)
            
        Returns:
            bool: True if export successful
        """
        if shape is None:
            shape = self.current_shape
            
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
    
    def get_bounding_box(self, shape: TopoDS_Shape = None) -> Tuple[float, float, float, float, float, float]:
        """
        Get bounding box of the shape.
        
        Args:
            shape: Shape to analyze (uses current_shape if None)
            
        Returns:
            Tuple of (xmin, ymin, zmin, xmax, ymax, zmax)
        """
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        
        if shape is None:
            shape = self.current_shape
            
        if shape is None:
            raise ValueError("No shape to analyze")
        
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return (xmin, ymin, zmin, xmax, ymax, zmax)

    @staticmethod
    def translate_shape(shape: TopoDS_Shape, dx: float, dy: float, dz: float = 0.0) -> TopoDS_Shape:
        """
        Create a translated copy of the given shape.
        """
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(dx, dy, dz))
        return BRepBuilderAPI_Transform(shape, trsf, True).Shape()

    @staticmethod
    def rotate_shape(shape: TopoDS_Shape,
                     angle_rad: float,
                     axis_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                     pivot: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> TopoDS_Shape:
        """
        Rotate a shape around an arbitrary axis passing through the specified pivot.
        """
        if abs(angle_rad) <= 1e-12:
            return shape
        axis_x, axis_y, axis_z = axis_dir
        if abs(axis_x) < 1e-12 and abs(axis_y) < 1e-12 and abs(axis_z) < 1e-12:
            return shape
        pivot_x, pivot_y, pivot_z = pivot
        axis = gp_Ax1(gp_Pnt(pivot_x, pivot_y, pivot_z), gp_Dir(axis_x, axis_y, axis_z))
        trsf = gp_Trsf()
        trsf.SetRotation(axis, angle_rad)
        return BRepBuilderAPI_Transform(shape, trsf, True).Shape()

    @staticmethod
    def rotate_shape_z(shape: TopoDS_Shape, angle_rad: float,
                       pivot_x: float = 0.0, pivot_y: float = 0.0, pivot_z: float = 0.0) -> TopoDS_Shape:
        """
        Rotate a shape around the Z axis passing through the specified pivot.
        """
        return StepExporter.rotate_shape(shape, angle_rad, (0.0, 0.0, 1.0), (pivot_x, pivot_y, pivot_z))

    @staticmethod
    def rotate_and_translate(shape: TopoDS_Shape,
                             angle_rad: float,
                             pivot: Tuple[float, float, float],
                             translation: Tuple[float, float, float],
                             axis_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> TopoDS_Shape:
        """
        Convenience helper that rotates a copy of shape around a specified axis (through pivot) and
        then applies a translation, returning the transformed copy.
        """
        trans_x, trans_y, trans_z = translation
        rotated = StepExporter.rotate_shape(shape, angle_rad, axis_dir, pivot)
        return StepExporter.translate_shape(rotated, trans_x, trans_y, trans_z)


class SpiralSurfaceBuilder:
    """
    Construct faces and solids by mapping planar (x, y) coordinates onto the spiral surface
    generated by extruding the spiral centreline along the global Y axis.
    """

    _EPS = 1e-9

    def __init__(self,
                 mapper: SpiralMapper,
                 max_length: float,
                 tolerance: float = 1e-6):
        self.mapper = mapper
        self.max_length = max(0.0, min(mapper.get_total_length(), max_length))
        if self.max_length <= 0.0:
            raise ValueError("SpiralSurfaceBuilder requires a positive max_length within spiral range")

        self.curve = mapper.build_bspline_curve(self.max_length)
        self.surface = Geom_SurfaceOfLinearExtrusion(self.curve, gp_Dir(0.0, 1.0, 0.0))
        self.curve_first = self.curve.FirstParameter()
        self.curve_last = self.curve.LastParameter()
        self.tolerance = tolerance

    def _length_to_param(self, length: float) -> float:
        length = max(0.0, min(self.max_length, length))
        if length <= 1e-12:
            return self.curve_first
        if length >= self.max_length - 1e-12:
            return self.curve_last
        try:
            abscissa = GCPnts_AbscissaPoint(self.curve, length, self.curve_first)
            return abscissa.Parameter()
        except Exception:
            ratio = length / self.max_length if self.max_length > 1e-12 else 0.0
            return self.curve_first + (self.curve_last - self.curve_first) * ratio

    def _make_edge_on_surface(self, surface, p_start: gp_Pnt2d, p_end: gp_Pnt2d):
        dx = p_end.X() - p_start.X()
        dy = p_end.Y() - p_start.Y()
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 1e-9:
            return None
        line2d = Geom2d_Line(p_start, gp_Dir2d(dx, dy))
        edge_builder = BRepBuilderAPI_MakeEdge(line2d, surface, 0.0, length)
        if not edge_builder.IsDone():
            raise RuntimeError("Failed to create edge on spiral surface")
        return edge_builder.Edge()

    def _compute_uv_points(self,
                           shape_curve: List[Tuple[float, float]],
                           x_offset: float,
                           y_offset: float,
                           x_origin: float,
                           x_transform=None) -> List[gp_Pnt2d]:
        uv_points: List[gp_Pnt2d] = []
        for x, y in shape_curve:
            base_x = x_transform(x) if x_transform else x
            s_val = x_offset + (base_x - x_origin)
            u_param = self._length_to_param(s_val)
            v_val = y_offset + y
            uv_points.append(gp_Pnt2d(u_param, v_val))

        if uv_points and uv_points[0].Distance(uv_points[-1]) > 1e-9:
            uv_points.append(uv_points[0])
        return uv_points

    def _build_wire(self, surface, uv_points: List[gp_Pnt2d]):
        wire_builder = BRepBuilderAPI_MakeWire()
        for idx in range(len(uv_points) - 1):
            edge = self._make_edge_on_surface(surface, uv_points[idx], uv_points[idx + 1])
            if edge is not None:
                wire_builder.Add(edge)
        wire = wire_builder.Wire()
        return wire

    def map_curve_points(self,
                         shape_curve: List[Tuple[float, float]],
                         x_offset: float = 0.0,
                         y_offset: float = 0.0,
                         x_origin: float = 0.0,
                         radial_offset: float = 0.0,
                         x_transform=None) -> List[gp_Pnt]:
        mapped_coords = self.mapper.map_curve_points(
            shape_curve,
            x_offset=x_offset,
            y_offset=y_offset,
            x_origin=x_origin,
            radial_offset=radial_offset,
            x_transform=x_transform
        )
        points: List[gp_Pnt] = []
        for x_val, y_val, z_val in mapped_coords:
            points.append(gp_Pnt(float(x_val), float(y_val), float(z_val)))
        return points

    @staticmethod
    def _strip_duplicate(points: List[gp_Pnt], eps: float = 1e-9) -> List[gp_Pnt]:
        if not points:
            return points
        if points[0].Distance(points[-1]) <= eps:
            return points[:-1]
        return points

    @staticmethod
    def _make_wire_from_points(points: List[gp_Pnt]) -> TopoDS_Wire:
        poly = BRepBuilderAPI_MakePolygon()
        for pt in points:
            poly.Add(pt)
        if points and points[0].Distance(points[-1]) > 1e-9:
            poly.Add(points[0])
        poly.Close()
        return poly.Wire()

    def create_face(self,
                    shape_curve: List[Tuple[float, float]],
                    x_offset: float = 0.0,
                    y_offset: float = 0.0,
                    x_origin: float = 0.0,
                    x_transform=None) -> TopoDS_Shape:
        """
        Map the planar closed curve onto the spiral surface and build a trimmed face.
        """
        if not shape_curve or len(shape_curve) < 3:
            raise ValueError("shape_curve must contain at least 3 points")

        uv_points = self._compute_uv_points(shape_curve, x_offset, y_offset, x_origin, x_transform)
        wire = self._build_wire(self.surface, uv_points)
        face_builder = BRepBuilderAPI_MakeFace(self.surface, wire, True)
        if not face_builder.IsDone():
            raise RuntimeError("Failed to create face on spiral surface")
        return face_builder.Face()

    def create_thick_solid(self,
                           shape_curve: List[Tuple[float, float]],
                           thickness: float,
                           x_offset: float = 0.0,
                           y_offset: float = 0.0,
                           x_origin: float = 0.0,
                           x_transform=None,
                           radial_direction: float = 1.0) -> TopoDS_Shape:
        if abs(thickness) <= 1e-12:
            raise ValueError("Thickness must be non-zero for solid creation")

        radial_sign = 1.0 if radial_direction >= 0 else -1.0
        half_thickness = thickness / 2.0

        top_points = self.map_curve_points(
            shape_curve,
            x_offset=x_offset,
            y_offset=y_offset,
            x_origin=x_origin,
            radial_offset=radial_sign * half_thickness,
            x_transform=x_transform
        )
        bottom_points = self.map_curve_points(
            shape_curve,
            x_offset=x_offset,
            y_offset=y_offset,
            x_origin=x_origin,
            radial_offset=-radial_sign * half_thickness,
            x_transform=x_transform
        )

        top_points = self._strip_duplicate(top_points)
        bottom_points = self._strip_duplicate(bottom_points)

        if len(top_points) < 3 or len(bottom_points) < 3:
            raise RuntimeError("Not enough points to build thick solid")

        top_wire = self._make_wire_from_points(top_points)
        bottom_wire = self._make_wire_from_points(list(reversed(bottom_points)))

        top_face = BRepBuilderAPI_MakeFace(top_wire).Face()
        bottom_face = BRepBuilderAPI_MakeFace(bottom_wire).Face()

        shell_builder = BRep_Builder()
        shell = TopoDS_Shell()
        shell_builder.MakeShell(shell)
        shell_builder.Add(shell, top_face)
        shell_builder.Add(shell, bottom_face)

        n = len(top_points)
        for idx in range(n):
            next_idx = (idx + 1) % n
            p_top_i = top_points[idx]
            p_top_j = top_points[next_idx]
            p_bot_i = bottom_points[idx]
            p_bot_j = bottom_points[next_idx]

            side_wire = self._make_wire_from_points([p_bot_i, p_bot_j, p_top_j, p_top_i])
            side_face = BRepBuilderAPI_MakeFace(side_wire).Face()
            shell_builder.Add(shell, side_face)

        solid_builder = BRepBuilderAPI_MakeSolid()
        solid_builder.Add(shell)
        solid_builder.Build()
        if not solid_builder.IsDone():
            raise RuntimeError("Failed to convert shell into solid")
        return solid_builder.Solid()


def create_motor_pattern_shape(pattern, offset: float, space: float, thickness: float = 0.047) -> TopoDS_Shape:
    """
    Convenience function to create a 3D shape from a Pattern object using GetShape().
    
    Args:
        pattern: Pattern object with GetShape() method
        offset: Assembly offset parameter
        space: Spacing parameter
        thickness: Extrusion thickness in mm
        
    Returns:
        TopoDS_Shape: 3D extruded shape
    """
    exporter = StepExporter(thickness=thickness)
    closed_curve = pattern.GetShape(offset, space)
    shape = exporter.create_shape_from_curve(closed_curve)
    return shape


if __name__ == "__main__":
    # Test with a simple pattern
    from pattern import Pattern
    
    # Create a test pattern
    pattern = Pattern(width=2.0, height=3.0)
    pattern.SetVariable('exponent', 1.5)
    
    # Create and export shape
    exporter = StepExporter(thickness=0.047)
    left_curve = pattern.GetCurve()
    shape = exporter.create_symmetric_shape(left_curve, pattern.width / 2.0)
    
    # Save to STEP file
    output_file = "test_pattern.step"
    exporter.save_step(output_file)
    print(f"Exported shape to {output_file}")
    
    # Print bounding box
    bbox = exporter.get_bounding_box()
    print(f"Bounding box: {bbox}")
