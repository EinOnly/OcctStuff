"""
STEP file generator for motor pattern shapes using OpenCASCADE.
"""

from typing import List, Tuple
import os

# Geometric primitives
from OCC.Core.gp import gp_Pnt, gp_Pnt2d, gp_Dir, gp_Dir2d, gp_Vec, gp_Ax1, gp_Ax2, gp_Trsf

# Curve & interpolation tools
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
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Shell
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.StlAPI import StlAPI_Writer

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

    def _mesh_shape(self,
                    shape: TopoDS_Shape | None = None,
                    linear_deflection: float = 0.1,
                    angular_deflection: float = 0.5,
                    relative: bool = False) -> TopoDS_Shape:
        """Create a triangulation of the given shape if needed."""
        if shape is None:
            shape = self.current_shape
        if shape is None:
            raise ValueError("No shape available for meshing")

        BRepMesh_IncrementalMesh(shape,
                                 linear_deflection,
                                 relative,
                                 angular_deflection,
                                 True)
        return shape

    def save_stl(self,
                 filename: str,
                 shape: TopoDS_Shape | None = None,
                 linear_deflection: float = 0.1,
                 angular_deflection: float = 0.5,
                 relative: bool = False,
                 ascii_mode: bool = True) -> bool:
        """Export the shape as an STL mesh."""
        shape = self._mesh_shape(shape, linear_deflection, angular_deflection, relative)
        if not filename.lower().endswith('.stl'):
            filename += '.stl'
        writer = StlAPI_Writer()
        writer.SetASCIIMode(ascii_mode)
        if not writer.Write(shape, filename):
            raise RuntimeError(f"Failed to write STL file: {filename}")
        return True

    def save_obj(self,
                 filename: str,
                 shape: TopoDS_Shape | None = None,
                 linear_deflection: float = 0.1,
                 angular_deflection: float = 0.5,
                 relative: bool = False) -> bool:
        """Export the shape as a Wavefront OBJ mesh."""
        shape = self._mesh_shape(shape, linear_deflection, angular_deflection, relative)
        if not filename.lower().endswith('.obj'):
            filename += '.obj'

        vertices: list[Tuple[float, float, float]] = []
        vertex_map: dict[Tuple[float, float, float], int] = {}
        faces: list[Tuple[int, int, int]] = []

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, loc)
            if triangulation is None:
                exp.Next()
                continue

            trans = loc.Transformation()
            local_index: dict[int, int] = {}

            for idx in range(1, triangulation.NbNodes() + 1):
                node = triangulation.Node(idx)
                pt = gp_Pnt(node.X(), node.Y(), node.Z())
                pt.Transform(trans)
                key = (round(pt.X(), 6), round(pt.Y(), 6), round(pt.Z(), 6))
                global_idx = vertex_map.get(key)
                if global_idx is None:
                    global_idx = len(vertices) + 1
                    vertices.append((pt.X(), pt.Y(), pt.Z()))
                    vertex_map[key] = global_idx
                local_index[idx] = global_idx

            for tri_idx in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(tri_idx)
                n1, n2, n3 = tri.Get()
                v1 = local_index[n1]
                v2 = local_index[n2]
                v3 = local_index[n3]
                if face.Orientation() == TopAbs_REVERSED:
                    faces.append((v1, v3, v2))
                else:
                    faces.append((v1, v2, v3))

            exp.Next()

        if not vertices or not faces:
            raise RuntimeError("Shape produced no triangulation data for OBJ export")

        with open(filename, 'w', encoding='utf-8') as obj_file:
            obj_file.write("# Generated by StepExporter\n")
            for vx, vy, vz in vertices:
                obj_file.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for f1, f2, f3 in faces:
                obj_file.write(f"f {f1} {f2} {f3}\n")

        return True

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
    """Manage offset spiral surfaces (outer/middle/inner) and build trimmed solids."""

    _EPS = 1e-9

    def __init__(self,
                 mapper: SpiralMapper,
                 max_length: float,
                 pattern_thickness: float,
                 tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.pattern_thickness = pattern_thickness

        base_radius = mapper.radius
        base_thick = mapper.thick
        turns = mapper.turns
        samples = mapper.samples_per_turn

        self.mapper_map: dict[str, SpiralMapper] = {}
        self.curve_map = {}
        self.surface_map = {}
        self.length_map = {}
        self.param_range = {}

        configs = [
            ('outer', mapper),
            ('middle', SpiralMapper(radius_value=max(0.0, base_radius - pattern_thickness / 2.0),
                                    thick_value=base_thick,
                                    turns=turns,
                                    samples_per_turn=samples)),
            ('inner', SpiralMapper(radius_value=max(0.0, base_radius - pattern_thickness),
                                   thick_value=base_thick,
                                   turns=turns,
                                   samples_per_turn=samples)),
        ]

        for key, mapper_obj in configs:
            self.mapper_map[key] = mapper_obj
            total_length = mapper_obj.get_total_length()
            effective_length = max(0.0, min(max_length, total_length))
            if effective_length <= self._EPS:
                effective_length = total_length
            self.length_map[key] = effective_length
            curve = mapper_obj.build_bspline_curve(effective_length)
            self.curve_map[key] = curve
            self.param_range[key] = (curve.FirstParameter(), curve.LastParameter())
            self.surface_map[key] = Geom_SurfaceOfLinearExtrusion(curve, gp_Dir(0.0, 1.0, 0.0))

    def _arc_length_to_param(self, key: str, arc_length: float) -> float:
        length = self.length_map[key]
        first, last = self.param_range[key]
        if length <= self._EPS:
            return first
        s = max(0.0, min(arc_length, length))
        ratio = s / length
        return first + ratio * (last - first)

    def _make_edge_on_surface(self, surface, p_start: gp_Pnt2d, p_end: gp_Pnt2d):
        dx = p_end.X() - p_start.X()
        dy = p_end.Y() - p_start.Y()
        if abs(dx) <= self._EPS and abs(dy) <= self._EPS:
            return None
        line2d = Geom2d_Line(p_start, gp_Dir2d(dx, dy))
        edge_builder = BRepBuilderAPI_MakeEdge(line2d, surface, p_start, p_end)
        if not edge_builder.IsDone():
            return None
        return edge_builder.Edge()

    def _compute_uv_points(self,
                           key: str,
                           shape_curve: List[Tuple[float, float]],
                           x_offset: float,
                           y_offset: float,
                           x_origin: float,
                           x_transform=None) -> List[gp_Pnt2d]:
        uv_points: List[gp_Pnt2d] = []
        for x, y in shape_curve:
            base_x = x_transform(x) if x_transform else x
            s_val = x_offset + (base_x - x_origin)
            param = self._arc_length_to_param(key, s_val)
            uv_points.append(gp_Pnt2d(param, y_offset + y))
        if uv_points:
            first = uv_points[0]
            if uv_points[-1].Distance(first) > self._EPS:
                uv_points.append(gp_Pnt2d(first.X(), first.Y()))
        return uv_points

    def _build_wire(self, key: str, uv_points: List[gp_Pnt2d]) -> 'TopoDS_Wire':
        surface = self.surface_map[key]
        wire_builder = BRepBuilderAPI_MakeWire()
        for idx in range(len(uv_points) - 1):
            edge = self._make_edge_on_surface(surface, uv_points[idx], uv_points[idx + 1])
            if edge is not None:
                wire_builder.Add(edge)
        return wire_builder.Wire()

    def _build_face(self,
                    key: str,
                    planar_points: List[Tuple[float, float]],
                    x_offset: float,
                    y_offset: float,
                    x_origin: float,
                    x_transform=None):
        uv_points = self._compute_uv_points(key, planar_points, x_offset, y_offset, x_origin, x_transform)
        wire = self._build_wire(key, uv_points)
        face_builder = BRepBuilderAPI_MakeFace(self.surface_map[key], wire, True)
        if not face_builder.IsDone():
            raise RuntimeError(f"Failed to create face on '{key}' surface")
        return face_builder.Face(), uv_points

    def _points_from_uv(self, key: str, uv_points: List[gp_Pnt2d]) -> List[gp_Pnt]:
        surface = self.surface_map[key]
        return [surface.Value(pt.X(), pt.Y()) for pt in uv_points[:-1]]

    @staticmethod
    def _prepare_planar_points(shape_curve: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        pts = list(shape_curve)
        if pts and abs(pts[0][0] - pts[-1][0]) <= SpiralSurfaceBuilder._EPS and abs(pts[0][1] - pts[-1][1]) <= SpiralSurfaceBuilder._EPS:
            pts = pts[:-1]
        return pts

    def _build_layer(self,
                     planar_points: List[Tuple[float, float]],
                     upper_key: str,
                     lower_key: str,
                     x_offset: float,
                     y_offset: float,
                     x_origin: float,
                     x_transform) -> TopoDS_Shape:
        upper_face, uv_upper = self._build_face(upper_key, planar_points, x_offset, y_offset, x_origin, x_transform)
        lower_face, uv_lower = self._build_face(lower_key, planar_points, x_offset, y_offset, x_origin, x_transform)

        upper_points = self._points_from_uv(upper_key, uv_upper)
        lower_points = self._points_from_uv(lower_key, uv_lower)

        shell_builder = BRep_Builder()
        shell = TopoDS_Shell()
        shell_builder.MakeShell(shell)
        shell_builder.Add(shell, upper_face)
        shell_builder.Add(shell, lower_face.Reversed())

        n = len(upper_points)
        for idx in range(n):
            next_idx = (idx + 1) % n
            quad = BRepBuilderAPI_MakePolygon()
            quad.Add(lower_points[idx])
            quad.Add(lower_points[next_idx])
            quad.Add(upper_points[next_idx])
            quad.Add(upper_points[idx])
            quad.Close()
            side_face = BRepBuilderAPI_MakeFace(quad.Wire()).Face()
            shell_builder.Add(shell, side_face)

        solid_builder = BRepBuilderAPI_MakeSolid()
        solid_builder.Add(shell)
        solid_builder.Build()
        if not solid_builder.IsDone():
            raise RuntimeError("Failed to convert shell into solid")
        return solid_builder.Solid()

    def create_thick_solid(self,
                           shape_curve: List[Tuple[float, float]],
                           thickness: float,
                           x_offset: float = 0.0,
                           y_offset: float = 0.0,
                           x_origin: float = 0.0,
                           x_transform=None,
                           layers: List[Tuple[str, str]] | None = None) -> TopoDS_Shape:
        planar_points = self._prepare_planar_points(shape_curve)
        if layers is None:
            layers = [('outer', 'middle'), ('middle', 'inner')]

        solids: List[TopoDS_Shape] = []
        for upper_key, lower_key in layers:
            solids.append(self._build_layer(planar_points, upper_key, lower_key, x_offset, y_offset, x_origin, x_transform))

        if not solids:
            raise RuntimeError("No layers generated for thick solid")
        if len(solids) == 1:
            return solids[0]

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for solid in solids:
            builder.Add(compound, solid)
        return compound


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
