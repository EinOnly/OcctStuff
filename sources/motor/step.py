"""
STEP file generator for motor pattern shapes using OpenCASCADE.
"""

from typing import List, Tuple
import os

# Geometric primitives
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax2, gp_Trsf

# Curve & interpolation tools
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2

# Shape construction
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism

# Boolean operations
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

# Topology
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound

# STEP file export
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone


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
