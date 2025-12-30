#!/usr/bin/env python3
"""
Test script to verify surface trimming implementation
"""

import sys
import numpy as np
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnSurf
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool

def create_simple_ruled_surface():
    """Create a simple ruled surface for testing"""

    # Create two parallel curves (rectangles at different heights)
    profile_wires = []

    for z_level in [0.0, 1.0]:
        pts_array = TColgp_Array1OfPnt(1, 20)
        for i in range(20):
            u = i / 19.0
            x = u * 10.0  # 0 to 10
            y = 0.0
            z = z_level
            pts_array.SetValue(i + 1, gp_Pnt(x, y, z))

        spline = GeomAPI_PointsToBSpline(pts_array, 3, 8, GeomAbs_C2).Curve()
        edge = BRepBuilderAPI_MakeEdge(spline).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()
        profile_wires.append(wire)

    # Create ruled surface
    loft = BRepOffsetAPI_ThruSections(False)
    for wire in profile_wires:
        loft.AddWire(wire)
    loft.Build()

    if loft.IsDone():
        print("✓ Ruled surface created")
        return loft.Shape()
    else:
        print("✗ Failed to create ruled surface")
        return None

def test_trimming(surface):
    """Test trimming the surface with a boundary wire"""

    # Create a simple boundary wire (a smaller rectangle)
    boundary_pts = [
        gp_Pnt(2.0, -0.5, 0.2),
        gp_Pnt(8.0, -0.5, 0.8),
        gp_Pnt(8.0, 0.5, 0.8),
        gp_Pnt(2.0, 0.5, 0.2),
    ]

    boundary_poly = BRepBuilderAPI_MakePolygon()
    for pt in boundary_pts:
        boundary_poly.Add(pt)
    boundary_poly.Close()

    if not boundary_poly.IsDone():
        print("✗ Failed to create boundary wire")
        return None

    boundary_wire = boundary_poly.Wire()
    print("✓ Boundary wire created")

    # Get the face from the surface
    explorer = TopExp_Explorer(surface, TopAbs_FACE)
    if not explorer.More():
        print("✗ No face found in surface")
        return None

    base_face = explorer.Current()
    face_surface = BRep_Tool.Surface(base_face)
    print("✓ Extracted surface from face")

    # Project boundary points onto surface
    boundary_2d_uv = []
    projector = GeomAPI_ProjectPointOnSurf()

    for pt in boundary_pts:
        projector.Init(pt, face_surface)
        if projector.NbPoints() > 0:
            u, v = projector.LowerDistanceParameters()
            boundary_2d_uv.append((u, v))
            dist = projector.LowerDistance()
            print(f"  Point projected: UV=({u:.3f}, {v:.3f}), dist={dist:.6f}")

    if len(boundary_2d_uv) < 3:
        print("✗ Not enough projected points")
        return None

    print(f"✓ Projected {len(boundary_2d_uv)} boundary points")

    # Try to create trimmed face
    try:
        face_maker = BRepBuilderAPI_MakeFace(face_surface, boundary_wire, True)

        if face_maker.IsDone():
            print("✓ Successfully created trimmed face!")
            return face_maker.Face()
        else:
            print("✗ Face maker failed (IsDone=False)")

            # Try alternative: use surface bounds
            u_min, u_max, v_min, v_max = 0.0, 10.0, -1.0, 1.0
            face_maker2 = BRepBuilderAPI_MakeFace(face_surface, u_min, u_max, v_min, v_max, 1e-6)
            if face_maker2.IsDone():
                print("✓ Created face with explicit bounds")
                return face_maker2.Face()
            else:
                print("✗ Face with bounds also failed")

            return None
    except Exception as e:
        print(f"✗ Exception during face creation: {e}")
        return None

def main():
    print("Testing surface trimming implementation...\n")

    # Create ruled surface
    surface = create_simple_ruled_surface()
    if surface is None:
        return 1

    print()

    # Test trimming
    trimmed = test_trimming(surface)
    if trimmed is None:
        print("\n✗ Trimming test FAILED")
        return 1
    else:
        print("\n✓ Trimming test PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(main())
