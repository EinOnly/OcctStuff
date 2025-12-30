#!/usr/bin/env python3
"""
Test script to verify surface extrusion implementation
"""

import sys
import numpy as np
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.BRepOffset import BRepOffset_Skin
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
from OCC.Core.GeomAbs import GeomAbs_Intersection

def test_simple_extrusion():
    """Test extruding a simple square face"""

    print("Testing surface extrusion...")

    # Create a simple square face
    pts = [
        gp_Pnt(0.0, 0.0, 0.0),
        gp_Pnt(1.0, 0.0, 0.0),
        gp_Pnt(1.0, 1.0, 0.0),
        gp_Pnt(0.0, 1.0, 0.0),
    ]

    poly = BRepBuilderAPI_MakePolygon()
    for pt in pts:
        poly.Add(pt)
    poly.Close()

    if not poly.IsDone():
        print("✗ Failed to create polygon")
        return False

    wire = poly.Wire()
    face_maker = BRepBuilderAPI_MakeFace(wire)

    if not face_maker.IsDone():
        print("✗ Failed to create face")
        return False

    face = face_maker.Face()
    print("✓ Created test face")

    # Test extrusion
    thickness = 0.047

    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        from OCC.Core.TopoDS import TopoDS_Shell
        from OCC.Core.TopAbs import TopAbs_SHELL

        offset_maker = BRepOffsetAPI_MakeOffsetShape()
        offset_maker.PerformByJoin(
            face,
            thickness,
            1e-6,
            BRepOffset_Skin,
            False,
            False,
            GeomAbs_Intersection
        )

        if offset_maker.IsDone():
            shape = offset_maker.Shape()
            print(f"✓ Successfully extruded face (thickness={thickness})")
            print(f"  Result type: {type(shape)}")
            print(f"  Shape type enum: {shape.ShapeType()}")

            # Convert shell to solid if needed
            if shape.ShapeType() == TopAbs_SHELL:
                print("  Converting shell to solid...")
                from OCC.Core.TopoDS import topods_Shell
                shell = topods_Shell(shape)
                solid_maker = BRepBuilderAPI_MakeSolid(shell)
                if solid_maker.IsDone():
                    solid = solid_maker.Solid()
                    print(f"  ✓ Converted to solid: {type(solid)}")
                    return True
                else:
                    print("  ✗ Failed to make solid from shell")
                    return False
            else:
                print("  Already a solid or other shape type")
                return True
        else:
            print("✗ MakeOffsetShape failed")
            return False

    except Exception as e:
        print(f"✗ Extrusion exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_extrusion()
    sys.exit(0 if success else 1)
