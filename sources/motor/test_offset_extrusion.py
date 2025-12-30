#!/usr/bin/env python3
"""
Minimal test for BRepOffsetAPI_MakeOffsetShape extrusion
"""

import sys
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
from OCC.Core.BRepOffset import BRepOffset_Skin
from OCC.Core.GeomAbs import GeomAbs_Intersection
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_SHELL

def test_offset_extrusion():
    """Test offset-based extrusion on a simple square face"""

    print("Testing BRepOffsetAPI_MakeOffsetShape extrusion...")

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

    # Test offset extrusion
    thickness = 0.047

    try:
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
            print(f"✓ Offset shape created")
            print(f"  Shape type: {shape.ShapeType()}")

            # Try to convert to solid if it's a shell
            if shape.ShapeType() == TopAbs_SHELL:
                print("  Converting shell to solid...")
                shell = topods.Shell(shape)
                solid_maker = BRepBuilderAPI_MakeSolid(shell)
                if solid_maker.IsDone():
                    solid = solid_maker.Solid()
                    print(f"  ✓ Converted to solid: {type(solid)}")
                    return True
                else:
                    print("  ✗ Failed to make solid from shell")
                    return False
            else:
                print(f"  Shape is type {shape.ShapeType()}")
                return True
        else:
            print("✗ MakeOffsetShape failed")
            return False

    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_offset_extrusion()
    sys.exit(0 if success else 1)
