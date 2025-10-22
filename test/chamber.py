from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopoDS import topods_Edge, topods_Face
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Display.SimpleGui import init_display

# 创建立方体
box = BRepPrimAPI_MakeBox(100, 100, 100).Shape()

# 初始化倒角器
chamfer = BRepFilletAPI_MakeChamfer(box)

# 手动遍历所有边
edge_explorer = TopExp_Explorer(box, TopAbs_EDGE)
while edge_explorer.More():
    edge = topods_Edge(edge_explorer.Current())

    # 手动查找当前 edge 对应的 face（遍历所有面，看是否包含该 edge）
    faces = []
    face_explorer = TopExp_Explorer(box, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())

        # 遍历面上的所有边，看是否匹配
        inner_edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while inner_edge_explorer.More():
            inner_edge = topods_Edge(inner_edge_explorer.Current())
            if edge.IsSame(inner_edge):
                faces.append(face)
                break
            inner_edge_explorer.Next()

        face_explorer.Next()

    if len(faces) >= 1:
        chamfer.Add(5.0, 5.0, edge, faces[0])

    edge_explorer.Next()

# 获取倒角后的形状
chamfered_shape = chamfer.Shape()

# 可视化
display, start_display, *_ = init_display()
display.DisplayShape(chamfered_shape, update=True)
start_display()