from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Circ, gp_Pln
from OCC.Core.Geom import Geom_Circle, Geom_BSplineCurve
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
import numpy as np

# 导入 PyQt 模块
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt

# 控制器类，用于控制样条曲线
class SplineController:
    def __init__(self, display):
        self.display = display
        self.spline_points = [gp_Pnt(3, 0, 10), gp_Pnt(4, 0, 7), gp_Pnt(5, 0, 3), gp_Pnt(6, 0, 0)]
        self.circle_wire = None
        self.hex_wire = None
        self.spline_edge = None
        self.loft_shape = None
        
        # 控制点的限制范围
        self.x_range = (1, 8)  # X坐标范围
        self.z_range = (0, 10)  # Z坐标范围
    
    def update_control_point(self, index, x, z):
        """更新控制点位置"""
        # 更新索引为index的控制点的坐标
        self.spline_points[index].SetX(x)
        self.spline_points[index].SetZ(z)
        
        # 重新创建和更新样条曲线及放样形状
        self.update_spline()
    
    def update_spline(self):
        """更新样条曲线和放样形状"""
        # 如果已有样条曲线，移除它
        if self.spline_edge is not None:
            self.display.Context.Remove(self.spline_edge, False)
        if self.loft_shape is not None:
            self.display.Context.Remove(self.loft_shape, False)
        
        # 重新创建样条曲线
        points_array = TColgp_Array1OfPnt(1, len(self.spline_points))
        for i, point in enumerate(self.spline_points, 1):
            points_array.SetValue(i, point)
        
        spline_curve = GeomAPI_PointsToBSpline(points_array, 3, 3).Curve()
        self.spline_edge = BRepBuilderAPI_MakeEdge(spline_curve).Edge()
        
        # 重新创建放样形状
        loft_builder = BRepOffsetAPI_ThruSections(True, True)
        loft_builder.AddWire(self.circle_wire)
        loft_builder.AddWire(self.hex_wire)
        self.loft_shape = loft_builder.Shape()
        
        # 显示更新后的形状
        self.display.DisplayShape(self.spline_edge, color="RED", update=False)
        self.display.DisplayShape(self.loft_shape, color="YELLOW", transparency=0.5, update=True)

# 创建Qt控制面板
class ControlPanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.initUI()
    
    def initUI(self):
        """初始化UI界面"""
        self.setWindowTitle('样条曲线控制器')
        self.setMinimumWidth(300)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 为控制点1(索引1)创建滑动条
        point1_group = QVBoxLayout()
        point1_group.addWidget(QLabel('控制点 1:'))
        
        # X坐标滑动条
        p1_x_layout = QHBoxLayout()
        p1_x_layout.addWidget(QLabel('X:'))
        self.p1_x_slider = QSlider(Qt.Horizontal)
        self.p1_x_slider.setRange(self.controller.x_range[0] * 100, self.controller.x_range[1] * 100)
        self.p1_x_slider.setValue(int(self.controller.spline_points[1].X() * 100))
        self.p1_x_slider.valueChanged.connect(self.on_p1_x_changed)
        self.p1_x_label = QLabel(f"{self.controller.spline_points[1].X():.2f}")
        p1_x_layout.addWidget(self.p1_x_slider)
        p1_x_layout.addWidget(self.p1_x_label)
        point1_group.addLayout(p1_x_layout)
        
        # Z坐标滑动条
        p1_z_layout = QHBoxLayout()
        p1_z_layout.addWidget(QLabel('Z:'))
        self.p1_z_slider = QSlider(Qt.Horizontal)
        self.p1_z_slider.setRange(self.controller.z_range[0] * 100, self.controller.z_range[1] * 100)
        self.p1_z_slider.setValue(int(self.controller.spline_points[1].Z() * 100))
        self.p1_z_slider.valueChanged.connect(self.on_p1_z_changed)
        self.p1_z_label = QLabel(f"{self.controller.spline_points[1].Z():.2f}")
        p1_z_layout.addWidget(self.p1_z_slider)
        p1_z_layout.addWidget(self.p1_z_label)
        point1_group.addLayout(p1_z_layout)
        
        main_layout.addLayout(point1_group)
        
        # 分隔线
        main_layout.addWidget(QLabel(''))
        
        # 为控制点2(索引2)创建滑动条
        point2_group = QVBoxLayout()
        point2_group.addWidget(QLabel('控制点 2:'))
        
        # X坐标滑动条
        p2_x_layout = QHBoxLayout()
        p2_x_layout.addWidget(QLabel('X:'))
        self.p2_x_slider = QSlider(Qt.Horizontal)
        self.p2_x_slider.setRange(self.controller.x_range[0] * 100, self.controller.x_range[1] * 100)
        self.p2_x_slider.setValue(int(self.controller.spline_points[2].X() * 100))
        self.p2_x_slider.valueChanged.connect(self.on_p2_x_changed)
        self.p2_x_label = QLabel(f"{self.controller.spline_points[2].X():.2f}")
        p2_x_layout.addWidget(self.p2_x_slider)
        p2_x_layout.addWidget(self.p2_x_label)
        point2_group.addLayout(p2_x_layout)
        
        # Z坐标滑动条
        p2_z_layout = QHBoxLayout()
        p2_z_layout.addWidget(QLabel('Z:'))
        self.p2_z_slider = QSlider(Qt.Horizontal)
        self.p2_z_slider.setRange(self.controller.z_range[0] * 100, self.controller.z_range[1] * 100)
        self.p2_z_slider.setValue(int(self.controller.spline_points[2].Z() * 100))
        self.p2_z_slider.valueChanged.connect(self.on_p2_z_changed)
        self.p2_z_label = QLabel(f"{self.controller.spline_points[2].Z():.2f}")
        p2_z_layout.addWidget(self.p2_z_slider)
        p2_z_layout.addWidget(self.p2_z_label)
        point2_group.addLayout(p2_z_layout)
        
        main_layout.addLayout(point2_group)
        
        # 重置按钮
        reset_button = QPushButton('重置')
        reset_button.clicked.connect(self.reset_controls)
        main_layout.addWidget(reset_button)
        
        self.setLayout(main_layout)
    
    def on_p1_x_changed(self, value):
        """控制点1的X坐标变化"""
        x = value / 100.0
        self.p1_x_label.setText(f"{x:.2f}")
        self.controller.update_control_point(1, x, self.controller.spline_points[1].Z())
    
    def on_p1_z_changed(self, value):
        """控制点1的Z坐标变化"""
        z = value / 100.0
        self.p1_z_label.setText(f"{z:.2f}")
        self.controller.update_control_point(1, self.controller.spline_points[1].X(), z)
    
    def on_p2_x_changed(self, value):
        """控制点2的X坐标变化"""
        x = value / 100.0
        self.p2_x_label.setText(f"{x:.2f}")
        self.controller.update_control_point(2, x, self.controller.spline_points[2].Z())
    
    def on_p2_z_changed(self, value):
        """控制点2的Z坐标变化"""
        z = value / 100.0
        self.p2_z_label.setText(f"{z:.2f}")
        self.controller.update_control_point(2, self.controller.spline_points[2].X(), z)
    
    def reset_controls(self):
        """重置控制点到初始位置"""
        # 重置控制点1
        self.controller.spline_points[1] = gp_Pnt(4, 0, 7)
        self.p1_x_slider.setValue(400)
        self.p1_z_slider.setValue(700)
        self.p1_x_label.setText("4.00")
        self.p1_z_label.setText("7.00")
        
        # 重置控制点2
        self.controller.spline_points[2] = gp_Pnt(5, 0, 3)
        self.p2_x_slider.setValue(500)
        self.p2_z_slider.setValue(300)
        self.p2_x_label.setText("5.00")
        self.p2_z_label.setText("3.00")
        
        # 更新样条曲线
        self.controller.update_spline()

# 初始化显示窗口和Qt控件
display, start_display, *_ = init_display()

# 1. 创建圆形曲线 A
circle_axis = gp_Ax2(gp_Pnt(0, 0, 10), gp_Dir(0, 0, 1))  # 圆在Z=10的平面上
circle_geom = Geom_Circle(circle_axis, 3.0)
circle_edge = BRepBuilderAPI_MakeEdge(circle_geom).Edge()
circle_wire = BRepBuilderAPI_MakeWire(circle_edge).Wire()

# 2. 创建六边形曲线 B
hex_radius = 6.0
hex_z = 0.0
hex_points = []
for i in range(6):
    angle = np.pi / 3 * i
    hex_points.append(
        gp_Pnt(hex_radius * np.cos(angle), hex_radius * np.sin(angle), hex_z)
    )
hex_points.append(hex_points[0])  # 封闭曲线

hex_edges = []
for i in range(6):
    edge = BRepBuilderAPI_MakeEdge(hex_points[i], hex_points[i + 1]).Edge()
    hex_edges.append(edge)

hex_wire_builder = BRepBuilderAPI_MakeWire()
for edge in hex_edges:
    hex_wire_builder.Add(edge)
hex_wire = hex_wire_builder.Wire()

# 初始化控制器
controller = SplineController(display)
controller.circle_wire = circle_wire
controller.hex_wire = hex_wire

# 创建初始B样条曲线用于显示
points_array = TColgp_Array1OfPnt(1, len(controller.spline_points))
for i, point in enumerate(controller.spline_points, 1):
    points_array.SetValue(i, point)
spline_curve = GeomAPI_PointsToBSpline(points_array, 3, 3).Curve()
controller.spline_edge = BRepBuilderAPI_MakeEdge(spline_curve).Edge()

# 创建初始放样形状
loft_builder = BRepOffsetAPI_ThruSections(True, True)
loft_builder.AddWire(circle_wire)
loft_builder.AddWire(hex_wire)
controller.loft_shape = loft_builder.Shape()

# 显示所有元素
display.DisplayShape(circle_wire, color="BLUE", update=False)
display.DisplayShape(hex_wire, color="GREEN", update=False)
display.DisplayShape(controller.spline_edge, color="RED", update=False)
display.DisplayShape(controller.loft_shape, color="YELLOW", transparency=0.5, update=False)

# 创建Qt控制面板
control_panel = ControlPanel(controller)
control_panel.show()

# 设置合适的视图角度
display.View.SetProj(1, 1, 1)
display.FitAll()

print("使用滑动条调整控制点位置以修改样条曲线")
start_display()
