import sys
import os
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QVBoxLayout, 
    QHBoxLayout, 
    QSlider, 
    QPushButton, 
    QWidget, 
    QLabel, 
    QFrame, 
    QSplitter,
    QGroupBox
)

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage

from log import CORELOG
# OCCT imports
try:
    # First load the Qt backend to prevent the "no backend has been imported" error
    from OCC.Display.backend import load_backend
    load_backend("pyqt5")
    
    # Now import the display module
    from OCC.Display import qtDisplay
    
    # Import other OpenCASCADE modules
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.AIS import AIS_Shape
    from OCC.Core.TopoDS import TopoDS_Shape
except ImportError as e:
    CORELOG.erro(f"Cannot import OpenCASCADE modules. Make sure PythonOCC is installed correctly.")
    CORELOG.erro(f"Import error details: {e}")
    sys.exit(1)

class ModelSignals(QObject):
    """Signal hub for model operations"""
    shape_changed = pyqtSignal(object)  # Emits shape data
    color_changed = pyqtSignal(tuple)   # Emits (r,g,b)
    transparency_changed = pyqtSignal(float)  # Emits transparency value 0-1
    texture_changed = pyqtSignal(str)   # Emits texture path
    
class OCCTViewer(qtDisplay.qtViewer3d):
    def __init__(self, *args):
        super(OCCTViewer, self).__init__(*args)
        self.InitDriver()
        self.display_shapes = []
        
    def display_shape(self, shape, color=None, transparency=None, update=True):
        if isinstance(shape, TopoDS_Shape):
            ais_shape = AIS_Shape(shape)
            if color:
                r, g, b = color
                ais_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
                ais_shape.SetColor(ais_color)
            if transparency is not None:
                ais_shape.SetTransparency(transparency)
            
            self._display.Context.Display(ais_shape, update)
            self.display_shapes.append(ais_shape)
            return ais_shape
        return None
        
    def clear(self):
        self._display.Context.RemoveAll(True)
        self.display_shapes = []
        
    def update_shape_color(self, shape_index, color):
        if 0 <= shape_index < len(self.display_shapes):
            shape = self.display_shapes[shape_index]
            r, g, b = color
            ais_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
            shape.SetColor(ais_color)
            self._display.Context.Redisplay(shape, True)
            
    def update_shape_transparency(self, shape_index, transparency):
        if 0 <= shape_index < len(self.display_shapes):
            shape = self.display_shapes[shape_index]
            shape.SetTransparency(transparency)
            self._display.Context.Redisplay(shape, True)

class TextureViewer(QFrame):
    def __init__(self, parent=None):
        super(TextureViewer, self).__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setMinimumSize(300, 200)
        
        self.layout = QVBoxLayout(self)
        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.img_label)
        
        # Default texture/placeholder
        self.set_placeholder_texture()
        
    def set_placeholder_texture(self):
        # Create a simple gradient texture as placeholder
        width, height = 256, 256
        image = QImage(width, height, QImage.Format_RGB888)
        
        for y in range(height):
            for x in range(width):
                r = int(255 * (x / width))
                g = int(255 * (y / height))
                b = 100
                image.setPixel(x, y, QtGui.qRgb(r, g, b))
        
        self.set_texture(image)
        
    def set_texture(self, image):
        if isinstance(image, QImage):
            pixmap = QPixmap.fromImage(image)
        elif isinstance(image, str) and os.path.isfile(image):
            pixmap = QPixmap(image)
        else:
            return
            
        self.img_label.setPixmap(pixmap.scaled(
            self.img_label.width(), 
            self.img_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

class SliderGroup(QWidget):
    """A reusable slider group with a label"""
    value_changed = pyqtSignal(str, int)  # name, value
    
    def __init__(self, name, min_val, max_val, default_val, parent=None):
        super().__init__(parent)
        self.name = name
        
        layout = QHBoxLayout(self)
        self.label = QLabel(name)
        self.label.setMinimumWidth(70)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default_val)
        
        self.value_label = QLabel(str(default_val))
        self.value_label.setMinimumWidth(30)
        
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        
        self.slider.valueChanged.connect(self._value_changed)
        
    def _value_changed(self, value):
        self.value_label.setText(str(value))
        self.value_changed.emit(self.name, value)
        
    def get_value(self):
        return self.slider.value()
        
    def set_value(self, value):
        self.slider.setValue(value)

class ButtonGroup(QWidget):
    """A group of buttons that can be customized"""
    button_clicked = pyqtSignal(str)  # button_name
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        if title:
            self.group_box = QGroupBox(title)
            self.button_layout = QHBoxLayout(self.group_box)
            self.layout.addWidget(self.group_box)
        else:
            self.button_layout = QHBoxLayout()
            self.layout.addLayout(self.button_layout)
        
        self.buttons = {}
        
    def add_button(self, name, display_text=None):
        """Add a button to the group"""
        if display_text is None:
            display_text = name
            
        button = QPushButton(display_text)
        self.button_layout.addWidget(button)
        self.buttons[name] = button
        
        button.clicked.connect(lambda: self.button_clicked.emit(name))
        return button
        
    def add_buttons(self, button_dict):
        """Add multiple buttons from a dict: {name: display_text}"""
        for name, display_text in button_dict.items():
            self.add_button(name, display_text)

class ControlPanel(QWidget):
    """
    A customizable control panel that can be configured with different controls
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(250)
        self.layout = QVBoxLayout(self)
        
        # Dictionary to store control groups
        self.controls = {}
        
    def add_button_group(self, group_id, title=""):
        """Add a group of buttons"""
        button_group = ButtonGroup(title, self)
        self.layout.addWidget(button_group)
        self.controls[group_id] = button_group
        return button_group
        
    def add_slider_group(self, group_id, title=""):
        """Add a group of sliders"""
        group = QGroupBox(title) if title else QWidget()
        group_layout = QVBoxLayout(group)
        self.layout.addWidget(group)
        
        # Store sliders in this group
        group.sliders = {}
        
        self.controls[group_id] = group
        return group
        
    def add_slider(self, group_id, name, min_val, max_val, default_val):
        """Add a slider to an existing group"""
        if group_id not in self.controls:
            return None
            
        group = self.controls[group_id]
        slider = SliderGroup(name, min_val, max_val, default_val, group)
        group.layout().addWidget(slider)
        
        # Add reference to the slider
        group.sliders[name] = slider
        return slider
        
    def add_spacer(self):
        """Add a spacer to the layout"""
        self.layout.addStretch(1)
        
    def get_slider_value(self, group_id, slider_name):
        """Get the value of a specific slider"""
        if group_id in self.controls:
            group = self.controls[group_id]
            if hasattr(group, 'sliders') and slider_name in group.sliders:
                return group.sliders[slider_name].get_value()
        return None

class ModelController:
    """Controller that handles model operations based on UI events"""
    
    def __init__(self, signals):
        self.signals = signals
        self.current_shape_type = None
        self.current_shape_params = None
        self.current_color = (0.7, 0.7, 0.9)  # Default color
        self.current_transparency = 0.0
        
    def create_box(self, x, y, z):
        """Create a box shape"""
        self.current_shape_type = 'box'
        self.current_shape_params = (x, y, z)
        shape = BRepPrimAPI_MakeBox(x, y, z).Shape()
        self.signals.shape_changed.emit({
            'type': 'box',
            'shape': shape,
            'params': (x, y, z),
            'color': self.current_color,
            'transparency': self.current_transparency
        })
        
    def create_cylinder(self, radius, height):
        """Create a cylinder shape"""
        self.current_shape_type = 'cylinder'
        self.current_shape_params = (radius, height)
        shape = BRepPrimAPI_MakeCylinder(radius, height).Shape()
        self.signals.shape_changed.emit({
            'type': 'cylinder',
            'shape': shape,
            'params': (radius, height),
            'color': self.current_color,
            'transparency': self.current_transparency
        })
        
    def update_color(self, r, g, b):
        """Update the color of the current shape"""
        self.current_color = (r, g, b)
        self.signals.color_changed.emit((r, g, b))
        
    def update_transparency(self, value):
        """Update the transparency of the current shape"""
        self.current_transparency = value
        self.signals.transparency_changed.emit(value)
        
    def update_texture(self, texture_path):
        """Update the texture"""
        self.signals.texture_changed.emit(texture_path)
        
    def recreate_current_shape(self):
        """Recreate the current shape with updated parameters"""
        if self.current_shape_type == 'box':
            x, y, z = self.current_shape_params
            self.create_box(x, y, z)
        elif self.current_shape_type == 'cylinder':
            radius, height = self.current_shape_params
            self.create_cylinder(radius, height)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCCT Model Viewer")
        self.setMinimumSize(1000, 700)
        
        # Create signals for model updates
        self.model_signals = ModelSignals()
        
        # Create the model controller
        self.model_controller = ModelController(self.model_signals)
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create UI components
        self.setup_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Show default shape
        self.model_controller.create_box(100, 60, 40)
        
    def setup_ui(self):
        # Left side: Control panel
        self.control_panel = ControlPanel()
        
        # Add shape selection buttons
        shape_buttons = self.control_panel.add_button_group("shapes", "Shape Selection")
        shape_buttons.add_button("box", "Box")
        shape_buttons.add_button("cylinder", "Cylinder")
        
        # Add size sliders
        size_group = self.control_panel.add_slider_group("size", "Size Parameters")
        self.control_panel.add_slider("size", "Size X", 10, 200, 100)
        self.control_panel.add_slider("size", "Size Y", 10, 200, 60)
        self.control_panel.add_slider("size", "Size Z", 10, 200, 40)
        
        # Add color sliders
        color_group = self.control_panel.add_slider_group("color", "Color")
        self.control_panel.add_slider("color", "Red", 0, 255, 150)
        self.control_panel.add_slider("color", "Green", 0, 255, 150)
        self.control_panel.add_slider("color", "Blue", 0, 255, 200)
        
        # Add transparency slider
        trans_group = self.control_panel.add_slider_group("appearance", "Appearance")
        self.control_panel.add_slider("appearance", "Transparency", 0, 10, 0)
        
        # Add texture button
        texture_group = self.control_panel.add_button_group("texture", "Texture")
        texture_group.add_button("load_texture", "Load Texture")
        
        # Add spacer
        self.control_panel.add_spacer()
        
        # Middle: OCCT viewer
        self.occ_viewer = OCCTViewer(self)
        
        # Right side: Texture viewer
        self.texture_viewer = TextureViewer()
        
        # Create a splitter for better UI control
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.control_panel)
        self.splitter.addWidget(self.occ_viewer)  
        self.splitter.addWidget(self.texture_viewer)
        
        # Set splitter proportions
        self.splitter.setSizes([250, 500, 250])
        
        self.main_layout.addWidget(self.splitter)
        
    def connect_signals(self):
        # Connect button signals
        shape_buttons = self.control_panel.controls["shapes"]
        shape_buttons.button_clicked.connect(self.handle_shape_button)
        
        texture_buttons = self.control_panel.controls["texture"]
        texture_buttons.button_clicked.connect(self.handle_texture_button)
        
        # Connect slider signals
        for group_name, group in self.control_panel.controls.items():
            if hasattr(group, 'sliders'):
                for name, slider in group.sliders.items():
                    slider.value_changed.connect(self.handle_slider_changed)
        
        # Connect model signals
        self.model_signals.shape_changed.connect(self.handle_shape_changed)
        self.model_signals.color_changed.connect(self.handle_color_changed)
        self.model_signals.transparency_changed.connect(self.handle_transparency_changed)
        self.model_signals.texture_changed.connect(self.handle_texture_changed)
        
    def handle_shape_button(self, button_name):
        """Handle shape button clicks"""
        if button_name == "box":
            x = self.control_panel.get_slider_value("size", "Size X")
            y = self.control_panel.get_slider_value("size", "Size Y")
            z = self.control_panel.get_slider_value("size", "Size Z")
            self.model_controller.create_box(x, y, z)
        elif button_name == "cylinder":
            radius = self.control_panel.get_slider_value("size", "Size X") / 2.0
            height = self.control_panel.get_slider_value("size", "Size Z")
            self.model_controller.create_cylinder(radius, height)
            
    def handle_texture_button(self, button_name):
        """Handle texture button clicks"""
        if button_name == "load_texture":
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Texture", "", "Images (*.png *.jpg *.bmp)"
            )
            
            if file_name:
                self.model_controller.update_texture(file_name)
                
    def handle_slider_changed(self, name, value):
        """Handle slider value changes"""
        # Handle size sliders
        if name in ["Size X", "Size Y", "Size Z"]:
            # Only recreate if we have a current shape
            if self.model_controller.current_shape_type:
                if self.model_controller.current_shape_type == 'box':
                    x = self.control_panel.get_slider_value("size", "Size X")
                    y = self.control_panel.get_slider_value("size", "Size Y")
                    z = self.control_panel.get_slider_value("size", "Size Z")
                    self.model_controller.current_shape_params = (x, y, z)
                elif self.model_controller.current_shape_type == 'cylinder':
                    radius = self.control_panel.get_slider_value("size", "Size X") / 2.0
                    height = self.control_panel.get_slider_value("size", "Size Z")
                    self.model_controller.current_shape_params = (radius, height)
                
                self.model_controller.recreate_current_shape()
                
        # Handle color sliders
        elif name in ["Red", "Green", "Blue"]:
            r = self.control_panel.get_slider_value("color", "Red") / 255.0
            g = self.control_panel.get_slider_value("color", "Green") / 255.0
            b = self.control_panel.get_slider_value("color", "Blue") / 255.0
            self.model_controller.update_color(r, g, b)
            
        # Handle transparency slider
        elif name == "Transparency":
            transparency = value / 10.0
            self.model_controller.update_transparency(transparency)
            
    def handle_shape_changed(self, shape_data):
        """Handle shape change events"""
        self.occ_viewer.clear()
        shape = shape_data['shape']
        color = shape_data['color']
        transparency = shape_data['transparency']
        
        self.occ_viewer.display_shape(shape, color=color, transparency=transparency)
        self.occ_viewer._display.FitAll()
        
    def handle_color_changed(self, color):
        """Handle color change events"""
        if not self.occ_viewer.display_shapes:
            return
            
        self.occ_viewer.update_shape_color(0, color)
        
    def handle_transparency_changed(self, transparency):
        """Handle transparency change events"""
        if not self.occ_viewer.display_shapes:
            return
            
        self.occ_viewer.update_shape_transparency(0, transparency)
        
    def handle_texture_changed(self, texture_path):
        """Handle texture change events"""
        self.texture_viewer.set_texture(texture_path)

def ui_main():
    window = MainWindow()
    window.show()
    
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
