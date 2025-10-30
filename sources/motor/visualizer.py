import math
import os
from pattern import Pattern
from assamly import AssemblyBuilder
from step import StepExporter
from PyQt5.QtWidgets import (QWidget, QLabel, QSlider, QLineEdit, QHBoxLayout, QVBoxLayout, QGridLayout, QApplication, QPushButton, QFileDialog, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QDoubleValidator
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# OCCT Display
from OCC.Display.backend import load_backend
load_backend('pyqt5')
from OCC.Display.qtDisplay import qtViewer3d

class Slider(QWidget):
    """A slider widget with label, slider, and input box for low-latency async updates"""
    valueChanged = pyqtSignal(str, float)  # Signal emits (label, value)
    
    def __init__(self, label, min_val, max_val, initial, step=0.1):
        super().__init__()
        self.label_text = label
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = max(self.min_val, min(self.max_val, initial))
        
        # Calculate slider resolution based on step
        self.resolution = self._calculate_resolution(min_val, max_val, step)
        
        # Create UI components
        self.label = QLabel(f"{label}:")
        self.label.setMinimumWidth(60)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.label.setStyleSheet("QLabel { color: #000; font-size: 12px; border: none; }")
        
        # QSlider (integer-based)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.resolution)
        self.slider.setValue(self._value_to_slider(self.value))
        self.slider.setMinimumWidth(150)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background: #d0d0d0;
                margin: 0px;
            }
            QSlider::sub-page:horizontal {
                background: #4a90e2;
                height: 4px;
                border: none;
            }
            QSlider::add-page:horizontal {
                background: #d0d0d0;
                height: 4px;
                border: none;
            }
            QSlider::handle:horizontal {
                background: #5c5c5c;
                border: none;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #787878;
            }
        """)
        
        # QLineEdit with double validator
        self.input_box = QLineEdit()
        self.input_box.setText(f"{self.value:.5f}")
        self.input_box.setMaximumWidth(80)
        self.input_box.setAlignment(Qt.AlignLeft)
        self.input_box.setStyleSheet("""
            QLineEdit { 
                border: none;
                padding: 3px; 
                background-color: white;
                color: #000;
                font-size: 12px;
            }
        """)
        validator = QDoubleValidator(min_val, max_val, 5)
        self.input_box.setValidator(validator)
        
        # Layout
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)  # Add stretch factor
        layout.addWidget(self.input_box)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)  # Add spacing between widgets
        self.setLayout(layout)
        
        # Connect signals for async low-latency updates
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.input_box.editingFinished.connect(self._on_input_finished)
        
        # Debounce timer for input box real-time updates
        self.input_timer = QTimer()
        self.input_timer.setSingleShot(True)
        self.input_timer.timeout.connect(self._on_input_timeout)
        self.input_box.textChanged.connect(self._on_input_text_changed)
        
    def _calculate_resolution(self, min_val, max_val, step):
        """Compute slider resolution ensuring we can reach the max value"""
        span = max_val - min_val
        if step <= 0 or span <= 0:
            return 0
        return max(1, int(math.ceil(span / step)))

    def _update_slider_bounds(self):
        """Refresh slider integer bounds after range/step updates"""
        self.resolution = self._calculate_resolution(self.min_val, self.max_val, self.step)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.resolution)

    def _value_to_slider(self, value):
        """Convert real value to slider position"""
        if self.step <= 0:
            return 0
        position = int(round((value - self.min_val) / self.step))
        return max(0, min(self.resolution, position))
    
    def _slider_to_value(self, slider_pos):
        """Convert slider position to real value"""
        return self.min_val + slider_pos * self.step
    
    def _on_slider_changed(self, slider_pos):
        """Handle slider movement - immediate update"""
        new_value = self._slider_to_value(slider_pos)
        self.value = new_value
        self.input_box.setText(f"{new_value:.5f}")
        self.valueChanged.emit(self.label_text, new_value)
    
    def _on_input_text_changed(self, text):
        """Handle input box text change - debounced update"""
        self.input_timer.stop()
        self.input_timer.start(300)  # 300ms debounce
    
    def _on_input_timeout(self):
        """Handle debounced input box update"""
        try:
            new_value = float(self.input_box.text())
            if self.min_val <= new_value <= self.max_val:
                self.value = new_value
                self.slider.setValue(self._value_to_slider(new_value))
                self.valueChanged.emit(self.label_text, new_value)
        except ValueError:
            # Invalid input, revert to current value
            self.input_box.setText(f"{self.value:.5f}")
    
    def _on_input_finished(self):
        """Handle input box editing finished (Enter or focus lost)"""
        try:
            new_value = float(self.input_box.text())
            if self.min_val <= new_value <= self.max_val:
                self.value = new_value
                self.slider.setValue(self._value_to_slider(new_value))
                self.valueChanged.emit(self.label_text, new_value)
            else:
                # Out of range, revert
                self.input_box.setText(f"{self.value:.5f}")
        except ValueError:
            # Invalid input, revert to current value
            self.input_box.setText(f"{self.value:.5f}")
    
    def set_value(self, new_value):
        """Programmatically set the slider value, clamping to valid range"""
        # Clamp value to valid range instead of raising error
        clamped_value = max(self.min_val, min(self.max_val, new_value))
        self.value = clamped_value
        self.slider.setValue(self._value_to_slider(clamped_value))
        self.input_box.setText(f"{clamped_value:.5f}")
    
    def set_range(self, new_min, new_max, new_step=None):
        """Update the slider's min/max range (and optionally step)"""
        self.min_val = new_min
        self.max_val = new_max
        if new_step is not None and new_step > 0:
            self.step = new_step
        self._update_slider_bounds()
        # Update validator
        self.input_box.setValidator(QDoubleValidator(new_min, new_max, 5))
        # Clamp current value to new range
        if self.value < new_min:
            self.set_value(new_min)
        elif self.value > new_max:
            self.set_value(new_max)
        else:
            # Just update the slider position
            self.slider.setValue(self._value_to_slider(self.value))
    
    def get_value(self):
        """Get current slider value"""
        return self.value

class Visualizer(QWidget):
    def __init__(self, height=200, multiple=2.5, spacing=10):

        super().__init__()
        self.height = height
        self.spacing = spacing
        self.multiple = multiple
        self.sliders = []
        self.slider_map = {}
        self.assembly_view_limits = None
        
        # OCCT Step exporter
        self.step_exporter = StepExporter(thickness=0.047)
        self.pattern = Pattern(width=4.702, height=7.5)
        self.assembly_builder = AssemblyBuilder(
            pattern=self.pattern,
            step_exporter=self.step_exporter,
        )
        self.assembly_params = self.assembly_builder.assembly
        self.spiral_params = self.assembly_builder.spiral
        self.thick = 0.544  # Default conductor width used for coil spacing
        self.assembly_params.coil_width = self.thick
        self.assembly_params.spacing = 0.06000
        self.assembly_params.update_offset_from_coil()
        self.assembly_params.count = 8
        self.spacing_input = None
        
        # Initialize the main window
        self.setWindowTitle("Motor Pattern Visualizer")
        
        # Create grid layout to keep rows and columns aligned
        main_layout = QGridLayout()
        main_layout.setSpacing(spacing)
        main_layout.setContentsMargins(spacing, spacing, spacing, spacing)
        
        # First-row widgets share a square width
        column_width = height
        chart_width = column_width
        assembly_width = column_width * 2 + spacing

        # Create three windows (QWidgets)
        self.windowInput = QWidget()
        self.windowInput.setMinimumSize(column_width, height)
        self.windowInput.setMaximumSize(column_width, height)
        self.windowInput.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        
        # Create matplotlib canvas for pattern window
        self.pattern_figure = Figure(figsize=(height/100, height/100), dpi=100)
        self.pattern_canvas = FigureCanvas(self.pattern_figure)
        self.pattern_ax = self.pattern_figure.add_subplot(111)
        self.pattern_canvas.setMinimumSize(column_width, height)
        self.pattern_canvas.setMaximumSize(column_width, height)
        
        self.windowPattern = QWidget()
        self.windowPattern.setMinimumSize(column_width, height)
        self.windowPattern.setMaximumSize(column_width, height)
        pattern_layout = QVBoxLayout()
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.addWidget(self.pattern_canvas)
        self.windowPattern.setLayout(pattern_layout)
        
        # Create matplotlib canvas for chart display
        self.chart_figure = Figure(figsize=(height/100, height/100), dpi=100)
        self.chart_canvas = FigureCanvas(self.chart_figure)
        self.chart_ax = self.chart_figure.add_subplot(111)
        self.chart_canvas.setMinimumSize(chart_width, height)
        self.chart_canvas.setMaximumSize(chart_width, height)

        self.windowChart = QWidget()
        self.windowChart.setMinimumSize(chart_width, height)
        self.windowChart.setMaximumSize(chart_width, height)
        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.addWidget(self.chart_canvas)
        self.windowChart.setLayout(chart_layout)

        # Create a dedicated slider panel window (occupies former chart slot)
        self.windowSliders = QWidget()
        self.windowSliders.setMinimumSize(height, height)
        self.windowSliders.setMaximumSize(height, height)
        self.windowSliders.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.slider_layout = QVBoxLayout()
        self.slider_layout.setContentsMargins(5, 5, 5, 5)
        self.slider_layout.setSpacing(5)
        self.windowSliders.setLayout(self.slider_layout)
        
        # Create matplotlib canvas for assembly window
        self.assembly_figure = Figure(figsize=(assembly_width/100, height/100), dpi=100)
        self.assembly_canvas = FigureCanvas(self.assembly_figure)
        self.assembly_ax = self.assembly_figure.add_subplot(111)
        self.assembly_canvas.setMinimumSize(assembly_width, height)
        self.assembly_canvas.setMaximumSize(assembly_width, height)
        
        self.windowAssamble = QWidget()
        self.windowAssamble.setMinimumSize(assembly_width, height)
        self.windowAssamble.setMaximumSize(assembly_width, height)
        assembly_layout = QVBoxLayout()
        assembly_layout.setContentsMargins(0, 0, 0, 0)
        assembly_layout.addWidget(self.assembly_canvas)
        self.windowAssamble.setLayout(assembly_layout)
        self.assembly_canvas.mpl_connect('scroll_event', self._on_assembly_scroll)
        
        # Spiral placement parameters (defaults follow curve.py constants)
        self.spiral_radius = None
        self.spiral_thickness = None
        self.spiral_turns = None
        
        # Cache for chart data
        self.chart_needs_update = True
        
        # Create slider panel for windowInput
        self.input_layout = QVBoxLayout()
        self.windowInput.setLayout(self.input_layout)
        self._build_input_panel()
        self._build_slider_panel()
        
        # Add widgets to grid layout for aligned columns
        main_layout.addWidget(self.windowInput, 0, 0)
        main_layout.addWidget(self.windowPattern, 0, 1)
        main_layout.addWidget(self.windowSliders, 0, 2)
        main_layout.addWidget(self.windowAssamble, 1, 0, 1, 2)
        main_layout.addWidget(self.windowChart, 1, 2)
        
        main_layout.setColumnMinimumWidth(0, column_width)
        main_layout.setColumnMinimumWidth(1, column_width)
        main_layout.setColumnMinimumWidth(2, chart_width)
        
        # Third row: 3D OCCT viewer spans all columns
        self.viewer3d = qtViewer3d()
        viewer_width = assembly_width + spacing + chart_width
        self.viewer3d.setMinimumSize(viewer_width, height)
        main_layout.addWidget(self.viewer3d, 2, 0, 1, 3)
        
        self.setLayout(main_layout)
        
        # Adjust window size (three rows now)
        # Width: 3 windows + 2 spacings between them + 2 spacings for margins
        first_row_width = column_width * 3 + spacing * 2
        total_width = first_row_width + spacing * 2
        # Height: 3 rows + 2 spacings between them + 2 spacings for margins
        total_height = height * 3 + spacing * 4
        self.setMinimumSize(total_width, total_height)
        
        # Enable keyboard focus for 3D viewer
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Don't draw here - wait until window is shown
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for 3D view control"""
        key = event.key()
        viewer_focus = self.viewer3d.hasFocus()
        
        # View switching: 1=Top, 2=Front, 3=Side
        if key == Qt.Key_1:
            if viewer_focus:
                # Top view (looking down Z axis)
                self.viewer3d._display.View_Top()
                print("View: Top (Z-axis)")
            else:
                self._reset_assembly_view()
                self._draw_assembly()
                print("Assembly view: Centered")
        elif key == Qt.Key_2:
            # Front view (looking along Y axis)
            self.viewer3d._display.View_Front()
            print("View: Front (Y-axis)")
        elif key == Qt.Key_3:
            # Side view (looking along X axis)
            self.viewer3d._display.View_Right()
            print("View: Side (X-axis)")
        
        elif key == Qt.Key_0:
            self._reset_assembly_view()
            self._draw_assembly()
            print("Assembly view: Reset zoom")
        
        # Pan controls with arrow keys
        elif key == Qt.Key_Left:
            if viewer_focus:
                self.viewer3d._display.Pan(-50, 0)
                print("Pan: Left (3D)")
            else:
                self._pan_assembly(-0.1, 0.0)
        elif key == Qt.Key_Right:
            if viewer_focus:
                self.viewer3d._display.Pan(50, 0)
                print("Pan: Right (3D)")
            else:
                self._pan_assembly(0.1, 0.0)
        elif key == Qt.Key_Up:
            if viewer_focus:
                self.viewer3d._display.Pan(0, 50)
                print("Pan: Up (3D)")
            else:
                self._pan_assembly(0.0, 0.1)
        elif key == Qt.Key_Down:
            if viewer_focus:
                self.viewer3d._display.Pan(0, -50)
                print("Pan: Down (3D)")
            else:
                self._pan_assembly(0.0, -0.1)
        
        # Zoom controls
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            if viewer_focus:
                self.viewer3d._display.ZoomFactor(1.2)
                print("Zoom: In (3D)")
            else:
                self._zoom_assembly(0.9)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            if viewer_focus:
                self.viewer3d._display.ZoomFactor(0.8)
                print("Zoom: Out (3D)")
            else:
                self._zoom_assembly(1.0 / 0.9)
        
        # R = Reset view (fit all)
        elif key == Qt.Key_R:
            self.viewer3d._display.FitAll()
            print("View: Reset (Fit All)")
        
        # I = Isometric view
        elif key == Qt.Key_I:
            self.viewer3d._display.View_Iso()
            print("View: Isometric")
        
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)
    
    def _build_input_panel(self):
        """Build input boxes for width, height, thick, and spacing parameters."""
        # Create a container widget for the input panel
        input_panel = QWidget()
        input_panel.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 3px;")
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(5)
        
        # Title
        title = QLabel("Dimensions")
        title.setStyleSheet("font-weight: bold; font-size: 11px; color: #333;")
        panel_layout.addWidget(title)
        
        # Width input
        width_layout = QHBoxLayout()
        width_layout.setSpacing(5)
        width_label = QLabel("Width:")
        width_label.setMinimumWidth(45)
        width_label.setStyleSheet("font-size: 10px; color: #000;")
        self.width_input = QLineEdit()
        self.width_input.setText(f"{self.assembly_builder.width:.5f}")
        self.width_input.setStyleSheet("""
            QLineEdit { 
                border: 1px solid #ccc;
                padding: 3px; 
                background-color: white;
                color: #000;
                font-size: 10px;
            }
        """)
        self.width_input.setValidator(QDoubleValidator(0.0, 1000.0, 5))
        self.width_input.editingFinished.connect(self._on_width_changed)
        width_layout.addWidget(width_label)
        width_layout.addWidget(self.width_input, 1)  # Stretch factor 1
        panel_layout.addLayout(width_layout)
        
        # Height input
        height_layout = QHBoxLayout()
        height_layout.setSpacing(5)
        height_label = QLabel("Height:")
        height_label.setMinimumWidth(45)
        height_label.setStyleSheet("font-size: 10px; color: #000;")
        self.height_input = QLineEdit()
        self.height_input.setText(f"{self.assembly_builder.height:.5f}")
        self.height_input.setStyleSheet("""
            QLineEdit { 
                border: 1px solid #ccc;
                padding: 3px; 
                background-color: white;
                color: #000;
                font-size: 10px;
            }
        """)
        self.height_input.setValidator(QDoubleValidator(0.0, 1000.0, 5))
        self.height_input.editingFinished.connect(self._on_height_changed)
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_input, 1)  # Stretch factor 1
        panel_layout.addLayout(height_layout)
        
        # Thick input
        thick_layout = QHBoxLayout()
        thick_layout.setSpacing(5)
        thick_label = QLabel("Thick:")
        thick_label.setMinimumWidth(45)
        thick_label.setStyleSheet("font-size: 10px; color: #000;")
        self.thick_input = QLineEdit()
        self.thick_input.setText(f"{self.thick:.5f}")
        self.thick_input.setStyleSheet("""
            QLineEdit { 
                border: 1px solid #ccc;
                padding: 3px; 
                background-color: white;
                color: #000;
                font-size: 10px;
            }
        """)
        self.thick_input.setValidator(QDoubleValidator(0.0, 100.0, 5))
        self.thick_input.editingFinished.connect(self._on_thick_changed)
        thick_layout.addWidget(thick_label)
        thick_layout.addWidget(self.thick_input, 1)  # Stretch factor 1
        panel_layout.addLayout(thick_layout)

        # Spacing input
        spacing_layout = QHBoxLayout()
        spacing_layout.setSpacing(5)
        spacing_label = QLabel("Spacing:")
        spacing_label.setMinimumWidth(45)
        spacing_label.setStyleSheet("font-size: 10px; color: #000;")
        self.spacing_input = QLineEdit()
        self.spacing_input.setText(f"{self.assembly_params.spacing:.5f}")
        self.spacing_input.setStyleSheet("""
            QLineEdit { 
                border: 1px solid #ccc;
                padding: 3px; 
                background-color: white;
                color: #000;
                font-size: 10px;
            }
        """)
        self.spacing_input.setValidator(QDoubleValidator(0.0, 10.0, 5))
        self.spacing_input.editingFinished.connect(self._on_spacing_changed)
        spacing_layout.addWidget(spacing_label)
        spacing_layout.addWidget(self.spacing_input, 1)
        panel_layout.addLayout(spacing_layout)
        
        # All buttons in a horizontal layout (3 columns)
        button_row = QHBoxLayout()
        button_row.setSpacing(3)
        
        self.calc_button = QPushButton("Calc")
        self.calc_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 4px;
                font-size: 9px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2868a8;
            }
        """)
        self.calc_button.clicked.connect(self._on_calculate_clicked)
        button_row.addWidget(self.calc_button)
        
        self.update_3d_button = QPushButton("3D")
        self.update_3d_button.setStyleSheet("""
            QPushButton {
                background-color: #5cb85c;
                color: white;
                border: none;
                padding: 8px 4px;
                font-size: 9px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4cae4c;
            }
            QPushButton:pressed {
                background-color: #449d44;
            }
        """)
        self.update_3d_button.clicked.connect(self._on_update_3d_clicked)
        button_row.addWidget(self.update_3d_button)
        
        self.save_step_button = QPushButton("STEP")
        self.save_step_button.setStyleSheet("""
            QPushButton {
                background-color: #f0ad4e;
                color: white;
                border: none;
                padding: 8px 4px;
                font-size: 9px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ec971f;
            }
            QPushButton:pressed {
                background-color: #d58512;
            }
        """)
        self.save_step_button.clicked.connect(self._on_save_step_clicked)
        button_row.addWidget(self.save_step_button)
        
        # Twist button in the same row
        self.twist_button = QPushButton("Twist")
        self.twist_button.setCheckable(True)
        self.twist_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 4px;
                font-size: 9px;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
            QPushButton:checked:hover {
                background-color: #218838;
            }
        """)
        self.twist_button.clicked.connect(self._on_twist_clicked)
        button_row.addWidget(self.twist_button)
        
        panel_layout.addLayout(button_row)
        
        input_panel.setLayout(panel_layout)
        self.input_layout.addWidget(input_panel)
    
    def _on_width_changed(self):
        """Handle width input change - reset pattern and mark for recalculation."""
        try:
            new_width = float(self.width_input.text())
            if new_width > 0:
                # Reset pattern with new dimensions
                self.assembly_builder.set_dimensions(width=new_width)
                self._reset_assembly_view()
                # Mark chart for update but don't calculate yet
                self.chart_needs_update = True
                self._rebuild_ui_without_chart()
        except ValueError:
            self.width_input.setText(f"{self.assembly_builder.width:.5f}")
    
    def _on_height_changed(self):
        """Handle height input change - reset pattern and mark for recalculation."""
        try:
            new_height = float(self.height_input.text())
            if new_height > 0:
                # Reset pattern with new dimensions
                self.assembly_builder.set_dimensions(height=new_height)
                self._reset_assembly_view()
                # Mark chart for update but don't calculate yet
                self.chart_needs_update = True
                self._rebuild_ui_without_chart()
        except ValueError:
            self.height_input.setText(f"{self.assembly_builder.height:.5f}")
    
    def _on_thick_changed(self):
        """Handle thick input change - update assembly offset, but don't recalculate yet."""
        try:
            new_thick = float(self.thick_input.text())
            if new_thick > 0:
                self.thick = new_thick
                self.assembly_params.coil_width = self.thick
                self.assembly_params.update_offset_from_coil()
                self._reset_assembly_view()
                # Don't trigger chart update - wait for calculate button
                self._update_views_without_chart()
        except ValueError:
            self.thick_input.setText(f"{self.thick:.5f}")

    def _on_spacing_changed(self):
        """Handle spacing input change - update assembly spacing and offset."""
        try:
            new_spacing = float(self.spacing_input.text())
            if new_spacing >= 0.0:
                self.assembly_params.spacing = new_spacing
                self.assembly_params.update_offset_from_coil()
                self.spacing_input.setText(f"{self.assembly_params.spacing:.5f}")
                self._reset_assembly_view()
                self._update_views_without_chart()
        except ValueError:
            self.spacing_input.setText(f"{self.assembly_params.spacing:.5f}")
    
    def _on_calculate_clicked(self):
        """Handle calculate button click - recalculate resistance and update chart."""
        self.chart_needs_update = True
        self._update_views()
    
    def _on_twist_clicked(self):
        """Toggle twist mode - flips bottom half of shape horizontally at pattern height center."""
        self.assembly_params.twist_enabled = self.twist_button.isChecked()
        
        # Update all views
        self._update_views_without_chart()
    
    def _rebuild_ui(self):
        """Rebuild UI after dimension changes (with full chart update)."""
        # Update input boxes
        self.width_input.setText(f"{self.assembly_builder.width:.5f}")
        self.height_input.setText(f"{self.assembly_builder.height:.5f}")
        self.thick_input.setText(f"{self.thick:.5f}")
        self.spacing_input.setText(f"{self.assembly_params.spacing:.5f}")
        
        # Rebuild sliders
        self._build_slider_panel()
        
        # Mark chart for update
        self.chart_needs_update = True
        
        # Redraw all views
        self._update_views()
    
    def _rebuild_ui_without_chart(self):
        """Rebuild UI after dimension changes (skip chart calculation)."""
        # Update input boxes
        self.width_input.setText(f"{self.assembly_builder.width:.5f}")
        self.height_input.setText(f"{self.assembly_builder.height:.5f}")
        self.thick_input.setText(f"{self.thick:.5f}")
        self.spacing_input.setText(f"{self.assembly_params.spacing:.5f}")
        
        # Rebuild sliders
        self._build_slider_panel()
        
        # Redraw views without chart
        self._update_views_without_chart()
    
    def _on_update_3d_clicked(self):
        """Update the 3D model in the viewer."""
        try:
            # Just call the unified update method
            self._update_3d_model()
            
        except Exception as e:
            print(f"Error updating 3D model: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_save_step_clicked(self):
        """Save all array patterns to STEP file."""
        try:
            curves = self.assembly_builder.get_curves()
            left_curve = curves['left']
            right_curve = curves['right']
            if not left_curve or not right_curve:
                print("No valid shape to save")
                return
            options = ["Straight Array", "Spiral Coil", "Both"]
            choice, ok = QInputDialog.getItem(
                self,
                "Select Geometry",
                "Choose which arrangement to export:",
                options,
                current=0,
                editable=False
            )
            if not ok:
                print("STEP export cancelled")
                return
            
            export_straight = choice in ("Straight Array", "Both")
            export_spiral = choice in ("Spiral Coil", "Both")

            all_shapes = []

            if export_straight:
                flat_results = self.assembly_builder.build_flat_solids()
                for inst in flat_results.get('left', []):
                    if inst.shape is not None:
                        all_shapes.append(inst.shape)
                    elif inst.error:
                        print(f"Skipping flat left #{inst.index}: {inst.error}")
                for inst in flat_results.get('right', []):
                    if inst.shape is not None:
                        all_shapes.append(inst.shape)
                    elif inst.error:
                        print(f"Skipping flat right #{inst.index}: {inst.error}")

            if export_spiral:
                spiral_results = self.assembly_builder.build_spiral_solids(
                    radius_override=self.spiral_radius,
                    thickness_override=self.spiral_thickness,
                    turns_override=self.spiral_turns,
                )
                if spiral_results.get('length_warning'):
                    print("Warning: Spiral length insufficient for export; truncating coil geometry.")
                for inst in spiral_results.get('left', []):
                    if inst.shape is not None:
                        all_shapes.append(inst.shape)
                    elif inst.error:
                        print(f"Skipping spiral left #{inst.index}: {inst.error}")
                for inst in spiral_results.get('right', []):
                    if inst.shape is not None:
                        all_shapes.append(inst.shape)
                    elif inst.error:
                        print(f"Skipping spiral right #{inst.index}: {inst.error}")

            if not all_shapes:
                print("No shapes selected for export.")
                return
            
            # Open file dialog
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Geometry",
                "motor_pattern_array.step",
                "STEP Files (*.step *.stp);;STL Files (*.stl);;OBJ Files (*.obj);;All Files (*)"
            )
            
            if filename:
                # Save all shapes as compound
                from OCC.Core.BRep import BRep_Builder
                from OCC.Core.TopoDS import TopoDS_Compound
                
                compound = TopoDS_Compound()
                builder = BRep_Builder()
                builder.MakeCompound(compound)
                
                for shape in all_shapes:
                    builder.Add(compound, shape)
                
                ext = os.path.splitext(filename)[1].lower()
                if ext in ('.stl',):
                    self.step_exporter.save_stl(filename, shape=compound)
                    print(f"Saved STL file: {filename} (array with {len(all_shapes)} shapes)")
                elif ext in ('.obj',):
                    self.step_exporter.save_obj(filename, shape=compound)
                    print(f"Saved OBJ file: {filename} (array with {len(all_shapes)} shapes)")
                else:
                    self.step_exporter.save_step(filename, shape=compound)
                    print(f"Saved STEP file: {filename} (array with {len(all_shapes)} shapes)")
                
        except Exception as e:
            print(f"Error saving STEP file: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_slider_panel(self):
        """Rebuild parameter sliders according to current pattern mode (excluding width and height)."""
        # Remove existing slider widgets
        while self.slider_layout.count() > 0:
            item = self.slider_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.sliders = []
        self.slider_map = {}

        sliders_data = self.assembly_builder.get_pattern_variables()
        for data in sliders_data:
            label = data['label']
            
            # Skip width and height - they are now input boxes
            if label in ['width', 'height']:
                continue
            
            slider = Slider(
                label=label,
                min_val=data['min'],
                max_val=data['max'],
                initial=data['value'],
                step=data.get('step', 0.05)
            )
            slider.valueChanged.connect(self._on_slider_changed)
            self.sliders.append(slider)
            self.slider_map[label] = slider
            self.slider_layout.addWidget(slider)

        # Assembly count control
        count_slider = Slider(
            label='count',
            min_val=1,
            max_val=600,
            initial=self.assembly_params.count,
            step=1
        )
        count_slider.valueChanged.connect(self._on_assembly_slider_changed)
        self.sliders.append(count_slider)
        self.slider_map['count'] = count_slider
        self.slider_layout.addWidget(count_slider)

        self.slider_layout.addStretch()

    def _on_slider_changed(self, label, value):
        """Handle slider value changes - update pattern and refresh views (without expensive calculations)"""
        previous_mode = self.assembly_builder.get_pattern_mode()

        self.assembly_builder.set_pattern_variable(label, value)
        current_mode = self.assembly_builder.get_pattern_mode()
        
        if previous_mode != current_mode:
            self._build_slider_panel()
            self._update_slider_ranges()
            self._update_sliders_from_pattern()
        else:
            self._update_sliders_from_pattern(skip_label=label)

        # Only update graphics, don't recalculate resistance
        self._update_views_without_chart()
    
    def _on_assembly_slider_changed(self, label, value):
        """Handle assembly slider changes - update assembly parameters and refresh"""
        if label == 'count':
            self.assembly_params.count = int(value)
            self._reset_assembly_view()
        
        # Only redraw assembly, don't auto-update 3D
        self._draw_assembly()
    
    def _update_slider_ranges(self):
        """Update slider min/max ranges based on current pattern dimensions"""
        # Get updated variable info from pattern
        variables = self.assembly_builder.get_pattern_variables()
        var_dict = {v['label']: v for v in variables}

        for label, slider in self.slider_map.items():
            if label in var_dict:
                slider.set_range(
                    var_dict[label]['min'],
                    var_dict[label]['max'],
                    var_dict[label].get('step', slider.step)
                )
    
    def _update_sliders_from_pattern(self, skip_label=None):
        """Update all slider values from pattern (after constraints applied)
        
        Args:
            skip_label: Optional label to skip updating (e.g., the slider user is currently changing)
        """
        values = self.assembly_builder.get_pattern_values()

        for label, slider in self.slider_map.items():
            if label == skip_label:
                continue
            if label in values:
                slider.set_value(values[label])
    
    def _update_views(self):
        """Update pattern, chart and assembly visualizations (full update with chart)"""
        self._draw_pattern()
        if self.chart_needs_update:
            self._draw_chart()
            self.chart_needs_update = False
        self._draw_assembly()
        # Don't auto-update 3D model - only update when 3D button is clicked
    
    def _update_views_without_chart(self):
        """Update only pattern and assembly visualizations (skip expensive chart calculation)"""
        self._draw_pattern()
        self._draw_assembly()
        # Don't auto-update 3D model - only update when 3D button is clicked

    def _reset_assembly_view(self):
        """Clear stored assembly view limits so the next draw uses defaults."""
        self.assembly_view_limits = None

    def _pan_assembly(self, dx_fraction: float, dy_fraction: float):
        """Pan the assembly view by a fraction of the current window."""
        if self.assembly_view_limits is None:
            current_xlim = self.assembly_ax.get_xlim()
            current_ylim = self.assembly_ax.get_ylim()
            self.assembly_view_limits = (current_xlim, current_ylim)

        x_limits, y_limits = self.assembly_view_limits
        range_x = x_limits[1] - x_limits[0]
        range_y = y_limits[1] - y_limits[0]

        dx = range_x * dx_fraction
        dy = range_y * dy_fraction

        new_xlim = (x_limits[0] + dx, x_limits[1] + dx)
        new_ylim = (y_limits[0] + dy, y_limits[1] + dy)

        self.assembly_view_limits = (new_xlim, new_ylim)
        self.assembly_ax.set_xlim(*new_xlim)
        self.assembly_ax.set_ylim(*new_ylim)
        self.assembly_canvas.draw_idle()

    def _zoom_assembly(self, factor: float, x_center: float | None = None, y_center: float | None = None):
        """Apply zoom to the assembly view."""
        if factor <= 0:
            return

        if self.assembly_view_limits is None:
            current_xlim = self.assembly_ax.get_xlim()
            current_ylim = self.assembly_ax.get_ylim()
        else:
            current_xlim, current_ylim = self.assembly_view_limits

        x_center = x_center if x_center is not None else (current_xlim[0] + current_xlim[1]) / 2.0
        y_center = y_center if y_center is not None else (current_ylim[0] + current_ylim[1]) / 2.0

        min_range = 1e-4
        current_range_x = max(current_xlim[1] - current_xlim[0], min_range)
        current_range_y = max(current_ylim[1] - current_ylim[0], min_range)

        new_range_x = max(current_range_x * factor, min_range)
        new_range_y = max(current_range_y * factor, min_range)

        new_xlim = (x_center - new_range_x / 2.0, x_center + new_range_x / 2.0)
        new_ylim = (y_center - new_range_y / 2.0, y_center + new_range_y / 2.0)

        self.assembly_view_limits = (new_xlim, new_ylim)
        self.assembly_ax.set_xlim(*new_xlim)
        self.assembly_ax.set_ylim(*new_ylim)
        self.assembly_canvas.draw_idle()

    def _on_assembly_scroll(self, event):
        """Handle mouse wheel zooming on the assembly canvas."""
        if event.inaxes != self.assembly_ax:
            return

        step = getattr(event, 'step', 0)
        if step == 0:
            return

        base_scale = 0.9
        scale = base_scale ** abs(step)
        if step < 0:
            scale = 1.0 / scale

        self._zoom_assembly(
            factor=scale,
            x_center=event.xdata,
            y_center=event.ydata
        )
    
    def _update_3d_model(self):
        """Update the 3D model in the viewer when pattern changes."""
        try:
            self.viewer3d._display.EraseAll()
            flat_results = self.assembly_builder.build_flat_solids()
            for inst in flat_results.get('left', []):
                if inst.shape is not None:
                    self.viewer3d._display.DisplayShape(inst.shape, update=False, color='BLUE', transparency=0.2)
                elif inst.error:
                    print(f"Flat left #{inst.index} error: {inst.error}")

            for inst in flat_results.get('right', []):
                if inst.shape is not None:
                    self.viewer3d._display.DisplayShape(inst.shape, update=False, color='CYAN', transparency=0.3)
                elif inst.error:
                    print(f"Flat right #{inst.index} error: {inst.error}")

            spiral_results = self.assembly_builder.build_spiral_solids(
                radius_override=self.spiral_radius,
                thickness_override=self.spiral_thickness,
                turns_override=self.spiral_turns,
            )
            if spiral_results.get('length_warning'):
                print("Warning: Spiral length insufficient for requested array; truncating coil display.")

            for inst in spiral_results.get('left', []):
                if inst.shape is not None:
                    self.viewer3d._display.DisplayShape(
                        inst.shape,
                        update=False,
                        color='GREEN',
                        transparency=0.35
                    )
                elif inst.error:
                    print(f"Spiral left #{inst.index} error: {inst.error}")

            for inst in spiral_results.get('right', []):
                if inst.shape is not None:
                    self.viewer3d._display.DisplayShape(
                        inst.shape,
                        update=False,
                        color='MAGENTA',
                        transparency=0.45
                    )
                elif inst.error:
                    print(f"Spiral right #{inst.index} error: {inst.error}")

            self.viewer3d._display.FitAll()
            self.viewer3d._display.Repaint()
            
        except Exception as e:
            print(f"Error updating 3D model: {e}")
    
    def _draw_curve(self, ax, curve, color, linewidth=1, alpha=1.0, x_offset=0):
        """Helper function to draw a curve (list of points)
        
        Args:
            ax: Matplotlib axis object
            curve: List of points [[x1,y1], [x2,y2], ...]
            color: Line color
            linewidth: Line width
            alpha: Transparency (0-1)
            x_offset: Horizontal offset to apply
        """
        if curve:
            xs = [p[0] + x_offset for p in curve]
            ys = [p[1] for p in curve]
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
    
    def _draw_pattern(self):
        """Draw the pattern in the pattern window"""
        self.pattern_ax.clear()
        
        # Get segments from pattern with shape
        segments = self.assembly_builder.get_segments(
            assembly_offset=self.assembly_params.offset,
            space=self.assembly_params.spacing
        )
        curves = self.assembly_builder.get_curves()
        
        # Draw bounding box (optional, can be commented out)
        bbox = segments['bbox_left']
        xs = [p[0] for p in bbox] + [bbox[0][0]]
        ys = [p[1] for p in bbox] + [bbox[0][1]]
        self.pattern_ax.plot(xs, ys, 'k-', linewidth=0.5, alpha=0.2)
        
        # Draw the closed shape if available
        if curves['left']:
            shape = curves['left']
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            # Fill the shape with light color
            self.pattern_ax.fill(xs, ys, color='lightblue', alpha=0.3)
            # Draw the shape outline
            self.pattern_ax.plot(xs, ys, 'b-', linewidth=0.5, alpha=0.8)
        
        # Draw left curve (full opacity)
        self._draw_curve(self.pattern_ax, segments['curve_left'], 'b', linewidth=1, alpha=1.0)
        
        # Draw right curve (mirrored, 30% transparent)
        # Mirror the left curve across the symmetry axis
        center_x = self.assembly_builder.width / 2.0
        curve_right = [(2 * center_x - p[0], p[1]) for p in segments['curve_left']]
        self._draw_curve(self.pattern_ax, curve_right, 'b', linewidth=1.0, alpha=0.3)
        
        # Draw symmetry axis (thin dashed gray line)
        symmetry = segments['symmetry']
        xs = [symmetry[0][0], symmetry[1][0]]
        ys = [symmetry[0][1], symmetry[1][1]]
        self.pattern_ax.plot(xs, ys, 'k--', linewidth=0.5, alpha=0.5)
        
        # Set equal aspect ratio
        self.pattern_ax.set_aspect('equal')
        
        # Calculate the center and max dimension for centering and filling
        center_x = self.assembly_builder.width / 2
        center_y = self.assembly_builder.height / 2
        max_dim = max(self.assembly_builder.width, self.assembly_builder.height)
        
        # Add 10% margin and center the view
        margin_factor = 1.1
        half_size = (max_dim * margin_factor) / 2
        
        self.pattern_ax.set_xlim(center_x - half_size, center_x + half_size)
        self.pattern_ax.set_ylim(center_y - half_size, center_y + half_size)
        
        # Remove axis labels and ticks for cleaner look
        self.pattern_ax.set_xticks([])
        self.pattern_ax.set_yticks([])
        self.pattern_ax.set_frame_on(False)
        
        # Refresh canvas
        self.pattern_figure.tight_layout(pad=0)
        self.pattern_canvas.draw()
    
    def _draw_assembly(self):
        """Draw the assembly with repeated shape patterns"""
        self.assembly_ax.clear()
        
        instances = self.assembly_builder.build_2d_instances()
        if not instances:
            self.assembly_view_limits = None
            self.assembly_canvas.draw()
            return

        # Set transparency for overlapping visualization
        base_alpha = 0.5
        
        # Draw repeated patterns
        for inst in instances:
            xs = [p[0] for p in inst.left_shape]
            ys = [p[1] for p in inst.left_shape]
            # Fill the shape
            self.assembly_ax.fill(xs, ys, color='lightblue', alpha=base_alpha * 0.6)
            # Draw the shape outline
            self.assembly_ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=base_alpha)
            
            # Draw right shape (mirrored)
            xs_right = [p[0] for p in inst.right_shape]
            ys_right = [p[1] for p in inst.right_shape]
            # Fill the mirrored shape
            self.assembly_ax.fill(xs_right, ys_right, color='lightblue', alpha=base_alpha * 0.3)
            # Draw the mirrored shape outline with dashed line
            self.assembly_ax.plot(xs_right, ys_right, 'b--', linewidth=0.8, alpha=base_alpha * 0.5)
        
        # Set equal aspect ratio
        self.assembly_ax.set_aspect('equal')
        
        # Calculate view bounds using current canvas aspect ratio (fallback to grid-defined width)
        canvas_width = self.assembly_canvas.width() or (self.height * 2 + self.spacing)
        canvas_height = self.assembly_canvas.height() or self.height
        assembly_aspect = canvas_width / max(1, canvas_height)
        view_height = self.assembly_builder.height * 1.1  # Add 10% margin
        view_width = view_height * assembly_aspect  # Match window aspect ratio
        
        # Center the view horizontally on the content
        last_start = instances[-1].offset if instances else 0.0
        content_width = last_start + self.assembly_builder.width
        center_x = content_width / 2
        
        default_xlim = (center_x - view_width/2, center_x + view_width/2)
        default_ylim = (-view_height * 0.05, view_height * 0.95)

        if self.assembly_view_limits is None:
            self.assembly_view_limits = (default_xlim, default_ylim)

        x_limits, y_limits = self.assembly_view_limits

        self.assembly_ax.set_xlim(*x_limits)
        self.assembly_ax.set_ylim(*y_limits)
        
        # Remove axis labels and ticks
        self.assembly_ax.set_xticks([])
        self.assembly_ax.set_yticks([])
        self.assembly_ax.set_frame_on(False)
        
        # Refresh canvas
        self.assembly_figure.tight_layout(pad=0)
        self.assembly_canvas.draw()
    
    def _draw_chart(self):
        """Draw area and resistance vs exponent/vbh chart in independent window"""
        # Clear the entire figure to remove all axes including twinx axes
        self.chart_figure.clear()
        self.chart_ax = self.chart_figure.add_subplot(111)
        
        # Calculate area, coefficient and resistance
        x_values = []
        area_values = []
        coeff_values = []
        resistance_values = []
        
        # Save current state so we can restore after sampling
        state_snapshot = self.assembly_builder.snapshot_pattern()
        is_mode_a = self.assembly_builder.get_pattern_mode() == 'A'
        
        if is_mode_a:
            # Mode A: vary vbh from 0 to half_height
            half_height = self.assembly_builder.height / 2.0
            vbh_start = 0.0
            vbh_end = half_height
            vbh_step = half_height / 100.0  # 100 steps
            
            original_value = state_snapshot['vbh']
            current_vbh = vbh_start
            try:
                while current_vbh <= vbh_end + 1e-9:
                    # Restore original geometry before applying a new vbh
                    self.assembly_builder.restore_pattern(state_snapshot)
                    self.assembly_builder.set_pattern_variable('vbh', current_vbh)

                    area = self.assembly_builder.get_shape_area(
                        offset=self.assembly_params.offset,
                        space=self.assembly_params.spacing
                    )
                    coeff = self.assembly_builder.get_equivalent_coefficient(
                        offset=self.assembly_params.offset,
                        space=self.assembly_params.spacing
                    )
                    resistance = self.assembly_builder.get_resistance(
                        offset=self.assembly_params.offset,
                        space=self.assembly_params.spacing
                    )

                    x_values.append(current_vbh)
                    area_values.append(area)
                    coeff_values.append(coeff)
                    resistance_values.append(resistance)

                    current_vbh += vbh_step
            finally:
                self.assembly_builder.restore_pattern(state_snapshot)
        else:
            # Mode B: vary exponent from 1.2 to 2.0
            exponent_start = 1.2
            exponent_end = 2.0
            exponent_step = 0.01
            
            original_value = state_snapshot['exponent']
            current_exp = exponent_start
            try:
                while current_exp <= exponent_end:
                    # Restore original geometry before applying a new exponent
                    self.assembly_builder.restore_pattern(state_snapshot)
                    self.assembly_builder.set_pattern_variable('exponent', current_exp)

                    area = self.assembly_builder.get_shape_area(
                        offset=self.assembly_params.offset,
                        space=self.assembly_params.spacing
                    )
                    coeff = self.assembly_builder.get_equivalent_coefficient(
                        offset=self.assembly_params.offset,
                        space=self.assembly_params.spacing
                    )
                    resistance = self.assembly_builder.get_resistance(
                        offset=self.assembly_params.offset,
                        space=self.assembly_params.spacing
                    )

                    x_values.append(current_exp)
                    area_values.append(area)
                    coeff_values.append(coeff)
                    resistance_values.append(resistance)

                    current_exp += exponent_step
            finally:
                self.assembly_builder.restore_pattern(state_snapshot)
        
        # Calculate current values for display
        current_area = self.assembly_builder.get_shape_area(offset=self.assembly_params.offset, space=self.assembly_params.spacing)
        s1 = current_area
        s2 = self.assembly_builder.get_rectangle_area(offset=self.assembly_params.offset, space=self.assembly_params.spacing)
        current_coeff = self.assembly_builder.get_equivalent_coefficient(offset=self.assembly_params.offset, space=self.assembly_params.spacing)
        current_resistance = self.assembly_builder.get_resistance(offset=self.assembly_params.offset, space=self.assembly_params.spacing)
        symmetric_area = self.assembly_builder.get_symmetric_curve_area()
        
        # Calculate area improvement rate
        if is_mode_a:
            # Mode A: compare min vbh (0) vs max vbh (half_height)
            area_baseline = area_values[0] if area_values else None  # vbh = 0
            area_optimized = area_values[-1] if area_values else None  # vbh = max
        else:
            # Mode B: compare exponent 2.0 vs 1.2
            area_baseline = None  # exponent = 2.0
            area_optimized = None  # exponent = 1.2
            for i, exp in enumerate(x_values):
                if abs(exp - 2.0) < 0.005:  # Find closest to 2.0
                    area_baseline = area_values[i]
                if abs(exp - 1.2) < 0.005:  # Find closest to 1.2
                    area_optimized = area_values[i]
        
        # Calculate improvement rate
        if area_baseline and area_optimized:
            improvement_rate = (area_optimized - area_baseline) / area_baseline * 100
            info_text = f'S1:{s1:.3f} S2:{s2:.3f} K:{current_coeff:.3f} ΔS:{improvement_rate:+.1f}%\nR:{current_resistance*1000:.3f}mΩ | Slot:{symmetric_area:.3f}mm²'
        else:
            info_text = f'S1:{s1:.3f} S2:{s2:.3f} K:{current_coeff:.3f}\nR:{current_resistance*1000:.3f}mΩ | Slot:{symmetric_area:.3f}mm²'
        
        self.chart_ax.set_title(info_text, fontsize=10, pad=8)
        
        # Plot area curve (left y-axis, blue)
        color_area = 'tab:blue'
        xlabel = 'vbh (mm)' if is_mode_a else 'Exponent'
        self.chart_ax.set_xlabel(xlabel, fontsize=10)
        self.chart_ax.set_ylabel('Area (mm²)', fontsize=10, color=color_area)
        self.chart_ax.plot(x_values, area_values, color=color_area, linewidth=2.0, label='Area')
        self.chart_ax.tick_params(axis='y', labelcolor=color_area, labelsize=9)
        
        # Mark current position for area
        self.chart_ax.plot([original_value], [current_area], 'o', color=color_area, markersize=6)
        
        # Create second y-axis for resistance (right y-axis, red)
        if resistance_values:
            resistance_mohm = [r * 1000 for r in resistance_values]
            color_resistance = 'tab:red'
            
            chart_ax2 = self.chart_ax.twinx()
            chart_ax2.set_ylabel('Resistance (mΩ)', fontsize=10, color=color_resistance)
            chart_ax2.plot(x_values, resistance_mohm, color=color_resistance, linewidth=2.0,
                          linestyle='--', label='Resistance')
            chart_ax2.tick_params(axis='y', labelcolor=color_resistance, labelsize=9)
            
            # Mark current position for resistance
            chart_ax2.plot([original_value], [current_resistance * 1000], 'o', 
                          color=color_resistance, markersize=6)
        
        # Add grid and styling
        self.chart_ax.grid(True, alpha=0.3, linewidth=0.5)
        self.chart_ax.tick_params(axis='x', labelsize=9)
        
        # Set x-axis limits to match the data range
        if is_mode_a:
            half_height = self.assembly_builder.height / 2.0
            self.chart_ax.set_xlim(-0.05 * half_height, 1.05 * half_height)
        else:
            self.chart_ax.set_xlim(1.15, 2.05)
        
        # Refresh canvas
        self.chart_figure.tight_layout(pad=1.0)
        self.chart_canvas.draw()
    
    def show(self, box=None, pattern=None):
        """Display the GUI and start the Qt application"""
        super().show()
        # Use QTimer to delay initial draw until window is fully rendered
        QTimer.singleShot(0, self._update_views)
        
    @staticmethod
    def run_app(visualizer):
        """Static method to run the Qt application"""
        import sys
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        visualizer.show()
        sys.exit(app.exec_())
