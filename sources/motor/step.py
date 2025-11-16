"""
STEP file generator and 3D viewer for motor pattern shapes using OpenCASCADE.
"""

from typing import List, Tuple, Dict, Any
import os

# PyQt5 imports
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt

# OpenCASCADE imports
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone

# OCCT Display
from OCC.Display.backend import load_backend
load_backend('pyqt5')
from OCC.Display.qtDisplay import qtViewer3d


class StepExporter:
    """Generate 3D OCCT shapes from 2D motor pattern curves and export to STEP."""

    def __init__(self, thickness: float = 0.047):
        """
        Initialize STEP exporter.

        Args:
            thickness: Extrusion thickness in mm (default 0.047 mm for copper foil)
        """
        self.thickness = thickness
        self.current_shapes = []

    def create_shape_from_curve(self, curve_points: List[Tuple[float, float]],
                                z_offset: float = 0.0) -> TopoDS_Shape:
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
        pts = list(curve_points)
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        # Create wire from line segments connecting consecutive points
        wire_maker = BRepBuilderAPI_MakeWire()

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]

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
        return shape

    def create_compound_from_layers(self, layers: Dict[str, Any]) -> TopoDS_Compound:
        """
        Create a compound shape from front and back layers.

        Args:
            layers: Dictionary with 'front' and 'back' layer data

        Returns:
            TopoDS_Compound: Compound containing all extruded shapes
        """
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)

        self.current_shapes = []

        # Process front layer at z=0
        front_shapes = layers.get("front", [])
        for shape_data in front_shapes:
            shape_curve = shape_data.get("shape")
            if shape_curve is not None and len(shape_curve) > 0:
                points = [(pt[0], pt[1]) for pt in shape_curve]
                solid = self.create_shape_from_curve(points, z_offset=0.0)
                builder.Add(compound, solid)
                self.current_shapes.append({
                    "shape": solid,
                    "layer": "front",
                    "color": shape_data.get("color", "#de7cfc")
                })

        # Process back layer at z=thickness (offset to avoid overlap)
        back_shapes = layers.get("back", [])
        for shape_data in back_shapes:
            shape_curve = shape_data.get("shape")
            if shape_curve is not None and len(shape_curve) > 0:
                points = [(pt[0], pt[1]) for pt in shape_curve]
                # Offset back layer slightly in Z to avoid z-fighting
                solid = self.create_shape_from_curve(points, z_offset=self.thickness * 1.5)
                builder.Add(compound, solid)
                self.current_shapes.append({
                    "shape": solid,
                    "layer": "back",
                    "color": shape_data.get("color", "#de7cfc")
                })

        return compound

    def save_step(self, filename: str, shape: TopoDS_Shape = None) -> bool:
        """
        Export shape to STEP file.

        Args:
            filename: Output STEP file path
            shape: Shape to export

        Returns:
            bool: True if export successful
        """
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


class StepViewer(QWidget):
    """3D viewer widget for OCCT shapes with save functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.exporter = StepExporter(thickness=0.047)
        self.current_compound = None
        self.layers_data = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create OCCT viewer
        self.viewer = qtViewer3d(self)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.viewer)

        # Enable keyboard focus for shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

        self.setMinimumSize(400, 400)

    def setLayers(self, layers: Dict[str, Any]):
        """
        Update the 3D view with new layer data.

        Args:
            layers: Dictionary containing 'front' and 'back' layer shapes
        """
        self.layers_data = layers
        self.refresh_view()

    def refresh_view(self):
        """Rebuild the 3D view from current layer data."""
        if not self.layers_data:
            return

        try:
            # Clear previous display
            self.viewer._display.Context.RemoveAll(True)

            # Create compound from layers
            self.current_compound = self.exporter.create_compound_from_layers(self.layers_data)

            # Display each shape with its color
            for shape_info in self.exporter.current_shapes:
                shape = shape_info["shape"]
                color_str = shape_info["color"]

                # Parse hex color
                color_str = color_str.lstrip('#')
                r = int(color_str[0:2], 16) / 255.0
                g = int(color_str[2:4], 16) / 255.0
                b = int(color_str[4:6], 16) / 255.0
                color = Quantity_Color(r, g, b, Quantity_TOC_RGB)

                # Display with color
                ais_shape = self.viewer._display.DisplayShape(
                    shape,
                    color=color,
                    transparency=0.3 if shape_info["layer"] == "back" else 0.1,
                    update=False
                )

            # Fit all and update
            self.viewer._display.FitAll()
            self.viewer._display.Repaint()

        except Exception as e:
            print(f"Error rebuilding view: {e}")
            import traceback
            traceback.print_exc()

    def save_step_file(self):
        """Save the current shapes to a STEP file."""
        if self.current_compound is None:
            QMessageBox.warning(self, "No Data", "No shapes to save. Please generate layers first.")
            return

        # Open file dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save STEP File",
            "",
            "STEP Files (*.step *.stp);;All Files (*)"
        )

        if not filename:
            return

        try:
            self.exporter.save_step(filename, self.current_compound)
            QMessageBox.information(self, "Success", f"STEP file saved successfully:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save STEP file:\n{str(e)}")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for view control."""
        from PyQt5.QtCore import Qt

        key = event.key()

        # 1 - Top view
        if key == Qt.Key_1:
            self.viewer._display.View_Top()
            self.viewer._display.Repaint()

        # 2 - Fit all (center)
        elif key == Qt.Key_2:
            self.viewer._display.FitAll()
            self.viewer._display.Repaint()

        # - (minus) - Zoom out
        elif key == Qt.Key_Minus:
            self.viewer._display.ZoomFactor(0.8)
            self.viewer._display.Repaint()

        # = (equals) - Zoom in
        elif key == Qt.Key_Equal:
            self.viewer._display.ZoomFactor(1.25)
            self.viewer._display.Repaint()

        # Arrow keys - Pan
        elif key == Qt.Key_Up:
            self.viewer._display.Pan(0, 50)
            self.viewer._display.Repaint()

        elif key == Qt.Key_Down:
            self.viewer._display.Pan(0, -50)
            self.viewer._display.Repaint()

        elif key == Qt.Key_Left:
            self.viewer._display.Pan(50, 0)
            self.viewer._display.Repaint()

        elif key == Qt.Key_Right:
            self.viewer._display.Pan(-50, 0)
            self.viewer._display.Repaint()

        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    viewer = StepViewer()
    viewer.show()

    # Test with sample data
    test_layers = {
        "front": [
            {
                "shape": [(0, 0), (10, 0), (10, 5), (5, 8), (0, 5)],
                "color": "#ff6b6b"
            }
        ],
        "back": [
            {
                "shape": [(2, 1), (8, 1), (8, 4), (5, 6), (2, 4)],
                "color": "#4ecdc4"
            }
        ]
    }
    viewer.setLayers(test_layers)

    sys.exit(app.exec_())
