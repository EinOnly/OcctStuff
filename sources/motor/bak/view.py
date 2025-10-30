"""
Simple STEP file viewer using pythonOCC and PyQt5.

Launch the script with a STEP file path to open and display the model:

    python sources/motor/view.py path/to/model.step

If no path is supplied, a file dialog allows selecting a STEP file.
Keyboard shortcuts:
    - 1: Top view (俯视图).
    - 2: Front view (正视图).
    - 3: Isometric view (透视图).
    - 4: Toggle transparent mode (遮挡透明显示).
    - Arrow keys pan the view.
    - `=` zooms in, `-` zooms out.
    - `F` fits the view to the model.
"""

import os
import sys

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QSizePolicy
from PyQt5.QtGui import QSurfaceFormat

from OCC.Display.backend import load_backend

load_backend("pyqt5")
from OCC.Display.qtDisplay import qtViewer3d  # noqa: E402
from OCC.Core.IFSelect import IFSelect_RetDone  # noqa: E402
from OCC.Core.STEPControl import STEPControl_Reader  # noqa: E402
from OCC.Core.AIS import AIS_Shape  # noqa: E402


# Improve rendering quality by enabling HiDPI support and multisampling
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

surface_format = QSurfaceFormat()
surface_format.setDepthBufferSize(24)
surface_format.setSamples(8)  # Enable MSAA for smoother edges
surface_format.setVersion(3, 3)
surface_format.setProfile(QSurfaceFormat.CoreProfile)
QSurfaceFormat.setDefaultFormat(surface_format)


def load_step(path: str):
    """Load a STEP file and return a TopoDS_Shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()


class StepViewer(QMainWindow):
    """Main window hosting the OCCT 3D viewer."""

    def __init__(self, shape, source_path: str):
        super().__init__()
        self._source_path = source_path
        self.viewer = qtViewer3d(self)
        self.viewer.InitDriver()
        self.viewer.setFocusPolicy(Qt.StrongFocus)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self.viewer)
        self.viewer.setMinimumSize(900, 700)

        self.display = self.viewer._display
        self._context = None
        self._ais_shape = None
        self._transparent = False

        # Increase triangulation quality so curved surfaces look smoother.
        context = getattr(self.display, "Context", None)
        if callable(context):
            context = context()
        if context is not None:
            self._context = context
            context.SetDeviationCoefficient(0.02)
            context.SetDeviationAngle(5.0)

        if self._context is not None:
            self._ais_shape = AIS_Shape(shape)
            self._context.Display(self._ais_shape, True)
        else:
            self._ais_shape = self.display.DisplayShape(shape, update=True)
        self.display.View.SetImmediateUpdate(True)
        params = self.display.View.ChangeRenderingParams()
        if hasattr(params, "RenderResolutionScale"):
            screen = QApplication.primaryScreen()
            pixel_ratio = screen.devicePixelRatio() if screen else 1.0
            params.RenderResolutionScale = max(1.0, pixel_ratio)

        self.display.FitAll()
        QTimer.singleShot(0, self._refresh_view)

        self.setWindowTitle(f"STEP Viewer - {os.path.basename(source_path)}")
        self.resize(900, 700)

    def _refresh_view(self):
        """Ensure the OpenGL viewport matches the widget size."""
        if self.display is not None:
            self.display.View.MustBeResized()
            self.display.View.Redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_view()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for panning and zooming."""
        key = event.key()
        pan_amount = 50

        if key == Qt.Key_1:
            self.display.View_Top()
            self.display.FitAll()
        elif key == Qt.Key_2:
            self.display.View_Front()
            self.display.FitAll()
        elif key == Qt.Key_3:
            self.display.View_Iso()
            self.display.FitAll()
        elif key == Qt.Key_4:
            self._toggle_transparency()
        elif key == Qt.Key_Left:
            self.display.Pan(-pan_amount, 0)
        elif key == Qt.Key_Right:
            self.display.Pan(pan_amount, 0)
        elif key == Qt.Key_Up:
            self.display.Pan(0, pan_amount)
        elif key == Qt.Key_Down:
            self.display.Pan(0, -pan_amount)
        elif key in (Qt.Key_Equal, Qt.Key_Plus):
            self.display.ZoomFactor(1.2)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.display.ZoomFactor(0.8)
        elif key == Qt.Key_F:
            self.display.FitAll()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        """Ensure the embedded viewer receives focus once visible."""
        super().showEvent(event)
        self.viewer.setFocus(Qt.OtherFocusReason)
        self._refresh_view()

    def _toggle_transparency(self):
        """Toggle between opaque and semi-transparent display."""
        if self._ais_shape is None or self._context is None:
            return
        self._transparent = not self._transparent
        alpha = 0.6 if self._transparent else 0.0
        self._context.SetTransparency(self._ais_shape, alpha, True)
        self._context.Redisplay(self._ais_shape, True)


def select_step_file():
    """Open a file dialog to select a STEP file."""
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilters(["STEP files (*.step *.stp)", "All files (*)"])
    if dialog.exec_():
        selected = dialog.selectedFiles()
        if selected:
            return selected[0]
    return None


def main(argv):
    """Entry point for launching the viewer."""
    app = QApplication(argv)

    path = argv[1] if len(argv) > 1 else None
    if not path:
        path = select_step_file()

    if not path:
        return 0

    path = os.path.abspath(path)
    if not os.path.isfile(path):
        QMessageBox.critical(None, "STEP Viewer", f"File not found:\n{path}")
        return 1

    try:
        shape = load_step(path)
    except Exception as exc:
        QMessageBox.critical(None, "STEP Viewer", f"无法加载 STEP 文件:\n{exc}")
        return 1

    window = StepViewer(shape, path)
    window.show()
    window.activateWindow()
    window.raise_()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
