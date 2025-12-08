from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QSlider, QVBoxLayout

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
from PyQt5.QtGui import QSurfaceFormat
fmt = QSurfaceFormat()
fmt.setSamples(4)
fmt.setDepthBufferSize(24)
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.CoreProfile)
QSurfaceFormat.setDefaultFormat(fmt)
from PyQt5.QtWidgets import QSizePolicy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from OCC.Display.backend import load_backend
load_backend("pyqt5")
from OCC.Display.qtDisplay import qtViewer3d

from OCC.Core.gp import gp_Pln, gp_Pnt, gp_Dir
from OCC.Display.qtDisplay import qtViewer3d
from OCC.Core.Graphic3d import Graphic3d_ClipPlane

# === matplotlib canvas ===
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, ax, parent=None):
        self.ax = ax
        fig = self.ax.figure
        fig.set_dpi(80)  # <<== recommanded dpi
        super().__init__(fig)
        self.setParent(parent)
        self.draw()  # Initial draw


# === OCC Viewer Widget ===
class OCCViewerWidget(QWidget):
    def __init__(self, parent=None, shapes=None):
        super().__init__(parent)
        self.viewer = qtViewer3d(self)
        self.viewer.InitDriver()
        self.display = self.viewer._display
        self.shapes = shapes

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer)
        self.viewer.setMinimumSize(600, 600)

        self._resized_once = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._resized_once:
            self.display.View.MustBeResized()
            self.display.Repaint()
            self._resized_once = True

    def display_units(self):
        self.display.EraseAll()
        self.display.DisplayShape(self.shapes, update=True)
        self.display.FitAll()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F:
            self.display.FitAll()
        elif key == Qt.Key_1:  # Front view
            self.set_view('front')
        elif key == Qt.Key_2:  # Back view
            self.set_view('back')
        elif key == Qt.Key_3:  # Left view
            self.set_view('left')
        elif key == Qt.Key_4:  # Right view
            self.set_view('right')
        elif key == Qt.Key_5:  # Top view
            self.set_view('top')
        elif key == Qt.Key_6:  # Bottom view
            self.set_view('bottom')
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.zoom(factor=1.1)  # zoom in
        elif key == Qt.Key_Minus or key == Qt.Key_Underscore:
            self.zoom(factor=0.9)  # zoom out
        event.accept()

    def zoom(self, factor=1.1):
        view = self.display.View

        # center of the view
        cx = 0.5 * self.width()
        cy = 0.5 * self.height()

        # zoom in/out
        # dx and dy are the distances from the center to the edges
        dx = int((1.0 - 1.0 / factor) * self.width() * 0.5)
        dy = int((1.0 - 1.0 / factor) * self.height() * 0.5)

        x1 = int(cx - dx)
        y1 = int(cy - dy)
        x2 = int(cx + dx)
        y2 = int(cy + dy)

        view.Zoom(x1, y1, x2, y2)
        self.display.Repaint()

    def set_view(self, name):
        view = self.display.View
        if name == 'front':
            view.SetProj(0, 0, 1)
        elif name == 'back':
            view.SetProj(0, 0, -1)
        elif name == 'left':
            view.SetProj(-1, 0, 0)
        elif name == 'right':
            view.SetProj(1, 0, 0)
        elif name == 'top':
            view.SetProj(0, 1, 0)
        elif name == 'bottom':
            view.SetProj(0, -1, 0)
        self.display.FitAll()

# === Main Window ===
class MainWindow(QMainWindow):
    def __init__(self, ax, shapes):
        super().__init__()
        self.setWindowTitle("Voronoi + OCCT Viewer")
        self.resize(1600, 1600)

        central = QWidget()
        layout = QHBoxLayout(central)

        self.canvas = MatplotlibCanvas(ax, self)
        self.canvas.setMinimumSize(800, 800)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.occt = OCCViewerWidget(parent=self, shapes=shapes)
        self.occt.setMinimumSize(800, 800)
        self.occt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.canvas)
        layout.addWidget(self.occt)

        self.setCentralWidget(central)
        self.occt.display_units()


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())