from log import CORELOG

from parameters import ParametersPanel
from layer import Layers
from pattern import Pattern


from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow,
    QGridLayout,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
    QFrame
)
from PyQt5.QtCore import Qt

class Application(QMainWindow):
    PANEL_PADDING = 10

    def __init__(self, parent=None) -> None:
        CORELOG.info("Initializing Application")
        super().__init__(parent)

        self.layers = []
        self._buildLayout()
        self.resize(1400, 800)

    def _buildLayout(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setSpacing(self.PANEL_PADDING)
        grid.setContentsMargins(
            self.PANEL_PADDING,
            0,
            self.PANEL_PADDING,
            self.PANEL_PADDING,
        )

        self.paramsWidget = self.buildParams()
        self._addPanelToGrid(
            grid,
            self.paramsWidget,
            0,
            0,
            3,
            1,
            top_margin=self.PANEL_PADDING,
            decorate=False,
        )

        self.patternWidget = self.buildPattern()
        self._addPanelToGrid(grid, self.patternWidget, 0, 1, top_margin=30)

        self.matplotWidget = self.buileMatplot()
        self._addPanelToGrid(grid, self.matplotWidget, 0, 2, top_margin=30)

        self.layersWidget = self.buildLayers()
        self._addPanelToGrid(grid, self.layersWidget, 1, 1, 1, 2)

        self.occtWidget = self.buildOcct()
        self._addPanelToGrid(grid, self.occtWidget, 2, 1, 1, 2)

    def _buildParameters(self):
        pass

    # --- Window registrations -------------------------------------------------
    def buildParams(self):
        return ParametersPanel()

    def buildPattern(self):
        return Pattern()

    def buileMatplot(self):
        return self._buildPlaceholder("Matplot")

    def buildLayers(self):
        return Layers()

    def buildOcct(self):
        return self._buildPlaceholder("OCCT")

    def _buildPlaceholder(self, title: str, *, top_padding=None) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"{title}\nPlaceholder")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        wrapper = self._wrapPanel(content)
        wrapper.setMinimumHeight(120)
        wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return wrapper

    def _wrapPanel(self, widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Plain)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        frame.setSizePolicy(widget.sizePolicy())
        return frame

    def _addPanelToGrid(
        self,
        grid: QGridLayout,
        widget: QWidget,
        row: int,
        col: int,
        row_span: int = 1,
        col_span: int = 1,
        *,
        top_margin: int = None,
        decorate: bool = True,
    ):
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(
            0,
            self.PANEL_PADDING if top_margin is None else top_margin,
            0,
            0,
        )
        container_layout.setSpacing(0)
        content = self._wrapPanel(widget) if decorate else widget
        container_layout.addWidget(content)
        container.setSizePolicy(widget.sizePolicy())
        grid.addWidget(container, row, col, row_span, col_span)

    def _buildLayers(self):
        pass

    def _buldSliders(self):
        pass


    def refresh(self):
        pass
    
    def register(self):
        pass
    
    def render(self):
        CORELOG.info("Main loop rendering.")
        super().show()
        pass
    
    def close(self):
        pass
