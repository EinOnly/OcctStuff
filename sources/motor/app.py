from log import CORELOG

from parameters import PPARAMS, LPARAMS, ParametersPanel
from layer import Layer
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
    PANEL_PADDING = ParametersPanel.OUTER_MARGIN

    def __init__(self, parent=None) -> None:
        CORELOG.info("Initializing Application")
        super().__init__(parent)

        self.layers = []
        self._buildLayout()
        self.resize(1000, 800)

    def _buildLayout(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setSpacing(self.PANEL_PADDING)
        grid.setContentsMargins(
            self.PANEL_PADDING,
            self.PANEL_PADDING,
            self.PANEL_PADDING,
            self.PANEL_PADDING,
        )

        self.paramsWidget = self.buildParams()
        grid.addWidget(self.paramsWidget, 0, 0, 3, 1)

        self.patternWidget = self.buildPattern()
        grid.addWidget(self.patternWidget, 0, 1)

        self.matplotWidget = self.buileMatplot()
        grid.addWidget(self.matplotWidget, 0, 2)

        self.layersWidget = self.buildLayers()
        grid.addWidget(self.layersWidget, 1, 1, 1, 2)

        self.occtWidget = self.buildOcct()
        grid.addWidget(self.occtWidget, 2, 1, 1, 2)

    def _buildParameters(self):
        pass

    # --- Window registrations -------------------------------------------------
    def buildParams(self):
        return ParametersPanel()

    def buildPattern(self):
        pattern = Pattern()
        return self._wrapPanel(pattern, top_padding=20)

    def buileMatplot(self):
        return self._buildPlaceholder("Matplot", top_padding=10)

    def buildLayers(self):
        return self._buildPlaceholder("Layers")

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
        wrapper = self._wrapPanel(content, top_padding=top_padding)
        wrapper.setMinimumHeight(120)
        wrapper.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return wrapper

    def _wrapPanel(self, widget: QWidget, *, padding=True, top_padding=None) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        if padding:
            left = right = bottom = self.PANEL_PADDING
            top = self.PANEL_PADDING
        else:
            left = right = bottom = top = 0
        if top_padding is not None:
            top = top_padding
        layout.setContentsMargins(left, top, right, bottom)
        layout.addWidget(widget)
        frame.setSizePolicy(widget.sizePolicy())
        return frame

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
