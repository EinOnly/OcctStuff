import numpy as np

from log import COLORLOG
from superellipse import Superellipse
from parameters import LPARAMS
from calculate import Calculate

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from typing import Dict, Any, Iterable, List, Tuple


class Layers(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._params = LPARAMS
        self._layerConfig = {}


        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = LayerCanvas()
        layout.addWidget(self.canvas)

        self.register()
        self._refresh_layers()


        COLORLOG.info(f"Layers initialized.")

    # ------------------------------------------------------------------
    # Data plumbing
    # ------------------------------------------------------------------
    def update(self):
        self._refresh_layers()

    def read(self) -> Dict[str, Any]:
        return self._params.snapshot()

    def register(self):
        self._params.changed.connect(self._on_param_changed)
        self._params.bulkChanged.connect(self._on_bulk_changed)

    def _refresh_layers(self):
        self.canvas.setLayers(self.getLayers(self._params.snapshot(), self._layerConfig))

    # ------------------------------------------------------------------
    # Signal callbacks
    # ------------------------------------------------------------------
    def _on_param_changed(self, key: str, value: Any):
        self._refresh_layers()

    def _on_bulk_changed(self, payload: Dict[str, Any]):
        self._refresh_layers()

    def _buildStart(self):
        pass

    def _buildNormal(self):
        pass

    def _buildEnd(self):
        pass

    def _buildLayers(self):
        self._buildStart()
        self._buildNormal()
        self._buildEnd()
    
    def getLayers(self, params: Dict[str, Any], layerConfig: Dict[str, Any]) -> Dict[str, Any]:
        pattern = {}
        return pattern

class LayerCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: Dict[str, Any] = {}
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setLayers(self, layers: Dict[str, Any]):
        self._layers = layers or {}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().window())

        self._draw_Layers(painter, )

    def _draw_Layers(self, painter: QPainter,points: List[Tuple[float, float]]):
        pass

    def _draw_shape(self, painter: QPainter, mapper, points: List[Tuple[float, float]]):
        """Draw the main shape with semi-transparent fill"""
        if len(points) < 3:
            return

        polygon = QPolygonF([mapper(pt) for pt in points])
        fill = QBrush(QColor(255, 140, 0, 100))  # Orange with transparency
        outline = QPen(QColor(255, 100, 0), 2.0)
        outline.setCosmetic(True)

        painter.setBrush(fill)
        painter.setPen(outline)
        painter.drawPolygon(polygon)