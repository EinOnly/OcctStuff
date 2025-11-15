from typing import Dict, Any, Iterable, List, Tuple
import numpy as np

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
)

from parameters import PPARAMS


class Pattern(QWidget):
    """Visualizes the current pattern geometry derived from PPARAMS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._params = PPARAMS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = PatternCanvas()
        layout.addWidget(self.canvas)

        self.register()
        self._refresh_pattern()

    # ------------------------------------------------------------------
    # Data plumbing
    # ------------------------------------------------------------------
    def update(self):
        self._refresh_pattern()

    def read(self) -> Dict[str, Any]:
        return self._params.snapshot()

    def register(self):
        self._params.changed.connect(self._on_param_changed)
        self._params.bulkChanged.connect(self._on_bulk_changed)

    def _refresh_pattern(self):
        self.canvas.setPattern(self.getPattern(self.read()))

    # ------------------------------------------------------------------
    # Signal callbacks
    # ------------------------------------------------------------------
    def _on_param_changed(self, key: str, value: Any):
        self._refresh_pattern()

    def _on_bulk_changed(self, payload: Dict[str, Any]):
        self._refresh_pattern()

    def _buildOuter(self):

        pass

    def _buildInner(self):
        pass

    def _buildConvexHull(self):
        pass

    def __buildShape(self):
        params = self.read()
        pass

    def _checkCache(self, params: Dict[str, Any]):
        pass

    def _buildAssist(self, params: Dict[str, Any]):
        '''
        psram = {
            "pattern_tp0": 0.0,
            "pattern_tp1": 0.0,
            "pattern_tp2": 0.0,
            "pattern_tp3": 0.0,
            "pattern_tnn": 2.0,
            "pattern_tmm": 2.0,
            "pattern_tcc": 0.0,
            "pattern_bp0": 0.0,
            "pattern_bp1": 0.0,
            "pattern_bp2": 0.0,
            "pattern_bp3": 0.0,
            "pattern_bnn": 2.0,
            "pattern_bmm": 2.0,
            "pattern_bcc": 0.0,
            "pattern_pbh": 100.0,
            "pattern_pbw": 200.0,
            "pattern_ppw": 5.0,
            "pattern_mode": "straight",
            "pattern_twist": False,
        }
        '''
        # counterclockwise
        # line : x1,y1,x2,y2
        half_width = float(params.get("pattern_pbw", 0.0) or 0.0) / 2.0
        half_height = float(params.get("pattern_pbh", 0.0) or 0.0) / 2.0

        top_points = np.array([
            [half_width,                     float(params.get("pattern_pbh", 0.0) or 0.0)],
            [float(params.get("pattern_tp1", 0.0) or 0.0), float(params.get("pattern_pbh", 0.0) or 0.0)],
            [0.0,                            half_height + float(params.get("pattern_tp3", 0.0) or 0.0)],
            [0.0,                            half_height],
        ], dtype=np.float64)

        bottom_points = np.array([
            [0.0,                            half_height],
            [0.0,                            float(params.get("pattern_bp2", 0.0) or 0.0)],
            [float(params.get("pattern_bp1", 0.0) or 0.0), 0.0],
            [half_width,                     0.0],
        ], dtype=np.float64)

        vertical_center = np.array([
            [half_width, 0.0],
            [half_width, float(params.get("pattern_pbh", 0.0) or 0.0)],
        ], dtype=np.float64)
        horizontal_center = np.array([
            [0.0, half_height],
            [float(params.get("pattern_pbw", 0.0) or 0.0), half_height],
        ], dtype=np.float64)
        return {
            "top": top_points,
            "bottom": bottom_points,
            "vcenter": vertical_center,
            "hcenter": horizontal_center,
        }

    def _buildBbox(self, params: Dict[str, Any]):
        return np.array([
            [0.0, 0.0],
            [float(params.get("pattern_pbw", 0.0) or 0.0), 0.0],
            [float(params.get("pattern_pbw", 0.0) or 0.0), float(params.get("pattern_pbh", 0.0) or 0.0)],
            [0.0, float(params.get("pattern_pbh", 0.0) or 0.0)],
        ], dtype=np.float64)

    def getPattern(self, params: Dict[str, Any]):
        """
        return {
            "bbox": {},
            "convex": {},
            "shape": {},
            "assist": {},
            "info": {}
        }
        """
        snapshot = params or self.read()
        return {
            "bbox": self._buildBbox(snapshot),
            "assist": self._buildAssist(snapshot),
        }


class PatternCanvas(QWidget):
    """Simple 2D canvas that draws bbox + assist polylines."""

    MARGIN = 12
    ASSIST_COLORS = [
        QColor("#ff6b6b"),
        QColor("#4ecdc4"),
        QColor("#ffa600"),
        QColor("#5c7cfa"),
        QColor("#b15cff"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pattern: Dict[str, Any] = {}
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPattern(self, pattern: Dict[str, Any]):
        self._pattern = pattern or {}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().window())

        if not self._pattern:
            return

        bbox_pts = self._to_point_list(self._pattern.get("bbox"))
        assist = self._pattern.get("assist") or {}
        assist_polys = {name: self._to_point_list(poly) for name, poly in assist.items()}

        all_points: List[Tuple[float, float]] = []
        all_points.extend(bbox_pts)
        for pts in assist_polys.values():
            all_points.extend(pts)

        if not all_points:
            return

        bounds = self._compute_bounds(all_points)
        if bounds is None:
            return
        mapper = self._build_mapper(bounds)

        self._draw_bbox(painter, mapper, bbox_pts)
        self._draw_assist(painter, mapper, assist_polys)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_bbox(self, painter: QPainter, mapper, points: List[Tuple[float, float]]):
        if len(points) < 3:
            return

        polygon = QPolygonF([mapper(pt) for pt in points])
        fill = QBrush(QColor(74, 144, 226, 60))
        outline = QPen(QColor(74, 144, 226), 1.4)
        outline.setCosmetic(True)

        painter.setBrush(fill)
        painter.setPen(outline)
        painter.drawPolygon(polygon)

    def _draw_assist(self, painter: QPainter, mapper, polylines: Dict[str, List[Tuple[float, float]]]):
        painter.setBrush(Qt.NoBrush)
        for idx, (name, pts) in enumerate(polylines.items()):
            if len(pts) < 2:
                continue
            color = self.ASSIST_COLORS[idx % len(self.ASSIST_COLORS)]
            pen = QPen(color, 0.5)
            pen.setStyle(Qt.DashLine if name not in ("vcenter", "hcenter") else Qt.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)

            poly = QPolygonF([mapper(pt) for pt in pts])
            painter.drawPolyline(poly)

            # Draw vertices for clarity
            vertex_pen = QPen(color)
            vertex_pen.setWidth(4)
            vertex_pen.setCosmetic(True)
            painter.setPen(vertex_pen)
            for point in pts:
                painter.drawPoint(mapper(point))

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _compute_bounds(self, points: Iterable[Tuple[float, float]]):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if not xs or not ys:
            return None
        return min(xs), max(xs), min(ys), max(ys)

    def _build_mapper(self, bounds):
        min_x, max_x, min_y, max_y = bounds
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)

        rect = self.rect().adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)
        available_w = max(rect.width(), 1)
        available_h = max(rect.height(), 1)

        scale = min(available_w / span_x, available_h / span_y)
        target_w = span_x * scale
        target_h = span_y * scale
        origin_x = rect.left() + (available_w - target_w) / 2.0
        origin_y = rect.top() + (available_h - target_h) / 2.0

        def mapper(point: Tuple[float, float]) -> QPointF:
            x = origin_x + (point[0] - min_x) * scale
            # invert Y so that larger Y goes up visually
            y = origin_y + target_h - (point[1] - min_y) * scale
            return QPointF(x, y)

        return mapper

    @staticmethod
    def _to_point_list(data) -> List[Tuple[float, float]]:
        if data is None:
            return []
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return []
        return [(float(x), float(y)) for x, y in arr]
