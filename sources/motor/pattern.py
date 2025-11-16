import numpy as np

from superellipse import Superellipse
from parameters import PPARAMS
from calculate import Calculate

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from typing import Dict, Any, Iterable, List, Tuple

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
        self.canvas.setPattern(Pattern.GetPattern(self.read(), self.read(), self.read()))

    # ------------------------------------------------------------------
    # Signal callbacks
    # ------------------------------------------------------------------
    def _on_param_changed(self, key: str, value: Any):
        self._refresh_pattern()

    def _on_bulk_changed(self, payload: Dict[str, Any]):
        self._refresh_pattern()

    @staticmethod
    def _buildTopOuter(assist: Dict[str, Any]) -> np.ndarray:
        """Build top portion of outer curve with tcc boundary clamping."""
        points = assist.get("top")
        mode = assist.get("mode", "straight")
        radius = assist.get("max_tx", 0.0)

        if points is None:
            return np.array([], dtype=np.float64).reshape(0, 2)

        top = np.asarray(points, dtype=np.float64)

        if mode == "straight":
            # Straight mode: return all points directly
            return top.copy()

        elif mode == "superelliptic":
            superellipse = Superellipse.getInstance()
            n = assist.get("tnn", 2.0)
            m = assist.get("tmm", 2.0)
            superellipse.setExponents(n, m)

            # Generate superellipse curve from top[0] to top[1]
            curve_points = superellipse.generate(radius, radius, 0, top[0][1], 0)

            # Convert to numpy array and reverse (skip last point which is the first)
            curve_arr = np.array(curve_points, dtype=np.float64)
            return curve_arr[::-1]  # Reverse and skip first (originally last)

        # Fallback
        return top.copy()

    @staticmethod
    def _buildTopInner(assist: Dict[str, Any], pwidth: float) -> np.ndarray:
        """Build top portion of outer curve with tcc boundary clamping."""
        base = Pattern._buildTopOuter(assist)
        spacing = assist.get("spacing", 0.0)
        shift = pwidth + spacing
        clamp = assist.get("max_tx", 0.0) + spacing
        base[:, 0] += shift
        inner = Calculate.Offset(base, spacing)
        # Vectorized x-offset: shift all x-coordinates
        inner = Calculate.Clamp(inner, clamp)
        # Apply tcc boundary clamping
        return inner[::-1]

    @staticmethod
    def _buildBottomOuter(assist: Dict[str, Any]) -> np.ndarray:
        """Build bottom portion of outer curve with bcc boundary clamping."""
        points = assist.get("bottom")
        mode = assist.get("mode", "straight")
        radius = assist.get("max_bx", 0.0)

        if points is None:
            return np.array([], dtype=np.float64).reshape(0, 2)

        bottom = np.asarray(points, dtype=np.float64)

        if mode == "straight":
            # Straight mode: return all points directly
            return bottom.copy()

        elif mode == "superelliptic":
            superellipse = Superellipse.getInstance()
            n = assist.get("bnn", 2.0)
            m = assist.get("bmm", 2.0)
            superellipse.setExponents(n, m)

            # Generate superellipse curve from bottom[0] to bottom[1]
            curve_points = superellipse.generate(radius, radius, 0, 0, 3)

            # Convert to numpy array and reverse (skip last point which is the first)
            curve_arr = np.array(curve_points, dtype=np.float64)
            return curve_arr[::-1][1:]  # Reverse and skip first (originally last)

        # Fallback
        return bottom.copy()

    @staticmethod
    def _buildBottomInner(assist: Dict[str, Any], pwidth: float) -> np.ndarray:
        """Build top portion of outer curve with tcc boundary clamping."""
        base = Pattern._buildBottomOuter(assist)
        spacing = assist.get("spacing", 0.0)
        shift = pwidth + spacing
        clamp = assist.get("max_bx", 0.0) + spacing
        base[:, 0] += shift
        inner = Calculate.Offset(base, spacing)
        # Vectorized x-offset: shift all x-coordinates
        inner = Calculate.Clamp(inner, clamp)
        # Apply tcc boundary clamping
        return inner[::-1]

    @staticmethod
    def _buildTop(currentAssist: Dict[str, Any], nextAssist: Dict[str, Any]) -> np.ndarray:
        outer = Pattern._buildTopOuter(currentAssist)
        inner = Pattern._buildTopInner(nextAssist, currentAssist.get("pwidth", 0.0))
        twist = currentAssist.get("twist", False)
        if twist:
            axis_x = currentAssist.get("pwidth", 0.0) / 2
            outer_t = Calculate.Mirror(outer.copy(), axis_x)
            inner_t = Calculate.Mirror(inner.copy(), axis_x)
            inner = outer_t[::-1]
            outer = inner_t[::-1]
        return (outer, inner)

    @staticmethod
    def _buildBottom(currentAssist: Dict[str, Any], nextAssist: Dict[str, Any]) -> np.ndarray:
        outer = Pattern._buildBottomOuter(currentAssist)
        inner = Pattern._buildBottomInner(nextAssist, currentAssist.get("pwidth", 0.0))
        return (outer, inner)

    @staticmethod
    def _buildConvexHull():
        pass

    @staticmethod
    def _buildShape(currentAssist: Dict[str, Any], nextAssist: Dict[str, Any], location:str = "normal") -> np.ndarray:
        """
        Build closed shape by combining outer and inner curves.

        Steps:
        1. Get outer curve from current pattern
        2. Get inner curve from next pattern (with offset)
        3. Translate inner curve by pattern_ppw in X direction
        4. If pattern_twist is True, mirror bottom portion of inner curve
        5. Combine into closed polygon

        Args:
            currentAssist: Current pattern assist data
            nextAssist: Next pattern assist data

        Returns:
            numpy array forming a closed shape
        """
        if location == "start":
            top = Pattern._buildTop(currentAssist, nextAssist)
            bottom = Pattern._buildBottom(currentAssist, currentAssist)
        elif location == "end":
            top = Pattern._buildTop(currentAssist, currentAssist)
            bottom = Pattern._buildBottom(currentAssist, nextAssist)
        else:
            top = Pattern._buildTop(currentAssist, nextAssist)
            bottom = Pattern._buildBottom(currentAssist, nextAssist)
        curve = np.vstack([
            top[0],
            bottom[0],
            bottom[1],
            top[1],
        ])
        return curve

    @staticmethod
    def _buildAssist(params: Dict[str, Any]):
        """
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
        """
        if params is None:
            return None
        
        # counterclockwise
        # line : x1,y1,x2,y2
        width = float(params.get("pattern_pbw", 0.0) or 0.0)
        height = float(params.get("pattern_pbh", 0.0) or 0.0)
        mode = params.get("pattern_mode", "straight")
        pattern_width = float(params.get("pattern_ppw", 0.0) or 0.0)

        # Get tp1, bp1 values
        tp1 = float(params.get("pattern_tp1", 0.0) or 0.0)
        tp3 = float(params.get("pattern_tp3", 0.0) or 0.0)
        bp1 = float(params.get("pattern_bp1", 0.0) or 0.0)
        bp2 = float(params.get("pattern_bp2", 0.0) or 0.0)

        tcc = float(params.get("pattern_tcc", width/2.0) or 0.0)
        bcc = float(params.get("pattern_bcc", width/2.0) or 0.0)

        top_points = None
        bottom_points = None
        if mode == "straight":
            top_points = np.array(
                [
                    [tp1,     height],
                    [tp1 - pattern_width/2,     height],
                    [0.0,     height/2.0 + tp3],
                    [0.0,     height/2.0],
                ],
                dtype=np.float64,
            )
            bottom_points = np.array(
                [
                    [0.0, height/2.0],
                    [0.0, bp2],
                    [bp1 - pattern_width/2, 0.0],
                    [bp1, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            top_points = np.array(
                [
                    [tp1,     height],
                    [0.0,     height/2.0 + tp3],
                    [0.0,     height/2.0],
                ],
                dtype=np.float64,
            )
            bottom_points = np.array(
                [
                    [0.0, height/2.0],
                    [0.0, bp2],
                    [bp1, 0.0],
                ],
                dtype=np.float64,
            )

        vertical_clamp = np.array(
            [
                [tp1, height],  # top: tcc at y=pbh
                [bp1, 0.0],  # bottom: bcc at y=0
            ],
            dtype=np.float64,
        )
        horizontal_clamp = np.array(
            [
                [0.0,   height/2.0],
                [width, height/2.0],
            ],
            dtype=np.float64,
        )
        return {
            "top": top_points,
            "bottom": bottom_points,
            "mode": mode,
            "twist": params.get("pattern_twist", False),
            "tnn": float(params.get("pattern_tnn", 2.0) or 2.0),
            "tmm": float(params.get("pattern_tmm", 2.0) or 2.0),
            "bnn": float(params.get("pattern_bnn", 2.0) or 2.0),
            "bmm": float(params.get("pattern_bmm", 2.0) or 2.0),
            "tcc": tcc,
            "bcc": bcc,
            "width": width,
            "height": height,
            "pwidth": pattern_width,
            "max_tx": tp1,
            "max_bx": bp1,
            "spacing": float(params.get("pattern_psp", 0.0) or 0.0),
            "vclamp": vertical_clamp,
            "hclamp": horizontal_clamp,
        }

    @staticmethod
    def _buildBbox(params: Dict[str, Any]):
        return np.array(
            [
                [0.0, 0.0],
                [float(params.get("pattern_pbw", 0.0) or 0.0), 0.0],
                [
                    float(params.get("pattern_pbw", 0.0) or 0.0),
                    float(params.get("pattern_pbh", 0.0) or 0.0),
                ],
                [0.0, float(params.get("pattern_pbh", 0.0) or 0.0)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def GetPattern(currentParams: Dict[str, Any], nextParams: Dict[str, Any], location:str = "normal") -> Dict[str, Any]:
        """
        return {
            "bbox": {},
            "assist": {},
            "shape": {},
            "convex": {},
        }
        """
        if currentParams is None or nextParams is None:
            raise ValueError("Both currentParams and nextParams are required")

        return {
            "bbox": Pattern._buildBbox(currentParams),
            "assist": Pattern._buildAssist(currentParams),
            "shape": Pattern._buildShape(
                Pattern._buildAssist(currentParams),
                Pattern._buildAssist(nextParams),
                location
            ),
            "convexhull": [],
            "convexhull_area": 0.0,
            "pattern_area": 0.0,
            "pattern_resistance": 0.0,
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
        assist_polys = {
            name: self._to_point_list(poly)
            for name, poly in assist.items()
            if not isinstance(poly, (str, bool, int, float))
        }
        shape_pts = self._to_point_list(self._pattern.get("shape"))

        all_points: List[Tuple[float, float]] = []
        all_points.extend(bbox_pts)
        for pts in assist_polys.values():
            all_points.extend(pts)
        all_points.extend(shape_pts)

        if not all_points:
            return

        bounds = self._compute_bounds(all_points)
        if bounds is None:
            return
        mapper = self._build_mapper(bounds)

        self._draw_bbox(painter, mapper, bbox_pts)
        self._draw_shape(painter, mapper, shape_pts)
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

    def _draw_assist(
        self, painter: QPainter, mapper, polylines: Dict[str, List[Tuple[float, float]]]
    ):
        painter.setBrush(Qt.NoBrush)
        for idx, (name, pts) in enumerate(polylines.items()):
            if len(pts) < 2:
                continue
            color = self.ASSIST_COLORS[idx % len(self.ASSIST_COLORS)]
            pen = QPen(color, 0.5)
            pen.setStyle(
                Qt.DashLine if name not in ("vclamp", "hclamp") else Qt.SolidLine
            )
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

        rect = self.rect().adjusted(
            self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN
        )
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
        try:
            arr = np.asarray(data, dtype=np.float64)
        except (ValueError, TypeError):
            return []
        if arr.ndim != 2 or arr.shape[1] < 2:
            return []
        return [(float(x), float(y)) for x, y in arr]
