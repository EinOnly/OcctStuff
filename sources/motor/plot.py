from typing import Dict, Any, List, Tuple

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


class Plot(QWidget):
    """Wrapper panel that hosts the PlotCanvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = PlotCanvas()
        layout.addWidget(self.canvas)

    def setLayers(self, layers: Dict[str, Any]):
        """Update the chart using freshly generated layer data."""
        self.canvas.setLayers(layers)


class PlotCanvas(QWidget):
    """Custom painter-based canvas to render pattern metrics."""

    MARGIN = 24
    RIGHT_MARGIN = 60  # Extra margin for right y-axis

    # Left y-axis metrics (areas)
    LEFT_METRICS = [
        ("convexhull_area", "Convex Hull Area", "mm²", QColor("#4ecdc4")),
        ("pattern_area", "Pattern Area", "mm²", QColor("#ffa600")),
    ]

    # Right y-axis metrics (resistance)
    RIGHT_METRICS = [
        ("pattern_resistance", "Resistance", "mΩ", QColor("#ff6b6b")),
    ]

    # All metrics combined for compatibility
    METRICS = LEFT_METRICS + RIGHT_METRICS

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points: List[Dict[str, Any]] = []
        self._layer_breaks: List[Tuple[int, str]] = []
        self._stats: Dict[str, Dict[str, Tuple[int, float]]] = {}
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setLayers(self, layers: Dict[str, Any]):
        layers = layers or {}
        self._points, self._layer_breaks = self._extract_points(layers.get("front", []))
        self._stats = self._compute_stats()
        self.update()

    def _extract_points(self, front_layers: List[Dict[str, Any]]):
        points: List[Dict[str, Any]] = []
        boundaries: List[Tuple[int, str]] = []
        last_layer_index = None

        for idx, entry in enumerate(front_layers):
            metrics = entry.get("metrics") or {}
            layer_index = entry.get("layer_index", 0)
            layer_label = entry.get("layer_label") or f"Layer {layer_index + 1}"

            if last_layer_index != layer_index:
                boundaries.append((idx, layer_label))
                last_layer_index = layer_index

            points.append({
                "x": idx,
                "layer": layer_index,
                "layer_label": layer_label,
                "pattern": entry.get("index", idx),
                "convexhull_area": float(metrics.get("convexhull_area", 0.0) or 0.0),
                "pattern_area": float(metrics.get("pattern_area", 0.0) or 0.0),
                "pattern_resistance": float(metrics.get("pattern_resistance", 0.0) or 0.0) * 1000.0,
            })

        if front_layers:
            boundaries.append((len(front_layers), None))

        return points, boundaries

    def _compute_stats(self) -> Dict[str, Dict[str, Tuple[int, float]]]:
        stats: Dict[str, Dict[str, Tuple[int, float]]] = {}
        for key, _label, _unit, _color in self.METRICS:
            values = [(pt["x"], pt[key]) for pt in self._points]
            if not values:
                continue
            stats[key] = {
                "max": max(values, key=lambda v: v[1]),
                "min": min(values, key=lambda v: v[1]),
            }
        return stats

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().window())

        if not self._points:
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "Generate layers to preview metrics.")
            return

        # Adjust plot rect with extra margin on the right for second y-axis
        plot_rect = self.rect().adjusted(self.MARGIN, self.MARGIN, -self.RIGHT_MARGIN, -self.MARGIN)
        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        # Calculate value ranges for left y-axis (areas)
        left_values = [
            pt[key]
            for pt in self._points
            for key, _label, _unit, _color in self.LEFT_METRICS
        ]
        valid_left = [v for v in left_values if v is not None and v > 0]

        # Calculate value ranges for right y-axis (resistance)
        right_values = [
            pt[key]
            for pt in self._points
            for key, _label, _unit, _color in self.RIGHT_METRICS
        ]
        valid_right = [v for v in right_values if v is not None and v > 0]

        if not valid_left and not valid_right:
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "No metric data available.")
            return

        # Left y-axis range (areas)
        if valid_left:
            min_left = min(valid_left)
            max_left = max(valid_left)
            if abs(max_left - min_left) < 1e-6:
                padding = 1.0 if max_left == 0 else abs(max_left) * 0.05
                min_left -= padding
                max_left += padding
            span_left = max(max_left - min_left, 1e-6)
        else:
            min_left, max_left, span_left = 0, 1, 1

        # Right y-axis range (resistance)
        if valid_right:
            min_right = min(valid_right)
            max_right = max(valid_right)
            if abs(max_right - min_right) < 1e-6:
                padding = 1.0 if max_right == 0 else abs(max_right) * 0.05
                min_right -= padding
                max_right += padding
            span_right = max(max_right - min_right, 1e-6)
        else:
            min_right, max_right, span_right = 0, 1, 1

        total_points = len(self._points)
        x_divisor = max(total_points - 1, 1)

        def map_x(idx: int) -> float:
            if total_points == 1:
                return plot_rect.left() + plot_rect.width() / 2.0
            return plot_rect.left() + (idx / x_divisor) * plot_rect.width()

        def map_y_left(value: float) -> float:
            """Map value to y coordinate using left axis scale (areas)."""
            ratio = (value - min_left) / span_left
            return plot_rect.bottom() - ratio * plot_rect.height()

        def map_y_right(value: float) -> float:
            """Map value to y coordinate using right axis scale (resistance)."""
            ratio = (value - min_right) / span_right
            return plot_rect.bottom() - ratio * plot_rect.height()

        # Draw axes
        axis_pen = QPen(QColor("#666666"), 1.2)
        painter.setPen(axis_pen)
        # X-axis (bottom)
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.bottomRight())
        # Left y-axis (areas)
        painter.setPen(QPen(QColor("#4ecdc4"), 1.5))
        painter.drawLine(plot_rect.bottomLeft(), plot_rect.topLeft())
        # Right y-axis (resistance)
        painter.setPen(QPen(QColor("#ff6b6b"), 1.5))
        right_axis_x = plot_rect.right()
        painter.drawLine(QPointF(right_axis_x, plot_rect.bottom()),
                        QPointF(right_axis_x, plot_rect.top()))

        # Horizontal guide lines with y-axis labels
        guide_pen = QPen(QColor("#cccccc"), 1, Qt.DashLine)
        painter.setPen(guide_pen)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot_rect.bottom() - frac * plot_rect.height()
            if frac > 0 and frac < 1.0:
                painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))

            # Left y-axis labels (areas) - in cyan color
            if valid_left:
                left_value = min_left + frac * span_left
                painter.setPen(QPen(QColor("#4ecdc4")))
                painter.drawText(QPointF(plot_rect.left() - self.MARGIN + 2, y + 4), f"{left_value:.1f}")

            # Right y-axis labels (resistance) - in red color
            if valid_right:
                right_value = min_right + frac * span_right
                painter.setPen(QPen(QColor("#ff6b6b")))
                painter.drawText(QPointF(plot_rect.right() + 4, y + 4), f"{right_value:.1f}")

        # Layer separators
        if total_points > 1:
            sep_pen = QPen(QColor("#aaaaaa"), 1, Qt.DotLine)
            painter.setPen(sep_pen)
            for idx, label in self._layer_breaks:
                if idx <= 0 or idx >= total_points:
                    continue
                x = map_x(idx)
                painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
                if label:
                    painter.save()
                    painter.setPen(Qt.gray)
                    painter.drawText(QPointF(x + 4, plot_rect.top() + 14), label)
                    painter.restore()

        # Draw metric lines
        for key, label, unit, color in self.METRICS:
            # Determine which y-axis to use
            is_left_axis = any(k == key for k, _, _, _ in self.LEFT_METRICS)
            map_y = map_y_left if is_left_axis else map_y_right

            # Create points using the appropriate y-axis mapping
            series = [QPointF(map_x(pt["x"]), map_y(pt[key]))
                     for pt in self._points if pt[key] > 0]
            if not series:
                continue

            painter.setPen(QPen(color, 2.0))
            painter.setBrush(Qt.NoBrush)

            if len(series) == 1:
                painter.drawEllipse(series[0], 2, 2)
            else:
                painter.drawPolyline(QPolygonF(series))
                painter.setBrush(QBrush(color))
                for pt in series:
                    painter.drawEllipse(pt, 2, 2)

            stat = self._stats.get(key)
            if not stat:
                continue

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color))
            max_idx, max_val = stat["max"]
            min_idx, min_val = stat["min"]

            # Skip if values are invalid
            if max_val <= 0 or min_val <= 0:
                continue

            max_point = QPointF(map_x(max_idx), map_y(max_val))
            min_point = QPointF(map_x(min_idx), map_y(min_val))
            painter.drawEllipse(max_point, 3, 3)
            painter.drawEllipse(min_point, 3, 3)

            painter.setPen(QPen(color))
            painter.drawText(max_point + QPointF(6, -6), f"max {max_val:.3f}{unit}")
            painter.drawText(min_point + QPointF(6, 14), f"min {min_val:.3f}{unit}")

        self._draw_summary(painter, plot_rect)

    def _draw_summary(self, painter: QPainter, plot_rect: QRectF):
        """Render min/max summary box inside the plotting area."""
        lines = []
        for key, label, unit, color in self.METRICS:
            stat = self._stats.get(key)
            if not stat:
                continue
            max_val = stat["max"][1]
            min_val = stat["min"][1]
            lines.append((color, f"{label}: min {min_val:.3f}{unit}, max {max_val:.3f}{unit}"))

        if not lines:
            return

        padding = 8
        line_height = 18
        metrics = painter.fontMetrics()
        box_width = max(220, max(metrics.horizontalAdvance(text) for _color, text in lines) + 24)
        box_height = padding * 2 + line_height * len(lines)
        rect = QRectF(
            plot_rect.left() + padding,
            plot_rect.top() + padding,
            box_width,
            box_height,
        )

        painter.save()
        painter.setBrush(QColor(255, 255, 255, 180))
        painter.setPen(QPen(QColor("#444444"), 1))
        painter.drawRoundedRect(rect, 6, 6)

        y = rect.top() + padding + line_height - 6
        for color, text in lines:
            marker = QRectF(rect.left() + padding - 4, y - 10, 10, 10)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRect(marker)
            painter.setPen(QPen(QColor("#222222")))
            painter.setBrush(Qt.NoBrush)
            painter.drawText(QPointF(rect.left() + padding + 12, y), text)
            y += line_height
        painter.restore()
