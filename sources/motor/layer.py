import numpy as np

from log import CORELOG
from parameters import LPARAMS
from pattern import Pattern
from calculate import Calculate

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QGestureRecognizer
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


        CORELOG.info(f"Layers initialized.")

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
        # import pprint
        # pprint.pprint(self._params.snapshot())
        self.canvas.setLayers(self.getLayers())

    # ------------------------------------------------------------------
    # Signal callbacks
    # ------------------------------------------------------------------
    def _on_param_changed(self, key: str, value: Any):
        self._refresh_layers()

    def _on_bulk_changed(self, payload: Dict[str, Any]):
        self._refresh_layers()

    def _buildPattern(self,
        layers: Dict[str, Any], 
        currentParams: Dict[str, Any], 
        nextParams: Dict[str, Any], 
        color: str,
        offset: float = 0.0,
        location: str = "normal",
        back: bool = False,
        index: int = 0

    ):
        pattern = Pattern.GetPattern(currentParams, nextParams, location)

        # Get shape and apply horizontal offset
        shape = pattern.get("shape")
        if shape is not None and len(shape) > 0:
            shape_offset = shape.copy()
            shape_offset[:, 0] += offset

            layers["front"].append({
                "shape": shape_offset,
                "color": color,
                "index": index
            })
            
            if back:
                shape_back = shape.copy()
                # flip this pattern here
                shape_back = Calculate.Mirror(shape_back, currentParams.get("pattern_ppw", 0))

                layers["back"].append({
                    "shape": shape_back,
                    "color": color,
                    "index": index
                })
        

    def _buildStart(self, layers: Dict[str, Any], currentConfig: Dict[str, Any], nextConfig: Dict[str, Any], start_offset: float = 0.0):
        """Build start layer: all patterns use same config except last one transitions to next layer."""
        layer_params = currentConfig.get("layer", {})
        count = layer_params.get("layer_pdc", 9)

        ppw = layer_params.get("layer_ppw", 0.5)
        psp = layer_params.get("layer_psp", 0.05)
        color = layer_params.get("color", "#de7cfc")
        offset = start_offset
        location = "normal"

        currentParams = None
        nextParams = None
        back = True

        for i in range(count):

            # Handled the last pattern of each layer separately
            if i == count - 1 and nextConfig is not None:
                # Last pattern transitions to next layer
                currentParams = currentConfig.get("layer", {}).copy()
                nextParams = nextConfig.get("layer", {}).copy()
                location = "end"
            elif i < 8:
                currentParams = currentConfig.get("layer", {}).copy()
                currentParams["pattern_twist"] = False
                currentParams["pattern_tp1"] += currentParams["pattern_ppw"]
                nextParams = currentParams
                back = False
            else:
                # Regular patterns (current -> current)
                currentParams = currentConfig.get("layer", {})
                nextParams = currentParams

            # build pattern here
            self._buildPattern( layers, currentParams, nextParams, color, offset, location, back, i )
            # Move offset for next pattern
            offset += ppw + psp

        return offset  # Return final offset for next layer

    def _buildNormal(self, layers: Dict[str, Any], preConfig: Dict[str, Any], currentConfig: Dict[str, Any], nextConfig: Dict[str, Any], start_offset: float = 0.0):
        """Build normal layer: all patterns transition to next layer."""
        layer_params = currentConfig.get("layer", {})
        count = layer_params.get("layer_pdc", 9)

        ppw = layer_params.get("layer_ppw", 0.5)
        psp = layer_params.get("layer_psp", 0.05)
        color = layer_params.get("color", "#de7cfc")
        offset = start_offset
        location = "normal"

        currentParams = None
        nextParams = None
        back = True

        for i in range(count):

            # Handled the last pattern of each layer separately
            if i == 0 and preConfig is not None:
                # Last pattern transitions to next layer
                currentParams = currentConfig.get("layer", {}).copy()
                nextParams = preConfig.get("layer", {}).copy()
                location = "start"
            elif i == count - 1 and nextConfig is not None:
                # Last pattern transitions to next layer
                currentParams = currentConfig.get("layer", {}).copy()
                nextParams = nextConfig.get("layer", {}).copy()
                location = "end"
            else:
                # Regular patterns (current -> current)
                currentParams = currentConfig.get("layer", {})
                nextParams = currentParams

            # build pattern here
            self._buildPattern( layers, currentParams, nextParams, color, offset, location, back, i )
            # Move offset for next pattern
            offset += ppw + psp

        return offset  # Return final offset for next layer

    def _buildEnd(self, layers: Dict[str, Any], preConfig: Dict[str, Any], currentConfig: Dict[str, Any], start_offset: float = 0.0):
        """Build end layer: all patterns use same config (no transition)."""
        layer_params = currentConfig.get("layer", {})
        count = layer_params.get("layer_pdc", 9)

        ppw = layer_params.get("layer_ppw", 0.5)
        psp = layer_params.get("layer_psp", 0.05)
        color = layer_params.get("color", "#de7cfc")
        offset = start_offset
        location = "normal"

        currentParams = None
        nextParams = None

        for i in range(count):

            # Handled the last pattern of each layer separately
            if i == 0 and preConfig is not None:
                # Last pattern transitions to next layer
                currentParams = currentConfig.get("layer", {}).copy()
                nextParams = preConfig.get("layer", {}).copy()
                location = "start"
            else:
                # Regular patterns (current -> current)
                currentParams = currentConfig.get("layer", {})
                nextParams = currentParams

            # build pattern here
            self._buildPattern( layers, currentParams, nextParams, color, offset, location, False, i )
            # Move offset for next pattern
            offset += ppw + psp

        return offset  # Return final offset for next layer

    def getLayers(self) -> Dict[str, Any]:
        '''
        Build all layers based on current parameters.
        Returns a dictionary with "front" and "back" layers.
        - special tags:
        - Layer tags
        1. start: the first layer
        2. mormal: normal layers between first and last
        3. end: the last layer
        - location tag:
        1. start: first pattern of each layer
        2. normal: ...
        2. end: last pattern of each layer
        - position:
        1. front
        2. back
        '''
        layers = { "front": [], "back": []}
        configs = self._params.snapshot().get("layers", [])
        cumulative_offset = 0.0

        for idx, currentLayerConfig in enumerate(configs):
            layer_type = currentLayerConfig.get("type", "normal")
            nexLayerConfig = configs[idx + 1] if idx + 1 < len(configs) else None
            preLayerConfig = configs[idx - 1] if idx - 1 >= 0 else None

            match layer_type:
                case "start":
                    CORELOG.info(f"Building start layer at index {idx}")
                    cumulative_offset = self._buildStart(
                        layers, 
                        currentLayerConfig,
                        nexLayerConfig,
                        cumulative_offset
                    )
                    CORELOG.info(f"Built layer at index {idx} successfully, offset: {cumulative_offset}")
                case "normal":
                    CORELOG.info(f"Building normal layer at index {idx}")
                    cumulative_offset = self._buildNormal(
                        layers, 
                        preLayerConfig,
                        currentLayerConfig, 
                        nexLayerConfig, 
                        cumulative_offset
                    )
                    CORELOG.info(f"Built layer at index {idx} successfully, offset: {cumulative_offset}")
                case "end":
                    CORELOG.info(f"Building end layer at index {idx}")
                    cumulative_offset = self._buildEnd(
                        layers, 
                        preLayerConfig,
                        currentLayerConfig, 
                        cumulative_offset
                    )
                    CORELOG.info(f"Built layer at index {idx} successfully, offset: {cumulative_offset}")
                case _:
                    CORELOG.warn(f"Unknown layer type: {layer_type} at index {idx}")
                    continue
        return layers

class LayerCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: Dict[str, Any] = {}
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Zoom and Pan state
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_mouse_pos = None
        self._is_panning = False

        # Gesture state
        self._pinch_scale_factor = 1.0
        self._last_wheel_pos = None

        # Enable gestures for Mac trackpad
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.PanGesture)

        # Enable mouse tracking for pan
        self.setMouseTracking(True)

        # Accept touch events
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)

    def setLayers(self, layers: Dict[str, Any]):
        self._layers = layers or {}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), self.palette().window())

        if not self._layers or not self._layers.get("front"):
            return

        front_shapes = self._layers.get("front", [])

        # Collect all points to compute bounds
        all_points: List[Tuple[float, float]] = []
        for shape_data in front_shapes:
            shape = shape_data.get("shape")
            if shape is not None and len(shape) > 0:
                all_points.extend([(pt[0], pt[1]) for pt in shape])

        if not all_points:
            return

        bounds = self._compute_bounds(all_points)
        if bounds is None:
            return

        mapper = self._build_mapper(bounds)

        # Draw all shapes
        for shape_data in front_shapes:
            shape = shape_data.get("shape")
            color = shape_data.get("color", "#de7cfc")
            if shape is not None and len(shape) > 0:
                points = [(pt[0], pt[1]) for pt in shape]
                self._draw_shape(painter, mapper, points, color)

    def _compute_bounds(self, points: List[Tuple[float, float]]):
        """Compute bounding box of all points."""
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Add some padding
        width = max_x - min_x
        height = max_y - min_y
        padding = max(width, height) * 0.1

        return {
            "min_x": min_x - padding,
            "max_x": max_x + padding,
            "min_y": min_y - padding,
            "max_y": max_y + padding,
        }

    def _build_mapper(self, bounds):
        """Build coordinate mapper from world to screen with zoom and pan."""
        MARGIN = 12
        w = self.width() - 2 * MARGIN
        h = self.height() - 2 * MARGIN

        world_w = bounds["max_x"] - bounds["min_x"]
        world_h = bounds["max_y"] - bounds["min_y"]

        if world_w == 0 or world_h == 0:
            base_scale = 1.0
        else:
            base_scale = min(w / world_w, h / world_h)

        # Apply zoom
        scale = base_scale * self._zoom

        def mapper(pt):
            # Transform to screen coordinates
            x = (pt[0] - bounds["min_x"]) * scale + MARGIN
            # Flip Y axis (screen Y increases downward)
            y = h - (pt[1] - bounds["min_y"]) * scale + MARGIN

            # Apply pan offset
            x += self._pan_x
            y += self._pan_y

            return QPointF(x, y)

        return mapper

    def _draw_shape(self, painter: QPainter, mapper, points: List[Tuple[float, float]], color_str: str):
        """Draw a single shape with semi-transparent fill."""
        if len(points) < 3:
            return

        polygon = QPolygonF([mapper(pt) for pt in points])

        # Parse color string
        color = QColor(color_str)
        fill = QBrush(QColor(color.red(), color.green(), color.blue(), 100))
        outline = QPen(color, 1.5)
        outline.setCosmetic(True)

        painter.setBrush(fill)
        painter.setPen(outline)
        painter.drawPolygon(polygon)

    # ------------------------------------------------------------------
    # Zoom and Pan Event Handlers
    # ------------------------------------------------------------------
    def event(self, event):
        """Handle gesture events (pinch, pan)."""
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def gestureEvent(self, event):
        """Handle pinch and pan gestures for Mac trackpad."""
        pinch = event.gesture(Qt.PinchGesture)
        pan = event.gesture(Qt.PanGesture)

        if pinch:
            self._handlePinchGesture(pinch)

        if pan:
            self._handlePanGesture(pan)

        return True

    def _handlePinchGesture(self, gesture):
        """Handle two-finger pinch gesture for zooming."""
        from PyQt5.QtCore import Qt as QtCore

        if gesture.state() == QtCore.GestureStarted:
            self._pinch_scale_factor = 1.0
        elif gesture.state() == QtCore.GestureUpdated or gesture.state() == QtCore.GestureFinished:
            # Get scale change
            scale_factor = gesture.scaleFactor()

            # Update zoom
            old_zoom = self._zoom
            self._zoom = max(0.1, min(10.0, self._zoom * scale_factor))

            # Adjust pan to zoom towards gesture center
            center_point = gesture.centerPoint()
            screen_center_x = self.width() / 2
            screen_center_y = self.height() / 2

            offset_x = center_point.x() - screen_center_x
            offset_y = center_point.y() - screen_center_y

            zoom_change = self._zoom / old_zoom - 1
            self._pan_x -= offset_x * zoom_change
            self._pan_y -= offset_y * zoom_change

            self.update()

    def _handlePanGesture(self, gesture):
        """Handle two-finger pan gesture for panning."""
        from PyQt5.QtCore import Qt as QtCore

        if gesture.state() == QtCore.GestureUpdated or gesture.state() == QtCore.GestureFinished:
            delta = gesture.delta()
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self.update()

    def mousePressEvent(self, event):
        """Start panning only with Command + left button (Mac)."""
        from PyQt5.QtCore import Qt
        # Only ⌘ + left click for pan (backup method)
        if event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier):
            self._is_panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handle panning while mouse button is held."""
        if self._is_panning and self._last_mouse_pos is not None:
            delta = event.pos() - self._last_mouse_pos
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Stop panning."""
        from PyQt5.QtCore import Qt
        if event.button() == Qt.LeftButton and self._is_panning:
            self._is_panning = False
            self._last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event):
        """Reset zoom and pan on double click."""
        from PyQt5.QtCore import Qt
        if event.button() == Qt.LeftButton:
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self.update()