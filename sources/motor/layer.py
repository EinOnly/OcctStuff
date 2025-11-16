import numpy as np
from typing import Dict, Any, Iterable, List, Tuple, Optional

# from log import CORELOG
from parameters import LPARAMS
from pattern import Pattern
from calculate import Calculate

from PyQt5.QtCore import Qt, QPointF, QCoreApplication
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QGestureRecognizer, QProgressDialog


class Layers(QWidget):
    # Multi-threading disabled by default due to Python GIL limitations
    USE_MULTITHREADING = True

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._params = LPARAMS
        self._layerConfig = {}
        self._progress_dialog = None
        self._progress_current = 0
        self._progress_total = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = LayerCanvas()
        layout.addWidget(self.canvas)

        self._needs_regeneration = True
        self.register()
        # CORELOG.info(f"Layers initialized.")

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

    def generate_layers(self):
        """Public entry for manual generation triggered by UI button."""
        self._refresh_layers()

    def _refresh_layers(self):
        # import pprint
        # pprint.pprint(self._params.snapshot())
        self.canvas.setLayers(self.getLayers())
        self._needs_regeneration = False

    # ------------------------------------------------------------------
    # Signal callbacks
    # ------------------------------------------------------------------
    def _on_param_changed(self, key: str, value: Any):
        self._needs_regeneration = True

    def _on_bulk_changed(self, payload: Dict[str, Any]):
        self._needs_regeneration = True

    @staticmethod
    def _compute_pattern(
        currentParams: Dict[str, Any],
        nextParams: Dict[str, Any],
        color: str,
        layer_index: int,
        layer_label: str,
        offset: float,
        location: str,
        front: bool,
        back: bool,
        index: int,
        mirror: bool
    ) -> Dict[str, Any]:
        """
        Compute a single pattern and return the result.
        This is a static method suitable for parallel execution.
        """
        pattern = Pattern.GetPattern(currentParams, nextParams, location)
        metrics = {
            "convexhull_area": pattern.get("convexhull_area", 0.0),
            "pattern_area": pattern.get("pattern_area", 0.0),
            "pattern_resistance": pattern.get("pattern_resistance", 0.0),
        }

        result = {"front": None, "back": None}
        shape = pattern.get("shape")

        if shape is not None and len(shape) > 0:
            if front:
                shape_offset = shape.copy()
                shape_offset[:, 0] += offset
                result["front"] = {
                    "shape": shape_offset,
                    "color": color,
                    "index": index,
                    "layer_index": layer_index,
                    "layer_label": layer_label,
                    "location": location,
                    "metrics": metrics,
                }

            if back:
                shape_back = shape.copy()
                mirror_x = currentParams.get("pattern_ppw", 0)/2 if mirror else None
                shape_back = Calculate.Mirror(shape_back, mirror_x, currentParams.get("pattern_pbh", 0)/2)
                shape_back[:, 0] += offset
                result["back"] = {
                    "shape": shape_back,
                    "color": color,
                    "index": index,
                    "layer_index": layer_index,
                    "layer_label": layer_label,
                    "location": location,
                    "metrics": metrics,
                }

        return result

    def _buildPattern(self,
        layers: Dict[str, Any],
        currentParams: Dict[str, Any],
        nextParams: Dict[str, Any],
        color: str,
        layer_index: int,
        layer_label: str,
        offset: float = 0.0,
        location: str = "normal",
        front: bool = True,
        back: bool = True,
        index: int = 0,
        mirror: bool = False
    ):
        pattern = Pattern.GetPattern(currentParams, nextParams, location)
        metrics = {
            "convexhull_area": pattern.get("convexhull_area", 0.0),
            "pattern_area": pattern.get("pattern_area", 0.0),
            "pattern_resistance": pattern.get("pattern_resistance", 0.0),
        }

        # Get shape and apply horizontal offset
        shape = pattern.get("shape")
        if shape is not None and len(shape) > 0:
            if front:
                shape_offset = shape.copy()
                shape_offset[:, 0] += offset

                layers["front"].append({
                    "shape": shape_offset,
                    "color": color,
                    "index": index,
                    "layer_index": layer_index,
                    "layer_label": layer_label,
                    "location": location,
                    "metrics": metrics,
                })

            if back:
                shape_back = shape.copy()
                # flip this pattern here
                mirror_x = currentParams.get("pattern_ppw", 0)/2 if mirror else None
                shape_back = Calculate.Mirror(shape_back, mirror_x, currentParams.get("pattern_pbh", 0)/2)

                shape_back[:, 0] += offset
                layers["back"].append({
                    "shape": shape_back,
                    "color": color,
                    "index": index,
                    "layer_index": layer_index,
                    "layer_label": layer_label,
                    "location": location,
                    "metrics": metrics,
                })

        # Update progress if dialog exists
        if self._progress_dialog is not None:
            self._progress_current += 1
            self._progress_dialog.setValue(self._progress_current)
            QCoreApplication.processEvents()  # Allow UI to update

    def _build_patterns_parallel(self, tasks: List[Dict[str, Any]], layers: Dict[str, Any]):
        """
        Execute pattern computation tasks (optionally in parallel) and collect results.

        Args:
            tasks: List of task dictionaries containing all required parameters
            layers: Dictionary to append results to
        """
        if not tasks:
            return

        if self.USE_MULTITHREADING:
            # Multi-threaded execution (may be slower due to GIL)
            from concurrent.futures import ThreadPoolExecutor
            max_workers = min(2, len(tasks))  # Reduced to 2 workers

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(Layers._compute_pattern, **task) for task in tasks]

                for idx, future in enumerate(futures):
                    result = future.result()

                    if result["front"] is not None:
                        layers["front"].append(result["front"])
                    if result["back"] is not None:
                        layers["back"].append(result["back"])

                    # Batch progress updates (every 5 patterns)
                    if self._progress_dialog is not None:
                        self._progress_current += 1
                        if idx % 5 == 0 or idx == len(futures) - 1:
                            self._progress_dialog.setValue(self._progress_current)
                            QCoreApplication.processEvents()
        else:
            # Single-threaded execution (faster for most cases)
            for idx, task in enumerate(tasks):
                result = Layers._compute_pattern(**task)

                if result["front"] is not None:
                    layers["front"].append(result["front"])
                if result["back"] is not None:
                    layers["back"].append(result["back"])

                # Update progress every 5 patterns to reduce overhead
                if self._progress_dialog is not None:
                    self._progress_current += 1
                    if idx % 5 == 0 or idx == len(tasks) - 1:
                        self._progress_dialog.setValue(self._progress_current)
                        QCoreApplication.processEvents()

    def _buildStart(self, layers: Dict[str, Any], currentConfig: Dict[str, Any], nextConfig: Dict[str, Any], layer_index: int, layer_label: str, start_offset: float = 0.0):
        """Build start layer: all patterns use same config except last one transitions to next layer."""
        layer_params = currentConfig.get("layer", {})
        count = layer_params.get("layer_pdc", 9)

        ppw = layer_params.get("layer_ppw", 0.5)
        psp = layer_params.get("layer_psp", 0.05)
        color = layer_params.get("color", "#de7cfc")

        # Prepare all tasks for parallel execution
        tasks = []
        for i in range(count):
            offset = start_offset + i * (ppw + psp)

            # Reset back to default for each pattern
            back = True
            front = True
            location = "normal"
            currentParams = None
            nextParams = None

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
                back = True
                if i == 8:
                    back = False

            tasks.append({
                "currentParams": currentParams,
                "nextParams": nextParams,
                "color": color,
                "layer_index": layer_index,
                "layer_label": layer_label,
                "offset": offset,
                "location": location,
                "front": front,
                "back": back,
                "index": i,
                "mirror": False
            })

        # Execute all patterns in parallel
        self._build_patterns_parallel(tasks, layers)

        return start_offset + count * (ppw + psp)  # Return final offset for next layer

    def _buildNormal(self, layers: Dict[str, Any], preConfig: Dict[str, Any], currentConfig: Dict[str, Any], nextConfig: Dict[str, Any], layer_index: int, layer_label: str, start_offset: float = 0.0):
        """Build normal layer: all patterns transition to next layer."""
        layer_params = currentConfig.get("layer", {})
        count = layer_params.get("layer_pdc", 9)

        ppw = layer_params.get("layer_ppw", 0.5)
        psp = layer_params.get("layer_psp", 0.05)
        color = layer_params.get("color", "#de7cfc")

        # Prepare all tasks for parallel execution
        tasks = []
        for i in range(count):
            offset = start_offset + i * (ppw + psp)

            # Reset back to default for each pattern
            back = True
            front = True
            location = "normal"
            currentParams = None
            nextParams = None

            # Handled the last pattern of each layer separately
            if i == 0 and preConfig is not None:
                # First pattern transitions from previous layer
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

            tasks.append({
                "currentParams": currentParams,
                "nextParams": nextParams,
                "color": color,
                "layer_index": layer_index,
                "layer_label": layer_label,
                "offset": offset,
                "location": location,
                "front": front,
                "back": back,
                "index": i,
                "mirror": False
            })

        # Execute all patterns in parallel
        self._build_patterns_parallel(tasks, layers)

        return start_offset + count * (ppw + psp)  # Return final offset for next layer

    def _buildEnd(self, layers: Dict[str, Any], preConfig: Dict[str, Any], currentConfig: Dict[str, Any], layer_index: int, layer_label: str, start_offset: float = 0.0):
        """Build end layer: all patterns use same config (no transition)."""
        layer_params = currentConfig.get("layer", {})
        count = layer_params.get("layer_pdc", 9)

        ppw = layer_params.get("layer_ppw", 0.5)
        psp = layer_params.get("layer_psp", 0.05)
        color = layer_params.get("color", "#de7cfc")

        # Prepare all tasks for parallel execution
        tasks = []
        for i in range(count):
            offset = start_offset + i * (ppw + psp)

            # Reset back to default for each pattern
            back = True
            front = True
            mirror = False
            location = "normal"
            currentParams = None
            nextParams = None

            # Handled the last pattern of each layer separately
            if i == 0 and preConfig is not None:
                # First pattern transitions from previous layer
                currentParams = currentConfig.get("layer", {}).copy()
                nextParams = preConfig.get("layer", {}).copy()
                location = "start"
            elif i > count - 10:
                currentParams = currentConfig.get("layer", {}).copy()
                currentParams["pattern_twist"] = False
                nextParams = currentParams
                front = False
                mirror = True
            else:
                # Regular patterns (current -> current)
                currentParams = currentConfig.get("layer", {})
                nextParams = currentParams

            tasks.append({
                "currentParams": currentParams,
                "nextParams": nextParams,
                "color": color,
                "layer_index": layer_index,
                "layer_label": layer_label,
                "offset": offset,
                "location": location,
                "front": front,
                "back": back,
                "index": i,
                "mirror": mirror
            })

        # Execute all patterns in parallel
        self._build_patterns_parallel(tasks, layers)

        return start_offset + count * (ppw + psp)  # Return final offset for next layer

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

        # Calculate total number of patterns for progress tracking
        total_patterns = 0
        for config in configs:
            layer_params = config.get("layer", {})
            total_patterns += layer_params.get("layer_pdc", 9)

        # Create progress dialog
        if total_patterns > 0:
            self._progress_dialog = QProgressDialog(
                "Generating layers...",
                "Cancel",
                0,
                total_patterns,
                self
            )
            self._progress_dialog.setWindowModality(Qt.WindowModal)
            self._progress_dialog.setMinimumDuration(0)  # Show immediately
            self._progress_current = 0
            self._progress_total = total_patterns

        cumulative_offset = 0.0

        try:
            for idx, currentLayerConfig in enumerate(configs):
                # Check if user cancelled
                if self._progress_dialog and self._progress_dialog.wasCanceled():
                    break

                layer_type = currentLayerConfig.get("type", "normal")
                layer_index = currentLayerConfig.get("index", idx)
                layer_label = currentLayerConfig.get("label") or f"Layer {layer_index + 1} ({layer_type})"
                nexLayerConfig = configs[idx + 1] if idx + 1 < len(configs) else None
                preLayerConfig = configs[idx - 1] if idx - 1 >= 0 else None

                # Update progress label
                if self._progress_dialog:
                    self._progress_dialog.setLabelText(f"Generating {layer_label}...")

                match layer_type:
                    case "start":
                        # CORELOG.info(f"Building start layer at index {idx}")
                        cumulative_offset = self._buildStart(
                            layers,
                            currentLayerConfig,
                            nexLayerConfig,
                            layer_index,
                            layer_label,
                            cumulative_offset
                        )
                        # CORELOG.info(f"Built layer at index {idx} successfully, offset: {cumulative_offset}")
                    case "normal":
                        # CORELOG.info(f"Building normal layer at index {idx}")
                        cumulative_offset = self._buildNormal(
                            layers,
                            preLayerConfig,
                            currentLayerConfig,
                            nexLayerConfig,
                            layer_index,
                            layer_label,
                            cumulative_offset
                        )
                        # CORELOG.info(f"Built layer at index {idx} successfully, offset: {cumulative_offset}")
                    case "end":
                        # CORELOG.info(f"Building end layer at index {idx}")
                        cumulative_offset = self._buildEnd(
                            layers,
                            preLayerConfig,
                            currentLayerConfig,
                            layer_index,
                            layer_label,
                            cumulative_offset
                        )
                        # CORELOG.info(f"Built layer at index {idx} successfully, offset: {cumulative_offset}")
                    case _:
                        # CORELOG.warn(f"Unknown layer type: {layer_type} at index {idx}")
                        continue
        finally:
            # Clean up progress dialog
            if self._progress_dialog:
                self._progress_dialog.close()
                self._progress_dialog = None

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

        if not self._layers:
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "Click 'Generate Layers' to render.")
            return

        # Collect all points from both front and back for bounds calculation
        all_points: List[Tuple[float, float]] = []
        for layer_name in ["front", "back"]:
            layer_shapes = self._layers.get(layer_name, [])
            for shape_data in layer_shapes:
                shape = shape_data.get("shape")
                if shape is not None and len(shape) > 0:
                    all_points.extend([(pt[0], pt[1]) for pt in shape])

        if not all_points:
            painter.setPen(Qt.gray)
            painter.drawText(self.rect(), Qt.AlignCenter, "No layer geometry generated yet.")
            return

        bounds = self._compute_bounds(all_points)
        if bounds is None:
            return

        mapper = self._build_mapper(bounds)

        # Draw back layer first (with higher transparency)
        self._draw_layer(painter, mapper, "back", alpha=50)

        # Draw front layer on top (with normal transparency)
        self._draw_layer(painter, mapper, "front", alpha=100)

    def _draw_layer(self, painter: QPainter, mapper, layer_name: str, alpha: int = 100):
        """
        Draw a specific layer (front or back) with specified alpha transparency.

        Args:
            painter: QPainter instance
            mapper: Coordinate mapping function
            layer_name: Name of the layer ("front" or "back")
            alpha: Alpha transparency value (0-255, default 100)
        """
        layer_shapes = self._layers.get(layer_name, [])
        if not layer_shapes:
            return

        for shape_data in layer_shapes:
            shape = shape_data.get("shape")
            color = shape_data.get("color", "#de7cfc")
            if shape is not None and len(shape) > 0:
                points = [(pt[0], pt[1]) for pt in shape]
                self._draw_shape(painter, mapper, points, color, alpha)

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

    def _draw_shape(self, painter: QPainter, mapper, points: List[Tuple[float, float]], color_str: str, alpha: int = 100):
        """
        Draw a single shape with semi-transparent fill.

        Args:
            painter: QPainter instance
            mapper: Coordinate mapping function
            points: List of (x, y) points
            color_str: Color string (hex or named color)
            alpha: Alpha transparency value (0-255, default 100)
        """
        if len(points) < 3:
            return

        polygon = QPolygonF([mapper(pt) for pt in points])

        # Parse color string and apply alpha
        color = QColor(color_str)
        fill = QBrush(QColor(color.red(), color.green(), color.blue(), alpha))
        outline = QPen(QColor(color.red(), color.green(), color.blue(), min(255, alpha * 2)), 1.5)
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
        from PyQt5.QtCore import Qt

        if self._is_panning:
            # Safety check: verify Command key is still held
            if not (event.modifiers() & Qt.ControlModifier):
                # Command key released during panning - stop immediately
                self._stopPanning()
                return

            # Safety check: verify left button is still pressed
            if not (event.buttons() & Qt.LeftButton):
                # Left button released - stop immediately
                self._stopPanning()
                return

            if self._last_mouse_pos is not None:
                delta = event.pos() - self._last_mouse_pos
                self._pan_x += delta.x()
                self._pan_y += delta.y()
                self._last_mouse_pos = event.pos()
                self.update()

    def mouseReleaseEvent(self, event):
        """Stop panning."""
        from PyQt5.QtCore import Qt
        if event.button() == Qt.LeftButton:
            self._stopPanning()

    def keyPressEvent(self, event):
        """Handle key press events."""
        from PyQt5.QtCore import Qt
        if event.key() == Qt.Key_Escape:
            # ESC key cancels panning
            self._stopPanning()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        """Stop panning when focus is lost."""
        self._stopPanning()
        super().focusOutEvent(event)

    def leaveEvent(self, event):
        """Stop panning when mouse leaves the widget."""
        self._stopPanning()
        super().leaveEvent(event)

    def enterEvent(self, event):
        """Ensure clean state when mouse enters the widget."""
        # Force stop panning if somehow still active
        if self._is_panning:
            self._stopPanning()
        super().enterEvent(event)

    def _stopPanning(self):
        """Stop panning and reset state - force cleanup even if state is inconsistent."""
        # Force reset all state, regardless of current _is_panning value
        was_panning = self._is_panning
        self._is_panning = False
        self._last_mouse_pos = None

        # Always reset cursor to ensure it's not stuck
        self.setCursor(Qt.ArrowCursor)

        # Only update if we were actually panning to avoid unnecessary redraws
        if was_panning:
            self.update()

    def mouseDoubleClickEvent(self, event):
        """Reset zoom and pan on double click."""
        from PyQt5.QtCore import Qt
        if event.button() == Qt.LeftButton:
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self.update()
