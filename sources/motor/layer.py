import csv
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

# from log import CORELOG
from parameters import LPARAMS
from pattern import Pattern
from calculate import Calculate

from PyQt5.QtCore import Qt, QPointF, QCoreApplication
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QGestureRecognizer, QProgressDialog

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False


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
        preConfig: Dict[str, Any],
        currentConfig: Dict[str, Any],
        nextConfig: Dict[str, Any],
        color: str,
        layer_index: int,
        layer_label: str,
        offset: float,
        layer: str,
        front: bool,
        back: bool,
        index: int,
        count: int
    ) -> Dict[str, Any]:
        """
        Compute a single pattern and return the result.
        This is a static method suitable for parallel execution.
        Front and back are generated independently by Pattern.
        """
        result = {"front": None, "back": None}
        
        if front:
            pattern = Pattern.GetPattern(
                preConfig=preConfig,
                currentConfig=currentConfig,
                nextConfig=nextConfig,
                side="front",
                layer=layer,
                layerIndex=layer_index,
                patternIndex=index,
                patternCount=count
            )
            
            shape = pattern.get("shape")
            if shape is not None and len(shape) > 0:
                shape_offset = shape.copy()
                shape_offset[:, 0] += offset
                
                metrics = {
                    "convexhull_area": pattern.get("convexhull_area", 0.0),
                    "pattern_area": pattern.get("pattern_area", 0.0),
                    "pattern_resistance": pattern.get("pattern_resistance", 0.0),
                }
                
                _uvw_map = {0: "u", 1: "v", 2: "w"}
                result["front"] = {
                    "shape": shape_offset,
                    "color": color,
                    "index": index,
                    "uvw": _uvw_map[(index // 3) % 3],
                    "layer_index": layer_index,
                    "layer_label": layer_label,
                    "location": "normal",
                    "metrics": metrics,
                }

        if back:
            pattern = Pattern.GetPattern(
                preConfig=preConfig,
                currentConfig=currentConfig,
                nextConfig=nextConfig,
                side="back",
                layer=layer,
                layerIndex=layer_index,
                patternIndex=index,
                patternCount=count
            )
            
            shape = pattern.get("shape")
            if shape is not None and len(shape) > 0:
                shape_offset = shape.copy()
                shape_offset[:, 0] += offset
                
                metrics = {
                    "convexhull_area": pattern.get("convexhull_area", 0.0),
                    "pattern_area": pattern.get("pattern_area", 0.0),
                    "pattern_resistance": pattern.get("pattern_resistance", 0.0),
                }
                
                _uvw_map = {0: "u", 1: "v", 2: "w"}
                result["back"] = {
                    "shape": shape_offset,
                    "color": color,
                    "index": index,
                    "uvw": _uvw_map[(index // 3) % 3],
                    "layer_index": layer_index,
                    "layer_label": layer_label,
                    "location": "normal",
                    "metrics": metrics,
                }

        return result

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
                        result["front"]["seq"] = self._front_seq
                        self._front_seq += 1
                        layers["front"].append(result["front"])
                    if result["back"] is not None:
                        result["back"]["seq"] = self._back_seq
                        self._back_seq += 1
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
                    result["front"]["seq"] = self._front_seq
                    self._front_seq += 1
                    layers["front"].append(result["front"])
                if result["back"] is not None:
                    result["back"]["seq"] = self._back_seq
                    self._back_seq += 1
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

            tasks.append({
                "preConfig": None,
                "currentConfig": currentConfig,
                "nextConfig": nextConfig,
                "color": color,
                "layer_index": layer_index,
                "layer_label": layer_label,
                "offset": offset,
                "layer": "begin",
                "front": True,
                "back": True,
                "index": i,
                "count": count
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

            tasks.append({
                "preConfig": preConfig,
                "currentConfig": currentConfig,
                "nextConfig": nextConfig,
                "color": color,
                "layer_index": layer_index,
                "layer_label": layer_label,
                "offset": offset,
                "layer": "mid",
                "front": True,
                "back": True,
                "index": i,
                "count": count
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

            tasks.append({
                "preConfig": preConfig,
                "currentConfig": currentConfig,
                "nextConfig": None,
                "color": color,
                "layer_index": layer_index,
                "layer_label": layer_label,
                "offset": offset,
                "layer": "end",
                "front": True,
                "back": True,
                "index": i,
                "count": count
            })

        # Execute all patterns in parallel
        self._build_patterns_parallel(tasks, layers)

        return start_offset + count * (ppw + psp)  # Return final offset for next layer    def _export_layer_metrics(self, layers: Dict[str, Any], export_filename: str = "layer_metrics.csv"):
        
    def _export_layer_metrics(self, layers: Dict[str, Any], export_filename: str = "layer_metrics.csv"):
        """
        Export convex hull, pattern area, and resistance metrics for selected patterns.
        Samples indices [0, 9, 20, 52, 53] from each layer on both sides.
        Calculates overall average resistance across all patterns.
        Output format: grouped by layer, then front/back within each layer, then overall average.
        """
        if not layers:
            return

        # Target indices to sample
        target_indices = [0, 9, 20, 52, 53]

        # Collect data for front and back separately
        front_rows: List[Dict[str, Any]] = []
        back_rows: List[Dict[str, Any]] = []

        # For calculating overall average resistance
        all_resistances: List[float] = []

        # Process front side
        if "front" in layers and layers["front"]:
            front_rows, front_resistances = self._process_side_metrics(
                layers["front"], "front", target_indices
            )
            all_resistances.extend(front_resistances)

        # Process back side
        if "back" in layers and layers["back"]:
            back_rows, back_resistances = self._process_side_metrics(
                layers["back"], "back", target_indices
            )
            all_resistances.extend(back_resistances)

        # Combine and sort by layer_index, then by side (front before back)
        all_data_rows = front_rows + back_rows
        # Sort by: layer_index (ascending), then side (back before front alphabetically, so reverse)
        all_data_rows.sort(key=lambda x: (x["layer_index"] or 0, x["side"]))

        # Calculate overall averages
        average_rows: List[Dict[str, Any]] = []
        if all_resistances:
            # Collect all convexhull_area and pattern_area values
            all_convexhull_areas = []
            all_pattern_areas = []

            for row in all_data_rows:
                convexhull = row.get("convexhull_area")
                pattern = row.get("pattern_area")
                if convexhull not in ("", None):
                    all_convexhull_areas.append(convexhull)
                if pattern not in ("", None):
                    all_pattern_areas.append(pattern)

            # Calculate averages
            overall_avg_resistance = sum(all_resistances) / len(all_resistances)
            overall_avg_convexhull = sum(all_convexhull_areas) / len(all_convexhull_areas) if all_convexhull_areas else ""
            overall_avg_pattern = sum(all_pattern_areas) / len(all_pattern_areas) if all_pattern_areas else ""

            average_rows.append({
                "side": "overall",
                "layer_index": "",
                "layer_label": "",
                "pattern_index": "AVG",
                "location": "average",
                "sampled_from": f"n={len(all_resistances)}",
                "convexhull_area": overall_avg_convexhull,
                "pattern_area": overall_avg_pattern,
                "pattern_resistance": overall_avg_resistance,
            })

        # Combine all rows: data sorted by layer, then overall average
        all_rows = all_data_rows + average_rows

        if not all_rows:
            return

        export_path = Path(__file__).resolve().parent / export_filename
        fieldnames = [
            "side",
            "layer_index",
            "layer_label",
            "pattern_index",
            "location",
            "sampled_from",
            "convexhull_area",
            "pattern_area",
            "pattern_resistance",
        ]

        with export_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    def _process_side_metrics(
        self,
        entries: List[Dict[str, Any]],
        side: str,
        target_indices: List[int]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Process metrics for one side (front or back).
        Returns a tuple of (rows, all_resistances).
        """
        rows: List[Dict[str, Any]] = []
        all_resistances: List[float] = []

        # Group by layer
        grouped: Dict[Tuple[Optional[int], Optional[str]], List[Dict[str, Any]]] = {}
        for entry in entries:
            key = (entry.get("layer_index"), entry.get("layer_label"))
            grouped.setdefault(key, []).append(entry)

        # Process each layer
        for (layer_idx, layer_label), bucket in grouped.items():
            if not bucket:
                continue

            sorted_bucket = sorted(bucket, key=lambda item: item.get("index", 0))

            # Collect all resistances for overall average calculation
            for entry in sorted_bucket:
                metrics = entry.get("metrics", {})
                resistance = metrics.get("pattern_resistance")
                if resistance is not None:
                    all_resistances.append(resistance)

            # Sample specific indices
            for target_idx in target_indices:
                if target_idx < 0 or target_idx >= len(sorted_bucket):
                    continue

                entry = sorted_bucket[target_idx]
                metrics = entry.get("metrics", {})
                rows.append({
                    "side": side,
                    "layer_index": layer_idx,
                    "layer_label": layer_label,
                    "pattern_index": entry.get("index"),
                    "location": entry.get("location"),
                    "sampled_from": f"index[{target_idx}]",
                    "convexhull_area": metrics.get("convexhull_area"),
                    "pattern_area": metrics.get("pattern_area"),
                    "pattern_resistance": metrics.get("pattern_resistance"),
                })

        return rows, all_resistances

    def _export_layers_to_dxf(self, layers: Dict[str, Any], export_filename: str = "layers_export.dxf"):
        """
        Export all layer polylines to DXF format, grouped by layer.

        Args:
            layers: Dictionary containing 'front' and 'back' layer shapes
            export_filename: Output DXF filename
        """
        if not HAS_EZDXF:
            print("Warning: ezdxf not installed. Skipping DXF export.")
            print("Install with: pip install ezdxf")
            return

        if not layers:
            return

        # Create new DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        # Group front shapes by layer_index
        front_shapes = layers.get("front", [])
        front_by_layer = {}
        for shape_data in front_shapes:
            layer_index = shape_data.get("layer_index", 0)
            layer_label = shape_data.get("layer_label", f"Layer{layer_index}")
            if layer_index not in front_by_layer:
                front_by_layer[layer_index] = {
                    "label": layer_label,
                    "shapes": [],
                    "color": shape_data.get("color", "#de7cfc")
                }
            front_by_layer[layer_index]["shapes"].append(shape_data)

        # Group back shapes by layer_index
        back_shapes = layers.get("back", [])
        back_by_layer = {}
        for shape_data in back_shapes:
            layer_index = shape_data.get("layer_index", 0)
            layer_label = shape_data.get("layer_label", f"Layer{layer_index}")
            if layer_index not in back_by_layer:
                back_by_layer[layer_index] = {
                    "label": layer_label,
                    "shapes": [],
                    "color": shape_data.get("color", "#de7cfc")
                }
            back_by_layer[layer_index]["shapes"].append(shape_data)

        # Export front layers
        for layer_idx in sorted(front_by_layer.keys()):
            layer_info = front_by_layer[layer_idx]
            layer_name = f"front-layer{layer_idx}"
            aci_color = self._hex_to_aci_color(layer_info["color"])

            for shape_data in layer_info["shapes"]:
                shape = shape_data.get("shape")
                if shape is not None and len(shape) > 0:
                    # Create polyline points (x, y) - DXF uses 2D coordinates
                    points = [(pt[0], pt[1]) for pt in shape]

                    # Ensure closed polyline
                    if points[0] != points[-1]:
                        points.append(points[0])

                    # Add polyline to DXF
                    polyline = msp.add_lwpolyline(points)
                    polyline.dxf.layer = layer_name
                    polyline.dxf.color = aci_color

        # Export back layers
        for layer_idx in sorted(back_by_layer.keys()):
            layer_info = back_by_layer[layer_idx]
            layer_name = f"back-layer{layer_idx}"
            aci_color = self._hex_to_aci_color(layer_info["color"])

            for shape_data in layer_info["shapes"]:
                shape = shape_data.get("shape")
                if shape is not None and len(shape) > 0:
                    # Create polyline points (x, y)
                    points = [(pt[0], pt[1]) for pt in shape]

                    # Ensure closed polyline
                    if points[0] != points[-1]:
                        points.append(points[0])

                    # Add polyline to DXF
                    polyline = msp.add_lwpolyline(points)
                    polyline.dxf.layer = layer_name
                    polyline.dxf.color = aci_color

        # Save DXF file
        export_path = Path(__file__).resolve().parent / export_filename
        doc.saveas(export_path)
        print(f"DXF exported to: {export_path}")

    def _hex_to_aci_color(self, hex_color: str) -> int:
        """
        Convert hex color to approximate AutoCAD Color Index (ACI).

        Args:
            hex_color: Hex color string (e.g., "#de7cfc")

        Returns:
            int: ACI color index (1-255)
        """
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')

        # Parse RGB values
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
        except (ValueError, IndexError):
            return 7  # Default white

        # Simple mapping to common ACI colors
        # This is a basic approximation
        if r > 200 and g < 100 and b < 100:
            return 1  # Red
        elif r < 100 and g > 200 and b < 100:
            return 3  # Green
        elif r < 100 and g < 100 and b > 200:
            return 5  # Blue
        elif r > 200 and g > 200 and b < 100:
            return 2  # Yellow
        elif r > 200 and g < 100 and b > 200:
            return 6  # Magenta
        elif r < 100 and g > 200 and b > 200:
            return 4  # Cyan
        elif r > 200 and g > 200 and b > 200:
            return 7  # White
        else:
            return 8  # Gray

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
        self._front_seq = 0
        self._back_seq = 0
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

        self._export_layer_metrics(layers)
        self._export_layers_to_dxf(layers)
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
