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
        params = self.read()
        # For preview, use a simple mid-layer pattern
        config = {"layer": params}
        self.canvas.setPattern(Pattern.GetPattern(
            preConfig=None,
            currentConfig=config,
            nextConfig=None,
            side="front",
            layer="mid",
            layerIndex=0,
            patternIndex=0,
            patternCount=9
        ))

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
        height = assist.get("height", 0.0)

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
            result = curve_arr[::-1]  # Reverse
            
            # The superellipse curve ends at (0, height - radius).
            # We need to extend it down to (0, height/2) with a vertical line segment
            # to ensure the middle section is perfectly parallel.
            mid_y = height / 2.0
            curve_end_y = result[-1][1] if len(result) > 0 else mid_y
            
            if len(result) > 0 and curve_end_y > mid_y:
                # Add a vertical line segment from curve end to (0, height/2)
                # Ensure the last point of superellipse is at x=0
                result[-1][0] = 0.0
                # Add the endpoint at (0, height/2)
                endpoint = np.array([[0.0, mid_y]], dtype=np.float64)
                result = np.vstack([result, endpoint])
            elif len(result) > 0:
                # Curve already reaches below mid_y, just ensure x=0
                result[-1] = [0.0, mid_y]
            
            return result

        # Fallback
        return top.copy()

    @staticmethod
    def _buildTopInner(assist: Dict[str, Any], pwidth: float) -> np.ndarray:
        """Build top portion of outer curve with tcc boundary clamping."""
        base = Pattern._buildTopOuter(assist)

        # Check if we have enough points for offset operation
        if len(base) < 2:
            return np.array([], dtype=np.float64).reshape(0, 2)

        spacing = assist.get("spacing", 0.0)
        shift = pwidth + spacing
        clamp = assist.get("max_tx", 0.0) + spacing
        base[:, 0] += shift

        # Remove duplicate consecutive points to avoid zero-length segments
        if len(base) > 1:
            diffs = np.linalg.norm(base[1:] - base[:-1], axis=1)
            valid_mask = np.concatenate(([True], diffs > 1e-9))
            base = base[valid_mask]

        # Check again after removing duplicates
        if len(base) < 2:
            return np.array([], dtype=np.float64).reshape(0, 2)

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
        height = assist.get("height", 0.0)

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
            result = curve_arr[::-1][1:]  # Reverse and skip first (originally last)
            
            # The superellipse curve starts at (0, radius).
            # We need to extend it up from (0, height/2) with a vertical line segment
            # to ensure the middle section is perfectly parallel.
            mid_y = height / 2.0
            curve_start_y = result[0][1] if len(result) > 0 else mid_y
            
            if len(result) > 0 and curve_start_y < mid_y:
                # Add a vertical line segment from (0, height/2) to curve start
                # Ensure the first point of superellipse is at x=0
                result[0][0] = 0.0
                # Add the start point at (0, height/2)
                startpoint = np.array([[0.0, mid_y]], dtype=np.float64)
                result = np.vstack([startpoint, result])
            elif len(result) > 0:
                # Curve already starts above mid_y, just ensure x=0
                result[0] = [0.0, mid_y]
            
            return result

        # Fallback
        return bottom.copy()

    @staticmethod
    def _buildBottomInner(assist: Dict[str, Any], pwidth: float) -> np.ndarray:
        """Build top portion of outer curve with tcc boundary clamping."""
        base = Pattern._buildBottomOuter(assist)

        # Check if we have enough points for offset operation
        if len(base) < 2:
            return np.array([], dtype=np.float64).reshape(0, 2)

        spacing = assist.get("spacing", 0.0)
        shift = pwidth + spacing
        clamp = assist.get("max_bx", 0.0) + spacing
        base[:, 0] += shift

        # Remove duplicate consecutive points to avoid zero-length segments
        if len(base) > 1:
            diffs = np.linalg.norm(base[1:] - base[:-1], axis=1)
            valid_mask = np.concatenate(([True], diffs > 1e-9))
            base = base[valid_mask]

        # Check again after removing duplicates
        if len(base) < 2:
            return np.array([], dtype=np.float64).reshape(0, 2)

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
    def _buildConvexHull(top_outer: np.ndarray, bottom_outer: np.ndarray, width: float, flip_y: bool = False) -> np.ndarray:
        """
        Build convex hull by connecting top/bottom outer curves and mirroring across center line.

        Creates a symmetric closed polygon by:
        1. Taking outer curves from top and bottom
        2. Clipping at center line (width/2)
        3. Mirroring across center line to create symmetric shape
        4. Optionally flipping y-axis for the right side

        Args:
            top_outer: Top outer curve points
            bottom_outer: Bottom outer curve points
            width: Pattern width for center line calculation
            flip_y: If True, also flip y-axis when mirroring (for non-twist patterns)

        Returns:
            numpy array of convex hull points forming a closed polygon
        """

        if len(top_outer) == 0 or len(bottom_outer) == 0:
            return np.array([], dtype=np.float64).reshape(0, 2)

        center_x = width / 2.0

        # Clip top outer at center line (keep points where x <= center_x)
        top_clipped = []
        for i in range(len(top_outer)):
            pt = top_outer[i]
            if pt[0] <= center_x:
                top_clipped.append(pt)
            elif i > 0:
                # Check if we're transitioning from left to right of center line
                prev_pt = top_outer[i - 1]
                if prev_pt[0] <= center_x:
                    # Linear interpolation to find intersection with center line
                    t = (center_x - prev_pt[0]) / (pt[0] - prev_pt[0])
                    y_intersect = prev_pt[1] + t * (pt[1] - prev_pt[1])
                    top_clipped.append([center_x, y_intersect])
                    break

        # Clip bottom outer at center line (keep points where x <= center_x)
        bottom_clipped = []
        for i in range(len(bottom_outer)):
            pt = bottom_outer[i]
            if pt[0] <= center_x:
                bottom_clipped.append(pt)
            elif i > 0:
                # Check if we're transitioning from left to right of center line
                prev_pt = bottom_outer[i - 1]
                if prev_pt[0] <= center_x:
                    # Linear interpolation to find intersection with center line
                    t = (center_x - prev_pt[0]) / (pt[0] - prev_pt[0])
                    y_intersect = prev_pt[1] + t * (pt[1] - prev_pt[1])
                    bottom_clipped.append([center_x, y_intersect])
                    break

        if not top_clipped or not bottom_clipped:
            return np.array([], dtype=np.float64).reshape(0, 2)

        top_clipped = np.array(top_clipped, dtype=np.float64)
        bottom_clipped = np.array(bottom_clipped, dtype=np.float64)

        if flip_y:
            # For non-twist patterns: mirror x, then swap top and bottom
            # This creates a vertical flip effect on the right side
            top_mirrored = Calculate.Mirror(bottom_clipped.copy(), axis_x=center_x)
            bottom_mirrored = Calculate.Mirror(top_clipped.copy(), axis_x=center_x)
        else:
            # For twist patterns: simple x-axis mirror
            top_mirrored = Calculate.Mirror(top_clipped.copy(), axis_x=center_x)
            bottom_mirrored = Calculate.Mirror(bottom_clipped.copy(), axis_x=center_x)

        # Build closed convex hull: top_left -> bottom_left -> bottom_right -> top_right
        convex_hull = np.vstack([
            top_clipped,           # Left side top
            bottom_clipped,  # Left side bottom (reversed)
            bottom_mirrored[::-1],       # Right side bottom
            top_mirrored[::-1],    # Right side top (reversed)
        ])

        return convex_hull

    @staticmethod
    def _buildShape(currentAssist: Dict[str, Any], nextAssist: Dict[str, Any], location:str = "normal") -> tuple:
        """
        Build closed shape by combining outer and inner curves.

        Steps:
        1. Get outer curve from current pattern
        2. Get inner curve from next pattern (with offset)
        3. Translate inner curve by pattern_ppw in X direction
        4. If pattern_twist is True, mirror bottom portion of inner curve
        5. Combine into closed polygon in counter-clockwise order

        Args:
            currentAssist: Current pattern assist data
            nextAssist: Next pattern assist data
            location: Pattern location ("start", "end", or "normal")

        Returns:
            tuple: (curve, outer_end_idx, top, bottom)
                - curve: Complete closed curve [top_outer, bottom_outer, bottom_inner, top_inner]
                - outer_end_idx: End index (exclusive) of outer conductor path
                - top: Tuple of (top_outer, top_inner) curves
                - bottom: Tuple of (bottom_outer, bottom_inner) curves
        """
        if currentAssist.get("twist", False):
            if location == "start":
                top = Pattern._buildTop(currentAssist, nextAssist)
                bottom = Pattern._buildBottom(currentAssist, currentAssist)
            elif location == "end":
                top = Pattern._buildTop(currentAssist, currentAssist)
                bottom = Pattern._buildBottom(currentAssist, nextAssist)
            else:
                top = Pattern._buildTop(currentAssist, nextAssist)
                bottom = Pattern._buildBottom(currentAssist, nextAssist)
        else:
            if location == "start":
                top = Pattern._buildTop(currentAssist, currentAssist)
                bottom = Pattern._buildBottom(currentAssist, currentAssist)
            elif location == "end":
                top = Pattern._buildTop(currentAssist, nextAssist)
                bottom = Pattern._buildBottom(currentAssist, nextAssist)
            else:
                top = Pattern._buildTop(currentAssist, nextAssist)
                bottom = Pattern._buildBottom(currentAssist, nextAssist)
        # Calculate index where outer path ends
        outer_end_idx = len(top[0]) + len(bottom[0])

        curve = np.vstack([
            top[0],      # top_outer: start to outer_end_idx
            bottom[0],   # bottom_outer
            bottom[1],   # bottom_inner: from outer_end_idx onwards
            top[1],      # top_inner
        ])
        
        # Remove consecutive duplicate points to avoid zero-length edges
        # which cause BRepBuilderAPI_MakeEdge to fail
        if len(curve) > 1:
            diffs = np.linalg.norm(curve[1:] - curve[:-1], axis=1)
            valid_mask = np.concatenate(([True], diffs > 1e-9))
            curve = curve[valid_mask]
            # Recalculate outer_end_idx based on removed duplicates
            # Count valid points in top[0] and bottom[0]
            top_outer_len = len(top[0])
            bottom_outer_len = len(bottom[0])
            # Check how many duplicates were removed before outer_end_idx
            original_outer_end = top_outer_len + bottom_outer_len
            removed_before_outer = np.sum(~valid_mask[:original_outer_end])
            outer_end_idx = original_outer_end - removed_before_outer
        
        return curve, outer_end_idx, top, bottom

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
                [width/2, height],  # top: tcc at y=pbh
                [width/2, 0.0],  # bottom: bcc at y=0
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
    def _empty_pattern() -> Dict[str, Any]:
        """Return an empty pattern (used when a side shouldn't be rendered)."""
        return {
            "bbox": np.array([], dtype=np.float64).reshape(0, 2),
            "assist": {},
            "shape": np.array([], dtype=np.float64).reshape(0, 2),
            "convexhull": np.array([], dtype=np.float64).reshape(0, 2),
            "convexhull_area": 0.0,
            "pattern_area": 0.0,
            "pattern_resistance": 0.0,
        }

    @staticmethod
    def GetPattern(
        preConfig: Dict[str, Any],
        currentConfig: Dict[str, Any],
        nextConfig: Dict[str, Any],
        side: str = "front",
        layer: str = "mid",
        layerIndex: int = 0,
        patternIndex: int = 0,
        patternCount: int = 9
    ) -> Dict[str, Any]:
        """
        Build a pattern based on layer context.
        
        Args:
            preConfig: Configuration of previous layer (can be None)
            currentConfig: Configuration of current layer
            nextConfig: Configuration of next layer (can be None)
            side: "front" or "back"
            layer: "begin", "mid", or "end"
            layerIndex: Index of current layer
            patternIndex: Index of pattern within layer (0-based)
            patternCount: Total number of patterns in this layer
            
        Returns:
            Dictionary with bbox, assist, shape, convexhull, and metrics
        """
        if currentConfig is None:
            raise ValueError("currentConfig is required")

        # Extract layer parameters
        currentParams = currentConfig.get("layer", {})
        preParams = preConfig.get("layer", {}) if preConfig else None
        nextParams = nextConfig.get("layer", {}) if nextConfig else None
        
        # Determine location and which params to use based on context
        location = "normal"
        params_current = None
        params_next = None
        twist = currentParams["pattern_twist"]

        if  twist:
            if layer == "begin":
                # Start layer logic
                if patternIndex == patternCount - 1 and nextParams is not None:
                    # Last pattern transitions to next layer
                    params_current = currentParams.copy()
                    params_next = nextParams.copy()
                    location = "end"
                elif patternIndex < 8:
                    # First 8 patterns use modified current params
                    params_current = currentParams.copy()
                    params_current["pattern_twist"] = False
                    params_current["pattern_tp1"] += params_current["pattern_ppw"] + params_current["pattern_psp"]
                    params_next = params_current
                    # Don't render back side for these patterns
                    if side == "back":
                        return Pattern._empty_pattern()
                else:
                    # Regular patterns (current -> current)
                    params_current = currentParams
                    params_next = currentParams
                    # Pattern 8 doesn't render back side
                    if patternIndex == 8 and side == "back":
                        return Pattern._empty_pattern()

            elif layer == "mid":
                # Normal layer logic
                if patternIndex == 0 and preParams is not None:
                    # First pattern transitions from previous layer
                    params_current = currentParams.copy()
                    params_next = preParams.copy()
                    location = "start"
                    
                    # When twist status changes at transition, adjust params_next for back side
                    if side == "back":
                        current_twist = params_current.get("pattern_twist", True)
                        pre_twist = params_next.get("pattern_twist", True)
                        
                        # If twist status differs, back side should use current params
                        # to maintain consistency with front side's bottom half
                        if current_twist != pre_twist:
                            if not current_twist:
                                # Current is not twisted, use current for both
                                params_next = params_current.copy()
                            
                elif patternIndex == patternCount - 1 and nextParams is not None:
                    # Last pattern transitions to next layer
                    params_current = currentParams.copy()
                    params_next = nextParams.copy()
                    location = "end"
                    
                    # When twist status changes at transition, adjust params_next for back side
                    if side == "back":
                        current_twist = params_current.get("pattern_twist", True)
                        next_twist = params_next.get("pattern_twist", True)
                        
                        # If twist status differs, back side should use current params
                        # to maintain consistency with front side's bottom half
                        if current_twist != next_twist:
                            if not current_twist:
                                # Current is not twisted, use current for both
                                params_next = params_current.copy()
                            
                else:
                    # Regular patterns (current -> current)
                    params_current = currentParams.copy()
                    params_next = params_current

            elif layer == "end":
                # End layer logic
                if patternIndex == 0 and preParams is not None:
                    # First pattern transitions from previous layer
                    params_current = currentParams.copy()
                    params_next = preParams.copy()
                    location = "start"
                elif patternIndex > patternCount - 10:
                    # Last 9 patterns use modified current params
                    params_current = currentParams.copy()
                    params_current["pattern_twist"] = False
                    params_next = params_current
                    # Don't render front side for these patterns
                    if side == "front":
                        return Pattern._empty_pattern()
                else:
                    # Regular patterns (current -> current)
                    params_current = currentParams
                    params_next = currentParams
        else:
            if layer == "begin":
                # Start layer logic
                if patternIndex == patternCount - 1 and nextParams is not None:
                    # Last pattern transitions to next layer
                    params_current = currentParams.copy()
                    params_next = nextParams.copy()
                    location = "end"
                elif patternIndex < 8:
                    # First 8 patterns use modified current params
                    params_current = currentParams.copy()
                    params_current["pattern_twist"] = False
                    params_current["pattern_tp1"] += params_current["pattern_ppw"] + params_current["pattern_psp"]
                    params_next = params_current
                    # Don't render back side for these patterns
                    if side == "back":
                        return Pattern._empty_pattern()
                else:
                    # Regular patterns (current -> current)
                    params_current = currentParams
                    params_next = currentParams
                    # Pattern 8 doesn't render back side
                    if patternIndex == 8 and side == "back":
                        return Pattern._empty_pattern()

            elif layer == "mid":
                # Normal layer logic
                if patternIndex == 0 and preParams is not None:
                    # First pattern transitions from previous layer
                    params_current = currentParams.copy()
                    params_next = currentParams.copy()
                    location = "start"
                    # When back and not twisted, this should align with previous layer's last pattern
                    if side == "back":
                        # For back side: use current->pre to align with previous layer's end
                        params_current = currentParams.copy()
                        params_next = preParams.copy()
                elif patternIndex == patternCount - 1 and nextParams is not None:
                    # Last pattern transitions to next layer
                    params_current = currentParams.copy()
                    params_next = nextParams.copy()
                    location = "end"
                    # When back and not twisted, this should align with next layer's first pattern
                    if side == "back":
                        # For back side: use next->current to align with next layer's start
                        params_current = nextParams.copy()
                        params_next = currentParams.copy()
                else:
                    # Regular patterns (current -> current)
                    params_current = currentParams.copy()
                    params_next = params_current

            elif layer == "end":
                # End layer logic
                if patternIndex == 0 and preParams is not None:
                    # First pattern transitions from previous layer
                    params_current = currentParams.copy()
                    params_next = preParams.copy()
                    location = "start"
                elif patternIndex > patternCount - 10:
                    # Last 9 patterns use modified current params
                    params_current = currentParams.copy()
                    params_current["pattern_twist"] = False
                    params_next = params_current
                    # Don't render front side for these patterns
                    if side == "front":
                        return Pattern._empty_pattern()
                else:
                    # Regular patterns (current -> current)
                    params_current = currentParams
                    params_next = currentParams

        if params_current is None or params_next is None:
            raise ValueError("Failed to determine pattern parameters")

        current_assist = Pattern._buildAssist(params_current)
        next_assist = Pattern._buildAssist(params_next)
        shape, outer_end_idx, top, bottom = Pattern._buildShape(current_assist, next_assist, location)
        
        # Apply mirroring for back side
        if side == "back":
            twist = params_current.get("pattern_twist", True)
            mirror_y = params_current.get("pattern_pbh", 0) / 2
            
            if twist:
                # When twisted: mirror only along Y axis (no X mirror)
                # This creates the twisted effect
                mirror_x = None
            else:
                # When not twisted: mirror along both X and Y axes
                # This keeps back side aligned with front side's bottom half
                mirror_x = params_current.get("pattern_ppw", 0) / 2
            
            shape = Calculate.Mirror(shape, mirror_x, mirror_y)
        
        # convexhull = Pattern._buildConvexHull(top[0], bottom[0], current_assist.get("width", 0.0))
        if params_current.get("pattern_twist", False):
            convexhull = Pattern._buildConvexHull(
                Calculate.Mirror(bottom[0], None, current_assist.get("height", 0.0)/2)[::-1], 
                bottom[0], 
                current_assist.get("width", 0.0)
            )
        else:
            convexhull = Pattern._buildConvexHull(
                top[0], 
                bottom[0], 
                current_assist.get("width", 0.0)
            )
        
        # Calculate resistance along outer curve (without twist)
        if len(shape) > 0:
            # Thickness: 0.047 mm (copper foil standard)
            thickness = 0.047
            # Calculate resistance from index 0 to outer_end_idx
            resistance = Calculate.ResistanceAlongPath(shape, 0, outer_end_idx, thickness)
        else:
            resistance = 0.0

        return {
            "bbox": Pattern._buildBbox(params_current),
            "assist": current_assist,
            "shape": shape,
            "convexhull": convexhull,
            "convexhull_area": Calculate.AreaOfClosedPolygon(convexhull),
            "pattern_area": Calculate.AreaOfClosedPolygon(shape),
            "pattern_resistance": resistance,
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
        convexhull_pts = self._to_point_list(self._pattern.get("convexhull"))

        all_points: List[Tuple[float, float]] = []
        all_points.extend(bbox_pts)
        for pts in assist_polys.values():
            all_points.extend(pts)
        all_points.extend(shape_pts)
        all_points.extend(convexhull_pts)

        if not all_points:
            return

        bounds = self._compute_bounds(all_points)
        if bounds is None:
            return
        mapper = self._build_mapper(bounds)

        self._draw_bbox(painter, mapper, bbox_pts)
        self._draw_convexhull(painter, mapper, convexhull_pts)
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

    def _draw_convexhull(self, painter: QPainter, mapper, points: List[Tuple[float, float]]):
        """Draw the convex hull with distinct styling."""
        if len(points) < 3:
            return

        polygon = QPolygonF([mapper(pt) for pt in points])
        # Green with transparency for convex hull
        fill = QBrush(QColor(76, 175, 80, 40))  # Light green fill
        outline = QPen(QColor(56, 142, 60), 2.0)  # Darker green outline
        outline.setCosmetic(True)
        outline.setStyle(Qt.DashLine)  # Dashed line for distinction

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
