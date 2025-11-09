from __future__ import annotations

import math
from typing import Dict, List, Tuple

from superellipse import Superellipse

POINT_EPSILON = 1e-12


class Pattern:
    """
    Pattern geometry based on a single quadrant definition.

    Parameters:
        - vbh_bottom: vertical straight segment for lower half of left edge
        - vbh_top: vertical straight segment for upper half of left edge
        - vth_top / vth_bottom: superellipse corner radii in Y
        - vlw_top / vlw_bottom: superellipse corner radii in X
        - vrw_top / vrw_bottom: horizontal straight segment lengths to symmetry axis

    Constraints:
        - (vbh_bottom + vth_bottom) == height / 2
        - (vbh_top + vth_top) == height / 2
        - (vlw_top + vrw_top) == width / 2
        - (vlw_bottom + vrw_bottom) == width / 2

    Mode A (exponent == 2):
        - vbh (bottom straight) and vlw (top horizontal radius) are user controlled.
        - Top and bottom halves stay symmetric.

    Mode B (exponent < 2):
        - Top and bottom corners expose independent sliders.
        - Straight segments are derived per half to satisfy constraints.
    """

    _EXP_TOLERANCE = 1e-6

    def __init__(self, width: float = 1.0, height: float = 1.0):
        self.width = max(0.0, width)
        self.height = max(0.0, height)

        # Superellipse exponent (default < 2, so start in Mode B)
        self.exponent = 0.80
        self.exponent_m = 0.80

        # Core parameters (initialised before applying constraints)
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        initial_corner = min(half_width, half_height) * 0.5
        self.vlw = initial_corner  # top corner horizontal radius
        self.vth = initial_corner  # top corner vertical radius
        self.vrw = max(0.0, half_width - self.vlw)  # top straight segment
        self.vbh = max(0.0, half_height - self.vth)  # bottom straight segment (Mode A control)

        # Independent corner controls for Mode B
        self.corner_top_value = initial_corner
        self.corner_bottom_value = initial_corner

        # Derived bottom half parameters
        self.vlw_bottom = initial_corner
        self.vth_bottom = initial_corner
        self.vrw_bottom = max(0.0, half_width - self.vlw_bottom)
        self.vbh_bottom = max(0.0, half_height - self.vth_bottom)
        self.vbh_top = max(0.0, half_height - self.vth)

        # Allow UI-controlled extension beyond geometric min corner limit
        self.corner_margin = 0.0

        # Manual mode control (default based on exponent)
        self.mode = 'A' if math.isclose(self.exponent, 2.0, abs_tol=self._EXP_TOLERANCE) else 'B'
        self._symmetric_envelope: List[Tuple[float, float]] = []

        # Superellipse helper
        self.superellipse = Superellipse.get_instance()
        self.superellipse.set_exponents(self.exponent, self.exponent_m)

        self._apply_constraints()

    def reset_with_dimensions(self, width: float, height: float):
        """Reset pattern with new width and height, reinitializing all parameters."""
        self.width = max(0.0, width)
        self.height = max(0.0, height)
        self._symmetric_envelope = []

        # Reinitialize core parameters
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        initial_corner = min(half_width, half_height) * 0.5
        self.vlw = initial_corner
        self.vth = initial_corner
        self.vrw = max(0.0, half_width - self.vlw)
        self.vbh = max(0.0, half_height - self.vth)

        self.corner_top_value = initial_corner
        self.corner_bottom_value = initial_corner

        self.vlw_bottom = initial_corner
        self.vth_bottom = initial_corner
        self.vrw_bottom = max(0.0, half_width - self.vlw_bottom)
        self.vbh_bottom = max(0.0, half_height - self.vth_bottom)
        self.vbh_top = max(0.0, half_height - self.vth)

        self._apply_constraints()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _is_mode_a(self, exponent: float | None = None) -> bool:
        if exponent is not None:
            return math.isclose(exponent, 2.0, abs_tol=self._EXP_TOLERANCE)
        return self.mode == 'A'

    def _apply_constraints(self):
        self._symmetric_envelope = []
        half_width = max(0.0, self.width / 2.0)
        half_height = max(0.0, self.height / 2.0)

        if self._is_mode_a():
            # Clamp user controlled parameters
            # In Mode A (exponent == 2), vlw max is reduced by 0.05
            # vlw_max = max(0.0, half_width - 0.05)
            vlw_max = max(0.0, half_width)
            self.vbh = self._clamp(self.vbh, 0.0, half_height)
            self.vlw = self._clamp(self.vlw, 0.0, vlw_max)
            self.vlw_bottom = self._clamp(self.vlw_bottom, 0.0, vlw_max)

            # Derived values
            self.vth = max(0.0, half_height - self.vbh)
            self.vrw = max(0.0, half_width - self.vlw)
            self.vrw_bottom = max(0.0, half_width - self.vlw_bottom)
            self.vth_bottom = self.vth

            # Mirror values to keep halves in sync while respecting spacing allowance
            corner_max = min(self.vth, self.vlw) + self.corner_margin
            corner_bottom_max = min(self.vth_bottom, self.vlw_bottom) + self.corner_margin
            self.corner_top_value = min(self.corner_top_value, corner_max)
            self.corner_bottom_value = min(self.corner_bottom_value, corner_bottom_max)
            self.vbh_bottom = self.vbh
            self.vbh_top = max(0.0, half_height - self.vth)
        else:
            # Independent corner sliders drive top/bottom radii
            max_corner = min(half_width, half_height) + self.corner_margin
            self.corner_top_value = self._clamp(self.corner_top_value, 0.0, max_corner)
            self.corner_bottom_value = self._clamp(self.corner_bottom_value, 0.0, max_corner)

            self.vth = self.corner_top_value
            self.vlw = self.corner_top_value
            self.vrw = max(0.0, half_width - self.vlw)
            self.vbh_top = max(0.0, half_height - self.vth)

            self.vth_bottom = self.corner_bottom_value
            self.vlw_bottom = self.corner_bottom_value
            self.vrw_bottom = max(0.0, half_width - self.vlw_bottom)
            self.vbh = max(0.0, half_height - self.vth_bottom)
            self.vbh_bottom = self.vbh

    def snapshot(self) -> Dict[str, float]:
        """Capture current state for temporary simulations."""
        return {
            'width': self.width,
            'height': self.height,
            'mode': self.mode,
            'exponent': self.exponent,
            'exponent_m': self.exponent_m,
            'vbh': self.vbh,
            'vth': self.vth,
            'vlw': self.vlw,
            'vrw': self.vrw,
            'vlw_bottom': self.vlw_bottom,
            'corner_top_value': self.corner_top_value,
            'corner_bottom_value': self.corner_bottom_value,
        }

    def restore(self, state: Dict[str, float]):
        """Restore state captured by snapshot()."""
        self.width = max(0.0, state['width'])
        self.height = max(0.0, state['height'])
        self.mode = state.get('mode', self.mode)
        self.exponent = self._clamp(state['exponent'], 0.5, 2.0)
        self.exponent_m = self._clamp(state.get('exponent_m', self.exponent_m), 0.1, 2.0)
        self.superellipse.set_exponents(self.exponent, self.exponent_m)

        self.vbh = max(0.0, state['vbh'])
        self.vlw = max(0.0, state['vlw'])
        self.vlw_bottom = max(0.0, state.get('vlw_bottom', self.vlw_bottom))
        self.corner_top_value = max(0.0, state.get('corner_top_value', self.corner_top_value))
        self.corner_bottom_value = max(0.0, state.get('corner_bottom_value', self.corner_bottom_value))
        # Derived fields will be reconstructed below
        self._apply_constraints()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_mode(self) -> str:
        """Return current pattern mode identifier."""
        return self.mode

    def GetVariables(self) -> List[Dict[str, float]]:
        half_width = max(0.0, self.width / 2.0)
        half_height = max(0.0, self.height / 2.0)

        variables: List[Dict[str, float]] = [
            {
                'label': 'width',
                'value': self.width,
                'min': 0.0,
                'max': max(self.width, 5.0),
                'step': 0.1,
            },
            {
                'label': 'height',
                'value': self.height,
                'min': 0.0,
                'max': max(self.height, 5.0),
                'step': 0.1,
            },
        ]

        if self._is_mode_a():
            # When exponent == 2 (Mode A), reduce vlw max by 0.05
            vlw_max = max(0.0, half_width)
            variables.extend([
                {
                    'label': 'vbh',
                    'value': self.vbh,
                    'min': 0.0,
                    'max': half_height,
                    'step': 0.01,
                },
                {
                    'label': 'vlw_top',
                    'value': self.vlw,
                    'min': 0.0,
                    'max': vlw_max,
                    'step': 0.01,
                },
                {
                    'label': 'vlw_bottom',
                    'value': self.vlw_bottom,
                    'min': 0.0,
                    'max': vlw_max,
                    'step': 0.01,
                },
            ])
        else:
            max_corner = min(half_width, half_height)
            variables.extend([
                {
                    'label': 'corner_bottom',
                    'value': self.corner_bottom_value,
                    'min': 0.0,
                    'max': max_corner,
                    'step': 0.01,
                },
                {
                    'label': 'corner_top',
                    'value': self.corner_top_value,
                    'min': 0.0,
                    'max': max_corner,
                    'step': 0.01,
                },
            ])

        variables.append({
            'label': 'exponent',
            'value': self.exponent,
            'min': 0.5,
            'max': 2.0,
            'step': 0.01,
        })
        variables.append({
            'label': 'exponent_m',
            'value': self.exponent_m,
            'min': 0.1,
            'max': 2.0,
            'step': 0.01,
        })

        return variables

    def SetVariable(self, label: str, value: float):
        if label == 'width':
            self.width = max(0.0, value)
        elif label == 'height':
            self.height = max(0.0, value)
        elif label == 'vbh':
            self.vbh = max(0.0, value)
        elif label == 'vlw':
            self.vlw = max(0.0, value)
        elif label == 'vlw_top':
            self.vlw = max(0.0, value)
        elif label == 'vlw_bottom':
            self.vlw_bottom = max(0.0, value)
        elif label == 'corner_bottom':
            self.corner_bottom_value = max(0.0, value)
        elif label == 'corner_top':
            self.corner_top_value = max(0.0, value)
        elif label == 'corner':
            # Backward compatibility: update both sliders
            sanitized = max(0.0, value)
            self.corner_top_value = sanitized
            self.corner_bottom_value = sanitized
        elif label == 'exponent':
            value = self._clamp(value, 0.5, 2.0)
            self.exponent = value
            self.superellipse.set_exponents(self.exponent, self.exponent_m)
        elif label in ('exponent_m', 'm'):
            value = self._clamp(value, 0.1, 2.0)
            self.exponent_m = value
            self.superellipse.set_exponents(self.exponent, self.exponent_m)
        else:
            raise ValueError(f"Unknown parameter label: {label}")

        self._apply_constraints()

    def set_mode(self, mode: str):
        """Manually switch between Mode A and Mode B."""
        normalized = mode.upper()
        if normalized not in ('A', 'B'):
            raise ValueError(f"Unknown mode: {mode}")
        if self.mode == normalized:
            return

        if normalized == 'A':
            # Carry existing corner definitions into straight/corner form
            self.vlw = min(self.corner_top_value, max(0.0, self.width / 2.0))
            self.vlw_bottom = min(self.corner_bottom_value, max(0.0, self.width / 2.0))
        else:
            # Carry straight definitions into independent corner sliders
            self.corner_top_value = min(self.vth, self.vlw)
            self.corner_bottom_value = min(self.vth_bottom, self.vlw_bottom)

        self.mode = normalized
        self._apply_constraints()

    def GetSymmetricEnvelope(self) -> List[Tuple[float, float]]:
        """Return the cached symmetric closed polygon, computing if needed."""
        if not self._symmetric_envelope:
            self.GetSymmetricCurveArea()
        return list(self._symmetric_envelope)

    # --------------------------------------------------------------------- #
    # Geometry helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _points_close(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return abs(a[0] - b[0]) <= POINT_EPSILON and abs(a[1] - b[1]) <= POINT_EPSILON

    @staticmethod
    def _fmt_point(x: float, y: float) -> Tuple[float, float]:
        return (round(x, 15), round(y, 15))

    def _x_limit_for_y(self, y: float) -> float:
        half_height = self.height / 2.0
        if y <= half_height + POINT_EPSILON:
            return max(0.0, self.vlw_bottom)
        return max(0.0, self.vlw)

    def _key_points(self) -> Dict[str, Tuple[float, float]]:
        half_height = self.height / 2.0
        bottom_limit = self._x_limit_for_y(0.0)
        top_limit = self._x_limit_for_y(self.height)

        return {
            'bottom_center': self._fmt_point(bottom_limit, 0.0),
            'bottom_corner_start': self._fmt_point(self.vlw_bottom, 0.0),
            'bottom_vertical_end': self._fmt_point(0.0, self.vth_bottom),
            'center_left': self._fmt_point(0.0, half_height),
            'top_vertical_start': self._fmt_point(0.0, self.height - self.vth),
            'top_corner_start': self._fmt_point(self.vlw, self.height),
            'top_center': self._fmt_point(top_limit, self.height),
        }

    def _split_curve_sections(self, curve: List[Tuple[float, float]],
                              split_y: float) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Split a monotonic curve into lower/upper sections at the given Y."""
        lower: List[Tuple[float, float]] = []
        upper: List[Tuple[float, float]] = []
        if not curve:
            return lower, upper

        def add_unique(target: List[Tuple[float, float]], point: Tuple[float, float]):
            if not target or not self._points_close(target[-1], point):
                target.append(point)

        eps = 1e-9
        split_idx = None
        for idx, point in enumerate(curve):
            if point[1] >= split_y - eps:
                split_idx = idx
                break
        if split_idx is None:
            return list(curve), []

        prev_index = max(0, split_idx - 1)
        prev_point = curve[prev_index]
        curr_point = curve[split_idx]

        if split_idx == 0:
            seam_point = self._fmt_point(curr_point[0], split_y)
        else:
            if abs(curr_point[1] - prev_point[1]) <= eps:
                seam_x = curr_point[0]
            else:
                t = (split_y - prev_point[1]) / (curr_point[1] - prev_point[1])
                t = max(0.0, min(1.0, t))
                seam_x = prev_point[0] + t * (curr_point[0] - prev_point[0])
            seam_point = self._fmt_point(seam_x, split_y)

        for idx in range(split_idx):
            add_unique(lower, curve[idx])
        add_unique(lower, seam_point)

        add_unique(upper, seam_point)
        for idx in range(split_idx, len(curve)):
            add_unique(upper, curve[idx])

        return lower, upper

    # --------------------------------------------------------------------- #
    # Geometry extraction
    # --------------------------------------------------------------------- #
    def GetBbox(self) -> Dict[str, List[Tuple[float, float]]]:
        half_width = self.width / 2.0
        bbox_left = [
            self._fmt_point(0.0, 0.0),
            self._fmt_point(half_width, 0.0),
            self._fmt_point(half_width, self.height),
            self._fmt_point(0.0, self.height),
        ]

        symmetry = [
            self._fmt_point(half_width, 0.0),
            self._fmt_point(half_width, self.height),
        ]

        return {
            'bbox_left': bbox_left,
            'symmetry': symmetry,
        }

    def GetCurve(self) -> List[Tuple[float, float]]:
        points = self._key_points()
        curve: List[Tuple[float, float]] = []

        def add_point(pt: Tuple[float, float]):
            if not curve or not self._points_close(curve[-1], pt):
                curve.append(pt)

        add_point(points['bottom_center'])

        # Bottom horizontal straight segment towards the corner
        if self.vrw_bottom > POINT_EPSILON:
            add_point(points['bottom_corner_start'])

        # Bottom-left corner
        corner_lb = self.superellipse.generate_corner_points(
            self.vlw_bottom,
            self.vth_bottom,
            3,
            0.0,
            0.0,
        )
        for idx, pt in enumerate(corner_lb):
            if idx == 0 and self._points_close(pt, points['bottom_corner_start']):
                continue
            add_point(pt)

        # Vertical straight to the centre
        lower_target = self._fmt_point(0.0, self.vth_bottom + self.vbh_bottom)
        add_point(lower_target)

        # Continue vertical straight to the start of the top corner
        upper_target = points['top_vertical_start']
        add_point(upper_target)

        # Top-left corner
        corner_lt = self.superellipse.generate_corner_points(
            self.vlw,
            self.vth,
            0,
            0.0,
            self.height,
        )
        for idx, pt in enumerate(corner_lt):
            if idx == 0 and self._points_close(pt, upper_target):
                continue
            add_point(pt)

        # Top horizontal towards the symmetry axis
        add_point(points['top_center'])

        return curve

    def _apply_ct_offset_state(self, ct_offset: float) -> bool:
        """Temporarily enlarge the top corner span by ct_offset."""
        if ct_offset <= POINT_EPSILON:
            return False

        target_top = max(0.0, self.vlw + ct_offset)
        if target_top <= self.vlw + POINT_EPSILON:
            return False

        required_width = max(self.width, target_top * 2.0)
        if required_width > self.width + POINT_EPSILON:
            self.width = required_width
            self._apply_constraints()

        if self._is_mode_a():
            self.SetVariable('vlw_top', target_top)
        else:
            self.SetVariable('corner_top', max(0.0, self.corner_top_value + ct_offset))
        return True

    def GetCurveWithCtOffset(self, ct_offset: float, split_y: float | None = None) -> List[Tuple[float, float]]:
        """
        Generate a curve where the upper section uses an increased CT value.
        The lower section is preserved from the original geometry.
        """
        if ct_offset <= POINT_EPSILON:
            return self.GetCurve()

        base_curve = self.GetCurve()
        if not base_curve:
            return []

        if split_y is None:
            split_target = max(0.0, min(self.height, self.height - self.vth))
        else:
            split_target = min(max(split_y, 0.0), self.height)

        original_state = self.snapshot()
        try:
            if not self._apply_ct_offset_state(ct_offset):
                return base_curve
            offset_curve = self.GetCurve()
        finally:
            self.restore(original_state)

        if not offset_curve:
            return base_curve

        lower_base, _ = self._split_curve_sections(base_curve, split_target)
        _, upper_variant = self._split_curve_sections(offset_curve, split_target)

        if not lower_base or not upper_variant:
            return base_curve

        combined = list(lower_base)
        if self._points_close(combined[-1], upper_variant[0]):
            combined.extend(upper_variant[1:])
        else:
            combined.extend(upper_variant)

        return combined


    def _build_clipped_left_curve(self) -> List[Tuple[float, float]]:
        """Return left curve clamped to the symmetry axis."""
        raw_curve = self.GetCurve()
        if not raw_curve:
            return []

        center_x = self.width / 2.0
        left_path: List[Tuple[float, float]] = []

        def add_point(pt: Tuple[float, float]):
            if not left_path or not self._points_close(left_path[-1], pt):
                left_path.append(pt)

        for idx, (x, y) in enumerate(raw_curve):
            x_clamped = min(x, center_x)
            add_point((x_clamped, y))

            if idx >= len(raw_curve) - 1:
                break

            x2, y2 = raw_curve[idx + 1]
            crosses_center = (x - center_x) * (x2 - center_x) < 0.0
            if crosses_center and not math.isclose(x2, x, abs_tol=POINT_EPSILON):
                t = (center_x - x) / (x2 - x)
                t = max(0.0, min(1.0, t))
                y_cross = y + t * (y2 - y)
                add_point((center_x, y_cross))
            elif x > center_x and x2 > center_x:
                add_point((center_x, y2))

        if left_path:
            left_path[0] = (center_x, left_path[0][1])
            left_path[-1] = (center_x, left_path[-1][1])

        return left_path

    def GetClippedLeftCurve(self) -> List[Tuple[float, float]]:
        """Expose the symmetry-clamped left curve for rendering/export."""
        return self._build_clipped_left_curve()

    def GetSegments(self, assembly_offset: float | None = None, space: float = 0.1
                    ) -> Dict[str, List[Tuple[float, float]]]:
        bbox = self.GetBbox()
        curve_left = self.GetClippedLeftCurve()

        result = {
            'bbox_left': bbox['bbox_left'],
            'curve_left': curve_left,
            'symmetry': bbox['symmetry'],
        }

        if assembly_offset is not None:
            result['shape'] = self.GetShape(assembly_offset, space)

        return result

    def GetCircuit(self, align: str = 'left') -> Dict[str, object]:
        segments = self.GetSegments()
        return {
            'curve': segments['curve_left'],
            'align': align,
            'width': self.width,
        }

    # --------------------------------------------------------------------- #
    # Shape utilities
    # --------------------------------------------------------------------- #
    def GetShape(self, offset: float, space: float,
                 curve_override: List[Tuple[float, float]] | None = None) -> List[Tuple[float, float]]:
        curve_left = curve_override if curve_override is not None else self.GetCurve()
        if not curve_left:
            return []

        max_y = self.height
        min_y = 0.0
        # Offset curve
        curve_b = [(x + offset, y) for x, y in curve_left]

        # Calculate inset curve (right boundary)
        pcl_raw: List[Tuple[float, float]] = []
        for i in range(len(curve_b)):
            if i == 0:
                dx = curve_b[i + 1][0] - curve_b[i][0]
                dy = curve_b[i + 1][1] - curve_b[i][1]
            elif i == len(curve_b) - 1:
                dx = curve_b[i][0] - curve_b[i - 1][0]
                dy = curve_b[i][1] - curve_b[i - 1][1]
            else:
                dx = curve_b[i + 1][0] - curve_b[i - 1][0]
                dy = curve_b[i + 1][1] - curve_b[i - 1][1]

            length = math.hypot(dx, dy)
            if length > POINT_EPSILON:
                tx = dx / length
                ty = dy / length
                nx = -ty
                ny = tx
            else:
                nx, ny = -1.0, 0.0

            pcl_x = curve_b[i][0] + nx * space
            pcl_y = curve_b[i][1] + ny * space
            pcl_raw.append((pcl_x, pcl_y))

        # Clip the inset curve to valid region (piecewise limits in X, 0 <= y <= max_y)
        pcl_clipped: List[Tuple[float, float]] = []
        
        for i in range(len(pcl_raw)):
            x1, y1 = pcl_raw[i]
            x2, y2 = pcl_raw[(i + 1) % len(pcl_raw)]
            
            # Check if segment needs clipping
            seg_points = self._clip_segment_piecewise(x1, y1, x2, y2, min_y, max_y)
            
            # Add clipped segment points (avoid duplicates)
            for pt in seg_points:
                if not pcl_clipped or not self._points_close(pcl_clipped[-1], pt):
                    pcl_clipped.append(self._fmt_point(pt[0], pt[1]))
        
        if not pcl_clipped:
            # If all points clipped out, fallback to simple shape
            pcl_clipped = [
                self._fmt_point(self.vlw_bottom, min_y),
                self._fmt_point(self.vlw, max_y),
            ]

        # Build closed shape: left curve + reversed clipped right boundary
        closed_shape = list(curve_left)
        closed_shape.extend(reversed(pcl_clipped))
        
        # Close the shape if needed
        if not self._points_close(closed_shape[-1], closed_shape[0]):
            closed_shape.append(curve_left[0])

        return closed_shape
    
    def _clip_segment_const(self, x1: float, y1: float, x2: float, y2: float,
                            max_x: float, min_y: float, max_y: float) -> List[Tuple[float, float]]:
        """
        Clip a line segment to the bounding box [0, max_x] × [min_y, max_y].
        Returns list of points representing the clipped segment.
        """
        # Check if both points are inside
        p1_inside = (x1 <= max_x and min_y <= y1 <= max_y)
        p2_inside = (x2 <= max_x and min_y <= y2 <= max_y)
        
        if p1_inside and p2_inside:
            return [(x1, y1)]
        
        result = []
        
        # Add first point if inside
        if p1_inside:
            result.append((x1, y1))
        
        # Check for intersections with boundaries
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > POINT_EPSILON or abs(dy) > POINT_EPSILON:
            # Check intersection with x = max_x
            if (x1 <= max_x < x2) or (x2 < max_x <= x1):
                t = (max_x - x1) / dx if abs(dx) > POINT_EPSILON else 0
                if 0 <= t <= 1:
                    y_int = y1 + t * dy
                    if min_y <= y_int <= max_y:
                        result.append((max_x, y_int))
            
            # Check intersection with y = min_y
            if (y1 >= min_y > y2) or (y2 > min_y >= y1):
                t = (min_y - y1) / dy if abs(dy) > POINT_EPSILON else 0
                if 0 <= t <= 1:
                    x_int = x1 + t * dx
                    if x_int <= max_x:
                        result.append((x_int, min_y))
            
            # Check intersection with y = max_y
            if (y1 <= max_y < y2) or (y2 < max_y <= y1):
                t = (max_y - y1) / dy if abs(dy) > POINT_EPSILON else 0
                if 0 <= t <= 1:
                    x_int = x1 + t * dx
                    if x_int <= max_x:
                        result.append((x_int, max_y))
        
        # Add second point if inside
        if p2_inside:
            result.append((x2, y2))
        
        return result
    
    def _clip_segment_piecewise(self, x1: float, y1: float, x2: float, y2: float,
                                min_y: float, max_y: float) -> List[Tuple[float, float]]:
        """
        Clip a segment against the dynamic x-limits defined by top/bottom radii.
        """
        half_height = self.height / 2.0
        points: List[Tuple[float, float]] = [(x1, y1)]

        # Split at the horizontal midline where the x-limit changes
        if (y1 - half_height) * (y2 - half_height) < -POINT_EPSILON:
            if abs(y2 - y1) > POINT_EPSILON:
                t = (half_height - y1) / (y2 - y1)
                xi = x1 + t * (x2 - x1)
                points.append((xi, half_height))

        points.append((x2, y2))

        result: List[Tuple[float, float]] = []
        for start, end in zip(points, points[1:]):
            mid_y = (start[1] + end[1]) / 2.0
            max_x = self._x_limit_for_y(mid_y)

            seg = self._clip_segment_const(start[0], start[1], end[0], end[1], max_x, min_y, max_y)
            for pt in seg:
                if not result or not self._points_close(result[-1], pt):
                    result.append(pt)

        return result

    def GetShapeArea(self, offset: float, space: float) -> float:
        shape = self.GetShape(offset, space)
        if not shape or len(shape) < 3:
            return 0.0

        area = 0.0
        for i in range(len(shape) - 1):
            x1, y1 = shape[i]
            x2, y2 = shape[i + 1]
            area += x1 * y2 - x2 * y1

        return abs(area) / 2.0

    def GetRectangleArea(self, offset: float, space: float) -> float:
        vertical_span = max(self.vbh_top + self.vbh_bottom, 0.0)
        return vertical_span * max(offset - space, 0.0)

    def GetEquivalentCoefficient(self, offset: float, space: float) -> float:
        s1 = self.GetShapeArea(offset, space)
        s2 = self.GetRectangleArea(offset, space)

        if s2 == 0.0:
            return 0.0

        return s1 / s2

    def GetSymmetricCurveArea(self) -> float:
        """Area enclosed by the symmetry-clamped envelope."""
        left_path = self._build_clipped_left_curve()
        if not left_path or len(left_path) < 2:
            self._symmetric_envelope = []
            return 0.0

        center_x = self.width / 2.0

        right_path: List[Tuple[float, float]] = []
        for x, y in reversed(left_path):
            mirrored = (2 * center_x - x, y)
            if not right_path or not self._points_close(right_path[-1], mirrored):
                right_path.append(mirrored)

        envelope: List[Tuple[float, float]] = list(left_path)
        for pt in right_path:
            if not envelope or not self._points_close(envelope[-1], pt):
                envelope.append(pt)

        if envelope and not self._points_close(envelope[0], envelope[-1]):
            envelope.append(envelope[0])

        self._symmetric_envelope = envelope

        area = 0.0
        for i in range(len(envelope) - 1):
            x1, y1 = envelope[i]
            x2, y2 = envelope[i + 1]
            area += x1 * y2 - x2 * y1

        return abs(area) / 2.0
    
    def GetResistance(self, offset: float, space: float, thick: float = 0.047, rho: float = 1.724e-8) -> float:
        """
        Calculate electrical resistance for flat conductor in motor slot.
        
        Integration along the left curve, finding nearest point on right boundary
        for each position to get accurate cross-sectional width.
        
        R = ρ ∫ (1/A(s)) ds where A(s) = w(s) × thick
        
        Args:
            offset: Assembly offset (mm)
            space: Spacing parameter (mm) 
            thick: Thickness of the conductor (mm), default 0.047 mm
            rho: Electrical resistivity (Ω·m), default 1.724e-8 Ω·m for copper
            
        Returns:
            Resistance in Ohms
        """
        curve_left = self.GetCurve()
        if not curve_left or len(curve_left) < 2:
            return 0.0
        
        # Convert units: mm to m
        thick_m = thick * 1e-3
        
        # Build right boundary using the same logic as GetShape
        curve_b = [(x + offset, y) for x, y in curve_left]
        
        # Calculate right boundary (inset by space along normal)
        curve_right: List[Tuple[float, float]] = []
        for i in range(len(curve_b)):
            if i == 0:
                dx = curve_b[i + 1][0] - curve_b[i][0]
                dy = curve_b[i + 1][1] - curve_b[i][1]
            elif i == len(curve_b) - 1:
                dx = curve_b[i][0] - curve_b[i - 1][0]
                dy = curve_b[i][1] - curve_b[i - 1][1]
            else:
                dx = curve_b[i + 1][0] - curve_b[i - 1][0]
                dy = curve_b[i + 1][1] - curve_b[i - 1][1]

            length = math.hypot(dx, dy)
            if length > POINT_EPSILON:
                tx = dx / length
                ty = dy / length
                nx = -ty  # Normal pointing inward
                ny = tx
            else:
                nx, ny = -1.0, 0.0

            # Right boundary point (inset along normal)
            pcl_x = curve_b[i][0] + nx * space
            pcl_y = curve_b[i][1] + ny * space
            curve_right.append((pcl_x, pcl_y))
        
        # Helper function to find nearest point on right boundary
        def find_nearest_point_on_right(left_point: Tuple[float, float]) -> Tuple[float, float, float]:
            """Returns (x_right, y_right, distance)"""
            x_left, y_left = left_point
            min_dist = float('inf')
            nearest_x, nearest_y = 0.0, 0.0
            
            # Check all segments of right boundary
            for i in range(len(curve_right)):
                x_r, y_r = curve_right[i]
                dist = math.hypot(x_r - x_left, y_r - y_left)
                if dist < min_dist:
                    min_dist = dist
                    nearest_x, nearest_y = x_r, y_r
            
            return nearest_x, nearest_y, min_dist
        
        # Integrate along the left curve
        integral = 0.0
        
        for i in range(len(curve_left) - 1):
            x_left_1, y_left_1 = curve_left[i]
            x_left_2, y_left_2 = curve_left[i + 1]
            
            # Calculate segment properties along left curve
            dx_left = x_left_2 - x_left_1
            dy_left = y_left_2 - y_left_1
            ds = math.hypot(dx_left, dy_left)  # Path length along left curve
            
            if ds < POINT_EPSILON:
                continue
            
            # Find nearest points on right boundary for both ends
            x_right_1, y_right_1, w1 = find_nearest_point_on_right((x_left_1, y_left_1))
            x_right_2, y_right_2, w2 = find_nearest_point_on_right((x_left_2, y_left_2))
            
            # Ensure positive widths
            w1 = max(w1, POINT_EPSILON)
            w2 = max(w2, POINT_EPSILON)
            
            # Convert to meters
            w1_m = w1 * 1e-3
            w2_m = w2 * 1e-3
            ds_m = ds * 1e-3
            
            # Cross-sectional areas at both ends
            A1 = w1_m * thick_m  # m²
            A2 = w2_m * thick_m  # m²
            
            # Trapezoidal integration along curve: ∫(1/A) ds
            integral += 0.5 * (1.0/A1 + 1.0/A2) * ds_m
        
        # Calculate resistance: R = ρ * integral
        resistance = rho * integral
        
        return resistance
