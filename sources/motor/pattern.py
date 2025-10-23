from __future__ import annotations

import math
from typing import Dict, List, Tuple

from superellipse import Superellipse

POINT_EPSILON = 1e-12


class Pattern:
    """
    Pattern geometry based on a single quadrant definition.

    Parameters:
        - vbh: vertical straight segment (lower half of left edge)
        - vth: vertical radius of the corner (upper half of left edge)
        - vlw: horizontal radius of the corner (left half of top edge)
        - vrw: horizontal straight segment (right half of top edge)

    Constraints:
        - (vbh + vth) * 2 == height
        - (vlw + vrw) * 2 == width

    Mode A (exponent == 2):
        - vbh and vlw are user controlled (sliders).
        - vth and vrw are derived to satisfy constraints.

    Mode B (exponent < 2):
        - vth == vlw == corner slider value.
        - vbh and vrw are derived to satisfy constraints.
    """

    _EXP_TOLERANCE = 1e-6

    def __init__(self, width: float = 1.0, height: float = 1.0):
        self.width = max(0.0, width)
        self.height = max(0.0, height)

        # Superellipse exponent (default < 2, so start in Mode B)
        self.exponent = 0.80

        # Core parameters (initialised before applying constraints)
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        initial_corner = min(half_width, half_height) * 0.5
        self.vlw = initial_corner
        self.vth = initial_corner
        self.vrw = max(0.0, half_width - self.vlw)
        self.vbh = max(0.0, half_height - self.vth)
        self.corner_value = initial_corner

        # Superellipse helper
        self.superellipse = Superellipse.get_instance()
        self.superellipse.set_exponent(self.exponent)

        self._apply_constraints()

    def reset_with_dimensions(self, width: float, height: float):
        """Reset pattern with new width and height, reinitializing all parameters."""
        self.width = max(0.0, width)
        self.height = max(0.0, height)

        # Reinitialize core parameters
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        initial_corner = min(half_width, half_height) * 0.5
        self.vlw = initial_corner
        self.vth = initial_corner
        self.vrw = max(0.0, half_width - self.vlw)
        self.vbh = max(0.0, half_height - self.vth)
        self.corner_value = initial_corner

        self._apply_constraints()

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _is_mode_a(self, exponent: float | None = None) -> bool:
        exp = self.exponent if exponent is None else exponent
        return math.isclose(exp, 2.0, abs_tol=self._EXP_TOLERANCE)

    def _apply_constraints(self):
        half_width = max(0.0, self.width / 2.0)
        half_height = max(0.0, self.height / 2.0)

        if self._is_mode_a():
            # Clamp user controlled parameters
            # In Mode A (exponent == 2), vlw max is reduced by 0.05
            vlw_max = max(0.0, half_width - 0.05)
            self.vbh = self._clamp(self.vbh, 0.0, half_height)
            self.vlw = self._clamp(self.vlw, 0.0, vlw_max)

            # Derived values
            self.vth = max(0.0, half_height - self.vbh)
            self.vrw = max(0.0, half_width - self.vlw)

            # Update stored corner slider baseline for mode switches
            self.corner_value = min(self.vth, self.vlw)
        else:
            # Unified corner value controls both radii
            max_corner = min(half_width, half_height)
            self.corner_value = self._clamp(self.corner_value, 0.0, max_corner)

            self.vth = self.corner_value
            self.vlw = self.corner_value
            self.vbh = max(0.0, half_height - self.vth)
            self.vrw = max(0.0, half_width - self.vlw)

    def snapshot(self) -> Dict[str, float]:
        """Capture current state for temporary simulations."""
        return {
            'width': self.width,
            'height': self.height,
            'exponent': self.exponent,
            'vbh': self.vbh,
            'vth': self.vth,
            'vlw': self.vlw,
            'vrw': self.vrw,
            'corner_value': self.corner_value,
        }

    def restore(self, state: Dict[str, float]):
        """Restore state captured by snapshot()."""
        self.width = max(0.0, state['width'])
        self.height = max(0.0, state['height'])
        self.exponent = self._clamp(state['exponent'], 0.5, 2.0)
        self.superellipse.set_exponent(self.exponent)

        self.vbh = max(0.0, state['vbh'])
        self.vlw = max(0.0, state['vlw'])
        self.corner_value = max(0.0, state['corner_value'])
        # Derived fields will be reconstructed below
        self._apply_constraints()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_mode(self) -> str:
        """Return current pattern mode identifier."""
        return 'A' if self._is_mode_a() else 'B'

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
            vlw_max = max(0.0, half_width - 0.05)
            variables.extend([
                {
                    'label': 'vbh',
                    'value': self.vbh,
                    'min': 0.0,
                    'max': half_height,
                    'step': 0.01,
                },
                {
                    'label': 'vlw',
                    'value': self.vlw,
                    'min': 0.0,
                    'max': vlw_max,
                    'step': 0.01,
                },
            ])
        else:
            variables.append({
                'label': 'corner',
                'value': self.vth,
                'min': 0.0,
                'max': min(half_width, half_height),
                'step': 0.01,
            })

        variables.append({
            'label': 'exponent',
            'value': self.exponent,
            'min': 0.5,
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
        elif label == 'corner':
            self.corner_value = max(0.0, value)
        elif label == 'exponent':
            prev_mode_a = self._is_mode_a()
            value = self._clamp(value, 0.5, 2.0)
            self.exponent = value
            self.superellipse.set_exponent(self.exponent)

            if prev_mode_a and not self._is_mode_a():
                # Preserve smooth transition into mode B
                self.corner_value = min(self.vth, self.vlw)
            elif not prev_mode_a and self._is_mode_a():
                # Carry over previous straight portion as baseline
                half_height = max(0.0, self.height / 2.0)
                half_width = max(0.0, self.width / 2.0)
                self.vbh = self._clamp(self.vbh, 0.0, half_height)
                self.vlw = self._clamp(self.vlw, 0.0, half_width)
        else:
            raise ValueError(f"Unknown parameter label: {label}")

        self._apply_constraints()

    # --------------------------------------------------------------------- #
    # Geometry helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _points_close(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return abs(a[0] - b[0]) <= POINT_EPSILON and abs(a[1] - b[1]) <= POINT_EPSILON

    @staticmethod
    def _fmt_point(x: float, y: float) -> Tuple[float, float]:
        return (round(x, 15), round(y, 15))

    def _key_points(self) -> Dict[str, Tuple[float, float]]:
        half_width = self.width / 2.0
        half_height = self.height / 2.0

        return {
            'bottom_center': self._fmt_point(half_width, 0.0),
            'bottom_corner_start': self._fmt_point(self.vlw, 0.0),
            'bottom_vertical_end': self._fmt_point(0.0, self.vth),
            'center_left': self._fmt_point(0.0, half_height),
            'top_vertical_start': self._fmt_point(0.0, self.height - self.vth),
            'top_corner_start': self._fmt_point(self.vlw, self.height),
            'top_center': self._fmt_point(half_width, self.height),
        }

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
        if self.vrw > POINT_EPSILON:
            add_point(points['bottom_corner_start'])

        # Bottom-left corner
        # Both top and bottom corners should use vth as the corner radius
        corner_lb = self.superellipse.generate_corner_points(
            self.vlw,
            self.vth,
            3,
            0.0,
            0.0,
        )
        for idx, pt in enumerate(corner_lb):
            if idx == 0 and self._points_close(pt, points['bottom_corner_start']):
                continue
            add_point(pt)

        # Vertical straight to the centre
        lower_target = self._fmt_point(0.0, self.vth + self.vbh)
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

    def GetSegments(self, assembly_offset: float | None = None, space: float = 0.1
                    ) -> Dict[str, List[Tuple[float, float]]]:
        bbox = self.GetBbox()
        curve_left = self.GetCurve()

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
    def GetShape(self, offset: float, space: float) -> List[Tuple[float, float]]:
        curve_left = self.GetCurve()
        if not curve_left:
            return []

        center_x = self.width / 2.0
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

        # Clip the inset curve to valid region (x <= center_x, 0 <= y <= max_y)
        pcl_clipped: List[Tuple[float, float]] = []
        
        for i in range(len(pcl_raw)):
            x1, y1 = pcl_raw[i]
            x2, y2 = pcl_raw[(i + 1) % len(pcl_raw)]
            
            # Check if segment needs clipping
            seg_points = self._clip_segment(x1, y1, x2, y2, center_x, min_y, max_y)
            
            # Add clipped segment points (avoid duplicates)
            for pt in seg_points:
                if not pcl_clipped or not self._points_close(pcl_clipped[-1], pt):
                    pcl_clipped.append(self._fmt_point(pt[0], pt[1]))
        
        if not pcl_clipped:
            # If all points clipped out, fallback to simple shape
            pcl_clipped = [self._fmt_point(center_x, min_y), 
                          self._fmt_point(center_x, max_y)]

        # Build closed shape: left curve + reversed clipped right boundary
        closed_shape = list(curve_left)
        closed_shape.extend(reversed(pcl_clipped))
        
        # Close the shape if needed
        if not self._points_close(closed_shape[-1], closed_shape[0]):
            closed_shape.append(curve_left[0])

        return closed_shape
    
    def _clip_segment(self, x1: float, y1: float, x2: float, y2: float,
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
        return (2.0 * self.vbh) * max(offset - space, 0.0)

    def GetEquivalentCoefficient(self, offset: float, space: float) -> float:
        s1 = self.GetShapeArea(offset, space)
        s2 = self.GetRectangleArea(offset, space)

        if s2 == 0.0:
            return 0.0

        return s1 / s2

    def GetSymmetricCurveArea(self) -> float:
        """
        Calculate area of the closed shape formed by the left curve 
        and its symmetric reflection along the y-axis (vertical symmetry).
        
        The curve goes from bottom to top on the left, then mirrors back 
        down on the right side to form a closed shape.
        
        Returns:
            Area in mm²
        """
        curve_left = self.GetCurve()
        if not curve_left or len(curve_left) < 2:
            return 0.0
        
        # Mirror the left curve along the vertical axis at x = width/2
        center_x = self.width / 2.0
        
        # Build closed polygon: left curve (bottom to top) + mirrored curve (top to bottom)
        closed_polygon: List[Tuple[float, float]] = []
        
        # Add left curve points (bottom to top)
        for x, y in curve_left:
            closed_polygon.append((x, y))
        
        # Add mirrored curve points in reverse order (top to bottom)
        for x, y in reversed(curve_left):
            x_mirrored = 2 * center_x - x  # Mirror across center_x
            closed_polygon.append((x_mirrored, y))
        
        # Calculate area using shoelace formula
        area = 0.0
        n = len(closed_polygon)
        for i in range(n):
            x1, y1 = closed_polygon[i]
            x2, y2 = closed_polygon[(i + 1) % n]
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
