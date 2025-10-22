"""
Superellipse Path Generator

A class that generates superellipse (rounded rectangle with configurable corner shape)
paths with caching for performance optimization.
"""

import math
from typing import List, Tuple, Dict

RADIUS_EPSILON = 1e-12


class Superellipse:
    """
    Superellipse path generator for smooth corner transitions.
    
    Features:
    - Cached superellipse point calculations
    - Mathematical superellipse formulas for smooth corners
    - Support for per-corner radii configuration
    """
    
    # Singleton instance
    _instance = None
    
    def __init__(self):
        """Initialize the Superellipse generator"""
        # Number of curve samples (smoothness of the path) - increased for smoother curves
        self.steps = 100
        # Shape exponent parameter (n) of the superellipse
        self.exponent = 0.80
        # LRU cache for superellipse points
        self.cache = {}
        # Max cache size to prevent memory leaks
        self.cache_size = 120
        # Decimal precision for coordinates
        self.precision = 15
        # Pre-calculated quarter superellipse corner points
        self.corner = self._sample_quarter_superellipse(n=self.exponent, steps=self.steps)
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)"""
        cls._instance = None
    
    def _fmt(self, v: float) -> float:
        """Format number to specified precision"""
        if not math.isfinite(v):
            return 0.0
        return round(v, self.precision)
    
    def _sample_quarter_superellipse(self, n: float, steps: int) -> List[Tuple[float, float]]:
        """
        Generate a quarter superellipse curve as an array of (x, y) points.
        Uses LRU cache for efficiency.
        
        Args:
            n: Shape exponent parameter
            steps: Number of sample points
            
        Returns:
            List of (x, y) coordinate tuples
        """
        normalized_n = round(n * 1000) / 1000
        normalized_steps = max(2, int(steps))
        cache_key = f"n{normalized_n}_s{normalized_steps}"
        
        # Check cache
        if cache_key in self.cache:
            # Move to end (LRU)
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            return value
        
        # Calculate points
        safe_n = max(0.1, min(10, normalized_n))
        inv_n = 2 / safe_n
        neg_n_half = -safe_n / 2
        points = []
        
        for i in range(normalized_steps + 1):
            theta = (math.pi / 2) * (i / normalized_steps)
            cos_val = math.cos(theta)
            sin_val = math.sin(theta)
            rr = math.pow(
                math.pow(abs(cos_val), inv_n) + math.pow(abs(sin_val), inv_n),
                neg_n_half
            )
            # Invert the curve to make it convex (bulge outward) instead of concave
            # We flip the coordinates: (x, y) -> (1-y, 1-x) to get the outer curve
            x = rr * cos_val
            y = rr * sin_val
            points.append((1.0 - y, 1.0 - x))
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[cache_key] = points
        return points
    
    def set_exponent(self, n: float):
        """
        Update the exponent and recalculate corner template
        
        Args:
            n: New exponent value (typically 0.5 - 2.0)
        """
        self.exponent = max(0.1, min(10, n))
        self.corner = self._sample_quarter_superellipse(n=self.exponent, steps=self.steps)
    
    def generate_corner_points(
        self,
        radius_x: float,
        radius_y: float,
        corner_idx: int,
        corner_x: float,
        corner_y: float
    ) -> List[Tuple[float, float]]:
        """
        Generate points for a specific corner.

        Args:
            radius_x: Horizontal radius (distance from corner apex along X).
            radius_y: Vertical radius (distance from corner apex along Y).
            corner_idx: Corner index (0=LT, 1=RT, 2=RB, 3=LB).
            corner_x: X coordinate of the corner apex.
            corner_y: Y coordinate of the corner apex.

        Returns:
            List of (x, y) coordinates for the corner curve. When radii differ,
            a straight polyline between the two endpoints is returned using
            the same sampling density as the superellipse.
        """
        radius_x = max(0.0, radius_x)
        radius_y = max(0.0, radius_y)

        if radius_x < RADIUS_EPSILON and radius_y < RADIUS_EPSILON:
            return [(self._fmt(corner_x), self._fmt(corner_y))]

        dir_map = {
            0: (1.0, -1.0),
            1: (-1.0, -1.0),
            2: (-1.0, 1.0),
            3: (1.0, 1.0),
        }
        sign_x, sign_y = dir_map.get(corner_idx, (1.0, -1.0))

        use_curve = (
            abs(radius_x - radius_y) <= RADIUS_EPSILON
            and radius_x >= RADIUS_EPSILON
            and radius_y >= RADIUS_EPSILON
        )

        if use_curve:
            points: List[Tuple[float, float]] = []
            for base_x, base_y in self.corner:
                sx = base_x * radius_x
                sy = base_y * radius_y
                px = corner_x + sign_x * sx
                py = corner_y + sign_y * sy
                points.append((self._fmt(px), self._fmt(py)))

            if corner_idx in (0, 2):
                points.reverse()
            return points

        steps = max(2, len(self.corner))
        start = (
            corner_x + sign_x * radius_x,
            corner_y + sign_y * 0.0,
        )
        end = (
            corner_x + sign_x * 0.0,
            corner_y + sign_y * radius_y,
        )
        if corner_idx in (0, 2):
            start, end = end, start

        points: List[Tuple[float, float]] = []
        for i in range(steps):
            t = 0.0 if steps == 1 else i / (steps - 1)
            px = start[0] + (end[0] - start[0]) * t
            py = start[1] + (end[1] - start[1]) * t
            points.append((self._fmt(px), self._fmt(py)))

        return points
    
    @staticmethod
    def get_cache_size() -> int:
        """Get current cache size"""
        instance = Superellipse.get_instance()
        return len(instance.cache)
    
    @staticmethod
    def clear_cache():
        """Clear the cache"""
        instance = Superellipse.get_instance()
        instance.cache.clear()
    
    @staticmethod
    def set_precision(precision: int):
        """Set decimal precision"""
        instance = Superellipse.get_instance()
        instance.precision = max(0, min(15, precision))
    
    @staticmethod
    def get_precision() -> int:
        """Get current precision"""
        instance = Superellipse.get_instance()
        return instance.precision
