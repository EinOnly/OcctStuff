"""
Rectangle Pattern Generator

This module handles rectangle geometry calculations and data generation.
It provides a clean interface for generating rectangle data without any visualization dependencies.
All matplotlib functionality has been moved to visualize.py for better separation of concerns.
"""


class RectangleGenerator:
    """
    A class to generate rectangle geometry data based on adjustable parameters.
    
    This class focuses purely on mathematical calculations and data structure generation,
    with no visualization dependencies.
    """
    
    def __init__(self, coef=1.0, W=0.0, H=0.0):
        """
        Initialize the rectangle generator with default parameters.
        
        Args:
            coef (float): Scaling coefficient for the base unit rectangle
            W (float): Additional width added to the center of horizontal sides
            H (float): Additional height added to the center of vertical sides
        """
        self.coef = coef
        self.W = W
        self.H = H
        
        # Default positioning
        self.origin_x = 2.0
        self.origin_y = 2.0
        
    def get_parameters(self):
        """
        Get current parameter values.
        
        Returns:
            dict: Dictionary containing current coef, W, and H values
        """
        return {
            'coef': self.coef,
            'W': self.W,
            'H': self.H
        }
        
    def set_coef(self, value):
        """Set the coef parameter."""
        self.coef = max(0.0, value)
        
    def set_w(self, value):
        """Set the W parameter."""
        self.W = max(0.0, value)
        
    def set_h(self, value):
        """Set the H parameter."""
        self.H = max(0.0, value)
        
    def get_dimensions(self):
        """
        Calculate rectangle dimensions.
        
        Returns:
            dict: Dictionary containing width and height
        """
        total_width = 2 * self.coef + self.W
        total_height = 2 * self.coef + self.H
        
        return {
            'width': total_width,
            'height': total_height
        }
        
    def generate_rectangle(self):
        """
        Generate complete rectangle data structure for visualization.
        
        Returns:
            dict: Complete rectangle data including segments, dimensions, and positioning
        """
        # Calculate basic dimensions
        total_width = 2 * self.coef + self.W
        total_height = 2 * self.coef + self.H
        center_x = self.origin_x + total_width / 2
        
        # Generate segment data for each edge
        bottom_segments = self._generate_horizontal_segments(self.origin_y)
        top_segments = self._generate_horizontal_segments(self.origin_y + total_height)
        left_segments = self._generate_vertical_segments(self.origin_x)
        right_segments = self._generate_vertical_segments(self.origin_x + total_width)
        
        return {
            'origin': (self.origin_x, self.origin_y),
            'total_width': total_width,
            'total_height': total_height,
            'center_x': center_x,
            'bottom_segments': bottom_segments,
            'top_segments': top_segments,
            'left_segments': left_segments,
            'right_segments': right_segments,
            'parameters': self.get_parameters()
        }
        
    def _generate_horizontal_segments(self, y_coord):
        """
        Generate horizontal edge segments (bottom or top edge).
        
        Args:
            y_coord (float): Y coordinate of the edge
            
        Returns:
            list: List of segment dictionaries with type, start, and end positions
        """
        segments = []
        current_x = self.origin_x
        
        # Left coef segment
        segments.append({
            'type': 'coef',
            'x_start': current_x,
            'x_end': current_x + self.coef,
            'y_coord': y_coord
        })
        current_x += self.coef
        
        # W segment (center)
        if self.W > 0:
            segments.append({
                'type': 'W',
                'x_start': current_x,
                'x_end': current_x + self.W,
                'y_coord': y_coord
            })
        current_x += self.W
        
        # Right coef segment
        segments.append({
            'type': 'coef',
            'x_start': current_x,
            'x_end': current_x + self.coef,
            'y_coord': y_coord
        })
        
        return segments
        
    def _generate_vertical_segments(self, x_coord):
        """
        Generate vertical edge segments (left or right edge).
        
        Args:
            x_coord (float): X coordinate of the edge
            
        Returns:
            list: List of segment dictionaries with type, start, and end positions
        """
        segments = []
        current_y = self.origin_y
        
        # Bottom coef segment
        segments.append({
            'type': 'coef',
            'x_coord': x_coord,
            'y_start': current_y,
            'y_end': current_y + self.coef
        })
        current_y += self.coef
        
        # H segment (center)
        if self.H > 0:
            segments.append({
                'type': 'H',
                'x_coord': x_coord,
                'y_start': current_y,
                'y_end': current_y + self.H
            })
        current_y += self.H
        
        # Top coef segment
        segments.append({
            'type': 'coef',
            'x_coord': x_coord,
            'y_start': current_y,
            'y_end': current_y + self.coef
        })
        
        return segments
        
    def get_formula_string(self):
        """
        Generate formula string representation.
        
        Returns:
            str: Human-readable formula string
        """
        return f"({self.coef:.1f}+{self.W:.1f}+{self.coef:.1f}) × ({self.coef:.1f}+{self.H:.1f}+{self.coef:.1f})"
        
    def validate_parameters(self, coef=None, W=None, H=None):
        """
        Validate parameter values.
        
        Args:
            coef (float, optional): Coef value to validate
            W (float, optional): W value to validate  
            H (float, optional): H value to validate
            
        Returns:
            dict: Dictionary of validated parameters
        """
        validated = {}
        
        if coef is not None:
            validated['coef'] = max(0.0, min(20.0, coef))
        if W is not None:
            validated['W'] = max(0.0, min(20.0, W))
        if H is not None:
            validated['H'] = max(0.0, min(20.0, H))
            
        return validated
