from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from parameter import AssemblyParameters, Parameter
from pattern import Pattern
from step import StepExporter


@dataclass
class Assembly2DInstance:
    """Single repeated 2D pattern instance."""
    index: int
    offset: float
    left_shape: List[Tuple[float, float]]
    right_shape: List[Tuple[float, float]]


@dataclass
class AssemblySolidInstance:
    """3D solid representation for either flat array or spiral mapped geometry."""
    index: int
    offset: float
    shape: Any  # TopoDS_Shape when OCC is available
    error: Optional[str] = None


class AssemblyBuilder:
    """
    Coordinate pattern geometry and STEP extrusion so that the UI layer only needs
    to format or display the returned data.
    """

    def __init__(self,
        pattern: Pattern | None = None,
        assembly: AssemblyParameters | None = None,
        step_exporter: StepExporter | None = None):
        
        self.pattern = pattern or Pattern()
        self.assembly = assembly or AssemblyParameters()
        self.step_exporter = step_exporter or StepExporter(thickness=self.assembly.layer_thickness)
        self.parameters = Parameter(self.pattern, self.assembly)

    # ------------------------------------------------------------------ #
    # Pattern accessors and modifiers
    # ------------------------------------------------------------------ #
    @property
    def width(self) -> float:
        return self.pattern.width

    @property
    def height(self) -> float:
        return self.pattern.height

    def set_dimensions(self, width: float | None = None, height: float | None = None):
        current_width = self.pattern.width if width is None else width
        current_height = self.pattern.height if height is None else height
        self.pattern.reset_with_dimensions(current_width, current_height)

    def get_pattern_variables(self):
        return self.parameters.pattern_specs()

    def get_pattern_mode(self) -> str:
        return self.parameters.get_mode()

    def set_pattern_variable(self, label: str, value: float):
        self.parameters.set_pattern_value(label, value)

    def get_pattern_values(self) -> Dict[str, float]:
        return self.parameters.pattern_values()

    def set_pattern_mode(self, mode: str):
        self.parameters.set_mode(mode)

    def get_assembly_variables(self) -> List[Dict[str, float]]:
        return self.parameters.assembly_specs()

    def set_assembly_variable(self, label: str, value: float):
        self.parameters.set_assembly_value(label, value)

    def get_assembly_values(self) -> Dict[str, float]:
        return self.parameters.assembly_values()

    def snapshot_pattern(self) -> Dict[str, float]:
        return self.pattern.snapshot()

    def restore_pattern(self, state: Dict[str, float]):
        self.pattern.restore(state)

    def get_segments(self, assembly_offset: float | None = None, space: float = 0.1) -> Dict[str, List[Tuple[float, float]]]:
        return self.pattern.GetSegments(assembly_offset=assembly_offset, space=space)

    def get_shape_area(self, offset: float, space: float) -> float:
        return self.pattern.GetShapeArea(offset=offset, space=space)

    def get_rectangle_area(self, offset: float, space: float) -> float:
        return self.pattern.GetRectangleArea(offset=offset, space=space)

    def get_equivalent_coefficient(self, offset: float, space: float) -> float:
        return self.pattern.GetEquivalentCoefficient(offset=offset, space=space)

    def get_resistance(self, offset: float, space: float, thick: float | None = None, rho: float = 1.724e-8) -> float:
        thickness = thick if thick is not None else self.step_exporter.thickness
        return self.pattern.GetResistance(offset=offset, space=space, thick=thickness, rho=rho)

    def get_symmetric_curve_area(self) -> float:
        return self.pattern.GetSymmetricCurveArea()

    # ------------------------------------------------------------------ #
    # 2D helpers
    # ------------------------------------------------------------------ #
    def _mirror_shape(self, shape: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
        center_x = self.pattern.width / 2.0
        return [(2 * center_x - x, y) for x, y in shape]

    def _apply_twist(self, shape_curve: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not self.assembly.twist_enabled or not shape_curve:
            return shape_curve

        y_center = self.pattern.height / 2.0
        x_center = self.assembly.coil_width / 2.0
        eps = 1e-9

        def add_unique(points: List[Tuple[float, float]], point: Tuple[float, float]):
            if not points:
                points.append(point)
                return
            last_x, last_y = points[-1]
            if abs(last_x - point[0]) > eps or abs(last_y - point[1]) > eps:
                points.append(point)

        num_points = len(shape_curve)
        upper_half: List[Tuple[float, float]] = []
        for idx in range(num_points):
            x1, y1 = shape_curve[idx]
            x2, y2 = shape_curve[(idx + 1) % num_points]

            if y1 >= y_center:
                add_unique(upper_half, (x1, y1))

            delta1 = y1 - y_center
            delta2 = y2 - y_center
            if delta1 * delta2 < 0.0:
                t = (y_center - y1) / (y2 - y1)
                x_cross = x1 + t * (x2 - x1)
                add_unique(upper_half, (round(x_cross, 15), round(y_center, 15)))

        if len(upper_half) < 2:
            return shape_curve

        rotated_half = [
            (round(2 * x_center - x, 15), round(2 * y_center - y, 15))
            for x, y in upper_half
        ]

        twisted_curve = list(upper_half)
        for point in rotated_half:
            add_unique(twisted_curve, point)

        first_x, first_y = twisted_curve[0]
        last_x, last_y = twisted_curve[-1]
        if abs(first_x - last_x) > eps or abs(first_y - last_y) > eps:
            twisted_curve.append((first_x, first_y))
        
        def mirror_horizontal(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            return [(round(2 * x_center - x, 15), y) for x, y in points]
        
        mirrored_curve = mirror_horizontal(twisted_curve)
        
        return mirrored_curve

    def _compute_base_shape(self) -> List[Tuple[float, float]]:
        shape = self.pattern.GetShape(
            offset=self.assembly.offset,
            space=self.assembly.spacing,
        )
        if not shape:
            return []
        return shape

    def build_2d_instances(self) -> List[Assembly2DInstance]:
        base_shape = self._compute_base_shape()
        if not base_shape:
            return []

        twisted_shape = self._apply_twist(base_shape)
        twist_shapes = {
            False: base_shape,
            True: twisted_shape,
        }
        mirrored_cache = {
            key: self._mirror_shape(shape) if shape else []
            for key, shape in twist_shapes.items()
        }

        instances: List[Assembly2DInstance] = []
        total_instances = max(0, int(self.assembly.count))
        for index, offset in enumerate(self.assembly.iter_offsets()):
            twist_left = self.assembly.should_twist_left(index, total_instances)
            twist_right = self.assembly.should_twist_right(index, total_instances)
            left_variant = twist_shapes[True] if twist_left else twist_shapes[False]
            right_variant = mirrored_cache[True] if twist_right else mirrored_cache[False]
            left_shape = [(x + offset, y) for x, y in left_variant]
            right_shape = [(x + offset, y) for x, y in right_variant]
            instances.append(Assembly2DInstance(
                index=index,
                offset=offset,
                left_shape=left_shape,
                right_shape=right_shape,
            ))
        return instances

    # ------------------------------------------------------------------ #
    # 3D helpers
    # ------------------------------------------------------------------ #
    def _prepare_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        base_curve = self._compute_base_shape()
        if len(base_curve) < 3:
            return {'left': [], 'right': [], 'envelope': []}
        use_twist_left = self.assembly.should_twist_left(0)
        use_twist_right = self.assembly.should_twist_right(0)
        left_curve = self._apply_twist(base_curve) if use_twist_left else base_curve
        right_source = self._apply_twist(base_curve) if use_twist_right else base_curve
        right_curve = self._mirror_shape(right_source)
        envelope = self.pattern.GetSymmetricEnvelope()
        return {'left': left_curve, 'right': right_curve, 'envelope': envelope}

    def get_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Public accessor for the twisted base curve and its mirrored counterpart.
        """
        return self._prepare_curves()

    def build_flat_solids(self) -> Dict[str, List[AssemblySolidInstance]]:
        thickness = self.step_exporter.thickness
        base_curve = self._compute_base_shape()
        if len(base_curve) < 3:
            return {'left': [], 'right': []}

        twisted_curve = self._apply_twist(base_curve)
        variant_curves = {
            False: base_curve,
            True: twisted_curve,
        }
        mirrored_variants = {
            flag: self._mirror_shape(curve) if curve else []
            for flag, curve in variant_curves.items()
        }

        left_cache: Dict[bool, Optional[Any]] = {False: None, True: None}
        right_cache: Dict[bool, Optional[Any]] = {False: None, True: None}

        for flag in (False, True):
            curve_left = variant_curves[flag]
            if curve_left:
                try:
                    left_cache[flag] = self.step_exporter.create_shape_from_curve(curve_left, z_offset=thickness)
                except Exception as err:
                    message = f"{type(err).__name__}: {err}"
                    return {
                        'left': [AssemblySolidInstance(index=0, offset=0.0, shape=None, error=message)],
                        'right': []
                    }
            curve_right = mirrored_variants[flag]
            if curve_right:
                try:
                    right_cache[flag] = self.step_exporter.create_shape_from_curve(curve_right, z_offset=0.0)
                except Exception as err:
                    message = f"{type(err).__name__}: {err}"
                    return {
                        'left': [],
                        'right': [AssemblySolidInstance(index=0, offset=0.0, shape=None, error=message)],
                    }

        total_instances = max(0, int(self.assembly.count))
        results_left: List[AssemblySolidInstance] = []
        results_right: List[AssemblySolidInstance] = []

        for index, offset in enumerate(self.assembly.iter_offsets()):
            twist_left = self.assembly.should_twist_left(index, total_instances)
            twist_right = self.assembly.should_twist_right(index, total_instances)
            base_left = left_cache.get(True if twist_left else False)
            base_right = right_cache.get(True if twist_right else False)
            if base_left:
                left_shape = self.step_exporter.translate_shape(base_left, offset, 0.0, 0.0)
                results_left.append(AssemblySolidInstance(index=index, offset=offset, shape=left_shape))
            if base_right:
                right_shape = self.step_exporter.translate_shape(base_right, offset, 0.0, 0.0)
                results_right.append(AssemblySolidInstance(index=index, offset=offset, shape=right_shape))

        return {'left': results_left, 'right': results_right}
