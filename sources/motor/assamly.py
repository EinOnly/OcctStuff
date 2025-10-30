from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from parameter import AssemblyParameters, SpiralParameters
from pattern import Pattern
from curve import SpiralMapper
from step import StepExporter, SpiralSurfaceBuilder


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
    Coordinate pattern geometry, STEP extrusion and spiral mapping so that the UI layer
    only needs to format or display the returned data.
    """

    def __init__(self,
        pattern: Pattern | None = None,
        assembly: AssemblyParameters | None = None,
        spiral: SpiralParameters | None = None,
        step_exporter: StepExporter | None = None):
        
        self.pattern = pattern or Pattern()
        self.assembly = assembly or AssemblyParameters()
        self.spiral = spiral or SpiralParameters()
        self.step_exporter = step_exporter or StepExporter(thickness=self.assembly.layer_thickness)

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
        return self.pattern.GetVariables()

    def get_pattern_mode(self) -> str:
        return self.pattern.get_mode()

    def set_pattern_variable(self, label: str, value: float):
        self.pattern.SetVariable(label, value)

    def get_pattern_values(self) -> Dict[str, float]:
        return {
            'width': self.pattern.width,
            'height': self.pattern.height,
            'vbh': self.pattern.vbh,
            'vlw': self.pattern.vlw,
            'corner': self.pattern.vth,
            'exponent': self.pattern.exponent,
        }

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

    def get_resistance(self, offset: float, space: float, thick: float = 0.047, rho: float = 1.724e-8) -> float:
        return self.pattern.GetResistance(offset=offset, space=space, thick=thick, rho=rho)

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
        return twisted_curve

    def _compute_base_shape(self) -> List[Tuple[float, float]]:
        shape = self.pattern.GetShape(
            offset=self.assembly.offset,
            space=self.assembly.spacing,
        )
        if not shape:
            return []
        return self._apply_twist(shape)

    def build_2d_instances(self) -> List[Assembly2DInstance]:
        base_shape = self._compute_base_shape()
        if not base_shape:
            return []

        mirrored = self._mirror_shape(base_shape)
        instances: List[Assembly2DInstance] = []
        for index, offset in enumerate(self.assembly.iter_offsets()):
            left_shape = [(x + offset, y) for x, y in base_shape]
            right_shape = [(x + offset, y) for x, y in mirrored]
            instances.append(Assembly2DInstance(index=index, offset=offset,
                                                left_shape=left_shape, right_shape=right_shape))
        return instances

    # ------------------------------------------------------------------ #
    # 3D helpers
    # ------------------------------------------------------------------ #
    def _prepare_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        left_curve = self._compute_base_shape()
        if len(left_curve) < 3:
            return {'left': [], 'right': []}
        right_curve = self._mirror_shape(left_curve)
        return {'left': left_curve, 'right': right_curve}

    def get_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Public accessor for the twisted base curve and its mirrored counterpart.
        """
        return self._prepare_curves()

    def build_flat_solids(self) -> Dict[str, List[AssemblySolidInstance]]:
        curves = self._prepare_curves()
        left_curve = curves['left']
        right_curve = curves['right']
        if not left_curve or not right_curve:
            return {'left': [], 'right': []}

        thickness = self.step_exporter.thickness
        results_left: List[AssemblySolidInstance] = []
        results_right: List[AssemblySolidInstance] = []

        try:
            base_left = self.step_exporter.create_shape_from_curve(left_curve, z_offset=thickness)
        except Exception as err:
            message = f"{type(err).__name__}: {err}"
            return {
                'left': [AssemblySolidInstance(index=0, offset=0.0, shape=None, error=message)],
                'right': []
            }

        try:
            base_right = self.step_exporter.create_shape_from_curve(right_curve, z_offset=0.0)
        except Exception as err:
            message = f"{type(err).__name__}: {err}"
            return {
                'left': [],
                'right': [AssemblySolidInstance(index=0, offset=0.0, shape=None, error=message)],
            }

        for index, offset in enumerate(self.assembly.iter_offsets()):
            left_shape = self.step_exporter.translate_shape(base_left, offset, 0.0, 0.0)
            right_shape = self.step_exporter.translate_shape(base_right, offset, 0.0, 0.0)
            results_left.append(AssemblySolidInstance(index=index, offset=offset, shape=left_shape))
            results_right.append(AssemblySolidInstance(index=index, offset=offset, shape=right_shape))

        return {'left': results_left, 'right': results_right}

    def build_spiral_solids(self,
                            radius_override: float | None = None,
                            thickness_override: float | None = None,
                            turns_override: int | None = None,
                            samples_override: int | None = None) -> Dict[str, List[AssemblySolidInstance]]:
        curves = self._prepare_curves()
        left_curve = curves['left']
        right_curve = curves['right']
        if not left_curve or not right_curve:
            return {'left': [], 'right': [], 'length_warning': False}

        center_x = self.pattern.width / 2.0
        thickness = self.step_exporter.thickness

        curve_min_x = min(p[0] for p in left_curve)
        curve_max_x = max(p[0] for p in left_curve)
        curve_span = max(0.0, curve_max_x - curve_min_x)

        mirror_min_x = min(p[0] for p in right_curve)
        mirror_max_x = max(p[0] for p in right_curve)
        mirror_span = max(0.0, mirror_max_x - mirror_min_x)

        max_u_required = 0.0
        for offset in self.assembly.iter_offsets():
            max_u_required = max(max_u_required,
                                 offset + curve_span,
                                 offset + mirror_span)
        if max_u_required <= 0.0:
            return {'left': [], 'right': [], 'length_warning': False}

        mapper = SpiralMapper(
            parameters=self.spiral,
            radius_value=radius_override,
            thick_value=thickness_override,
            turns=turns_override,
            samples_per_turn=samples_override
        )

        total_length = mapper.get_total_length()
        if total_length <= 0.0:
            return {'left': [], 'right': [], 'length_warning': False}
        length_warning = max_u_required > total_length
        effective_length = min(max_u_required, total_length)

        builder = SpiralSurfaceBuilder(
            mapper=mapper,
            max_length=effective_length,
            pattern_thickness=thickness
        )

        left_results: List[AssemblySolidInstance] = []
        right_results: List[AssemblySolidInstance] = []

        for index, offset in enumerate(self.assembly.iter_offsets()):
            try:
                left_solid = builder.create_thick_solid(
                    left_curve,
                    thickness,
                    x_offset=offset,
                    x_origin=curve_min_x,
                    layers=[('outer', 'middle')],
                )
                left_solid = self.step_exporter.translate_shape(left_solid, offset, 0.0, 0.0)
                left_results.append(AssemblySolidInstance(index=index, offset=offset, shape=left_solid))
            except Exception as err:
                message = f"{type(err).__name__}: {err}"
                left_results.append(AssemblySolidInstance(index=index, offset=offset, shape=None, error=message))

            try:
                right_solid = builder.create_thick_solid(
                    right_curve,
                    thickness,
                    x_offset=offset,
                    x_origin=curve_min_x,
                    x_transform=lambda xv: 2 * center_x - xv,
                    layers=[('middle', 'inner')],
                )
                right_solid = self.step_exporter.translate_shape(right_solid, offset, 0.0, -thickness)
                right_results.append(AssemblySolidInstance(index=index, offset=offset, shape=right_solid))
            except Exception as err:
                message = f"{type(err).__name__}: {err}"
                right_results.append(AssemblySolidInstance(index=index, offset=offset, shape=None, error=message))

        return {'left': left_results, 'right': right_results, 'length_warning': length_warning}
