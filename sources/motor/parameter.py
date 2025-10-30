from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


class PatternParameters:
    """
    Manage all user-controllable parameters that describe the 2D pattern bounding box
    and superellipse corner radii. This class is deliberately independent from any
    geometry generation so it can be reused by both pattern builders and visualisers.
    """

    _EXP_TOLERANCE = 1e-6

    def __init__(self, width: float = 1.0, height: float = 1.0, exponent: float = 0.80):
        self.width = 0.0
        self.height = 0.0
        self.exponent = 2.0  # placeholder until reset

        self.vbh = 0.0  # vertical bottom straight segment
        self.vth = 0.0  # vertical top radius portion
        self.vlw = 0.0  # horizontal left radius portion
        self.vrw = 0.0  # horizontal right straight segment
        self.corner_value = 0.0  # slider value for Mode B

        self.reset(width=width, height=height, exponent=exponent)

    # ------------------------------------------------------------------ #
    # Initialisation & constraints
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _is_mode_a(self, exponent: float | None = None) -> bool:
        exp = self.exponent if exponent is None else exponent
        return abs(exp - 2.0) <= self._EXP_TOLERANCE

    def reset(self,
            width: float | None = None,
            height: float | None = None,
            exponent: float | None = None):
        """
        Reset the parameter set with optional new dimensions and exponent.
        Derived values are recomputed via the same logic used by the original Pattern class.
        """
        if width is not None:
            self.width = max(0.0, float(width))
        if height is not None:
            self.height = max(0.0, float(height))
        if exponent is not None:
            self.exponent = self._clamp(float(exponent), 0.5, 2.0)

        half_width = self.width / 2.0
        half_height = self.height / 2.0
        initial_corner = min(half_width, half_height) * 0.5

        self.vlw = initial_corner
        self.vth = initial_corner
        self.vrw = max(0.0, half_width - self.vlw)
        self.vbh = max(0.0, half_height - self.vth)
        self.corner_value = initial_corner

        self._apply_constraints()

    def _apply_constraints(self):
        """
        Apply mode-dependent constraints to keep the parameters self-consistent.
        Mode A (exponent == 2): user controls vbh/vlw, other values derived.
        Mode B (exponent < 2): a unified corner slider controls vth/vlw.
        """
        half_width = max(0.0, self.width / 2.0)
        half_height = max(0.0, self.height / 2.0)

        if self._is_mode_a():
            # Clamp straight segments
            vlw_max = max(0.0, half_width - 0.05)
            self.vbh = self._clamp(self.vbh, 0.0, half_height)
            self.vlw = self._clamp(self.vlw, 0.0, vlw_max)

            self.vth = max(0.0, half_height - self.vbh)
            self.vrw = max(0.0, half_width - self.vlw)

            self.corner_value = min(self.vth, self.vlw)
        else:
            max_corner = min(half_width, half_height)
            self.corner_value = self._clamp(self.corner_value, 0.0, max_corner)

            self.vth = self.corner_value
            self.vlw = self.corner_value
            self.vbh = max(0.0, half_height - self.vth)
            self.vrw = max(0.0, half_width - self.vlw)

    # ------------------------------------------------------------------ #
    # Inspection helpers
    # ------------------------------------------------------------------ #
    def get_mode(self) -> str:
        return 'A' if self._is_mode_a() else 'B'

    def snapshot(self) -> Dict[str, float]:
        """Return a shallow copy of the current parameter state."""
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
        """Restore state produced by snapshot()."""
        if not state:
            return
        self.width = max(0.0, float(state.get('width', self.width)))
        self.height = max(0.0, float(state.get('height', self.height)))
        self.exponent = self._clamp(float(state.get('exponent', self.exponent)), 0.5, 2.0)
        self.vbh = max(0.0, float(state.get('vbh', self.vbh)))
        self.vlw = max(0.0, float(state.get('vlw', self.vlw)))
        self.corner_value = max(0.0, float(state.get('corner_value', self.corner_value)))
        self._apply_constraints()

    def get_variables(self) -> List[Dict[str, float]]:
        """
        Provide metadata for UI sliders/inputs. Mirrors the original Pattern.GetVariables().
        """
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

    def set_value(self, label: str, value: float):
        """Update a parameter value and reapply constraints."""
        if label == 'width':
            self.width = max(0.0, float(value))
        elif label == 'height':
            self.height = max(0.0, float(value))
        elif label == 'vbh':
            self.vbh = max(0.0, float(value))
        elif label == 'vlw':
            self.vlw = max(0.0, float(value))
        elif label == 'corner':
            self.corner_value = max(0.0, float(value))
        elif label == 'exponent':
            prev_mode_a = self._is_mode_a()
            value_clamped = self._clamp(float(value), 0.5, 2.0)
            self.exponent = value_clamped

            if prev_mode_a and not self._is_mode_a():
                self.corner_value = min(self.vth, self.vlw)
            elif not prev_mode_a and self._is_mode_a():
                half_height = max(0.0, self.height / 2.0)
                half_width = max(0.0, self.width / 2.0)
                self.vbh = self._clamp(self.vbh, 0.0, half_height)
                self.vlw = self._clamp(self.vlw, 0.0, half_width)
        else:
            raise ValueError(f"Unknown parameter label: {label}")

        self._apply_constraints()

    # ------------------------------------------------------------------ #
    # Bounding box helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fmt_point(x: float, y: float) -> Tuple[float, float]:
        return (round(x, 15), round(y, 15))

    def get_bbox(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Return basic bounding-box related control geometry used by the UI.
        """
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


@dataclass
class SpiralParameters:
    """Parameters governing the logarithmic spiral used for 3D coil placement."""
    radius: float = 6.2055
    thickness: float = 0.1315
    turns: int = 6
    samples_per_turn: int = 1500


@dataclass
class AssemblyParameters:
    """
    Parameters controlling how 2D patterns are arrayed before mapping into 3D.
    """
    count: int = 18
    spacing: float = 0.06
    twist_enabled: bool = False
    layer_thickness: float = 0.047
    coil_width: float = 0.544
    offset: float = coil_width + spacing

    def update_offset_from_coil(self):
        """
        Synchronise the array offset with the coil width plus an optional clearance.
        """
        self.offset = max(self.coil_width + self.spacing, 1e-6)

    def iter_offsets(self) -> Iterable[float]:
        for idx in range(max(0, int(self.count))):
            yield idx * self.offset
