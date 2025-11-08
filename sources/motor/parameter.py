from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from pattern import Pattern

__all__ = ("Parameter", "AssemblyParameters")


@dataclass
class AssemblyParameters:
    """2D array spacing and extrusion controls."""
    count: int = 18
    spacing: float = 0.05
    twist_enabled: bool = False
    layer_thickness: float = 0.047
    coil_width: float = 0.544
    offset: float = coil_width + spacing

    def update_offset_from_coil(self) -> None:
        """Keep offset aligned with current coil width and spacing."""
        self.offset = max(self.coil_width + self.spacing, 1e-6)

    def iter_offsets(self) -> Iterable[float]:
        for idx in range(max(0, int(self.count))):
            yield idx * self.offset


class Parameter:
    """Centralised parameter controller used by UI and geometry layers."""

    _PATTERN_LABEL_MAP: Dict[str, str] = {
        "width": "w",
        "height": "h",
        "vbh": "vb",
        "corner_bottom": "cb",
        "corner_top": "ct",
        "exponent": "epn",
        "exponent_m": "epm",
    }

    _ASSEMBLY_LABEL_MAP: Dict[str, str] = {"count": "cnt"}

    def __init__(self, pattern: Pattern, assembly: AssemblyParameters):
        self._pattern = pattern
        self._assembly = assembly

        # Reverse lookups for setters
        self._pattern_reverse = {v: k for k, v in self._PATTERN_LABEL_MAP.items()}
        self._assembly_reverse = {v: k for k, v in self._ASSEMBLY_LABEL_MAP.items()}

    @staticmethod
    def _fmt(value: float) -> float:
        return round(float(value), 5)

    def pattern_specs(self) -> List[Dict[str, float]]:
        """Return slider metadata (all params with 0.001 steps, 5dp values)."""
        mode = self._pattern.get_mode()
        self._pattern.corner_margin = max(0.0, self._assembly.spacing * 2.0)
        enabled_map = {
            "vbh": mode == "A",
            "corner_bottom": mode == "B",
            "corner_top": mode == "B",
        }

        # Cache current pattern-provided ranges for active variables.
        var_data = {item["label"]: item for item in self._pattern.GetVariables()}

        attr_lookup = {
            "width": "width",
            "height": "height",
            "vbh": "vbh",
            "corner_bottom": "corner_bottom_value",
            "corner_top": "corner_top_value",
            "exponent": "exponent",
            "exponent_m": "exponent_m",
        }

        specs: List[Dict[str, float]] = []
        for original, short in self._PATTERN_LABEL_MAP.items():
            item = var_data.get(original)
            value = getattr(self._pattern, attr_lookup[original])
            if item is not None:
                min_val = item.get("min", value)
                max_val = item.get("max", value)
            else:
                min_val = value
                max_val = value

            if original in ("corner_bottom", "corner_top"):
                max_val = max_val + self._assembly.spacing * 2.0

            specs.append({
                "label": short,
                "value": self._fmt(value),
                "min": self._fmt(min_val),
                "max": self._fmt(max_val),
                "step": 0.001,
                "enabled": enabled_map.get(original, True),
            })

        return specs

    def pattern_values(self) -> Dict[str, float]:
        return {spec["label"]: spec["value"] for spec in self.pattern_specs()}

    def set_pattern_value(self, label: str, value: float) -> None:
        self._pattern.corner_margin = max(0.0, self._assembly.spacing * 2.0)
        original = self._pattern_reverse.get(label)
        if original is None:
            if label in self._PATTERN_LABEL_MAP:
                original = label
            else:
                raise ValueError(f"Unknown pattern parameter: {label}")
        self._pattern.SetVariable(original, self._fmt(value))

    def set_mode(self, mode: str):
        self._pattern.corner_margin = max(0.0, self._assembly.spacing * 2.0)
        self._pattern.set_mode(mode)

    def get_mode(self) -> str:
        return self._pattern.get_mode()

    def assembly_specs(self) -> List[Dict[str, float]]:
        count_max = 96
        return [{
            "label": self._ASSEMBLY_LABEL_MAP["count"],
            "value": float(self._assembly.count),
            "min": 1.0,
            "max": float(count_max),
            "step": 1.0,
        }]

    def assembly_values(self) -> Dict[str, float]:
        return {self._ASSEMBLY_LABEL_MAP["count"]: float(self._assembly.count)}

    def set_assembly_value(self, label: str, value: float) -> None:
        original = self._assembly_reverse.get(label)
        if not original:
            raise ValueError(f"Unknown assembly parameter: {label}")
        if original == "count":
            self._assembly.count = max(1, int(round(value)))
            return
        raise ValueError(f"Unsupported assembly parameter: {label}")

    @property
    def pattern(self) -> Pattern:
        return self._pattern

    @property
    def assembly(self) -> AssemblyParameters:
        return self._assembly
