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
    no_twist_prefix: int = 0
    no_twist_suffix: int = 0
    no_twist_left_prefix: int = 0
    no_twist_left_suffix: int = 0
    no_twist_right_prefix: int = 0
    no_twist_right_suffix: int = 0

    def update_offset_from_coil(self) -> None:
        """Keep offset aligned with current coil width and spacing."""
        self.offset = max(self.coil_width + self.spacing, 1e-6)

    def iter_offsets(self) -> Iterable[float]:
        for idx in range(max(0, int(self.count))):
            yield idx * self.offset

    def should_twist(self, index: int, total_count: int | None = None) -> bool:
        """Determine if a given instance index should apply twist."""
        return self.should_twist_left(index, total_count)

    def should_twist_left(self, index: int, total_count: int | None = None) -> bool:
        if not self.twist_enabled:
            return False
        total = total_count if total_count is not None else max(0, int(self.count))
        if total <= 0:
            return False
        if index < max(0, self.no_twist_prefix):
            return False
        suffix_limit = max(0, self.no_twist_suffix)
        if suffix_limit > 0 and index >= max(0, total - suffix_limit):
            return False
        if index < max(0, self.no_twist_left_prefix):
            return False
        left_suffix = max(0, self.no_twist_left_suffix)
        if left_suffix > 0 and index >= max(0, total - left_suffix):
            return False
        return True

    def should_twist_right(self, index: int, total_count: int | None = None) -> bool:
        if not self.twist_enabled:
            return False
        total = total_count if total_count is not None else max(0, int(self.count))
        if total <= 0:
            return False
        prefix_limit = max(0, self.no_twist_right_prefix)
        if prefix_limit > 0 and index < prefix_limit:
            return False
        suffix_limit = max(0, self.no_twist_right_suffix)
        if suffix_limit > 0 and index >= max(0, total - suffix_limit):
            return False
        return True


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
            range_source = item
            if original == "corner_top" and mode == "A":
                value = self._pattern.vlw
                range_source = var_data.get("vlw_top") or range_source
            elif original == "corner_bottom" and mode == "A":
                value = self._pattern.vlw_bottom
                range_source = var_data.get("vlw_bottom") or range_source
            if range_source is not None:
                min_val = range_source.get("min", value)
                max_val = range_source.get("max", value)
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
        mode = self._pattern.get_mode()
        if label in {"ct", "cb"} and mode == "A":
            target = "vlw_top" if label == "ct" else "vlw_bottom"
            self._pattern.SetVariable(target, self._fmt(value))
            return
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
