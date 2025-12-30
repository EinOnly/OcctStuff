# test_params_ui.py
# PParams + LParams with constraints, registrations, and a single test window

import sys
from typing import Callable, Optional, Dict, Any, Set, Tuple

from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSlider, QLineEdit, QComboBox, QGroupBox, QSizePolicy, QCheckBox, QPushButton
)


# ------------------------------
# Helpers
# ------------------------------

def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return v


# ------------------------------
# PParams: local pattern parameters with constraints
# ------------------------------

class PParams(QObject):
    changed = pyqtSignal(str, object)
    bulkChanged = pyqtSignal(dict)

    _RESEED_KEYS = {"pattern_pbw", "pattern_pbh"}

    def __init__(self):
        super().__init__()
        self._v: Dict[str, Any] = {
            # top parameters
            "pattern_tp0": 0.00,
            "pattern_tp1": 1.25,
            "pattern_tp2": 0.00,
            "pattern_tp3": 2.00,
            "pattern_tnn": 2.00,
            "pattern_tmm": 2.00,
            "pattern_tcc": 2.50,

            # bottom parameters
            "pattern_bp0": 0.00,
            "pattern_bp1": 1.25,
            "pattern_bp2": 0.00,
            "pattern_bp3": 2.00,
            "pattern_bnn": 2.00,
            "pattern_bmm": 2.00,
            "pattern_bcc": 2.50,

            # bounding box (width/height) — renamed: pbw/pbh
            "pattern_pbh": 7.5,    # height
            "pattern_pbw": 5.0,    # width

            # per-wave width
            "pattern_ppw": 0.5,
            "pattern_psp": 0.05,  # spacing

            # mode / twist
            "pattern_mode": "straight",     # straight | superelliptic
            "pattern_type": "wave",         # wave | lap
            "pattern_twist": False,
            "pattern_symmetry": True,
        }

        self._slider: Dict[str, Dict[str, Any]] = {}
        self._input: Dict[str, Dict[str, Any]] = {}
        self._block = False
        self._apply_constraints(None)

    # ----- accessors -----
    def get(self, key, default=None):
        return self._v.get(key, default)

    def snapshot(self):
        return dict(self._v)

    # ----- UI registration -----
    def sliderRegister(
        self,
        key: str,
        slider: QSlider,
        *,
        vmin: Optional[Callable[[], float]] = None,
        vmax: Optional[Callable[[], float]] = None,
        scale: float = 1.0
    ):
        slider.setMinimum(0)
        slider.setMaximum(1000)

        self._slider[key] = {
            "widget": slider,
            "vmin": vmin,
            "vmax": vmax,
            "scale": scale,
        }

        self._update_slider_range(key)
        self._push_slider_value(key, self.get(key))
        slider.valueChanged.connect(lambda iv: self._on_slider_changed(key, iv))

    def inputRegister(self, key: str, line: QLineEdit, *, fmt=None, parse=None):
        if fmt is None:
            fmt = lambda v: f"{v}"
        if parse is None:
            parse = lambda s: float(s)

        self._input[key] = {"widget": line, "fmt": fmt, "parse": parse}
        self._push_input_value(key, self.get(key))
        line.editingFinished.connect(lambda: self._on_input_finished(key))

    # ----- UI events -----
    def _on_slider_changed(self, key: str, iv: int):
        reg = self._slider[key]
        vmin = reg["vmin"]() if callable(reg["vmin"]) else 0.0
        vmax = reg["vmax"]() if callable(reg["vmax"]) else 1.0
        scale = reg["scale"]

        ratio = iv / 1000.0
        raw = vmin + (vmax - vmin) * ratio
        if scale > 0:
            steps = round(raw / scale)
            val = steps * scale
        else:
            val = raw
        self.set(key, val)

    def _on_input_finished(self, key: str):
        reg = self._input[key]
        try:
            val = reg["parse"](reg["widget"].text())
        except Exception:
            return
        self.set(key, val)

    # ----- UI push -----
    def _update_slider_range(self, key: str):
        if key not in self._slider:
            return
        reg = self._slider[key]
        vmin = reg["vmin"]() if callable(reg["vmin"]) else 0.0
        vmax = reg["vmax"]() if callable(reg["vmax"]) else 1.0
        if vmax < vmin:
            vmax = vmin
        reg["last_min"] = vmin
        reg["last_max"] = vmax
        w: QSlider = reg["widget"]
        w.setToolTip(f"{key}: [{vmin}, {vmax}]")

    def _push_slider_value(self, key: str, value: float):
        if key not in self._slider:
            return
        reg = self._slider[key]
        w: QSlider = reg["widget"]
        vmin = reg.get("last_min", 0.0)
        vmax = reg.get("last_max", 1.0)
        scale = reg["scale"]
        if vmax == vmin:
            pos = 0
        else:
            value = clamp(value, vmin, vmax)
            pos = int(round((value - vmin) / (vmax - vmin) * 1000.0))
        w.blockSignals(True)
        w.setValue(pos)
        w.blockSignals(False)
        w.setToolTip(f"{key}: {value:.6f} (range {vmin:.6f}..{vmax:.6f}, step {scale})")

    def _push_input_value(self, key: str, value: Any):
        if key not in self._input:
            return
        reg = self._input[key]
        w: QLineEdit = reg["widget"]
        fmt = reg["fmt"]
        w.blockSignals(True)
        w.setText(fmt(value))
        w.blockSignals(False)

    # ----- core set with constraints -----
    def set(self, key: str, value: Any, emit=True):
        if self._block:
            return
        self._block = True

        self._v[key] = value
        reseed_key = None if key in self._RESEED_KEYS else key
        self._apply_constraints(reseed_key)
        self._reflect_ui(None)

        self._block = False
        if emit:
            self.changed.emit(key, self._v[key])

    def update_bulk(self, kv: dict, emit=True):
        if self._block:
            return
        self._block = True
        for k, v in kv.items():
            self._v[k] = v
            self._apply_constraints(None if k in self._RESEED_KEYS else k)
        self._reflect_ui(None)
        self._block = False
        if emit:
            self.bulkChanged.emit(self.snapshot())

    # ----- constraints -----
    def _apply_constraints(self, key_changed: str):
        v = self._v
        pbw = float(v["pattern_pbw"])   # width
        pbh = float(v["pattern_pbh"])   # height
        ppw = float(v["pattern_ppw"])
        reseed = key_changed in (None, "pattern_pbw", "pattern_pbh")

        half_w = pbw / 2.0
        half_h = pbh / 2.0

        if v["pattern_mode"] == "superelliptic":
            tp0_min = min(half_w, half_h) - max(half_w, half_h)
            tp0_max = half_w - ppw
        else:
            tp0_min = -max(0.0, pbw) * 0.25
            tp0_max = max(0.0, pbw * 0.5 - ppw)
        tp3_min, tp3_max = 0.0, max(0.0, pbh * 0.5 - ppw)
        bp0_min, bp0_max = tp0_min, tp0_max
        bp3_min, bp3_max = tp3_min, tp3_max

        # centers clamp to [ppw, pbw], default to pbw/2 when pbw changes from 0
        center_min = min(ppw, pbw)
        default_center = pbw / 2.0 if pbw > 0 else center_min
        v["pattern_tcc"] = clamp(
            v["pattern_tcc"] if key_changed != "pattern_pbw" or v["pattern_tcc"] != 0.0 else default_center,
            center_min,
            pbw
        )
        v["pattern_bcc"] = clamp(
            v["pattern_bcc"] if key_changed != "pattern_pbw" or v["pattern_bcc"] != 0.0 else default_center,
            center_min,
            pbw
        )

        # enforce primary ranges
        v["pattern_tp0"] = clamp(float(v["pattern_tp0"]), tp0_min, tp0_max)
        v["pattern_tp3"] = clamp(float(v["pattern_tp3"]), tp3_min, tp3_max)
        v["pattern_bp0"] = clamp(float(v["pattern_bp0"]), bp0_min, bp0_max)
        v["pattern_bp3"] = clamp(float(v["pattern_bp3"]), bp3_min, bp3_max)

        # sums coupling: p0<->p1, p2<->p3 for top/bottom
        half_w = pbw / 2.0
        half_h = pbh / 2.0

        def couple(a_key, b_key, target_sum, prefer_key):
            a = float(v[a_key]); b = float(v[b_key])
            if key_changed == prefer_key or reseed:
                v[b_key] = target_sum - a
            else:
                v[a_key] = target_sum - b

        # Sync bp3 and bp2 to tp3 and tp2 BEFORE coupling
        if reseed:
            v["pattern_bp3"] = clamp(float(v["pattern_tp3"]), bp3_min, bp3_max)
            v["pattern_bp2"] = clamp(float(v["pattern_tp2"]), 0.0, half_h)

        couple("pattern_tp0", "pattern_tp1", half_w, "pattern_tp0")
        couple("pattern_tp3", "pattern_tp2", half_h, "pattern_tp3")
        couple("pattern_bp0", "pattern_bp1", half_w, "pattern_bp0")
        couple("pattern_bp3", "pattern_bp2", half_h, "pattern_bp3")

        # mode-specific constraints
        mode = v["pattern_mode"]
        if mode == "straight":
            v["pattern_tnn"] = 2.0
            v["pattern_tmm"] = 2.0
            v["pattern_bnn"] = 2.0
            v["pattern_bmm"] = 2.0
            shared_src = "bottom" if key_changed in ("pattern_bp2", "pattern_bp3") else "top"
            if reseed:
                shared_src = "top"
            if shared_src == "bottom":
                shared = clamp(float(v["pattern_bp2"]), 0.0, half_h)
            else:
                shared = clamp(float(v["pattern_tp2"]), 0.0, half_h)
            v["pattern_tp2"] = shared
            v["pattern_bp2"] = shared
            v["pattern_tp3"] = clamp(half_h - shared, tp3_min, tp3_max)
            v["pattern_bp3"] = clamp(half_h - shared, bp3_min, bp3_max)
        elif mode == "superelliptic":
            for k in ("pattern_tnn", "pattern_tmm", "pattern_bnn", "pattern_bmm"):
                v[k] = clamp(float(v[k]), 0.3, 2.0)
            self._enforce_superelliptic_symmetry(
                v,
                half_w=half_w,
                half_h=half_h,
                tp0_bounds=(tp0_min, tp0_max),
                tp3_bounds=(tp3_min, tp3_max),
                bp0_bounds=(bp0_min, bp0_max),
                bp3_bounds=(bp3_min, bp3_max),
                reseed=reseed,
            )

        if v.get("pattern_symmetry", False):
            shared = None
            if key_changed in ("pattern_tp0", "pattern_bp0"):
                shared = float(v[key_changed])
            else:
                shared = (float(v["pattern_tp0"]) + float(v["pattern_bp0"])) * 0.5
            shared = clamp(shared, tp0_min, tp0_max)
            v["pattern_tp0"] = shared
            v["pattern_bp0"] = clamp(shared, bp0_min, bp0_max)
            v["pattern_tp1"] = half_w - v["pattern_tp0"]
            v["pattern_bp1"] = half_w - v["pattern_bp0"]

        # re-clamp after sums
        v["pattern_tp0"] = clamp(float(v["pattern_tp0"]), tp0_min, tp0_max)
        v["pattern_tp3"] = clamp(float(v["pattern_tp3"]), tp3_min, tp3_max)
        v["pattern_bp0"] = clamp(float(v["pattern_bp0"]), bp0_min, bp0_max)
        v["pattern_bp3"] = clamp(float(v["pattern_bp3"]), bp3_min, bp3_max)

    def _enforce_superelliptic_symmetry(
        self,
        values: Dict[str, Any],
        *,
        half_w: float,
        half_h: float,
        tp0_bounds: Tuple[float, float],
        tp3_bounds: Tuple[float, float],
        bp0_bounds: Tuple[float, float],
        bp3_bounds: Tuple[float, float],
        reseed: bool = False,
    ):
        """Ensure tp1==tp2 and bp1==bp2 when superelliptic mode is active."""
        tp0_min, tp0_max = tp0_bounds
        tp3_min, tp3_max = tp3_bounds
        bp0_min, bp0_max = bp0_bounds
        bp3_min, bp3_max = bp3_bounds

        top_base_max = max(0.0, half_w - tp0_min)
        if reseed:
            top_base = min(half_w, half_h)
        else:
            top_base = float(values.get("pattern_tp1", half_w - float(values.get("pattern_tp0", 0.0))))
        top_base = clamp(top_base, 0.0, top_base_max)
        values["pattern_tp1"] = top_base
        values["pattern_tp2"] = top_base
        values["pattern_tp0"] = clamp(half_w - top_base, tp0_min, tp0_max)
        values["pattern_tp3"] = clamp(half_h - top_base, tp3_min, tp3_max)

        bottom_base_max = max(0.0, half_w - bp0_min)
        if reseed:
            bottom_base = min(half_w, half_h)
        else:
            bottom_base = float(values.get("pattern_bp1", half_w - float(values.get("pattern_bp0", 0.0))))
        bottom_base = clamp(bottom_base, 0.0, bottom_base_max)
        values["pattern_bp1"] = bottom_base
        values["pattern_bp2"] = bottom_base
        values["pattern_bp0"] = clamp(half_w - bottom_base, bp0_min, bp0_max)
        values["pattern_bp3"] = clamp(half_h - bottom_base, bp3_min, bp3_max)

    def _reflect_ui(self, keys: Optional[Set[str]]):
        # Always refresh ranges for all sliders that depend on pbw/pbh/ppw/mode
        dep_keys = (
            "pattern_tp0", "pattern_tp3",
            "pattern_bp0", "pattern_bp3",
            "pattern_tcc", "pattern_bcc",
            "pattern_tnn", "pattern_tmm", "pattern_bnn", "pattern_bmm",
        )
        for k in dep_keys:
            if k in self._slider:
                self._update_slider_range(k)
                self._push_slider_value(k, self._v.get(k, 0.0))

        # Push all inputs to reflect latest numbers (including layer-driven pbw/pbh/ppw echoes)
        for k in self._input.keys():
            self._push_input_value(k, self._v.get(k, ""))


# ------------------------------
# LParams: layer parameters with 2-way link to PParams.mode
# and inputs that drive PParams pbw/pbh/ppw
# ------------------------------

class LParams(QObject):
    changed = pyqtSignal(str, object)
    bulkChanged = pyqtSignal(dict)

    def __init__(self, pparams: PParams):
        super().__init__()
        self._p = pparams

        # Load layers configuration from settings
        self._layers_config = self._load_layers_config()

        self._v = {
            "layer_ldc": 1,
            "layer_pdc": 1,
            "layer_cfg": self._layers_config,
            "layer_mod": "even",       # even | gradual
            "layer_type": "wave",      # wave | lap
            "layer_sel": 0,            # selected layer index

            # new: layer-driven bounding box + per-wave width
            "layer_psp": self._p.get("pattern_psp"),
            "layer_ptc": 0.047,
            "layer_pmd": "straight",   # couples to PParams.pattern_mode
            "layer_pwt": False,
            "layer_psy": bool(self._p.get("pattern_symmetry")),
            "layer_pbw": self._p.get("pattern_pbw"),
            "layer_pbh": self._p.get("pattern_pbh"),
            "layer_ppw": self._p.get("pattern_ppw"),
        }
        self._input: Dict[str, Dict[str, Any]] = {}
        self._combo: Dict[str, QComboBox] = {}
        self._toggle: Dict[str, Any] = {}  # Supports QCheckBox and QPushButton
        self._block = False

        # listen to PParams to sync mode and pbw/pbh/ppw back
        self._p.changed.connect(self._on_pparams_changed)

        # Apply initial layer (layer 0) to PParams
        # Note: UI inputs will be updated when they're registered in ParametersPanel
        self._apply_selected_layer_to_pparams()

    def _load_layers_config(self) -> Dict[str, Any]:
        """Load layers configuration from settings.py"""
        try:
            import settings
            return settings.layers_b
        except Exception:
            # Fallback to default config
            return {
                "global": {
                    "layer_psp": 0.05,
                    "layer_ptc": 0.047,
                    "layer_pmd": "straight",
                    "layer_mod": "even",
                },
                "layers": []
            }

    def _map_layer_to_pattern_params(self, layer_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Map layer_* parameters to pattern_* parameters for PParams."""
        pparams_dict = {}
        for key, value in layer_dict.items():
            if key.startswith("pattern_"):
                # Direct pattern parameter
                pparams_dict[key] = value
            elif key == "layer_pmd":
                pparams_dict["pattern_mode"] = value
            elif key == "layer_type":
                pparams_dict["pattern_type"] = value
            elif key == "layer_pbw":
                pparams_dict["pattern_pbw"] = value
            elif key == "layer_pbh":
                pparams_dict["pattern_pbh"] = value
            elif key == "layer_ppw":
                pparams_dict["pattern_ppw"] = value
            elif key == "layer_psp":
                pparams_dict["pattern_psp"] = value
            elif key == "layer_pwt":
                pparams_dict["pattern_twist"] = value
            elif key == "layer_psy":
                pparams_dict["pattern_symmetry"] = value
        return pparams_dict

    def get(self, key, default=None):
        return self._v.get(key, default)

    def snapshot(self):
        """
        Return complete layers configuration with computed PParams for each layer.
        """
        layers_cfg = self._v.get("layer_cfg", {})
        global_params = layers_cfg.get("global", {})
        layers_list = layers_cfg.get("layers", [])

        result_layers = []

        for idx, layer_def in enumerate(layers_list):
            layer_type = layer_def.get("type", "normal")
            layer_params = layer_def.get("layer", {})

            # Merge global + layer-specific params
            merged = {**global_params, **layer_params}

            # Map layer_* parameters to pattern_* parameters for PParams
            pparams_dict = self._map_layer_to_pattern_params(merged)

            # Apply to PParams to get computed values
            temp_snapshot = self._p.snapshot()  # Save current state

            # Update PParams with mapped params
            if pparams_dict:
                self._p.update_bulk(pparams_dict, emit=False)

            # Get computed PParams
            computed_pparams = self._p.snapshot()

            # Restore original PParams state
            self._p.update_bulk(temp_snapshot, emit=False)

            # Build complete layer entry
            result_layers.append({
                "type": layer_type,
                "index": idx,
                "layer": {**merged, **computed_pparams}
            })

        return {
            "global": global_params,
            "layers": result_layers
        }

    def _apply_selected_layer_to_pparams(self):
        """Apply currently selected layer parameters to PParams and update LParams inputs."""
        sel_idx = self._v.get("layer_sel", 0)
        layers_cfg = self._v.get("layer_cfg", {})
        global_params = layers_cfg.get("global", {})
        layers_list = layers_cfg.get("layers", [])

        if 0 <= sel_idx < len(layers_list):
            layer_params = layers_list[sel_idx].get("layer", {})
            merged = {**global_params, **layer_params}

            # Update LParams values to reflect selected layer (for UI inputs)
            for key, value in merged.items():
                if key in self._v and not key.startswith("pattern_"):
                    self._v[key] = value
                    # Update UI for this field
                    if key in self._input:
                        self._push_input_value(key, value)
                    elif key in self._combo:
                        self._push_combo_value(key, value)
                    elif key in self._toggle:
                        self._push_toggle_value(key, value)

            # Map layer_* parameters to pattern_* parameters for PParams
            pparams_dict = self._map_layer_to_pattern_params(merged)

            # Update PParams with mapped params
            if pparams_dict:
                self._p.update_bulk(pparams_dict, emit=True)

    def _sync_config_global_param(self, key: str, value: Any):
        """Update a global parameter in the configuration."""
        layers_cfg = self._v.get("layer_cfg")
        if not layers_cfg:
            return
        if isinstance(layers_cfg.get("global"), dict):
            layers_cfg["global"][key] = value

    def _sync_config_pattern_mode(self, mode: str):
        """Ensure every layer config entry mirrors the active pattern mode."""
        layers_cfg = self._v.get("layer_cfg")
        if not layers_cfg:
            return

        if isinstance(layers_cfg.get("global"), dict):
            layers_cfg["global"]["layer_pmd"] = mode

        for layer_entry in layers_cfg.get("layers", []):
            layer_dict = layer_entry.get("layer") if isinstance(layer_entry, dict) else None
            if isinstance(layer_dict, dict):
                layer_dict["layer_pmd"] = mode

    def set(self, key: str, value: Any, emit=True):
        if self._block:
            return
        self._block = True

        self._v[key] = value

        # Handle layer selection change
        if key == "layer_sel":
            self._apply_selected_layer_to_pparams()

        # coupling to PParams
        elif key == "layer_pmd":
            mode = str(value)
            self._p.set("pattern_mode", mode)
            self._sync_config_pattern_mode(mode)
        elif key == "layer_mod":
            self._sync_config_global_param(key, value)
        elif key == "layer_type":
            self._p.set("pattern_type", str(value))
            self._sync_config_global_param(key, value)
        elif key == "layer_pwt":
            self._p.set("pattern_twist", bool(value))
        elif key == "layer_psy":
            self._p.set("pattern_symmetry", bool(value))
        elif key == "layer_psp":
            self._p.set("pattern_psp", float(value))
        elif key == "layer_pbw":
            self._p.set("pattern_pbw", float(value))
        elif key == "layer_pbh":
            self._p.set("pattern_pbh", float(value))
        elif key == "layer_ppw":
            self._p.set("pattern_ppw", float(value))

        # reflect UI
        if key in self._input:
            self._push_input_value(key, self._v[key])
        if key in self._combo:
            self._push_combo_value(key, self._v[key])
        if key in self._toggle:
            self._push_toggle_value(key, self._v[key])

        self._block = False
        if emit:
            self.changed.emit(key, self._v[key])

    def _on_pparams_changed(self, k: str, v: Any):
        # sync back to layer for UI consistency
        if k == "pattern_mode":
            self.set("layer_pmd", v)
        elif k == "pattern_type":
            self.set("layer_type", v)
        elif k == "pattern_pbw":
            self.set("layer_pbw", v)
        elif k == "pattern_pbh":
            self.set("layer_pbh", v)
        elif k == "pattern_ppw":
            self.set("layer_ppw", v)
        elif k == "pattern_twist":
            self.set("layer_pwt", v)
        elif k == "pattern_psp":
            self.set("layer_psp", v)
        elif k == "pattern_symmetry":
            self.set("layer_psy", v)

    # inputs
    def inputRegister(self, key: str, line: QLineEdit, *, fmt=None, parse=None):
        if fmt is None:
            fmt = lambda x: f"{x}"
        if parse is None:
            if key in ("layer_psp", "layer_ptc", "layer_pbw", "layer_pbh", "layer_ppw"):
                parse = lambda s: float(s)
            else:
                parse = lambda s: int(s)
        self._input[key] = {"widget": line, "fmt": fmt, "parse": parse}
        self._push_input_value(key, self._v.get(key, 0))
        line.editingFinished.connect(lambda: self._on_input_finished(key))

    def toggleRegister(self, key: str, widget):
        """Register a toggle widget (QCheckBox or checkable QPushButton)."""
        self._toggle[key] = widget
        widget.setChecked(bool(self._v.get(key, False)))

        # Use appropriate signal based on widget type
        from PyQt5.QtWidgets import QPushButton
        if isinstance(widget, QPushButton):
            # QPushButton uses toggled signal (emits bool directly)
            widget.toggled.connect(lambda checked, k=key: self.set(k, checked))
        else:
            # QCheckBox uses stateChanged signal (emits Qt.CheckState)
            widget.stateChanged.connect(lambda state, k=key: self.set(k, bool(state)))

    def _on_input_finished(self, key: str):
        reg = self._input[key]
        try:
            val = reg["parse"](reg["widget"].text())
        except Exception:
            return
        self.set(key, val)

    def _push_input_value(self, key: str, value: Any):
        reg = self._input[key]
        w: QLineEdit = reg["widget"]
        w.blockSignals(True)
        w.setText(reg["fmt"](value))
        w.blockSignals(False)

    def _push_toggle_value(self, key: str, value: Any):
        widget = self._toggle.get(key)
        if not widget:
            return
        widget.blockSignals(True)
        widget.setChecked(bool(value))
        widget.blockSignals(False)

    # combos
    def comboRegister(self, key: str, combo: QComboBox, items):
        combo.clear()
        for it in items:
            combo.addItem(it)
        self._combo[key] = combo
        self._push_combo_value(key, self._v.get(key, items[0]))
        combo.currentTextChanged.connect(lambda txt: self.set(key, txt))

    def _push_combo_value(self, key: str, value: Any):
        c = self._combo[key]
        c.blockSignals(True)

        # Special handling for layer_sel which uses integer index
        if key == "layer_sel" and isinstance(value, int):
            if 0 <= value < c.count():
                c.setCurrentIndex(value)
            else:
                c.setCurrentIndex(0)
        else:
            # Standard text-based lookup for other combos
            idx = c.findText(str(value))
            if idx < 0:
                idx = 0
            c.setCurrentIndex(idx)

        c.blockSignals(False)


# ------------------------------
# Global instances
# ------------------------------

PPARAMS = PParams()
LPARAMS = LParams(PPARAMS)


# ------------------------------
# Parameter Panel (embeddable)
# ------------------------------

class ParametersPanel(QWidget):
    OUTER_MARGIN = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)
        self.setMinimumSize(300, 800)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Group: Top/Bottom coupled ---
        gb_tb = QGroupBox("Top / Bottom Coupled Parameters")
        g2 = QGridLayout(gb_tb)

        # Top sliders
        self.s_tp0 = self._mk_slider()
        self.s_tp3 = self._mk_slider()
        self.s_tcc = self._mk_slider()

        # Bottom sliders
        self.s_bp0 = self._mk_slider()
        self.s_bp3 = self._mk_slider()
        self.s_bcc = self._mk_slider()

        # nn/mm (mode-dependent)
        self.s_tnn = self._mk_slider()
        self.s_tmm = self._mk_slider()
        self.s_bnn = self._mk_slider()
        self.s_bmm = self._mk_slider()

        # Layout rows
        g2.addWidget(QLabel("pattern_tp0"), 0, 0); g2.addWidget(self.s_tp0, 0, 1)
        g2.addWidget(QLabel("pattern_tp3"), 1, 0); g2.addWidget(self.s_tp3, 1, 1)
        g2.addWidget(QLabel("pattern_tcc"), 2, 0); g2.addWidget(self.s_tcc, 2, 1)

        g2.addWidget(QLabel("pattern_bp0"), 3, 0); g2.addWidget(self.s_bp0, 3, 1)
        g2.addWidget(QLabel("pattern_bp3"), 4, 0); g2.addWidget(self.s_bp3, 4, 1)
        g2.addWidget(QLabel("pattern_bcc"), 5, 0); g2.addWidget(self.s_bcc, 5, 1)

        g2.addWidget(QLabel("pattern_tnn"), 6, 0); g2.addWidget(self.s_tnn, 6, 1)
        g2.addWidget(QLabel("pattern_tmm"), 7, 0); g2.addWidget(self.s_tmm, 7, 1)
        g2.addWidget(QLabel("pattern_bnn"), 8, 0); g2.addWidget(self.s_bnn, 8, 1)
        g2.addWidget(QLabel("pattern_bmm"), 9, 0); g2.addWidget(self.s_bmm, 9, 1)

        layout.addWidget(gb_tb)

        # Register dynamic bounds (based on pbw/pbh/ppw)
        PPARAMS.sliderRegister(
            "pattern_tp0", self.s_tp0,
            vmin=lambda: self._corner_bounds()[0],
            vmax=lambda: self._corner_bounds()[1],
            scale=0.001
        )
        PPARAMS.sliderRegister(
            "pattern_tp3", self.s_tp3,
            vmin=lambda: 0.0,
            vmax=lambda: max(0.0, PPARAMS.get("pattern_pbh") * 0.5 - PPARAMS.get("pattern_ppw")),
            scale=0.001
        )
        PPARAMS.sliderRegister(
            "pattern_tcc", self.s_tcc,
            vmin=lambda: min(float(PPARAMS.get("pattern_ppw") or 0.0), float(PPARAMS.get("pattern_pbw") or 0.0)),
            vmax=lambda: float(PPARAMS.get("pattern_pbw") or 0.0),
            scale=0.001
        )

        PPARAMS.sliderRegister(
            "pattern_bp0", self.s_bp0,
            vmin=lambda: self._corner_bounds()[0],
            vmax=lambda: self._corner_bounds()[1],
            scale=0.001
        )
        PPARAMS.sliderRegister(
            "pattern_bp3", self.s_bp3,
            vmin=lambda: 0.0,
            vmax=lambda: max(0.0, PPARAMS.get("pattern_pbh") * 0.5 - PPARAMS.get("pattern_ppw")),
            scale=0.001
        )
        PPARAMS.sliderRegister(
            "pattern_bcc", self.s_bcc,
            vmin=lambda: min(float(PPARAMS.get("pattern_ppw") or 0.0), float(PPARAMS.get("pattern_pbw") or 0.0)),
            vmax=lambda: float(PPARAMS.get("pattern_pbw") or 0.0),
            scale=0.001
        )

        # Mode-dependent nn/mm sliders
        for key, sld in (("pattern_tnn", self.s_tnn),
                        ("pattern_tmm", self.s_tmm),
                        ("pattern_bnn", self.s_bnn),
                        ("pattern_bmm", self.s_bmm)):
            PPARAMS.sliderRegister(
                key, sld,
                vmin=lambda: 0.3 if PPARAMS.get("pattern_mode") == "superelliptic" else 2.0,
                vmax=lambda: 2.0,
                scale=0.01
            )

        # --- Readout group for p0..p3 (Top/Bottom) ---
        gb_read = QGroupBox("Readout: p0..p3 (auto-updated on mode/sliders)")
        g4 = QGridLayout(gb_read)

        self.lab_tp0 = QLabel(); self.lab_tp1 = QLabel()
        self.lab_tp2 = QLabel(); self.lab_tp3 = QLabel()
        self.lab_bp0 = QLabel(); self.lab_bp1 = QLabel()
        self.lab_bp2 = QLabel(); self.lab_bp3 = QLabel()

        g4.addWidget(QLabel("Top p0"), 0, 0); g4.addWidget(self.lab_tp0, 0, 1)
        g4.addWidget(QLabel("Top p1"), 1, 0); g4.addWidget(self.lab_tp1, 1, 1)
        g4.addWidget(QLabel("Top p2"), 2, 0); g4.addWidget(self.lab_tp2, 2, 1)
        g4.addWidget(QLabel("Top p3"), 3, 0); g4.addWidget(self.lab_tp3, 3, 1)

        g4.addWidget(QLabel("Bottom p0"), 0, 2); g4.addWidget(self.lab_bp0, 0, 3)
        g4.addWidget(QLabel("Bottom p1"), 1, 2); g4.addWidget(self.lab_bp1, 1, 3)
        g4.addWidget(QLabel("Bottom p2"), 2, 2); g4.addWidget(self.lab_bp2, 2, 3)
        g4.addWidget(QLabel("Bottom p3"), 3, 2); g4.addWidget(self.lab_bp3, 3, 3)

        layout.addWidget(gb_read)

        PPARAMS.changed.connect(lambda *_: self._refresh_p_labels())
        PPARAMS.bulkChanged.connect(lambda *_: self._refresh_p_labels())
        self._refresh_p_labels()

        # --- Layer inputs / modes (now also drive pbw/pbh/ppw) ---
        gb_layer = QGroupBox("Layer Inputs / Modes (drive pattern_pbw/pbh/ppw)")
        g3 = QGridLayout(gb_layer)

        self.in_psp = QLineEdit(); self.in_ptc = QLineEdit()
        self.in_ldc = QLineEdit(); self.in_pdc = QLineEdit()
        self.in_phw = QLineEdit(); self.in_phb = QLineEdit(); self.in_ppw = QLineEdit()

        # Two inputs per row: use 4 columns (label1, input1, label2, input2)
        g3.addWidget(QLabel("spacing"), 0, 0); g3.addWidget(self.in_psp, 0, 1)
        g3.addWidget(QLabel("thickness"), 0, 2); g3.addWidget(self.in_ptc, 0, 3)

        g3.addWidget(QLabel("layer count"), 1, 0); g3.addWidget(self.in_ldc, 1, 1)
        g3.addWidget(QLabel("pattern count"), 1, 2); g3.addWidget(self.in_pdc, 1, 3)

        g3.addWidget(QLabel("bbox width"), 2, 0); g3.addWidget(self.in_phw, 2, 1)
        g3.addWidget(QLabel("bbox height"), 2, 2); g3.addWidget(self.in_phb, 2, 3)

        g3.addWidget(QLabel("pattern width"), 3, 0); g3.addWidget(self.in_ppw, 3, 1)

        self.cmb_pmd = QComboBox(); self.cmb_mod = QComboBox(); self.cmb_sel = QComboBox()
        self.cmb_type = QComboBox()

        g3.addWidget(QLabel("layer select"), 3, 2); g3.addWidget(self.cmb_sel, 3, 3)

        g3.addWidget(QLabel("pattern mode"), 4, 0); g3.addWidget(self.cmb_pmd, 4, 1)
        g3.addWidget(QLabel("layer mode"), 4, 2); g3.addWidget(self.cmb_mod, 4, 3)
        g3.addWidget(QLabel("layer type"), 5, 0); g3.addWidget(self.cmb_type, 5, 1)

        # Create toggle buttons instead of checkboxes
        self.btn_pwt = QPushButton("Twist")
        self.btn_pwt.setCheckable(True)
        self.btn_psy = QPushButton("Symmetry")
        self.btn_psy.setCheckable(True)

        # Add spiral mode toggle
        self.btn_spiral_mode = QPushButton("Spiral Mode")
        self.btn_spiral_mode.setCheckable(True)
        self.btn_spiral_mode.setChecked(False)

        # Create action buttons (non-checkable)
        self.btn_save_step = QPushButton("Save STEP")
        self.btn_refresh_view = QPushButton("Refresh View")
        self.btn_generate_layers = QPushButton("Generate Layers")

        # Create horizontal layout for toggle buttons
        toggle_layout = QHBoxLayout()
        toggle_layout.addWidget(self.btn_pwt)
        toggle_layout.addWidget(self.btn_psy)
        toggle_layout.addWidget(self.btn_spiral_mode)

        # Create horizontal layout for action buttons
        action_layout = QHBoxLayout()
        action_layout.addWidget(self.btn_save_step)
        action_layout.addWidget(self.btn_refresh_view)
        action_layout.addWidget(self.btn_generate_layers)

        # Add button layouts spanning full width
        g3.addLayout(toggle_layout, 5, 2, 1, 2)  # Toggles on right side of row 5
        g3.addLayout(action_layout, 6, 0, 1, 4)  # Actions span all columns

        layout.addWidget(gb_layer)

        # inputs
        LPARAMS.inputRegister("layer_psp", self.in_psp, fmt=lambda x: f"{x:.3f}", parse=lambda s: float(s))
        LPARAMS.inputRegister("layer_ptc", self.in_ptc, fmt=lambda x: f"{x:.3f}", parse=lambda s: float(s))
        LPARAMS.inputRegister("layer_ldc", self.in_ldc, fmt=lambda x: f"{x}", parse=lambda s: int(s))
        LPARAMS.inputRegister("layer_pdc", self.in_pdc, fmt=lambda x: f"{x}", parse=lambda s: int(s))

        # new: layer_pbw/phb/ppw inputs that drive PParams
        LPARAMS.inputRegister("layer_pbw", self.in_phw, fmt=lambda x: f"{x:.3f}", parse=lambda s: float(s))
        LPARAMS.inputRegister("layer_pbh", self.in_phb, fmt=lambda x: f"{x:.3f}", parse=lambda s: float(s))
        LPARAMS.inputRegister("layer_ppw", self.in_ppw, fmt=lambda x: f"{x:.3f}", parse=lambda s: float(s))

        # Build layer selection items: "index: type"
        layers_cfg = LPARAMS.get("layer_cfg", {})
        layers_list = layers_cfg.get("layers", [])
        layer_items = [f"{idx}: {layer.get('type', 'normal')}" for idx, layer in enumerate(layers_list)]
        if not layer_items:
            layer_items = ["0: default"]

        # combos - special handling for layer_sel to parse index
        self.cmb_sel.clear()
        for it in layer_items:
            self.cmb_sel.addItem(it)

        # Register combo in LPARAMS (but don't use standard comboRegister due to index parsing)
        LPARAMS._combo["layer_sel"] = self.cmb_sel

        # Connect with custom handler to extract index from "index: type" format
        def on_layer_sel_changed(txt):
            try:
                idx = int(txt.split(":")[0].strip())
                LPARAMS.set("layer_sel", idx)
            except Exception:
                pass

        self.cmb_sel.currentTextChanged.connect(on_layer_sel_changed)

        # Set initial value
        initial_idx = LPARAMS.get("layer_sel", 0)
        if 0 <= initial_idx < len(layer_items):
            self.cmb_sel.setCurrentIndex(initial_idx)
        LPARAMS.comboRegister("layer_pmd", self.cmb_pmd, ["straight", "superelliptic"])
        LPARAMS.comboRegister("layer_mod", self.cmb_mod, ["even", "gradual"])
        LPARAMS.comboRegister("layer_type", self.cmb_type, ["wave", "lap"])
        LPARAMS.toggleRegister("layer_pwt", self.btn_pwt)
        LPARAMS.toggleRegister("layer_psy", self.btn_psy)

        # Tune slider UI to avoid handle clipping
        self._tune_slider_ui([
            self.s_tp0, self.s_tp3, self.s_tcc,
            self.s_bp0, self.s_bp3, self.s_bcc,
            self.s_tnn, self.s_tmm, self.s_bnn, self.s_bmm
        ])

        # Suggested minimum footprint so embedding layouts reserve enough space
        self.setMinimumSize(200, 800)

    def _refresh_p_labels(self):
        bw = PPARAMS.get("pattern_pbw")
        bh = PPARAMS.get("pattern_pbh")

        tp0 = PPARAMS.get("pattern_tp0"); tp1 = PPARAMS.get("pattern_tp1")
        tp2 = PPARAMS.get("pattern_tp2"); tp3 = PPARAMS.get("pattern_tp3")

        bp0 = PPARAMS.get("pattern_bp0"); bp1 = PPARAMS.get("pattern_bp1")
        bp2 = PPARAMS.get("pattern_bp2"); bp3 = PPARAMS.get("pattern_bp3")

        self.lab_tp0.setText(f"{tp0:.6f}")
        self.lab_tp1.setText(f"{tp1:.6f}")
        self.lab_tp2.setText(f"{tp2:.6f}")
        self.lab_tp3.setText(f"{tp3:.6f}")

        self.lab_bp0.setText(f"{bp0:.6f}")
        self.lab_bp1.setText(f"{bp1:.6f}")
        self.lab_bp2.setText(f"{bp2:.6f}")
        self.lab_bp3.setText(f"{bp3:.6f}")

    @staticmethod
    def _mk_slider() -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setMinimum(0); s.setMaximum(1000)
        return s

    @staticmethod
    @staticmethod
    def _corner_bounds():
        pbw = float(PPARAMS.get("pattern_pbw") or 0.0)
        pbh = float(PPARAMS.get("pattern_pbh") or 0.0)
        ppw = float(PPARAMS.get("pattern_ppw") or 0.0)
        half_w = pbw / 2.0
        if PPARAMS.get("pattern_mode") == "superelliptic":
            return half_w - min(pbw, pbh), half_w
        return -max(0.0, pbw) * 0.25, max(0.0, half_w - ppw)

    @staticmethod
    def _tune_slider_ui(sliders):
        style = """
        QSlider::groove:horizontal { height: 6px; margin: 12px 12px; }
        QSlider::handle:horizontal { width: 18px; margin: -10px 0; border-radius: 9px; }
        """
        for s in sliders:
            s.setMinimumHeight(28)
            s.setStyleSheet(style)


# ------------------------------
# Test Window wrapper
# ------------------------------

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PParams / LParams Interactive Test")
        panel = ParametersPanel()
        self.setCentralWidget(panel)
        self.resize(310, 800)


# ------------------------------
# Main
# ------------------------------

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
