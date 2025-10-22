"""
Visualization module for interactive rectangle drawing.

This module handles all matplotlib-related functionality including:
- Figure and axes setup
- Slider and text box widgets
- Drawing operations
- User interaction handling

The refactored version exposes a modular structure that decouples the
visualizer from a specific rectangle generator implementation and
supports asynchronous parameter updates.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterable, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, TextBox


class RectangleModelProtocol(Protocol):
    """Protocol describing the data provider expected by the visualizer."""

    def get_parameters(self) -> Mapping[str, float]:
        ...

    def generate_rectangle(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class LayoutConfig:
    """Layout related configuration."""

    height: int = 200
    multiple: float = 2.5
    spacing: float = 0.05

    def clamped_spacing(self) -> float:
        return max(0.0, min(0.2, self.spacing))


@dataclass(frozen=True)
class StyleConfig:
    """Visual styling configuration."""

    coef_color: str = "blue"
    wh_color: str = "red"
    axis_color: str = "black"
    line_width: float = 2.0
    left_alpha: float = 1.0
    right_alpha: float = 0.3
    axis_alpha: float = 0.8
    info_box_facecolor: str = "wheat"
    info_box_alpha: float = 0.8
    legend_location: str = "upper right"


@dataclass(frozen=True)
class ParameterBinding:
    """Metadata describing how a single parameter is exposed to the UI."""

    name: str
    label: str
    minimum: float
    maximum: float
    setter: Callable[[float], None]

    def clamp(self, value: float) -> float:
        return max(self.minimum, min(self.maximum, value))


class AsyncInputAdapter:
    """
    Bridge that accepts asynchronous or threaded updates and feeds them into the UI.

    The adapter uses a thread-safe queue and a matplotlib timer to drain updates
    on the UI thread, ensuring the visualizer stays responsive.
    """

    def __init__(self, apply_callback: Callable[[Mapping[str, float]], None], poll_interval: float = 0.1):
        self._apply_callback = apply_callback
        self._poll_interval = max(0.01, poll_interval)
        self._queue: "queue.Queue[Mapping[str, float]]" = queue.Queue()
        self._timer = None
        self._figure: Optional[Figure] = None

    def bind(self, figure: Figure) -> None:
        """Attach the adapter to a matplotlib figure so it can drain the queue."""
        if self._timer is not None:
            return
        self._figure = figure
        self._timer = figure.canvas.new_timer(interval=int(self._poll_interval * 1000))
        self._timer.add_callback(self._dispatch_updates)
        self._timer.start()

    def submit(self, update: Mapping[str, float]) -> None:
        """Submit an update from any thread."""
        self._queue.put(dict(update))

    async def submit_async(self, update: Mapping[str, float]) -> None:
        """Submit an update from an asyncio coroutine."""
        await asyncio.to_thread(self.submit, update)

    async def stream(self, updates: AsyncIterable[Mapping[str, float]]) -> None:
        """Consume an asynchronous stream of updates."""
        async for update in updates:
            await self.submit_async(update)

    def _dispatch_updates(self) -> None:
        """Drain queued updates and forward them to the visualizer."""
        processed = False
        while True:
            try:
                update = self._queue.get_nowait()
            except queue.Empty:
                break
            self._apply_callback(update)
            processed = True
        if processed and self._figure is not None:
            self._figure.canvas.draw_idle()


class RectangleArtist:
    """Encapsulates all drawing logic for the rectangle visualization."""

    def __init__(self, ax: Axes, style: StyleConfig):
        self.ax = ax
        self.style = style
        self._cached_artists = []  # Cache for reusable drawing elements
        self._last_draw_call = 0
        self._draw_throttle = 0.016  # ~60 FPS limit

    def render(self, rect_data: Mapping[str, Any], parameters: Mapping[str, float]) -> None:
        import time
        current_time = time.time()
        
        # Throttle drawing to prevent excessive redraws
        if current_time - self._last_draw_call < self._draw_throttle:
            return
        
        self._clear()
        self._draw_bottom_edge(rect_data)
        self._draw_top_edge(rect_data)
        self._draw_left_edge(rect_data)
        self._draw_right_edge(rect_data)
        self._draw_vertical_axis(rect_data)
        self._add_parameter_info(rect_data, parameters)
        
        self._last_draw_call = current_time
        # Only call draw_idle once at the end
        self.ax.figure.canvas.draw_idle()

    def _clear(self) -> None:
        # More efficient clearing using clear() method
        if hasattr(self.ax, 'clear'):
            # Store axis properties before clearing
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xlabel = self.ax.get_xlabel()
            ylabel = self.ax.get_ylabel()
            title = self.ax.get_title()
            
            self.ax.clear()
            
            # Restore axis properties
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_xlabel(xlabel, fontsize=8)
            self.ax.set_ylabel(ylabel, fontsize=8)
            self.ax.set_title(title, fontsize=10)
            self.ax.grid(True, alpha=0.3)
            self.ax.tick_params(labelsize=7)
            self.ax.set_aspect("equal")
        else:
            # Fallback to original method if clear() is not available
            for line in self.ax.lines[:]:
                line.remove()
            for text in self.ax.texts[:]:
                text.remove()
            legend = self.ax.get_legend()
            if legend:
                legend.remove()

    def _draw_bottom_edge(self, rect_data: Mapping[str, Any]) -> None:
        center_x = rect_data["center_x"]
        y0 = rect_data["origin"][1]
        for segment in rect_data["bottom_segments"]:
            self._draw_horizontal_segment(segment, y0, center_x)

    def _draw_top_edge(self, rect_data: Mapping[str, Any]) -> None:
        center_x = rect_data["center_x"]
        y_top = rect_data["origin"][1] + rect_data["total_height"]
        for segment in rect_data["top_segments"]:
            self._draw_horizontal_segment(segment, y_top, center_x)

    def _draw_left_edge(self, rect_data: Mapping[str, Any]) -> None:
        x0 = rect_data["origin"][0]
        for segment in rect_data["left_segments"]:
            color = self._color_for_segment(segment["type"])
            self.ax.plot(
                [x0, x0],
                [segment["y_start"], segment["y_end"]],
                color=color,
                linewidth=self.style.line_width,
                alpha=self.style.left_alpha,
            )

    def _draw_right_edge(self, rect_data: Mapping[str, Any]) -> None:
        x_right = rect_data["origin"][0] + rect_data["total_width"]
        for segment in rect_data["right_segments"]:
            color = self._color_for_segment(segment["type"])
            self.ax.plot(
                [x_right, x_right],
                [segment["y_start"], segment["y_end"]],
                color=color,
                linewidth=self.style.line_width,
                alpha=self.style.right_alpha,
            )

    def _draw_horizontal_segment(self, segment: Mapping[str, Any], y_coord: float, center_x: float) -> None:
        x_start = segment["x_start"]
        x_end = segment["x_end"]
        color = self._color_for_segment(segment["type"])
        if x_start < center_x < x_end:
            self.ax.plot(
                [x_start, center_x],
                [y_coord, y_coord],
                color=color,
                linewidth=self.style.line_width,
                alpha=self.style.left_alpha,
            )
            self.ax.plot(
                [center_x, x_end],
                [y_coord, y_coord],
                color=color,
                linewidth=self.style.line_width,
                alpha=self.style.right_alpha,
            )
        else:
            alpha = self.style.left_alpha if x_end <= center_x else self.style.right_alpha
            self.ax.plot(
                [x_start, x_end],
                [y_coord, y_coord],
                color=color,
                linewidth=self.style.line_width,
                alpha=alpha,
            )

    def _draw_vertical_axis(self, rect_data: Mapping[str, Any]) -> None:
        x0, y0 = rect_data["origin"]
        center_x = rect_data["center_x"]
        total_height = rect_data["total_height"]
        self.ax.plot(
            [center_x, center_x],
            [y0, y0 + total_height],
            color=self.style.axis_color,
            linewidth=2,
            linestyle="--",
            alpha=self.style.axis_alpha,
        )
        mark_size = min(rect_data["total_width"], total_height) * 0.02
        self.ax.plot(
            [center_x - mark_size, center_x + mark_size],
            [y0, y0],
            color=self.style.axis_color,
            linewidth=2,
            alpha=self.style.axis_alpha,
        )
        self.ax.plot(
            [center_x - mark_size, center_x + mark_size],
            [y0 + total_height, y0 + total_height],
            color=self.style.axis_color,
            linewidth=2,
            alpha=self.style.axis_alpha,
        )

    def _add_parameter_info(self, rect_data: Mapping[str, Any], parameters: Mapping[str, float]) -> None:
        x0, y0 = rect_data["origin"]
        total_width = rect_data["total_width"]
        total_height = rect_data["total_height"]
        info_text = (
            "Parameters:\n"
            f'Coef = {parameters["coef"]:.1f}\n'
            f'W = {parameters["W"]:.1f}\n'
            f'H = {parameters["H"]:.1f}\n\n'
            "Dimensions:\n"
            f"Width = {total_width:.1f}\n"
            f"Height = {total_height:.1f}\n\n"
            "Formula:\n"
            f'({parameters["coef"]:.1f}+{parameters["W"]:.1f}+{parameters["coef"]:.1f}) × '
            f'({parameters["coef"]:.1f}+{parameters["H"]:.1f}+{parameters["coef"]:.1f})'
        )
        self.ax.text(
            x0 + total_width + 1,
            y0 + total_height,
            info_text,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor=self.style.info_box_facecolor,
                alpha=self.style.info_box_alpha,
            ),
        )
        handles = [
            plt.Line2D([0], [0], color=self.style.coef_color, linewidth=self.style.line_width, label="Coef segments"),
            plt.Line2D([0], [0], color=self.style.wh_color, linewidth=self.style.line_width, label="W/H segments"),
            plt.Line2D([0], [0], color=self.style.axis_color, linewidth=2, linestyle="--", label="Center axis"),
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linewidth=self.style.line_width,
                alpha=self.style.right_alpha,
                label="Right side (30% opacity)",
            ),
        ]
        self.ax.legend(handles=handles, loc=self.style.legend_location, fontsize=8)

    def _color_for_segment(self, segment_type: str) -> str:
        return self.style.coef_color if segment_type == "coef" else self.style.wh_color


class ControlPanel:
    """Encapsulates slider/textbox creation and synchronisation."""

    def __init__(
        self,
        fig: Figure,
        panel_axis: Axes,
        bindings: Mapping[str, ParameterBinding],
        initial_values: Mapping[str, float],
        on_change: Callable[[str, float], None],
    ):
        self.fig = fig
        self.panel_axis = panel_axis
        self.bindings = bindings
        self.on_change = on_change
        self.sliders: Dict[str, Slider] = {}
        self.textboxes: Dict[str, TextBox] = {}
        self._suspend_events = False
        
        # Debouncing mechanism
        self._debounce_timer: Optional[threading.Timer] = None
        self._pending_changes: Dict[str, float] = {}
        self._debounce_delay = 0.05  # 50ms debounce delay

        self.panel_axis.set_xlim(0, 1)
        self.panel_axis.set_ylim(0, 1)
        self.panel_axis.axis("off")
        self.panel_axis.text(
            0.5,
            0.95,
            "Controls",
            fontsize=10,
            weight="bold",
            ha="center",
            va="top",
            transform=self.panel_axis.transAxes,
        )

        self._build_controls(initial_values)

    def _build_controls(self, initial_values: Mapping[str, float]) -> None:
        panel_box = self.panel_axis.get_position()
        panel_width = panel_box.width
        panel_height = panel_box.height
        left = panel_box.x0
        bottom = panel_box.y0

        binding_order: Iterable[ParameterBinding] = (self.bindings[name] for name in self.bindings)
        bindings_list = list(binding_order)
        rows = max(1, len(bindings_list))
        
        # Calculate the actual content area first
        content_padding = 0.01  # Small padding inside the border
        content_left = left + content_padding
        content_bottom = bottom + content_padding
        content_width = panel_width - 2 * content_padding
        content_height = panel_height - 2 * content_padding
        
        # Calculate row dimensions
        row_height = content_height / rows
        control_height = row_height * 0.7  # Height for slider and input
        vertical_margin = row_height * 0.15  # Margins above and below each row
        
        # Within each row: slider (70%) + input (25%) + spacing (5%)
        slider_width = content_width * 0.70
        input_width = content_width * 0.25
        horizontal_gap = content_width * 0.05
        
        # Calculate the actual bounding box of all controls
        first_row_bottom = content_bottom + content_height - row_height + vertical_margin
        last_row_bottom = content_bottom + vertical_margin
        controls_top = first_row_bottom + control_height
        controls_bottom = last_row_bottom
        controls_left = content_left
        controls_right = content_left + slider_width + horizontal_gap + input_width
        
        # Draw border around the actual control area with small padding
        border_padding = 0.005
        border_left = controls_left - border_padding
        border_bottom = controls_bottom - border_padding
        border_width = controls_right - controls_left + 2 * border_padding
        border_height = controls_top - controls_bottom + 2 * border_padding
        
        border_rect = plt.Rectangle(
            (border_left, border_bottom), border_width, border_height,
            linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.2
        )
        self.panel_axis.add_patch(border_rect)

        for index, binding in enumerate(bindings_list):
            # Calculate row position (from top to bottom)
            row_bottom = content_bottom + content_height - (index + 1) * row_height + vertical_margin
            
            # Slider position (includes the tag as slider label)
            slider_left = content_left
            slider_ax = self.fig.add_axes([slider_left, row_bottom, slider_width, control_height])
            
            # Input position
            input_left = slider_left + slider_width + horizontal_gap
            input_ax = self.fig.add_axes([input_left, row_bottom, input_width, control_height])

            # Create slider with tag as label
            slider = Slider(
                slider_ax,
                binding.label,  # Tag becomes the slider label
                binding.minimum,
                binding.maximum,
                valinit=binding.clamp(float(initial_values.get(binding.name, binding.minimum))),
                valfmt="%.2f",  # Keep format but hide the text
            )
            # Hide the value text but keep the label
            slider.valtext.set_visible(False)
            slider.valtext.set_text("")  # Clear the text content
            slider.label.set_fontsize(9)
            slider.label.set_weight('bold')
            slider.on_changed(lambda value, name=binding.name: self._handle_slider_change(name, value))

            # Create input box
            textbox = TextBox(input_ax, "", initial=f'{initial_values.get(binding.name, binding.minimum):.2f}')
            textbox.on_submit(lambda text, name=binding.name: self._handle_text_change(name, text))
            textbox.text_disp.set_fontsize(9)

            self.sliders[binding.name] = slider
            self.textboxes[binding.name] = textbox

    def _handle_slider_change(self, name: str, value: float) -> None:
        if self._suspend_events:
            return
        binding = self.bindings[name]
        clamped = binding.clamp(value)
        self._update_textbox(name, clamped)
        
        # Store the change and debounce
        self._pending_changes[name] = clamped
        self._debounce_parameter_change()

    def _handle_text_change(self, name: str, text: str) -> None:
        if self._suspend_events:
            return
        binding = self.bindings[name]
        try:
            value = float(text)
        except ValueError:
            self._sync_single(name, self.sliders[name].val)
            return
        clamped = binding.clamp(value)
        self._sync_single(name, clamped)
        # Text changes are immediate (no debouncing for manual input)
        self.on_change(name, clamped)

    def _debounce_parameter_change(self) -> None:
        """Debounce parameter changes to reduce update frequency during rapid slider movements."""
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
        
        def apply_changes():
            if self._pending_changes:
                # Apply all pending changes at once
                for name, value in self._pending_changes.items():
                    self.on_change(name, value)
                self._pending_changes.clear()
        
        self._debounce_timer = threading.Timer(self._debounce_delay, apply_changes)
        self._debounce_timer.start()

    def sync_parameters(self, parameters: Mapping[str, float]) -> None:
        self._suspend_events = True
        try:
            for name, value in parameters.items():
                if name in self.bindings:
                    clamped = self.bindings[name].clamp(float(value))
                    self._sync_single(name, clamped)
        finally:
            self._suspend_events = False

    def cleanup(self) -> None:
        """Clean up resources, particularly the debounce timer."""
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
            self._debounce_timer = None

    def _sync_single(self, name: str, value: float) -> None:
        if name in self.sliders:
            self.sliders[name].set_val(value)
        self._update_textbox(name, value)

    def _update_textbox(self, name: str, value: float) -> None:
        textbox = self.textboxes.get(name)
        if textbox:
            textbox.set_val(f"{value:.2f}")


class RectangleVisualizer:
    """
    A class to visualize and manage interactive rectangle display.

    The class is now composed of smaller building blocks (layout, artist, controls, async input)
    making it easier to reuse or extend individual pieces.
    """

    def __init__(
        self,
        rectangle_model: RectangleModelProtocol,
        height: int = 200,
        multiple: float = 2.5,
        spacing: float = 0.05,
        *,
        style: Optional[StyleConfig] = None,
        parameter_bindings: Optional[Sequence[ParameterBinding]] = None,
        async_poll_interval: float = 0.1,
    ):
        self.model = rectangle_model
        self.layout = LayoutConfig(height=height, multiple=multiple, spacing=spacing)
        self.style = style or StyleConfig()
        self.bindings = self._resolve_bindings(parameter_bindings)

        self.fig, self.ax_controls, self.ax_main, self.ax_reserved = self._setup_figure()
        self.artist = RectangleArtist(self.ax_main, self.style)
        self.control_panel = ControlPanel(
            self.fig,
            self.ax_controls,
            self.bindings,
            self.model.get_parameters(),
            self._handle_parameter_change,
        )

        self._async_adapter = AsyncInputAdapter(self._apply_external_updates, async_poll_interval)
        self._async_adapter.bind(self.fig)

        self._render()

    def _resolve_bindings(self, parameter_bindings: Optional[Sequence[ParameterBinding]]) -> Dict[str, ParameterBinding]:
        if parameter_bindings:
            return {binding.name: binding for binding in parameter_bindings}

        defaults = [
            ("coef", "Coef", 0.0, 20.0, "set_coef"),
            ("W", "W", 0.0, 20.0, "set_w"),
            ("H", "H", 0.0, 20.0, "set_h"),
        ]

        resolved: Dict[str, ParameterBinding] = {}
        for name, label, minimum, maximum, setter_name in defaults:
            setter = getattr(self.model, setter_name, None)
            if not callable(setter):
                raise AttributeError(f"Rectangle model is missing required setter '{setter_name}'")
            resolved[name] = ParameterBinding(name, label, minimum, maximum, setter)
        return resolved

    def _setup_figure(self) -> tuple[Figure, Axes, Axes, Axes]:
        fig_width = (self.layout.height * (2 + self.layout.multiple)) / 80
        fig_height = self.layout.height / 60
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.canvas.toolbar_visible = False

        widths = [1, 1, self.layout.multiple]
        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=widths,
            left=0.03,
            right=0.97,
            top=0.92,
            bottom=0.08,
            wspace=self.layout.clamped_spacing(),
        )

        controls_ax = fig.add_subplot(grid[0, 0])
        main_ax = fig.add_subplot(grid[0, 1])
        reserved_ax = fig.add_subplot(grid[0, 2])

        for ax in (main_ax, reserved_ax):
            ax.set_xlim(-1, 25)
            ax.set_ylim(-1, 25)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X coordinate", fontsize=8)
            ax.set_ylabel("Y coordinate", fontsize=8)
            ax.tick_params(labelsize=7)

        main_ax.set_aspect("equal")
        main_ax.set_title("Rectangle Display (Window A)", fontsize=10)

        reserved_ax.set_title("Reserved Display (Window B)", fontsize=10)
        reserved_ax.set_aspect(1.0 / self.layout.multiple)
        reserved_ax.text(
            0.5,
            0.5,
            "Window B\n(Reserved for future use)",
            transform=reserved_ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.5,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
        )

        return fig, controls_ax, main_ax, reserved_ax

    def _handle_parameter_change(self, name: str, value: float) -> None:
        binding = self.bindings[name]
        binding.setter(binding.clamp(value))
        self._render()

    def _apply_external_updates(self, updates: Mapping[str, float]) -> None:
        changed = False
        for name, value in updates.items():
            if name not in self.bindings:
                continue
            binding = self.bindings[name]
            clamped = binding.clamp(float(value))
            binding.setter(clamped)
            changed = True
        if changed:
            self._render()

    def _render(self) -> None:
        parameters = dict(self.model.get_parameters())
        rectangle_data = self.model.generate_rectangle()
        self.control_panel.sync_parameters(parameters)
        self.artist.render(rectangle_data, parameters)

    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self.control_panel, 'cleanup'):
            self.control_panel.cleanup()
        plt.close(self.fig)

    @property
    def async_input(self) -> AsyncInputAdapter:
        """Expose the async adapter so callers can push updates."""
        return self._async_adapter

    async def consume_async_updates(self, updates: AsyncIterable[Mapping[str, float]]) -> None:
        """Convenience wrapper to pipe an async iterable into the adapter."""
        await self._async_adapter.stream(updates)

    def update_parameters(self, updates: Mapping[str, float]) -> None:
        """Apply a batch of parameter updates programmatically."""
        self._apply_external_updates(updates)

    def show(self, *, block: bool = True) -> None:
        """Display the interactive plot."""
        plt.show(block=block)


# End of RectangleVisualizer class
