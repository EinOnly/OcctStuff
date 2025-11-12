# spiral_plot.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Sequence
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.TColgp import TColgp_Array1OfPnt
except ModuleNotFoundError:
    gp_Pnt = None
    GeomAPI_PointsToBSpline = None
    GeomAbs_C2 = None
    TColgp_Array1OfPnt = None

# ===== 参数 =====
radius = 6.2055   # 外接圆半径（外边贴圆）
thick  = 0.1315   # 螺旋带宽度（也是圈距）
layer_count  = 6        # 卷绕圈数
save_path = "spiral.png"

# ===== 矩形参数 =====
rect_width = 0.3777   # 矩形沿螺旋线方向的宽度
rect_gap = 0.05       # 矩形之间的间隔
rect_thick = 0.047*2
# 矩形厚度 = thick（沿法线方向）


# ===== 渐变矩形参数 =====
rect_width_start = 0.3777  # 起始宽度
width_change_rate = -0.02  # 每圈宽度变化量（负数表示递减）


_RECT_COLOR_BASE: Tuple[Tuple[float, float, float], ...] = (
    (0.20, 0.42, 0.88),  # 深蓝
    (0.88, 0.24, 0.24),  # 红色
    (0.22, 0.65, 0.32),  # 绿色备用
)

AREA_DIVISION_RANGE = (17, 100)


def _build_rect_palette(num_divisions: int) -> Tuple[Tuple[float, float, float], ...]:
    """Choose a palette that alternates blue/red and adds a third tone when needed."""
    if num_divisions <= 0:
        return _RECT_COLOR_BASE[:2]
    if num_divisions % 2 == 0 or num_divisions <= 2:
        return _RECT_COLOR_BASE[:2]
    return _RECT_COLOR_BASE


def _color_from_index(index: int, palette: Tuple[Tuple[float, float, float], ...]) -> Tuple[float, float, float]:
    """Return a stable color for a rectangle based on its index and palette."""
    if not palette:
        return (0.5, 0.5, 0.5)
    return palette[index % len(palette)]


def _estimate_rect_width(rect: np.ndarray) -> float:
    """Approximate rectangle width along the spiral tangent."""
    if rect.shape[0] < 4:
        return 0.0
    top = np.linalg.norm(rect[1] - rect[0])
    bottom = np.linalg.norm(rect[2] - rect[3])
    return float(0.5 * (top + bottom))


def _calculate_spiral_area(radius_value: float, thick_value: float, turn_count: int) -> float:
    """Compute spiral band area by accumulating ring areas per turn."""
    if turn_count <= 0:
        return 0.0
    b = thick_value / (2.0 * np.pi)
    spiral_area = 0.0
    for turn in range(turn_count):
        r_out = radius_value - b * 2.0 * np.pi * turn
        r_in = radius_value - b * 2.0 * np.pi * (turn + 1)
        spiral_area += np.pi * (r_out ** 2 - r_in ** 2)
    return float(max(0.0, spiral_area))

def spiral_band(radius: float, thick: float, count: int, N_per_turn: int = 1500):
    # 保证带宽正、圈数正
    if radius <= 0 or thick <= 0 or count <= 0:
        raise ValueError("radius、thick、count 必须为正数。")

    # 圈距 s = 2πb = thick
    b = thick / (2.0 * np.pi)

    # 外圆贴边：中心线外边距 radius - thick/2
    r0 = radius - thick / 2.0

    # 6圈：θ_max = 2π * count
    theta_max = 2.0 * np.pi * count

    # 采样
    N = max(int(N_per_turn * count), 800)
    theta = np.linspace(0.0, theta_max, N)
    r = r0 - b * theta

    # 中心线坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 计算法线方向，做 ±thick/2 偏移
    dr_dtheta = -b
    dx_dtheta = dr_dtheta * np.cos(theta) - r * np.sin(theta)
    dy_dtheta = dr_dtheta * np.sin(theta) + r * np.cos(theta)

    nx = -dy_dtheta
    ny = dx_dtheta
    n_norm = np.hypot(nx, ny) + 1e-12
    nx /= n_norm
    ny /= n_norm

    half_w = thick / 2.0
    x_outer = x + half_w * nx
    y_outer = y + half_w * ny
    x_inner = x - half_w * nx
    y_inner = y - half_w * ny

    # 组合成多边形
    poly_x = np.concatenate([x_outer, x_inner[::-1]])
    poly_y = np.concatenate([y_outer, y_inner[::-1]])

    return (poly_x, poly_y), (x, y), (theta, r)

def _compute_arc_lengths(theta_array: np.ndarray, r_array: np.ndarray, thick: float) -> np.ndarray:
    """
    Compute cumulative arc lengths along the spiral centerline defined by theta/r arrays.
    
    Args:
        theta_array: Parameter angles along the spiral.
        r_array: Corresponding spiral radii.
        thick: Spiral band thickness (used to derive dr/dtheta).
        
    Returns:
        np.ndarray: Cumulative arc length array (same length as theta_array).
    """
    if len(theta_array) != len(r_array):
        raise ValueError("theta_array and r_array must have the same length")
    
    arc_lengths = np.zeros(len(theta_array))
    if len(theta_array) < 2:
        return arc_lengths
    
    b = thick / (2.0 * np.pi)
    dr_dtheta = -b
    
    for i in range(1, len(theta_array)):
        dtheta = theta_array[i] - theta_array[i - 1]
        r_mid = (r_array[i] + r_array[i - 1]) / 2.0
        ds = np.sqrt(r_mid ** 2 + dr_dtheta ** 2) * dtheta
        arc_lengths[i] = arc_lengths[i - 1] + ds
    
    return arc_lengths

def generate_spiral_path(num_points: int,
                         spacing: float,
                         radius_value: float | None = None,
                         thick_value: float | None = None,
                         turns: int | None = None,
                         samples_per_turn: int = 1500) -> list[dict]:
    """
    Sample positions and tangents along the spiral centerline spaced by arc length.
    
    Args:
        num_points: Number of samples requested.
        spacing: Desired arc-length spacing between samples.
        radius_value: Override radius (defaults to module constant).
        thick_value: Override band thickness (defaults to module constant).
        turns: Override spiral turn count (defaults to module constant).
        samples_per_turn: Resolution for sampling the spiral (default 1500).
        
    Returns:
        List[dict]: Each entry contains {'x', 'y', 'angle'} where angle is the tangent direction in radians.
    """
    if num_points <= 0 or spacing <= 0.0:
        return []
    
    radius_use = radius if radius_value is None else radius_value
    thick_use = thick if thick_value is None else thick_value
    count_use = layer_count if turns is None else turns
    
    (_, _), (cx, cy), (theta_array, r_array) = spiral_band(radius_use, thick_use, count_use, N_per_turn=samples_per_turn)
    arc_lengths = _compute_arc_lengths(theta_array, r_array, thick_use)
    
    if len(arc_lengths) == 0:
        return []
    
    total_length = arc_lengths[-1]
    if total_length <= 0.0:
        return []
    
    results: list[dict] = []
    b = thick_use / (2.0 * np.pi)
    dr_dtheta = -b
    
    for idx_point in range(num_points):
        target_s = idx_point * spacing
        if target_s > total_length:
            break
        
        seg_idx = int(np.searchsorted(arc_lengths, target_s, side='right'))
        if seg_idx == 0:
            theta_val = theta_array[0]
            x_val = cx[0]
            y_val = cy[0]
        else:
            seg_idx = min(seg_idx, len(theta_array) - 1)
            s0 = arc_lengths[seg_idx - 1]
            s1 = arc_lengths[seg_idx]
            if s1 - s0 <= 1e-12:
                lerp = 0.0
            else:
                lerp = (target_s - s0) / (s1 - s0)
            theta0 = theta_array[seg_idx - 1]
            theta1 = theta_array[seg_idx]
            theta_val = theta0 + (theta1 - theta0) * lerp
            x0, y0 = cx[seg_idx - 1], cy[seg_idx - 1]
            x1, y1 = cx[seg_idx], cy[seg_idx]
            x_val = x0 + (x1 - x0) * lerp
            y_val = y0 + (y1 - y0) * lerp
        
        r_val = np.interp(theta_val, theta_array, r_array)
        dx_dtheta = dr_dtheta * np.cos(theta_val) - r_val * np.sin(theta_val)
        dy_dtheta = dr_dtheta * np.sin(theta_val) + r_val * np.cos(theta_val)
        angle = float(np.arctan2(dy_dtheta, dx_dtheta))
        
        results.append({
            'x': float(x_val),
            'y': float(y_val),
            'angle': angle,
        })
    
    return results


class SpiralMapper:
    """
    Helper for arc-length based evaluation of the spiral centreline and for constructing
    OCCT geometry (curve/surface) parameterised by the accumulated arc length.
    """

    def __init__(self,
                 radius_value: float | None = None,
                 thick_value: float | None = None,
                 turns: int | None = None,
                 samples_per_turn: int = 2000):
        self.radius = radius if radius_value is None else radius_value
        self.thick = thick if thick_value is None else thick_value
        self.turns = layer_count if turns is None else turns
        self.samples_per_turn = max(800, int(samples_per_turn))
        self._precompute()

    def _precompute(self):
        b = self.thick / (2.0 * np.pi)
        theta_max = 2.0 * np.pi * self.turns
        samples = max(int(self.samples_per_turn * self.turns), 2000)
        self.theta = np.linspace(0.0, theta_max, samples)
        r0 = self.radius - self.thick / 2.0
        self.r = r0 - b * self.theta
        self.dr_dtheta = -b * np.ones_like(self.theta)

        self.x = self.r * np.cos(self.theta)
        self.z = self.r * np.sin(self.theta)

        self.dx_dtheta = self.dr_dtheta * np.cos(self.theta) - self.r * np.sin(self.theta)
        self.dz_dtheta = self.dr_dtheta * np.sin(self.theta) + self.r * np.cos(self.theta)

        # Arc-length accumulation
        self.arc_lengths = np.zeros_like(self.theta)
        for i in range(1, len(self.theta)):
            dtheta = self.theta[i] - self.theta[i - 1]
            norm = np.hypot(self.dx_dtheta[i - 1], self.dz_dtheta[i - 1])
            if norm <= 0.0:
                ds = 0.0
            else:
                ds = norm * dtheta
            self.arc_lengths[i] = self.arc_lengths[i - 1] + ds

        self.total_length = float(self.arc_lengths[-1])

    def get_total_length(self) -> float:
        """Return total available arc length for the configured spiral."""
        return self.total_length

    def _interpolate_theta(self, s: float) -> Tuple[float, float, float, float]:
        """Interpolate theta and derivative information for a given arc length s."""
        if s <= 0.0:
            idx = 1
            t = 0.0
        elif s >= self.total_length:
            idx = len(self.arc_lengths) - 1
            t = 1.0
        else:
            idx = int(np.searchsorted(self.arc_lengths, s))
            if idx == 0:
                idx = 1
            s0 = self.arc_lengths[idx - 1]
            s1 = self.arc_lengths[idx]
            if abs(s1 - s0) <= 1e-12:
                t = 0.0
            else:
                t = (s - s0) / (s1 - s0)

        theta = self.theta[idx - 1] + t * (self.theta[idx] - self.theta[idx - 1])
        dx_dtheta = self.dx_dtheta[idx - 1] + t * (self.dx_dtheta[idx] - self.dx_dtheta[idx - 1])
        dz_dtheta = self.dz_dtheta[idx - 1] + t * (self.dz_dtheta[idx] - self.dz_dtheta[idx - 1])
        r_val = (self.r[idx - 1] + t * (self.r[idx] - self.r[idx - 1]))
        return theta, r_val, dx_dtheta, dz_dtheta

    def evaluate(self, s: float) -> dict:
        """
        Evaluate spiral at arc length s.

        Returns dict with position (x, z), tangent vector (3D), radial vector (3D) and theta.
        """
        theta, r_val, dx_dtheta, dz_dtheta = self._interpolate_theta(s)
        x_val = r_val * np.cos(theta)
        z_val = r_val * np.sin(theta)

        tangent = np.array([dx_dtheta, 0.0, dz_dtheta])
        norm = np.linalg.norm(tangent)
        if norm <= 1e-12:
            tangent = np.array([0.0, 0.0, 1.0])
        else:
            tangent = tangent / norm

        # Radial direction from cross product (Y axis × tangent)
        radial = np.cross(np.array([0.0, 1.0, 0.0]), tangent)
        radial_norm = np.linalg.norm(radial)
        if radial_norm <= 1e-12:
            radial = np.array([1.0, 0.0, 0.0])
        else:
            radial = radial / radial_norm

        return {
            'theta': theta,
            'x': x_val,
            'z': z_val,
            'tangent': tangent,
            'radial': radial,
        }

    def map_point(self,
                  arc_length: float,
                  y_offset: float = 0.0,
                  radial_offset: float = 0.0) -> Tuple[float, float, float]:
        """
        Map a single panel point defined by arc length (x direction), vertical offset (y)
        and thickness offset (radial) onto the 3D spiral surface.
        """
        data = self.evaluate(arc_length)
        base = np.array([data['x'], 0.0, data['z']], dtype=float)
        vertical = np.array([0.0, y_offset, 0.0], dtype=float)
        radial_vec = np.array(data['radial'], dtype=float) * radial_offset
        mapped = base + vertical + radial_vec
        return float(mapped[0]), float(mapped[1]), float(mapped[2])

    def map_curve_points(self,
                         shape_curve: List[Tuple[float, float]],
                         x_offset: float = 0.0,
                         y_offset: float = 0.0,
                         x_origin: float = 0.0,
                         radial_offset: float = 0.0,
                         x_transform=None) -> List[Tuple[float, float, float]]:
        """
        Map an entire 2D curve onto the spiral surface while preserving relative offsets.
        """
        mapped: List[Tuple[float, float, float]] = []
        for x_val, y_val in shape_curve:
            base_x = x_transform(x_val) if x_transform else x_val
            arc_length = x_offset + (base_x - x_origin)
            mapped.append(self.map_point(arc_length, y_offset + y_val, radial_offset))
        return mapped

    def build_bspline_curve(self, max_length: float):
        """
        Build an OCC Geom_BSplineCurve parameterised by arc length (s) up to max_length.
        """
        max_length = max(0.0, min(max_length, self.total_length))
        if max_length <= 0.0:
            raise ValueError("max_length must be positive and within spiral span")

        if gp_Pnt is None or GeomAPI_PointsToBSpline is None or TColgp_Array1OfPnt is None:
            raise RuntimeError("OpenCASCADE is required to build BSpline curves for the spiral")

        # Determine number of samples proportionally to requested length
        if self.total_length <= 0.0:
            raise ValueError("Spiral total length is zero; cannot build curve")

        ratio = max_length / self.total_length
        sample_count = max(5, int(self.samples_per_turn * ratio) + 2)

        s_values = np.linspace(0.0, max_length, sample_count)
        points = TColgp_Array1OfPnt(1, sample_count)

        for idx, s_val in enumerate(s_values, start=1):
            data = self.evaluate(float(s_val))
            points.SetValue(idx, gp_Pnt(float(data['x']), 0.0, float(data['z'])))

        spline_builder = GeomAPI_PointsToBSpline(points, 3, 8, GeomAbs_C2)
        return spline_builder.Curve()

    def build_offset_curve(self, offset: float, max_length: float, samples: int | None = None):
        if gp_Pnt is None or GeomAPI_PointsToBSpline is None:
            raise RuntimeError("OpenCASCADE is required to build offset curves for the spiral")

        max_length = max(0.0, min(max_length, self.total_length))
        if max_length <= 0.0:
            raise ValueError("max_length must be positive and within spiral span")

        if samples is None:
            samples = max(5, int(self.samples_per_turn * (max_length / self.total_length)) + 2)
        s_values = np.linspace(0.0, max_length, samples)

        points = TColgp_Array1OfPnt(1, samples)
        for idx, s_val in enumerate(s_values, start=1):
            data = self.evaluate(float(s_val))
            base = np.array([data['x'], 0.0, data['z']], dtype=float)
            radial = np.array(data['radial'], dtype=float) * offset
            mapped = base + radial
            points.SetValue(idx, gp_Pnt(float(mapped[0]), 0.0, float(mapped[2])))

        return GeomAPI_PointsToBSpline(points, 3, 8, GeomAbs_C2).Curve()


def generate_rectangles(theta_array, r_array, rect_width: float, rect_gap: float, thick: float):
    """
    在螺旋线上生成矩形
    
    参数:
        theta_array: 螺旋线角度数组
        r_array: 螺旋线半径数组
        rect_width: 矩形沿螺旋线方向的宽度
        rect_gap: 矩形之间的间隔
        thick: 矩形厚度（沿法线方向）
    
    返回:
        rectangles: 矩形列表，每个矩形是一个 (4x2) 的顶点数组
    """
    # 计算螺旋线的弧长
    b = thick / (2.0 * np.pi)
    
    # 使用数值积分计算累积弧长
    arc_lengths = np.zeros(len(theta_array))
    for i in range(1, len(theta_array)):
        dtheta = theta_array[i] - theta_array[i-1]
        r_mid = (r_array[i] + r_array[i-1]) / 2.0
        dr_dtheta = -b
        # ds = sqrt(r^2 + (dr/dtheta)^2) * dtheta
        ds = np.sqrt(r_mid**2 + dr_dtheta**2) * dtheta
        arc_lengths[i] = arc_lengths[i-1] + ds
    
    total_length = arc_lengths[-1]
    
    # 计算矩形位置
    rect_pitch = rect_width + rect_gap  # 矩形周期
    num_rects = int(total_length / rect_pitch)
    
    rectangles = []
    
    for i in range(num_rects):
        # 矩形起始和结束位置（沿弧长）
        s_start = i * rect_pitch
        s_end = s_start + rect_width
        
        if s_end > total_length:
            break
        
        # 找到对应的 theta 索引
        idx_start = np.searchsorted(arc_lengths, s_start)
        idx_end = np.searchsorted(arc_lengths, s_end)
        
        if idx_start >= len(theta_array) - 1 or idx_end >= len(theta_array) - 1:
            break
        
        # 获取起始和结束点的参数
        theta_s = theta_array[idx_start]
        r_s = r_array[idx_start]
        theta_e = theta_array[idx_end]
        r_e = r_array[idx_end]
        
        # 起始点和结束点坐标
        x_s, y_s = r_s * np.cos(theta_s), r_s * np.sin(theta_s)
        x_e, y_e = r_e * np.cos(theta_e), r_e * np.sin(theta_e)
        
        # 计算法线方向（在起始点和结束点）
        dr_dtheta = -b
        
        # 起始点法线
        dx_dtheta_s = dr_dtheta * np.cos(theta_s) - r_s * np.sin(theta_s)
        dy_dtheta_s = dr_dtheta * np.sin(theta_s) + r_s * np.cos(theta_s)
        nx_s = -dy_dtheta_s
        ny_s = dx_dtheta_s
        n_norm_s = np.hypot(nx_s, ny_s) + 1e-12
        nx_s /= n_norm_s
        ny_s /= n_norm_s
        
        # 结束点法线
        dx_dtheta_e = dr_dtheta * np.cos(theta_e) - r_e * np.sin(theta_e)
        dy_dtheta_e = dr_dtheta * np.sin(theta_e) + r_e * np.cos(theta_e)
        nx_e = -dy_dtheta_e
        ny_e = dx_dtheta_e
        n_norm_e = np.hypot(nx_e, ny_e) + 1e-12
        nx_e /= n_norm_e
        ny_e /= n_norm_e
        
        # 矩形四个顶点
        half_thick = thick / 2.0
        p1 = np.array([x_s + half_thick * nx_s, y_s + half_thick * ny_s])  # 起始外侧
        p2 = np.array([x_e + half_thick * nx_e, y_e + half_thick * ny_e])  # 结束外侧
        p3 = np.array([x_e - half_thick * nx_e, y_e - half_thick * ny_e])  # 结束内侧
        p4 = np.array([x_s - half_thick * nx_s, y_s - half_thick * ny_s])  # 起始内侧
        
        rect = np.array([p1, p2, p3, p4])
        rectangles.append(rect)
    
    return rectangles

def generate_gradient_rectangles_aligned(theta_array, r_array, rect_width_start: float,
                                        width_change_rate: float, rect_gap: float,
                                        thick: float, count: int, num_divisions: int = 96,
                                        force_width: float | None = None):
    """
    基于径向 96 分割生成对齐的渐变矩形。

    步骤：
        1. 从圆心发出 num_divisions 条径向分割线，与每圈螺旋线求交得到交点 N。
        2. 对每个交点 N，取前后相邻交点 A、B，并计算 NA、NB 的中点（记为 NOA、NOB）。
        3. 将 NOA、NOB 沿切向分别偏移 rect_gap/2，作为该矩形在中心线上允许的边界。
        4. 在边界内根据渐变宽度截取矩形主体，并沿法向展开 thick 厚度。

    参数:
        theta_array: 螺旋线角度数组
        r_array: 螺旋线半径数组
        rect_width_start: 起始矩形宽度
        width_change_rate: 每圈宽度变化量
        rect_gap: 相邻矩形在切向上的间隔
        thick: 矩形厚度（沿法线方向）
        count: 总圈数
        num_divisions: 径向分割数（默认 96）

    返回:
        List[np.ndarray]: 每个矩形的四个顶点（按顺时针顺序）
    """
    rectangles = []
    if num_divisions < 3:
        return rectangles

    radial_angles = np.linspace(0.0, 2.0 * np.pi, num_divisions, endpoint=False)
    gap_offset = max(rect_gap / 2.0, 0.0)
    half_thick = thick / 2.0
    eps = 1e-9

    # 预计算插值辅助数据
    dr_dtheta_array = np.gradient(r_array, theta_array)

    def eval_radius(theta_val: float) -> float:
        return float(np.interp(theta_val, theta_array, r_array))

    def eval_radius_derivative(theta_val: float) -> float:
        return float(np.interp(theta_val, theta_array, dr_dtheta_array))

    def eval_position(theta_val: float) -> np.ndarray:
        r_val = eval_radius(theta_val)
        return np.array([r_val * np.cos(theta_val), r_val * np.sin(theta_val)])

    def eval_tangent(theta_val: float, r_val: float | None = None) -> np.ndarray:
        r_v = eval_radius(theta_val) if r_val is None else r_val
        dr_v = eval_radius_derivative(theta_val)
        dx = dr_v * np.cos(theta_val) - r_v * np.sin(theta_val)
        dy = dr_v * np.sin(theta_val) + r_v * np.cos(theta_val)
        norm = np.hypot(dx, dy)
        if norm < eps:
            return np.array([0.0, 0.0])
        return np.array([dx / norm, dy / norm])

    full_turn = 2.0 * np.pi

    for turn in range(count):
        current_width = rect_width_start + turn * width_change_rate
        if force_width is not None:
            current_width = force_width
        if current_width <= 0.0:
            break

        theta_start = turn * full_turn
        theta_end = (turn + 1) * full_turn
        theta_points = radial_angles + theta_start

        if theta_points[-1] > theta_array[-1] + 1e-9:
            break

        positions = [eval_position(theta_val) for theta_val in theta_points]
        tangents = [eval_tangent(theta_val) for theta_val in theta_points]

        for idx, (theta_curr, pos_curr, tangent_curr) in enumerate(zip(theta_points, positions, tangents)):
            if np.linalg.norm(tangent_curr) < eps:
                continue

            theta_prev = theta_points[idx - 1]
            theta_next = theta_points[(idx + 1) % len(theta_points)]

            delta_prev = (theta_curr - theta_prev) % full_turn
            delta_next = (theta_next - theta_curr) % full_turn

            theta_mid_prev = theta_prev + 0.5 * delta_prev
            theta_mid_next = theta_curr + 0.5 * delta_next

            pos_mid_prev = eval_position(theta_mid_prev)
            pos_mid_next = eval_position(theta_mid_next)
            tangent_mid_prev = eval_tangent(theta_mid_prev)
            tangent_mid_next = eval_tangent(theta_mid_next)

            if np.linalg.norm(tangent_mid_prev) < eps or np.linalg.norm(tangent_mid_next) < eps:
                continue

            start_boundary = pos_mid_prev + gap_offset * tangent_mid_prev
            end_boundary = pos_mid_next - gap_offset * tangent_mid_next

            start_coord = np.dot(start_boundary - pos_curr, tangent_curr)
            end_coord = np.dot(end_boundary - pos_curr, tangent_curr)

            available_length = end_coord - start_coord
            if available_length <= eps:
                continue

            segment_start = start_coord
            segment_end = end_coord

            start_point = pos_curr + segment_start * tangent_curr
            end_point = pos_curr + segment_end * tangent_curr

            normal_dir = np.array([-tangent_curr[1], tangent_curr[0]])

            p1 = start_point + half_thick * normal_dir
            p2 = end_point + half_thick * normal_dir
            p3 = end_point - half_thick * normal_dir
            p4 = start_point - half_thick * normal_dir

            rectangles.append(np.array([p1, p2, p3, p4]))

    return rectangles

def generate_constant_count_rectangles(theta_array,
                                       r_array,
                                       rect_width: float,
                                       rect_gap: float,
                                       thick: float,
                                       count: int,
                                       constant_count: int = 96,
                                       mode: str = "fixed_count") -> dict:
    """
    生成按圈对齐的矩形，并提供图层对齐信息。

    Args:
        theta_array: 螺旋角度序列
        r_array: 对应半径序列
        rect_width: 参考矩形宽度（用于 outer_layer 模式的初始估计）
        rect_gap: 矩形间距
        thick: 螺旋厚度
        count: 圈数
        constant_count: 固定数量模式下的分段数
        mode: "fixed_count"（模式 B）或 "outer_layer"（模式 A，原 outer_width）

    Returns:
        dict 包含:
            rectangles: 顶点数组列表
            used_count: 目标分段数量
            metadata: 每个矩形的层/序号信息
            widths_per_layer: 每层计算得到的矩形宽度
            layer_lengths: 每层的弧长
            layer_rect_counts: 每层实际生成的矩形数量
            mode: 归一化后的模式名称
    """
    if rect_gap < 0.0:
        return {
            'rectangles': [],
            'used_count': 0,
            'metadata': [],
            'widths_per_layer': [],
            'layer_lengths': [],
            'layer_rect_counts': [],
            'mode': (mode or "").lower(),
        }

    if constant_count <= 0:
        constant_count = 1

    arc_lengths = _compute_arc_lengths(theta_array, r_array, thick)
    if len(arc_lengths) == 0:
        return {
            'rectangles': [],
            'used_count': 0,
            'metadata': [],
            'widths_per_layer': [],
            'layer_lengths': [],
            'layer_rect_counts': [],
            'mode': (mode or "").lower(),
        }

    dr_dtheta_array = np.gradient(r_array, theta_array)

    def theta_from_arc(s_val: float) -> float:
        s_clamped = max(arc_lengths[0], min(s_val, arc_lengths[-1]))
        return float(np.interp(s_clamped, arc_lengths, theta_array))

    def position_at_theta(theta_val: float) -> np.ndarray:
        r_val = float(np.interp(theta_val, theta_array, r_array))
        return np.array([r_val * np.cos(theta_val), r_val * np.sin(theta_val)])

    def tangent_normal_at_theta(theta_val: float) -> Tuple[np.ndarray, np.ndarray]:
        r_val = float(np.interp(theta_val, theta_array, r_array))
        dr_val = float(np.interp(theta_val, theta_array, dr_dtheta_array))
        dx = dr_val * np.cos(theta_val) - r_val * np.sin(theta_val)
        dy = dr_val * np.sin(theta_val) + r_val * np.cos(theta_val)
        norm = np.hypot(dx, dy)
        if norm < 1e-12:
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
        tangent = np.array([dx / norm, dy / norm])
        normal = np.array([-tangent[1], tangent[0]])
        return tangent, normal

    mode_key = (mode or "").lower()
    if mode_key == "outer_width":
        mode_key = "outer_layer"
    if mode_key not in ("fixed_count", "outer_layer"):
        mode_key = "fixed_count"

    rectangles: List[np.ndarray] = []
    metadata: List[dict] = []
    widths_per_layer: List[float] = []
    layer_rect_counts: List[int] = []
    half_thick = thick / 2.0
    full_turn = 2.0 * np.pi
    eps = 1e-9
    max_theta = theta_array[-1]
    used_count = max(1, int(round(constant_count)))

    layer_segments: List[Tuple[float, float, float, float]] = []
    for turn in range(count):
        theta_start = turn * full_turn
        theta_end = min((turn + 1) * full_turn, max_theta)
        if theta_start >= max_theta - eps:
            break
        if theta_end - theta_start <= eps:
            continue
        s_start = float(np.interp(theta_start, theta_array, arc_lengths))
        s_end = float(np.interp(theta_end, theta_array, arc_lengths))
        if s_end - s_start <= eps:
            continue
        layer_segments.append((theta_start, theta_end, s_start, s_end))

    if not layer_segments:
        return {
            'rectangles': [],
            'used_count': used_count,
            'metadata': [],
            'widths_per_layer': [],
            'layer_lengths': [],
            'layer_rect_counts': [],
            'mode': mode_key,
        }

    layer_lengths = [seg[3] - seg[2] for seg in layer_segments]

    if mode_key == "outer_layer":
        denom = rect_width + rect_gap
        if denom <= eps:
            used_count = max(1, used_count)
        else:
            estimated = int(np.floor(layer_lengths[0] / denom))
            used_count = max(1, estimated)
        while used_count > 1:
            min_width = min((length / used_count) - rect_gap for length in layer_lengths)
            if min_width > eps:
                break
            used_count -= 1
        used_count = max(1, used_count)

    for layer_idx, (_, _, s_start, s_end) in enumerate(layer_segments):
        layer_length = s_end - s_start
        if used_count <= 0 or layer_length <= eps:
            widths_per_layer.append(0.0)
            layer_rect_counts.append(0)
            continue
        width_val = (layer_length / used_count) - rect_gap
        widths_per_layer.append(max(0.0, width_val))
        if width_val <= eps:
            layer_rect_counts.append(0)
            continue

        current_s = s_start
        layer_generated = 0
        for div_idx in range(used_count):
            start_s = current_s
            end_s = min(s_end, start_s + width_val)

            theta0 = theta_from_arc(start_s)
            theta1 = theta_from_arc(end_s)
            pos0 = position_at_theta(theta0)
            pos1 = position_at_theta(theta1)
            tangent0, normal0 = tangent_normal_at_theta(theta0)
            tangent1, normal1 = tangent_normal_at_theta(theta1)

            if np.linalg.norm(tangent0) < eps or np.linalg.norm(tangent1) < eps:
                current_s = start_s + width_val + rect_gap
                continue

            p1 = pos0 + half_thick * normal0
            p2 = pos1 + half_thick * normal1
            p3 = pos1 - half_thick * normal1
            p4 = pos0 - half_thick * normal0
            rectangles.append(np.array([p1, p2, p3, p4]))
            metadata.append({'layer': layer_idx, 'index': div_idx})
            layer_generated += 1
            current_s = end_s + rect_gap
        layer_rect_counts.append(layer_generated)

    return {
        'rectangles': rectangles,
        'used_count': used_count,
        'metadata': metadata,
        'widths_per_layer': widths_per_layer,
        'layer_lengths': layer_lengths,
        'layer_rect_counts': layer_rect_counts,
        'mode': mode_key,
    }


def _draw_spiral_panels(ax_fixed, ax_equal, ax_gradient,
                        radius_value: float,
                        thick_value: float,
                        count_value: int,
                        rect_width_value: float,
                        rect_width_start_value: float,
                        width_change_rate_value: float,
                        rect_gap_value: float,
                        constant_mode: str = "fixed_count",
                        lock_gradient_width: bool = False,
                        constant_count_value: int = 96) -> dict:
    """Render the three comparison panels onto provided axes."""
    axes = [ax_fixed, ax_equal, ax_gradient]
    for ax in axes:
        ax.clear()

    (poly_x, poly_y), (cx, cy), (theta, r) = spiral_band(radius_value, thick_value, count_value)

    ax_fixed.fill(poly_x, poly_y, alpha=0.3, color='lightblue', label='Spiral Band')
    ax_fixed.plot(cx, cy, linewidth=0.8, color="k", label='Center Line')
    rectangles_fixed = generate_rectangles(theta, r, rect_width_value, rect_gap_value, thick_value)
    area_info_fixed = calculate_area_ratio(
        radius_value, thick_value, count_value, rectangles_fixed,
        rect_width=rect_width_value, rect_thickness=rect_thick)
    equal_data = generate_constant_count_rectangles(
        theta, r, rect_width_value, rect_gap_value, thick_value, count_value,
        constant_count=constant_count_value,
        mode=constant_mode)
    equal_rectangles = equal_data['rectangles']
    equal_per_turn = equal_data['used_count']
    equal_meta = equal_data['metadata']
    per_layer_widths = equal_data['widths_per_layer']
    layer_rect_counts = equal_data['layer_rect_counts']
    layer_lengths = equal_data['layer_lengths']
    palette = _build_rect_palette(equal_per_turn or max(1, int(round(constant_count_value))))
    for idx, rect in enumerate(rectangles_fixed):
        rect_closed = np.vstack([rect, rect[0]])
        color = _color_from_index(idx, palette)
        ax_fixed.fill(rect_closed[:, 0], rect_closed[:, 1],
                      alpha=0.7, color=color, edgecolor='darkred', linewidth=0.5)

    ax_equal.fill(poly_x, poly_y, alpha=0.3, color='lightblue', label='Spiral Band')
    ax_equal.plot(cx, cy, linewidth=0.8, color="k", label='Center Line')
    effective_turns = len(layer_lengths) if layer_lengths else count_value
    area_info_equal = calculate_area_ratio(
        radius_value, thick_value, effective_turns,
        widths_per_layer=per_layer_widths,
        count_per_layer=layer_rect_counts,
        rect_thickness=rect_thick)
    for rect, info in zip(equal_rectangles, equal_meta):
        rect_closed = np.vstack([rect, rect[0]])
        color = _color_from_index(info.get('index', 0), palette)
        ax_equal.fill(rect_closed[:, 0], rect_closed[:, 1],
                      alpha=0.7, color=color, edgecolor='darkred', linewidth=0.5)

    rectangles_gradient = generate_gradient_rectangles_aligned(
        theta, r, rect_width_start_value, width_change_rate_value, rect_gap_value,
        thick_value, count_value, force_width=rect_width_start_value if lock_gradient_width else None)
    ax_gradient.fill(poly_x, poly_y, alpha=0.3, color='lightblue', label='Spiral Band')
    ax_gradient.plot(cx, cy, linewidth=0.8, color="k", label='Center Line')
    area_info_gradient = calculate_area_ratio(
        radius_value, thick_value, count_value, rectangles_gradient,
        rect_thickness=rect_thick)
    for idx, rect in enumerate(rectangles_gradient):
        rect_closed = np.vstack([rect, rect[0]])
        color = _color_from_index(idx, palette)
        ax_gradient.fill(rect_closed[:, 0], rect_closed[:, 1],
                         alpha=0.7, color=color, edgecolor='darkred', linewidth=0.5)

    t = np.linspace(0, 2*np.pi, 720)
    lim = radius_value + thick_value
    for ax in axes:
        ax.plot(radius_value*np.cos(t), radius_value*np.sin(t), linestyle="--",
                color="gray", linewidth=0.8, label='Outer Circle')
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_fixed.set_title(
        f"Fixed Width Rectangles\nwidth={rect_width_value}, gap={rect_gap_value}\n"
        f"Area Ratio: {area_info_fixed['ratio_percent']:.2f}%"
    )

    mode_label = "Mode B: Fixed Count" if constant_mode == "fixed_count" else "Mode A: Outer Derived"
    count_text = equal_per_turn if equal_per_turn else "auto"
    ax_equal.set_title(
        f"{mode_label} Rectangles\ncount/turn={count_text}, gap={rect_gap_value}\n"
        f"Area Ratio: {area_info_equal['ratio_percent']:.2f}%"
    )

    if lock_gradient_width:
        gradient_width_desc = f"width: {rect_width_start_value} (locked)"
    else:
        gradient_width_desc = f"width: {rect_width_start_value} + {width_change_rate_value}/turn"
    ax_gradient.set_title(
        "Gradient Width Rectangles (Radially Aligned)\n"
        f"{gradient_width_desc}, gap={rect_gap_value}\n"
        f"Area Ratio: {area_info_gradient['ratio_percent']:.2f}%"
    )

    return {
        'fixed': area_info_fixed,
        'constant': area_info_equal,
        'gradient': area_info_gradient,
        'count_per_turn': equal_per_turn,
        'per_layer_widths': per_layer_widths,
        'layer_lengths': layer_lengths,
        'layer_rect_counts': layer_rect_counts,
        'mode': equal_data.get('mode', constant_mode),
    }


def plot_spiral(radius: float, thick: float, count: int, save_path: str | None = None, 
                show_rectangles: bool = True):
    (poly_x, poly_y), (cx, cy), (theta, r) = spiral_band(radius, thick, count)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill(poly_x, poly_y, alpha=0.3, color='lightblue', label='Spiral Band')
    ax.plot(cx, cy, linewidth=0.8, color="k", label='Center Line')

    # 绘制矩形
    if show_rectangles:
        rectangles = generate_rectangles(theta, r, rect_width, rect_gap, thick)
        print(f"生成了 {len(rectangles)} 个矩形")
        palette = _build_rect_palette(len(rectangles))
        for idx, rect in enumerate(rectangles):
            # 闭合矩形路径
            rect_closed = np.vstack([rect, rect[0]])
            color = _color_from_index(idx, palette)
            ax.fill(rect_closed[:, 0], rect_closed[:, 1], 
                   alpha=0.7, color=color, edgecolor='darkred', linewidth=0.5)

    # 外接圆参考线
    t = np.linspace(0, 2*np.pi, 720)
    ax.plot(radius*np.cos(t), radius*np.sin(t), linestyle="--", 
           color="gray", linewidth=0.8, label='Outer Circle')

    ax.set_aspect('equal')
    lim = radius + thick
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Spiral with Rectangles\nradius={radius}, thick={thick}, turns={count}\n"
                f"rect_width={rect_width}, gap={rect_gap}")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存图像到: {save_path}")
    plt.show()

def plot_spiral_gradient(radius: float, thick: float, count: int, 
                        rect_width_start: float, width_change_rate: float,
                        rect_gap: float, save_path: str | None = None):
    """绘制带渐变矩形的螺旋线"""
    (poly_x, poly_y), (cx, cy), (theta, r) = spiral_band(radius, thick, count)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.fill(poly_x, poly_y, alpha=0.3, color='lightblue', label='Spiral Band')
    ax.plot(cx, cy, linewidth=0.8, color="k", label='Center Line')

    # 绘制渐变矩形
    rectangles = generate_gradient_rectangles_aligned(theta, r, rect_width_start, 
                                                     width_change_rate, rect_gap, 
                                                     thick, count)
    print(f"生成了 {len(rectangles)} 个渐变矩形")
    palette = _build_rect_palette(len(rectangles))

    for idx, rect in enumerate(rectangles):
        # 闭合矩形路径
        rect_closed = np.vstack([rect, rect[0]])
        color = _color_from_index(idx, palette)
        ax.fill(rect_closed[:, 0], rect_closed[:, 1], 
               alpha=0.7, color=color, edgecolor='darkred', linewidth=0.5)

    # 外接圆参考线
    t = np.linspace(0, 2*np.pi, 720)
    ax.plot(radius*np.cos(t), radius*np.sin(t), linestyle="--", 
           color="gray", linewidth=0.8, label='Outer Circle')

    ax.set_aspect('equal')
    lim = radius + thick
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Spiral with Gradient Rectangles (Radially Aligned)\n"
                f"radius={radius}, thick={thick}, turns={count}\n"
                f"width: {rect_width_start} + {width_change_rate}/turn, gap={rect_gap}")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存图像到: {save_path}")
    plt.show()

def calculate_area_ratio(radius: float,
                         thick: float,
                         count: int,
                         rectangles: list | None = None,
                         *,
                         rect_width: float | None = None,
                         widths_per_layer: Sequence[float] | None = None,
                         count_per_layer: Sequence[int] | None = None,
                         rect_thickness: float | None = None) -> dict:
    """
    计算螺旋带面积和矩形面积的占比（按矩形宽度*厚度估算）。

    Args:
        radius: 外接圆半径
        thick: 螺旋带宽度
        count: 圈数
        rectangles: 矩形列表（可选）
        rect_width: 若提供，则所有矩形使用统一宽度
        widths_per_layer: 每层矩形宽度列表
        count_per_layer: 每层矩形数量列表（须与 widths_per_layer 长度一致）

    Returns:
        包含螺旋带面积、矩形总面积和占比的字典。
    """
    spiral_area = _calculate_spiral_area(radius, thick, count)

    rect_thickness_use = rect_thickness if rect_thickness is not None else rect_thick

    total_rect_area = 0.0
    if widths_per_layer is not None and count_per_layer is not None:
        for width_val, count_val in zip(widths_per_layer, count_per_layer):
            if width_val <= 0.0 or count_val <= 0:
                continue
            total_rect_area += float(width_val) * rect_thickness_use * int(count_val)
    elif rectangles:
        if rect_width is not None:
            total_rect_area = float(len(rectangles)) * rect_width * rect_thickness_use
        else:
            total_rect_area = rect_thickness_use * sum(_estimate_rect_width(rect) for rect in rectangles)
    elif rect_width is not None and count > 0:
        total_rect_area = rect_width * rect_thickness_use * count

    ratio = (total_rect_area / spiral_area * 100.0) if spiral_area > 1e-12 else 0.0

    return {
        'spiral_area': spiral_area,
        'rectangles_area': total_rect_area,
        'ratio_percent': ratio,
    }


def compute_manual_area_curve(layer_lengths: Sequence[float],
                              rect_gap: float,
                              radius_value: float,
                              turn_count: int,
                              division_min: int = 17,
                              division_max: int = 100,
                              rect_thickness: float | None = None,
                              spiral_thickness: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算手动分段模式下的面积占比曲线。

    返回:
        (divisions_array, ratio_array)
    """
    if division_max < division_min:
        division_min, division_max = division_max, division_min

    layer_lengths = list(layer_lengths)
    rect_thickness_use = rect_thickness if rect_thickness is not None else rect_thick
    spiral_thickness_use = spiral_thickness if spiral_thickness is not None else thick
    if not layer_lengths or turn_count <= 0 or rect_thickness_use <= 0.0:
        return np.array([]), np.array([])

    spiral_area = _calculate_spiral_area(radius_value, spiral_thickness_use, turn_count)
    if spiral_area <= 1e-12:
        return np.array([]), np.array([])

    divisions = np.arange(division_min, division_max + 1, dtype=float)
    ratios = np.zeros_like(divisions)
    for idx, div_val in enumerate(divisions):
        n = int(max(1, round(div_val)))
        widths = []
        for length in layer_lengths:
            width_val = (length / n) - rect_gap
            widths.append(width_val if width_val > 0.0 else 0.0)
        if not widths or min(widths) <= 0.0:
            ratios[idx] = 0.0
            continue
        total_area = rect_thickness_use * n * sum(widths)
        ratios[idx] = (total_area / spiral_area) * 100.0

    return divisions, ratios

def plot_both_spirals(radius: float, thick: float, count: int, 
                     rect_width: float, rect_width_start: float, 
                     width_change_rate: float, rect_gap: float,
                     save_path: str | None = None,
                     interactive: bool = False,
                     constant_mode: str = "fixed_count",
                     lock_gradient_width: bool = False):
    """在一个窗口中显示三个螺旋线策略对比图。
    
    Args:
        radius: 外接圆半径
        thick: 螺旋带宽度
        count: 圈数
        rect_width: 固定宽度策略的矩形宽度
        rect_width_start: 渐变策略的起始宽度
        width_change_rate: 渐变策略的每圈变化量
        rect_gap: 所有策略共用的矩形间隔
        save_path: 输出文件路径（可选）
        interactive: 若为 True，改为展示带 Slider 的交互界面（忽略 save_path）
        constant_mode: "fixed_count" 或 "outer_layer"（兼容旧值 "outer_width"）
    """
    if constant_mode in ("outer_layer", "outer_width"):
        mode = "outer_layer"
    elif constant_mode == "fixed_count":
        mode = "fixed_count"
    else:
        mode = "fixed_count"

    if interactive:
        return interactive_radius_slider(
            radius_min=radius * 0.2,
            radius_max=radius * 1.4,
            radius_init=radius,
            thick_value=thick,
            count_value=count,
            rect_width_value=rect_width,
            rect_width_start_value=rect_width_start,
            width_change_rate_value=width_change_rate,
            rect_gap_value=rect_gap,
            constant_mode=mode,
            lock_gradient_width=lock_gradient_width,
        )

    fig, axes = plt.subplots(1, 4, figsize=(32, 8), constrained_layout=True)
    ax1, ax2, ax3, ratio_ax = axes
    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect(1)
    ratio_ax.set_box_aspect(1)
    results = _draw_spiral_panels(
        ax1, ax2, ax3,
        radius, thick, count,
        rect_width, rect_width_start,
        width_change_rate, rect_gap,
        constant_mode=mode,
        lock_gradient_width=(mode == "outer_layer") or lock_gradient_width,
        constant_count_value=96,
    )

    ratio_ax.set_xlabel("num_divisions")
    ratio_ax.set_ylabel("Area %")
    ratio_ax.grid(True, alpha=0.3)
    div_min, div_max = AREA_DIVISION_RANGE
    ratio_ax.set_xlim(div_min, div_max)
    ratio_ax.set_ylim(0.0, 5.0)
    ratio_value_text = ratio_ax.text(
        0.02, 0.92, "", transform=ratio_ax.transAxes, fontsize=10,
        color='orange', fontweight='bold', ha='left', va='top',
    )
    layer_lengths = results.get('layer_lengths', [])
    turns_effective = len(layer_lengths) if layer_lengths else count
    ratio_value_text.set_text("")
    if mode == "fixed_count":
        divisions, ratios = compute_manual_area_curve(
            layer_lengths,
            rect_gap,
            radius,
            turns_effective,
            division_min=div_min,
            division_max=div_max,
            rect_thickness=rect_thick,
            spiral_thickness=thick,
        )
        if divisions.size > 0 and ratios.size > 0:
            ratio_ax.plot(divisions, ratios, color='purple', linewidth=1.6, alpha=0.8)
            count_val = results.get('count_per_turn')
            if count_val:
                count_clamped = max(div_min, min(div_max, count_val))
                ratio_ax.axvline(count_clamped, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
                current_ratio = results.get('constant', {}).get('ratio_percent')
                if current_ratio is not None:
                    ratio_ax.plot([count_clamped], [current_ratio],
                                  marker='o', color='orange', markersize=6)
                    ratio_value_text.set_text(
                        f"num_divisions={count_clamped}\narea={current_ratio:.2f}%"
                    )
                else:
                    ratio_value_text.set_text(f"num_divisions={count_clamped}\narea=--")
                x_min = float(min(divisions[0], count_clamped))
                x_max = float(max(divisions[-1], count_clamped))
                ratio_ax.set_xlim(x_min, x_max)
            max_ratio = float(np.max(ratios))
            ratio_ax.set_ylim(0.0, max(5.0, max_ratio * 1.1))
        else:
            ratio_value_text.set_text("No manual samples")
        ratio_ax.set_title("Area Ratio vs num_divisions (manual mode)")
    else:
        auto_ratio = results.get('constant', {}).get('ratio_percent')
        auto_count = results.get('count_per_turn')
        if auto_ratio is not None and auto_count:
            ratio_value_text.set_text(
                f"auto divisions={auto_count}\narea={auto_ratio:.2f}%"
            )
        else:
            ratio_value_text.set_text("Auto mode")
        ratio_ax.set_title("Area Ratio (manual mode only)")

    plt.suptitle(f"Spiral Comparison: radius={radius}, thick={thick}, turns={count}", 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存图像到: {save_path}")
    plt.show()
    return results


def interactive_radius_slider(radius_min: float = 2.0,
                              radius_max: float = 8.0,
                              radius_init: float | None = None,
                              *,
                              thick_value: float | None = None,
                              count_value: int | None = None,
                              rect_width_value: float | None = None,
                              rect_width_start_value: float | None = None,
                              width_change_rate_value: float | None = None,
                              rect_gap_value: float | None = None,
                              constant_mode: str = "fixed_count",
                              lock_gradient_width: bool = False):
    """
    使用 Matplotlib Slider 交互调节半径，并观察三种矩形策略的变化。
    其他参数若未提供，将回落到模块级默认值。
    """
    if radius_init is None:
        radius_init = radius
    radius_init = max(radius_min, min(radius_max, radius_init))

    thick_use = thick if thick_value is None else thick_value
    count_use = layer_count if count_value is None else count_value
    rect_width_use = rect_width if rect_width_value is None else rect_width_value
    rect_width_start_use = rect_width_start if rect_width_start_value is None else rect_width_start_value
    width_change_rate_use = width_change_rate if width_change_rate_value is None else width_change_rate_value
    rect_gap_use = rect_gap if rect_gap_value is None else rect_gap_value

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))
    ax1, ax2, ax3, ratio_ax = axes
    plt.subplots_adjust(bottom=0.32, wspace=0.25, top=0.88)
    for ax in (ax1, ax2, ax3):
        ax.set_box_aspect(1)
    ratio_ax.set_box_aspect(1)
    ratio_ax.set_title("Area Ratio (Rect width × thickness)")
    ratio_ax.set_xlabel("num_divisions")
    ratio_ax.set_ylabel("Area %")
    ratio_ax.grid(True, alpha=0.3)
    div_min, div_max = AREA_DIVISION_RANGE
    ratio_ax.set_xlim(div_min, div_max)
    ratio_ax.set_ylim(0.0, 5.0)
    ratio_line, = ratio_ax.plot([], [], color='purple', linewidth=1.6, alpha=0.8)
    ratio_marker, = ratio_ax.plot([], [], marker='o', color='orange', markersize=6, linestyle='None')
    ratio_marker.set_visible(False)
    ratio_current_line = ratio_ax.axvline(div_min, color='gray', linestyle='--', linewidth=0.9, alpha=0.7)
    ratio_current_line.set_visible(False)
    ratio_value_text = ratio_ax.text(
        0.02, 0.92, "", transform=ratio_ax.transAxes, fontsize=10,
        color='orange', fontweight='bold', ha='left', va='top',
    )

    mode_state = (constant_mode or "").lower()
    if mode_state == "outer_width":
        mode_state = "outer_layer"
    if mode_state not in ("fixed_count", "outer_layer"):
        mode_state = "fixed_count"

    suptitle = fig.suptitle("", fontsize=14, fontweight='bold')

    slider_ax = fig.add_axes([0.2, 0.20, 0.6, 0.03])
    radius_slider = Slider(
        slider_ax,
        "Radius",
        radius_min,
        radius_max,
        valinit=radius_init,
        valstep=0.01,
    )

    division_ax = fig.add_axes([0.2, 0.14, 0.6, 0.03])
    division_slider = Slider(
        division_ax,
        "Divisions",
        AREA_DIVISION_RANGE[0],
        AREA_DIVISION_RANGE[1],
        valinit=96,
        valstep=1,
    )

    constant_button_ax = fig.add_axes([0.25, 0.05, 0.5, 0.05])
    constant_button = Button(constant_button_ax, "")

    _division_slider_updating = False

    def _apply(radius_value: float | None = None):
        nonlocal _division_slider_updating
        val = radius_slider.val if radius_value is None else radius_value
        divisions_val = int(round(division_slider.val))
        info = _draw_spiral_panels(
            ax1, ax2, ax3,
            val, thick_use, count_use,
            rect_width_use, rect_width_start_use,
            width_change_rate_use, rect_gap_use,
            constant_mode=mode_state,
            lock_gradient_width=(mode_state == "outer_layer"),
            constant_count_value=divisions_val,
        )
        count_actual = info.get('count_per_turn', divisions_val)
        current_ratio = info.get('constant', {}).get('ratio_percent')
        layer_lengths = info.get('layer_lengths', [])
        turns_effective = len(layer_lengths) if layer_lengths else count_use
        ratio_value_text.set_text("")
        if mode_state == "fixed_count":
            divs, ratios = compute_manual_area_curve(
                layer_lengths,
                rect_gap_use,
                val,
                turns_effective,
                division_min=div_min,
                division_max=div_max,
                rect_thickness=rect_thick,
                spiral_thickness=thick_use,
            )
            has_curve = divs.size > 0 and ratios.size > 0
            marker_x = count_actual if count_actual else divisions_val
            marker_clamped = None
            if marker_x:
                marker_clamped = max(div_min, min(div_max, marker_x))

            if has_curve:
                ratio_line.set_data(divs, ratios)
                if marker_clamped is not None:
                    x_min = float(min(divs[0], marker_clamped))
                    x_max = float(max(divs[-1], marker_clamped))
                else:
                    x_min = float(divs[0])
                    x_max = float(divs[-1])
                ratio_ax.set_xlim(x_min, x_max)
                max_ratio = float(np.max(ratios))
                ratio_ax.set_ylim(0.0, max(5.0, max_ratio * 1.1))
            else:
                ratio_line.set_data([], [])
                ratio_ax.set_xlim(div_min, div_max)
                ratio_ax.set_ylim(0.0, 5.0)

            if marker_clamped is not None:
                ratio_current_line.set_xdata([marker_clamped, marker_clamped])
                ratio_current_line.set_visible(True)
                if current_ratio is not None:
                    ratio_marker.set_data([marker_clamped], [current_ratio])
                    ratio_marker.set_visible(True)
                    ratio_value_text.set_text(
                        f"num_divisions={marker_clamped}\narea={current_ratio:.2f}%"
                    )
                else:
                    ratio_marker.set_data([], [])
                    ratio_marker.set_visible(False)
                    ratio_value_text.set_text(f"num_divisions={marker_clamped}\narea=--")
            else:
                ratio_current_line.set_visible(False)
                ratio_marker.set_visible(False)
                ratio_value_text.set_text("No manual value")
            ratio_ax.set_title("Area Ratio vs num_divisions (manual mode)")
        else:
            ratio_line.set_data([], [])
            ratio_marker.set_data([], [])
            ratio_marker.set_visible(False)
            ratio_current_line.set_visible(False)
            ratio_ax.set_xlim(div_min, div_max)
            ratio_ax.set_ylim(0.0, 5.0)
            auto_ratio = info.get('constant', {}).get('ratio_percent')
            if count_actual and auto_ratio is not None:
                ratio_value_text.set_text(
                    f"auto divisions={count_actual}\narea={auto_ratio:.2f}%"
                )
            else:
                ratio_value_text.set_text("Auto mode")
            ratio_ax.set_title("Area Ratio (manual mode only)")

        if mode_state == "outer_layer" and count_actual:
            if abs(division_slider.val - count_actual) > 1e-6:
                _division_slider_updating = True
                division_slider.set_val(count_actual)
                _division_slider_updating = False
        _update_controls(count_actual)
        suptitle.set_text(
            f"Spiral Comparison: radius={val:.3f}, thick={thick_use}, turns={count_use}"
        )
        fig.canvas.draw_idle()
        return info

    def _update_controls(actual_count: int | None = None):
        mode_text = "Mode: B (Fixed Count)" if mode_state == "fixed_count" else "Mode: A (Outer Derived)"
        constant_button.label.set_text(mode_text)
        if mode_state == "outer_layer":
            label = f"Divisions (auto={actual_count})" if actual_count else "Divisions (auto)"
            division_slider.label.set_text(label)
        else:
            division_slider.label.set_text("Divisions (17-100)")
        fig.canvas.draw_idle()

    def _on_radius_change(val):
        _apply(val)

    def _toggle_constant(event):
        nonlocal mode_state
        mode_state = "outer_layer" if mode_state == "fixed_count" else "fixed_count"
        _update_controls()
        _apply()

    def _on_division_change(val):
        if _division_slider_updating or mode_state == "outer_layer":
            return
        _apply()

    radius_slider.on_changed(_on_radius_change)
    division_slider.on_changed(_on_division_change)
    constant_button.on_clicked(_toggle_constant)

    _update_controls()
    _apply(radius_init)
    plt.show()
    return radius_slider

if __name__ == "__main__":
    print("=" * 50)
    print("绘制螺旋线对比图（按需拖动滑块调整半径）...")
    print("=" * 50)
    interactive_radius_slider()
