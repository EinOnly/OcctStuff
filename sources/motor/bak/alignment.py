# spiral_plot.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

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
count  = 6        # 卷绕圈数
save_path = "spiral.png"

# ===== 矩形参数 =====
rect_width = 0.3777   # 矩形沿螺旋线方向的宽度
rect_gap = 0.05       # 矩形之间的间隔
# 矩形厚度 = thick（沿法线方向）

# ===== 渐变矩形参数 =====
rect_width_start = 0.3777  # 起始宽度
width_change_rate = -0.02  # 每圈宽度变化量（负数表示递减）

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
    count_use = count if turns is None else turns
    
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
        self.turns = count if turns is None else turns
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
                                        thick: float, count: int, num_divisions: int = 96):
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
        
        for rect in rectangles:
            # 闭合矩形路径
            rect_closed = np.vstack([rect, rect[0]])
            ax.fill(rect_closed[:, 0], rect_closed[:, 1], 
                   alpha=0.7, color='orange', edgecolor='red', linewidth=0.5)

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
    
    # 使用颜色映射显示不同圈的矩形
    colors = plt.cm.rainbow(np.linspace(0, 1, len(rectangles)))
    
    for idx, rect in enumerate(rectangles):
        # 闭合矩形路径
        rect_closed = np.vstack([rect, rect[0]])
        ax.fill(rect_closed[:, 0], rect_closed[:, 1], 
               alpha=0.7, color=colors[idx], edgecolor='darkred', linewidth=0.5)

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

def calculate_area_ratio(radius: float, thick: float, count: int, rectangles: list) -> dict:
    """
    计算螺旋带面积和矩形面积的占比
    
    参数:
        radius: 外接圆半径
        thick: 螺旋带宽度
        count: 卷绕圈数
        rectangles: 矩形列表
    
    返回:
        字典包含：螺旋带面积、矩形总面积、占比
    """
    # 计算螺旋带面积
    # 螺旋带可以看作是一个环形带，外半径从 radius 到内半径
    b = thick / (2.0 * np.pi)
    r_outer = radius
    r_inner = radius - thick / 2.0 - b * 2.0 * np.pi * count
    
    # 使用更精确的方法：计算每一圈的面积并累加
    spiral_area = 0.0
    for turn in range(count):
        # 当前圈的外半径和内半径
        r_out = radius - b * 2.0 * np.pi * turn
        r_in = radius - b * 2.0 * np.pi * (turn + 1)
        
        # 环形面积
        ring_area = np.pi * (r_out**2 - r_in**2)
        spiral_area += ring_area
    
    # 计算矩形总面积（使用 Shoelace 公式）
    total_rect_area = 0.0
    for rect in rectangles:
        # Shoelace 公式计算多边形面积
        x = rect[:, 0]
        y = rect[:, 1]
        area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x)-1)))
        total_rect_area += area
    
    # 计算占比
    ratio = (total_rect_area / spiral_area * 100) if spiral_area > 0 else 0.0
    
    return {
        'spiral_area': spiral_area,
        'rectangles_area': total_rect_area,
        'ratio_percent': ratio
    }

def plot_both_spirals(radius: float, thick: float, count: int, 
                     rect_width: float, rect_width_start: float, 
                     width_change_rate: float, rect_gap: float,
                     save_path: str | None = None):
    """在一个窗口中显示两个螺旋线"""
    (poly_x, poly_y), (cx, cy), (theta, r) = spiral_band(radius, thick, count)
    
    # 创建一个包含两个子图的窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === 左图：固定宽度矩形 ===
    ax1.fill(poly_x, poly_y, alpha=0.3, color='lightblue', label='Spiral Band')
    ax1.plot(cx, cy, linewidth=0.8, color="k", label='Center Line')
    
    # 生成固定宽度矩形
    rectangles_fixed = generate_rectangles(theta, r, rect_width, rect_gap, thick)
    print(f"固定宽度：生成了 {len(rectangles_fixed)} 个矩形")
    
    # 计算面积占比
    area_info_fixed = calculate_area_ratio(radius, thick, count, rectangles_fixed)
    print(f"固定宽度 - 螺旋带面积: {area_info_fixed['spiral_area']:.4f}")
    print(f"固定宽度 - 矩形总面积: {area_info_fixed['rectangles_area']:.4f}")
    print(f"固定宽度 - 占比: {area_info_fixed['ratio_percent']:.2f}%")
    
    for rect in rectangles_fixed:
        rect_closed = np.vstack([rect, rect[0]])
        ax1.fill(rect_closed[:, 0], rect_closed[:, 1], 
               alpha=0.7, color='orange', edgecolor='red', linewidth=0.5)
    
    # 外接圆参考线
    t = np.linspace(0, 2*np.pi, 720)
    ax1.plot(radius*np.cos(t), radius*np.sin(t), linestyle="--", 
           color="gray", linewidth=0.8, label='Outer Circle')
    
    ax1.set_aspect('equal')
    lim = radius + thick
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"Fixed Width Rectangles\nwidth={rect_width}, gap={rect_gap}\n"
                 f"Area Ratio: {area_info_fixed['ratio_percent']:.2f}%")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # === 右图：渐变宽度矩形 ===
    ax2.fill(poly_x, poly_y, alpha=0.3, color='lightblue', label='Spiral Band')
    ax2.plot(cx, cy, linewidth=0.8, color="k", label='Center Line')
    
    # 生成渐变宽度矩形
    rectangles_gradient = generate_gradient_rectangles_aligned(theta, r, rect_width_start, 
                                                              width_change_rate, rect_gap, 
                                                              thick, count)
    print(f"渐变宽度：生成了 {len(rectangles_gradient)} 个矩形")
    
    # 计算面积占比
    area_info_gradient = calculate_area_ratio(radius, thick, count, rectangles_gradient)
    print(f"渐变宽度 - 螺旋带面积: {area_info_gradient['spiral_area']:.4f}")
    print(f"渐变宽度 - 矩形总面积: {area_info_gradient['rectangles_area']:.4f}")
    print(f"渐变宽度 - 占比: {area_info_gradient['ratio_percent']:.2f}%")
    
    # 使用颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(rectangles_gradient)))
    
    for idx, rect in enumerate(rectangles_gradient):
        rect_closed = np.vstack([rect, rect[0]])
        ax2.fill(rect_closed[:, 0], rect_closed[:, 1], 
               alpha=0.7, color=colors[idx], edgecolor='darkred', linewidth=0.5)
    
    # 外接圆参考线
    ax2.plot(radius*np.cos(t), radius*np.sin(t), linestyle="--", 
           color="gray", linewidth=0.8, label='Outer Circle')
    
    ax2.set_aspect('equal')
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(f"Gradient Width Rectangles (Radially Aligned)\n"
                 f"width: {rect_width_start} + {width_change_rate}/turn, gap={rect_gap}\n"
                 f"Area Ratio: {area_info_gradient['ratio_percent']:.2f}%")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Spiral Comparison: radius={radius}, thick={thick}, turns={count}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"已保存图像到: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("=" * 50)
    print("绘制螺旋线对比图...")
    print("=" * 50)
    plot_both_spirals(radius, thick, count, rect_width, rect_width_start, 
                     width_change_rate, rect_gap, "spiral_comparison.png")
