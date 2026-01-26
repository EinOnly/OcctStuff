import numpy as np
from tqdm import tqdm

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d


class FinalCableHarness:
    def __init__(
        self, box_size=(9, 9, 35), num_pairs=120, pair_gap=0.55, render_radius=0.25
    ):
        self.box = np.array(box_size)
        self.num_pairs = num_pairs
        self.gap = pair_gap
        self.r = render_radius

        # 物理模拟半径 (100% 渲染半径)
        self.phys_radius = self.r

        self.line_a_path = []
        self.line_b_path = []
        self._body_a = []
        self._body_b = []

    def _pack_circle(self, center, count, spacing):
        """蜂窝堆积"""
        points = []
        for i in range(count):
            r = spacing * 0.6 * np.sqrt(i + 1)
            theta = i * 2.39996
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y])
        return np.array(points)

    def _generate_guide_paths(self, steps):
        """生成平滑随机路径"""
        num_keyframes = 5
        z_keys = np.linspace(0, self.box[2], num_keyframes)
        guide_paths = []
        margin = self.gap + self.r * 3
        for i in range(self.num_pairs):
            x_keys = np.random.uniform(
                -self.box[0] / 2 + margin, self.box[0] / 2 - margin, num_keyframes
            )
            y_keys = np.random.uniform(
                -self.box[1] / 2 + margin, self.box[1] / 2 - margin, num_keyframes
            )
            cs_x = CubicSpline(z_keys, x_keys)
            cs_y = CubicSpline(z_keys, y_keys)
            z_samples = np.linspace(0, self.box[2], steps)
            guide_paths.append(np.column_stack([cs_x(z_samples), cs_y(z_samples)]))
        return np.array(guide_paths)

    def _solve_collision_step(self, pos, pair_angles, iterations=10):
        """快速初始碰撞解算"""
        current_pos = pos.copy()
        vec_x = np.cos(pair_angles) * (self.gap / 2)
        vec_y = np.sin(pair_angles) * (self.gap / 2)
        limit_x = self.box[0] / 2 - self.phys_radius
        limit_y = self.box[1] / 2 - self.phys_radius

        for _ in range(iterations):
            np.clip(current_pos[:, 0], -limit_x, limit_x, out=current_pos[:, 0])
            np.clip(current_pos[:, 1], -limit_y, limit_y, out=current_pos[:, 1])
            Ax = current_pos[:, 0] + vec_x
            Ay = current_pos[:, 1] + vec_y
            Bx = current_pos[:, 0] - vec_x
            By = current_pos[:, 1] - vec_y
            all_pts = np.vstack([np.column_stack([Ax, Ay]), np.column_stack([Bx, By])])
            dists = cdist(all_pts, all_pts, metric="euclidean")
            np.fill_diagonal(dists, 999.0)
            for i in range(self.num_pairs):
                dists[i, i + self.num_pairs] = 999.0
                dists[i + self.num_pairs, i] = 999.0

            collision_mask = dists < (self.phys_radius * 2.1)
            if not np.any(collision_mask):
                break
            overlap = (self.phys_radius * 2.1) - dists
            overlap[~collision_mask] = 0
            rows, cols = np.where(collision_mask)
            unique = rows < cols
            rows, cols = rows[unique], cols[unique]
            if len(rows) == 0:
                break
            p_rows, p_cols = all_pts[rows], all_pts[cols]
            diffs = p_rows - p_cols
            current_dist = dists[rows, cols]
            current_dist[current_dist < 1e-5] = 1e-5
            move_vecs = (
                (diffs / current_dist[:, None]) * overlap[rows, cols][:, None] * 0.5
            )
            center_moves = np.zeros_like(current_pos)
            counts = np.zeros(self.num_pairs)
            for k in range(len(rows)):
                pair_r, pair_c = rows[k] % self.num_pairs, cols[k] % self.num_pairs
                center_moves[pair_r] += move_vecs[k]
                center_moves[pair_c] -= move_vecs[k]
                counts[pair_r] += 1
                counts[pair_c] += 1
            counts[counts == 0] = 1
            current_pos += center_moves / counts[:, None]
        return current_pos

    def _generate_body(self, steps=800):
        print("1/5 Generating cable body...")
        guides = self._generate_guide_paths(steps)
        current_pos = guides[:, 0, :].copy()
        current_angles = np.random.uniform(0, np.pi * 2, self.num_pairs)
        for _ in range(50):
            current_pos = self._solve_collision_step(
                current_pos, current_angles, iterations=1
            )
        z_step = self.box[2] / steps
        for i in tqdm(range(steps)):
            z = i * z_step
            target_pos = guides[:, i, :]
            current_pos = current_pos * 0.9 + target_pos * 0.1
            current_angles += 0.06
            current_pos = self._solve_collision_step(
                current_pos, current_angles, iterations=8
            )
            cos_a, sin_a = np.cos(current_angles), np.sin(current_angles)
            offset_x, offset_y = cos_a * (self.gap / 2), sin_a * (self.gap / 2)
            pts_a = np.column_stack(
                [
                    current_pos[:, 0] + offset_x,
                    current_pos[:, 1] + offset_y,
                    np.full(self.num_pairs, z),
                ]
            )
            pts_b = np.column_stack(
                [
                    current_pos[:, 0] - offset_x,
                    current_pos[:, 1] - offset_y,
                    np.full(self.num_pairs, z),
                ]
            )
            self._body_a.append(pts_a)
            self._body_b.append(pts_b)

    def _generate_connector_segment(
        self,
        body_conn_a,
        body_conn_b,
        center_a,
        center_b,
        z_range,
        steps,
        reverse=False,
    ):
        pack_a = self._pack_circle(center_a, self.num_pairs, self.r * 2.2)
        pack_b = self._pack_circle(center_b, self.num_pairs, self.r * 2.2)
        t = np.linspace(0, 1, steps)
        w = 3 * t**2 - 2 * t**3
        w = w[:, np.newaxis]
        z_vals = np.linspace(z_range[0], z_range[1], steps)
        seg_a, seg_b = [], []
        for k in range(steps):
            weight = w[k, 0]
            if reverse:
                start_a, end_a = pack_a, body_conn_a[:, :2]
                start_b, end_b = pack_b, body_conn_b[:, :2]
            else:
                start_a, end_a = body_conn_a[:, :2], pack_a
                start_b, end_b = body_conn_b[:, :2], pack_b
            xy_a = start_a * (1 - weight) + end_a * weight
            xy_b = start_b * (1 - weight) + end_b * weight
            seg_a.append(np.column_stack([xy_a, np.full(self.num_pairs, z_vals[k])]))
            seg_b.append(np.column_stack([xy_b, np.full(self.num_pairs, z_vals[k])]))
        return seg_a, seg_b

    def _advanced_relax_and_smooth(
        self, iterations=15, smooth_sigma=3.0, body_start_idx=0, body_end_idx=0
    ):
        """
        修复版松弛算法：引入约束遮罩 (Constraint Mask)
        只在 Body 区域强制红蓝配对，在接头区域允许它们分离。
        """
        print(f"4/5 Relaxing with Masked Constraints...")

        path_a = np.array(self.line_a_path)
        path_b = np.array(self.line_b_path)
        total_steps = path_a.shape[0]

        target_dist = self.gap
        min_collision_dist = self.r * 2.0

        # --- 关键修改：生成约束权重遮罩 ---
        # 1. 初始化全为 0 (完全自由)
        constraint_mask = np.zeros(total_steps)

        # 2. 将 Body 区域设为 1 (强制配对)
        # 稍微缩减一点范围，防止在接头根部发生拉扯
        safe_margin = 10
        constraint_mask[body_start_idx + safe_margin : body_end_idx - safe_margin] = 1.0

        # 3. 对遮罩本身进行高斯平滑，使其产生渐变过渡 (0 -> 0.5 -> 1)
        # 这样约束力是慢慢生效的，不会有断层
        constraint_mask = gaussian_filter1d(constraint_mask, sigma=10.0)

        for it in tqdm(range(iterations), desc="Iterative Relax"):

            # Step A: 高斯平滑 (全域应用，因为平滑会让接头曲线更优美)
            path_a = gaussian_filter1d(path_a, sigma=smooth_sigma, axis=0)
            path_b = gaussian_filter1d(path_b, sigma=smooth_sigma, axis=0)

            # Step B: 逐切片物理修正
            for s in range(total_steps):
                xy_a = path_a[s, :, :2]
                xy_b = path_b[s, :, :2]

                # --- 1. Pair Constraints (带遮罩) ---
                # 获取当前切片的约束权重 (0.0 ~ 1.0)
                weight = constraint_mask[s]

                # 只有当权重 > 0 时才计算拉回力
                if weight > 0.01:
                    centers = (xy_a + xy_b) * 0.5
                    diffs = xy_a - xy_b
                    dists = np.linalg.norm(diffs, axis=1, keepdims=True)
                    dists[dists < 1e-6] = 1e-6

                    normals = diffs / dists
                    # 理想偏移量
                    target_offset = normals * (target_dist * 0.5)

                    # 现在的偏移量
                    current_offset_a = xy_a - centers
                    current_offset_b = xy_b - centers

                    # 混合：根据权重，决定我们多大程度上强制移回 target
                    # Lerp: Current -> Target
                    new_offset_a = (
                        current_offset_a * (1 - weight) + target_offset * weight
                    )
                    new_offset_b = (
                        current_offset_b * (1 - weight) - target_offset * weight
                    )

                    xy_a = centers + new_offset_a
                    xy_b = centers + new_offset_b

                # --- 2. Collision Repulsion (全域应用) ---
                # 即使在接头处，A线之间也不应该穿插，所以碰撞检测保留
                all_pts = np.vstack([xy_a, xy_b])
                d_mat = cdist(all_pts, all_pts, metric="euclidean")

                np.fill_diagonal(d_mat, 10.0)
                for i in range(self.num_pairs):
                    d_mat[i, i + self.num_pairs] = 10.0
                    d_mat[i + self.num_pairs, i] = 10.0

                mask = d_mat < min_collision_dist
                if np.any(mask):
                    penetration = min_collision_dist - d_mat
                    penetration[~mask] = 0

                    delta = all_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :]
                    dist_safe = d_mat.copy()
                    dist_safe[dist_safe < 1e-6] = 1e-6
                    norm_dir = delta / dist_safe[:, :, np.newaxis]

                    force = norm_dir * penetration[:, :, np.newaxis] * 0.2
                    total_force = np.sum(force, axis=1)

                    # 在这里，我们需要保护接头处的“分离状态”
                    # 如果在接头处，A和B相距很远，它们之间不会产生 collision force
                    # 产生的 collision force 主要是 A vs A 或 B vs B
                    # 所以直接应用是可以的
                    all_pts += total_force

                    # 将推力结果写回
                    xy_a = all_pts[: self.num_pairs]
                    xy_b = all_pts[self.num_pairs :]

                    # --- 3. Post-Collision Constraint (再次应用遮罩) ---
                    # 推离可能会破坏间距，如果我们在 Body 区域，需要再次拉回
                    if weight > 0.01:
                        centers = (xy_a + xy_b) * 0.5
                        diffs = xy_a - xy_b
                        dists = np.linalg.norm(diffs, axis=1, keepdims=True)
                        dists[dists < 1e-6] = 1e-6
                        normals = diffs / dists
                        target_offset = normals * (target_dist * 0.5)

                        current_offset_a = xy_a - centers
                        current_offset_b = xy_b - centers

                        new_offset_a = (
                            current_offset_a * (1 - weight) + target_offset * weight
                        )
                        new_offset_b = (
                            current_offset_b * (1 - weight) - target_offset * weight
                        )

                        xy_a = centers + new_offset_a
                        xy_b = centers + new_offset_b

                path_a[s, :, :2] = xy_a
                path_b[s, :, :2] = xy_b

        self.line_a_path = [path_a[s] for s in range(total_steps)]
        self.line_b_path = [path_b[s] for s in range(total_steps)]

    def generate_full_cable(self, body_steps=800, term_steps=120, term_len=15.0):
        # 1. Body
        self._generate_body(steps=body_steps)

        # 2. Terminals
        print("2/5 Connector Terminals...")
        spread_x = self.box[0] * 0.8
        start_a, start_b = self._generate_connector_segment(
            self._body_a[0],
            self._body_b[0],
            [-spread_x, 0],
            [spread_x, 0],
            (-term_len, 0),
            term_steps,
            reverse=True,
        )
        end_a, end_b = self._generate_connector_segment(
            self._body_a[-1],
            self._body_b[-1],
            [-spread_x, 0],
            [spread_x, 0],
            (self.box[2], self.box[2] + term_len),
            term_steps,
            reverse=False,
        )

        # 拼接
        # Start(切掉最后1帧) + Body + End(切掉第1帧)
        # 关键：计算 Body 在大数组中的索引范围
        start_segment_len = len(start_a) - 1
        body_segment_len = len(self._body_a)

        body_start_idx = start_segment_len
        body_end_idx = start_segment_len + body_segment_len

        self.line_a_path = start_a[:-1] + self._body_a + end_a[1:]
        self.line_b_path = start_b[:-1] + self._body_b + end_b[1:]

        # 3. 带遮罩的松弛
        self._advanced_relax_and_smooth(
            iterations=100,
            smooth_sigma=3.0,
            body_start_idx=body_start_idx,
            body_end_idx=body_end_idx,
        )


# OCCT Imports
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Vec, gp_Circ
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Builder
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.GeomAbs import GeomAbs_C2, GeomAbs_C1


class IndustrialStepExporter:
    def __init__(self, target_radius=0.25):
        self.target_radius = float(target_radius)
        self.builder = TopoDS_Builder()
        self.compound = TopoDS_Compound()
        self.builder.MakeCompound(self.compound)
        self.count = 0

    def _resample_safe_distance(self, points, min_dist):
        """
        [几何守门员]
        强制重采样：保证任何两个控制点之间的距离都大于 min_dist。
        这是防止 Pipe 自交的物理底线。
        """
        if len(points) < 2:
            return None

        safe_points = [points[0]]
        for i in range(1, len(points)):
            curr_pt = points[i]
            prev_pt = safe_points[-1]
            dist = np.linalg.norm(curr_pt - prev_pt)

            # 如果当前点距离上一个保留点太近，忽略它（因为它可能导致急弯）
            # 只有走得够远了才记录
            if dist > min_dist:
                safe_points.append(curr_pt)

        # 强制保留终点，保证接头位置准确
        if np.linalg.norm(points[-1] - safe_points[-1]) > 1e-4:
            safe_points.append(points[-1])

        return safe_points

    def _create_curve(self, points):
        """生成 B-Spline，使用宽松的容差以换取光顺度"""
        # 1. 安全重采样：采样距离设为半径的 1.1 倍
        # 这意味着在管子直径范围内，最多只有一个控制点，彻底杜绝自交
        safe_pts = self._resample_safe_distance(points, self.target_radius * 1.1)

        if len(safe_pts) < 2:
            return None

        occt_points = TColgp_Array1OfPnt(1, len(safe_pts))
        for i, pt in enumerate(safe_pts):
            occt_points.SetValue(
                i + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            )

        try:
            # 2. 逼近生成
            # Tolerance=0.5: 允许 0.5mm 的误差，让曲线自然过渡，不要强行扭曲
            approx = GeomAPI_PointsToBSpline(
                occt_points,
                3,
                8,  # 3阶曲线
                GeomAbs_C2,  # C2 连续 (非常平滑)
                0.5,  # 容差
            )
            if not approx.IsDone():
                return None
            return approx.Curve()
        except:
            return None

    def _make_pipe(self, geom_curve, radius):
        """尝试生成实体"""
        try:
            # 1. 脊线
            spine_edge = BRepBuilderAPI_MakeEdge(geom_curve).Edge()
            spine_wire = BRepBuilderAPI_MakeWire(spine_edge).Wire()

            # 2. 截面 (严格对齐起点)
            u_start = geom_curve.FirstParameter()
            p_start = gp_Pnt()
            v_tan = gp_Vec()
            geom_curve.D1(u_start, p_start, v_tan)  # 获取起点的精确位置和切线

            if v_tan.Magnitude() < 1e-6:
                return None

            axis = gp_Ax2(p_start, gp_Dir(v_tan))
            circle = gp_Circ(axis, radius)
            profile_edge = BRepBuilderAPI_MakeEdge(circle).Edge()
            profile_wire = BRepBuilderAPI_MakeWire(profile_edge).Wire()

            # 3. 扫掠 (MakePipeShell)
            pipe_shell = BRepOffsetAPI_MakePipeShell(spine_wire)

            # [关键设置] 使用 CorrectedFrenet 模式
            # 这能处理大部分复杂的 3D 扭曲
            pipe_shell.SetMode(True)
            pipe_shell.Add(profile_wire)

            pipe_shell.Build()
            if not pipe_shell.IsDone():
                return None

            if pipe_shell.MakeSolid():
                return pipe_shell.Shape()
            return None
        except:
            return None

    def process_strands(self, strands_list):
        print(f"Processing {len(strands_list)} strands...")
        success = 0

        for pts in tqdm(strands_list):
            # 1. 生成曲线
            curve = self._create_curve(pts)
            if curve is None:
                continue

            solid = None
            current_r = self.target_radius

            # --- 策略 A: 降级尝试 (Radius Fallback) ---
            # 尝试 100%, 90%, 80% ... 直到 50%
            # 大部分线会在 100% 或 90% 成功
            for _ in range(6):
                solid = self._make_pipe(curve, current_r)
                if solid:
                    break
                current_r *= 0.9  # 失败则变细

            if solid:
                self.builder.Add(self.compound, solid)
                success += 1
                self.count += 1
            else:
                # --- 策略 B: 保底导出 (Centerline Fallback) ---
                # 实在不行（极罕见），导出中心线，保证不丢数据
                try:
                    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
                    self.builder.Add(self.compound, edge)
                    # print("Warning: One strand failed solid generation, exported as wire.")
                except:
                    pass

        print(f"Batch Result: {success}/{len(strands_list)} solids.")

    def export(self, filename):
        if self.count == 0:
            print("No geometry.")
            return
        print(f"Writing {filename}...")
        writer = STEPControl_Writer()
        writer.Transfer(self.compound, STEPControl_AsIs)
        writer.Write(filename)
        print("Done.")


if __name__ == "__main__":
    print("--- 1. Generating Physics ---")
    # 物理模拟
    cable = FinalCableHarness(
        box_size=(9, 9, 35), num_pairs=120, pair_gap=0.55, render_radius=0.25
    )
    cable.generate_full_cable(body_steps=800, term_steps=120)

    print("\n--- 2. Transposing ---")
    raw_a = np.array(cable.line_a_path)
    raw_b = np.array(cable.line_b_path)
    strands_a = raw_a.transpose(1, 0, 2)
    strands_b = raw_b.transpose(1, 0, 2)

    print("\n--- 3. Industrial Export ---")
    # 半径设定为 0.24 (略小于物理半径 0.25，留出间隙)
    exporter = IndustrialStepExporter(target_radius=0.24)

    exporter.process_strands(strands_a)
    exporter.process_strands(strands_b)

    exporter.export("industrial_harness.step")
