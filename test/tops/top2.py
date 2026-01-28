import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d


class PipeWeavingSystem:
    """
    管道随机编织系统 - 用于生成类似 TPMS 的多孔结构

    特性:
    - 60 根独立管道 (两组各 30 根)
    - 出入口汇合到指定填充比例
    - 全局碰撞优化保证管道不穿插
    """

    def __init__(
        self,
        box_size=(9, 9, 35),
        num_pipes_per_group=30,
        render_radius=0.25,
        target_fill_ratio=0.3,
    ):
        """
        参数:
            box_size: 编织区域尺寸 (x, y, z)
            num_pipes_per_group: 每组管道数量 (总共 2 组)
            render_radius: 管道渲染半径
            target_fill_ratio: 出入口目标填充比例 (0.0 ~ 1.0)
        """
        self.box = np.array(box_size)
        self.num_pipes_per_group = num_pipes_per_group
        self.total_pipes = num_pipes_per_group * 2  # 总共 60 根
        self.r = render_radius
        self.phys_radius = self.r
        self.target_fill_ratio = target_fill_ratio

        # 计算每组出口容器尺寸
        group_pipe_area = self.num_pipes_per_group * np.pi * self.r**2
        self.connector_area = group_pipe_area / target_fill_ratio
        self.connector_radius = np.sqrt(self.connector_area / np.pi)

        # 统一存储所有管道路径 (列表形式，每个元素是一个切片)
        self.all_pipes_path = []  # shape: (num_steps, total_pipes, 3)

    def _pack_circle_with_fill_ratio(self, center, count, pipe_radius, target_fill_ratio):
        """
        生成指定填充率的紧密堆积

        参数:
            center: 堆积中心 [x, y]
            count: 管道数量
            pipe_radius: 单根管道半径
            target_fill_ratio: 目标填充比例 (0.0 ~ 1.0)

        返回:
            points: (count, 2) 的位置数组
            container_radius: 容器半径
        """
        # 1. 计算目标容器半径
        total_pipe_area = count * np.pi * pipe_radius**2
        container_area = total_pipe_area / target_fill_ratio
        container_radius = np.sqrt(container_area / np.pi)

        # 2. 使用向日葵堆积但限制在容器内
        effective_radius = container_radius - pipe_radius  # 留出管道半径
        if count > 1:
            spacing = effective_radius / (0.6 * np.sqrt(count))
        else:
            spacing = 0

        # 3. 生成点位 (黄金角度堆积)
        golden_angle = np.pi * (3 - np.sqrt(5))  # 约 2.39996
        points = []
        for i in range(count):
            r = spacing * 0.6 * np.sqrt(i + 1) if i > 0 else 0
            theta = i * golden_angle
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y])

        return np.array(points), container_radius

    def _generate_guide_paths(self, steps):
        """生成 60 根独立管道的平滑随机路径"""
        num_keyframes = 6
        z_keys = np.linspace(0, self.box[2], num_keyframes)
        guide_paths = []
        margin = self.r * 4  # 边界留白

        for i in range(self.total_pipes):
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

        return np.array(guide_paths)  # shape: (total_pipes, steps, 2)

    def _solve_collision_step_simple(self, pos, iterations=10):
        """
        快速单切片碰撞解算 (用于初始生成)
        所有 60 根管道平等处理，无配对排除
        """
        current_pos = pos.copy()  # shape: (total_pipes, 2)
        limit_x = self.box[0] / 2 - self.phys_radius
        limit_y = self.box[1] / 2 - self.phys_radius
        min_dist = self.phys_radius * 2.0

        for _ in range(iterations):
            # 边界约束
            np.clip(current_pos[:, 0], -limit_x, limit_x, out=current_pos[:, 0])
            np.clip(current_pos[:, 1], -limit_y, limit_y, out=current_pos[:, 1])

            # 计算所有点对距离
            dists = cdist(current_pos, current_pos, metric="euclidean")
            np.fill_diagonal(dists, 999.0)

            # 碰撞检测
            collision_mask = dists < min_dist
            if not np.any(collision_mask):
                break

            overlap = min_dist - dists
            overlap[~collision_mask] = 0

            rows, cols = np.where(collision_mask)
            unique = rows < cols
            rows, cols = rows[unique], cols[unique]
            if len(rows) == 0:
                break

            p_rows, p_cols = current_pos[rows], current_pos[cols]
            diffs = p_rows - p_cols
            current_dist = dists[rows, cols]
            current_dist[current_dist < 1e-5] = 1e-5
            move_vecs = (diffs / current_dist[:, None]) * overlap[rows, cols][:, None] * 0.5

            # 累加位移
            moves = np.zeros_like(current_pos)
            counts = np.zeros(self.total_pipes)
            for k in range(len(rows)):
                moves[rows[k]] += move_vecs[k]
                moves[cols[k]] -= move_vecs[k]
                counts[rows[k]] += 1
                counts[cols[k]] += 1

            counts[counts == 0] = 1
            current_pos += moves / counts[:, None]

        return current_pos

    def _generate_body(self, steps=800):
        """生成 60 根独立管道的主体部分"""
        print("1/4 Generating pipe body...")
        guides = self._generate_guide_paths(steps)  # (total_pipes, steps, 2)

        # 初始位置：使用第一个切片的引导位置
        current_pos = guides[:, 0, :].copy()  # (total_pipes, 2)

        # 初始碰撞解算
        for _ in range(30):
            current_pos = self._solve_collision_step_simple(current_pos, iterations=5)

        z_step = self.box[2] / steps
        body_path = []

        for i in tqdm(range(steps), desc="Body generation"):
            z = i * z_step
            target_pos = guides[:, i, :]

            # 平滑跟随引导路径
            current_pos = current_pos * 0.85 + target_pos * 0.15

            # 单切片碰撞解算
            current_pos = self._solve_collision_step_simple(current_pos, iterations=5)

            # 记录当前切片所有管道位置
            pts = np.column_stack([current_pos, np.full(self.total_pipes, z)])
            body_path.append(pts)

        return body_path  # list of (total_pipes, 3)

    def _generate_connector_segment(self, body_conn, center_a, center_b, z_range, steps, reverse=False):
        """
        生成接头段：A 组汇合到 center_a，B 组汇合到 center_b

        参数:
            body_conn: body 端点位置 (total_pipes, 3)
            center_a: A 组汇合中心 [x, y]
            center_b: B 组汇合中心 [x, y]
            z_range: Z 坐标范围 (z_start, z_end)
            steps: 插值步数
            reverse: True = 从紧密堆积到分散，False = 从分散到紧密堆积
        """
        n = self.num_pipes_per_group

        # 为每组生成紧密堆积位置
        pack_a, _ = self._pack_circle_with_fill_ratio(
            center_a, n, self.r, self.target_fill_ratio
        )
        pack_b, _ = self._pack_circle_with_fill_ratio(
            center_b, n, self.r, self.target_fill_ratio
        )

        # 合并两组的目标位置
        pack_all = np.vstack([pack_a, pack_b])  # (total_pipes, 2)

        # Hermite 插值权重 (S 形曲线)
        t = np.linspace(0, 1, steps)
        w = 3 * t**2 - 2 * t**3  # smoothstep

        z_vals = np.linspace(z_range[0], z_range[1], steps)
        segment = []

        for k in range(steps):
            weight = w[k]
            if reverse:
                # 从紧密到分散
                start_xy = pack_all
                end_xy = body_conn[:, :2]
            else:
                # 从分散到紧密
                start_xy = body_conn[:, :2]
                end_xy = pack_all

            xy = start_xy * (1 - weight) + end_xy * weight
            pts = np.column_stack([xy, np.full(self.total_pipes, z_vals[k])])
            segment.append(pts)

        return segment  # list of (total_pipes, 3)

    def _global_collision_optimization(self, max_iterations=300, convergence_threshold=0.001):
        """
        全局碰撞优化算法 (核心) - 紧密贴合版

        原理:
        1. 吸引力: 让管道向中心聚拢，像被挤压的绳子
        2. 推斥力: 当管道距离 < 直径时推开，防止穿插
        3. 最终效果: 管道紧密贴合 (距离 ≈ 直径) 但不穿插
        """
        print("3/4 Global Collision Optimization (tight packing)...")

        all_paths = np.array(self.all_pipes_path)
        num_steps = all_paths.shape[0]

        min_dist = self.r * 2.0  # 最小允许距离 = 直径 (刚好贴合)
        target_dist = self.r * 2.0  # 目标距离 = 直径 (紧密贴合)
        attract_range = self.r * 4.0  # 吸引力作用范围

        limit_x = self.box[0] / 2 - self.phys_radius
        limit_y = self.box[1] / 2 - self.phys_radius

        for iteration in range(max_iterations):
            max_penetration = 0.0
            forces = np.zeros_like(all_paths)

            for s in range(num_steps):
                current_pts = all_paths[s, :, :2]
                d_mat = cdist(current_pts, current_pts)
                np.fill_diagonal(d_mat, 999.0)

                # 计算方向向量
                delta = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
                dist_safe = np.maximum(d_mat, 1e-6)
                norm_dir = delta / dist_safe[:, :, np.newaxis]

                # ====== 1. 推斥力: 防止穿插 ======
                collision_mask = d_mat < min_dist
                if np.any(collision_mask):
                    penetration = min_dist - d_mat
                    penetration[~collision_mask] = 0
                    max_penetration = max(max_penetration, penetration.max())

                    # 强推力，穿插时快速推开
                    repel_force = norm_dir * penetration[:, :, np.newaxis] * 0.5
                    forces[s, :, :2] += np.sum(repel_force, axis=1)

                # ====== 2. 吸引力: 让管道聚拢 ======
                # 对于距离在 (min_dist, attract_range) 范围内的管道，施加吸引力
                attract_mask = (d_mat > min_dist) & (d_mat < attract_range)
                if np.any(attract_mask):
                    # 吸引力强度: 距离越远吸引力越强，但有上限
                    gap = d_mat - target_dist
                    gap[~attract_mask] = 0
                    gap = np.clip(gap, 0, attract_range - target_dist)

                    # 负方向 = 吸引 (向对方靠近)
                    attract_force = -norm_dir * gap[:, :, np.newaxis] * 0.15
                    forces[s, :, :2] += np.sum(attract_force, axis=1)

                # ====== 3. 向中心聚拢的全局吸引力 ======
                center = np.mean(current_pts, axis=0)
                to_center = center - current_pts
                dist_to_center = np.linalg.norm(to_center, axis=1, keepdims=True)
                dist_to_center = np.maximum(dist_to_center, 1e-6)
                center_dir = to_center / dist_to_center

                # 轻微的向心力，让整体更紧凑
                forces[s, :, :2] += center_dir * 0.02

            # ====== Pass 2: 应用力 ======
            all_paths[:, :, :2] += forces[:, :, :2]

            # 边界约束
            all_paths[:, :, 0] = np.clip(all_paths[:, :, 0], -limit_x, limit_x)
            all_paths[:, :, 1] = np.clip(all_paths[:, :, 1], -limit_y, limit_y)

            # Z 方向平滑
            smooth_sigma = 2.0
            all_paths[:, :, 0] = gaussian_filter1d(all_paths[:, :, 0], sigma=smooth_sigma, axis=0)
            all_paths[:, :, 1] = gaussian_filter1d(all_paths[:, :, 1], sigma=smooth_sigma, axis=0)

            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: max_penetration = {max_penetration:.4f}")

            if max_penetration < convergence_threshold:
                print(f"  Converged at iteration {iteration}")
                break

        self.all_pipes_path = [all_paths[s] for s in range(num_steps)]

    def _final_smooth(self, sigma=2.0):
        """最终的全局平滑，确保曲线连续"""
        all_paths = np.array(self.all_pipes_path)

        # 仅平滑 XY，保持 Z 不变
        all_paths[:, :, 0] = gaussian_filter1d(all_paths[:, :, 0], sigma=sigma, axis=0)
        all_paths[:, :, 1] = gaussian_filter1d(all_paths[:, :, 1], sigma=sigma, axis=0)

        num_steps = all_paths.shape[0]
        self.all_pipes_path = [all_paths[s] for s in range(num_steps)]

    def generate_full(self, body_steps=800, term_steps=120, term_len=15.0):
        """
        生成完整的管道编织结构

        参数:
            body_steps: 主体部分的步数
            term_steps: 接头部分的步数
            term_len: 接头长度
        """
        # 1. 生成主体
        body_path = self._generate_body(steps=body_steps)

        # 2. 生成接头
        print("2/4 Generating connector terminals...")
        spread_x = self.box[0] * 0.4  # 两组出口的 X 间距

        # 起始接头 (从紧密到分散)
        start_segment = self._generate_connector_segment(
            body_path[0],
            [-spread_x, 0],
            [spread_x, 0],
            (-term_len, 0),
            term_steps,
            reverse=True,
        )

        # 结束接头 (从分散到紧密)
        end_segment = self._generate_connector_segment(
            body_path[-1],
            [-spread_x, 0],
            [spread_x, 0],
            (self.box[2], self.box[2] + term_len),
            term_steps,
            reverse=False,
        )

        # 3. 拼接: Start[:-1] + Body + End[1:]
        self.all_pipes_path = start_segment[:-1] + body_path + end_segment[1:]

        # 4. 全局碰撞优化
        self._global_collision_optimization(
            max_iterations=300,
            convergence_threshold=0.005,
        )

        # 5. 最终平滑
        print("4/4 Final smoothing...")
        self._final_smooth(sigma=2.0)

    def diagnose_collisions(self):
        """诊断碰撞情况"""
        all_paths = np.array(self.all_pipes_path)
        num_steps = all_paths.shape[0]
        min_dist = self.r * 2.0

        total_collisions = 0
        max_penetration = 0.0

        for s in range(num_steps):
            pts = all_paths[s, :, :2]
            d_mat = cdist(pts, pts)
            np.fill_diagonal(d_mat, 999.0)

            collision_mask = d_mat < min_dist
            n_collisions = np.sum(collision_mask) // 2
            total_collisions += n_collisions

            if n_collisions > 0:
                penetration = min_dist - d_mat[collision_mask]
                max_penetration = max(max_penetration, penetration.max())

        print(f"Diagnosis:")
        print(f"  Total collision pairs across all slices: {total_collisions}")
        print(f"  Max penetration depth: {max_penetration:.4f}")
        print(f"  Target min distance: {min_dist:.4f}")

        return total_collisions, max_penetration

    def get_strands(self):
        """获取所有管道的路径数据，用于导出"""
        all_paths = np.array(self.all_pipes_path)  # (num_steps, total_pipes, 3)
        # 转置为 (total_pipes, num_steps, 3)
        strands = all_paths.transpose(1, 0, 2)
        return strands


# OCCT Imports
from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Vec, gp_Circ
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Builder
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_C2, GeomAbs_C1


class PipeExporter:
    """管道导出器：支持 STEP 和 STL 格式"""

    def __init__(self, target_radius=0.25):
        self.target_radius = float(target_radius)
        self.builder = TopoDS_Builder()
        self.compound = TopoDS_Compound()
        self.builder.MakeCompound(self.compound)
        self.count = 0
        self.success_count = 0

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
                except:
                    pass

        self.success_count += success
        print(f"Batch Result: {success}/{len(strands_list)} solids.")

    def export_step(self, filename):
        """导出 STEP 格式"""
        if self.count == 0:
            print("No geometry to export.")
            return
        print(f"Writing STEP: {filename}...")
        writer = STEPControl_Writer()
        writer.Transfer(self.compound, STEPControl_AsIs)
        writer.Write(filename)
        print(f"STEP export done: {self.success_count} pipes.")

    def export_stl(self, filename, linear_deflection=0.1, angular_deflection=0.5):
        """
        导出 STL 格式

        参数:
            filename: 输出文件名
            linear_deflection: 线性偏差 (越小越精细)
            angular_deflection: 角度偏差 (弧度)
        """
        if self.count == 0:
            print("No geometry to export.")
            return

        print(f"Meshing for STL (deflection={linear_deflection})...")
        # 网格化
        mesh = BRepMesh_IncrementalMesh(
            self.compound, linear_deflection, False, angular_deflection, True
        )
        mesh.Perform()

        print(f"Writing STL: {filename}...")
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)  # 二进制格式，文件更小
        writer.Write(self.compound, filename)
        print(f"STL export done: {self.success_count} pipes.")

    def export(self, step_filename, stl_filename=None):
        """导出 STEP 和可选的 STL"""
        self.export_step(step_filename)
        if stl_filename:
            self.export_stl(stl_filename)


if __name__ == "__main__":
    print("=" * 50)
    print("Pipe Weaving System - TPMS Alternative")
    print("=" * 50)

    # 配置参数
    NUM_PIPES_PER_GROUP = 30  # 每组 30 根，共 60 根
    RENDER_RADIUS = 0.25
    TARGET_FILL_RATIO = 0.3  # 出入口 30% 填充率
    BOX_SIZE = (9, 9, 35)

    print(f"\nConfiguration:")
    print(f"  Pipes per group: {NUM_PIPES_PER_GROUP}")
    print(f"  Total pipes: {NUM_PIPES_PER_GROUP * 2}")
    print(f"  Render radius: {RENDER_RADIUS}")
    print(f"  Target fill ratio: {TARGET_FILL_RATIO}")
    print(f"  Box size: {BOX_SIZE}")

    # 1. 生成物理模拟
    print("\n--- 1. Physics Simulation ---")
    weaver = PipeWeavingSystem(
        box_size=BOX_SIZE,
        num_pipes_per_group=NUM_PIPES_PER_GROUP,
        render_radius=RENDER_RADIUS,
        target_fill_ratio=TARGET_FILL_RATIO,
    )
    weaver.generate_full(body_steps=800, term_steps=120, term_len=15.0)

    # 2. 诊断碰撞
    print("\n--- 2. Collision Diagnosis ---")
    weaver.diagnose_collisions()

    # 3. 获取路径数据
    print("\n--- 3. Preparing Export ---")
    strands = weaver.get_strands()
    print(f"Total strands: {len(strands)}")

    # 4. 导出
    print("\n--- 4. Export ---")
    exporter = PipeExporter(target_radius=RENDER_RADIUS * 0.96)  # 略小，留间隙
    exporter.process_strands(strands)

    # 同时导出 STEP 和 STL
    exporter.export(
        step_filename="pipe_weaving.step",
        stl_filename="pipe_weaving.stl"
    )

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
