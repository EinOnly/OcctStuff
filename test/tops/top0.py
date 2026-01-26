import numpy as np
from tqdm import tqdm

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d


class FinalCableHarness:
    def __init__(self, box_size=(9, 9, 35), num_pairs=120, 
                 pair_gap=0.55, render_radius=0.25):
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
        """ 蜂窝堆积 """
        points = []
        for i in range(count):
            r = spacing * 0.6 * np.sqrt(i + 1)
            theta = i * 2.39996 
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y])
        return np.array(points)

    def _generate_guide_paths(self, steps):
        """ 生成平滑随机路径 """
        num_keyframes = 5
        z_keys = np.linspace(0, self.box[2], num_keyframes)
        guide_paths = []
        margin = self.gap + self.r * 3
        for i in range(self.num_pairs):
            x_keys = np.random.uniform(-self.box[0]/2 + margin, self.box[0]/2 - margin, num_keyframes)
            y_keys = np.random.uniform(-self.box[1]/2 + margin, self.box[1]/2 - margin, num_keyframes)
            cs_x = CubicSpline(z_keys, x_keys)
            cs_y = CubicSpline(z_keys, y_keys)
            z_samples = np.linspace(0, self.box[2], steps)
            guide_paths.append(np.column_stack([cs_x(z_samples), cs_y(z_samples)]))
        return np.array(guide_paths)

    def _solve_collision_step(self, pos, pair_angles, iterations=10):
        """ 快速初始碰撞解算 """
        current_pos = pos.copy()
        vec_x = np.cos(pair_angles) * (self.gap / 2)
        vec_y = np.sin(pair_angles) * (self.gap / 2)
        limit_x = self.box[0]/2 - self.phys_radius
        limit_y = self.box[1]/2 - self.phys_radius
        
        for _ in range(iterations):
            np.clip(current_pos[:, 0], -limit_x, limit_x, out=current_pos[:, 0])
            np.clip(current_pos[:, 1], -limit_y, limit_y, out=current_pos[:, 1])
            Ax = current_pos[:, 0] + vec_x
            Ay = current_pos[:, 1] + vec_y
            Bx = current_pos[:, 0] - vec_x
            By = current_pos[:, 1] - vec_y
            all_pts = np.vstack([np.column_stack([Ax, Ay]), np.column_stack([Bx, By])])
            dists = cdist(all_pts, all_pts, metric='euclidean')
            np.fill_diagonal(dists, 999.0)
            for i in range(self.num_pairs):
                dists[i, i + self.num_pairs] = 999.0
                dists[i + self.num_pairs, i] = 999.0
            
            collision_mask = dists < (self.phys_radius * 2.1)
            if not np.any(collision_mask): break
            overlap = (self.phys_radius * 2.1) - dists
            overlap[~collision_mask] = 0
            rows, cols = np.where(collision_mask)
            unique = rows < cols
            rows, cols = rows[unique], cols[unique]
            if len(rows) == 0: break
            p_rows, p_cols = all_pts[rows], all_pts[cols]
            diffs = p_rows - p_cols
            current_dist = dists[rows, cols]
            current_dist[current_dist < 1e-5] = 1e-5
            move_vecs = (diffs / current_dist[:, None]) * overlap[rows, cols][:, None] * 0.5
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
        current_angles = np.random.uniform(0, np.pi*2, self.num_pairs)
        for _ in range(50):
            current_pos = self._solve_collision_step(current_pos, current_angles, iterations=1)
        z_step = self.box[2] / steps
        for i in tqdm(range(steps)):
            z = i * z_step
            target_pos = guides[:, i, :]
            current_pos = current_pos * 0.9 + target_pos * 0.1
            current_angles += 0.06
            current_pos = self._solve_collision_step(current_pos, current_angles, iterations=8)
            cos_a, sin_a = np.cos(current_angles), np.sin(current_angles)
            offset_x, offset_y = cos_a * (self.gap / 2), sin_a * (self.gap / 2)
            pts_a = np.column_stack([current_pos[:,0] + offset_x, current_pos[:,1] + offset_y, np.full(self.num_pairs, z)])
            pts_b = np.column_stack([current_pos[:,0] - offset_x, current_pos[:,1] - offset_y, np.full(self.num_pairs, z)])
            self._body_a.append(pts_a)
            self._body_b.append(pts_b)

    def _generate_connector_segment(self, body_conn_a, body_conn_b, center_a, center_b, z_range, steps, reverse=False):
        pack_a = self._pack_circle(center_a, self.num_pairs, self.r * 2.2)
        pack_b = self._pack_circle(center_b, self.num_pairs, self.r * 2.2)
        t = np.linspace(0, 1, steps)
        w = 3*t**2 - 2*t**3
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

    def _advanced_relax_and_smooth(self, iterations=15, smooth_sigma=3.0, 
                                   body_start_idx=0, body_end_idx=0):
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
                    new_offset_a = current_offset_a * (1 - weight) + target_offset * weight
                    new_offset_b = current_offset_b * (1 - weight) - target_offset * weight
                    
                    xy_a = centers + new_offset_a
                    xy_b = centers + new_offset_b

                # --- 2. Collision Repulsion (全域应用) ---
                # 即使在接头处，A线之间也不应该穿插，所以碰撞检测保留
                all_pts = np.vstack([xy_a, xy_b])
                d_mat = cdist(all_pts, all_pts, metric='euclidean')
                
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
                    xy_a = all_pts[:self.num_pairs]
                    xy_b = all_pts[self.num_pairs:]

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
                        
                        new_offset_a = current_offset_a * (1 - weight) + target_offset * weight
                        new_offset_b = current_offset_b * (1 - weight) - target_offset * weight
                        
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
            self._body_a[0], self._body_b[0], 
            [-spread_x, 0], [spread_x, 0], 
            (-term_len, 0), term_steps, reverse=True
        )
        end_a, end_b = self._generate_connector_segment(
            self._body_a[-1], self._body_b[-1], 
            [-spread_x, 0], [spread_x, 0], 
            (self.box[2], self.box[2]+term_len), term_steps, reverse=False
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
            iterations=20, 
            smooth_sigma=3.0,
            body_start_idx=body_start_idx,
            body_end_idx=body_end_idx
        )


from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Builder
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.GeomAbs import GeomAbs_C2


class CurveOnlyExporter:
    def __init__(self):
        self.builder = TopoDS_Builder()
        self.compound = TopoDS_Compound()
        self.builder.MakeCompound(self.compound)
        self.count = 0

    def _create_bspline_edge(self, points):
        """ 将点集转换为 B-Spline 曲线 """
        # 1. 简单的距离去重 (防止重合点)
        clean_points = [points[0]]
        for pt in points[1:]:
            if np.linalg.norm(pt - clean_points[-1]) > 1e-4:
                clean_points.append(pt)
        
        num_points = len(clean_points)
        if num_points < 2: return None

        # 2. 转换为 OCCT 点
        occt_points = TColgp_Array1OfPnt(1, num_points)
        for i, pt in enumerate(clean_points):
            occt_points.SetValue(i + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))

        try:
            # 3. 逼近生成曲线 (C2 连续，非常光滑)
            approx = GeomAPI_PointsToBSpline(
                occt_points, 
                3, 8,           # 阶数范围
                GeomAbs_C2,     # 二阶导数连续(最滑)
                0.1             # 允许 0.1mm 的拟合误差
            )
            if not approx.IsDone(): return None
            
            return BRepBuilderAPI_MakeEdge(approx.Curve()).Edge()

        except Exception:
            return None

    def add_paths(self, paths_array):
        """ 
        注意：这里接收的是已经是 [N_Paths, N_Steps, 3] 格式的数组 
        """
        print(f"Converting {len(paths_array)} strands to curves...")
        
        # 遍历每一根单独的线
        for pts in tqdm(paths_array):
            edge = self._create_bspline_edge(pts)
            if edge:
                self.builder.Add(self.compound, edge)
                self.count += 1

    def export(self, filename="cables_curves.step"):
        if self.count == 0:
            print("No curves to export.")
            return

        print(f"Writing STEP file ({self.count} curves)...")
        writer = STEPControl_Writer()
        writer.Transfer(self.compound, STEPControl_AsIs)
        status = writer.Write(filename)
        if status == 1:
            print(f"Success! Saved to {filename}")
        else:
            print("Export Failed.")

if __name__ == "__main__":
    print("--- 1. Generating Physics ---")
    # 设置线缆数量
    N_PAIRS = 120 # 120对 = 240根线
    
    cable = FinalCableHarness(
        box_size=(9, 9, 35), 
        num_pairs=N_PAIRS,
        pair_gap=0.55, 
        render_radius=0.25
    )
    cable.generate_full_cable(body_steps=800, term_steps=120)

    print("\n--- 2. Transposing Data (Slice -> Strand) ---")
    
    # [关键修正步骤]
    # 原始数据是 list of steps: [Steps, Num_Pairs, 3]
    # 我们需要转置成: [Num_Pairs, Steps, 3]
    
    # 1. 转换为 Numpy 数组
    raw_a = np.array(cable.line_a_path) # Shape: (1040, 120, 3)
    raw_b = np.array(cable.line_b_path) # Shape: (1040, 120, 3)
    
    # 2. 执行转置 (交换 0轴 和 1轴)
    # 现在的 shape 变成了 (120, 1040, 3)，这才是 120 根完整的线
    strands_a = raw_a.transpose(1, 0, 2)
    strands_b = raw_b.transpose(1, 0, 2)
    
    print(f"Data Transposed. Ready to export {len(strands_a) + len(strands_b)} curves.")

    print("\n--- 3. Exporting ---")
    exporter = CurveOnlyExporter()
    
    exporter.add_paths(strands_a) # 添加 A 组线
    exporter.add_paths(strands_b) # 添加 B 组线
    
    exporter.export("final_centerlines.step")