import numpy as np
from heapq import heappush, heappop

class SlimeAgent:
    def __init__(self, solver, max_fraction=0.6):
        self.ny, self.nx = solver.ny, solver.nx
        self.max_fraction = max_fraction
        self.slime = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.memory = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.path_mask = np.zeros_like(self.slime, dtype=bool)

        # --- 调整后的权重 ---
        self.target_fill = 0.5    # 目标填充率
        self.learning_rate = 0.2  # 提高学习率，反应更快
        self.max_step = 0.1       # 允许更大的单步突变
        self.weights = {
            "body": 0.2,          # 维持现有身体
            "heat": 1.2,          # 强烈追逐热量 (仅在流体区有效)
            "calm": 0.1,          # 稍微偏好低速区
            "cohesion": 0.1,      # 聚集性
            "path": 0.5,          # 初始路径维持
            "frontier": 2.0,      # [关键] 极大地鼓励向外扩张
            "noise": 0.6,         # [关键] 增加随机性以打破死锁
            "sparsity": 0.1,      # 惩罚过密的块，鼓励分支
            "area_penalty": 0.1,  
            "area_gain": 3.0,     # 未填满时强力奖励生长
            "memory": 0.3,        
        }

        # 自动探测入口/出口位置
        inlet_rows = np.where(solver.inlet_mask[:, 0])[0]
        outlet_rows = np.where(solver.outlet_mask[:, -1])[0]
        inlet_row = int(np.mean(inlet_rows)) if inlet_rows.size else self.ny // 2
        outlet_row = int(np.mean(outlet_rows)) if outlet_rows.size else self.ny // 2
        
        self.start_seed = (inlet_row, 1)
        self.goal_seeds = [
            (max(5, outlet_row - self.ny // 3), self.nx - 2),
            (outlet_row, self.nx - 2),
            (min(self.ny - 6, outlet_row + self.ny // 3), self.nx - 2),
        ]

    def initialize_from_fields(self, solver):
        """初始化：基于初始流场或几何中心生成初始路径"""
        # 这里我们需要一个全局可生长掩码，而不仅仅是当前的流体
        # 如果 solver 初始化时大部分是墙，我们需要假定整个空间都是潜在可生长的
        active_fluid = solver.get_fluid_mask()
        
        # 判定：如果当前流体太少（可能是初始化状态），则允许全图规划
        if np.count_nonzero(active_fluid) < self.ny * self.nx * 0.05:
            growable = ~solver.base_boundary
        else:
            growable = active_fluid | ~solver.base_boundary # 逻辑上应该是全图

        growable = ~solver.base_boundary # 强制全图可生长

        v_norm = self._normalize(np.sqrt(solver.u**2 + solver.v**2), active_fluid)
        t_norm = self._normalize(solver.temp, active_fluid)

        # 初始路径规划：倾向于走中间或顺着流场
        cost = 0.15 * v_norm + 0.85 * (1 - t_norm) + 0.02
        self.path_mask = self._path_union(cost, growable)

        noise = np.random.rand(self.ny, self.nx).astype(np.float32) * growable
        
        # 初始欲望分布
        desirability = (
            0.5 * (1 - v_norm) 
            + 0.5 * self._dilate(self.path_mask, steps=3).astype(np.float32)
            + 0.2 * noise
        ) * growable

        # 选取初始种子
        target_cells = int(self.target_fill * np.count_nonzero(growable) * 0.2) # 初始只长 20%
        flat = desirability[growable]
        if target_cells > 0 and flat.size > target_cells:
            thresh = np.partition(flat, -target_cells)[-target_cells]
            seed_mask = (desirability >= thresh) & growable
        else:
            seed_mask = growable.copy()

        # 确保连通性
        connected = self._connected_component(seed_mask, self.start_seed)
        self.slime = connected.astype(np.float32)
        # 强制根部存活
        sy, sx = self.start_seed
        self.slime[sy-2:sy+3, 0:5] = 1.0
        self.memory = self.slime.copy()

    def update(self, solver, slow_factor=1.0):
        # --- 1. 获取感知域与生长域 ---
        # active_fluid: 有真实物理量(速度/温度)的区域
        active_fluid = solver.get_fluid_mask()
        # growable_domain: Slime 潜在可以占据的所有区域 (除了永久边界)
        # [修复点] 之前这里使用的是 active_fluid，导致 Slime 无法看到外部空间
        growable_domain = ~solver.base_boundary

        # --- 2. 感知物理量 (仅在 active_fluid 内有效) ---
        vmag = np.sqrt(solver.u**2 + solver.v**2)
        v_norm = self._normalize(vmag, active_fluid)
        t_norm = self._normalize(solver.temp, active_fluid)

        # 路径规划成本：在流体区基于物理量，在未知区给予基础成本
        cost = np.ones_like(self.slime) * 0.5 
        if np.any(active_fluid):
            # 速度越快、温度越高，成本越低(或高，取决于策略，这里假设喜欢热/低速)
            cost[active_fluid] = 0.2 * v_norm[active_fluid] + 0.8 * (1 - t_norm[active_fluid])

        # --- 3. 计算各个驱动因子 ---
        
        # 路径因子：始终指引大方向
        path_mask = self._path_union(cost, growable_domain)
        if path_mask.any():
            self.path_mask = path_mask
        corridor = self._dilate(self.path_mask, steps=2)

        # 内部聚集因子
        cohesion = self._neighbor_mean(self.slime)
        
        # [关键] 边界探测因子 (Frontier)
        # 探测 Slime 边缘向外一圈的区域，这是扩张的主要动力
        # 即使这些区域目前没有流体，它们也在 growable_domain 内，应该被高亮
        frontier = np.clip(self._dilate(self.slime > 0.05, steps=2).astype(np.float32) - self.slime, 0, 1)

        # 随机因子
        noise = np.random.rand(self.ny, self.nx).astype(np.float32)
        
        # 密度统计
        thickness = self._neighbor_mean(self.slime)
        fill_ratio = float(self.slime.sum()) / (np.count_nonzero(growable_domain) + 1e-6)
        area_under = max(0.0, self.target_fill - fill_ratio)

        # --- 4. 综合欲望值计算 ---
        desirability = (
            self.weights["body"] * self.slime
            + self.weights["heat"] * t_norm           # 物理驱动 (仅内部)
            + self.weights["calm"] * (1 - v_norm)     # 物理驱动 (仅内部)
            + self.weights["cohesion"] * cohesion
            + self.weights["path"] * corridor.astype(np.float32)
            + self.weights["frontier"] * frontier     # [关键] 外部驱动
            + self.weights["noise"] * noise           # [关键] 外部随机驱动
            - self.weights["sparsity"] * thickness
            + self.weights["area_gain"] * area_under
            + self.weights["memory"] * self.memory
        )

        # [关键修复] 允许欲望在整个可生长区域生效，而不仅是当前流体区
        desirability *= growable_domain

        # --- 5. 状态更新 ---
        lr = self.learning_rate * slow_factor
        target = self.slime + lr * (desirability - self.slime)
        
        # 限制单步变化幅度，防止震荡
        delta = np.clip(
            target - self.slime,
            -self.max_step * slow_factor,
            self.max_step * slow_factor,
        )
        updated = self.slime + delta

        # --- 6. 后处理与约束 ---
        
        # 截断 (Cap Fraction)
        updated = self._cap_fraction(updated, growable_domain)
        updated[~growable_domain] = 0

        # 保护主要路径，防止断裂
        updated[corridor] = np.maximum(updated[corridor], 0.6)
        
        # 记忆惯性
        updated = np.maximum(updated, 0.2 * self.slime)

        # 强制根部存活 (防止源头被切断导致全死)
        sy, sx = self.start_seed
        updated[max(0,sy-2):min(self.ny,sy+3), 0:4] = 1.0

        # 连通性检查 (只保留连接到源头的最大连通域)
        # 适当降低阈值，允许微弱连接存在，防止过早断裂
        connected = self._connected_component(updated >= 0.1, self.start_seed)
        updated *= connected.astype(np.float32)

        # 平滑处理
        smoothed = 0.2 * updated + 0.8 * self._neighbor_mean(updated)
        self.slime = np.clip(smoothed, 0, 1)
        
        # 更新长期记忆
        self.memory = np.clip(0.98 * self.memory + 0.02 * self.slime, 0, 1)

    def get_obstacle_mask(self, threshold=0.15):
        """返回给 LBM 的障碍物掩码 (True = Fluid, False = Solid/Obstacle)"""
        # 注意：simulation.py 里使用的是 slime_mask = ~boundary
        # 所以这里返回 True 的地方会变成流体
        mask = self.slime >= threshold
        return mask.astype(bool)

    # --- Helpers ---
    def _normalize(self, field, mask):
        vals = field[mask]
        if vals.size == 0:
            return np.zeros_like(field, dtype=np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-12:
            return np.zeros_like(field, dtype=np.float32)
        norm = (field - vmin) / (vmax - vmin)
        norm[~mask] = 0
        return norm.astype(np.float32)

    def _dilate(self, mask, steps=1):
        out = mask.copy()
        for _ in range(steps):
            grown = out.copy()
            # 8邻域膨胀
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    grown |= np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            out = grown
        return out

    def _cap_fraction(self, field, active_domain):
        max_cells = int(self.max_fraction * np.count_nonzero(active_domain))
        if max_cells <= 0: return np.zeros_like(field)
        
        vals = field[active_domain].ravel()
        if vals.size <= max_cells:
            return field * active_domain
            
        # 找到阈值保留前 N 个
        threshold = np.partition(vals, -max_cells)[-max_cells]
        capped = np.where(field >= threshold, field, 0)
        capped[~active_domain] = 0
        return capped.astype(np.float32)

    def _neighbor_mean(self, field):
        # 简单的卷积平滑
        kernel = np.array([[0.5, 1, 0.5], [1, 4, 1], [0.5, 1, 0.5]])
        kernel /= kernel.sum()
        # 这里用 scipy.signal.convolve2d 会更准，但为了减少依赖手写一个近似
        accum = 4.0 * field
        accum += np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
        accum += np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
        accum += 0.5 * (np.roll(np.roll(field, 1, axis=0), 1, axis=1) + 
                        np.roll(np.roll(field, -1, axis=0), -1, axis=1) +
                        np.roll(np.roll(field, 1, axis=0), -1, axis=1) + 
                        np.roll(np.roll(field, -1, axis=0), 1, axis=1))
        return accum / 10.0

    def _connected_component(self, mask, seed):
        sy, sx = seed
        # 保护：如果种子点死了，强行复活，否则程序逻辑崩溃
        if not mask[sy, sx]: mask[sy, sx] = True
            
        visited = np.zeros_like(mask, dtype=bool)
        stack = [(sy, sx)]
        visited[sy, sx] = True
        
        # 简单的 DFS 寻找连通域
        while stack:
            y, x = stack.pop()
            neighbors = [(y+1,x), (y-1,x), (y,x+1), (y,x-1)]
            for ny, nx in neighbors:
                if 0 <= ny < self.ny and 0 <= nx < self.nx:
                    if not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        return visited

    def _shortest_path(self, cost, traversable, start, goal):
        # Dijkstra 算法
        sy, sx = start
        gy, gx = goal
        path_mask = np.zeros((self.ny, self.nx), dtype=bool)
        
        # 即使终点不可达，也尝试规划
        dist = np.full((self.ny, self.nx), np.inf, dtype=np.float32)
        prev = np.full((self.ny, self.nx, 2), -1, dtype=np.int32)
        heap = []
        
        if traversable[sy, sx]:
            dist[sy, sx] = 0.0
            heappush(heap, (0.0, sy, sx))
            
        while heap:
            d, y, x = heappop(heap)
            if d > dist[y, x]: continue
            if (y, x) == (gy, gx): break
            
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.ny and 0 <= nx < self.nx and traversable[ny, nx]:
                    nd = d + cost[ny, nx] # 简化距离计算
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        prev[ny, nx] = (y, x)
                        heappush(heap, (nd, ny, nx))
                        
        # 回溯
        cy, cx = gy, gx
        if np.isinf(dist[cy, cx]): return path_mask, dist # 无法到达
        
        while cy != -1:
            path_mask[cy, cx] = True
            cy, cx = prev[cy, cx]
            
        return path_mask, dist

    def _path_union(self, cost, traversable):
        union = np.zeros((self.ny, self.nx), dtype=bool)
        for g in self.goal_seeds:
            p, _ = self._shortest_path(cost, traversable, self.start_seed, g)
            union |= p
        return union

# --- 独立实时运行模块 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import time

    class RealTimeDummySolver:
        def __init__(self, nx=200, ny=100):
            self.nx, self.ny = nx, ny
            self.inlet_mask = np.zeros((ny, nx), dtype=bool)
            self.outlet_mask = np.zeros((ny, nx), dtype=bool)
            self.base_boundary = np.zeros((ny, nx), dtype=bool)
            
            # 设置入口和出口
            self.inlet_mask[10:30, 0] = True
            self.outlet_mask[-30:-10, -1] = True
            # 设置边界墙
            self.base_boundary[0, :] = True
            self.base_boundary[-1, :] = True
            self.base_boundary[:, 0] = True # Inlet处会被mask覆盖
            self.base_boundary[:, -1] = True 

            # 模拟物理场
            y = np.linspace(0, 1, ny)[:, None]
            x = np.linspace(0, 1, nx)[None, :]
            # 温度场：假设出口处热，入口处冷
            self.temp = x * 0.8 + 0.2 * np.random.rand(ny, nx)
            # 速度场：假设中间快
            self.u = 0.5 * (1 - (2*y - 1)**2) 
            self.v = np.zeros_like(self.u)

        def get_fluid_mask(self):
            # 模拟中，这里返回的是物理场有效的区域
            # 在Dummy中我们假设物理场全图有效，或者只在 slime 区域有效
            # 为了测试 Slime 扩张，我们先返回全图有效 (模拟完全开放流体)
            return ~self.base_boundary

    print("启动实时独立 Slime 演化演示...")
    solver = RealTimeDummySolver()
    slime = SlimeAgent(solver)
    slime.initialize_from_fields(solver)

    # 绘图设置
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = LinearSegmentedColormap.from_list("slime", ["#000000", "#00FF00"])
    
    img = ax.imshow(slime.slime, cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Real-time Slime Evolution (Standalone)")
    
    try:
        for i in range(500):
            # 模拟环境变化 (添加一点随机性)
            solver.temp += 0.01 * (np.random.rand(solver.ny, solver.nx) - 0.5)
            
            # 核心更新
            slime.update(solver)
            
            # 渲染
            img.set_data(slime.slime)
            plt.pause(0.01)
            if i % 10 == 0:
                print(f"Step {i}, Slime Coverage: {slime.slime.mean():.2%}")
                
    except KeyboardInterrupt:
        print("Stopped.")
    plt.ioff()
    plt.show()