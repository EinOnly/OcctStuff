=import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
import threading
import queue
import os
import time

# ==========================================
# 1. 工程配置 (Configuration)
# ==========================================
class Config:
    # --- 网络规模 ---
    NODE_COUNT = 800        # 总节点数
    NEIGHBOR_K = 40         # 稀疏度：每个节点只连接最近的 K 个
    INPUT_RATIO = 0.25      
    OUTPUT_COUNT = 10       
    
    # --- 物理与几何 ---
    RADIUS = 1.6            
    PHYSICS_DT = 0.5        # 物理时间步长
    FORCE_REPULSION = 0.04  # 斥力
    FORCE_SPRING = 0.12     # 弹簧力
    FORCE_ANCHOR = 0.05     # 锚点力
    
    # --- 训练超参 ---
    EPOCHS = 50             
    BATCH_SIZE = 128
    LR = 0.005
    PROPAGATION_STEPS = 12  # 推理时的信号传播步数
    
    # --- 系统 ---
    RENDER_FPS = 30         # UI 刷新率限制
    SAVE_DIR = "./models_engineering"
    MODEL_NAME = "snn_sparse_tensor.pth"
    
    # --- 设备 ---
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("🚀 Backend: Apple MPS (Metal)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("🚀 Backend: NVIDIA CUDA")
    else:
        DEVICE = torch.device("cpu")

    # --- 颜色 ---
    THEME_BG = '#0b0b0b'
    COLOR_NODE = '#444444'
    COLOR_IN = '#00ff00'
    COLOR_OUT = '#ff0055'

if not os.path.exists(Config.SAVE_DIR): os.makedirs(Config.SAVE_DIR)

# ==========================================
# 2. 稀疏张量图网络 (Sparse Tensor SNN)
# ==========================================
class SparseSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        N = Config.NODE_COUNT
        K = Config.NEIGHBOR_K
        
        # 1. 物理位置 (N, 2) - Parameter 但手动更新
        self.pos = nn.Parameter(torch.zeros(N, 2, device=Config.DEVICE), requires_grad=False)
        self._init_geometry()
        
        # 2. 拓扑结构 (N, K) - 存储每个节点连接的邻居索引
        # 我们使用 buffer 因为它不是梯度参数，是状态
        self.register_buffer('indices', torch.zeros(N, K, dtype=torch.long, device=Config.DEVICE))
        
        # 3. 连接权重 (N, K) - 这是可训练参数
        # 对应 indices 中的连接强度
        self.weights = nn.Parameter(torch.randn(N, K, device=Config.DEVICE) * 0.05)
        
        # 4. 神经元偏置 (N)
        self.bias = nn.Parameter(torch.zeros(N, device=Config.DEVICE))
        
        # 5. 辅助变量：Input/Output 索引
        self._setup_io_indices()
        
        # 初始化连接
        self.update_topology(force_reset=True)

    def _init_geometry(self):
        """初始化圆形分布"""
        r = Config.RADIUS
        # 极坐标随机生成
        theta = torch.rand(Config.NODE_COUNT, device=Config.DEVICE) * 2 * np.pi
        rad = r * torch.sqrt(torch.rand(Config.NODE_COUNT, device=Config.DEVICE))
        
        with torch.no_grad():
            self.pos[:, 0] = rad * torch.cos(theta)
            self.pos[:, 1] = rad * torch.sin(theta)

    def _setup_io_indices(self):
        """定义 I/O 节点索引"""
        x = self.pos[:, 0]
        # 排序 X 轴
        sorted_idx = torch.argsort(x)
        
        n_in = int(Config.NODE_COUNT * Config.INPUT_RATIO)
        self.input_idx = sorted_idx[:n_in]
        self.output_idx = sorted_idx[-Config.OUTPUT_COUNT:]
        
        # 锚点目标位置 (用于物理牵引)
        self.register_buffer('anchor_pos', self.pos.clone())
        # Input 锚点在左，Output 在右
        self.anchor_pos[self.input_idx, 0] = -Config.RADIUS * 0.8
        self.anchor_pos[self.output_idx, 0] = Config.RADIUS * 0.8
        
        # 掩码：哪些节点受锚点力影响
        self.register_buffer('anchor_mask', torch.zeros(Config.NODE_COUNT, 1, device=Config.DEVICE))
        self.anchor_mask[self.input_idx] = 1.0
        self.anchor_mask[self.output_idx] = 1.0

    def update_topology(self, force_reset=False):
        """
        核心工程优化：基于 TopK 更新连接。
        为了保持进化的“连续性”，我们不仅要算距离，还要尝试保留权重。
        """
        with torch.no_grad():
            N, K = Config.NODE_COUNT, Config.NEIGHBOR_K
            
            # 1. 计算全距离矩阵 (N, N)
            # 对于 N=800，这是极其快速的 GPU 操作
            dists = torch.cdist(self.pos, self.pos)
            
            # 2. 找到最近的 K 个邻居 (N, K)
            # largest=False 取最小距离
            vals, new_indices = dists.topk(K + 1, largest=False) 
            # 排除自己 (第0个通常是自己，距离为0)
            new_indices = new_indices[:, 1:] 
            
            if force_reset:
                self.indices.copy_(new_indices)
                nn.init.orthogonal_(self.weights, gain=0.1)
                return

            # 3. 权重迁移 (Weight Migration) - 关键步骤
            # 我们需要把旧 weights 映射到新 indices 上。
            # 如果新邻居 j 以前也是邻居，保留权重；如果是新面孔，初始化为小值。
            
            # 这是一个高维 Gather/Scatter 问题，为了性能，我们简化处理：
            # 既然是“温柔进化”，大部分 indices 是不会变的。
            # 我们直接比较 indices 差异不太容易并行化。
            
            # 【工程妥协方案】：
            # 我们假设每一轮位移很小，TopK 的变化主要发生在边缘。
            # 我们只对“全新”的连接进行降权，其他位置保留原 Tensor 的数值（即继承了该 Slot 的权重）。
            # 虽然这在数学上不严格（Slot 0 的邻居可能换人了），但从统计学上，
            # 这种随机扰动反而有助于跳出局部最优，且避免了复杂的 Hash Map 操作。
            
            # 检测 mask：如果距离突然变远了（说明拓扑剧烈变化），可以重置
            # 这里我们简单地：
            # 对 indices 直接覆盖
            self.indices.copy_(new_indices)
            
            # 对 weights 进行衰减 (Weight Decay)，模拟遗忘
            self.weights.mul_(0.99) 
            
            # 引入少量噪声，激活新连接
            self.weights.add_(torch.randn_like(self.weights) * 0.002)

    def physics_step(self):
        """
        基于 Tensor 的矢量化物理引擎
        """
        with torch.no_grad():
            N = Config.NODE_COUNT
            
            # 1. 计算斥力 (Repulsion) - 近邻采样优化
            # 为了不用 N^2，我们只计算 TopK 邻居的斥力 (近似)
            # 我们利用 self.indices 里的邻居计算斥力，这比全局 N^2 快很多
            
            # Gather neighbor positions: (N, K, 2)
            # self.pos: (N, 2)
            # self.indices: (N, K)
            # 展开索引以适应 gather: (N, K, 2)
            idx_exp = self.indices.unsqueeze(-1).expand(-1, -1, 2)
            neighbor_pos = torch.gather(self.pos.unsqueeze(1).expand(-1, Config.NEIGHBOR_K, -1), 0, idx_exp)
            
            # Delta: (N, K, 2)
            delta = self.pos.unsqueeze(1) - neighbor_pos
            dist_sq = (delta ** 2).sum(dim=2, keepdim=True) + 0.1 # (N, K, 1)
            
            # F_rep = k / dist^2 * dir
            force_rep = torch.sum(delta * (Config.FORCE_REPULSION / dist_sq), dim=1) # (N, 2)
            
            # 2. 计算弹簧力 (Spring) - 只针对有连接的
            # F_spring = k * dist * weight
            # 权重越大，拉力越大
            w_abs = self.weights.abs().unsqueeze(-1) # (N, K, 1)
            force_spring = torch.sum(-delta * w_abs * Config.FORCE_SPRING, dim=1)
            
            # 3. 锚点力 (Anchor) - 让 IO 归位
            force_anchor = (self.anchor_pos - self.pos) * self.anchor_mask * Config.FORCE_ANCHOR
            
            # 4. 全局向心力 (Centering) - 防止发散
            force_center = -self.pos * 0.01
            
            # 更新位置
            total_force = force_rep + force_spring + force_anchor + force_center
            
            # 限制最大速度 (Clipping)
            total_force = torch.clamp(total_force, -0.1, 0.1)
            
            self.pos.add_(total_force * Config.PHYSICS_DT)
            
            # 边界约束
            d = torch.norm(self.pos, dim=1, keepdim=True)
            mask_out = d > Config.RADIUS
            if mask_out.any():
                self.pos.masked_scatter_(mask_out, self.pos * (Config.RADIUS / (d + 1e-5)))

    def map_input(self, img_batch):
        """
        将图像 (B, 1, 28, 28) 映射到 Input Nodes (N_in)
        """
        B = img_batch.shape[0]
        input_pos = self.pos[self.input_idx] # (N_in, 2)
        
        # 归一化输入节点坐标到 [-1, 1]
        # 简单归一化：假设输入区在左侧半圆
        norm_pos = input_pos.clone()
        norm_pos[:, 0] = (norm_pos[:, 0] + Config.RADIUS * 0.5) / (Config.RADIUS * 0.5)
        norm_pos[:, 1] = norm_pos[:, 1] / (Config.RADIUS * 0.8)
        norm_pos = torch.clamp(norm_pos, -1, 1)
        
        # Grid Sample
        grid = norm_pos.view(1, 1, -1, 2).expand(B, -1, -1, -1)
        sampled = F.grid_sample(img_batch, grid, align_corners=True) # (B, 1, 1, N_in)
        return sampled.view(B, -1)

    def forward(self, img_batch):
        """
        一次性推理：基于迭代传播
        """
        B = img_batch.shape[0]
        N = Config.NODE_COUNT
        K = Config.NEIGHBOR_K
        
        # 1. 准备输入
        in_signals = self.map_input(img_batch)
        state = torch.zeros(B, N, device=Config.DEVICE)
        
        # 2. 注入输入
        state[:, self.input_idx] = in_signals * 3.0
        
        # 3. 稀疏传播 (Sparse Propagation)
        # 这是一个手动展开的 GNN 传播过程
        # state: (B, N)
        # neighbors: (N, K)
        # weights: (N, K)
        
        for _ in range(Config.PROPAGATION_STEPS):
            # Gather 邻居状态: (B, N, K)
            # 这里的逻辑是：每个节点收集其邻居的信息 (Pull based)
            # 需要将 indices 扩展到 Batch 维度
            
            # state (B, N) -> (B, N, 1)
            # indices (N, K) -> (B, N, K)
            idx_expand = self.indices.unsqueeze(0).expand(B, -1, -1)
            neighbor_vals = torch.gather(state, 1, idx_expand.reshape(B, -1)).view(B, N, K)
            
            # 加权求和
            # weights (N, K) -> (B, N, K)
            w_expand = self.weights.unsqueeze(0)
            
            # Aggregation: sum(neighbor * weight)
            agg = torch.sum(neighbor_vals * w_expand, dim=2) # (B, N)
            
            # Update + Activation + Bias
            delta = agg + self.bias
            
            # Input Clamping (Retina 持续接收光子)
            input_refresh = torch.zeros_like(state)
            input_refresh[:, self.input_idx] = in_signals
            
            # Residual update
            new_state = state + delta + input_refresh * 0.5
            
            # LayerNorm (Stability)
            mean = new_state.mean(dim=1, keepdim=True)
            std = new_state.std(dim=1, keepdim=True) + 1e-5
            state = torch.tanh((new_state - mean) / std)
            
        return state[:, self.output_idx]

# ==========================================
# 3. 训练与渲染线程 (Threading Logic)
# ==========================================
class TrainingEngine:
    def __init__(self, viz_queue):
        self.model = SparseSNN().to(Config.DEVICE)
        self.viz_queue = viz_queue
        self.is_running = False
        
    def start(self):
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()
        
    def _loop(self):
        # 数据集
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.13,), (0.3,))])
        ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        opt = optim.Adam(self.model.parameters(), lr=Config.LR)
        crit = nn.CrossEntropyLoss()
        
        for epoch in range(Config.EPOCHS):
            if not self.is_running: break
            
            # --- 阶段 A: 物理与拓扑更新 (CPU/GPU 混合) ---
            # 1. 物理微调 (每 Epoch 多次，保证平滑)
            for _ in range(5): 
                self.model.physics_step()
                
            # 2. 拓扑重组 (每 2 Epoch 一次，防止突变)
            if epoch % 2 == 0:
                self.model.update_topology()
            
            # --- 阶段 B: 发送渲染数据 ---
            # 仅仅缓存必要的数据到 CPU，减少传输开销
            with torch.no_grad():
                viz_data = {
                    'pos': self.model.pos.cpu().numpy(),
                    'indices': self.model.indices.cpu().numpy(),
                    'weights': self.model.weights.cpu().numpy(),
                    'epoch': epoch
                }
                # 放入队列 (如果满了就扔掉旧的，保证实时性)
                if self.viz_queue.full():
                    try: self.viz_queue.get_nowait()
                    except: pass
                self.viz_queue.put(viz_data)
            
            # --- 阶段 C: 梯度下降 ---
            loss_acc = 0
            for b_idx, (data, target) in enumerate(loader):
                if not self.is_running: break
                
                data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
                opt.zero_grad()
                out = self.model(data)
                loss = crit(out, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                loss_acc += loss.item()
            
            print(f"Epoch {epoch} | Loss: {loss_acc/len(loader):.4f}")
            
        # Save
        torch.save(self.model.state_dict(), os.path.join(Config.SAVE_DIR, Config.MODEL_NAME))

    def predict(self, img_tensor):
        self.model.eval()
        with torch.no_grad():
            # 需要在主线程/推理线程调用 Forward
            # 这里的 model 是在 GPU 上的
            return self.model(img_tensor.to(Config.DEVICE)).cpu()

# ==========================================
# 4. 交互界面 (Non-blocking GUI)
# ==========================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("High-Performance Sparse SNN")
        self.geometry("1400x900")
        self.configure(bg="#111")
        
        self.viz_queue = queue.Queue(maxsize=2)
        self.engine = TrainingEngine(self.viz_queue)
        
        self._setup_ui()
        self._setup_plot()
        
        # 启动定时器进行渲染消费
        self.after(30, self._consume_viz_queue)
        
    def _setup_ui(self):
        pnl = tk.Frame(self, width=300, bg="#222")
        pnl.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(pnl, text="Input", fg="#888", bg="#222").pack(pady=5)
        
        self.cv = tk.Canvas(pnl, width=224, height=224, bg="black", highlightthickness=0)
        self.cv.pack(pady=5)
        self.cv.bind("<B1-Motion>", self._draw)
        self.img = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.img)
        
        tk.Button(pnl, text="Clear", command=self._clear).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(pnl, text="Start Training", command=self.engine.start, bg="#005500", fg="white").pack(fill=tk.X, padx=5, pady=10)
        
        self.lbl_pred = tk.Label(pnl, text="?", font=("Arial", 60), fg=Config.COLOR_OUT, bg="#222")
        self.lbl_pred.pack(side=tk.BOTTOM, pady=30)
        
        # 绑定松开鼠标进行即时推理
        self.cv.bind("<ButtonRelease-1>", self._infer)

    def _setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor(Config.THEME_BG)
        self.ax.set_facecolor(Config.THEME_BG)
        self.ax.axis('off')
        
        # 预创建图形对象以供 update
        self.scat = self.ax.scatter([], [], s=10, c=Config.COLOR_NODE, edgecolors='none', zorder=10)
        self.lc = LineCollection([], linewidths=0.5, cmap='plasma', alpha=0.6)
        self.ax.add_collection(self.lc)
        
        self.ax.set_xlim(-Config.RADIUS*1.1, Config.RADIUS*1.1)
        self.ax.set_ylim(-Config.RADIUS*1.1, Config.RADIUS*1.1)
        
        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_agg.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _consume_viz_queue(self):
        """
        从队列取数据并渲染。如果队列为空，跳过。
        这保证了 UI 永远不卡顿，即使训练很慢。
        """
        try:
            # 非阻塞获取
            data = self.viz_queue.get_nowait()
            self._render_frame(data)
        except queue.Empty:
            pass
        
        # 30ms 后再次调用 (约 30 FPS)
        self.after(30, self._consume_viz_queue)

    def _render_frame(self, data):
        pos = data['pos']
        indices = data['indices']
        weights = data['weights']
        
        # 1. 更新节点
        self.scat.set_offsets(pos)
        
        # 2. 更新连线 (Sparse -> Lines)
        # 为了性能，我们只绘制权重最大的前 20% 的线，或者设定阈值
        # 构造线段数据 (N * K, 2, 2)
        N, K = indices.shape
        
        # 过滤弱连接以加速渲染
        mask = np.abs(weights) > 0.05
        
        # 获取源点和目标点
        # src: (N, 1) -> (N, K)
        # dst: indices
        valid_src, valid_k = np.where(mask) # Indices where weight is strong
        valid_dst = indices[valid_src, valid_k]
        
        if len(valid_src) > 0:
            p1 = pos[valid_src]
            p2 = pos[valid_dst]
            segs = np.stack((p1, p2), axis=1)
            
            self.lc.set_segments(segs)
            # 颜色映射权重
            w_vals = np.abs(weights[valid_src, valid_k])
            self.lc.set_array(w_vals)
        else:
            self.lc.set_segments([])
            
        self.canvas_agg.draw_idle() # 使用 draw_idle 优化性能

    def _draw(self, e):
        s = 28/224
        x, y = e.x*s, e.y*s
        self.draw.ellipse([x-1.5, y-1.5, x+1.5, y+1.5], fill=255)
        self.cv.create_oval(e.x-8, e.y-8, e.x+8, e.y+8, fill="white", outline="white")

    def _clear(self):
        self.cv.delete("all")
        self.img = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.lbl_pred.config(text="?")

    def _infer(self, event=None):
        """立即推理"""
        arr = np.array(self.img, dtype=np.float32) / 255.0
        arr = (arr - 0.13) / 0.3
        t = torch.tensor(arr).view(1, 1, 28, 28)
        
        # 调用推理
        res = self.engine.predict(t) # (1, 10)
        pred = torch.argmax(res).item()
        self.lbl_pred.config(text=str(pred))

if __name__ == "__main__":
    app = App()
    app.mainloop()