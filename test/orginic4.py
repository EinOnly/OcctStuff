import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# --- 1. 定义几何生成 (复用之前的逻辑) ---
def get_tpms_slice(nx, ny, period_x=40, period_y=10):
    """生成一个 2D 的各向异性 TPMS 切片作为障碍物"""
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    X, Y = np.meshgrid(x, y)
    
    # 拉伸型 Schwarz D 的 2D 投影模拟
    # sin(x/Px) * cos(y/Py) > t
    k_x = 2 * np.pi / period_x
    k_y = 2 * np.pi / period_y
    
    # 场函数
    field = np.sin(X * k_x) * np.cos(Y * k_y)
    
    # 障碍物掩码 (Mask): True 表示是墙壁 (固体), False 表示是流体
    # 我们取 field > 0.3 为墙壁
    obstacles = field > 0.3
    
    # 强制边界：上下壁面封死
    obstacles[0, :] = True
    obstacles[-1, :] = True
    
    return obstacles

# --- 2. LBM 求解器核心 (D2Q9 模型) ---
def run_lbm_simulation(nx=300, ny=100, steps=1000):
    print("初始化 LBM 求解器...")
    
    # LBM 参数
    tau = 0.6            # 松弛时间 (决定粘度)
    omega = 1.0 / tau
    u_in = 0.1           # 入口流速 (小于 0.1 以保持稳定)
    
    # D2Q9 权重和向量
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    # c: 离散速度方向 [x, y]
    c = np.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], 
                  [1,1], [-1,1], [-1,-1], [1,-1]])
    
    # 反弹方向索引 (用于处理边界)
    noslip = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    
    # 获取几何障碍物
    obstacles = get_tpms_slice(nx, ny, period_x=80, period_y=15)
    
    # 初始化分布函数 (f)
    # 初始状态：密度 rho=1, 速度 u=0
    n_pop = 9
    f = np.zeros((n_pop, ny, nx))
    feq = np.zeros((n_pop, ny, nx))
    rho = np.ones((ny, nx))
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    
    # 初始化为平衡态
    for i in range(n_pop):
        f[i, :, :] = w[i] * rho

    print(f"开始计算 {steps} 步 (纯 Python 会比较慢，请耐心等待)...")
    
    # --- 主循环 ---
    for step in range(steps):
        
        # 1. 宏观量计算 (Macro)
        rho = np.sum(f, axis=0)
        ux = (np.sum(f[[1, 5, 8], :, :], axis=0) - np.sum(f[[3, 6, 7], :, :], axis=0)) / rho
        uy = (np.sum(f[[2, 5, 6], :, :], axis=0) - np.sum(f[[4, 7, 8], :, :], axis=0)) / rho
        
        # 2. 强制入口/出口边界条件 (Zou-He 或者 简单设定)
        # 这里使用简单的强制速度：左侧入口向右吹风
        ux[:, 0] = u_in
        uy[:, 0] = 0
        rho[:, 0] = 1.0 / (1.0 - ux[:, 0]) * (np.sum(f[[0, 2, 4], :, 0], axis=0) + 2*np.sum(f[[3, 6, 7], :, 0], axis=0))
        
        # 3. 碰撞步 (Collision) - BGK 模型
        # 计算平衡态 F_eq
        u2 = ux**2 + uy**2
        for i in range(n_pop):
            cu = c[i,0]*ux + c[i,1]*uy
            feq[i] = rho * w[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
            
        # 碰撞
        f = f + omega * (feq - f)
        
        # 4. 障碍物处理 (Bounce-back)
        # 在固体内部，粒子反弹回来的方向
        # (先简单处理：不更新固体内的f，稍后覆盖)
        
        # 5. 流动步 (Streaming)
        # 粒子移动到邻居格子
        for i in range(n_pop):
            f[i, :, :] = np.roll(np.roll(f[i, :, :], c[i, 0], axis=1), c[i, 1], axis=0)
            
        # 6. 修正障碍物边界 (反弹)
        # 刚才流进去障碍物的粒子，被弹回原来的格子，且方向相反
        # 这一步是 LBM 处理复杂几何的神技
        # 找到障碍物位置
        mask = obstacles
        bounced_f = f[noslip, :, :]
        # 将障碍物位置的 f 替换为反弹后的值
        for i in range(n_pop):
             f[i][mask] = bounced_f[i][mask]

        if step % 100 == 0:
            print(f"Step: {step}/{steps}")

    return ux, uy, obstacles

# --- 可视化 ---
def visualize_results(ux, uy, obstacles):
    velocity_mag = np.sqrt(ux**2 + uy**2)
    
    # 遮挡住固体部分以便显示
    velocity_mag_masked = np.ma.masked_where(obstacles, velocity_mag)

    plt.figure(figsize=(12, 4))
    plt.title("LBM Simulation of TPMS Slice (Re = Low)")
    
    # 绘制流速云图
    plt.imshow(velocity_mag_masked, cmap='jet', origin='lower')
    plt.colorbar(label='Velocity Magnitude')
    
    # 绘制流线
    # 下采样一下流线，不然太密
    Y, X = np.mgrid[0:ux.shape[0], 0:ux.shape[1]]
    skip = 4
    plt.streamplot(X[::skip, ::skip], Y[::skip, ::skip], 
                   ux[::skip, ::skip], uy[::skip, ::skip], 
                   color='white', linewidth=0.5, density=1.5, arrowsize=0.5)
    
    # 绘制障碍物轮廓
    plt.contour(obstacles, levels=[0.5], colors='black', linewidths=1)
    
    plt.xlabel("X (Flow Direction)")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行轻量级仿真
    ux, uy, obstacles = run_lbm_simulation(nx=300, ny=100, steps=1500)
    visualize_results(ux, uy, obstacles)