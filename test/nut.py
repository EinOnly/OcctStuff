import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_nutation_trajectory():
    # --- 1. 参数设置 (Parameters) ---
    num_teeth = 48
    radius = 17.0        # 假设分度圆半径 approx 17mm (对应 OD 34-37mm)
    nutation_angle_deg = 3.0
    theta = np.deg2rad(nutation_angle_deg)
    
    # 模拟驱动轴旋转一圈 (360度)
    steps = 360
    drive_angles = np.linspace(0, 2 * np.pi, steps)

    # --- 2. 初始化齿轮齿中心 (Initial Gear Teeth Centers) ---
    # 在平躺状态下 (Z=0) 的 48 个点的坐标
    # 齿 0 在 X 轴正方向
    tooth_indices = np.arange(num_teeth)
    initial_angles = 2 * np.pi * tooth_indices / num_teeth
    
    # shape: (48, 3) -> [[x0, y0, 0], [x1, y1, 0], ...]
    points_initial = np.zeros((num_teeth, 3))
    points_initial[:, 0] = radius * np.cos(initial_angles)
    points_initial[:, 1] = radius * np.sin(initial_angles)
    points_initial[:, 2] = 0

    # 用于存储所有齿在所有时间步的轨迹
    # shape: (steps, 48, 3)
    trajectories = np.zeros((steps, num_teeth, 3))

    # --- 3. 核心计算循环 (Calculation Loop) ---
    print(f"开始计算 {num_teeth} 个齿的章动轨迹 (无自转模式)...")
    
    for i, phi in enumerate(drive_angles):
        # 这里的 phi 是驱动轴的角度（也就是章动最高点的方位角）
        
        # --- 核心算法：无自转章动 (Pure Wobble) ---
        # 原理：要让圆盘倒向 phi 方向，我们需要绕 "垂直于 phi 的轴" 旋转 theta 角
        # 旋转轴向量 k 在 XY 平面上
        
        # 1. 确定旋转轴 k (Unit Vector)
        # 如果倾斜方向是 phi，旋转轴就是 phi + 90度 (或者 -90度)
        # 这里我们就取 (-sin(phi), cos(phi), 0)
        kx = -np.sin(phi)
        ky = np.cos(phi)
        kz = 0
        
        # Rodrigues 旋转公式 (绕任意轴 k 旋转 theta 角度)
        # v_rot = v * cos(t) + (k x v) * sin(t) + k * (k . v) * (1 - cos(t))
        
        # 为了高效，我们对所有点同时做向量运算
        # P: (48, 3)
        P = points_initial
        
        # K: (3,) -> 广播到 (48, 3)
        K = np.array([kx, ky, kz])
        
        # 点乘 (K . P) -> shape (48,)
        k_dot_p = np.dot(P, K) 
        k_dot_p = k_dot_p[:, np.newaxis] # reshape to (48, 1) for broadcasting
        
        # 叉乘 (K x P) -> shape (48, 3)
        k_cross_p = np.cross(K, P)
        
        # 应用公式
        P_rot = (P * np.cos(theta) + 
                 k_cross_p * np.sin(theta) + 
                 K * k_dot_p * (1 - np.cos(theta)))
        
        trajectories[i, :, :] = P_rot

    # --- 4. 绘图 (Visualization) ---
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)

    # [图 1] 单个齿的 Z 轴高度变化 (验证正弦波)
    ax1 = fig.add_subplot(2, 2, 1)
    target_tooth_idx = 0 # 追踪第0号齿
    z_values = trajectories[:, target_tooth_idx, 2]
    
    ax1.plot(np.degrees(drive_angles), z_values, 'b-', linewidth=2)
    ax1.set_title(f'Tooth #{target_tooth_idx} Vertical Motion (Z-Axis)', fontsize=12)
    ax1.set_xlabel('Driver Angle (Degrees)')
    ax1.set_ylabel('Height (mm)')
    ax1.grid(True)
    ax1.text(0, min(z_values), "  Pure Sine Wave!", color='red')

    # [图 2] 某一瞬间所有齿的空间分布 (侧视)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    snapshot_idx = 45 # 取第45步 (45度时)
    
    xs = trajectories[snapshot_idx, :, 0]
    ys = trajectories[snapshot_idx, :, 1]
    zs = trajectories[snapshot_idx, :, 2]
    
    # 画出圆盘面
    ax2.plot_trisurf(xs, ys, zs, alpha=0.3, color='orange')
    ax2.scatter(xs, ys, zs, c='r', s=10) # 齿中心点
    
    # 画出最高点和最低点
    max_z_idx = np.argmax(zs)
    min_z_idx = np.argmin(zs)
    ax2.scatter(xs[max_z_idx], ys[max_z_idx], zs[max_z_idx], c='g', s=50, label='Highest')
    ax2.scatter(xs[min_z_idx], ys[min_z_idx], zs[min_z_idx], c='b', s=50, label='Lowest')
    
    ax2.set_zlim(-2, 2)
    ax2.set_title(f'Snapshot of Rotor at {snapshot_idx} degrees', fontsize=12)
    ax2.legend()

    # [图 3] 验证“无自转”：第0号齿的俯视轨迹 (X-Y 平面)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # 取出第0号齿的所有轨迹
    t0_x = trajectories[:, 0, 0]
    t0_y = trajectories[:, 0, 1]
    
    ax3.plot(t0_x, t0_y, 'k.-', markersize=2, alpha=0.5)
    # 画出初始位置
    ax3.plot(t0_x[0], t0_y[0], 'ro', label='Start')
    
    # 强制等比例，否则看不出它是圆还是点
    ax3.axis('equal') 
    ax3.set_title('Top View of Tooth #0 Path (Proof of No Rotation)', fontsize=12)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.legend()
    
    # 计算它跑了多远（应该非常小，只在原地画8字或弧线）
    drift_distance = np.sqrt((max(t0_x)-min(t0_x))**2 + (max(t0_y)-min(t0_y))**2)
    ax3.text(min(t0_x), min(t0_y), f"Max XY Drift: {drift_distance:.4f} mm\n(Essentially 0 for human eye)", verticalalignment='top')

    # [图 4] 展开图 (所有齿的波浪)
    ax4 = fig.add_subplot(2, 2, 4)
    # 取 snapshot 时刻，所有齿的 Z 值按齿的索引排列
    current_zs = trajectories[snapshot_idx, :, 2]
    ax4.bar(np.arange(num_teeth), current_zs, color='purple', alpha=0.6)
    ax4.set_title('Instantaneous Z-Height of All 48 Teeth', fontsize=12)
    ax4.set_xlabel('Tooth Index (0-47)')
    ax4.set_ylabel('Z Height (mm)')
    ax4.grid(True, axis='y')

    plt.suptitle(f"Nutation Gear Motion Analysis (Tilt: {nutation_angle_deg}deg, Teeth: {num_teeth})", fontsize=16)
    plt.show()

if __name__ == "__main__":
    calculate_nutation_trajectory()