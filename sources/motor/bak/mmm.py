import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. 参数设定 (模拟 15mm 直径超扁平电机)
# ==========================================
D_motor = 15.0          # 电机总直径 mm
R_motor = D_motor / 2
H_gap = 1.5             # 定子与中心平面的半气隙高度
Tilt_angle_deg = 5.0    # 转子倾斜角度 (章动角)
Tilt_angle_rad = np.radians(Tilt_angle_deg)

# 网格精度
grid_res = 30

print(f"--- 双边轴向磁通章动电机可视化 ---")
print(f"直径: {D_motor}mm")
print(f"章动角: {Tilt_angle_deg}°")
print("原理: 旋转的轴向磁拉力导致转子进动(滚动)")

# ==========================================
# 2. 几何生成函数
# ==========================================
def create_disk_grid(radius, resolution):
    """生成圆盘的基础网格数据 (极坐标转笛卡尔)"""
    r = np.linspace(0, radius, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return X, Y, R, Theta

def get_tilted_rotor_z(R, Theta, current_phase, tilt_rad):
    """
    计算倾斜转子的Z高度表面。
    核心原理：Z高度取决于半径和相对于当前相位角的角度差。
    最低点出现在 Theta = current_phase 的位置。
    """
    # 使用余弦函数模拟倾斜表面，负号使得 phase 处为最低点
    Z = -R * np.tan(tilt_rad) * np.cos(Theta - current_phase)
    return Z

# ==========================================
# 3. 初始化绘图场景
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-R_motor*1.2, R_motor*1.2)
ax.set_ylim(-R_motor*1.2, R_motor*1.2)
ax.set_zlim(-H_gap*2, H_gap*2)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
# 调整视角以更好地观察章动
ax.view_init(elev=30, azim=45)
ax.set_title("Axial Flux Nutating Motor Concept\n(Double-Sided Stator Drive)", fontsize=12)

# --- 绘制静态部件 (双边定子) ---
X_base, Y_base, R_base, Theta_base = create_disk_grid(R_motor, grid_res)
# 下定子 (Bottom Stator) - 半透明蓝色
ax.plot_surface(X_base, Y_base, np.full_like(X_base, -H_gap), color='blue', alpha=0.2, edgecolor='none')
ax.text(R_motor, 0, -H_gap, "Bottom Stator (Coils)", color='blue')
# 上定子 (Top Stator) - 半透明蓝色
ax.plot_surface(X_base, Y_base, np.full_like(X_base, H_gap), color='blue', alpha=0.2, edgecolor='none')
ax.text(R_motor, 0, H_gap, "Top Stator (Coils)", color='blue')

# --- 初始化动态部件 ---
# 1. 转子 (Rotor) - 初始状态 (相位0)
Z_rotor_init = get_tilted_rotor_z(R_base, Theta_base, 0, Tilt_angle_rad)
# 使用 coolwarm colormap 表示高低，红色高，蓝色低
rotor_surf = ax.plot_surface(X_base, Y_base, Z_rotor_init, cmap='coolwarm', alpha=0.8, linewidth=0.5, edgecolors='k')

# 2. 力的可视化 (Force Vectors)
# 我们用一个大的红色箭头表示当前最大的“轴向吸力”位置
# 初始位置在角度 0，半径 R_motor*0.8 处，方向向下
q_force_pos_x = R_motor * 0.8
q_force_pos_y = 0
q_force_pos_z = H_gap * 0.8 # 从上定子下方开始
q_force_dir_z = -H_gap * 1.6 # 向下指
force_quiver = ax.quiver(q_force_pos_x, q_force_pos_y, q_force_pos_z, 
                         0, 0, q_force_dir_z, 
                         color='red', linewidth=3, arrow_length_ratio=0.3, label='Magnetic Pull Force')

# 3. 接触点标记 (Contact Point Marker)
# 标记转子最低点，也就是理论上的滚动接触点
contact_point, = ax.plot([R_motor], [0], [-R_motor*np.tan(Tilt_angle_rad)], 'ro', markersize=10, markeredgecolor='y', label='Rolling Contact Point')

# 添加图例和说明文字
ax.legend(loc='upper left')
status_text = fig.text(0.05, 0.05, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# ==========================================
# 4. 动画更新函数
# ==========================================
def update(frame):
    # frame 是当前磁场旋转的角度 (相位)
    current_phase = np.radians(frame)
    
    # --- 更新转子姿态 ---
    # 计算新的倾斜表面 Z 值
    Z_rotor_new = get_tilted_rotor_z(R_base, Theta_base, current_phase, Tilt_angle_rad)
    
    # Matplotlib 3D surface 更新需要移除旧的再添加新的
    global rotor_surf
    rotor_surf.remove()
    rotor_surf = ax.plot_surface(X_base, Y_base, Z_rotor_new, cmap='coolwarm', alpha=0.8, linewidth=0.5, edgecolors='k')
    
    # --- 更新力的矢量 (旋转的吸力) ---
    # 力矢量的位置随着相位旋转
    fx = R_motor * 0.8 * np.cos(current_phase)
    fy = R_motor * 0.8 * np.sin(current_phase)
    fz_start = H_gap * 0.8
    fz_dir = -H_gap * 1.6
    
    global force_quiver
    force_quiver.remove()
    force_quiver = ax.quiver(fx, fy, fz_start, 0, 0, fz_dir, color='red', linewidth=3, arrow_length_ratio=0.3)
    
    # --- 更新接触点标记 ---
    # 接触点在当前相位的最低处
    cx = R_motor * np.cos(current_phase)
    cy = R_motor * np.sin(current_phase)
    cz = -R_motor * np.tan(Tilt_angle_rad) # 简化的最低点高度
    contact_point.set_data([cx], [cy])
    contact_point.set_3d_properties([cz])
    
    # --- 更新说明文字 ---
    # 计算一个简化的机械角度显示 (假设减速比 50:1)
    mech_angle = frame / 50.0
    status_text.set_text(f"Magnetic Field Phase: {frame:.1f}°\n"
                         f"Driving Force: Rotating Axial Pull\n"
                         f"Rotor Motion: Nutation (Wobbling)\n"
                         f"Output (Contact Point): Slow Rolling (~{mech_angle:.1f}°)")
    
    return rotor_surf, force_quiver, contact_point

# ==========================================
# 5. 生成动画
# ==========================================
# 磁场旋转 360 度
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), 
                    interval=30, blit=False)

plt.show()
