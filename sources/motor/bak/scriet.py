import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# ==========================================
# 1. 核心参数 (单位: mm)
# ==========================================
# 这是一个经典的几何组合
Z_rotor = 10             # 摆轮齿数 (为了视觉清晰，改用10:11的比例，更明显)
Z_pins = 11              # 针轮齿数 (Z_rotor + 1)
e = 0.8                  # 偏心距 (稍微加大一点，让摆动更明显)
R_pin = 1.0              # 针销半径 (真实的针销通常比较粗)

# --- 关键尺寸自动计算 ---
# 针轮节圆半径 (R_stator)
# 工业设计中通常设定 K = R_stator / (e * Z_pins)
# 这里我们反推：为了不干涉，且齿形饱满
# 设定一个基准 R_stator
R_stator_case = 15.0     

# 减速比
reduction_ratio = Z_rotor  # 输入转一圈，输出转 1/Z_rotor (如果是针轮固定)
# 这里的配置是：输入(偏心)转，输出(针轮)减速转。 ratio = Z_pins / (Z_pins - Z_rotor) = 11
ratio_output = Z_pins / (Z_pins - Z_rotor)

print(f"=== 真实摆线参数 ===")
print(f"针轮节圆直径: {R_stator_case*2}mm")
print(f"偏心距: {e}mm")
print(f"针销半径: {R_pin}mm")
print(f"减速比: {int(ratio_output)}:1")

# ==========================================
# 2. 真实摆线方程 (Epitrochoid Equidistant)
# ==========================================
def get_real_cycloid_shape(R_stator, e, r_pin, z_rotor, points=2000):
    """
    计算短幅外摆线等距曲线 (真正的摆线齿轮形状)
    R_stator: 针轮节圆半径
    e: 偏心距
    r_pin: 针销半径
    z_rotor: 摆轮齿数
    """
    z_pins = z_rotor + 1
    # 参数角 psi (0 到 2pi * z_rotor，闭合曲线需要转多圈参数，或者利用周期性)
    # 对于标准方程，0-2pi 是一周
    psi = np.linspace(0, 2*np.pi, points)
    
    # --- 第一步：计算外摆线 (针销中心在摆轮坐标系下的轨迹) ---
    # 注意公式中的符号，取决于坐标系定义，这里采用标准型
    # A = R_stator
    # B = e * z_pins
    
    # 基础摆线 (Epicycloid center path)
    # 摆轮坐标系下，针销中心的轨迹
    x0 = R_stator * np.cos(psi) - e * np.cos(z_pins * psi)
    y0 = -R_stator * np.sin(psi) + e * np.sin(z_pins * psi) # 注意y方向符号配合旋转方向
    
    # --- 第二步：计算法向量并向内收缩 (等距曲线) ---
    # 我们需要减去针销半径。需要计算切线/法线角度。
    
    # 对 psi 求导
    dx = -R_stator * np.sin(psi) + e * z_pins * np.sin(z_pins * psi)
    dy = -R_stator * np.cos(psi) + e * z_pins * np.cos(z_pins * psi)
    
    # 计算法线分量 (归一化)
    # 距离因子
    dist = np.sqrt(dx**2 + dy**2)
    
    # 法线方向的偏移量 (垂直于切线)
    # Normal vector: (-dy, dx) or (dy, -dx) depending on direction
    # 对于内收缩，我们通常从中心指向外? 不，是从轨迹指向内。
    # 经过推导，等距曲线坐标为：
    
    x = x0 + r_pin * (dy / dist)
    y = y0 - r_pin * (dx / dist)
    
    return x, y

# ==========================================
# 3. 初始化绘图
# ==========================================
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_aspect('equal')
lim = R_stator_case + R_pin + 2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_title(f'True Cycloidal Meshing (Equidistant Curve)\nNote the contact area ("Meshing") vs Gap', fontsize=12)
ax.grid(True, alpha=0.2)

# 1. 绘制摆轮 (实体轮廓)
# 获取真实形状
rotor_x_raw, rotor_y_raw = get_real_cycloid_shape(R_stator_case, e, R_pin, Z_rotor)
rotor_polygon, = ax.plot([], [], 'b-', linewidth=1.5, label='Cycloid Rotor')
rotor_fill = ax.fill([], [], 'blue', alpha=0.1)[0] # 填充颜色方便看干涉

# 2. 绘制针销 (Output Pins)
pins = []
pin_x_ref, pin_y_ref = [], []
angles_pins = np.linspace(0, 2*np.pi, Z_pins, endpoint=False)
# 针销初始位置
pin_centers_x = R_stator_case * np.cos(angles_pins)
pin_centers_y = R_stator_case * np.sin(angles_pins)

for i in range(Z_pins):
    # 红色圆圈代表针销
    pin = patches.Circle((0, 0), R_pin, facecolor='#ffcccc', edgecolor='red', linewidth=1)
    ax.add_patch(pin)
    pins.append(pin)

# 3. 驱动偏心轴
eccentric_point, = ax.plot([], [], 'k+', markersize=10, markeredgewidth=2, label='Eccentric Input')
eccentric_arm, = ax.plot([], [], 'k-', alpha=0.3)

# 4. 辅助圆 (针销节圆)
ax.add_patch(patches.Circle((0,0), R_stator_case, fill=False, linestyle=':', alpha=0.3))

ax.legend(loc='upper right')

# ==========================================
# 4. 动画逻辑
# ==========================================
def update(frame):
    # 输入轴角度 (Input Shaft)
    theta_input = np.radians(frame)
    
    # --- 运动学解算 ---
    # 设定：针轮(外壳)旋转输出，摆轮仅平移(在此时参照系下)
    # 实际上，如果摆轮不自转，针轮必须反向旋转来啮合
    
    # 减速比 = Z_pins / (Z_pins - Z_rotor) = 11
    theta_output = theta_input / ratio_output
    
    # 1. 更新针轮位置 (输出旋转)
    cos_out = np.cos(theta_output)
    sin_out = np.sin(theta_output)
    
    current_pin_x = pin_centers_x * cos_out - pin_centers_y * sin_out
    current_pin_y = pin_centers_x * sin_out + pin_centers_y * cos_out
    
    for i, p in enumerate(pins):
        p.center = (current_pin_x[i], current_pin_y[i])
        
    # 2. 更新摆轮位置 (偏心平移)
    # 摆轮中心坐标
    cx = e * np.cos(theta_input)
    cy = e * np.sin(theta_input)
    
    # 平移摆轮
    # 关键：由于针轮转了，我们需要确保摆轮的相位是匹配的。
    # 在这个生成函数里，摆轮形状已经是“包含了自转几何”的静止形状，
    # 所以我们只需要平移它即可。
    
    final_x = rotor_x_raw + cx
    final_y = rotor_y_raw + cy
    
    rotor_polygon.set_data(final_x, final_y)
    
    # 更新填充
    xy = np.column_stack((final_x, final_y))
    rotor_fill.set_xy(xy)
    
    # 3. 更新偏心轴
    eccentric_point.set_data([cx], [cy])
    eccentric_arm.set_data([0, cx], [0, cy])
    
    return [rotor_polygon, eccentric_point, eccentric_arm, rotor_fill] + pins

# ==========================================
# 5. 执行
# ==========================================
# 慢动作演示啮合细节
ani = FuncAnimation(fig, update, frames=np.arange(0, 360*2, 2), 
                    interval=20, blit=True)

plt.show()