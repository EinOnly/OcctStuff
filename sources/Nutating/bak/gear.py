import numpy as np

# --- V1 原型机参数 (40mm) ---
D_OUTER = 40e-3      # 外径 40mm
D_INNER = 20e-3      # 内径 20mm (留给球轴承的空间)
GAP = 0.5e-3         # 气隙 (0.5mm, V1 保守一点)
B_FIELD = 1.0        # 磁感应强度 (1T, 钕铁硼 N42 级)
NUTATION_ANGLE = 3.0 # 章动角 (3度, 常用值)
N_ROTOR = 48         # 转子齿数
N_SHELL = 50         # 外壳齿数
FRICTION_COEFF = 0.1 # 摩擦系数 (树脂对树脂/润滑)

# --- 物理常数 ---
MU_0 = 4 * np.pi * 1e-7

# --- 1. 计算轴向吸力 (Maxwell Force) ---
# 有效磁作用面积 (假设只有 1/6 的区域处于强吸合状态)
Area_total = np.pi * ((D_OUTER/2)**2 - (D_INNER/2)**2)
Area_effective = Area_total / 6.0

# F = B^2 * A / 2*mu_0
F_axial = (B_FIELD**2 * Area_effective) / (2 * MU_0)

# --- 2. 计算产生的翻转力矩 (Tilting Torque T0) ---
# 力臂大约是平均半径
R_mean = (D_OUTER + D_INNER) / 4.0
T_0 = F_axial * R_mean

# --- 3. 计算输出扭矩 (Output Torque T1) ---
# 楔形放大倍数 = 1 / tan(alpha)
Wedge_Gain = 1 / np.tan(np.radians(NUTATION_ANGLE))

# 减速比增益 (对于 50:48，减速比 25)
Gear_Ratio = N_SHELL / (N_SHELL - N_ROTOR)

# 估算效率 (考虑摩擦)
# eta = tan(alpha) / tan(alpha + atan(mu))
eta = np.tan(np.radians(NUTATION_ANGLE)) / np.tan(np.radians(NUTATION_ANGLE) + np.arctan(FRICTION_COEFF))

T_out_theoretical = T_0 * Wedge_Gain * eta * Gear_Ratio # 这里减速比是否乘进去取决于能量守恒，保守估算先不乘，因为那是速度比

print(f"=== 40mm V1 原型机理论参数 ===")
print(f"轴向吸力 (F_axial): {F_axial:.2f} N (约 {F_axial/9.8:.1f} kg)")
print(f"翻转力矩 (T_0):    {T_0:.4f} Nm")
print(f"楔形放大倍数:       {Wedge_Gain:.1f} 倍")
print(f"预估传动效率:       {eta*100:.1f} %")
print(f"最终输出扭矩 (保守): {T_0 * Wedge_Gain * eta:.4f} Nm (直驱推力)")
print(f"  *注：如果算上减速比带来的能量转换，扭矩会更大，但首先要克服摩擦自锁。")


import pandas as pd

def generate_crown_profile(teeth, radius, amplitude, filename):
    """
    生成一个圆周上的波浪形点云，保存为CSV用于Fusion 360导入
    """
    # 分辨率：一圈 360 个点足够
    thetas = np.linspace(0, 2*np.pi, 361)
    
    # 坐标列表
    points = []
    
    for theta in thetas:
        # X, Y 是圆
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        # Z 是正弦波 (模拟齿高)
        # 加上一点 bias 保证是单面的
        z = amplitude * np.cos(teeth * theta)
        
        points.append([x, y, z])
        
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False, header=False) # Fusion 读取无头CSV
    print(f"已生成: {filename} (半径 {radius}mm, 齿数 {teeth})")

# 参数配置 (V1 40mm)
R_mean = 15.0  # 30mm直径处的平均齿圆
H_tooth = 1.0  # 齿高 1mm

generate_crown_profile(48, R_mean, H_tooth, 'rotor_48t_profile.csv')
generate_crown_profile(50, R_mean, H_tooth, 'shell_50t_profile.csv')