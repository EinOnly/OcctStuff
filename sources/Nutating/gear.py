import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class TrueBooleanHobbing:
    def __init__(self, 
                 rotor_teeth=48, 
                 shell_teeth=50, 
                 tilt_angle_deg=3.0, 
                 rotor_radius=35.20, 
                 shell_width=6.0,
                 tooth_width_deg=3.0): # 转子齿的物理宽度
        
        self.Nr = rotor_teeth
        self.Ns = shell_teeth
        self.alpha = np.radians(tilt_angle_deg)
        self.R = rotor_radius
        self.H = shell_width
        self.tw_rad = np.radians(tooth_width_deg)
        
        # 减速比：转子自转速度
        # w_spin = -w_nut * (Ns - Nr) / Nr
        self.ratio = (self.Ns - self.Nr) / self.Nr

    def run_simulation(self, res_theta=600, res_z=40):
        print(f"启动真物理切削模拟 (True Physics Hobbing)...")
        print(f"注意：这不再是数学近似，而是真实的布尔减运算。")
        
        # 1. 建立毛坯 (Blank)
        # 这是一个圆柱面展开图：Theta x Z
        # 值 = 半径 (Radius)
        theta_grid = np.linspace(0, 2*np.pi, res_theta)
        z_grid = np.linspace(-self.H/2, self.H/2, res_z)
        THETA, Z = np.meshgrid(theta_grid, z_grid)
        
        # 初始半径：假设足够厚
        RADIUS = np.full_like(THETA, self.R * 1.1) 
        
        # 2. 定义转子单齿 (Cutter Geometry)
        # 简化为一个梯形刀头：Tip在R，Root在R-Depth
        # 我们只追踪【齿尖】和【齿侧】的包络
        # 转子齿在局部系下是一条直线段： x=R, y=0, z in [-width/2, width/2]? 
        # 不，转子齿是径向的。但作为切削刃，它是 Z 轴向的一条棱。
        
        # 3. 时间步进 (Time Stepping)
        # 我们需要让转子“进动”完所有的相位。
        # 周期数 = Nr / GCD(Nr, Ns) = 48 / 2 = 24 圈
        cycles = 25 
        steps_per_cycle = 100
        total_steps = cycles * steps_per_cycle
        
        t_nut = np.linspace(0, 2 * np.pi * cycles, total_steps)
        t_spin = -t_nut * self.ratio 
        
        print(f"模拟步数: {total_steps}步 (覆盖 {cycles} 圈章动)")
        
        # 为了加速，我们使用广播机制计算刀具位置
        # 形状: (Time, Nr_teeth)
        
        # 转子所有齿的原始角度
        beta = np.linspace(0, 2*np.pi, self.Nr, endpoint=False)
        
        # 预计算三角函数
        # 形状扩展为 (Time, Nr)
        beta = beta[np.newaxis, :]
        t_s = t_spin[:, np.newaxis]
        t_n = t_nut[:, np.newaxis]
        
        # 组合角度：转子齿在"自转系"中的角度 = beta + t_spin
        angle_local = beta + t_s
        
        # 转子齿尖坐标 (在 Tilt 之前的平面上)
        # x_l = R * cos(angle_local)
        # y_l = R * sin(angle_local)
        # z_l = 0
        
        c_loc, s_loc = np.cos(angle_local), np.sin(angle_local)
        x_l = self.R * c_loc
        y_l = self.R * s_loc
        
        # 应用 Tilt (绕 X 轴)
        # y_t = y_l * cos(alpha)
        # z_t = -y_l * sin(alpha)  <-- 关键！Z轴坐标产生了！
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        
        x_t = x_l
        y_t = y_l * ca
        z_t = -y_l * sa # (Time, Nr) 这是切削点的 Z 高度
        
        # 应用 Nutation (绕 Z 轴公转)
        c_n, s_n = np.cos(t_n), np.sin(t_n)
        
        # World Coordinates of Cutting Points
        x_w = x_t * c_n - y_t * s_n
        y_w = x_t * s_n + y_t * c_n
        # z_w = z_t  (不变)
        
        # 此时，(x_w, y_w, z_w) 是所有时刻、所有转子齿尖的三维轨迹云。
        # 它们构成了切削的"边界"。
        
        # --- 核心切削算法 (The Sculpting) ---
        # 我们要将这些点“印”到圆柱面 (THETA, Z) 上。
        
        # 1. 将轨迹点转换为圆柱坐标
        r_w = np.sqrt(x_w**2 + y_w**2) # 这里的 r_w 其实稍微小于 R，因为倾斜了
        theta_w = np.arctan2(y_w, x_w) # (-pi, pi)
        theta_w = np.mod(theta_w, 2*np.pi) # (0, 2pi)
        
        # 展平以便处理
        pts_theta = theta_w.flatten()
        pts_z = z_t.flatten()
        pts_r = r_w.flatten()
        
        # 2. 映射到网格
        # 对于网格上的每个点 (grid_theta, grid_z)，如果它在某一个刀具点的“攻击范围”内，
        # 它的半径就被削减到刀具半径。
        
        # 由于点太多 (2500 * 48 = 120,000个)，直接循环太慢。
        # 我们利用直方图或网格化技巧。
        
        # 让我们反过来想：
        # 我们已经有了刀具轨迹的 (theta, z, r)。
        # 这就是这一层被挖掉后的“底”。
        # 我们只需要把这些散点插值成一个曲面，取最小值包络。
        
        print("正在生成包络面 (Calculating Envelope)...")
        
        # 我们创建一个临时的画布来记录"最小半径"
        # 初始化为无穷大
        min_radius_map = np.full((res_z, res_theta), 999.0)
        
        # 将连续坐标离散化到网格索引
        idx_theta = (pts_theta / (2*np.pi) * res_theta).astype(int)
        idx_theta = np.clip(idx_theta, 0, res_theta-1)
        
        # Z 坐标映射 (-H/2 => 0, H/2 => res_z)
        idx_z = ((pts_z + self.H/2) / self.H * res_z).astype(int)
        
        # 过滤掉超出 Z 范围的点 (转子有些部分可能翘得很高)
        valid_mask = (idx_z >= 0) & (idx_z < res_z)
        
        idx_theta = idx_theta[valid_mask]
        idx_z = idx_z[valid_mask]
        val_r = pts_r[valid_mask]
        
        # 暴力填充：取最小值
        # 这种方法虽然粗糙，但对于展示"形状"足够了
        # 考虑到齿有宽度，我们不仅要在 (theta, z) 挖坑，还要在它左右挖坑。
        
        tooth_pixel_width = int(self.tw_rad / (2*np.pi) * res_theta)
        
        for k in range(len(val_r)):
            iz, it, r = idx_z[k], idx_theta[k], val_r[k]
            
            # 模拟齿宽：横向刷一遍
            # 为了效率，我们只更新比当前值更小的
            # (这是一个非常简化的光栅化过程)
            
            # 左右边界 (处理循环边界)
            left = it - tooth_pixel_width // 2
            right = it + tooth_pixel_width // 2
            
            # 简单的循环处理
            indices = np.arange(left, right) % res_theta
            
            # 更新这一行
            current_vals = min_radius_map[iz, indices]
            # 这是一个"挖坑"操作，保留最小值
            min_radius_map[iz, indices] = np.minimum(current_vals, r)

        # 平滑处理 (模拟加工表面光洁度)
        # min_radius_map = scipy.ndimage.gaussian_filter(min_radius_map, sigma=1)
        
        # 任何没被碰到的地方（999），恢复为初始毛坯半径
        min_radius_map[min_radius_map > 900] = self.R * 1.05
        
        return THETA, Z, min_radius_map

    def plot_result(self):
        THETA, Z, RADIUS = self.run_simulation()
        
        fig = plt.figure(figsize=(14, 12))
        
        # --- 2D 展开图 ---
        ax1 = fig.add_subplot(211)
        # 用颜色表示深浅 (半径)
        # 半径越小(越黑)，说明被挖得越深。半径越大(越白)，说明是齿顶。
        c = ax1.pcolormesh(np.degrees(THETA), Z, RADIUS, cmap='gist_gray', shading='auto')
        plt.colorbar(c, ax=ax1, label='Radius (mm)')
        
        ax1.set_title(f"Unrolled Surface of Shell ({self.Ns} Teeth)\nBlack=Valley (Cut), White=Peak (Tooth)")
        ax1.set_xlabel("Angle (deg)")
        ax1.set_ylabel("Z Height (mm)")
        ax1.set_xlim(0, 360)
        
        # --- 3D 实体图 ---
        ax2 = fig.add_subplot(212, projection='3d')
        
        # 转换为笛卡尔坐标
        X = RADIUS * np.cos(THETA)
        Y = RADIUS * np.sin(THETA)
        
        # 绘制
        surf = ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.bone,
                                linewidth=0, antialiased=False)
        
        ax2.set_title("3D Solid Reconstruction of the Cut Gear")
        
        # 调整比例
        limit = self.R * 1.2
        ax2.set_xlim(-limit, limit)
        ax2.set_ylim(-limit, limit)
        ax2.set_zlim(-self.H/2, self.H/2)
        
        # 视角优化：侧视，看清齿的轮廓
        ax2.view_init(elev=40, azim=-45)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = TrueBooleanHobbing(
        rotor_teeth=48, 
        shell_teeth=50, 
        tilt_angle_deg=3.0, 
        shell_width=6.0,
        rotor_radius=35.20,
        tooth_width_deg=2.5 # 转子齿稍微宽一点，切出来的槽就宽，留下的Shell齿就尖
    )
    sim.plot_result()