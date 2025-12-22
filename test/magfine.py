import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MicroNutatingSimV2:
    def __init__(self):
        # --- 用户指定的核心参数 ---
        self.num_magnets = 12
        self.radius = 6.46      # 分布半径 (mm)
        self.mag_dia = 3.0      # 磁铁直径 (mm)
        self.mag_h = 0.5        # 磁铁厚度 (mm)
        self.tilt = 10.0        # 章动/锥角 (deg)
        
        # --- 线圈与定子参数 ---
        self.clearance = 0.2    # 最小机械气隙 (mm)
        self.coil_h = 1.6       # 线圈厚度 (mm)
        
        # 物理常数 (N52钕铁硼)
        self.Br = 1.48          # Tesla
        self.mu0 = 4 * np.pi * 1e-7
        
        # 计算单个磁铁的磁矩 (Magnetic Moment)
        # 体积 V = pi * r^2 * h (转换为米)
        volume = np.pi * ((self.mag_dia/2)*1e-3)**2 * (self.mag_h*1e-3)
        self.m_mag = (1/self.mu0) * self.Br * volume
        
        # 存储磁铁数据
        self.magnets = []
        self._build_rotor()

    def _rot_x(self, deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rot_y(self, deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rot_z(self, deg):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _build_rotor(self):
        """构建章动转子几何"""
        R_nut = self._rot_x(self.tilt)
        
        # 为了提高近场计算精度，使用 7-dipole 近似
        sub_offsets = []
        sub_offsets.append(np.array([0,0,0]))
        r_sub = self.mag_dia / 4.0 
        for k in range(6):
            th = k * 60 * (np.pi/180)
            sub_offsets.append(np.array([r_sub*np.cos(th), r_sub*np.sin(th), 0]))
        
        m_sub_mag = self.m_mag / 7.0

        for i in range(self.num_magnets):
            phi = i * (360 / self.num_magnets)
            pol = 1 if i % 2 == 0 else -1
            
            R_bevel = self._rot_y(self.tilt)
            p_local_center = np.array([self.radius, 0, 0])
            m_local_vec = np.array([0, 0, pol])
            
            R_z_pos = self._rot_z(phi)
            R_total = R_nut @ R_z_pos @ R_bevel
            
            pos_center = R_total @ p_local_center
            m_vec_final = (R_total @ m_local_vec) * m_sub_mag
            
            current_mag_subs = []
            for off in sub_offsets:
                off_rot = R_total @ off
                p_sub = pos_center + off_rot
                current_mag_subs.append({"pos": p_sub, "m": m_vec_final})
            
            self.magnets.append({
                "id": i,
                "center": pos_center,
                "subs": current_mag_subs,
                "polarity": pol
            })

    def get_stator_geometry(self):
        """计算定子平面高度"""
        lowest_center_z = -self.radius * np.sin(np.deg2rad(self.tilt))
        magnet_surface_z_at_lowest = lowest_center_z - (self.mag_h / 2)
        stator_surface_z = magnet_surface_z_at_lowest - self.clearance
        return abs(stator_surface_z)

    def calculate_b_field_at_point(self, point_mm, use_back_iron=True):
        """计算空间某点的磁感应强度 B (Vector)"""
        B_total = np.zeros(3)
        stator_z = self.get_stator_geometry()
        
        # 背铁平面位置 (mm)
        iron_plane_z_top = stator_z + self.coil_h
        iron_plane_z_bot = -stator_z - self.coil_h
        
        for mag in self.magnets:
            for sub in mag["subs"]:
                # 1. 真实磁源
                B_total += self._dipole_formula(sub["pos"], sub["m"], point_mm)
                
                if use_back_iron:
                    # 2. 顶部背铁镜像
                    p_src = sub["pos"]
                    p_img_top = np.array([p_src[0], p_src[1], 2*iron_plane_z_top - p_src[2]])
                    
                    # 镜像磁矩 (垂直分量不变，水平分量反向)
                    m_src = sub["m"]
                    m_img = np.array([-m_src[0], -m_src[1], m_src[2]])
                    
                    B_total += self._dipole_formula(p_img_top, m_img, point_mm)
                    
                    # 3. 底部背铁镜像
                    p_img_bot = np.array([p_src[0], p_src[1], 2*iron_plane_z_bot - p_src[2]])
                    B_total += self._dipole_formula(p_img_bot, m_img, point_mm)
        
        return B_total

    def _dipole_formula(self, pos_src_mm, m_src_SI, pos_eval_mm):
        """
        偶极子磁场公式 (修正版)
        输入: 坐标(mm), 磁矩(A·m²)
        输出: 磁场(Tesla)
        """
        # [关键修正] 将毫米坐标转换为米
        r_vec_meters = (pos_eval_mm - pos_src_mm) * 1e-3
        
        r_sq = np.dot(r_vec_meters, r_vec_meters)
        r_mag = np.sqrt(r_sq)
        
        if r_mag < 1e-6: return np.zeros(3)
        
        dot = np.dot(m_src_SI, r_vec_meters)
        
        # B = (mu0 / 4pi) * ...
        # 1e-7 是 mu0/4pi
        return 1e-7 * (3 * dot * r_vec_meters - m_src_SI * r_sq) / (r_mag**5)

    def analyze_forces(self):
        stator_surface_z = self.get_stator_geometry()
        # 评估点：线圈中心
        eval_z = stator_surface_z + (self.coil_h / 2.0)
        
        print(f"--- 仿真参数校正 ---")
        print(f"  磁铁尺寸: {self.mag_dia}mm x {self.mag_h}mm")
        print(f"  线圈中心平面: +/- {eval_z:.3f} mm")
        print(f"  背铁距离磁铁: > {self.clearance + self.coil_h:.3f} mm")
        
        results_no_iron = []
        results_iron = []
        
        for mag in self.magnets:
            center = mag["center"]
            # 投影到定子平面上线圈中心位置
            eval_pos = np.array([center[0], center[1], eval_z])
            
            # Case 1
            B1 = self.calculate_b_field_at_point(eval_pos, use_back_iron=False)
            results_no_iron.append({
                "id": mag["id"],
                "Bz": abs(B1[2]),
                "Br": np.linalg.norm(B1[:2]),
                "B_vec": B1
            })
            
            # Case 2
            B2 = self.calculate_b_field_at_point(eval_pos, use_back_iron=True)
            results_iron.append({
                "id": mag["id"],
                "Bz": abs(B2[2]),
                "Br": np.linalg.norm(B2[:2]),
                "B_vec": B2
            })
            
        return results_no_iron, results_iron, eval_z

    def plot_results(self, res_no, res_iron, eval_z):
        ids = [r["id"] for r in res_no]
        bz_no = [r["Bz"] for r in res_no]
        bz_iron = [r["Bz"] for r in res_iron]
        br_iron = [r["Br"] for r in res_iron]
        
        fig = plt.figure(figsize=(14, 6))
        
        # 图1: 背铁增益
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(ids, bz_no, 'o--', label="No Iron (Air Core)", color='gray')
        ax1.plot(ids, bz_iron, 'o-', label="With Back Iron", color='blue', linewidth=2)
        ax1.fill_between(ids, bz_no, bz_iron, color='blue', alpha=0.1)
        ax1.set_title("1. Back Iron Gain (Real T values)")
        ax1.set_ylabel("|Bz| Field Strength (Tesla)")
        ax1.set_xlabel("Magnet ID")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 计算最大值的增益
        max_no = max(bz_no)
        max_iron = max(bz_iron)
        print(f"Peak Field (No Iron): {max_no:.4f} T")
        print(f"Peak Field (With Iron): {max_iron:.4f} T")
        print(f"Gain: {((max_iron/max_no)-1)*100:.1f}%")

        # 图2: 力矢量分布
        ax2 = fig.add_subplot(1, 2, 2)
        x = np.arange(len(ids))
        width = 0.35
        ax2.bar(x - width/2, bz_iron, width, label='Axial (Thrust)', color='green')
        ax2.bar(x + width/2, br_iron, width, label='Lateral (Vibration)', color='red')
        ax2.set_title("2. Force Distribution (Tesla)")
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 运行修正版
sim = MicroNutatingSimV2()
res_no, res_iron, eval_z = sim.analyze_forces()
sim.plot_results(res_no, res_iron, eval_z)