import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


class NutatingMotorSim:
    def __init__(
        self,
        num_magnets=12,
        radius=6.46,
        mag_d=3.0,
        mag_h=0.5,
        rotor_gap=1.28,
        cone_tilt=10.0,
        nutation_tilt=10.0,
        resolution=24,
    ):
        """
        Visualizer for Beveled Rotor + Flat Stator

        Key Physics:
        Rotor Cone Angle (10 deg) - Nutation Angle (10 deg) = 0 deg (Parallel to Flat Stator)
        """
        self.n = num_magnets
        self.r = radius
        self.d = mag_d
        self.h = mag_h
        self.gap = rotor_gap
        self.cone_deg = cone_tilt  # Rotor Bevel Angle
        self.nut_deg = nutation_tilt  # Nutation Tilt Angle
        self.res = resolution

        self.magnets_data = []
        self.magnets_physics = []  # Store physics data (position, normal)
        self._calculate_rotor_geometry()

    # --- Matrix Helpers ---
    def _rot_x(self, deg):
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rot_y(self, deg):
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rot_z(self, deg):
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _apply_transform(self, mesh, matrix, translation):
        shape = mesh.shape
        flat = mesh.reshape(3, -1)
        transformed = matrix @ flat
        transformed = transformed + translation[:, np.newaxis]
        return transformed.reshape(shape)
    
    def _apply_transform_point(self, point, matrix, translation):
        """Applies transform to a single 3D point."""
        p = np.array(point)
        return matrix @ p + translation

    def _apply_transform_vector(self, vector, matrix):
        """Applies transform to a direction vector (rotation only)."""
        v = np.array(vector)
        return matrix @ v

    # --- Magnet Mesh Generation ---
    def _create_solid_cylinder_meshes(self, r, h, res):
        theta = np.linspace(0, 2 * np.pi, res)
        z = np.linspace(-h / 2, h / 2, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_side = r * np.cos(theta_grid)
        y_side = r * np.sin(theta_grid)
        mesh_side = np.array([x_side, y_side, z_grid])

        r_steps = np.linspace(0, r, 2)
        theta_grid_cap, r_grid_cap = np.meshgrid(theta, r_steps)
        x_cap = r_grid_cap * np.cos(theta_grid_cap)
        y_cap = r_grid_cap * np.sin(theta_grid_cap)

        z_top = np.full_like(x_cap, h / 2)
        mesh_top = np.array([x_cap, y_cap, z_top])
        z_bot = np.full_like(x_cap, -h / 2)
        mesh_bot = np.array([x_cap, y_cap, z_bot])

        return {"side": mesh_side, "top": mesh_top, "bottom": mesh_bot}

    def _transform_solid_part(self, solid_dict, matrix, translation):
        transformed_dict = {}
        for key, mesh in solid_dict.items():
            transformed_dict[key] = self._apply_transform(mesh, matrix, translation)
        return transformed_dict

    def _calculate_rotor_geometry(self):
        base_solid = self._create_solid_cylinder_meshes(self.d / 2, self.h, self.res)

        # Global Nutation Matrix (Entire rotor tilts 10 deg)
        R_nutation = self._rot_x(self.nut_deg)

        for i in range(self.n):
            phi = i * (360 / self.n)
            is_north = i % 2 == 0
            color = "#DC143C" if is_north else "#4169E1"

            R_pos = self._rot_z(phi)

            # --- Top Magnet (Beveled) ---
            # Local Tilt = +10 deg (Cone shape)
            R_local_top = self._rot_y(self.cone_deg)
            trans_local_top = np.array([self.r, 0, self.gap / 2 + self.h / 2])

            # Mesh Transformation
            solid_top = self._transform_solid_part(
                base_solid, R_local_top, trans_local_top
            )
            solid_top = self._transform_solid_part(solid_top, R_pos, np.zeros(3))
            solid_top_final = self._transform_solid_part(
                solid_top, R_nutation, np.zeros(3)
            )
            
            # Physics Calculation (Track Center of Outer Face)
            # Top magnet outer face is at local z = +h/2
            p_top_center = np.array([0, 0, self.h / 2])
            # Apply same transforms
            p_top = self._apply_transform_point(p_top_center, R_local_top, trans_local_top)
            p_top = self._apply_transform_point(p_top, R_pos, np.zeros(3))
            p_top = self._apply_transform_point(p_top, R_nutation, np.zeros(3))
            
            # Normal vector (pointing out of the face)
            n_top = np.array([0, 0, 1])
            n_top = self._apply_transform_vector(n_top, R_local_top)
            n_top = self._apply_transform_vector(n_top, R_pos)
            n_top = self._apply_transform_vector(n_top, R_nutation)


            # --- Bottom Magnet (Beveled) ---
            # Local Tilt = -10 deg
            R_local_bot = self._rot_y(-self.cone_deg)
            trans_local_bot = np.array([self.r, 0, -(self.gap / 2 + self.h / 2)])

            # Mesh Transformation
            solid_bot = self._transform_solid_part(
                base_solid, R_local_bot, trans_local_bot
            )
            solid_bot = self._transform_solid_part(solid_bot, R_pos, np.zeros(3))
            solid_bot_final = self._transform_solid_part(
                solid_bot, R_nutation, np.zeros(3)
            )
            
            # Physics Calculation (Track Center of Outer Face)
            # Bottom magnet outer face is at local z = -h/2
            p_bot_center = np.array([0, 0, -self.h / 2])
            p_bot = self._apply_transform_point(p_bot_center, R_local_bot, trans_local_bot)
            p_bot = self._apply_transform_point(p_bot, R_pos, np.zeros(3))
            p_bot = self._apply_transform_point(p_bot, R_nutation, np.zeros(3))

            # Normal vector (pointing out of the face, i.e., down)
            n_bot = np.array([0, 0, -1])
            n_bot = self._apply_transform_vector(n_bot, R_local_bot)
            n_bot = self._apply_transform_vector(n_bot, R_pos)
            n_bot = self._apply_transform_vector(n_bot, R_nutation)

            self.magnets_data.append({"parts": solid_top_final, "color": color})
            self.magnets_data.append({"parts": solid_bot_final, "color": color})
            
            # Store physics data for the group
            self.magnets_physics.append({
                "group_id": i,
                "top": {"center": p_top, "normal": n_top},
                "bottom": {"center": p_bot, "normal": n_bot}
            })

    # --- 2. Create Flat Stator Planes ---
    def _create_flat_stator(self, side="top"):
        """
        Generates a FLAT plane (Standard PCB) that perfectly touches the magnet gap.
        """
        r_inner = self.r - self.d - 2
        r_outer = self.r + self.d + 2

        res_theta = 40
        res_r = 5

        theta = np.linspace(0, 2 * np.pi, res_theta)
        r = np.linspace(r_inner, r_outer, res_r)
        theta_grid, r_grid = np.meshgrid(theta, r)

        x = r_grid * np.cos(theta_grid)
        y = r_grid * np.sin(theta_grid)

        # --- 修正Z轴位置 ---
        # 1. 找到转子最低点磁铁的表面 Z 坐标
        # 转子中心在 Z=0
        # 磁铁中心在 +/- (Gap/2 + H/2)
        # 磁铁表面在 +/- (Gap/2 + H)
        # 再算上章动引起的垂直位移: R * sin(10 deg)

        # 简化计算：
        # 我们知道在接触点，磁铁表面到中心的垂直距离是固定的。
        # 我们希望气隙 Gap_Close_mm = 0.1

        # 假设转子不动时，磁铁表面 Z = +/- (self.gap/2 + self.h)
        # 当章动发生时，接触点的磁铁表面 Z 会下降 R * sin(nutation)

        # 但我们用反向逻辑：
        # 我们已经画出了转子。我们只需要把平面放到离转子最近点 0.1mm 的地方。
        # 通过实验或几何推导，接触点磁铁表面的 Z 大约在：
        magnet_surface_z = (
            (self.gap / 2) + self.h + (self.r * np.sin(np.deg2rad(self.nut_deg)))
        )

        # 定子应该在磁铁表面再往外 0.1mm
        stator_z = magnet_surface_z + 0.1

        if side == "top":
            z = np.full_like(x, stator_z)
        else:
            z = np.full_like(x, -stator_z)

        return x, y, z

    def calculate_static_forces(self):
        """
        Calculates the static magnetic pull force for each magnet group (Top + Bottom)
        against the Flat Stator (Iron Core).
        Returns a list of results.
        """
        print("\n--- Static Magnetic Force Simulation (N52 NdFeB) ---")
        
        # Constants for N52
        Br = 1.48  # Tesla
        mu0 = 4 * np.pi * 1e-7
        
        # Magnet Dimensions
        R = (self.d / 2) * 1e-3  # meters
        H = self.h * 1e-3        # meters
        Area = np.pi * R**2
        
        # Calculate Surface Field B_surf at center of magnet face
        B_surf = (Br / 2) * (H / np.sqrt(R**2 + H**2))
        
        # Max Force (at zero gap)
        F_max = (B_surf**2 * Area) / (2 * mu0)
        
        # Calculate Stator Z Position
        magnet_lowest_z_dist = (self.gap / 2) + self.h + (self.r * np.sin(np.deg2rad(self.nut_deg)))
        stator_z_pos = magnet_lowest_z_dist + 0.1  # mm
        
        results = []
        
        print("\nGroup Forces (Top + Bottom Pair) - STATIC:")
        print(f"{'Group':<6} | {'Gap Top':<10} | {'F_Top(N)':<10} | {'Gap Bot':<10} | {'F_Bot(N)':<10} | {'Net Z(N)':<10}")
        print("-" * 70)
        
        for group in self.magnets_physics:
            gid = group["group_id"]
            
            # --- Top Magnet ---
            p_top = group["top"]["center"]
            gap_top = stator_z_pos - p_top[2]
            g_m = gap_top * 1e-3
            f_top = F_max / (1 + g_m / R)**2
            
            # --- Bottom Magnet ---
            p_bot = group["bottom"]["center"]
            gap_bot = p_bot[2] - (-stator_z_pos)
            g_m_bot = gap_bot * 1e-3
            f_bot = F_max / (1 + g_m_bot / R)**2
            
            # Net Force on Rotor (Z direction) - Static: Both Pull
            # Top Pulls UP (+), Bottom Pulls DOWN (-)
            net_force = f_top - f_bot
            
            results.append({
                "group_id": gid,
                "gap_top": gap_top,
                "force_top": f_top,
                "gap_bot": gap_bot,
                "force_bot": f_bot,
                "net_force": net_force
            })

            print(f"{gid:<6} | {gap_top:<10.4f} | {f_top:<10.4f} | {gap_bot:<10.4f} | {f_bot:<10.4f} | {net_force:<10.4f}")

        return results

    def calculate_active_forces(self):
        """
        Calculates the ACTIVE magnetic force (Coils Energized).
        Logic: 
        - We assume the coils are commutated sinusoidally to match the nutation.
        - At the 'Close' side (Peak), we Boost the Pull.
        - At the 'Far' side (Trough), we Reverse the Pull (Push).
        - At the Nodes (90 deg), the coil effect is zero.
        
        Model:
        F_active = F_static * (1 + Gain * Sin(phi))
        """
        print("\n--- Active Magnetic Force Simulation (Energized) ---")
        
        # Constants (Same as static)
        Br = 1.48
        mu0 = 4 * np.pi * 1e-7
        R = (self.d / 2) * 1e-3
        H = self.h * 1e-3
        Area = np.pi * R**2
        B_surf = (Br / 2) * (H / np.sqrt(R**2 + H**2))
        F_max = (B_surf**2 * Area) / (2 * mu0)
        
        magnet_lowest_z_dist = (self.gap / 2) + self.h + (self.r * np.sin(np.deg2rad(self.nut_deg)))
        stator_z_pos = magnet_lowest_z_dist + 0.1
        
        results = []
        
        # Gain factor for Active Coils. 
        # Gain > 1.0 ensures that the 'Far' side becomes a Push (Negative Force).
        ACTIVE_GAIN = 2.0 
        
        print("\nGroup Forces (Top + Bottom Pair) - ACTIVE:")
        print(f"{'Group':<6} | {'Gap Top':<10} | {'F_Top(N)':<10} | {'Gap Bot':<10} | {'F_Bot(N)':<10} | {'Net Z(N)':<10}")
        print("-" * 70)
        
        for group in self.magnets_physics:
            gid = group["group_id"]
            
            # Calculate Phase Angle for this group
            # Group 0 is at 0 deg. Group 3 is at 90 deg (Peak).
            # We want Peak Effect at 90 deg.
            phi = gid * (360 / self.n)
            # Sin(phi) is 1 at 90, -1 at 270.
            factor = np.sin(np.deg2rad(phi))
            
            # --- Top Magnet ---
            p_top = group["top"]["center"]
            gap_top = stator_z_pos - p_top[2]
            g_m = gap_top * 1e-3
            f_top_static = F_max / (1 + g_m / R)**2
            
            # Active Modulation:
            # At 90 deg (Top Close), factor=1. We want Boost Pull. -> (1 + G*1)
            # At 270 deg (Top Far), factor=-1. We want Push (or reduced Pull). -> (1 + G*-1)
            f_top_active = f_top_static * (1 + ACTIVE_GAIN * factor)
            
            # --- Bottom Magnet ---
            p_bot = group["bottom"]["center"]
            gap_bot = p_bot[2] - (-stator_z_pos)
            g_m_bot = gap_bot * 1e-3
            f_bot_static = F_max / (1 + g_m_bot / R)**2
            
            # Active Modulation:
            # At 90 deg (Bot Far), factor=1. We want Push (Negative Force).
            # Static is Pull (+). We want Result < 0.
            # So we need (1 - G*factor). If G=2, factor=1 -> (1-2) = -1. Push!
            # At 270 deg (Bot Close), factor=-1. We want Boost Pull.
            # (1 - G*-1) = (1 + G). Boost!
            f_bot_active = f_bot_static * (1 - ACTIVE_GAIN * factor)
            
            # Net Force on Rotor (Z direction)
            # Top pulls UP (+), Bottom pulls DOWN (-).
            # If f_bot_active is negative (Push), it means it Pushes UP (+).
            # So we subtract the bottom force vector (which points down).
            # Net = F_top_vec + F_bot_vec
            # F_top_vec = f_top_active * (+Z)
            # F_bot_vec = f_bot_active * (-Z)
            # Net Z = f_top_active - f_bot_active
            
            net_force = f_top_active - f_bot_active
            
            results.append({
                "group_id": gid,
                "gap_top": gap_top,
                "force_top": f_top_active,
                "gap_bot": gap_bot,
                "force_bot": f_bot_active,
                "net_force": net_force
            })

            print(f"{gid:<6} | {gap_top:<10.4f} | {f_top_active:<10.4f} | {gap_bot:<10.4f} | {f_bot_active:<10.4f} | {net_force:<10.4f}")

        return results

    def visualize(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Render Rotor Magnets
        draw_stride = max(1, self.res // 10)
        print("Rendering Nutating Beveled Rotor...")
        for m in self.magnets_data:
            parts = m["parts"]
            c = m["color"]
            kw = dict(
                color=c,
                alpha=1.0,
                shade=True,
                linewidth=0,
                rstride=draw_stride,
                cstride=draw_stride,
            )
            ax.plot_surface(parts["side"][0], parts["side"][1], parts["side"][2], **kw)
            ax.plot_surface(parts["top"][0], parts["top"][1], parts["top"][2], **kw)
            ax.plot_surface(
                parts["bottom"][0], parts["bottom"][1], parts["bottom"][2], **kw
            )

        # 2. Render FLAT Stator PCBs
        print("Rendering Flat PCB Planes...")

        # Top PCB (Flat)
        x_st, y_st, z_st = self._create_flat_stator("top")
        ax.plot_wireframe(x_st, y_st, z_st, color="#00AA00", alpha=0.3, linewidth=0.5)
        ax.plot_surface(x_st, y_st, z_st, color="green", alpha=0.1, shade=False)

        # Bottom PCB (Flat)
        x_sb, y_sb, z_sb = self._create_flat_stator("bottom")
        ax.plot_wireframe(x_sb, y_sb, z_sb, color="#00AA00", alpha=0.3, linewidth=0.5)
        ax.plot_surface(x_sb, y_sb, z_sb, color="green", alpha=0.1, shade=False)

        # 3. Visualize Forces
        print("Visualizing Forces...")
        max_f = 0
        for group in self.magnets_physics:
            if "force_top" in group:
                # Top Force (Red Arrow Up)
                p_top = group["top"]["center"]
                f_top = group["force_top"]
                # Scale factor for visibility if needed, currently 1 unit = 1 Newton
                scale = 1.0 
                ax.quiver(p_top[0], p_top[1], p_top[2], 0, 0, f_top * scale, color="red", linewidth=1.5, arrow_length_ratio=0.3)
                
                # Bottom Force (Blue Arrow Down)
                p_bot = group["bottom"]["center"]
                f_bot = group["force_bot"]
                # Force is attractive towards bottom stator (-Z direction)
                ax.quiver(p_bot[0], p_bot[1], p_bot[2], 0, 0, -f_bot * scale, color="blue", linewidth=1.5, arrow_length_ratio=0.3)
                
                max_f = max(max_f, f_top, f_bot)

        # 4. View Settings
        limit = self.r + 5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-5, 5)
        ax.set_box_aspect([1, 1, 0.5])

        ax.set_title(
            f"Correct Geometry: Beveled Rotor + FLAT Stators\n(Parallel Gap at Contact Line)"
        )
        ax.set_xlabel("X (mm)")

        # Legend
        handles = [
            mpatches.Patch(color="#DC143C", label="Rotor Magnet N"),
            mpatches.Patch(color="#4169E1", label="Rotor Magnet S"),
            mpatches.Patch(color="#00FF00", alpha=0.3, label="Flat PCB Stator"),
            mpatches.Patch(color="red", label="Top Attraction Force"),
            mpatches.Patch(color="blue", label="Bottom Attraction Force"),
        ]
        ax.legend(handles=handles, loc="upper right")

        # Set view to see the parallel gap clearly
        ax.view_init(elev=0, azim=0)  # Side view

        plt.tight_layout()
        # plt.show()  # Don't block here, show all at end

    def plot_radar_forces(self, static_results, active_results):
        """
        Plots a radar chart (polar plot) comparing Static vs Active magnetic forces.
        """
        if not static_results or not active_results:
            print("No force data to plot.")
            return

        # Data Preparation
        groups = [g["group_id"] for g in static_results]
        
        # Extract Net Forces (Use Absolute Value for Radar Plot to avoid negative radius flipping)
        net_static = [abs(g["net_force"]) for g in static_results]
        net_active = [abs(g["net_force"]) for g in active_results]
        
        # Close the loop for radar chart
        groups_plot = groups + [groups[0]]
        net_static_plot = net_static + [net_static[0]]
        net_active_plot = net_active + [net_active[0]]

        # Angles
        theta = np.linspace(0, 2 * np.pi, len(groups_plot), endpoint=True)

        # Plotting
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="polar")

        # Plot Static Force
        ax.plot(theta, net_static_plot, "o-", color="blue", linewidth=2, label="Static |Net Force|")
        ax.fill(theta, net_static_plot, color="blue", alpha=0.1)

        # Plot Active Force
        ax.plot(theta, net_active_plot, "o-", color="red", linewidth=2, label="Active |Net Force|")
        ax.fill(theta, net_active_plot, color="red", alpha=0.1)

        ax.set_title("Magnitude of Net Z Force per Group (Static vs Active)\nNote: 90°=Pull Up, 270°=Pull Down", va='bottom', y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        
        # Set angular ticks
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels([f"G{g}" for g in groups])
        
        plt.tight_layout()


# === Run ===
# Note: Cone Tilt and Nutation Tilt MUST be equal for this to work on flat stators!
sim = NutatingMotorSim(num_magnets=12, radius=6.5, cone_tilt=10, nutation_tilt=10)
static_res = sim.calculate_static_forces()
active_res = sim.calculate_active_forces()
sim.visualize()
sim.plot_radar_forces(static_res, active_res)
plt.show()
