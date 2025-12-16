import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches


class MagArray:
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
        Initialize MagArray (Nutating Magnet Array)

        :param num_magnets: Number of magnets (default 12)
        :param radius: Distribution radius (mm)
        :param mag_d: Magnet diameter (mm)
        :param mag_h: Magnet thickness (mm)
        :param rotor_gap: Rotor edge thickness/gap (mm)
        :param cone_tilt: Cone half-angle (degrees), corresponds to +/- 10 degrees
        :param nutation_tilt: Global nutation tilt angle (degrees)
        :param resolution: Mesh segmentation count (Default 24 for balance)
        """
        self.n = num_magnets
        self.r = radius
        self.d = mag_d
        self.h = mag_h
        self.gap = rotor_gap
        self.cone_deg = cone_tilt
        self.nut_deg = nutation_tilt
        self.res = resolution

        # Storage for calculated data
        # Each item is a dict containing 'parts' (side, top, bottom meshes)
        self.magnets_data = []
        self.planes_data = []

        self._calculate_geometry()

    def _rot_x(self, deg):
        """Rotation matrix around X-axis"""
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rot_y(self, deg):
        """Rotation matrix around Y-axis"""
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _rot_z(self, deg):
        """Rotation matrix around Z-axis"""
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _create_solid_cylinder_meshes(self, r, h, res):
        """
        Generate vertex data for a SOLID cylinder (Side + Caps).
        Returns a dictionary of meshes.
        """
        # 1. Side Mesh
        theta = np.linspace(0, 2 * np.pi, res)
        z = np.linspace(-h / 2, h / 2, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_side = r * np.cos(theta_grid)
        y_side = r * np.sin(theta_grid)
        mesh_side = np.array([x_side, y_side, z_grid])

        # 2. Cap Meshes (Top & Bottom)
        # Optimization: Caps only need 2 radial steps to look solid
        r_steps = np.linspace(0, r, 2)
        theta_grid_cap, r_grid_cap = np.meshgrid(theta, r_steps)
        x_cap = r_grid_cap * np.cos(theta_grid_cap)
        y_cap = r_grid_cap * np.sin(theta_grid_cap)

        # Top Cap Z
        z_top = np.full_like(x_cap, h / 2)
        mesh_top = np.array([x_cap, y_cap, z_top])

        # Bottom Cap Z
        z_bot = np.full_like(x_cap, -h / 2)
        mesh_bot = np.array([x_cap, y_cap, z_bot])

        return {"side": mesh_side, "top": mesh_top, "bottom": mesh_bot}

    def _apply_transform(self, mesh, matrix, translation):
        """Apply matrix transformation and translation to a mesh array"""
        shape = mesh.shape
        flat = mesh.reshape(3, -1)
        transformed = matrix @ flat
        transformed = transformed + translation[:, np.newaxis]
        return transformed.reshape(shape)

    def _transform_solid_part(self, solid_dict, matrix, translation):
        """Helper to transform all parts (side, top, bottom) of a solid cylinder dictionary"""
        transformed_dict = {}
        for key, mesh in solid_dict.items():
            transformed_dict[key] = self._apply_transform(mesh, matrix, translation)
        return transformed_dict

    def _calculate_geometry(self):
        """Core Geometry Calculation Engine"""

        # 1. Generate Base Meshes (Dictionary with side, top, bottom)
        base_solid = self._create_solid_cylinder_meshes(self.d / 2, self.h, self.res)

        # Global Nutation Matrix (Entire rotor tilt)
        R_nutation = self._rot_x(self.nut_deg)

        # === A. Calculate Magnets ===
        for i in range(self.n):
            phi = i * (360 / self.n)
            is_north = i % 2 == 0
            polarity = "N" if is_north else "S"
            color = "#DC143C" if is_north else "#4169E1"  # Crimson Red / Royal Blue

            # Rotation matrix for azimuthal position
            R_pos = self._rot_z(phi)

            # --- Top Magnet ---
            # 1. Local Tilt (+10 deg around Y - Cone Shape)
            R_local_top = self._rot_y(self.cone_deg)
            # 2. Translation (Move to radius + Z offset)
            trans_local_top = np.array([self.r, 0, self.gap / 2 + self.h / 2])

            # Apply: Local Tilt -> Translate -> Rotate Z (Position) -> Global Nutation
            # Step 1: Local Tilt & Translate
            solid_top = self._transform_solid_part(
                base_solid, R_local_top, trans_local_top
            )
            # Step 2: Rotate around Z (Position in ring)
            solid_top = self._transform_solid_part(solid_top, R_pos, np.zeros(3))
            # Step 3: Global Nutation
            solid_top_final = self._transform_solid_part(
                solid_top, R_nutation, np.zeros(3)
            )

            # --- Bottom Magnet ---
            # 1. Local Reverse Tilt (-10 deg around Y)
            R_local_bot = self._rot_y(-self.cone_deg)
            # 2. Translation (Downwards)
            trans_local_bot = np.array([self.r, 0, -(self.gap / 2 + self.h / 2)])

            # Step 1: Local Tilt & Translate
            solid_bot = self._transform_solid_part(
                base_solid, R_local_bot, trans_local_bot
            )
            # Step 2: Rotate around Z
            solid_bot = self._transform_solid_part(solid_bot, R_pos, np.zeros(3))
            # Step 3: Global Nutation
            solid_bot_final = self._transform_solid_part(
                solid_bot, R_nutation, np.zeros(3)
            )

            self.magnets_data.append(
                {
                    "index": i,
                    "side": "top",
                    "polarity": polarity,
                    "color": color,
                    "parts": solid_top_final,
                }
            )
            self.magnets_data.append(
                {
                    "index": i,
                    "side": "bottom",
                    "polarity": polarity,
                    "color": color,
                    "parts": solid_bot_final,
                }
            )

        # === B. Calculate Contact Planes ===
        # Use higher resolution for the ring to look smooth
        res_plane = max(40, self.res * 2)  # Ensure ring is smooth
        theta = np.linspace(0, 2 * np.pi, res_plane)
        r_range = [self.r - self.d, self.r + self.d]  # Slightly larger than magnets

        for side, tilt_sign, z_offset in [("top", 1, 1), ("bottom", -1, -1)]:
            X, Y, Z = [], [], []
            for r in r_range:
                row_x, row_y, row_z = [], [], []
                for t in theta:
                    # Construct points with tilt logic
                    # Tilt normal vector locally
                    R_tilt = self._rot_y(self.cone_deg * tilt_sign)
                    p_tilt = R_tilt @ np.array([r, 0, 0])
                    p_tilt[2] += (self.gap / 2) * z_offset

                    # Rotate to azimuthal position
                    R_azi = self._rot_z(np.rad2deg(t))
                    p_final = R_azi @ p_tilt

                    row_x.append(p_final[0])
                    row_y.append(p_final[1])
                    row_z.append(p_final[2])
                X.append(row_x)
                Y.append(row_y)
                Z.append(row_z)

            mesh_plane = np.array([X, Y, Z])
            # Apply Global Nutation to plane
            mesh_plane_final = self._apply_transform(
                mesh_plane, R_nutation, np.zeros(3)
            )

            self.planes_data.append(
                {
                    "mesh": mesh_plane_final,
                    "color": "cyan" if side == "top" else "magenta",
                }
            )

    def visualize(self):
        """Render the 3D Scene (Optimized for Performance)"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # === Performance Optimization ===
        # 'stride' determines how many grid points are skipped during drawing.
        # Dynamically calculated based on resolution to ensure UI responsiveness.
        draw_stride = max(1, self.res // 10)

        print(f"Rendering {len(self.magnets_data)} magnets...")
        print(f"Optimization: Mesh Resolution={self.res}, Draw Stride={draw_stride}")

        # 1. Render Solid Magnets
        for m in self.magnets_data:
            parts = m["parts"]
            c = m["color"]

            # rstride/cstride: Downsampling for rendering
            # shade=True: Adds lighting (slower but 3D effect)
            # linewidth=0: Removes mesh wireframe lines (faster, cleaner look)
            kw = dict(
                color=c,
                alpha=1.0,
                shade=True,
                linewidth=0,
                antialiased=False,
                rstride=draw_stride,
                cstride=draw_stride,
            )

            ax.plot_surface(parts["side"][0], parts["side"][1], parts["side"][2], **kw)
            ax.plot_surface(parts["top"][0], parts["top"][1], parts["top"][2], **kw)
            ax.plot_surface(
                parts["bottom"][0], parts["bottom"][1], parts["bottom"][2], **kw
            )

        # 2. Render Contact Planes (FIXED: Uncommented and set stride)
        # Planes are simpler, so we use a fixed small stride for smoothness
        print(f"Rendering {len(self.planes_data)} contact planes...")
        for p in self.planes_data:
            mesh = p["mesh"]
            # Using rstride=1, cstride=2 for a good balance of speed and look for rings
            ax.plot_surface(
                mesh[0],
                mesh[1],
                mesh[2],
                color=p["color"],
                alpha=0.2,
                rstride=1,
                cstride=2,
                linewidth=0,
                shade=False,
            )

        # 3. View Settings
        limit = self.r + 5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-5, 5)
        # Flatten Z-axis visually to match physical aspect
        ax.set_box_aspect([1, 1, 0.4])

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(
            f"Optimized Nutating Rotor & Stator Planes\nResolution: {self.res} (Display Stride: {draw_stride})"
        )

        # Legend
        handles = [
            mpatches.Patch(color="#DC143C", label="North (N)"),
            mpatches.Patch(color="#4169E1", label="South (S)"),
            mpatches.Patch(color="cyan", alpha=0.3, label="Stator Top Plane"),
            mpatches.Patch(color="magenta", alpha=0.3, label="Stator Bottom Plane"),
        ]
        ax.legend(handles=handles, loc="upper right")

        plt.tight_layout()
        plt.show()


# ================= Execute =================

# Instantiate with high resolution
# Visualize method handles optimization automatically.
rotor = MagArray(
    num_magnets=12, radius=6.46, cone_tilt=10, nutation_tilt=10, resolution=24
)

rotor.visualize()
