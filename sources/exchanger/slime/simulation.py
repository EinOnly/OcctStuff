import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from numba import njit
import threading

from slime import SlimeAgent


# =========================
# Numba kernels
# =========================
@njit
def get_equilibrium_numba(rho, u, v, w, c, Q, ny, nx):
    feq = np.zeros((Q, ny, nx))
    usqr = u * u + v * v
    for i in range(Q):
        cu = c[i, 0] * u + c[i, 1] * v
        feq[i] = w[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usqr)
    return feq


@njit
def bgk_collision(f, feq, tau):
    return f - (f - feq) / tau


# =========================
# Thermal + Fluid LBM Solver
# =========================
class ThermalLBM:
    def __init__(self, nx=300, ny=150, inlet_size=30):
        self.nx, self.ny = nx, ny
        self.inlet_size = inlet_size

        # D2Q9
        self.c = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ]
        )
        self.w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Physical parameters
        self.tau_f = 0.6
        self.tau_g = 0.8  # 🔥 热扩散降低一半
        self.u_in = 0.2
        self.cs_sq = 1.0 / 3.0
        self.advection_strength = 3.0  # 🔥 对流强度系数，越大流体带走热量越多

        # Fields
        self.f = np.ones((9, ny, nx)) * (1 / 9)
        self.g = np.zeros((9, ny, nx))
        self.rho = np.ones((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.temp = np.zeros((ny, nx))

        # Smoothed velocity for thermal stability
        self.u_smooth = np.zeros((ny, nx))
        self.v_smooth = np.zeros((ny, nx))
        self.smooth_alpha = 0.2

        # Boundaries & threading
        self.setup_boundaries()
        # slime_mask now represents FLUID domain (True = slime/open fluid)
        self.slime_mask = ~self.base_boundary
        self.update_combined_boundary()

        self.solid_feq = get_equilibrium_numba(
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            self.w,
            self.c,
            9,
            1,
            1,
        )
        self.lock = threading.Lock()
        self.running = True

    # =========================
    # Boundaries
    # =========================
    def setup_boundaries(self):
        self.base_boundary = np.ones((self.ny, self.nx), dtype=bool)
        self.base_boundary[1:-1, 1:-1] = False

        self.inlet_mask = np.zeros_like(self.base_boundary)
        self.inlet_mask[1 : self.inlet_size, 0] = True
        self.base_boundary[1 : self.inlet_size, 0] = False

        self.outlet_mask = np.zeros_like(self.base_boundary)
        self.outlet_mask[self.ny - self.inlet_size : self.ny - 1, -1] = True
        self.base_boundary[self.ny - self.inlet_size : self.ny - 1, -1] = False

    def update_combined_boundary(self):
        slime_fluid = self.slime_mask | self.inlet_mask | self.outlet_mask
        self.boundary = self.base_boundary | (~slime_fluid)
        self.boundary[self.inlet_mask | self.outlet_mask] = False

    def apply_slime_mask(self, slime_mask):
        previous_fluid = self.slime_mask.copy()
        slime_fluid = slime_mask | self.inlet_mask | self.outlet_mask
        self.slime_mask = slime_fluid
        self.update_combined_boundary()

        # Cells that became solid: wipe fluid/heat
        to_solid = previous_fluid & (~slime_fluid)
        if np.any(to_solid):
            for i in range(9):
                self.f[i, to_solid] = self.solid_feq[i, 0, 0]
                self.g[i, to_solid] = 0.0

        # Cells that reopened to fluid: reset to near-rest state
        reopened = (~previous_fluid) & slime_fluid
        if np.any(reopened):
            feq_open = get_equilibrium_numba(
                np.array([[1.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                self.w,
                self.c,
                9,
                1,
                1,
            )
            for i in range(9):
                self.f[i, reopened] = feq_open[i, 0, 0]
                self.g[i, reopened] = 0.0

    def get_fluid_mask(self):
        return self.slime_mask & ~self.inlet_mask & ~self.outlet_mask

    def step(self, steps=1):
        for _ in range(steps):
            self.fluid_step()
            self.thermal_step()

    # =========================
    # Fluid step
    # =========================
    def fluid_step(self):
        if not np.isfinite(self.f).all():
            # Clamp numerical blow-ups to keep simulation running
            self.f = np.nan_to_num(self.f, nan=1 / 9, posinf=1 / 9, neginf=1e-9)
        self.f = np.clip(self.f, 1e-9, 5.0)

        # Macroscopic
        self.rho = np.sum(self.f, axis=0)
        self.rho[self.rho < 1e-12] = 1e-12

        u = np.sum(self.f * self.c[:, 0, None, None], axis=0) / self.rho
        v = np.sum(self.f * self.c[:, 1, None, None], axis=0) / self.rho

        with self.lock:
            self.u = u
            self.v = v
        self.u = np.clip(self.u, -0.8, 0.8)
        self.v = np.clip(self.v, -0.8, 0.8)

        # Collision
        feq = get_equilibrium_numba(self.rho, u, v, self.w, self.c, 9, self.ny, self.nx)
        self.f = bgk_collision(self.f, feq, self.tau_f)

        # Streaming
        for i in range(9):
            self.f[i] = np.roll(self.f[i], self.c[i, 0], axis=1)
            self.f[i] = np.roll(self.f[i], self.c[i, 1], axis=0)

        # Kill periodicity
        self.f[:, :, 0] = self.f[:, :, 1]
        self.f[:, :, -1] = self.f[:, :, -2]
        self.f[:, 0, :] = self.f[:, 1, :]
        self.f[:, -1, :] = self.f[:, -2, :]

        # Bounce-back walls
        for i in range(9):
            self.f[i, self.boundary] = self.f[self.opp[i], self.boundary]

        # Inlet (equilibrium injection)
        feq_in = get_equilibrium_numba(
            np.array([[1.03]]),
            np.array([[self.u_in]]),
            np.array([[0.0]]),
            self.w,
            self.c,
            9,
            1,
            1,
        )
        for i in range(9):
            self.f[i, self.inlet_mask] = feq_in[i, 0, 0]

        # Velocity smoothing for thermal solver
        with self.lock:
            self.u_smooth = (
                self.smooth_alpha * self.u + (1 - self.smooth_alpha) * self.u_smooth
            )
            self.v_smooth = (
                self.smooth_alpha * self.v + (1 - self.smooth_alpha) * self.v_smooth
            )

    # =========================
    # Thermal step
    # =========================
    def thermal_step(self):
        if not np.isfinite(self.g).all():
            self.g = np.nan_to_num(self.g, nan=0.0, posinf=0.0, neginf=0.0)
        self.g = np.clip(self.g, -2.0, 2.0)

        self.temp = np.sum(self.g, axis=0)

        with self.lock:
            u = self.u_smooth.copy()
            v = self.v_smooth.copy()

        # ====== 体积热源: 在整个流体域内均匀加热 ======
        # 创建内部流体区域的掩码
        fluid_domain = ~self.boundary & ~self.inlet_mask & ~self.outlet_mask

        # 在流体域内均匀添加热量 (体积热源)
        heat_rate = 0.0001  # 每步的加热功率
        self.temp[fluid_domain] += heat_rate

        # 增强对流效果：速度越大，温度平衡态偏移越多
        u_enhanced = np.clip(u * self.advection_strength, -1.0, 1.0)
        v_enhanced = np.clip(v * self.advection_strength, -1.0, 1.0)

        geq = get_equilibrium_numba(
            self.temp, u_enhanced, v_enhanced, self.w, self.c, 9, self.ny, self.nx
        )
        self.g = bgk_collision(self.g, geq, self.tau_g)

        for i in range(9):
            self.g[i] = np.roll(self.g[i], self.c[i, 0], axis=1)
            self.g[i] = np.roll(self.g[i], self.c[i, 1], axis=0)

        # ====== 边界条件 ======

        # 1. 入口: 冷流体进入 (温度为0)
        geq_cold = get_equilibrium_numba(
            np.array([[0.0]]),
            np.array([[self.u_in]]),
            np.array([[0.0]]),
            self.w,
            self.c,
            9,
            1,
            1,
        )
        for i in range(9):
            self.g[i, self.inlet_mask] = geq_cold[i, 0, 0]

        # 2. 出口: 对流出流 (自然流出,允许热量带走)
        for i in range(9):
            if self.c[i, 0] > 0:  # 向右流动的方向
                self.g[i, self.outlet_mask] = self.g[i, :, -2][self.outlet_mask[:, -1]]

        # 3. 固体壁面: 绝热边界 (反弹)
        for i in range(9):
            self.g[i, self.boundary] = self.g[self.opp[i], self.boundary]

    # =========================
    # Pressure drop (correct)
    # =========================
    def get_pressure_drop(self):
        x1 = int(self.nx * 0.15)
        x2 = int(self.nx * 0.85)
        rho1 = np.mean(self.rho[10:-10, x1])
        rho2 = np.mean(self.rho[10:-10, x2])
        return (rho1 - rho2) * self.cs_sq


# =========================
# Run
# =========================
def start_sim():
    FLUID_BLOCK = 10
    SLOW_SLIME = 0.85  # pre-fluid update
    LIVE_SLOW = 0.85  # post-fluid update

    solver = ThermalLBM()
    slime = SlimeAgent(solver)

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    img_v = axes[0].imshow(
        np.zeros((solver.ny, solver.nx)),
        cmap="magma",
        origin="lower",
        norm=PowerNorm(gamma=0.5, vmin=1e-6, vmax=0.18),
    )
    axes[0].set_title("Velocity magnitude (log-compressed)")
    plt.colorbar(img_v, ax=axes[0])

    img_t = axes[1].imshow(
        np.zeros((solver.ny, solver.nx)),
        cmap="inferno",
        origin="lower",
        norm=PowerNorm(gamma=0.7, vmin=1e-6, vmax=1.0),
    )
    axes[1].set_title("Temperature (log-compressed)")
    plt.colorbar(img_t, ax=axes[1])

    slime_cmap = LinearSegmentedColormap.from_list("slime_red_blue", ["red", "blue"])
    img_s = axes[2].imshow(
        np.zeros((solver.ny, solver.nx)),
        cmap=slime_cmap,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title("Slime (blue) vs open fluid (red)")
    plt.colorbar(img_s, ax=axes[2])

    def refresh(label):
        vmag = np.sqrt(solver.u**2 + solver.v**2)
        vmag_safe = np.maximum(vmag, 1e-6)
        temp_safe = np.maximum(solver.temp, 1e-6)

        img_v.set_data(vmag_safe)
        img_t.set_data(temp_safe)
        img_s.set_data(slime.slime)

        heat_out = np.sum(solver.u[:, -2] * solver.temp[:, -2])
        dp = solver.get_pressure_drop()
        fig.suptitle(f"{label} | ΔP={dp:.6f} | Heat Out={heat_out:.4f}")
        plt.pause(0.01)

    # Seed slime immediately without warmup
    slime.initialize_from_fields(solver)
    solver.apply_slime_mask(slime.get_obstacle_mask())
    refresh("Init slime placement")

    # Continuous co-evolution: slime → fluid block → slime (loop)
    cycle = 0
    block_steps = FLUID_BLOCK
    try:
        while plt.fignum_exists(fig.number):
            cycle += 1
            slime.update(solver, slow_factor=SLOW_SLIME)
            solver.apply_slime_mask(slime.get_obstacle_mask())
            refresh(f"Cycle {cycle}: slime relax (block {block_steps})")

            solver.step(block_steps)
            refresh(f"Cycle {cycle}: fluid {block_steps}")

            slime.update(solver, slow_factor=LIVE_SLOW)
            solver.apply_slime_mask(slime.get_obstacle_mask())
            refresh(f"Cycle {cycle}: post-fluid slime")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    start_sim()
