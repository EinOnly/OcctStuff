import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Lorenz 系统参数
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# 初始状态 & 时间范围
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# 数值求解
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

# 结果
x, y, z = sol.y

# ======= 静态图 =======
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# ======= 动画演示 =======
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], lw=1)

ax.set_xlim((min(x), max(x)))
ax.set_ylim((min(y), max(y)))
ax.set_zlim((min(z), max(z)))
ax.set_title("Lorenz Attractor Animation")

def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(x), interval=1, blit=True)
plt.show()