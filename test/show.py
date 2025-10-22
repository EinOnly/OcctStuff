import numpy as np
import matplotlib.pyplot as plt

# 参数
k = 3                   # 控制花瓣数量
num_curves = 3       # 绘制多少条旋转的玫瑰曲线
theta = np.linspace(0, 2 * np.pi, 1000)

# 创建画布
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.set_rticks([])  # 去掉半径刻度
ax.set_xticks([])  # 去掉角度刻度
ax.grid(False)

# 绘制多条相位旋转后的玫瑰曲线
for i in range(num_curves):
    phase = (2 * np.pi / num_curves) * i
    r = np.abs(np.sin(k * theta + phase))  # abs 是为了更对称美观
    ax.plot(theta, r, color='black', linewidth=1.5)

plt.tight_layout()
plt.show()