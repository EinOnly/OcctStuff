import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 复制边界设置逻辑
nx, ny = 300, 150
inlet_size = 30

# 1. 强制整圈封闭
boundary = np.ones((ny, nx), dtype=np.bool_)
boundary[1:-1, 1:-1] = False

# 2. 挖开入口 (左下角)
inlet_mask = np.zeros((ny, nx), dtype=np.bool_)
inlet_mask[1 : inlet_size, 0] = True
boundary[1 : inlet_size, 0] = False

# 3. 挖开出口 (右上角)
outlet_mask = np.zeros((ny, nx), dtype=np.bool_)
outlet_mask[ny - inlet_size : ny - 1, -1] = True
boundary[ny - inlet_size : ny - 1, -1] = False

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 显示边界mask
axes[0, 0].imshow(boundary, cmap='RdYlGn_r', origin='lower', aspect='auto')
axes[0, 0].set_title('Boundary (Red=Wall, Green=Open)')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# 显示入口mask
axes[0, 1].imshow(inlet_mask, cmap='Greens', origin='lower', aspect='auto')
axes[0, 1].set_title('Inlet Mask (Green=Inlet)')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')

# 显示出口mask
axes[1, 0].imshow(outlet_mask, cmap='Reds', origin='lower', aspect='auto')
axes[1, 0].set_title('Outlet Mask (Red=Outlet)')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')

# 综合显示
combined = np.zeros((ny, nx, 3))
combined[boundary] = [1, 0, 0]  # 红色=墙壁
combined[inlet_mask] = [0, 1, 0]  # 绿色=入口
combined[outlet_mask] = [0, 0, 1]  # 蓝色=出口

axes[1, 1].imshow(combined, origin='lower', aspect='auto')
axes[1, 1].set_title('Combined (Red=Wall, Green=Inlet, Blue=Outlet)')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')

# 标注关键位置
for ax in axes.flat:
    # 标注四个角
    ax.plot(0, 0, 'wo', markersize=8, markeredgecolor='yellow', markeredgewidth=2)
    ax.plot(nx-1, 0, 'wo', markersize=8, markeredgecolor='yellow', markeredgewidth=2)
    ax.plot(0, ny-1, 'wo', markersize=8, markeredgecolor='yellow', markeredgewidth=2)
    ax.plot(nx-1, ny-1, 'wo', markersize=8, markeredgecolor='yellow', markeredgewidth=2)

    ax.text(0, 0, 'BL', color='yellow', fontsize=10, ha='left', va='bottom')
    ax.text(nx-1, 0, 'BR', color='yellow', fontsize=10, ha='right', va='bottom')
    ax.text(0, ny-1, 'TL', color='yellow', fontsize=10, ha='left', va='top')
    ax.text(nx-1, ny-1, 'TR', color='yellow', fontsize=10, ha='right', va='top')

plt.tight_layout()
plt.savefig('/Users/ein/EinDev/OcctStuff/test/boundary_debug.png', dpi=150)
print("Saved boundary debug visualization to: test/boundary_debug.png")

# 打印关键信息
print(f"\nGrid size: {nx} x {ny}")
print(f"Inlet size: {inlet_size}")
print(f"\nInlet range: y=[1:{inlet_size}], x=[0]")
print(f"Outlet range: y=[{ny-inlet_size}:{ny-1}], x=[{nx-1}]")
print(f"\nCorner checks:")
print(f"  Bottom-left (0,0):     boundary={boundary[0,0]}, inlet={inlet_mask[0,0]}, outlet={outlet_mask[0,0]}")
print(f"  Bottom-right (0,{nx-1}):  boundary={boundary[0,nx-1]}, inlet={inlet_mask[0,nx-1]}, outlet={outlet_mask[0,nx-1]}")
print(f"  Top-left ({ny-1},0):     boundary={boundary[ny-1,0]}, inlet={inlet_mask[ny-1,0]}, outlet={outlet_mask[ny-1,0]}")
print(f"  Top-right ({ny-1},{nx-1}): boundary={boundary[ny-1,nx-1]}, inlet={inlet_mask[ny-1,nx-1]}, outlet={outlet_mask[ny-1,nx-1]}")
print(f"\nInlet start (1,0):     boundary={boundary[1,0]}, inlet={inlet_mask[1,0]}")
print(f"Outlet start ({ny-inlet_size},{nx-1}): boundary={boundary[ny-inlet_size,nx-1]}, outlet={outlet_mask[ny-inlet_size,nx-1]}")

plt.show()
