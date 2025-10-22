import pyvoro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

points = np.random.rand(20, 2)
limits = [[0, 1], [0, 1]]
cells = pyvoro.compute_2d_voronoi(points, limits, 1.0)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

for cell in cells:
    poly = Polygon(cell['vertices'], closed=True, edgecolor='k', fill=True, alpha=0.2)
    ax.add_patch(poly)
    # 兼容数组/标量/float等所有类型
    orig = cell['original']
    i = int(np.array(orig).item())
    x, y = points[i]
    ax.plot(x, y, 'ro')
    ax.text(x, y, str(i), color="blue", fontsize=10)

plt.title("PyVoro 2D Voronoi Diagram")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()