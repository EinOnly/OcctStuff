import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def catmull_rom_chain(P, n_points=100):
    def catmull_rom(p0, p1, p2, p3, n_points):
        t = np.linspace(0, 1, n_points)
        t = t[:, np.newaxis]
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t**3
        )
    
    P = np.array(P)
    curve = []
    for i in range(len(P)):
        p0 = P[i - 1]
        p1 = P[i]
        p2 = P[(i + 1) % len(P)]
        p3 = P[(i + 2) % len(P)]
        segment = catmull_rom(p0, p1, p2, p3, n_points // len(P))
        curve.append(segment)
    return np.concatenate(curve, axis=0).T

# 示例用法
polygon = [(0, 0), (1, 2), (3, 3), (4, 1), (2, -1)]
x, y = catmull_rom_chain(polygon)

plt.plot(*zip(*polygon + [polygon[0]]), 'ro--', label='Polygon')
plt.plot(x, y, 'b-', label='Catmull-Rom')
plt.axis('equal')
plt.legend()
plt.show()