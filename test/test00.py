import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def smooth_closed_curve(points, smoothness=0):
    """通过 B 样条平滑闭合曲线"""
    points = np.array(points)
    points = np.vstack([points, points[0]])  # 闭合
    tck, u = splprep([points[:, 0], points[:, 1]], s=smoothness, per=True)
    unew = np.linspace(0, 1, 200)
    out = splev(unew, tck)
    return np.vstack(out).T

def offset_curve(curve, offset):
    """沿法线方向偏移封闭曲线（简单实现）"""
    dx = np.gradient(curve[:, 0])
    dy = np.gradient(curve[:, 1])
    length = np.hypot(dx, dy)
    dx /= length
    dy /= length
    normals = np.column_stack([-dy, dx])
    return curve + normals * offset

# 原始多边形点
points = np.array([[1, 1], [5, 1], [4, 4], [2, 5]])

# 平滑曲线（拟合封闭贝塞尔样条）
smooth = smooth_closed_curve(points)

# 生成“内切曲线”：向内偏移一定距离
inner = offset_curve(smooth, -0.2)

# 绘图展示
plt.figure(figsize=(6, 6))
plt.plot(*points.T, 'k--o', label="原始多边形")
plt.plot(*smooth.T, 'skyblue', linewidth=2, label="光滑封闭曲线")
plt.plot(*inner.T, 'orangered', linewidth=2, label="内切曲线")
plt.gca().set_aspect('equal')
plt.legend()
plt.title("平滑封闭曲线与内切曲线")
plt.show()