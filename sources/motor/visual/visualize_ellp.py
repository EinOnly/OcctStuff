import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def superellipse_to_nurbs(a, b, m, n, num_samples=100, degree=3):
    """
    将超椭圆转换为 NURBS 曲线
    
    参数：
    a, b: 半轴长度
    m, n: 指数参数
    num_samples: 采样点数
    degree: B样条次数
    """
    # 生成超椭圆的采样点
    t = np.linspace(0, 2*np.pi, num_samples)
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/m)
    y = b * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/n)
    
    # 拟合 NURBS 曲线
    # splprep 返回：(knots, coefficients, degree) 和 参数u
    tck, u = splprep([x, y], s=0, k=degree, per=True)
    
    # 生成平滑的 NURBS 曲线
    u_new = np.linspace(0, 1, 1000)
    x_nurbs, y_nurbs = splev(u_new, tck)
    
    return tck, (x_nurbs, y_nurbs), (x, y)

# 使用示例
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

configs = [
    (3, 2, 2, 2, "标准椭圆"),
    (3, 2, 4, 2, "x更方"),
    (3, 2, 2, 4, "y更方"),
    (3, 2, 0.5, 2, "x内凹"),
    (3, 2, 2, 0.5, "y内凹"),
    (3, 2, 4, 0.5, "混合"),
]

for ax, (a, b, m, n, title) in zip(axes.flat, configs):
    # 拟合 NURBS
    tck, (x_nurbs, y_nurbs), (x_sample, y_sample) = superellipse_to_nurbs(a, b, m, n)
    # 绘制
    ax.plot(x_sample, y_sample, 'ro', markersize=3, label='采样点', alpha=0.5)
    ax.plot(x_nurbs, y_nurbs, 'b-', linewidth=2, label='NURBS曲线')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()