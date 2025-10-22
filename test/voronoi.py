import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d
from noise import pnoise3
from scipy.interpolate import splprep, splev
from matplotlib.path import Path

def generate_perlin_density(t, res, scale=10.0):
    density = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            x = i / res * scale
            y = j / res * scale
            density[i, j] = pnoise3(x, y, t)
    density = density - density.min()
    density = density / density.max()
    return density

def move_points_by_density(points, density, bounds, res, dt):
    h, w = density.shape
    grad_y, grad_x = np.gradient(density)

    xmin, xmax, ymin, ymax = bounds
    for i in range(len(points)):
        px, py = points[i]
        xi = int((px - xmin) / (xmax - xmin) * (w - 1))
        yi = int((py - ymin) / (ymax - ymin) * (h - 1))
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)
        # gx = grad_x[yi, xi]
        # gy = grad_y[yi, xi]

        gx = grad_x[yi, xi]
        gy = grad_y[yi, xi]
        gx = gx / (np.sqrt(gx**2 + gy**2) + 1e-8)
        gy = gy / (np.sqrt(gx**2 + gy**2) + 1e-8)
        strength = density[yi, xi] ** 2  # 密度越大，移动越快

        points[i, 0] += gx * dt * strength * 3.0
        points[i, 1] += gy * dt * strength * 3.0
        points[i, 0] += gx * dt
        points[i, 1] += gy * dt

    points[:, 0] = np.clip(points[:, 0], xmin, xmax)
    points[:, 1] = np.clip(points[:, 1], ymin, ymax)
    return points

def relaxed_voronoi(points, bounds, iterations):
    xmin, xmax, ymin, ymax = bounds
    centroids = []
    for _ in range(iterations):
        try:
            vor = Voronoi(points)
        except:
            return points, None, []
        new_points = []
        centroids.clear()  # ✅ 每轮清空，只保留最后一轮的质心
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                new_points.append(points[i])
                centroids.append(points[i])
                continue
            polygon = [vor.vertices[j] for j in region if j >= 0]
            if len(polygon) == 0:
                new_points.append(points[i])
                centroids.append(points[i])
                continue
            centroid = np.mean(polygon, axis=0)
            centroid[0] = np.clip(centroid[0], xmin, xmax)
            centroid[1] = np.clip(centroid[1], ymin, ymax)
            new_points.append(centroid)
            centroids.append(centroid)
        points = np.array(new_points)
    try:
        return points, Voronoi(points), centroids
    except:
        return points, None, centroids

def chaikin_smoothing(points, iterations=4):
    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new_points += [Q, R]
        new_points.append(new_points[0])  # close the loop
        points = np.array(new_points)
    return points

def shrink_polygon(points, factor=0.85):
    center = points.mean(axis=0)
    shrunk = center + (points - center) * factor
    return shrunk

# generate points that are close to regular polygons
def generate_uniform_grid_points(num_points, bounds, aspect_ratio=1.0):
    """
    Generate a uniform grid of points with adjustable x/y spacing ratio.
    
    Parameters:
    - num_points: total number of points (approximate)
    - bounds: [xmin, xmax, ymin, ymax]
    - aspect_ratio: x_spacing / y_spacing (e.g., 2.0 means x-spacing is 2× y-spacing)
    """
    xmin, xmax, ymin, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin

    # Adjust grid shape based on aspect ratio
    area = width * height
    adjusted_height = np.sqrt(area / (num_points * aspect_ratio))
    adjusted_width = adjusted_height * aspect_ratio

    grid_cols = int(width / adjusted_width)
    grid_rows = int(height / adjusted_height)

    # Generate grid coordinates
    x = np.linspace(xmin, xmax, grid_cols)
    y = np.linspace(ymin, ymax, grid_rows)
    xv, yv = np.meshgrid(x, y)

    return np.column_stack((xv.ravel(), yv.ravel()))

def main():
    # 参数配置
    num_points = 600
    num_relaxations = 6
    bounds = [0, 100, 0, 100]
    res = 200
    scale = 1.0
    speed = 0.05
    dt = 0.5
    frames = 300
    interval = 100

    # 初始化点集
    points = np.random.rand(num_points, 2)
    points[:, 0] *= (bounds[1] - bounds[0])
    points[:, 1] *= (bounds[3] - bounds[2])

    # 初始化画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')

    def update(frame):
        nonlocal points

        ax.clear()
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')
        ax.set_title("Smooth Voronoi Driven by Perlin Density Field")

        # 📌 密度图生成与归一化
        t = frame * speed
        density = generate_perlin_density(t, res, scale=scale)
        density = (density - density.min()) / (density.max() - density.min())
        density = density ** 3.0
        ax.imshow(density, cmap='viridis', extent=bounds, origin='lower', alpha=0.85)

        # 📌 梯度预计算（全局只算一次）
        grad_y, grad_x = np.gradient(density)
        h, w = density.shape
        xmin, xmax, ymin, ymax = bounds

        # 📌 点移动
        points = move_points_by_density(points, density, bounds, res, dt)
        # points = generate_uniform_grid_points(num_points, bounds, 1.0)

        # 📌 Voronoi 构造与质心
        relaxed_points, vor, centroids = relaxed_voronoi(points, bounds, num_relaxations)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=0.8, point_size=1.5)

        # 📌 绘制细胞中心点
        if False and centroids:
            centroids = np.array(centroids)
            ax.scatter(centroids[:, 0], centroids[:, 1], color='blue', s=5, zorder=5)

            # 🔴 每个中心点绘制红色箭头（密度梯度方向）
            for cx, cy in centroids:
                xi = int((cx - xmin) / (xmax - xmin) * (w - 1))
                yi = int((cy - ymin) / (ymax - ymin) * (h - 1))
                xi = np.clip(xi, 0, w - 1)
                yi = np.clip(yi, 0, h - 1)

                gx = grad_x[yi, xi]
                gy = grad_y[yi, xi]

                norm = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
                dx = gx / norm * 3.0
                dy = gy / norm * 3.0
                ax.arrow(cx, cy, dx, dy, head_width=0.6, head_length=0.5, fc='red', ec='red', linewidth=0.8)

            # Render original points before relaxation
            ax.scatter(points[:, 0], points[:, 1], color='green', s=3, alpha=0.7, zorder=4)

        # 📌 绘制 Voronoi 区域轮廓和平滑边界
        if vor:
            for region_idx in vor.point_region:
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    continue

                polygon = [vor.vertices[i] for i in region]
                poly = np.array(polygon)
                if len(poly) < 3:
                    continue

                # 原始边界绘制（浅线）
                # ax.fill(*zip(*poly), edgecolor='black', facecolor='none', linewidth=0.1)

                # 📌 平滑 + 收缩
                poly = np.vstack([poly, poly[0]])  # 闭合
                smooth_poly = chaikin_smoothing(poly, iterations=4)
                smooth_poly = shrink_polygon(smooth_poly, factor=0.9)

                # 📌 额外：沿密度梯度方向的垂直方向压缩 smooth_poly
                center = smooth_poly.mean(axis=0)
                xi = int((center[0] - xmin) / (xmax - xmin) * (w - 1))
                yi = int((center[1] - ymin) / (ymax - ymin) * (h - 1))
                xi = np.clip(xi, 0, w - 1)
                yi = np.clip(yi, 0, h - 1)

                gx = grad_x[yi, xi]
                gy = grad_y[yi, xi]
                norm = np.sqrt(gx**2 + gy**2) + 1e-8
                gx /= norm
                gy /= norm
                nx, ny = -gy, gx  # 垂直方向
                squash_factor = 0.7  # 压缩强度

                for i in range(len(smooth_poly)):
                    v = smooth_poly[i] - center
                    d = v[0] * nx + v[1] * ny
                    offset = np.array([nx, ny]) * d * (1 - squash_factor)
                    smooth_poly[i] = smooth_poly[i] - offset

                # 📌 绘制最终变形后的边界
                path = Path(poly)
                if np.all(path.contains_points(smooth_poly)):
                    ax.plot(smooth_poly[:, 0], smooth_poly[:, 1], color='black', linewidth=0.8)
                else:
                    ax.plot(poly[:, 0], poly[:, 1], color='gray', linewidth=0.5)
                    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.show()


if __name__ == "__main__":
    main()