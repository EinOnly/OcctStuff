import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter

# 参数
num_points = 1000
num_relaxations = 10
bounds = [0, 100, 0, 100]
density_map_resolution = 200

# 1️⃣ 生成随机密度图
def generate_density_map(res=200, smoothness=5):
    density = np.random.rand(res, res)
    density = gaussian_filter(density, sigma=smoothness)
    density /= density.max()  # 归一化到 [0, 1]
    return density

# 2️⃣ 根据密度图 importance sampling
def sample_points_from_density(density, num_samples, bounds):
    h, w = density.shape
    flat = density.flatten()
    flat /= flat.sum()
    indices = np.random.choice(len(flat), size=num_samples, p=flat)
    ys, xs = np.divmod(indices, w)

    # 映射回真实坐标
    xmin, xmax, ymin, ymax = bounds
    x_coords = xmin + (xmax - xmin) * xs / w
    y_coords = ymin + (ymax - ymin) * ys / h
    return np.column_stack((x_coords, y_coords))

# 3️⃣ Lloyd Relaxation 保持不变
def lloyd_relaxation(points, bounds, iterations):
    xmin, xmax, ymin, ymax = bounds
    for _ in range(iterations):
        vor = Voronoi(points)
        new_points = []
        for point_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                new_points.append(points[point_idx])
                continue
            polygon = [vor.vertices[i] for i in region]
            poly = np.array(polygon)
            centroid = poly.mean(axis=0)
            centroid[0] = np.clip(centroid[0], xmin, xmax)
            centroid[1] = np.clip(centroid[1], ymin, ymax)
            new_points.append(centroid)
        points = np.array(new_points)
    return points

# 执行流程
density_map = generate_density_map(density_map_resolution)
points = sample_points_from_density(density_map, num_points, bounds)
relaxed_points = lloyd_relaxation(points, bounds, num_relaxations)
vor = Voronoi(relaxed_points)

# 4️⃣ 绘图
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# 左图：密度图
axs[0].imshow(density_map, cmap='viridis', extent=bounds, origin='lower')
axs[0].set_title("Density Map (higher = more points)")

# 右图：Voronoi 图
voronoi_plot_2d(vor, ax=axs[1], show_vertices=False, line_colors='black', line_width=0.8, point_size=1.5)
axs[1].set_xlim(bounds[0], bounds[1])
axs[1].set_ylim(bounds[2], bounds[3])
axs[1].set_aspect('equal')
axs[1].set_title(f"Relaxed Voronoi with Density-Controlled Sampling\n({num_relaxations} iterations)")

plt.tight_layout()
plt.show()