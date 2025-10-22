import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter

# Generate a random density map
def generate_density_map(res=200, smoothness=5):
    density = np.random.rand(res, res)
    density = gaussian_filter(density, sigma=smoothness)
    density /= density.max()  # Normalize to [0, 1]
    return density

# Perform importance sampling based on the density map
def sample_points_from_density(density, num_samples, bounds):
    h, w = density.shape
    flat = density.flatten()
    flat /= flat.sum()
    indices = np.random.choice(len(flat), size=num_samples, p=flat)
    ys, xs = np.divmod(indices, w)

    # Map indices back to real-world coordinates
    xmin, xmax, ymin, ymax = bounds
    x_coords = xmin + (xmax - xmin) * xs / w
    y_coords = ymin + (ymax - ymin) * ys / h
    return np.column_stack((x_coords, y_coords))

# Perform Lloyd relaxation to improve uniformity
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
    # Parameters
    num_points = 1000
    num_relaxations = 2
    bounds = [0, 100, 0, 100]
    density_map_resolution = 200

    # Step 1: Generate density map
    density_map = generate_density_map(density_map_resolution)

    # Step 2: Sample points based on density
    # points = sample_points_from_density(density_map, num_points, bounds)
    points = generate_uniform_grid_points(num_points, bounds, 1)
    
    # Step 3: Apply Lloyd relaxation
    relaxed_points = lloyd_relaxation(points, bounds, num_relaxations)

    # Step 4: Compute Voronoi diagram
    vor = Voronoi(relaxed_points)

    # Step 5: Plot results
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Density map
    axs[0].imshow(density_map, cmap='viridis', extent=bounds, origin='lower')
    axs[0].set_title("Density Map (higher = more points)")

    # Right: Voronoi diagram
    voronoi_plot_2d(vor, ax=axs[1], show_vertices=False, line_colors='black', line_width=0.8, point_size=1.5)
    # axs[1].set_xlim(bounds[0], bounds[1])
    # axs[1].set_ylim(bounds[2], bounds[3])
    # axs[1].set_aspect('equal')
    # axs[1].set_title(f"Relaxed Voronoi with Density-Controlled Sampling\n({num_relaxations} iterations)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()