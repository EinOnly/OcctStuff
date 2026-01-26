import numpy as np
from scipy.spatial import Voronoi

def relaxed_voronoi(points, bounds, iterations):
    xmin, xmax, ymin, ymax = bounds
    centroids = []
    for _ in range(iterations):
        try:
            vor = Voronoi(points)
        except:
            return points, None, []
        new_points = []
        centroids.clear()  # clear centroids for each iteration
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
        vor =  Voronoi(points)

        # Extract complementary points by merging vertices with short edges
        def extract_complementary_points(vor, bounds, threshold):
            """
            Extract Voronoi vertices as complementary seed points.
            Merge vertices that are connected by edges shorter than threshold.

            Parameters:
            - vor: Voronoi diagram
            - bounds: [xmin, xmax, ymin, ymax]
            - threshold: edge length threshold for merging vertices
            """
            xmin, xmax, ymin, ymax = bounds

            # Build a graph of Voronoi vertices and their connections
            from collections import defaultdict
            vertex_neighbors = defaultdict(set)

            for region_index in vor.point_region:
                region = vor.regions[region_index]
                if not region or -1 in region:
                    continue

                # Connect adjacent vertices in the region (they share edges)
                for i in range(len(region)):
                    v1 = region[i]
                    v2 = region[(i + 1) % len(region)]
                    if v1 >= 0 and v2 >= 0:
                        vertex_neighbors[v1].add(v2)
                        vertex_neighbors[v2].add(v1)

            # Merge vertices connected by short edges using Union-Find
            parent = {}

            def find(x):
                if x not in parent:
                    parent[x] = x
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Merge vertices with short edges
            for v1 in vertex_neighbors:
                for v2 in vertex_neighbors[v1]:
                    if v1 < v2:  # avoid duplicate processing
                        edge_length = np.linalg.norm(vor.vertices[v1] - vor.vertices[v2])
                        if edge_length < threshold:
                            union(v1, v2)

            # Group vertices by their root and compute centroids
            groups = defaultdict(list)
            for v in vertex_neighbors:
                root = find(v)
                groups[root].append(v)

            # Compute centroid for each group
            merged_points = []
            for group_vertices in groups.values():
                vertices_coords = [vor.vertices[v] for v in group_vertices]
                centroid = np.mean(vertices_coords, axis=0)
                # Clip to bounds
                cx = np.clip(centroid[0], xmin, xmax)
                cy = np.clip(centroid[1], ymin, ymax)
                merged_points.append([cx, cy])

            return np.array(merged_points) if merged_points else np.array([])

        # Calculate average edge length from original Voronoi diagram
        edge_lengths = []
        for region_index in vor.point_region:
            region = vor.regions[region_index]
            if not region or -1 in region:
                continue
            for i in range(len(region)):
                v1 = region[i]
                v2 = region[(i + 1) % len(region)]
                if v1 >= 0 and v2 >= 0:
                    edge_length = np.linalg.norm(vor.vertices[v1] - vor.vertices[v2])
                    edge_lengths.append(edge_length)

        avg_edge_length = np.mean(edge_lengths) if edge_lengths else 5.0
        target_points = len(points)  # Target number of complementary points

        # Binary search to find the optimal coefficient
        def find_optimal_coefficient(avg_edge_length, target_count, tolerance=0.1):
            """
            Use binary search to find coefficient that gives desired point count.

            Parameters:
            - avg_edge_length: average edge length
            - target_count: desired number of complementary points
            - tolerance: acceptable ratio difference (e.g., 0.1 means 10% tolerance)

            Returns:
            - optimal coefficient
            """
            low, high = 0.5, 1.5  # Search range for coefficient
            best_coef = 0.93
            best_diff = float('inf')

            for _ in range(10):  # Max 10 iterations
                mid = (low + high) / 2
                test_points = extract_complementary_points(vor, bounds, avg_edge_length * mid)
                count = len(test_points)

                diff = abs(count - target_count)
                if diff < best_diff:
                    best_diff = diff
                    best_coef = mid

                # Check if we're close enough
                ratio = count / target_count if target_count > 0 else 0
                if abs(ratio - 1.0) < tolerance:
                    return mid

                # Adjust search range
                if count > target_count:
                    # Too many points, need smaller coefficient (more merging)
                    low = mid
                else:
                    # Too few points, need larger coefficient (less merging)
                    high = mid

            return best_coef

        optimal_coef = find_optimal_coefficient(avg_edge_length, target_points, tolerance=0.15)
        points_complementary = extract_complementary_points(vor, bounds, avg_edge_length * optimal_coef)
    try:
        return points, Voronoi(points), centroids, points_complementary
    except:
        return points, None, centroids, None

def relaxed_voronoi_with_mask(points, bounds, iterations, valid_mask=None, mask_extent=None):
    """
    Lloyd's relaxation on a Voronoi diagram, restricted to valid_mask region.

    Parameters:
        points: (N, 2) input seed points
        bounds: [xmin, xmax, ymin, ymax]
        iterations: number of relaxation steps
        valid_mask: optional bool mask (H, W)
        mask_extent: [xmin, xmax, ymin, ymax] extent of valid_mask
    """
    xmin, xmax, ymin, ymax = bounds
    centroids = []

    for _ in range(iterations):
        try:
            vor = Voronoi(points)
        except Exception:
            return points, None, []

        new_points = []
        centroids.clear()

        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                # unbounded region
                new_points.append(points[i])
                centroids.append(points[i])
                continue

            polygon = [vor.vertices[j] for j in region if j >= 0]
            if len(polygon) == 0:
                new_points.append(points[i])
                centroids.append(points[i])
                continue

            centroid = np.mean(polygon, axis=0)
            cx, cy = centroid

            # Restrict within bounds
            cx = np.clip(cx, xmin, xmax)
            cy = np.clip(cy, ymin, ymax)

            # Check valid mask
            if valid_mask is not None and mask_extent is not None:
                mx0, mx1, my0, my1 = mask_extent
                h, w = valid_mask.shape

                u = (cx - mx0) / (mx1 - mx0)
                v = (cy - my0) / (my1 - my0)
                ix = int(u * w)
                iy = int(v * h)

                if not (0 <= ix < w and 0 <= iy < h) or not valid_mask[iy, ix]:
                    # invalid centroid, keep old point
                    new_points.append(points[i])
                    centroids.append(points[i])
                    continue

            new_points.append([cx, cy])
            centroids.append([cx, cy])

        points = np.array(new_points)

    try:
        return points, Voronoi(points), centroids
    except Exception:
        return points, None, centroids

def push_apart_vertices(vertices, min_dist=0.01, strength=0.2, iterations=3):
    """
    Push apart all Voronoi vertices that are too close to each other.

    Parameters:
    - vertices: np.ndarray of shape (N, 2)
    - min_dist: minimum distance between vertices
    - strength: push amount (0 ~ 1)
    - iterations: repeat passes

    Returns:
    - np.ndarray: modified vertices
    """
    verts = np.array(vertices, dtype=np.float64)
    N = len(verts)

    for _ in range(iterations):
        disp = np.zeros_like(verts)
        for i in range(N):
            for j in range(i + 1, N):
                d = verts[i] - verts[j]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-8:
                    offset = (min_dist - dist) * strength * d / dist
                    disp[i] += offset
                    disp[j] -= offset
        verts += disp

    return verts

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial import voronoi_plot_2d
    from shapely.geometry import Polygon
    import sys
    sys.path.append('..')
    from points import Points

    # Test complementary Voronoi cells
    print("Generating complementary Voronoi cells...")

    # 1. Define region and generate points using Points class
    bounds = [0, 100, 0, 100]

    # Create a rectangular region with a hole
    outer = [(5, 5), (95, 5), (95, 95), (5, 95)]
    hole = [(35, 35), (65, 35), (65, 65), (35, 65)]
    region_shape = Polygon(outer, [hole])

    print("Generating points using Points class...")
    points_gen = Points(shape=region_shape, spacing=4.0, offset_layers=1)
    points_a = np.array(points_gen.get_points())
    print(f"Generated {len(points_a)} points")

    # 2. Create valid mask from region
    print("\nCreating valid mask from region...")
    res_x, res_y = 200, 200
    xx, yy = np.meshgrid(np.linspace(bounds[0], bounds[1], res_x),
                         np.linspace(bounds[2], bounds[3], res_y))
    points_grid = np.c_[xx.ravel(), yy.ravel()]

    from shapely.geometry import Point
    mask = np.array([region_shape.contains(Point(p)) for p in points_grid]).reshape(res_y, res_x)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask coverage: {np.sum(mask)} / {mask.size} pixels")

    # 3. Generate density map
    print("\nGenerating density map...")
    from maps import generate_density_with_mask
    density = generate_density_with_mask(
        size=(100, 100),
        valid_mask=mask,
        circles=None,
        gradient_direction=(0, -1),
        gradient_strength=0.5,
        falloff=10.0
    )
    print(f"Density shape: {density.shape}")

    # 4. Apply Lloyd relaxation and get complementary points
    points_a_relaxed, vor_a, _, points_b = relaxed_voronoi(
        points_a, bounds=bounds, iterations=10
    )

    # 3. Create Voronoi diagrams
    vor_b = Voronoi(points_b) if points_b is not None and len(points_b) > 0 else None

    # 4. Plot both diagrams overlapped
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Helper function to draw region boundary
    from matplotlib.patches import Polygon as MplPolygon
    def draw_region(ax, shape):
        patch = MplPolygon(
            list(shape.exterior.coords),
            closed=True, fill=False,
            edgecolor="black", linewidth=2, linestyle="--", alpha=0.8
        )
        ax.add_patch(patch)
        for interior in shape.interiors:
            hole_patch = MplPolygon(
                list(interior.coords),
                closed=True, fill=False,
                edgecolor="red", linewidth=1.5, linestyle="--", alpha=0.8
            )
            ax.add_patch(hole_patch)

    # Plot A only
    ax1 = axes[0]
    voronoi_plot_2d(
        vor_a,
        ax=ax1,
        show_vertices=False,
        line_colors="blue",
        line_width=1.5,
        point_size=3,
    )
    ax1.plot(points_a_relaxed[:, 0], points_a_relaxed[:, 1], "bo", markersize=5, label="Points A")
    draw_region(ax1, region_shape)
    ax1.set_xlim(bounds[0], bounds[1])
    ax1.set_ylim(bounds[2], bounds[3])
    ax1.set_title("Voronoi A (Original)", fontsize=14)
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot B only
    ax2 = axes[1]
    if vor_b is not None:
        voronoi_plot_2d(
            vor_b,
            ax=ax2,
            show_vertices=False,
            line_colors="red",
            line_width=1.5,
            point_size=3,
        )
        ax2.plot(points_b[:, 0], points_b[:, 1], "ro", markersize=5, label="Points B (Complementary)")
    draw_region(ax2, region_shape)
    ax2.set_xlim(bounds[0], bounds[1])
    ax2.set_ylim(bounds[2], bounds[3])
    ax2.set_title("Voronoi B (Complementary)", fontsize=14)
    ax2.set_aspect("equal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot both overlapped
    ax3 = axes[2]
    voronoi_plot_2d(
        vor_a,
        ax=ax3,
        show_vertices=False,
        line_colors="blue",
        line_width=1.5,
        point_size=3,
    )
    ax3.plot(points_a_relaxed[:, 0], points_a_relaxed[:, 1], "bo", markersize=5, label="Points A", alpha=0.7)

    if vor_b is not None:
        voronoi_plot_2d(
            vor_b,
            ax=ax3,
            show_vertices=False,
            line_colors="red",
            line_width=1.5,
            point_size=3,
        )
        ax3.plot(points_b[:, 0], points_b[:, 1], "ro", markersize=5, label="Points B (Complementary)", alpha=0.7)

    draw_region(ax3, region_shape)
    ax3.set_xlim(bounds[0], bounds[1])
    ax3.set_ylim(bounds[2], bounds[3])
    ax3.set_title("Both Overlapped (Complementary Tessellation)", fontsize=14)
    ax3.set_aspect("equal")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/Users/ein/EinDev/OcctStuff/.cache/complementary_voronoi.png", dpi=150, bbox_inches="tight")

    # Print statistics
    print(f"\n{'='*60}")
    print(f"✓ Saved visualization to: /Users/ein/EinDev/OcctStuff/.cache/complementary_voronoi.png")
    print(f"{'='*60}")
    print(f"  Points A (original):      {len(points_a_relaxed)}")
    print(f"  Points B (complementary): {len(points_b) if points_b is not None else 0}")

    if points_b is not None and len(points_b) > 0:
        from scipy.spatial.distance import pdist
        dist_a = pdist(points_a_relaxed)
        dist_b = pdist(points_b)
        ratio_count = len(points_b) / len(points_a_relaxed)
        print(f"\n  Point count ratio (B/A):  {ratio_count:.2f}x")
        print(f"  Avg nearest-neighbor distance:")
        print(f"    Points A:               {np.mean(dist_a):.2f}")
        print(f"    Points B:               {np.mean(dist_b):.2f}")
        print(f"    Ratio (B/A):            {np.mean(dist_b)/np.mean(dist_a):.2f}x")
        print(f"\n  Total points (A+B):       {len(points_a_relaxed) + len(points_b)}")
        print(f"  Coverage improvement:     {((len(points_a_relaxed) + len(points_b)) / len(points_a_relaxed)):.1f}x denser")
    print(f"{'='*60}\n")

    plt.show()