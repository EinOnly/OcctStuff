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
        
        # Extract all valid vertices
        def extract_all_valid_vertices(vor):
            points = set()
            for region_index in vor.point_region:
                region = vor.regions[region_index]
                if not region or -1 in region:
                    continue  # ignore unbounded regions
                for vertex_index in region:
                    vertex = vor.vertices[vertex_index]
                    points.add(tuple(vertex))  # add as a tuple to avoid duplicates
            return np.array(list(points))
        points_edge = extract_all_valid_vertices(vor)
    try:
        return points, Voronoi(points), centroids, points_edge
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