import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep
from scipy.spatial import cKDTree
from shapely.ops import unary_union
from scipy.spatial import Voronoi

class Points:
    def __init__(
            self, 
            shape: Polygon = None,
            spacing: float = 1.0, 
            offset_layers: int = 1, 
            boundary_density: float = 1.0,
            logger=None
        ):
        if not isinstance(shape, Polygon):
            raise ValueError("Expected a shapely Polygon for 'shape'.")

        self.shape = shape
        self.shape_prep = prep(shape)
        self.spacing = spacing
        self.offset_layers = offset_layers
        self.boundary_density = boundary_density
        self.logger = logger

        self.points = []
        self.fixed = []

        self._generate()

    def _generate(self):
        bounds = self.shape.bounds  # (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = bounds

        temp_points = []
        temp_fixed_mask = []

        # 1. Outer boundary points using interpolation (precise)
        line = LineString(self.shape.exterior)
        boundary_length = line.length
        n_points = max(int(boundary_length / (self.spacing / self.boundary_density)), 2)
        for i in range(n_points):
            pt = line.interpolate(i / n_points, normalized=True)
            temp_points.append([pt.x, pt.y])
            temp_fixed_mask.append(True)

        # 2. Offset inward layers, interpolate points with decreasing density
        for layer in range(1, self.offset_layers + 1):
            offset = -layer * self.spacing
            offset_shape = self.shape.buffer(offset)
            if offset_shape.is_empty or not isinstance(offset_shape, Polygon):
                continue

            outline = LineString(offset_shape.exterior)
            layer_length = outline.length
            compression_factor = 1.0 / (1.0 + 0.5 * layer)
            n_layer_points = max(int(layer_length / (self.spacing / (self.boundary_density * compression_factor))), 2)
            for i in range(n_layer_points):
                pt = outline.interpolate(i / n_layer_points, normalized=True)
                if self.shape_prep.contains(pt):
                    temp_points.append([pt.x, pt.y])
                    temp_fixed_mask.append(True)

        # 3. Interior area filled with staggered grid
        inner_area = self.shape.buffer(-self.offset_layers * self.spacing - self.offset_layers)
        if inner_area.is_empty or not isinstance(inner_area, Polygon):
            inner_area = self.shape
        inner_prep = prep(inner_area)

        x_range = np.arange(xmin, xmax + self.spacing, self.spacing)
        y_range = np.arange(ymin, ymax + self.spacing, self.spacing * np.sqrt(3)/2)

        for j, y in enumerate(y_range):
            x_offset = 0.5 * self.spacing if j % 2 == 1 else 0.0
            for x in x_range:
                pt = [x + x_offset, y]
                if inner_prep.contains(Point(pt)):
                    temp_points.append(pt)
                    temp_fixed_mask.append(False)

        # 4. Deduplicate and synchronize fixed status
        points_arr = np.array(temp_points)
        fixed_arr = np.array(temp_fixed_mask, dtype=bool)

        if len(points_arr) == 0:
            self.points = np.empty((0, 2))
            self.fixed = np.array([], dtype=bool)
            return

        combined = np.hstack([points_arr, fixed_arr[:, None]])
        unique_combined = np.unique(combined, axis=0)

        self.points = unique_combined[:, :2]
        self.fixed = unique_combined[:, 2].astype(bool)

        if self.logger:
            self.logger.info(f"Generated {len(self.points)} points (fixed: {np.sum(self.fixed)}).")

    def move(self, density, dt, mask_extent, iterations=1, repel_radius=1.5, repel_strength=0.05):
        moved = self.points.copy()
        h, w = density.shape
        grad_y, grad_x = np.gradient(density)
        xmin, xmax, ymin, ymax = mask_extent
        margin = 1e-6

        for _ in range(iterations):
            new_points = moved.copy()
            tree = cKDTree(moved)

            for i, (px, py) in enumerate(moved):
                if self.fixed[i]:
                    continue

                # 1. attraction force (gradient direction)
                xi = int((px - xmin) / (xmax - xmin) * (w - 1))
                yi = int((py - ymin) / (ymax - ymin) * (h - 1))
                xi = np.clip(xi, 0, w - 1)
                yi = np.clip(yi, 0, h - 1)

                gx = grad_x[yi, xi]
                gy = grad_y[yi, xi]
                norm = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
                gx /= norm
                gy /= norm
                strength = density[yi, xi] ** 2

                dx = gx * dt * (1.0 + 3.0 * strength)
                dy = gy * dt * (1.0 + 3.0 * strength)

                # 2. repulsion force (from neighbors)
                repel_fx, repel_fy = 0.0, 0.0
                neighbors = tree.query_ball_point([px, py], repel_radius)
                for j in neighbors:
                    if i == j:
                        continue
                    dxij = px - moved[j][0]
                    dyij = py - moved[j][1]
                    dist2 = dxij ** 2 + dyij ** 2 + 1e-4
                    repel_fx += dxij / dist2
                    repel_fy += dyij / dist2

                dx += repel_strength * repel_fx
                dy += repel_strength * repel_fy

                newx = px + dx
                newy = py + dy
                pt = Point(newx, newy)
                if self.shape_prep.contains(pt):
                    new_points[i, 0] = newx
                    new_points[i, 1] = newy

            moved = new_points

        self.points = moved

    def relaxation(self, iterations=10):
        points = self.points.copy()
        for _ in range(iterations):
            vor = Voronoi(points)
            new_points = []
            for i, region_idx in enumerate(vor.point_region):
                if self.fixed[i]:
                    new_points.append(points[i])
                    continue
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    new_points.append(points[i])
                    continue
                polygon = np.array([vor.vertices[j] for j in region])
                centroid = polygon.mean(axis=0)
                pt = Point(*centroid)
                if self.shape_prep.contains(pt):
                    new_points.append(centroid)
                else:
                    new_points.append(points[i])
            points = np.array(new_points)
        self.points = points

    def get_points(self):
        return self.points

    def get_mask(self):
        return self.fixed

if __name__ == "__main__":
    # Example usage
    from shapely.wkt import loads as wkt_loads
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, MultiPolygon
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utilities.log import CORELOG

    def plot_shape_with_points(
        shape,
        points=None,
        fixed_mask=None,
        density=None,
        mask_extent=None,
        color="black",
        alpha=0.5
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # 1. draw density map if given
        if density is not None and mask_extent is not None:
            xmin, xmax, ymin, ymax = mask_extent
            extent = [xmin, xmax, ymin, ymax]
            ax.imshow(
                density,
                extent=extent,
                origin="lower",
                cmap="viridis",
                alpha=0.6,
                interpolation="bilinear",
            )

        # 2. draw shape outline
        if isinstance(shape, Polygon):
            patch = plt.Polygon(
                list(shape.exterior.coords),
                closed=True,
                fill=True,
                edgecolor=color,
                alpha=alpha
            )
            ax.add_patch(patch)
            for interior in shape.interiors:
                hole = plt.Polygon(
                    list(interior.coords),
                    closed=True,
                    fill=False,
                    edgecolor="red",
                    linestyle="--"
                )
                ax.add_patch(hole)
        elif isinstance(shape, MultiPolygon):
            for poly in shape.geoms:
                plot_shape_with_points(
                    poly, points, fixed_mask, density, mask_extent,
                    color=color, alpha=alpha
                )

        # 3. draw points
        if points is not None:
            points = np.array(points)
            if fixed_mask is not None:
                points_fixed = points[fixed_mask]
                points_free = points[~fixed_mask]
                ax.plot(points_fixed[:, 0], points_fixed[:, 1], "ro", markersize=2, label="Fixed")
                ax.plot(points_free[:, 0], points_free[:, 1], "bo", markersize=2, label="Free")
            else:
                ax.plot(points[:, 0], points[:, 1], "go", markersize=2, label="Points")

        ax.set_aspect("equal")
        ax.autoscale()
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    path = "/Users/ein/EinDev/OcctStuff/.cache/test00"
    # Load region edge
    edge_path = os.path.join(path, "region_edge.wkt")
    shape = None
    mask = None
    if os.path.exists(edge_path):
        with open(edge_path, "r") as f:
            shape = wkt_loads(f.read())

    mask_path = os.path.join(path, "mask.npz")
    if os.path.exists(mask_path):
        mask = np.load(mask_path)["mask"]

    w, h = 126.46, 54.46
    shape_bounds = shape.bounds
    xmin, ymin, xmax, ymax = shape_bounds
    # example circles
    circles = [
        (11.23, 11.23, 6.00, 'out'),
        (115.23, 43.23, 5.75, 'in'),
    ]

    # Generate density map
    from modules.maps import generate_density_with_mask
    density = generate_density_with_mask(
        size=(w, h),
        valid_mask=mask,
        circles=circles,
        gradient_direction=(-1, 0),
        gradient_strength=0.5,
        falloff=125.0
    )

    points = Points(shape=shape, spacing=3.0, offset_layers=1, logger=CORELOG)
    points.move(density, dt=0.1, mask_extent=(0, w, 0, h), iterations=50)
    points.relaxation(iterations=100)
    points_i = points.get_points()
    fixed = points.get_mask()

    plot_shape_with_points(
        shape=shape,
        points=points_i,
        fixed_mask=fixed,
        density=density,
        mask_extent=[xmin, xmax, ymin, ymax]
    )