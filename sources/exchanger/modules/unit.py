import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
from tqdm import tqdm

class unit:
    """
    Represents a Voronoi cell unit with its geometry, center point, and shap shape.
    """
    def __init__(self, 
                 vertices: list, 
                 index: int, 
                 center: list,
                 color: list = None, 
                 curve_bezier: list = None,
                 vector: list = None,
                 mask = None,
                 mask_extent=None,
                 ):
        """
        Parameters:
        - vertices: list of 2D coordinates (x, y)
        - index: index of the region
        - center: the input point that generated this region
        - color: optional RGB color
        - curve_bezier: optional custom closed curve (list of [x, y]) defining a shape
        """
        self.vertices = vertices
        self.index = index
        self.center = center
        self.color = color if color is not None else [255, 255, 255]
        self.edge = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]
        self.vector = vector

        # curve_bezier is a closed curve
        self.curve_bezier = curve_bezier if curve_bezier is not None else self._generate_curve_bezier()
        self.boundingbox = self._generate_bounding_box()

        # This scale is used to shrink the shape toward the center to create the basic shape
        self.curve_bezier = self._offset_proportional(curve=self.curve_bezier, center=center, factor=0.7)

        # top inner
        self.curve_wire_ti = None
        # top outer
        self.curve_wire_to = None
        # bottom inner
        self.curve_wire_bi = None
        # bottom outer
        self.curve_wire_bo = None

        # side curve to loft
        self.curve_wire_side = None

        # determine valid status based on mask
        self.valid = False
        if mask is not None and self.boundingbox is not None:
            xmin, xmax, ymin, ymax = self.boundingbox
            mx0, mx1, my0, my1 = mask_extent
            h, w = mask.shape

            # map 4 corners of bounding box to mask indices
            corners = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymin),
                (xmax, ymax)
            ]

            all_inside = True
            for x, y in corners:
                u = (x - mx0) / (mx1 - mx0)
                v = (y - my0) / (my1 - my0)
                i = int(u * w)
                j = int(v * h)

                if not (0 <= i < w and 0 <= j < h) or not mask[j, i]:
                    all_inside = False
                    break

            self.valid = all_inside
    
    def _generate_curve_bezier(self, samples_per_segment=30, t=0.1, min_edge_length=1e-4):
        """
        Generate a closed smooth Bézier curve from polygon with collapsed segments for short edges.

        Parameters:
        - samples_per_segment: number of points per Bézier
        - t: control point offset factor toward neighbor (0 ~ 1)
        - min_edge_length: edges shorter than this are collapsed to a single point

        Returns:
        - np.ndarray: smooth closed curve
        """
        import numpy as np

        verts = np.array(self.vertices)
        t = np.clip(t, 0.01, 0.49)
        n = len(verts)
        curve = []

        for i in range(n):
            A = verts[(i - 1) % n]
            B = verts[i]
            C = verts[(i + 1) % n]

            len_ab = np.linalg.norm(B - A)
            len_bc = np.linalg.norm(C - B)

            # Short edge → collapse to B
            is_short = (len_ab < min_edge_length or len_bc < min_edge_length)
            if is_short:
                curve.extend([B.copy()] * samples_per_segment)
                continue

            # Normal Bézier segment
            M0 = (A + B) / 2
            M1 = (B + C) / 2
            CP1 = B + (A - B) * t
            CP2 = B + (C - B) * t

            ts = np.linspace(0, 1, samples_per_segment, endpoint=False)
            for u in ts:
                mu = 1 - u
                pt = (
                    mu**3 * M0 +
                    3 * mu**2 * u * CP1 +
                    3 * mu * u**2 * CP2 +
                    u**3 * M1
                )
                curve.append(pt)

        if curve:
            curve.append(curve[0])  # close loop
            return np.array(curve)
        else:
            return np.array(self.vertices)

    def _generate_bounding_box(self):
        """
        Compute the bounding box of the shap curve.

        Returns:
        - (min_x, max_x, min_y, max_y) as a tuple
        """
        if self.curve_bezier is None or len(self.curve_bezier) < 1:
            return None

        shape = np.array(self.curve_bezier)
        min_x, min_y = np.min(shape, axis=0)
        max_x, max_y = np.max(shape, axis=0)
        return (min_x, max_x, min_y, max_y)

    def _scale_curve(self, curve: np.ndarray, center: np.ndarray, factor: float = 0.9) -> np.ndarray:
        """
        Scale a 2D/3D curve toward or away from the center.

        Parameters:
        - curve: np.ndarray of shape (N, 2) or (N, 3)
        - center: np.ndarray of shape (2,) or (3,)
        - factor: <1.0 shrink, >1.0 expand

        Returns:
        - np.ndarray: scaled curve
        """
        if curve is None or len(curve) < 2:
            return curve
        curve = np.array(curve)
        center = np.array(center)
        relative = curve - center
        return center + factor * relative

    def _offset_proportional(self, curve: np.ndarray, center: np.ndarray, factor: float = 0.9) -> np.ndarray:
        """
        Offset each point proportionally along the center-to-point direction.

        Parameters:
        - curve: np.ndarray of shape (N, 2) or (N, 3)
        - center: np.ndarray of shape (2,) or (3,)
        - factor: <1.0 inward, >1.0 outward

        Returns:
        - np.ndarray: offset curve
        """
        if curve is None or len(curve) < 2:
            return curve
        curve = np.array(curve)
        center = np.array(center)
        vectors = curve - center
        distances = np.linalg.norm(vectors, axis=1, keepdims=True)
        directions = vectors / (distances + 1e-8)
        offset = directions * distances * (factor - 1)
        return curve + offset

    def _offset_fixed(self, curve: np.ndarray, center: np.ndarray, offset_length: float = -1.0) -> np.ndarray:
        """
        Offset each point a fixed distance along the direction from the center.

        Parameters:
        - curve: np.ndarray of shape (N, 2) or (N, 3)
        - center: np.ndarray of shape (2,) or (3,)
        - offset_length: <0 inward, >0 outward

        Returns:
        - np.ndarray: offset curve
        """
        if curve is None or len(curve) < 2:
            return curve
        curve = np.array(curve)
        center = np.array(center)
        vectors = curve - center
        directions = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        offset = directions * offset_length
        return curve + offset

    def _lift_z(self, curve: np.ndarray, offset_z: float = 0.0) -> np.ndarray:
        """
        Lift a 2D curve to 3D by assigning a z-coordinate.

        Parameters:
        - curve: np.ndarray of shape (N, 2) or (N, 3)
        - offset_z: target z value

        Returns:
        - np.ndarray: lifted 3D curve of shape (N, 3)
        """
        if curve is None or len(curve) < 2:
            return curve
        curve = np.array(curve)
        if curve.shape[1] == 3:
            lifted = curve.copy()
            lifted[:, 2] = offset_z
            return lifted
        elif curve.shape[1] == 2:
            z_column = np.full((curve.shape[0], 1), offset_z)
            return np.hstack((curve, z_column))
        else:
            raise ValueError("Curve must be of shape (N, 2) or (N, 3)")

class Units:
    """
    Represents a collection of Voronoi cell units.
    """
    def __init__(self, units=None, vor: Voronoi = None):
        self.units = units if units is not None else []
        self.vor = vor

    @staticmethod
    def generate_from(vor: Voronoi, density=None, bounds=None, mask=None, mask_extent=None, logger=None):
        """
        Static method to generate Units from a Voronoi object.
        Returns a Units instance.
        """
        units_list = []

        # Add tqdm progress bar for unit generation
        for i, region_index in tqdm(
            enumerate(vor.point_region), 
            total=len(vor.point_region), 
            desc="GUS", 
            bar_format="{desc}: |{bar} [{n_fmt:>05}/{total_fmt:>05}]"
        ):
            region = vor.regions[region_index]
            if not region or -1 in region:
                continue

            vertices = [vor.vertices[j] for j in region]
            center = vor.points[i]

            # Sample vector from density if given
            if density is not None and bounds is not None:
                vector = Units.sample_gradient_from_density(density, bounds, center)
            else:
                vector = [0.0, 0.0]

            u = unit(
                vertices=vertices, 
                index=i, 
                center=center, 
                vector=vector, 
                curve_bezier=None, 
                mask=mask, 
                mask_extent=mask_extent
            )
            units_list.append(u)

        return Units(units_list, vor)
    
    @staticmethod
    def sample_gradient_from_density(density, bounds, point):
        """
        Sample normalized gradient vector at a given 2D point in world coordinates.
        
        Parameters:
        - density: 2D numpy array
        - bounds: [xmin, xmax, ymin, ymax]
        - point: [x, y] world-space coordinate

        Returns:
        - np.ndarray of shape (2,) representing the gradient vector
        """
        import numpy as np

        xmin, xmax, ymin, ymax = bounds
        h, w = density.shape
        x, y = point

        # Map point to image coordinates
        px = int((x - xmin) / (xmax - xmin) * (w - 1))
        py = int((y - ymin) / (ymax - ymin) * (h - 1))

        px = np.clip(px, 1, w - 2)
        py = np.clip(py, 1, h - 2)

        # Central differences
        dx = (density[py, px + 1] - density[py, px - 1]) * 0.5
        dy = (density[py + 1, px] - density[py - 1, px]) * 0.5

        grad = np.array([dx, dy], dtype=np.float64)
        norm = np.linalg.norm(grad)
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        return grad / norm

    def draw(self, ax, 
            draw_index=True, 
            edge_color='black',
            draw_shap=True, 
            fill_shap=True,
            draw_vector=False, 
            vector_scale=0.05, 
            vector_color='red',
            density_map=None,
            bounds=None,
            offset=(0.0, 0.0), 
        ):
        """
        Draw all units on the given matplotlib Axes.

        Parameters:
        - ax: matplotlib Axes
        - draw_index: whether to draw the index label at center
        - edge_color: color of the polygon edge
        - draw_shap: whether to draw the shap curve
        - fill_shap: whether to fill the shap shape with translucent color
        - draw_vector: whether to draw the unit vector as an arrow
        - vector_scale: scaling factor for vector arrow length
        - vector_color: color of the vector arrow
        - density_map: optional 2D numpy array for background density
        - offset: tuple of (x, y) to shift everything in the drawing
        """
        ox, oy = offset  # <== unfold offset

        # --- draw Voronoi edges ---
        voronoi_plot_2d(
            self.vor,
            ax=ax,
            show_vertices=False,
            line_colors=edge_color,
            line_width=0.3,
            point_size=0.5
        )

        shap_polys = []
        shap_colors = []
        shap_lines = []

        for u in self.units:
            # --- draw index ---
            if draw_index:
                ax.text(u.center[0] + ox, u.center[1] + oy, str(u.index),
                        color='blue', fontsize=8, ha='center', va='center')  # <== 加偏移

            # --- draw vector ---
            if draw_vector and hasattr(u, "vector"):
                start = u.center
                length = np.linalg.norm(u.vector)
                if length < 1e-8:
                    continue
                scaled_vec = (u.vector / length) * (length * vector_scale)
                vec = np.array(u.vector) * scaled_vec
                ax.arrow(
                    start[0] + ox, start[1] + oy, vec[0], vec[1],  # <== 加偏移
                    head_width=0.5, head_length=0.5,
                    fc=vector_color, ec=vector_color, linewidth=1
                )

            # --- draw shap ---
            if draw_shap and u.curve_bezier is not None and len(u.curve_bezier) > 2:
                points = np.array(u.curve_bezier)
                points += np.array(offset)  # <== 整体偏移

                if fill_shap:
                    poly = Polygon(
                        points,
                        closed=True,
                        facecolor=[c / 255 for c in u.color] + [0.4],
                        edgecolor='none'
                    )
                    ax.add_patch(poly)
                else:
                    xs, ys = np.vstack([points, points[0]]).T
                    ax.plot(xs, ys, color='red', linestyle='-', linewidth=1.0)

        # --- batch fallback（keep offset parameters） ---
        if fill_shap and shap_polys:
            collection = PolyCollection(
                shap_polys,
                facecolors=shap_colors,
                edgecolors='none'
            )
            ax.add_collection(collection)

        if not fill_shap and shap_lines:
            line_coll = LineCollection(shap_lines, colors='orange', linewidths=1.0)
            ax.add_collection(line_coll)

        # --- draw density map ---
        if density_map is not None and bounds is not None:
            ax.imshow(density_map, cmap='viridis', extent=bounds, origin='lower')

if __name__ == "__main__":
    # Example usage
    pass