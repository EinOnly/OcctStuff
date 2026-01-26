import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
from tqdm import tqdm


class unit:
    """
    Represents a Voronoi cell unit with its geometry, center point, and shap shape.
    """

    def __init__(
        self,
        vertices: list,
        index: int,
        center: list,
        color: list = None,
        curve_bezier: list = None,
        vector: list = None,
        scale_perp: float = 0.5,
        mask=None,
        mask_extent=None,
    ):
        """
        Parameters:
        - vertices: list of 2D coordinates (x, y)
        - index: index of the region
        - center: the input point that generated this region
        - color: optional RGB color
        - curve_bezier: optional custom closed curve (list of [x, y]) defining a shape
        - vector: direction vector for water droplet transformation
        - scale_perp: droplet effect intensity (0.0=no effect, 1.0=strong teardrop, default 0.5)
        """
        self.vertices = vertices
        self.index = index
        self.center = center
        self.color = color if color is not None else [255, 255, 255]
        self.edge = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]
        self.vector = vector

        # curve_bezier is a closed curve
        self.curve_bezier = (
            curve_bezier if curve_bezier is not None else self._generate_curve_bezier()
        )
        self.boundingbox = self._generate_bounding_box()

        # This scale is used to shrink the shape toward the center to create the basic shape
        self.curve_bezier = self._offset_proportional(
            curve=self.curve_bezier, center=center, factor=0.7
        )

        # Apply anisotropic affine transformation based on vector
        if vector is not None and np.linalg.norm(vector) > 1e-8:
            self.curve_bezier = self._affine_scale_perpendicular(
                curve=self.curve_bezier,
                center=center,
                vector=vector,
                scale_perp=scale_perp
            )

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
            corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

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

    def _generate_curve_bezier(
        self, samples_per_segment=30, t=0.1, min_edge_length=1e-4
    ):
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
            is_short = len_ab < min_edge_length or len_bc < min_edge_length
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
                pt = mu**3 * M0 + 3 * mu**2 * u * CP1 + 3 * mu * u**2 * CP2 + u**3 * M1
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

    def _scale_curve(
        self, curve: np.ndarray, center: np.ndarray, factor: float = 0.9
    ) -> np.ndarray:
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

    def _offset_proportional(
        self, curve: np.ndarray, center: np.ndarray, factor: float = 0.9
    ) -> np.ndarray:
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

    def _offset_fixed(
        self, curve: np.ndarray, center: np.ndarray, offset_length: float = -1.0
    ) -> np.ndarray:
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

    def _affine_scale_perpendicular(
        self, curve: np.ndarray, center: np.ndarray, vector: np.ndarray, scale_perp: float = 0.5
    ) -> np.ndarray:
        """
        Apply water droplet transformation: create a teardrop shape with the pointed end 
        facing the vector direction.

        Parameters:
        - curve: np.ndarray of shape (N, 2) or (N, 3)
        - center: np.ndarray of shape (2,) or (3,) - the center point for transformation
        - vector: np.ndarray of shape (2,) - the direction vector (will be normalized)
        - scale_perp: intensity of droplet effect (0.0=no effect, 1.0=strong tapering)

        Returns:
        - np.ndarray: transformed curve with water droplet shape
        """
        if curve is None or len(curve) < 2:
            return curve
        if vector is None or np.linalg.norm(vector) < 1e-8:
            return curve

        curve = np.array(curve)
        center = np.array(center)
        vector = np.array(vector)

        # Normalize the vector
        vec_norm = vector / np.linalg.norm(vector)

        # Get the perpendicular vector (2D rotation by 90 degrees)
        perp_vec = np.array([-vec_norm[1], vec_norm[0]])

        # Translate curve to origin (relative to center)
        relative = curve[:, :2] - center[:2]

        # Decompose each point into parallel and perpendicular components
        parallel_proj = np.dot(relative, vec_norm)[:, np.newaxis]  # scalar projection along vector
        perp_component = np.dot(relative, perp_vec)[:, np.newaxis] * perp_vec  # perpendicular component
        
        # Create water droplet effect:
        # - Points in front (positive projection) taper towards the tip
        # - Points behind (negative projection) remain rounded
        
        # Normalize parallel projection to [-1, 1] range for smooth scaling
        max_proj = np.max(np.abs(parallel_proj)) + 1e-8
        normalized_proj = parallel_proj / max_proj  # range: [-1, 1]
        
        # Create tapering function:
        # For positive projection (front): gradually taper to point
        # For negative projection (back): keep rounded
        taper_scale = np.ones_like(normalized_proj)
        
        for i, proj in enumerate(normalized_proj):
            if proj > 0:  # Front side (towards vector direction)
                # Exponential taper: stronger as we move forward
                # proj=0 -> scale=1.0, proj=1 -> scale=0.0 (point)
                taper_factor = 1.0 - scale_perp * (proj ** 1.5)  # Use power for smooth tapering
                taper_scale[i] = max(0.01, taper_factor)  # Minimum scale to avoid singularity
            else:  # Back side (opposite to vector direction)
                # Slight expansion for rounded back
                taper_scale[i] = 1.0 + scale_perp * 0.2 * abs(proj)
        
        # Apply tapering to perpendicular component only
        # This creates the droplet shape by narrowing width as we move forward
        scaled_perp = perp_component * taper_scale
        
        # Reconstruct the relative position
        parallel_component = parallel_proj * vec_norm
        scaled_relative = parallel_component + scaled_perp
        
        # Translate back
        result = scaled_relative + center[:2]

        # Preserve z-coordinate if curve is 3D
        if curve.shape[1] == 3:
            result = np.hstack((result, curve[:, 2:3]))

        return result


class Units:
    """
    Represents a collection of Voronoi cell units.
    """

    def __init__(self, units=None, vor: Voronoi = None):
        self.units = units if units is not None else []
        self.vor = vor

    @staticmethod
    def find_neighbors(vor: Voronoi, point_idx: int):
        """
        Find neighboring point indices for a given point in Voronoi diagram.
        
        Parameters:
        - vor: Voronoi object
        - point_idx: index of the point
        
        Returns:
        - set of neighboring point indices
        """
        neighbors = set()
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        
        if not region or -1 in region:
            return neighbors
        
        # Find all regions that share vertices with this region
        for vertex_idx in region:
            # Find all regions containing this vertex
            for i, other_region_idx in enumerate(vor.point_region):
                if i == point_idx:
                    continue
                other_region = vor.regions[other_region_idx]
                if vertex_idx in other_region and -1 not in other_region:
                    neighbors.add(i)
        
        return neighbors

    @staticmethod
    def merge_cells(vertices1, vertices2, center1, center2, waist_factor=0.7):
        """
        Merge two adjacent cells by combining their vertices in correct order to create a peanut-like shape.
        
        This method preserves all original vertices, applies waist effect, and sorts vertices by angle
        around the merged center to ensure correct ordering and avoid twisted curves.
        
        Parameters:
        - vertices1, vertices2: lists of vertices for each cell (in CCW order)
        - center1, center2: centers of each cell
        - waist_factor: factor (0.0-1.0) controlling the "waist" tightness at connection
                       (0.0 = very tight waist, 1.0 = no waist effect)
        
        Returns:
        - merged_vertices: list of vertices for merged cell in correct CCW order
        - merged_center: center of merged cell
        """
        import numpy as np
        from scipy.spatial import distance_matrix
        
        vertices1 = np.array(vertices1)
        vertices2 = np.array(vertices2)
        center1 = np.array(center1)
        center2 = np.array(center2)
        
        # Compute midpoint between two centers (this will be the new center)
        midpoint = (center1 + center2) / 2
        
        # Direction vector from center1 to center2
        direction = center2 - center1
        dist = np.linalg.norm(direction)
        
        if dist < 1e-8:
            # Centers are too close, just combine all vertices and sort by angle
            all_vertices = np.vstack([vertices1, vertices2])
            angles = np.arctan2(all_vertices[:, 1] - midpoint[1], 
                              all_vertices[:, 0] - midpoint[0])
            sorted_indices = np.argsort(angles)
            merged_vertices = all_vertices[sorted_indices].tolist()
            merged_center = midpoint.tolist()
            return merged_vertices, merged_center
        
        direction = direction / dist
        
        # Find shared vertices (vertices that are very close between two cells)
        dist_matrix = distance_matrix(vertices1, vertices2)
        threshold = 1e-2  # vertices closer than this are considered shared
        shared_pairs = np.argwhere(dist_matrix < threshold)
        
        shared_v1_indices = set(shared_pairs[:, 0]) if len(shared_pairs) > 0 else set()
        shared_v2_indices = set(shared_pairs[:, 1]) if len(shared_pairs) > 0 else set()
        
        # Apply waist effect to all vertices
        def apply_waist_effect(vertex, cell_center, direction_sign):
            """Apply waist shrinking effect to a vertex"""
            v_rel = vertex - cell_center
            projection = np.dot(v_rel, direction * direction_sign)
            
            # Apply waist effect if on the side facing the other cell
            if projection > 0:
                # Distance to midpoint along direction
                progress = projection / (dist / 2)
                progress = np.clip(progress, 0.0, 1.0)
                
                # Apply smooth waist effect (stronger near midpoint)
                waist_strength = 1.0 - waist_factor
                scale = 1.0 - waist_strength * (1.0 - abs(1.0 - progress))
                
                # Shrink vertex towards the line connecting centers
                point_on_line = cell_center + direction * direction_sign * projection
                v_processed = point_on_line + (vertex - point_on_line) * scale
                return v_processed
            else:
                return vertex
        
        # Process vertices from cell 1 (skip shared ones)
        processed_vertices = []
        
        for i, v in enumerate(vertices1):
            if i not in shared_v1_indices:  # Skip shared vertices
                v_processed = apply_waist_effect(v, center1, 1.0)
                processed_vertices.append(v_processed)
        
        # Process vertices from cell 2 (skip shared ones)
        for i, v in enumerate(vertices2):
            if i not in shared_v2_indices:  # Skip shared vertices
                v_processed = apply_waist_effect(v, center2, -1.0)
                processed_vertices.append(v_processed)
        
        # Convert to numpy array
        processed_vertices = np.array(processed_vertices)
        
        # Sort all vertices by angle around the merged center to ensure CCW order
        angles = np.arctan2(processed_vertices[:, 1] - midpoint[1], 
                          processed_vertices[:, 0] - midpoint[0])
        sorted_indices = np.argsort(angles)
        merged_vertices = processed_vertices[sorted_indices].tolist()
        
        # Compute merged center
        merged_center = midpoint.tolist()
        
        return merged_vertices, merged_center

    @staticmethod
    def generate_from(
        vor: Voronoi,
        density=None,
        bounds=None,
        mask=None,
        mask_extent=None,
        scale_perp=0.5,
        merge_probability=0.0,
        merge_waist_factor=0.7,
        max_merge_neighbors=1,
        logger=None,
    ):
        """
        Static method to generate Units from a Voronoi object.
        Returns a Units instance.

        Parameters:
        - vor: Voronoi object
        - density: optional density map for gradient-based vector sampling
        - bounds: bounds for density map [xmin, xmax, ymin, ymax]
        - mask: optional mask for valid region
        - mask_extent: extent of mask
        - scale_perp: droplet effect intensity (0.0=no effect, 1.0=strong teardrop, default 0.5)
        - merge_probability: probability (0.0-1.0) to merge adjacent cells (default 0.0)
        - merge_waist_factor: waist tightness for merged cells (0.0=tight, 1.0=no waist, default 0.7)
        - max_merge_neighbors: max number of neighbors a cell can merge with (default 1, creates branches when >1)
        - logger: optional logger
        """
        import numpy as np
        import random
        
        units_list = []
        merged_indices = set()  # Track which cells have been merged
        
        # Build neighbor map
        neighbor_map = {}
        for i in range(len(vor.points)):
            neighbor_map[i] = Units.find_neighbors(vor, i)
        
        # Add tqdm progress bar for unit generation
        for i, region_index in tqdm(
            enumerate(vor.point_region),
            total=len(vor.point_region),
            desc="GUS",
            bar_format="{desc}: |{bar} [{n_fmt:>05}/{total_fmt:>05}]",
        ):
            # Skip if already merged
            if i in merged_indices:
                continue
                
            region = vor.regions[region_index]
            if not region or -1 in region:
                continue

            vertices = [vor.vertices[j] for j in region]
            center = vor.points[i]
            
            # Try to merge with multiple neighbors to create branch-like structures
            should_merge = random.random() < merge_probability
            merge_count = 0
            merged_neighbors = []
            
            if should_merge and i in neighbor_map:
                # Get valid neighbors (not already merged)
                valid_neighbors = [n for n in neighbor_map[i] if n not in merged_indices and n > i]
                
                if valid_neighbors:
                    # Determine how many neighbors to merge (up to max_merge_neighbors)
                    num_to_merge = min(max_merge_neighbors, len(valid_neighbors))
                    
                    # Randomly select neighbors to merge
                    neighbors_to_merge = random.sample(valid_neighbors, num_to_merge)
                    
                    # Merge with each selected neighbor sequentially
                    for neighbor_idx in neighbors_to_merge:
                        neighbor_region_idx = vor.point_region[neighbor_idx]
                        neighbor_region = vor.regions[neighbor_region_idx]
                        
                        if neighbor_region and -1 not in neighbor_region:
                            neighbor_vertices = [vor.vertices[j] for j in neighbor_region]
                            neighbor_center = vor.points[neighbor_idx]
                            
                            # Merge the cells (accumulate vertices)
                            vertices, center = Units.merge_cells(
                                vertices, neighbor_vertices, center, neighbor_center,
                                waist_factor=merge_waist_factor
                            )
                            merged_indices.add(neighbor_idx)
                            merged_neighbors.append(neighbor_idx)
                            merge_count += 1
                    
                    if logger and merge_count > 0:
                        neighbor_str = ", ".join(map(str, merged_neighbors))
                        logger.info(f"   Merged cell {i} with {merge_count} neighbor(s): [{neighbor_str}]")

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
                scale_perp=scale_perp,
                curve_bezier=None,
                mask=mask,
                mask_extent=mask_extent,
            )
            units_list.append(u)

        return Units(units_list, vor)

    @staticmethod
    def visualize_generation(
        vor,
        units,
        shape=None,
        density=None,
        bounds=None,
        title="Units Generation Result"
    ):
        """
        Visualize the result of Units.generate_from()

        Parameters:
        - vor: Voronoi diagram
        - units: Units object or list of unit objects
        - shape: optional shapely Polygon for boundary
        - density: optional density map
        - bounds: [xmin, xmax, ymin, ymax]
        - title: plot title
        """
        import matplotlib.pyplot as plt
        from scipy.spatial import voronoi_plot_2d
        from matplotlib.patches import Polygon as MplPolygon

        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

        # 1. Draw density map if provided
        if density is not None and bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            ax.imshow(
                density,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap="viridis",
                alpha=0.3,
                interpolation="bilinear"
            )

        # 2. Draw shape boundary if provided
        if shape is not None:
            from shapely.geometry import Polygon
            if isinstance(shape, Polygon):
                patch = MplPolygon(
                    list(shape.exterior.coords),
                    closed=True,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                    linestyle="--"
                )
                ax.add_patch(patch)
                for interior in shape.interiors:
                    hole = MplPolygon(
                        list(interior.coords),
                        closed=True,
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                        linestyle="--"
                    )
                    ax.add_patch(hole)

        # 3. Draw Voronoi diagram
        if vor is not None:
            voronoi_plot_2d(
                vor,
                ax=ax,
                show_vertices=False,
                line_colors="blue",
                line_width=0.5,
                line_alpha=0.6,
                point_size=0
            )

        # 4. Draw units
        units_list = units.units if hasattr(units, 'units') else units
        valid_count = 0
        invalid_count = 0

        for u in units_list:
            # Draw center point
            color = "green" if u.valid else "gray"
            marker = "o" if u.valid else "x"
            ax.plot(u.center[0], u.center[1], marker,
                   color=color, markersize=4, alpha=0.8)

            # Draw curve_bezier if exists
            if u.curve_bezier is not None and len(u.curve_bezier) > 2:
                points = np.array(u.curve_bezier)
                line_color = "red" if u.valid else "lightgray"
                ax.plot(points[:, 0], points[:, 1],
                       color=line_color, linewidth=1.0, alpha=0.7)

            # Draw vector if exists
            if hasattr(u, 'vector') and u.vector is not None:
                vec_norm = np.linalg.norm(u.vector)
                if vec_norm > 1e-8:
                    vec_scaled = np.array(u.vector) * 2.0  # Scale for visibility
                    ax.arrow(
                        u.center[0], u.center[1],
                        vec_scaled[0], vec_scaled[1],
                        head_width=0.3, head_length=0.3,
                        fc="orange", ec="orange",
                        alpha=0.6, linewidth=0.5
                    )

            if u.valid:
                valid_count += 1
            else:
                invalid_count += 1

        # 5. Set limits and labels
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            ax.set_xlim(xmin - 1, xmax + 1)
            ax.set_ylim(ymin - 1, ymax + 1)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{title}\nValid: {valid_count}, Invalid: {invalid_count}", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # 6. Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label=f'Valid units ({valid_count})'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='gray',
                   markersize=8, label=f'Invalid units ({invalid_count})'),
            Line2D([0], [0], color='red', linewidth=2, label='Unit shapes'),
            Line2D([0], [0], color='orange', linewidth=2, label='Density vectors'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig, ax

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

    def draw(
        self,
        ax,
        draw_index=True,
        edge_color="black",
        draw_shap=True,
        fill_shap=True,
        draw_vector=False,
        vector_scale=0.05,
        vector_color="red",
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
            point_size=0.5,
        )

        shap_polys = []
        shap_colors = []
        shap_lines = []

        for u in self.units:
            # --- draw index ---
            if draw_index:
                ax.text(
                    u.center[0] + ox,
                    u.center[1] + oy,
                    str(u.index),
                    color="blue",
                    fontsize=8,
                    ha="center",
                    va="center",
                )  # <== 加偏移

            # --- draw vector ---
            if draw_vector and hasattr(u, "vector"):
                start = u.center
                length = np.linalg.norm(u.vector)
                if length < 1e-8:
                    continue
                scaled_vec = (u.vector / length) * (length * vector_scale)
                vec = np.array(u.vector) * scaled_vec
                ax.arrow(
                    start[0] + ox,
                    start[1] + oy,
                    vec[0],
                    vec[1],  # <== 加偏移
                    head_width=0.5,
                    head_length=0.5,
                    fc=vector_color,
                    ec=vector_color,
                    linewidth=1,
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
                        edgecolor="none",
                    )
                    ax.add_patch(poly)
                else:
                    xs, ys = np.vstack([points, points[0]]).T
                    ax.plot(xs, ys, color="red", linestyle="-", linewidth=1.0)

        # --- batch fallback（keep offset parameters） ---
        if fill_shap and shap_polys:
            collection = PolyCollection(
                shap_polys, facecolors=shap_colors, edgecolors="none"
            )
            ax.add_collection(collection)

        if not fill_shap and shap_lines:
            line_coll = LineCollection(shap_lines, colors="orange", linewidths=1.0)
            ax.add_collection(line_coll)

        # --- draw density map ---
        if density_map is not None and bounds is not None:
            ax.imshow(density_map, cmap="viridis", extent=bounds, origin="lower")

def visualize_cells_from_plate(
    valid_region_edge,
    valid_region_mask,
    bottom_width,
    bottom_height,
    cell_region_offset,
    density_map,
    spacing=3.0,
    move_iterations=1,
    num_relaxations=6,
    merge_probability=0.0,
    merge_waist_factor=0.7,
    max_merge_neighbors=1,
    logger=None
):
    """
    Generate cells and visualize cells, curves, and vectors.
    
    Parameters:
    - valid_region_edge: Shapely Polygon representing the valid region boundary
    - valid_region_mask: numpy array mask for valid region
    - bottom_width: plate width
    - bottom_height: plate height
    - cell_region_offset: offset from edges
    - density_map: density map for point movement
    - spacing: initial point spacing
    - move_iterations: number of density-based movement iterations
    - num_relaxations: number of Lloyd relaxation iterations
    - merge_probability: probability (0.0-1.0) to merge adjacent cells (default 0.0)
    - merge_waist_factor: waist tightness for merged cells (0.0=tight, 1.0=no waist, default 0.7)
    - max_merge_neighbors: max number of neighbors a cell can merge with (default 1, >1 creates branches)
    - logger: logger object for output
    
    Returns:
    - cells_A: Units object containing generated cells
    - points: final point positions
    - fig, ax: matplotlib figure and axis for visualization
    """
    from points import Points
    from pattern import relaxed_voronoi
    import matplotlib.pyplot as plt
    
    if logger:
        logger.warn("Starting cell generation and visualization...")
    
    # Define bounds
    bounds = [
        cell_region_offset, 
        bottom_width - cell_region_offset, 
        cell_region_offset, 
        bottom_height - cell_region_offset
    ]
    
    # Generate points
    if logger:
        logger.warn("Generating initial points...")
    points_obj = Points(
        shape=valid_region_edge, 
        spacing=spacing, 
        offset_layers=1, 
        logger=logger
    )
    
    # Move points by density
    if logger:
        logger.warn(f"Moving points by density: {move_iterations} times...")
    points_obj.move(
        density_map, 
        dt=0.1, 
        mask_extent=(0, bottom_width, 0, bottom_height), 
        iterations=move_iterations
    )
    
    # Apply Lloyd relaxation
    if logger:
        logger.warn("Applying Lloyd relaxation...")
    points_obj.relaxation(iterations=num_relaxations)
    points = points_obj.get_points()
    
    # Compute Voronoi diagram
    if logger:
        logger.warn("Computing Voronoi diagram...")
    _, vor_a, _, points_b = relaxed_voronoi(
        points,
        bounds=bounds,
        iterations=10
    )
    
    # Generate cells
    if logger:
        logger.warn("Generating Voronoi units...")
    cells_A = Units.generate_from(
        vor_a, 
        density=density_map, 
        bounds=bounds, 
        mask=valid_region_mask, 
        mask_extent=[0, bottom_width, 0, bottom_height],
        scale_perp=0.7,  # 0.7 = strong water droplet effect (0.0=none, 1.0=extreme)
        merge_probability=merge_probability,
        merge_waist_factor=merge_waist_factor,
        max_merge_neighbors=max_merge_neighbors,
        logger=logger
    )
    
    # Create visualization
    if logger:
        logger.warn("Creating visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw density map
    ax.imshow(density_map, extent=[0, bottom_width, 0, bottom_height], 
              origin='lower', cmap='viridis', alpha=0.3)
    
    # Draw cells, curves, and vectors
    cells_A.draw(
        ax, 
        draw_index=True, 
        draw_shap=True, 
        fill_shap=False, 
        draw_vector=True, 
        vector_scale=3,
        density_map=density_map,
        bounds=[0, bottom_width, 0, bottom_height],
        offset=(0.0, 0.0)
    )
    
    # Draw valid region boundary
    if valid_region_edge:
        from shapely.geometry import Polygon
        geoms = [valid_region_edge] if isinstance(valid_region_edge, Polygon) else valid_region_edge.geoms
        for geom in geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='yellow', linewidth=2, linestyle='-', label='Valid Region')
    
    ax.set_xlim(bounds[0]-1, bounds[1]+1)
    ax.set_ylim(bounds[2]-1, bounds[3]+1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Cells Visualization\nTotal: {len(cells_A.units)} cells, Valid: {sum(1 for u in cells_A.units if u.valid)}', 
                 fontsize=14)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='blue', label='Cell Boundaries'),
        Patch(facecolor='none', edgecolor='red', label='Bezier Curves'),
        Patch(facecolor='none', edgecolor='yellow', linewidth=2, label='Valid Region'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if logger:
        logger.info(f"✅ Visualization complete: {len(cells_A.units)} cells generated")
        logger.info(f"   Valid cells: {sum(1 for u in cells_A.units if u.valid)}")
    
    return cells_A, points, fig, ax

if __name__ == "__main__":
    """
    Test visualize_cells_from_plate function with plate-like geometry.
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, Point
    import sys
    sys.path.append('..')
    
    print("="*60)
    print("Cell Generation and Visualization Test")
    print("="*60)

    # 1. Define plate-like region (similar to plates.py)
    print("\n1. Defining plate region...")
    bottom_width = 126.46
    bottom_height = 54.46
    bottom_fillet = 11.00
    cell_region_offset = 1.00
    
    # Create rounded rectangle points
    def rounded_rect_points(cx, cy, w, h, r, segments=8):
        """Generate points for a rounded rectangle."""
        points = []
        r = min(r, w / 2, h / 2)
        corner_centers = [
            (cx + w - r, cy + r),        # Bottom-right
            (cx + w - r, cy + h - r),    # Top-right
            (cx + r,     cy + h - r),    # Top-left
            (cx + r,     cy + r),        # Bottom-left
        ]
        angles = [
            (270, 360),  # bottom-right
            (0, 90),     # top-right
            (90, 180),   # top-left
            (180, 270),  # bottom-left
        ]
        for (ccx, ccy), (start, end) in zip(corner_centers, angles):
            theta = np.linspace(np.radians(start), np.radians(end), segments)
            for t in theta:
                x = ccx + r * np.cos(t)
                y = ccy + r * np.sin(t)
                points.append((x, y))
        points.append(points[0])
        return points
    
    # Create outer boundary (shrunk by offset)
    outer_points = rounded_rect_points(
        cx=cell_region_offset,
        cy=cell_region_offset,
        w=bottom_width - 2 * cell_region_offset,
        h=bottom_height - 2 * cell_region_offset,
        r=max(0, bottom_fillet - cell_region_offset),
        segments=8
    )
    
    # Define holes (like in plates.py)
    holes_def = [
        {"location": [11.23, 11.23], "radius": 5.75, "offset": 2.00},
        {"location": [115.23, 43.23], "radius": 5.75, "offset": 2.00},
        {"location": [115.23, 11.23], "radius": 6.00, "offset": 2.00},
        {"location": [11.23, 43.23], "radius": 6.00, "offset": 2.00},
    ]
    
    # Create hole polygons
    hole_polygons = []
    for hole in holes_def:
        x, y = hole["location"]
        rad = hole["radius"] + hole["offset"]
        circle = Point(x, y).buffer(rad, resolution=32)
        hole_polygons.append(circle)
    
    # Create valid region (outer polygon minus holes)
    from shapely.ops import unary_union
    valid_region_edge = Polygon(shell=outer_points)
    full_holes = unary_union(hole_polygons)
    valid_region_edge = valid_region_edge.difference(full_holes)
    
    print(f"   Plate size: {bottom_width} x {bottom_height} mm")
    print(f"   Fillet radius: {bottom_fillet} mm")
    print(f"   Holes: {len(holes_def)}")

    # 2. Create valid mask from region
    print("\n2. Creating valid region mask...")
    resolution = 300
    short = min(bottom_width, bottom_height)
    px_size = short / resolution
    nx = int(np.ceil(bottom_width / px_size))
    ny = int(np.ceil(bottom_height / px_size))
    
    xs = (np.arange(nx) + 0.5) * px_size
    ys = (np.arange(ny) + 0.5) * px_size
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    
    from shapely.prepared import prep
    prepared_polygon = prep(valid_region_edge)
    mask = np.array([prepared_polygon.contains(Point(x, y)) for x, y in pts], dtype=bool)
    valid_region_mask = mask.reshape((ny, nx))
    
    print(f"   Mask shape: {valid_region_mask.shape}")
    print(f"   Valid pixels: {np.sum(valid_region_mask)} / {valid_region_mask.size}")

    # 3. Generate density map
    print("\n3. Generating density map...")
    from maps import generate_density_with_mask
    
    circles = [
        (11.23, 11.23, 6.00, 'out'),
        (115.23, 43.23, 5.75, 'in'),
    ]
    
    density_map = generate_density_with_mask(
        size=(bottom_width, bottom_height),
        valid_mask=valid_region_mask,
        circles=circles,
        gradient_direction=(-1, 0),
        gradient_strength=0.58,
        falloff=125.0
    )
    print(f"   Density shape: {density_map.shape}")

    # 4. Create simple logger
    class SimpleLogger:
        def warn(self, msg):
            print(f"[WARN] {msg}")
        def info(self, msg):
            print(f"[INFO] {msg}")
    
    logger = SimpleLogger()

    # 5. Call visualize_cells_from_plate
    print("\n4. Generating and visualizing cells...")
    cells_A, points, fig, ax = visualize_cells_from_plate(
        valid_region_edge=valid_region_edge,
        valid_region_mask=valid_region_mask,
        bottom_width=bottom_width,
        bottom_height=bottom_height,
        cell_region_offset=cell_region_offset,
        density_map=density_map,
        spacing=3.0,
        move_iterations=1,
        num_relaxations=6,
        merge_probability=0.0,  # 20% chance to merge adjacent cells
        merge_waist_factor=0.6,  # 0.6 = moderate waist tightness (0.0=tight, 1.0=no waist)
        max_merge_neighbors=3,   # Each cell can merge with up to 3 neighbors (creates tree branches!)
        logger=logger
    )

    # 6. Save and display
    output_path = "/Users/ein/EinDev/OcctStuff/.cache/unit_cells_visualization.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    print(f"\n{'='*60}")
    print(f"✓ Saved visualization to: {output_path}")
    print(f"{'='*60}")
    print(f"  Total cells: {len(cells_A.units)}")
    print(f"  Valid cells: {sum(1 for u in cells_A.units if u.valid)}")
    print(f"  Points generated: {len(points)}")
    print(f"{'='*60}\n")
    
    plt.show()