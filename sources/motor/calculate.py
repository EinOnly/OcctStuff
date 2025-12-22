import numpy as np

class Calculate:
    
    @classmethod
    def Offset(cls, points: np.ndarray, distance: float) -> np.ndarray:
        """
        Offset an open polyline inward by a given distance.
        The "inward" side is determined from the signed area of the closed polygon.
        """
        pts = np.asarray(points, dtype=np.float64)
        n = pts.shape[0]
        if n < 2:
            raise ValueError("Need at least two points to offset a polyline")

        # Compute polygon signed area using closed polygon (avoid vstack)
        # Positive => CCW, Negative => CW
        # Cross product: pts[i].x * pts[i+1].y - pts[i+1].x * pts[i].y
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * (np.sum(x[:-1] * y[1:]) - np.sum(x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])

        # Compute segment directions (optimized length calculation)
        seg_vecs = pts[1:] - pts[:-1]
        # Use faster sqrt(dx^2 + dy^2) instead of np.linalg.norm
        seg_len_sq = seg_vecs[:, 0]**2 + seg_vecs[:, 1]**2
        if np.any(seg_len_sq == 0):
            raise ValueError("Zero length segment in polyline")

        seg_len = np.sqrt(seg_len_sq)[:, np.newaxis]
        seg_dirs = seg_vecs / seg_len

        # Compute inward normals directly (left_normal = (-dy, dx))
        # If area > 0 (CCW) => inward is -left_normal = (dy, -dx)
        # If area < 0 (CW)  => inward is +left_normal = (-dy, dx)
        inward_sign = -1.0 if area > 0 else 1.0
        inward_normals = np.column_stack([
            inward_sign * -seg_dirs[:, 1],
            inward_sign * seg_dirs[:, 0]
        ])

        # Allocate output array
        new_pts = np.empty_like(pts)

        # First and last points: simple offset
        new_pts[0] = pts[0] + inward_normals[0] * distance
        new_pts[-1] = pts[-1] + inward_normals[-1] * distance

        if n > 2:
            # Vectorized interior points: intersection of adjacent offset lines
            offset_dist = inward_normals * distance

            # Offset points from adjacent segments
            p1s = pts[1:-1] + offset_dist[:-1]
            p2s = pts[1:-1] + offset_dist[1:]

            # Segment directions
            d1s = seg_dirs[:-1]
            d2s = seg_dirs[1:]

            # Compute line intersections: det = d1.x * d2.y - d1.y * d2.x
            dets = d1s[:, 0] * d2s[:, 1] - d1s[:, 1] * d2s[:, 0]

            # Detect parallel segments before division
            parallel_mask = np.abs(dets) < 1e-12
            safe_dets = np.where(parallel_mask, 1.0, dets)

            # Solve for intersection parameter: t = (delta.x * d2.y - delta.y * d2.x) / det
            delta = p2s - p1s
            ts = (delta[:, 0] * d2s[:, 1] - delta[:, 1] * d2s[:, 0]) / safe_dets

            # Compute intersections: p1 + t * d1
            new_pts[1:-1] = p1s + ts[:, np.newaxis] * d1s

            # Fix parallel cases with midpoint
            if np.any(parallel_mask):
                new_pts[1:-1][parallel_mask] = 0.5 * (p1s[parallel_mask] + p2s[parallel_mask])

        return new_pts
    
    @classmethod
    def Clamp(cls, points: np.ndarray, x_limit: float) -> np.ndarray:
        """
        Clamp a monotonic-x polyline against vertical line x = x_limit.

        Rules:
        - Points are monotonic in x (either increasing or decreasing).
        - Find the segment AB that straddles x_limit using binary search.
        - Compute intersection point P between AB and x = x_limit.
        - Keep the part on the left side of x_limit (x <= x_limit) plus P.
        For decreasing-x input, this means keeping P and all points with x <= x_limit,
        but the original order is preserved.
        """
        pts = np.asarray(points, dtype=np.float64)

        # Handle empty or single-point input
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        n = pts.shape[0]
        if n == 1:
            if pts[0, 0] <= x_limit:
                return pts.copy()
            return np.empty((0, 2), dtype=np.float64)

        # Detect x direction: increasing or decreasing
        reversed_flag = False
        if pts[0, 0] > pts[-1, 0]:
            # Reverse to make x increasing for simpler logic
            pts = pts[::-1].copy()
            reversed_flag = True

        # Now x is non-decreasing
        xmin = pts[0, 0]
        xmax = pts[-1, 0]
        eps = 1e-12

        # Case 1: x_limit is to the left of all points -> nothing to keep
        if x_limit < xmin - eps:
            result = np.empty((0, 2), dtype=np.float64)
            if reversed_flag:
                result = result[::-1].copy()
            return result

        # Case 2: x_limit is to the right of or equal to all points -> keep all
        if x_limit >= xmax - eps:
            result = pts.copy()
            if reversed_flag:
                result = result[::-1].copy()
            return result

        # Case 3: xmin <= x_limit < xmax, find last index with x <= x_limit via binary search
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if pts[mid, 0] <= x_limit:
                lo = mid + 1
            else:
                hi = mid - 1

        # hi is the last index with x <= x_limit
        iA = hi
        iB = hi + 1

        A = pts[iA]
        B = pts[iB]

        # Compute intersection P between segment AB and vertical line x = x_limit
        dx = B[0] - A[0]
        if abs(dx) < 1e-12:
            # Segment almost vertical; use A.y as approximation
            P = np.array([x_limit, A[1]], dtype=np.float64)
        else:
            t = (x_limit - A[0]) / dx
            yP = A[1] + t * (B[1] - A[1])
            P = np.array([x_limit, yP], dtype=np.float64)

        # Keep all points up to A, then append P
        left_part = pts[:iA + 1]

        # Avoid duplicate if last point is already numerically at P
        if np.linalg.norm(left_part[-1] - P) < 1e-12:
            result = left_part
        else:
            result = np.vstack([left_part, P])

        # Restore original order if input was decreasing in x
        if reversed_flag:
            result = result[::-1].copy()

        return result

    @classmethod
    def Mirror(cls, points: np.ndarray, axis_x: float = None, axis_y: float = None) -> np.ndarray:
        """
        Mirror points across vertical line x = axis_x and/or horizontal line y = axis_y.

        Args:
            points: Array of points to mirror (shape: n x 2)
            axis_x: X coordinate of vertical mirror axis (optional)
            axis_y: Y coordinate of horizontal mirror axis (optional)

        Returns:
            Mirrored points. If both axes provided, mirrors across both.

        For each point, the distance to the axis is preserved but direction is reversed:
        - mirrored_x = 2 * axis_x - point_x (if axis_x provided)
        - mirrored_y = 2 * axis_y - point_y (if axis_y provided)
        """
        mirrored = points.copy()

        if axis_x is not None:
            mirrored[:, 0] = 2 * axis_x - mirrored[:, 0]

        if axis_y is not None:
            mirrored[:, 1] = 2 * axis_y - mirrored[:, 1]

        return mirrored

    @classmethod
    def AreaOfClosedPolygon(cls, points: np.ndarray) -> float:
        """
        Calculate the area of a closed polygon using the Shoelace formula.

        Args:
            points: Array of points forming a closed polygon (shape: n x 2)

        Returns:
            Area of the polygon
        """
        if points.size == 0 or len(points) < 3:
            return 0.0

        pts = np.asarray(points, dtype=np.float64)

        # Shoelace formula (vectorized): Area = 0.5 * |Σ(x[i] * y[i+1] - x[i+1] * y[i])|
        x = pts[:, 0]
        y = pts[:, 1]

        # Compute cross products, including wrap-around from last to first point
        area = 0.5 * abs(
            np.sum(x[:-1] * y[1:]) - np.sum(x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]
        )

        return area

    @classmethod
    def Resistance(cls,
        points: np.ndarray,
        thick: float,
        start: np.ndarray,
        end: np.ndarray,
        rho: float = 1.724e-8
    ) -> float:
        """
        Calculate electrical resistance along boundary path from start to end.

        Integration along the boundary curve between start and end points,
        computing perpendicular width at each position.

        R = ρ ∫ (1/A(s)) ds where A(s) = w(s) × thick

        Args:
            points: Closed polygon boundary points (n x 2 array, in mm)
            thick: Thickness of the conductor (mm), default 0.047 mm
            start: Starting point on boundary (2D array, in mm)
            end: Ending point on boundary (2D array, in mm)
            rho: Electrical resistivity (Ω·m), default 1.724e-8 Ω·m for copper

        Returns:
            Resistance in Ohms
        """
        pts = np.asarray(points, dtype=np.float64)
        start_pt = np.asarray(start, dtype=np.float64)
        end_pt = np.asarray(end, dtype=np.float64)

        if pts.size == 0 or len(pts) < 3:
            return 0.0

        # Find nearest boundary points to start and end
        def find_nearest_boundary_index(target: np.ndarray) -> int:
            """Find index of boundary point nearest to target."""
            distances = np.linalg.norm(pts - target, axis=1)
            return np.argmin(distances)

        idx_start = find_nearest_boundary_index(start_pt)
        idx_end = find_nearest_boundary_index(end_pt)

        if idx_start == idx_end:
            return 0.0

        # Extract path along boundary from start to end
        # Choose shorter path (forward or backward along boundary)
        n = len(pts)
        if idx_start < idx_end:
            forward_len = idx_end - idx_start
            backward_len = n - forward_len
        else:
            forward_len = n - idx_start + idx_end
            backward_len = n - forward_len

        # Extract boundary path
        if forward_len <= backward_len:
            # Forward path
            if idx_start < idx_end:
                path_indices = list(range(idx_start, idx_end + 1))
            else:
                path_indices = list(range(idx_start, n)) + list(range(0, idx_end + 1))
        else:
            # Backward path (reverse)
            if idx_start > idx_end:
                path_indices = list(range(idx_start, idx_end - 1, -1))
            else:
                path_indices = list(range(idx_start, -1, -1)) + list(range(n - 1, idx_end - 1, -1))

        boundary_path = pts[path_indices]

        # Helper: find width perpendicular to segment direction
        def find_width_at_segment(pos: np.ndarray, tangent_dir: np.ndarray) -> float:
            """
            Find perpendicular width at position pos with given tangent direction.
            Returns width in mm, or None if invalid.
            """
            # Perpendicular direction (left normal)
            perp_dir = np.array([-tangent_dir[1], tangent_dir[0]])

            # Find intersections with all boundary segments
            intersections = []

            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]

                seg_vec = p2 - p1
                A = np.column_stack([perp_dir, -seg_vec])
                b = p1 - pos

                det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
                if abs(det) < 1e-12:
                    continue

                s = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
                u = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

                if 0 <= u <= 1:
                    intersections.append(s)

            if len(intersections) < 2:
                return None  # Invalid width

            intersections = np.array(intersections)
            width = abs(np.max(intersections) - np.min(intersections))

            # Return None for unreasonably small widths (< 0.01mm)
            return width if width >= 0.01 else None

        # Integrate along boundary path
        thick_m = thick * 1e-3
        integral = 0.0
        valid_segments = 0
        total_segments = 0

        # First pass: collect valid widths to compute average
        valid_widths = []
        for i in range(len(boundary_path) - 1):
            p1 = boundary_path[i]
            p2 = boundary_path[i + 1]
            seg_vec = p2 - p1
            ds = np.linalg.norm(seg_vec)
            if ds < 1e-12:
                continue
            tangent_dir = seg_vec / ds
            w1 = find_width_at_segment(p1, tangent_dir)
            w2 = find_width_at_segment(p2, tangent_dir)
            if w1 is not None:
                valid_widths.append(w1)
            if w2 is not None:
                valid_widths.append(w2)

        if not valid_widths:
            return 0.0  # No valid widths found

        # Compute average width for fallback
        avg_width = np.mean(valid_widths)

        # Second pass: integrate using valid or average widths
        for i in range(len(boundary_path) - 1):
            p1 = boundary_path[i]
            p2 = boundary_path[i + 1]

            # Segment vector and length
            seg_vec = p2 - p1
            ds = np.linalg.norm(seg_vec)

            if ds < 1e-12:
                continue

            total_segments += 1

            # Tangent direction
            tangent_dir = seg_vec / ds

            # Widths at both ends of segment
            w1 = find_width_at_segment(p1, tangent_dir)
            w2 = find_width_at_segment(p2, tangent_dir)

            # Use average width if individual width is invalid
            w1 = w1 if w1 is not None else avg_width
            w2 = w2 if w2 is not None else avg_width

            # Convert to meters
            w1_m = w1 * 1e-3
            w2_m = w2 * 1e-3
            ds_m = ds * 1e-3

            # Cross-sectional areas
            A1 = w1_m * thick_m
            A2 = w2_m * thick_m

            # Trapezoidal integration: ∫(1/A) ds
            integral += 0.5 * (1.0 / A1 + 1.0 / A2) * ds_m
            valid_segments += 1

        # Calculate resistance: R = ρ * integral
        resistance = rho * integral

        return resistance

    @classmethod
    def ResistanceAlongPath(cls,
        curve: np.ndarray,
        start_idx: int,
        end_idx: int,
        thick: float,
        rho: float = 1.724e-8
    ) -> float:
        """
        Calculate electrical resistance along the pattern boundary.

        The curve is structured as [outer_path, inner_path]. This improved version
        resamples the inner curve to match the outer curve's point count for accurate
        width pairing, which is critical for curves with different point densities.

        R = ρ ∫ (1/A(s)) ds where A(s) = w(s) × thick

        Args:
            curve: Complete closed curve points (n x 2 array, in mm)
            start_idx: Starting index of conductor path (usually 0)
            end_idx: Ending index of conductor path (exclusive)
            thick: Thickness of the conductor (mm), default 0.047 mm
            rho: Electrical resistivity (Ω·m), default 1.724e-8 Ω·m for copper

        Returns:
            Resistance in Ohms
        """
        if curve is None or len(curve) == 0:
            return 0.0
        if start_idx >= end_idx or end_idx > len(curve):
            return 0.0

        # Extract outer path (conductor path) and inner path
        outer = np.asarray(curve[start_idx:end_idx], dtype=np.float64)
        inner = np.asarray(curve[end_idx:], dtype=np.float64)

        if len(outer) < 2 or len(inner) < 2:
            return 0.0

        # Re-index so both sides start from the bottom seam (outer_end_idx)
        outer = outer[::-1].copy()  # bottom -> top

        def cumulative_lengths(pts: np.ndarray) -> np.ndarray:
            """Calculate cumulative arc lengths along curve."""
            if len(pts) < 2:
                return np.array([0.0], dtype=np.float64)
            diffs = pts[1:] - pts[:-1]
            seg_lengths = np.linalg.norm(diffs, axis=1)
            return np.concatenate(([0.0], np.cumsum(seg_lengths)))

        def resample_curve(pts: np.ndarray, num_points: int) -> np.ndarray:
            """
            Resample curve to have exactly num_points by interpolating along arc length.
            This ensures outer and inner curves have matching point counts for pairing.
            """
            if len(pts) == num_points:
                return pts

            cum_lengths = cumulative_lengths(pts)
            total_length = cum_lengths[-1]

            if total_length < 1e-12:
                # Degenerate curve, just repeat first point
                return np.tile(pts[0], (num_points, 1))

            # Generate target arc lengths uniformly distributed
            target_lengths = np.linspace(0, total_length, num_points)
            resampled = np.zeros((num_points, 2), dtype=np.float64)

            for i, target in enumerate(target_lengths):
                if target <= 0.0:
                    resampled[i] = pts[0]
                elif target >= total_length:
                    resampled[i] = pts[-1]
                else:
                    # Find segment containing target
                    idx = np.searchsorted(cum_lengths, target, side="right") - 1
                    idx = max(0, min(idx, len(pts) - 2))

                    # Interpolate within segment
                    seg_len = cum_lengths[idx + 1] - cum_lengths[idx]
                    if seg_len < 1e-12:
                        resampled[i] = pts[idx]
                    else:
                        t = (target - cum_lengths[idx]) / seg_len
                        resampled[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])

            return resampled

        # CRITICAL FIX: Resample inner curve to match outer curve's point count
        # This ensures proper 1-to-1 point pairing for width calculation
        inner_resampled = resample_curve(inner, len(outer))

        # Now outer and inner_resampled have the same number of points
        # Calculate width vectors by direct pairing
        width_vectors = inner_resampled - outer

        tol = 1e-6
        segment_data = []
        valid_widths = []

        for i in range(len(outer) - 1):
            p1 = outer[i]
            p2 = outer[i + 1]
            seg_vec = p2 - p1
            ds = np.linalg.norm(seg_vec)
            if ds < tol:
                continue
            tangent_dir = seg_vec / ds
            normal_dir = np.array([-tangent_dir[1], tangent_dir[0]])

            # Calculate perpendicular widths
            raw_w1 = abs(np.dot(width_vectors[i], normal_dir))
            raw_w2 = abs(np.dot(width_vectors[i + 1], normal_dir))

            # Fallback to Euclidean distance if perpendicular width is too small
            if raw_w1 < tol:
                raw_w1 = np.linalg.norm(width_vectors[i])
            if raw_w2 < tol:
                raw_w2 = np.linalg.norm(width_vectors[i + 1])

            if raw_w1 > tol:
                valid_widths.append(raw_w1)
            if raw_w2 > tol:
                valid_widths.append(raw_w2)

            segment_data.append((ds, raw_w1, raw_w2))

        if not segment_data or not valid_widths:
            return 0.0

        avg_width = float(np.mean(valid_widths))
        thick_m = thick * 1e-3
        integral = 0.0

        for ds, raw_w1, raw_w2 in segment_data:
            w1 = raw_w1 if raw_w1 > tol else avg_width
            w2 = raw_w2 if raw_w2 > tol else avg_width

            w1_m = w1 * 1e-3
            w2_m = w2 * 1e-3
            ds_m = ds * 1e-3

            A1 = w1_m * thick_m
            A2 = w2_m * thick_m

            integral += 0.5 * (1.0 / A1 + 1.0 / A2) * ds_m

        resistance = rho * integral
        return resistance
