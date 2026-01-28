import os
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure
from skimage.morphology import skeletonize
import networkx as nx
import trimesh

# -----------------------------
# 1) TPMS scalar field (Gyroid)
# f(x,y,z) = sin x cos y + sin y cos z + sin z cos x - t
# -----------------------------
def gyroid_field(X, Y, Z, t=0.0):
    return (
        np.sin(X) * np.cos(Y) +
        np.sin(Y) * np.cos(Z) +
        np.sin(Z) * np.cos(X) - t
    )

# -----------------------------
# 2) Build voxel domains A/B from TPMS field sign
# A: f > 0, B: f < 0
# -----------------------------
def build_domains(n=128, periods=2, t=0.0):
    # Domain: [0, 2π*periods]^3
    L = 2.0 * np.pi * periods
    xs = np.linspace(0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    f = gyroid_field(X, Y, Z, t=t)
    A = f > 0
    B = f < 0
    return A, B, L

# -----------------------------
# 3) Clean voxel domains for better skeleton
# -----------------------------
def clean_binary(vol, iters=1):
    # Use 26-neighborhood structure
    st = generate_binary_structure(3, 2)
    out = vol.copy()
    for _ in range(iters):
        out = binary_closing(out, structure=st)
        out = binary_opening(out, structure=st)
    return out

# -----------------------------
# 4) Skeletonize and convert skeleton voxels to graph
# Each voxel is a node, connect 26-neighborhood
# Then compress to a simpler graph by pruning degree-2 chains
# -----------------------------
_NEIGHBORS_26 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if not (i == j == k == 0)]

def skeleton_to_graph(skel):
    pts = np.argwhere(skel)
    pts_set = {tuple(p) for p in pts}

    G = nx.Graph()
    for p in pts_set:
        G.add_node(p)
        x, y, z = p
        for dx, dy, dz in _NEIGHBORS_26:
            q = (x + dx, y + dy, z + dz)
            if q in pts_set:
                G.add_edge(p, q)
    return G

def compress_graph(G):
    # Keep junctions (deg!=2) and endpoints (deg==1), remove degree-2 chains into polyline edges
    keep = [n for n in G.nodes if G.degree[n] != 2]
    keep_set = set(keep)

    H = nx.Graph()
    for n in keep:
        H.add_node(n)

    visited_edges = set()

    for n in keep:
        for nbr in G.neighbors(n):
            ekey = tuple(sorted([n, nbr]))
            if ekey in visited_edges:
                continue

            # Walk along chain
            path = [n, nbr]
            prev = n
            cur = nbr
            while cur not in keep_set and G.degree[cur] == 2:
                nxts = [x for x in G.neighbors(cur) if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                path.append(nxt)
                prev, cur = cur, nxt

            end = cur
            visited_edges.add(ekey)
            # mark all chain edges as visited
            for i in range(len(path) - 1):
                visited_edges.add(tuple(sorted([path[i], path[i + 1]])))

            if end == n:
                continue
            if end not in H:
                H.add_node(end)

            # Store polyline as edge attribute
            H.add_edge(n, end, polyline=path)

    return H

# -----------------------------
# 5) Turn each edge-polyline into an independent tube mesh
# Use trimesh "sweep" by building a polyline and sweeping a circle
# -----------------------------
def voxel_to_world(p, n, L):
    # Map voxel index to world coordinate in [0,L)
    return (np.array(p, dtype=np.float64) / n) * L

def make_tube_from_polyline(polyline_vox, n, L, radius=0.08, sections=16):
    # Convert polyline to world coords
    P = np.array([voxel_to_world(p, n, L) for p in polyline_vox], dtype=np.float64)

    # Remove consecutive duplicates
    d = np.linalg.norm(P[1:] - P[:-1], axis=1)
    keep = np.hstack([[True], d > 1e-9])
    P = P[keep]
    if len(P) < 2:
        return None

    # Build tube by connecting cylindrical segments
    segments = []
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        seg_vec = p1 - p0
        height = np.linalg.norm(seg_vec)
        if height < 1e-9:
            continue

        # Create cylinder along Z axis, then transform
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

        # Build rotation from Z to seg_vec direction
        z_axis = np.array([0, 0, 1])
        seg_dir = seg_vec / height

        # Rotation using Rodrigues formula
        cross = np.cross(z_axis, seg_dir)
        dot = np.dot(z_axis, seg_dir)

        if np.linalg.norm(cross) < 1e-9:
            if dot > 0:
                R = np.eye(3)
            else:
                R = np.diag([1, -1, -1])  # 180 degree rotation
        else:
            cross_norm = cross / np.linalg.norm(cross)
            angle = np.arccos(np.clip(dot, -1, 1))
            K = np.array([
                [0, -cross_norm[2], cross_norm[1]],
                [cross_norm[2], 0, -cross_norm[0]],
                [-cross_norm[1], cross_norm[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        # Transform: rotate then translate to midpoint
        midpoint = (p0 + p1) / 2
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = midpoint

        cyl.apply_transform(transform)
        segments.append(cyl)

    if not segments:
        return None

    return trimesh.util.concatenate(segments)

# -----------------------------
# 6) Main pipeline: TPMS -> A/B skeleton -> independent tubes export
# -----------------------------
def generate_AB_independent_tubes(
    out_dir="out_tpms_weave",
    n=128,
    periods=2,
    t=0.0,
    clean_iters=1,
    tube_radius=0.07,
    min_edge_len_vox=8,
):
    os.makedirs(out_dir, exist_ok=True)

    A, B, L = build_domains(n=n, periods=periods, t=t)
    A = clean_binary(A, iters=clean_iters)
    B = clean_binary(B, iters=clean_iters)

    results = {}
    for name, vol in [("A", A), ("B", B)]:
        # Skeleton
        skel = skeletonize(vol)
        G = skeleton_to_graph(skel)
        H = compress_graph(G)

        tubes = []
        edge_id = 0
        group_dir = os.path.join(out_dir, name)
        os.makedirs(group_dir, exist_ok=True)

        for u, v, data in H.edges(data=True):
            poly = data.get("polyline", None)
            if poly is None:
                continue
            if len(poly) < min_edge_len_vox:
                continue

            mesh = make_tube_from_polyline(poly, n=n, L=L, radius=tube_radius)
            if mesh is None:
                continue

            # Export each tube as an independent STL
            fp = os.path.join(group_dir, f"{name}_tube_{edge_id:05d}.stl")
            mesh.export(fp)
            tubes.append(fp)
            edge_id += 1

        results[name] = tubes
        print(f"[{name}] tubes exported: {len(tubes)} -> {group_dir}")

    # Optional: also export combined A/B for quick preview
    for name in ["A", "B"]:
        meshes = [trimesh.load(p) for p in results[name]]
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.export(os.path.join(out_dir, f"{name}_combined.stl"))

    return results

if __name__ == "__main__":
    generate_AB_independent_tubes(
        out_dir="out_tpms_weave",
        n=160,            # higher => smoother, slower
        periods=2,        # how many TPMS periods inside cube
        t=0.0,            # threshold shift; changes volume ratio A/B
        clean_iters=1,
        tube_radius=0.06, # world units in [0, 2π*periods]
        min_edge_len_vox=10
    )