import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# TPMS: Gyroid implicit field
# f(x,y,z) = sin x cos y + sin y cos z + sin z cos x
# -------------------------
def f_gyroid(p):
    x, y, z = p
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)

def grad_gyroid(p):
    x, y, z = p
    # df/dx, df/dy, df/dz
    dfx = np.cos(x) * np.cos(y) - np.sin(z) * np.sin(x)
    dfy = -np.sin(x) * np.sin(y) + np.cos(y) * np.cos(z)
    dfz = -np.sin(y) * np.sin(z) + np.cos(z) * np.cos(x)
    return np.array([dfx, dfy, dfz], dtype=np.float64)

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / max(n, eps)

def wrap_periodic(p, L=2*np.pi):
    # Wrap to [0, L)
    return (p % L + L) % L

def project_to_levelset(p, c, iters=5):
    # Newton-like projection: p <- p - (f(p)-c) * grad / |grad|^2
    q = p.copy()
    for _ in range(iters):
        g = grad_gyroid(q)
        gg = np.dot(g, g)
        if gg < 1e-12:
            break
        q = q - (f_gyroid(q) - c) * g / gg
    return q

def random_seed_on_levelset(c, L=2*np.pi, tries=2000):
    # Random sample then project
    for _ in range(tries):
        p = np.random.rand(3) * L
        p = project_to_levelset(p, c, iters=10)
        if abs(f_gyroid(p) - c) < 1e-4:
            return wrap_periodic(p, L)
    raise RuntimeError("Failed to find seed on levelset. Try different c.")

def trace_streamline_on_levelset(
    c,
    steps=2000,
    ds=0.01,
    L=2*np.pi,
    a=np.array([1.0, 0.0, 0.0]),
    seed=None,
    proj_iters=3,
):
    # Direction field tangent to levelset:
    # v = normalize( grad f × a )  (perpendicular to grad, thus tangent)
    # If grad is near-parallel to a, switch a automatically.
    if seed is None:
        p = random_seed_on_levelset(c, L=L)
    else:
        p = project_to_levelset(np.array(seed, dtype=np.float64), c, iters=10)
        p = wrap_periodic(p, L)

    pts = np.zeros((steps, 3), dtype=np.float64)
    pts[0] = p

    a1 = a.astype(np.float64)

    for i in range(1, steps):
        g = grad_gyroid(p)

        # If grad almost parallel to a, pick a different a
        if abs(np.dot(normalize(g), normalize(a1))) > 0.95:
            a1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        v = np.cross(g, a1)
        v = normalize(v)

        # Step
        p = p + ds * v

        # Wrap periodic cell
        p = wrap_periodic(p, L)

        # Pull back to levelset to remove drift
        p = project_to_levelset(p, c, iters=proj_iters)
        p = wrap_periodic(p, L)

        pts[i] = p

    return pts

def generate_tpms_tubes(
    n_tubes=12,
    c_values=None,
    steps=2500,
    ds=0.01,
    L=2*np.pi,
):
    if c_values is None:
        # Pick multiple level offsets around 0, small offsets look "channel-like"
        # You can widen this range if you want more spread
        c_values = np.linspace(-0.35, 0.35, n_tubes)

    tubes = []
    for c in c_values:
        pts = trace_streamline_on_levelset(
            c=float(c),
            steps=steps,
            ds=ds,
            L=L,
            a=np.array([1.0, 0.0, 0.0]),
            seed=None,
            proj_iters=3,
        )
        tubes.append(pts)
    return tubes

def plot_tubes(tubes, L=2*np.pi):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for P in tubes:
        ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=2)

    ax.set_title("Gyroid TPMS-like independent tubes (streamlines on f(x,y,z)=c)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)

    tubes = generate_tpms_tubes(
        n_tubes=14,
        steps=2200,
        ds=0.012,
        L=2*np.pi,
        # c_values=np.linspace(-0.4, 0.4, 14),
    )
    plot_tubes(tubes)