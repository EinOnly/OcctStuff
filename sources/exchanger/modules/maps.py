import numpy as np
from scipy.ndimage import gaussian_filter
from noise import pnoise3

# Generate a random density map
def generate_density_map(res=200, smoothness=50):
    density = np.random.rand(res, res)
    density = gaussian_filter(density, sigma=smoothness)
    density /= density.max()  # Normalize to [0, 1]
    return density

def generate_perlin_density(t=200, res=200, scale=1.0):
    density = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            x = i / res * scale
            y = j / res * scale
            density[i, j] = pnoise3(x, y, t)
    density = density - density.min()
    density = density / density.max()
    return density

def generate_density_with_circles(
    size=(100, 100),
    res=200,
    circles=None,
    gradient_direction=(0, -1),
    gradient_strength=0.3,
    falloff=10.0
):
    w, h = size
    density = np.zeros((res, res), dtype=np.float32)
    xx, yy = np.meshgrid(np.linspace(0, w, res), np.linspace(0, h, res))

    if circles:
        for cx, cy, r, mode in circles:
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

            layer = np.zeros_like(density)

            # circle area invalid
            layer[dist <= r] = 1.0

            # falloff area
            mask_falloff = (dist > r) & (dist <= r + falloff)
            alpha = (dist[mask_falloff] - r) / falloff

            if mode == 'in':
                # from circle center to edge pressure linearly decrease (capped to 1.0)
                layer[mask_falloff] = 1.0 - alpha
            elif mode == 'out':
                # from edge to center pressure linearly increase (capped to 1.0)
                layer[mask_falloff] = alpha

            # apply falloff
            density = np.maximum(density, layer)

    density = np.clip(density, 0, 1)

    if gradient_strength > 0:
        gx, gy = np.array(gradient_direction, dtype=np.float32)
        norm = np.sqrt(gx**2 + gy**2) + 1e-8
        gx /= norm
        gy /= norm

        direction_map = gx * xx + gy * yy
        direction_map = (direction_map - direction_map.min()) / (direction_map.max() - direction_map.min())
        density = (1 - gradient_strength) * density + gradient_strength * direction_map
    # Calculate inverse density (1 - density) if needed
    # This creates the opposite effect where high values become low and vice versa
    density = 1 - density
    return np.clip(density, 0, 1)

def generate_density_with_mask(
    size=(100, 100),
    valid_mask=None,             # Boolean mask of valid region
    circles=None,                # List of (cx, cy, r, mode)
    gradient_direction=(0, -1),  # e.g., gravity = (0, -1)
    gradient_strength=0.3,       # [0, 1] blending with gradient
    falloff=10.0                 # Edge soft falloff
):
    '''
    Generate a density map using circle-based influence, within the valid region mask.
    
    Params:
        size: (w, h) in mm (same as plate size)
        valid_mask: bool array of shape (H, W)
        circles: list of (cx, cy, r, mode) where mode = 'in' or 'out'
        gradient_direction: gradient vector direction (e.g., gravity)
        gradient_strength: how much the gradient blends into final density
        falloff: range outside the circle to soften edge effect
    '''
    w, h = size
    if valid_mask is None:
        raise ValueError("valid_mask must be provided.")
    
    res_y, res_x = valid_mask.shape
    xx, yy = np.meshgrid(np.linspace(0, w, res_x), np.linspace(0, h, res_y))

    density = np.zeros_like(valid_mask, dtype=np.float32)

    if circles:
        for cx, cy, r, mode in circles:
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

            layer = np.zeros_like(density)

            # inner full region
            layer[dist <= r] = 1.0

            # falloff ring
            mask_falloff = (dist > r) & (dist <= r + falloff)
            alpha = (dist[mask_falloff] - r) / falloff

            if mode == 'in':
                layer[mask_falloff] = 1.0 - alpha
            elif mode == 'out':
                layer[mask_falloff] = alpha

            density = np.maximum(density, layer)

    density = np.clip(density, 0, 1)

    # Add directional gradient
    if gradient_strength > 0:
        gx, gy = np.array(gradient_direction, dtype=np.float32)
        norm = np.sqrt(gx**2 + gy**2) + 1e-8
        gx /= norm
        gy /= norm

        direction_map = gx * xx + gy * yy
        direction_map = (direction_map - direction_map.min()) / (direction_map.max() - direction_map.min())

        density = (1 - gradient_strength) * density + gradient_strength * direction_map

    # Only invert valid area
    density[valid_mask] = 1.0 - density[valid_mask]

    # Optional: mask out invalid area (if you still want 0 outside)
    density[~valid_mask] = 0.0

    return density


if __name__ == "__main__":
    bounds = [0, 120, 0, 55]
    circles=[
        (11.5, 11.5, 6, 'in'),
        (108.5, 43.5, 6, 'out'),
    ]
    density_map = generate_density_map()

    # Plot the density map
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(density_map, cmap='viridis', extent=bounds, origin='lower')
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    for cx, cy, _, _ in circles:
        ax.plot(cx, cy, 'ro')

    plt.tight_layout()
    plt.show()
        
