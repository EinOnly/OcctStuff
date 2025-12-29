import numpy as np
import trimesh
from skimage import measure

def create_spinodal_structure(
    box_size=(100, 100, 20), 
    resolution=1.0,
    feature_size=8.0,      # 类似 TPMS 的周期，决定孔的大小
    wall_thickness=0.5,    # 隔离壁厚度
    seed=42                # 随机种子
):
    print(f"1. 初始化空间 {box_size}...")
    np.random.seed(seed)
    
    # 增加 padding
    pad = 4
    res = resolution
    shape = (int(box_size[0]/res) + pad*2, 
             int(box_size[1]/res) + pad*2, 
             int(box_size[2]/res) + pad*2)
    
    print("2. 生成高斯随机场 (Gaussian Random Field)...")
    # 旋节分解可以通过叠加大量随机相位的波来模拟
    # 为了得到特定大小的孔 (feature_size)，我们需要在频域上进行带通滤波
    
    # 初始化 k 空间 (频域)
    kx = np.fft.fftfreq(shape[0])
    ky = np.fft.fftfreq(shape[1])
    kz = np.fft.fftfreq(shape[2])
    k_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # 计算频率模长 |k|
    k_norm = np.sqrt(k_grid[0]**2 + k_grid[1]**2 + k_grid[2]**2)
    
    # 目标频率 (对应 feature_size)
    # k = 1 / lambda
    target_k = 1.0 / (feature_size / res)
    bandwidth = 0.2 * target_k # 频带宽度，越窄结构越均匀，越宽结构越混乱
    
    # 构建带通滤波器 (高斯形状)
    # 我们只允许特定频率的波通过，这样生成的噪点就会变成均匀的"斑点"
    filter_mask = np.exp(-0.5 * ((k_norm - target_k) / (bandwidth/2.355))**2)
    
    # 生成白噪声
    white_noise = np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape)
    
    # 频域滤波
    spectrum = np.fft.fftn(white_noise)
    filtered_spectrum = spectrum * filter_mask
    
    # 逆傅里叶变换变回实空间
    field = np.real(np.fft.ifftn(filtered_spectrum))
    
    # 归一化到 [-1, 1] (近似)
    field = field / np.std(field)

    print("3. 切割实体流道...")
    
    # 这里的 field 就是我们的 SDF
    # 旋节分解天然将空间分为两半： >t 和 <-t
    
    # 边界 Box
    x = np.linspace(0, box_size[0], shape[0])
    y = np.linspace(0, box_size[1], shape[1])
    z = np.linspace(0, box_size[2], shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # 简单的 Box SDF
    d_x = np.abs(xx - box_size[0]/2) - box_size[0]/2
    d_y = np.abs(yy - box_size[1]/2) - box_size[1]/2
    d_z = np.abs(zz - box_size[2]/2) - box_size[2]/2
    sdf_box = np.maximum(np.maximum(d_x, d_y), d_z)
    
    t = wall_thickness
    
    # 提取 Tube A (Cahn-Hilliard Phase A)
    sdf_A = np.maximum(-(field - t), sdf_box)
    
    # 提取 Tube B (Cahn-Hilliard Phase B)
    sdf_B = np.maximum((field + t), sdf_box)

    print("4. 生成 Mesh...")
    mesh_a = measure_to_mesh(sdf_A, res)
    mesh_a.visual.face_colors = [255, 100, 50, 255] # 珊瑚红
    
    mesh_b = measure_to_mesh(sdf_B, res)
    mesh_b.visual.face_colors = [50, 100, 255, 255] # 深海蓝
    
    return mesh_a, mesh_b

def measure_to_mesh(sdf, res):
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0, spacing=(res, res, res))
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except:
        return trimesh.Trimesh()

if __name__ == "__main__":
    print("生成 Spinodal (旋节分解) 换热器...")
    print("这种结构模仿了油水分离的物理过程。")
    print("它没有 TPMS 那种人工的'晶格感'，而是完全各向同性的随机双连续结构。")
    
    mesh_a, mesh_b = create_spinodal_structure(
        box_size=(150, 150, 20), # 稍微小一点方便演示 FFT 计算
        resolution=1.0,
        feature_size=12.0,       # 孔径大小
        wall_thickness=0.3
    )
    
    combined = trimesh.util.concatenate([mesh_a, mesh_b])
    combined.export("Spinodal_Exchanger.stl")
    print("完成。")