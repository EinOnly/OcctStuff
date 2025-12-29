import numpy as np
import trimesh
from skimage import measure

def create_organic_interwoven_tubes(
    box_size=(60, 60, 60), 
    resolution=0.6, 
    base_periodicity=18.0,
    warp_strength=4.0,
    warp_frequency=0.15,
    wall_thickness=0.2
):
    print(f"1. 初始化空间 {box_size}...")
    pad = 2
    res = resolution
    shape = (
        int(box_size[0]/res) + pad*2, 
        int(box_size[1]/res) + pad*2, 
        int(box_size[2]/res) + pad*2
    )
    
    x = np.linspace(-box_size[0]/2 - pad, box_size[0]/2 + pad, shape[0])
    y = np.linspace(-box_size[1]/2 - pad, box_size[1]/2 + pad, shape[1])
    z = np.linspace(-box_size[2]/2 - pad, box_size[2]/2 + pad, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    print("2. 应用域扭曲 (Domain Warping)...")
    # 这就是“有机”的来源。我们不再使用标准的 xx, yy, zz，
    # 而是使用被“噪声”扭曲过的坐标去采样 TPMS。
    
    # 模拟 3D 噪声：简单的多频正弦波叠加 (伪 Perlin 噪声)
    def pseudo_noise(px, py, pz, freq, seed=0):
        # 使用不同的相位偏移来模拟随机性
        n = (np.sin(px * freq + seed) + 
             np.sin(py * freq + seed*1.3) + 
             np.sin(pz * freq + seed*1.7))
        return n

    # 对坐标进行偏移
    # x' = x + strength * noise(z, y)
    # 这种非线性映射会让直线变成曲线
    freq = warp_frequency
    distort_x = pseudo_noise(xx, yy, zz, freq, seed=1.1) * warp_strength
    distort_y = pseudo_noise(xx, yy, zz, freq, seed=2.2) * warp_strength
    distort_z = pseudo_noise(xx, yy, zz, freq, seed=3.3) * warp_strength

    xx_warped = xx + distort_x
    yy_warped = yy + distort_y
    zz_warped = zz + distort_z

    print("3. 计算扭曲后的 Schwarz D 场...")
    # 使用扭曲后的坐标计算 TPMS
    k = 2 * np.pi / base_periodicity
    gx, gy, gz = xx_warped * k, yy_warped * k, zz_warped * k
    
    # Schwarz D 公式
    sdf_tpms = (np.sin(gx) * np.sin(gy) * np.sin(gz) + 
                np.sin(gx) * np.cos(gy) * np.cos(gz) + 
                np.cos(gx) * np.sin(gy) * np.cos(gz) + 
                np.cos(gx) * np.cos(gy) * np.sin(gz))
    
    # --- 下面是与之前相同的“隔离”逻辑 ---
    
    print("4. 布尔运算切割...")
    # 计算 Box 边界 (使用原始未扭曲的坐标，因为我们想要方方正正的盒子)
    d_x = np.abs(xx) - box_size[0]/2
    d_y = np.abs(yy) - box_size[1]/2
    d_z = np.abs(zz) - box_size[2]/2
    sdf_box = np.maximum(np.maximum(d_x, d_y), d_z)

    # 定义隔离阈值
    t = wall_thickness 

    # 提取 Tube A (SDF > t)
    sdf_fluid_A = np.maximum(-(sdf_tpms - t), sdf_box)
    
    # 提取 Tube B (SDF < -t)
    sdf_fluid_B = np.maximum((sdf_tpms + t), sdf_box)

    print("5. 生成 STL...")
    mesh_A = measure_to_mesh(sdf_fluid_A, res)
    mesh_A.visual.face_colors = [200, 50, 50, 255] # 暗红/血管色

    mesh_B = measure_to_mesh(sdf_fluid_B, res)
    mesh_B.visual.face_colors = [50, 100, 200, 255] # 静脉色

    return mesh_A, mesh_B

def measure_to_mesh(sdf, res):
    # 辅助函数
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0, spacing=(res, res, res))
    return trimesh.Trimesh(vertices=verts, faces=faces)

if __name__ == "__main__":
    # 调整 warp_strength 可以控制“有机”程度
    # warp_strength = 0 -> 规则的晶格
    # warp_strength = 5 -> 像树根或血管一样纠缠
    mesh_a, mesh_b = create_organic_interwoven_tubes(
        box_size=(30, 20, 60),
        base_periodicity=20.0,  
        warp_strength=6.0,      # 强扭曲
        warp_frequency=0.10,    # 低频扭曲（大范围弯曲）
        wall_thickness=0.5
    )
    
    # 组合导出
    combined = trimesh.util.concatenate([mesh_a, mesh_b])
    combined.export("Organic_Interwoven.stl")
    print("生成完毕。请查看 'Organic_Interwoven.stl'")
    print("你会发现结构虽然还是互锁的，但不再是笔直的网格，而是充满了自然的弯曲和流动感。")


    