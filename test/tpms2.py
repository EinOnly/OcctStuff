import numpy as np
import trimesh
from skimage import measure

def create_isolated_tubes_schwarz_d(
    box_size=(60, 60, 60), 
    resolution=0.6, 
    periodicity=20.0,      # 周期越大，管子越粗，结构越稀疏
    wall_thickness_factor=0.5 # 控制隔开两根管子的壁厚 (0.1~1.0)
):
    print(f"1. 初始化体素空间 {box_size}...")
    # 增加 padding，确保我们在计算时不切断边界，而是通过布尔运算来切
    pad = 2
    res = resolution
    shape = (int(box_size[0]/res) + pad*2, 
             int(box_size[1]/res) + pad*2, 
             int(box_size[2]/res) + pad*2)
    
    x = np.linspace(-box_size[0]/2 - pad, box_size[0]/2 + pad, shape[0])
    y = np.linspace(-box_size[1]/2 - pad, box_size[1]/2 + pad, shape[1])
    z = np.linspace(-box_size[2]/2 - pad, box_size[2]/2 + pad, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    print("2. 计算 Schwarz D (Diamond) 场...")
    # Schwarz D 公式: sin(x)sin(y)sin(z) + sin(x)cos(y)cos(z) + ...
    # 这是一个非常经典的类似“钻石晶格”的管道结构
    k = 2 * np.pi / periodicity
    gx, gy, gz = xx * k, yy * k, zz * k
    
    # 原始 Schwarz D SDF
    sdf_tpms = (np.sin(gx) * np.sin(gy) * np.sin(gz) + 
                np.sin(gx) * np.cos(gy) * np.cos(gz) + 
                np.cos(gx) * np.sin(gy) * np.cos(gz) + 
                np.cos(gx) * np.cos(gy) * np.sin(gz))
    
    # --- 关键逻辑：定义两个分离的空间 ---
    # 空间 A: SDF > t
    # 空间 B: SDF < -t
    # 中间 (-t 到 t) 是实体壁，把它们隔开
    t = wall_thickness_factor 
    
    print("3. 构建封闭边界 (Boolean Intersection)...")
    # 为了证明它们是独立的，我们要生成“封闭的实体”(Watertight Solids)
    # 这就像是生成“管道里的水”的形状
    
    # 计算 Box 的 SDF (负值在箱内，正值在箱外)
    # d = max(|x|-w/2, |y|-h/2, |z|-d/2)
    d_x = np.abs(xx) - box_size[0]/2
    d_y = np.abs(yy) - box_size[1]/2
    d_z = np.abs(zz) - box_size[2]/2
    sdf_box = np.maximum(np.maximum(d_x, d_y), d_z)

    # --- 生成流体域 A (Tube A) ---
    # 逻辑：(是 TPMS A 部分) AND (在箱子里面)
    # Boolean Intersection 在 SDF 中通常用 max(sdf1, sdf2)
    # 我们希望提取的物体是 sdf < 0 的部分
    # TPMS A 原始定义是 > t。为了配合 marching_cubes (提取 < 0)，我们需要反转符号。
    # 目标区域：sdf_tpms > t  -->  -(sdf_tpms - t) < 0
    sdf_fluid_A = np.maximum(-(sdf_tpms - t), sdf_box)

    # --- 生成流体域 B (Tube B) ---
    # 目标区域：sdf_tpms < -t -->  (sdf_tpms + t) < 0
    sdf_fluid_B = np.maximum((sdf_tpms + t), sdf_box)

    print("4. 提取网格 (Tube A)...")
    verts_A, faces_A, _, _ = measure.marching_cubes(sdf_fluid_A, level=0.0, spacing=(res, res, res))
    mesh_A = trimesh.Trimesh(vertices=verts_A, faces=faces_A)
    mesh_A.visual.face_colors = [255, 50, 50, 255] # 红色

    print("5. 提取网格 (Tube B)...")
    verts_B, faces_B, _, _ = measure.marching_cubes(sdf_fluid_B, level=0.0, spacing=(res, res, res))
    mesh_B = trimesh.Trimesh(vertices=verts_B, faces=faces_B)
    mesh_B.visual.face_colors = [50, 50, 255, 255] # 蓝色

    return mesh_A, mesh_B

if __name__ == "__main__":
    # 生成
    mesh_a, mesh_b = create_isolated_tubes_schwarz_d(
        box_size=(50, 50, 50),
        resolution=0.5,       # 如果太慢可改为 0.8 或 1.0
        periodicity=25.0,     # 周期
        wall_thickness_factor=0.6 # 这个值越大，管子越细，两管之间的空隙（壁厚）越大
    )

    # 导出
    print("导出 STL...")
    mesh_a.export("SchwarzD_Tube_A.stl")
    mesh_b.export("SchwarzD_Tube_B.stl")
    
    # 组合预览
    combined = trimesh.util.concatenate([mesh_a, mesh_b])
    combined.export("SchwarzD_Assembly.stl")
    
    print("完成！")
    print("请查看 'SchwarzD_Assembly.stl'。")
    print("你应该能看到红、蓝两套形状像'原子结构'或'复杂管道'的物体。")
    print("它们互相紧密缠绕，但在数学上和几何上是完全没有任何面接触的。")