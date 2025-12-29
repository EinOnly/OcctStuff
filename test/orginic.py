import numpy as np
import trimesh
from skimage import measure

def create_streamlined_exchanger(
    dims=(300, 200, 20), 
    resolution=0.8,         # 建议 0.5 用于生产
    flow_axis='x',          # 流动方向
    cross_section_period=8.0, # 截面(YZ)的周期：越小，管子越细越密
    flow_period=100.0,      # 流向(X)的周期：极大，拉直管路
    wavy_amplitude=3.0,     # 管道弯曲幅度 (mm)
    wall_thickness=0.6      # 壁厚 (mm)
):
    print(f"1. 初始化空间 {dims}...")
    pad = 2
    res = resolution
    shape = (int(dims[0]/res) + pad*2, 
             int(dims[1]/res) + pad*2, 
             int(dims[2]/res) + pad*2)
    
    x = np.linspace(-dims[0]/2 - pad, dims[0]/2 + pad, shape[0])
    y = np.linspace(-dims[1]/2 - pad, dims[1]/2 + pad, shape[1])
    z = np.linspace(-dims[2]/2 - pad, dims[2]/2 + pad, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    print("2. 应用流线型映射 (Streamline Mapping)...")
    # 这里的核心思想是：不要随机扭曲，而是“规律弯曲”。
    # 让管子像蛇一样轻轻摆动，产生二次流，但不要阻挡主流。

    # 定义流动方向的坐标 (Longitudinal) 和 截面坐标 (Transverse)
    if flow_axis == 'x':
        long_coord = xx
        trans_coord_1 = yy
        trans_coord_2 = zz
    
    # 关键步骤：各向异性拉伸
    # 我们希望 X 方向的变化非常缓慢 (Period = 100mm)
    # Y/Z 方向的变化非常快 (Period = 8mm)
    
    k_flow = 2 * np.pi / flow_period
    k_cross = 2 * np.pi / cross_section_period

    # --- 添加“波浪” (Waviness) ---
    # 让直管子稍微弯一点点，避免完全层流导致的换热效率低
    # y' = y + A * sin(k * x)
    offset_1 = np.sin(long_coord * k_flow * 2.0) * wavy_amplitude
    offset_2 = np.cos(long_coord * k_flow * 2.0) * wavy_amplitude # 相位差90度，形成螺旋感

    # 映射坐标
    # 注意：只在横截面坐标上施加正弦波，流向坐标保持线性
    tc1_mapped = (trans_coord_1 + offset_1) * k_cross
    tc2_mapped = (trans_coord_2 + offset_2) * k_cross
    lc_mapped  = long_coord * k_flow # 流向被极度拉伸

    print("3. 计算各向异性 Schwarz D...")
    # Schwarz D: sin(x)sin(y)sin(z) + ...
    # 这里的输入坐标已经被"拉伸"且"弯曲"了
    
    # 使用长坐标 lc_mapped 和 短坐标 tc_mapped
    # 公式调整为适应各向异性
    term1 = np.sin(lc_mapped) * np.sin(tc1_mapped) * np.sin(tc2_mapped)
    term2 = np.sin(lc_mapped) * np.cos(tc1_mapped) * np.cos(tc2_mapped)
    term3 = np.cos(lc_mapped) * np.sin(tc1_mapped) * np.cos(tc2_mapped)
    term4 = np.cos(lc_mapped) * np.cos(tc1_mapped) * np.sin(tc2_mapped)
    
    sdf_tpms = term1 + term2 + term3 + term4

    print("4. 生成实体流道...")
    # 边界
    d_x = np.abs(xx) - dims[0]/2
    d_y = np.abs(yy) - dims[1]/2
    d_z = np.abs(zz) - dims[2]/2
    sdf_box = np.maximum(np.maximum(d_x, d_y), d_z)

    # 阈值 (壁厚)
    # 由于我们拉伸了坐标，SDF 的梯度不再是 1，壁厚可能会变形
    # 这里使用经验值，建议在 CAD 中复测
    t_val = 0.5

    # 提取 Tube A (热)
    sdf_A = np.maximum(-(sdf_tpms - t_val), sdf_box)
    mesh_A = measure_to_mesh(sdf_A, res)
    mesh_A.visual.face_colors = [220, 50, 50, 255]

    # 提取 Tube B (冷)
    sdf_B = np.maximum((sdf_tpms + t_val), sdf_box)
    mesh_B = measure_to_mesh(sdf_B, res)
    mesh_B.visual.face_colors = [50, 50, 220, 255]

    return mesh_A, mesh_B

def measure_to_mesh(sdf, res):
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0, spacing=(res, res, res))
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except:
        return trimesh.Trimesh()

if __name__ == "__main__":
    print("生成流线型低压降换热器...")
    print("特性：")
    print("1. 流向(X)无阻挡：管路被拉直，没有特斯拉阀效应。")
    print("2. 截面(YZ)高密度：保持8mm微细管径，表面积巨大。")
    print("3. 微波浪：管路呈平缓S型，促进管内流体翻滚换热。")

    mesh_a, mesh_b = create_streamlined_exchanger(
        dims=(30, 20, 20), 
        resolution=1.0, 
        cross_section_period=10.0, # 横截面管子有多密 (10mm)
        flow_period=150.0,         # 纵向拉伸程度 (150mm，拉得很长)
        wavy_amplitude=2.0         # 只有2mm的轻微摆动
    )
    
    combined = trimesh.util.concatenate([mesh_a, mesh_b])
    combined.export("Streamlined_Channels.stl")
    print("完成。")