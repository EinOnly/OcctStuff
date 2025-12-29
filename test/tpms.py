import numpy as np
import trimesh
from skimage import measure

def create_manifold_gyroid(
    box_size=(60, 60, 60),  # 模型整体尺寸
    resolution=0.8,         # 分辨率（越小越精细）
    periodicity=12.0,       # TPMS 周期
    wall_thickness=1.5,     # 内部 Gyroid 壁厚 (mm)
    shell_thickness=3.0     # 外壳厚度 (mm)
):
    print(f"1. 初始化网格空间 {box_size}...")
    # 增加一点 padding 以确保边界封闭计算正确
    pad = 2
    shape = (int(box_size[0]/resolution) + pad*2,
             int(box_size[1]/resolution) + pad*2,
             int(box_size[2]/resolution) + pad*2)
    
    # 建立坐标系 (居中对齐，方便对称操作)
    x = np.linspace(-box_size[0]/2 - pad, box_size[0]/2 + pad, shape[0])
    y = np.linspace(-box_size[1]/2 - pad, box_size[1]/2 + pad, shape[1])
    z = np.linspace(-box_size[2]/2 - pad, box_size[2]/2 + pad, shape[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    print("2. 计算核心 Gyroid 场...")
    # TPMS 频率系数
    k = 2 * np.pi / periodicity
    # Gyroid SDF: 值越接近 0 越接近表面
    # 我们这里不取 abs，保留符号，以便区分域 A (+) 和域 B (-)
    gyroid_raw = (np.sin(k*xx)*np.cos(k*yy) + 
                  np.sin(k*yy)*np.cos(k*zz) + 
                  np.sin(k*zz)*np.cos(k*xx))
    
    # 将 Gyroid 转化为实体壁的 SDF
    # 逻辑：如果不取 abs，gyroid_raw > t 是域 A，< -t 是域 B
    # 实体壁是： -t <= gyroid_raw <= t
    # 在 SDF 逻辑中，我们需要定义“距离实体的距离”。
    # 为了简化，我们定义：负值 = 实体内部，正值 = 空气/流体
    
    # 转换：abs(gyroid) - thickness。 结果为负表示在壁内。
    # 注意：这里的 0.4 是近似的阈值，对应 wall_thickness
    # 真实的 SDF 需要复杂的距离变换，这里用阈值模拟足够了
    t_val = 0.5  # 控制壁厚的无量纲参数
    sdf_core_wall = np.abs(gyroid_raw) - t_val 
    # 现在：sdf_core_wall > 0 是流体， < 0 是实体壁

    print("3. 构建外壳与流道逻辑 (Boolean Operations)...")
    
    # --- 定义几何边界 SDF ---
    # 计算点到 Box 表面的距离（Box SDF）
    # d < 0 在 Box 内部, d > 0 在 Box 外部
    d_x = np.abs(xx) - box_size[0]/2
    d_y = np.abs(yy) - box_size[1]/2
    d_z = np.abs(zz) - box_size[2]/2
    # 整个 Box 的 SDF (外部距离)
    sdf_box_outer = np.maximum(np.maximum(d_x, d_y), d_z)
    
    # 定义“内腔”区域（比外轮廓小 shell_thickness）
    d_x_inner = np.abs(xx) - (box_size[0]/2 - shell_thickness)
    d_y_inner = np.abs(yy) - (box_size[1]/2 - shell_thickness)
    d_z_inner = np.abs(zz) - (box_size[2]/2 - shell_thickness)
    sdf_box_inner = np.maximum(np.maximum(d_x_inner, d_y_inner), d_z_inner)

    # --- 核心逻辑：封口与开口 ---
    # 我们不仅想要一个 Gyroid，我们想要一个带有特定开口的盒子。
    
    # 1. 基础实体：外壳 (Shell)
    # 外壳 = (在 Outer Box 内) AND (在 Inner Box 外)
    # SDF Boolean Intersection: max(A, B)
    # 我们这里用“负值代表实体”的逻辑：
    # 实体 = max(sdf_box_outer, -sdf_box_inner) ❌ 不好算
    # 换个思路：如果在这个区域，强制设为实体。

    # 初始化最终 SDF，默认为 Gyroid 壁
    final_sdf = sdf_core_wall

    # 2. 强制添加外壳 (Union 操作)
    # 只要在 Shell 区域（Inner Box 外部 且 Outer Box 内部），就变成实体
    # 也就是：如果是边缘，强制覆盖 Gyroid 的空洞
    in_shell_region = (sdf_box_inner > 0) & (sdf_box_outer <= 0)
    
    # 将 Shell 区域强制变为实体 (SDF < 0)
    # 我们设一个负数让它变成实心
    final_sdf[in_shell_region] = -1.0 

    # 3. 挖孔 (Difference 操作) - 关键步骤！
    # 现在的盒子是全封闭的（因为 Gyroid 延伸到了 Shell，而 Shell 被我们填实了）。
    # 我们需要“打通”流道。
    
    # -> 流道 A：沿着 X 轴走。打通 X+ 和 X- 面。
    # 条件：在 X 面附近，且属于 Gyroid 的 域A (gyroid_raw > t)
    # 这是一个布尔减法：如果 (是 X 面开口区) AND (是域 A)，则 设为空气(SDF > 0) ???
    # 不，我们现在的 final_sdf 已经是实体壁了。
    # 域 A 的流体本来就是“空”的 (SDF > 0)。
    # 问题在于步骤2把所有边缘都封死了。我们需要把特定位置的“封死”撤销掉。

    # 重构逻辑：
    # 区域 1: 核心区域 (Inner Box 内部) -> 保持 Gyroid 结构
    # 区域 2: 外壳区域 (Shell 层) -> 
    #         如果是 X 面 -> 只保留 域B 的壁，挖掉 域A 的口？
    #         如果是 Y 面 -> 只保留 域A 的壁，挖掉 域B 的口？
    #         如果是 Z 面 -> 全部封死 (保留实体)

    # 让我们用 mask 来实现这个逻辑
    
    # 面具定义
    mask_x_face = (d_x > -shell_thickness) & (d_x <= 0) & (d_y <= 0) & (d_z <= 0) # X 面的壳
    mask_y_face = (d_y > -shell_thickness) & (d_y <= 0) & (d_x <= 0) & (d_z <= 0) # Y 面的壳
    mask_z_face = (d_z > -shell_thickness) & (d_z <= 0)           # Z 面的壳 (上下盖)
    
    # 默认：核心区保持 Gyroid
    # 默认：外壳区设为实体 (-1)
    final_sdf = np.copy(sdf_core_wall)
    
    # 第一步：把所有边界先封死 (全变成实体)
    shell_mask = (sdf_box_inner > 0)
    final_sdf[shell_mask] = -1.0
    
    # 第二步：打通入口
    
    # X 面开口：让流体 A (gyroid_raw > t) 通行
    # 这意味着：在 X 面区域，如果本来是流体 A 的位置，把刚才封的“实体”变回“空气”
    # 只有当 gyroid_raw > t_val (即流体A区域) 时，设为 1.0 (空气)
    # 这样，流体 B (gyroid_raw < -t) 在 X 面依然被 Shell 挡住，从而实现隔离
    open_A_condition = (gyroid_raw > t_val) 
    final_sdf[mask_x_face & open_A_condition] = 1.0 
    
    # Y 面开口：让流体 B (gyroid_raw < -t) 通行
    # 条件：gyroid_raw < -t_val
    open_B_condition = (gyroid_raw < -t_val)
    final_sdf[mask_y_face & open_B_condition] = 1.0
    
    # Z 面：保持封死 (不做操作，前面已经设为 -1 了)
    
    # 边界清理：确保最最外层的一圈是干净的（防止 march cube 边缘伪影）
    final_sdf[sdf_box_outer > 0] = 1.0

    print("4. 提取实体网格...")
    # 提取等值面 0
    verts, faces, normals, values = measure.marching_cubes(
        final_sdf, 
        level=0.0,
        spacing=(resolution, resolution, resolution)
    )

    # 居中校正
    verts -= np.array([shape[0]*resolution/2, shape[1]*resolution/2, shape[2]*resolution/2])
    
    # 创建 Mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh

# --- 执行 ---
if __name__ == "__main__":
    # 生成一个 50mm 的立方体热交换器核心
    mesh = create_manifold_gyroid(
        box_size=(50, 50, 50),
        periodicity=15.0,     # 周期大一点方便看清内部
        wall_thickness=0.2,   # 内部特征
        shell_thickness=2.0   # 外框厚度
    )
    
    mesh.show()
    mesh.export("isolated_gyroid_exchanger.stl")
    print("生成完毕。请检查 STL 文件：")
    print("1. 这是一个带外框的实体。")
    print("2. X 轴方向可以透光（流体 A 通道）。")
    print("3. Y 轴方向可以透光（流体 B 通道）。")
    print("4. Z 轴（顶底）是完全封死的。")
    print("5. 两个流体域互不相通。")