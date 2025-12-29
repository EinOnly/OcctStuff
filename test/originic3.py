import numpy as np
import trimesh

def create_visual_guides_nozzle(
    dims=(200, 150, 20), 
    num_seeds=(15, 5),     # (Y方向数量, Z方向数量) - 种子点密度
    line_radius=0.8,       # 导向线的粗细
    steps=100              # 沿路径的采样点数
):
    print(f"1. 初始化导向生成器 {dims}...")
    w, h, l = dims
    
    # --- 步骤 1: 在入口处 (X负半轴) 定义种子点 ---
    # 我们在 Y-Z 平面上生成一个网格点作为起点
    y_seeds = np.linspace(-h/2 * 0.9, h/2 * 0.9, num_seeds[0]) # 稍微内缩一点避免贴边
    z_seeds = np.linspace(-l/2 * 0.9, l/2 * 0.9, num_seeds[1])
    yy_s, zz_s = np.meshgrid(y_seeds, z_seeds)
    
    # 所有种子点的起始 X 坐标
    x_start = -w/2
    
    # 将网格展平为点列表: [[x0, y0, z0], [x0, y1, z1], ...]
    seeds = np.stack([np.full_like(yy_s, x_start), yy_s, zz_s], axis=-1).reshape(-1, 3)
    print(f"   - 生成了 {len(seeds)} 条流线的起点。")

    # --- 步骤 2: 定义喷管流场规则 (与 TPMS 生成保持一致) ---
    # 规则回顾：
    # 压缩因子 C(x) = 1.0 + (x_norm + 1.0) * 0.75
    # 流函数守恒：Y_curr * C(x_curr) = Y_start * C(x_start)
    # 所以：Y_curr = Y_start * [C(x_start) / C(x_curr)]
    
    def get_compression(x_val, width):
        x_norm = x_val / (width/2)
        # 这里必须与上一段 TPMS 代码中的公式完全一致
        return 1.0 + (x_norm + 1.0) * 0.75

    # 计算起点的压缩因子
    comp_start = get_compression(x_start, w)
    
    # 定义沿 X 轴的路径点
    x_path = np.linspace(-w/2, w/2, steps)
    
    streamlines_meshes = []
    print("2. 追踪流线路径并生成实体管...")

    # --- 步骤 3: 追踪每一条流线 ---
    for i, seed in enumerate(seeds):
        y0, z0 = seed[1], seed[2]
        path_points = []
        
        for x_curr in x_path:
            # 计算当前位置的压缩因子
            comp_curr = get_compression(x_curr, w)
            
            # 根据流函数守恒计算当前的 Y 和 Z
            # 压缩比 = 起点压缩 / 当前压缩
            ratio = comp_start / comp_curr
            
            y_curr = y0 * ratio
            z_curr = z0 * ratio
            
            path_points.append([x_curr, y_curr, z_curr])
            
        path_points = np.array(path_points)
        
        # --- 步骤 4: 将路径转换为实体细管 (用于可视化) ---
        # 如果路径太短或点太少，跳过
        if len(path_points) < 2: continue

        try:
            # 创建一个圆形截面用于扫掠
            # 截面要在 XY 平面，然后沿着路径 Z 轴扫掠，trimesh 会自动处理方向
            angle = np.linspace(0, 2*np.pi, 12) # 12边形近似圆形
            theta = np.linspace(0, 2*np.pi, 12)
            circle_shape = np.column_stack((np.cos(theta), np.sin(theta))) * line_radius
            
            # 沿着路径扫掠生成管子
            tube = trimesh.creation.sweep(circle_shape, path_points)
            streamlines_meshes.append(tube)
        except Exception as e:
            # 有时路径曲率过大可能导致生成失败，忽略该线条
            # print(f"Warning: Skipped line {i} due to generation error: {e}")
            pass
            
    print(f"3. 合并 {len(streamlines_meshes)} 条导向线...")
    if not streamlines_meshes:
        print("Error: 没有生成任何线条。")
        return trimesh.Trimesh()
        
    combined_guides = trimesh.util.concatenate(streamlines_meshes)
    # 给导向线一个显眼的颜色 (例如亮黄色)
    combined_guides.visual.face_colors = [255, 220, 50, 255]
    
    return combined_guides

if __name__ == "__main__":
    # 确保这里的尺寸与你生成 TPMS 的尺寸一致
    W, H, L = 200, 150, 20
    
    print("开始生成可视化导向流线...")
    guide_mesh = create_visual_guides_nozzle(
        dims=(W, H, L),
        num_seeds=(20, 4), # Y方向20根线，Z方向4层线
        line_radius=0.5    # 线条做细一点，看起来更像骨架
    )
    
    output_name = "Visual_Guides_Nozzle.stl"
    guide_mesh.export(output_name)
    print(f"完成。请查看 {output_name}")
    print("提示：在 3D 软件中，将此 STL 与之前生成的 TPMS STL 同时打开叠加显示。")