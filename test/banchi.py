import taichi as ti
import numpy as np
import math

# 初始化 Taichi，尝试使用 GPU (Cuda/Vulkan/Metal)，如果不支持则回退到 CPU
ti.init(arch=ti.gpu)

# --- 配置参数 (可以修改这些参数来观察不同的"演化"行为) ---
WIDTH, HEIGHT = 800, 600
PARTICLE_COUNT = 200000      # 粒子数量，越多越清晰，但对显卡要求越高
EVAPORATION_RATE = 0.95      # 轨迹消散速度 (0-1)
DIFFUSION_RATE = 0.5         # 扩散速度
SENSOR_ANGLE = 45.0 * (math.pi / 180.0)  # 传感器探测角度
SENSOR_DIST = 10.0           # 传感器探测距离
TURN_SPEED = 20.0            # 转向速度
MOVE_SPEED = 1.0             # 移动速度

# --- 数据结构定义 ---
# 屏幕像素 (用于显示)
pixels = ti.Vector.field(3, dtype=float, shape=(WIDTH, HEIGHT))
# 费洛蒙网格 (用于记录轨迹)
grid = ti.field(dtype=float, shape=(WIDTH, HEIGHT))
# 粒子位置
pos = ti.Vector.field(2, dtype=float, shape=PARTICLE_COUNT)
# 粒子角度
angle = ti.field(dtype=float, shape=PARTICLE_COUNT)

@ti.kernel
def init_particles():
    """初始化粒子位置：这里设置为随机分布，也可以改为从中心爆发"""
    for i in range(PARTICLE_COUNT):
        # 随机分布在屏幕中心的一个圆内
        random_angle = ti.random() * math.pi * 2
        random_r = ti.sqrt(ti.random()) * (HEIGHT * 0.4)
        pos[i] = ti.Vector([WIDTH/2 + ti.cos(random_angle)*random_r, 
                            HEIGHT/2 + ti.sin(random_angle)*random_r])
        angle[i] = ti.random() * math.pi * 2

@ti.func
def sense(particle_idx, angle_offset):
    """感知函数：探测前方特定角度的费洛蒙浓度"""
    sensor_angle = angle[particle_idx] + angle_offset
    # 计算探测点坐标
    sensor_x = pos[particle_idx][0] + ti.cos(sensor_angle) * SENSOR_DIST
    sensor_y = pos[particle_idx][1] + ti.sin(sensor_angle) * SENSOR_DIST
    
    # 边界处理（转换为整数索引）
    ix = ti.cast(sensor_x, ti.i32)
    iy = ti.cast(sensor_y, ti.i32)
    
    # 简单的边界限制
    val = 0.0
    if ix >= 0 and ix < WIDTH and iy >= 0 and iy < HEIGHT:
        val = grid[ix, iy]
    return val

@ti.kernel
def update_particles():
    """核心逻辑：移动和转向"""
    for i in range(PARTICLE_COUNT):
        # 1. 感知阶段 (左，中，右)
        w_left = sense(i, SENSOR_ANGLE)
        w_center = sense(i, 0)
        w_right = sense(i, -SENSOR_ANGLE)
        
        random_turn = (ti.random() - 0.5) * 0.2 # 增加一点随机扰动，模拟自然界的混沌
        
        # 2. 决策阶段 (根据浓度调整角度)
        if w_center > w_left and w_center > w_right:
            # 前方浓度最高，保持方向 (加微小扰动)
            angle[i] += random_turn
        elif w_center < w_left and w_center < w_right:
            # 前方浓度最低，随机大幅转向
            angle[i] += (ti.random() - 0.5) * 2 * TURN_SPEED * 0.1
        elif w_left > w_right:
            # 左边浓度高，向左转
            angle[i] += random_turn + TURN_SPEED * 0.1
        elif w_right > w_left:
            # 右边浓度高，向右转
            angle[i] += random_turn - TURN_SPEED * 0.1
        
        # 3. 移动阶段
        pos[i][0] += ti.cos(angle[i]) * MOVE_SPEED
        pos[i][1] += ti.sin(angle[i]) * MOVE_SPEED
        
        # 边界处理：碰到墙壁随机反弹
        if pos[i][0] < 0 or pos[i][0] >= WIDTH:
            pos[i][0] = ti.max(0, ti.min(pos[i][0], WIDTH-0.1))
            angle[i] = ti.random() * math.pi * 2
        if pos[i][1] < 0 or pos[i][1] >= HEIGHT:
            pos[i][1] = ti.max(0, ti.min(pos[i][1], HEIGHT-0.1))
            angle[i] = ti.random() * math.pi * 2
            
        # 4. 留下痕迹 (Deposit)
        ix = ti.cast(pos[i][0], ti.i32)
        iy = ti.cast(pos[i][1], ti.i32)
        grid[ix, iy] += 1.0 # 在当前位置增加费洛蒙强度

@ti.kernel
def process_grid():
    """网格处理：扩散(Diffusion) 和 衰减(Decay)"""
    for i, j in grid:
        # 简单的 3x3 均值模糊实现扩散
        sum_val = 0.0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = i + dx, j + dy
                # 边界检查
                if nx >= 0 and nx < WIDTH and ny >= 0 and ny < HEIGHT:
                    sum_val += grid[nx, ny]
        
        # 平均并应用衰减
        blur_val = sum_val / 9.0
        grid[i, j] = blur_val * EVAPORATION_RATE

@ti.kernel
def render():
    """将数据渲染到屏幕"""
    for i, j in pixels:
        val = grid[i, j]
        # 颜色映射：根据浓度显示颜色 (这里用青色/绿色风格)
        # 限制最大亮度，防止过曝
        brightness = ti.min(val * 0.5, 1.0)
        pixels[i, j] = ti.Vector([0.0, brightness, brightness * 0.5])

# --- 主程序 ---
def main():
    gui = ti.GUI("Physarum Slime Mold Evolution", res=(WIDTH, HEIGHT), fast_gui=True)
    init_particles()
    
    print(f"Start simulation with {PARTICLE_COUNT} particles...")
    print("Press ESC to exit.")

    frame_count = 0
    while gui.running:
        # 执行多次更新步骤以加快演化速度
        update_particles()
        process_grid()
        
        # 渲染
        render()
        gui.set_image(pixels)
        gui.show()
        
        frame_count += 1
        if frame_count % 100 == 0:
            # 偶尔重置一下费洛蒙过高的情况，或者可以在这里改变参数
            pass

if __name__ == "__main__":
    main()