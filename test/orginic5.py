import taichi as ti
import numpy as np
import math

# 初始化 Taichi
ti.init(arch=ti.gpu) 

# --- 1. 仿真参数 ---
NX, NY, NZ = 128, 64, 64
STEPS = 4000

tau = 0.6
omega = 1.0 / tau
u_inlet = 0.1

# --- 2. 数据结构 ---
f = ti.Vector.field(19, dtype=float, shape=(NX, NY, NZ))
f_new = ti.Vector.field(19, dtype=float, shape=(NX, NY, NZ))
rho = ti.field(dtype=float, shape=(NX, NY, NZ))
vel = ti.Vector.field(3, dtype=float, shape=(NX, NY, NZ))
solid = ti.field(dtype=int, shape=(NX, NY, NZ))

# LBM 常量 Field
w = ti.field(dtype=float, shape=19)
e = ti.Vector.field(3, dtype=int, shape=19)
reverse = ti.field(dtype=int, shape=19)
image_buffer = ti.Vector.field(3, dtype=float, shape=(NX, NY))

# Numpy 常量 (CPU 端)
w_np = np.array([1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
e_np = np.array([[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                 [1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0], [1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1], [0,1,1], [0,1,-1], [0,-1,1], [0,-1,-1]], dtype=np.int32)
reverse_np = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15], dtype=np.int32)

# --- 修复点 1: 在 Python 范围初始化常量，而不是 Kernel 内 ---
def init_constants():
    # 这部分运行在 CPU (Python)，直接把值塞给 Taichi Field
    # Taichi 会自动把数据同步到 GPU
    for i in range(19):
        w[i] = w_np[i]
        reverse[i] = int(reverse_np[i])
        e[i] = [int(e_np[i,0]), int(e_np[i,1]), int(e_np[i,2])]

# --- 3. 几何生成内核 (有机 TPMS) ---
@ti.func
def pseudo_noise(px, py, pz, freq):
    return (ti.sin(px * freq) + ti.sin(py * freq * 1.3) + ti.sin(pz * freq * 1.7))

@ti.kernel
def init_geometry_field():
    # 仅负责几何与流场初始化
    period = 20.0
    warp_str = 4.0
    warp_freq = 0.15
    thickness = 0.4
    
    for i, j, k in solid:
        x, y, z = float(i), float(j), float(k)
        
        # 域扭曲
        dx = pseudo_noise(x, y, z, warp_freq) * warp_str
        dy = pseudo_noise(y, z, x, warp_freq) * warp_str
        dz = pseudo_noise(z, x, y, warp_freq) * warp_str
        
        xw, yw, zw = x + dx, y + dy, z + dz
        
        # Schwarz D
        factor = 2 * math.pi / period
        gx, gy, gz = xw * factor, yw * factor, zw * factor
        
        sdf = (ti.sin(gx)*ti.sin(gy)*ti.sin(gz) + 
               ti.sin(gx)*ti.cos(gy)*ti.cos(gz) + 
               ti.cos(gx)*ti.sin(gy)*ti.cos(gz) + 
               ti.cos(gx)*ti.cos(gy)*ti.sin(gz))
        
        if ti.abs(sdf) < thickness:
            solid[i, j, k] = 1
        else:
            solid[i, j, k] = 0
            
        # 边界封闭
        if j == 0 or j == NY-1 or k == 0 or k == NZ-1:
            solid[i, j, k] = 1

        # 初始化流场
        rho[i, j, k] = 1.0
        vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        for d in range(19):
            f[i, j, k][d] = w[d] * 1.0

# --- 4. LBM 求解器 ---
@ti.kernel
def lbm_step():
    # 1. 碰撞 & 流动
    for i, j, k in f:
        if solid[i, j, k] == 0:
            r = rho[i, j, k]
            v = vel[i, j, k]
            v_sq = v.norm_sqr()
            
            for d in range(19):
                ed_v = e[d].dot(v)
                feq = w[d] * r * (1 + 3*ed_v + 4.5*ed_v*ed_v - 1.5*v_sq)
                
                # Streaming 到 f_new
                new_pos = ti.Vector([i, j, k]) + e[d]
                if 0 <= new_pos.x < NX and 0 <= new_pos.y < NY and 0 <= new_pos.z < NZ:
                    f_new[new_pos.x, new_pos.y, new_pos.z][d] = f[i, j, k][d] * (1 - omega) + feq * omega

    # 2. 边界条件
    # A. 入口 (Inlet X=0)
    for j, k in ti.ndrange(NY, NZ):
        if solid[0, j, k] == 0:
            r = 1.0
            v = ti.Vector([u_inlet, 0.0, 0.0])
            v_sq = v.norm_sqr()
            for d in range(19):
                 ed_v = e[d].dot(v)
                 # 强制设定入口分布
                 f_new[0, j, k][d] = w[d] * r * (1 + 3*ed_v + 4.5*ed_v*ed_v - 1.5*v_sq)

    # B. 出口 (Outlet X=End) - 零梯度
    for j, k in ti.ndrange(NY, NZ):
        if solid[NX-1, j, k] == 0:
            for d in range(19):
                f_new[NX-1, j, k][d] = f_new[NX-2, j, k][d]

    # 3. 更新 f, rho, vel
    for i, j, k in f:
        # 将 f_new 复制回 f
        for d in range(19):
            f[i, j, k][d] = f_new[i, j, k][d]
            
        if solid[i, j, k] == 1:
            vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        else:
            current_f = f[i, j, k]
            local_rho = 0.0
            local_vel = ti.Vector([0.0, 0.0, 0.0])
            for d in range(19):
                val = current_f[d]
                local_rho += val
                local_vel += e[d] * val
            
            rho[i, j, k] = local_rho
            if local_rho > 0:
                vel[i, j, k] = local_vel / local_rho
            
            # 简单反弹修正 (Half-way bounce back simplified logic)
            # 在更新完宏观量后，检查是否需要修正流向墙壁的分量
            # 为了代码稳定性，这里只做基础更新，不覆盖 f

# --- 5. 渲染内核 ---
@ti.kernel
def render_slice(z_index: int):
    for i, j in image_buffer:
        v = vel[i, j, z_index].norm()
        if solid[i, j, z_index] == 1:
            image_buffer[i, j] = ti.Vector([0.3, 0.3, 0.3]) 
        else:
            val = v / (u_inlet * 2.5) # 调整颜色敏感度
            image_buffer[i, j] = ti.Vector([val, val*0.5, 1.0-val])

def main():
    print("初始化 LBM 常量 (CPU)...")
    init_constants()
    
    print("初始化 3D 有机几何 (GPU)...")
    init_geometry_field()
    
    window = ti.ui.Window("3D Organic TPMS Flow", (NX*4, NY*4))
    canvas = window.get_canvas()
    
    z_slice = NZ // 2
    print("开始仿真循环... 按 W/S 移动切片")
    
    while window.running:
        for _ in range(10): # 每帧计算 10 步
            lbm_step()
        
        if window.is_pressed('w'): z_slice = min(NZ-1, z_slice + 1)
        if window.is_pressed('s'): z_slice = max(0, z_slice - 1)
            
        render_slice(z_slice)
        
        # 旋转图像以便更符合直觉
        img = image_buffer.to_numpy()
        img = np.transpose(img, (1, 0, 2))[::-1, :] 
        canvas.set_image(img)
        window.show()

if __name__ == "__main__":
    main()