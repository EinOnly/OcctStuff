import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

# ==========================================
# 1. 物理参数定义 (V1 原型机 - 40mm)
# ==========================================
DIAMETER = 40.0
RADIUS = DIAMETER / 2.0
ROTOR_TEETH = 48
SHELL_TEETH = 50
NUTATION_ANGLE = 3.0   # 章动角
GEAR_RATIO = SHELL_TEETH / (SHELL_TEETH - ROTOR_TEETH) # 25:1
TOOTH_HEIGHT = 1.5

# ==========================================
# 2. 几何生成函数 (生成网格数据)
# ==========================================
def generate_crown_mesh_data(teeth, r_out, r_in, color):
    """
    生成冠状齿轮的 顶点(vertexes) 和 面(faces) 数据
    用于 PyQtGraph 的 GLMeshItem
    """
    cols = 360  # 圆周分辨率
    rows = 2    # 径向分辨率 (内圈和外圈)
    
    verts = []
    faces = []
    
    # 1. 生成顶点
    # r_in 到 r_out
    rs = np.linspace(r_in, r_out, rows)
    thetas = np.linspace(0, 2*np.pi, cols, endpoint=False)
    
    for r in rs:
        for theta in thetas:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            # Z轴波动: Crown Gear 形状
            z = (TOOTH_HEIGHT / 2.0) * np.cos(teeth * theta)
            verts.append([x, y, z])
            
    verts = np.array(verts)
    
    # 2. 生成面索引 (连接顶点形成三角形)
    # 这是一个简单的网格连接算法
    n_theta = len(thetas)
    for r_idx in range(rows - 1):
        for t_idx in range(n_theta):
            # 当前层的两个点
            p1 = r_idx * n_theta + t_idx
            p2 = r_idx * n_theta + (t_idx + 1) % n_theta
            # 下一层的两个点
            p3 = (r_idx + 1) * n_theta + (t_idx + 1) % n_theta
            p4 = (r_idx + 1) * n_theta + t_idx
            
            # 两个三角形组成一个矩形面
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
            
    faces = np.array(faces)
    
    # 生成颜色数组 (让它看起来有立体感)
    colors = np.ones((len(faces), 4))
    colors[:, 0] = color[0] # R
    colors[:, 1] = color[1] # G
    colors[:, 2] = color[2] # B
    colors[:, 3] = color[3] # Alpha
    
    return verts, faces, colors

# ==========================================
# 3. PyQt 主窗口类
# ==========================================
class SimulationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 窗口设置
        self.setWindowTitle(f"Nutating Gear Simulation (PyQt) - Ratio {int(GEAR_RATIO)}:1")
        self.resize(1000, 800)
        
        # --- A. 创建 3D 视图控件 ---
        self.view = gl.GLViewWidget()
        self.setCentralWidget(self.view)
        
        # 设置摄像机距离
        self.view.setCameraPosition(distance=80, elevation=30)
        # 添加网格平面作为参考
        g = gl.GLGridItem()
        g.setSize(100, 100)
        self.view.addItem(g)
        
        # --- B. 创建转子 (蓝色) ---
        verts_r, faces_r, cols_r = generate_crown_mesh_data(
            ROTOR_TEETH, RADIUS, RADIUS-5, (0.2, 0.2, 1.0, 0.8)
        )
        self.rotor_mesh = gl.GLMeshItem(
            vertexes=verts_r, faces=faces_r, faceColors=cols_r, 
            smooth=True, drawEdges=True, edgeColor=(0,0,0,0.2), shader='balloon'
        )
        self.view.addItem(self.rotor_mesh)
        
        # 添加一个标记点(小球)在转子上，证明它不自转
        self.marker_pos = np.array([RADIUS, 0, 0]) # 边缘一点
        self.marker = gl.GLScatterPlotItem(pos=self.marker_pos, color=(1,1,0,1), size=10, pxMode=True)
        self.view.addItem(self.marker)

        # --- C. 创建外壳 (红色) ---
        # 为了看清内部，我们用线框模式或者半透明
        verts_s, faces_s, cols_s = generate_crown_mesh_data(
            SHELL_TEETH, RADIUS+2, RADIUS-2, (1.0, 0.2, 0.2, 0.3)
        )
        self.shell_mesh = gl.GLMeshItem(
            vertexes=verts_s, faces=faces_s, faceColors=cols_s, 
            smooth=False, drawEdges=True, edgeColor=(1,0,0,0.5), shader='balloon'
        )
        # 把外壳抬高一点，或者翻转一下以模拟扣合
        self.shell_mesh.translate(0, 0, 5) 
        self.view.addItem(self.shell_mesh)

        # --- D. 动画定时器 ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(20) # 50 FPS
        
        self.frame_count = 0

    def update_animation(self):
        self.frame_count += 1
        t = self.frame_count * 0.05
        
        # 1. 计算角度
        # 磁场旋转一圈 (0 -> 360)
        angle_mag = (t * 2 * np.pi) % (2 * np.pi)
        angle_mag_deg = np.degrees(angle_mag)
        
        # 外壳应该转过的角度 (减速后)
        angle_shell = angle_mag / GEAR_RATIO
        angle_shell_deg = np.degrees(angle_shell)
        
        # --- 2. 变换转子 (Rotor) ---
        # 逻辑：固定不自转。
        # 变换顺序：重置 -> 绕Z转到磁场方向 -> 绕X倾斜 -> 绕Z转回 (抵消自转)
        
        self.rotor_mesh.resetTransform()
        self.rotor_mesh.rotate(angle_mag_deg, 0, 0, 1)  # Z+
        self.rotor_mesh.rotate(NUTATION_ANGLE, 1, 0, 0) # X (倾斜)
        self.rotor_mesh.rotate(-angle_mag_deg, 0, 0, 1) # Z- (取消自转)
        
        # 更新标记点的位置 (手动应用矩阵变换，因为ScatterPlotItem不方便直接transform)
        # 这里为了演示简单，我们只更新Mesh，你可以看到蓝色的网格本身没有旋转，只是在波动
        
        # --- 3. 变换外壳 (Shell) ---
        # 简单自转
        self.shell_mesh.resetTransform()
        self.shell_mesh.translate(0, 0, 5) # 保持位置
        self.shell_mesh.rotate(angle_shell_deg, 0, 0, 1)

# ==========================================
# 4. 程序入口
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimulationWindow()
    window.show()
    sys.exit(app.exec_())