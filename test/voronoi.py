import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi
from noise import pnoise3
from scipy.interpolate import splprep, splev
from matplotlib.path import Path

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2

def occ_make_extruded_shape_bspl(smooth_poly, height=10.0):
    if not np.allclose(smooth_poly[0], smooth_poly[-1]):
        smooth_poly = np.vstack([smooth_poly, smooth_poly[0]])

    n = len(smooth_poly)
    array = TColgp_Array1OfPnt(1, n)
    for i in range(n):
        x, y = smooth_poly[i]
        array.SetValue(i + 1, gp_Pnt(x, y, 0))

    # ✅ 构造闭合 B 样条曲线
    bspline_builder = GeomAPI_PointsToBSpline(array, 3, 8, GeomAbs_C2, 1e-6)
    bspline = bspline_builder.Curve()

    # ✅ 创建边、线框、面并挤出
    edge = BRepBuilderAPI_MakeEdge(bspline).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    face = BRepBuilderAPI_MakeFace(wire).Face()
    vec = gp_Vec(0, 0, height)
    return BRepPrimAPI_MakePrism(face, vec).Shape()

def occ_make_extruded_shape_bspl_closed(smooth_poly, height=10.0):
    # ✅ 只参考点，不管首尾是否相连，直接构造闭合曲线
    n = len(smooth_poly)
    array = TColgp_Array1OfPnt(1, n)
    for i in range(n):
        x, y = smooth_poly[i]
        array.SetValue(i + 1, gp_Pnt(x, y, 0))

    # ✅ 构造 B 样条，设置为闭合（periodic），不严格插值
    bspline_builder = GeomAPI_PointsToBSpline(array, 3, 8, GeomAbs_C2, 1e-2)
    bspline = bspline_builder.Curve()

    # 强制设置为周期性闭合
    if not bspline.IsClosed() or not bspline.IsPeriodic():
        bspline.SetPeriodic()

    # ✅ 构造挤出体
    edge = BRepBuilderAPI_MakeEdge(bspline).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    if not BRepBuilderAPI_MakeFace(wire).IsDone():
        print("❌ Failed to make face from wire.")
        return None

    face = BRepBuilderAPI_MakeFace(wire).Face()
    shape = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, height)).Shape()
    return shape


# 导出 STEP 文件
def export_to_step(shape, filename="voronoi_cell.stp"):
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filename)
    if status == IFSelect_RetDone:
        print(f"✅ STEP file saved to {filename}")
    else:
        print("❌ Failed to write STEP file.")


def generate_perlin_density(t, res, scale=10.0):
    density = np.zeros((res, res))
    for i in range(res):
        for j in range(res):
            x = i / res * scale
            y = j / res * scale
            density[i, j] = pnoise3(x, y, t)
    density = density - density.min()
    density = density / density.max()
    return density

def move_points_by_density(points, density, bounds, res, dt):
    h, w = density.shape
    grad_y, grad_x = np.gradient(density)

    xmin, xmax, ymin, ymax = bounds
    for i in range(len(points)):
        px, py = points[i]
        xi = int((px - xmin) / (xmax - xmin) * (w - 1))
        yi = int((py - ymin) / (ymax - ymin) * (h - 1))
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)
        # gx = grad_x[yi, xi]
        # gy = grad_y[yi, xi]

        gx = grad_x[yi, xi]
        gy = grad_y[yi, xi]
        gx = gx / (np.sqrt(gx**2 + gy**2) + 1e-8)
        gy = gy / (np.sqrt(gx**2 + gy**2) + 1e-8)
        strength = density[yi, xi] ** 2  # 密度越大，移动越快

        points[i, 0] += gx * dt * strength * 3.0
        points[i, 1] += gy * dt * strength * 3.0
        points[i, 0] += gx * dt
        points[i, 1] += gy * dt

    points[:, 0] = np.clip(points[:, 0], xmin, xmax)
    points[:, 1] = np.clip(points[:, 1], ymin, ymax)
    return points

def relaxed_voronoi(points, bounds, iterations):
    xmin, xmax, ymin, ymax = bounds
    centroids = []
    for _ in range(iterations):
        try:
            vor = Voronoi(points)
        except:
            return points, None, []
        new_points = []
        centroids.clear()  # ✅ 每轮清空，只保留最后一轮的质心
        for i, region_index in enumerate(vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                new_points.append(points[i])
                centroids.append(points[i])
                continue
            polygon = [vor.vertices[j] for j in region if j >= 0]
            if len(polygon) == 0:
                new_points.append(points[i])
                centroids.append(points[i])
                continue
            centroid = np.mean(polygon, axis=0)
            centroid[0] = np.clip(centroid[0], xmin, xmax)
            centroid[1] = np.clip(centroid[1], ymin, ymax)
            new_points.append(centroid)
            centroids.append(centroid)
        points = np.array(new_points)
    try:
        return points, Voronoi(points), centroids
    except:
        return points, None, centroids

def chaikin_smoothing(points, iterations=4):
    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            new_points += [Q, R]
        new_points.append(new_points[0])  # close the loop
        points = np.array(new_points)
    return points

def shrink_polygon(points, factor=0.85):
    center = points.mean(axis=0)
    shrunk = center + (points - center) * factor
    return shrunk

def main():
    # 参数配置
    num_points = 600
    num_relaxations = 6
    bounds = [0, 100, 0, 100]
    res = 200
    scale = 1.0
    speed = 0.05
    dt = 0.5
    frames = 300
    interval = 100

    # 初始化点集
    points = np.random.rand(num_points, 2)
    points[:, 0] *= (bounds[1] - bounds[0])
    points[:, 1] *= (bounds[3] - bounds[2])

    # 初始化画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')

    def update(frame):
        nonlocal points

        ax.clear()
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')
        ax.set_title("Smooth Voronoi Driven by Perlin Density Field")

        # 📌 密度图生成与归一化
        t = frame * speed
        density = generate_perlin_density(t, res, scale=scale)
        density = (density - density.min()) / (density.max() - density.min())
        density = density ** 3.0
        ax.imshow(density, cmap='viridis', extent=bounds, origin='lower', alpha=0.85)

        # 📌 梯度预计算（全局只算一次）
        grad_y, grad_x = np.gradient(density)
        h, w = density.shape
        xmin, xmax, ymin, ymax = bounds

        # 📌 点移动
        points = move_points_by_density(points, density, bounds, res, dt)

        # 📌 Voronoi 构造与质心
        relaxed_points, vor, centroids = relaxed_voronoi(points, bounds, num_relaxations)

        # 📌 绘制细胞中心点
        if centroids:
            centroids = np.array(centroids)
            ax.scatter(centroids[:, 0], centroids[:, 1], color='blue', s=5, zorder=5)

            # 🔴 每个中心点绘制红色箭头（密度梯度方向）
            for cx, cy in centroids:
                xi = int((cx - xmin) / (xmax - xmin) * (w - 1))
                yi = int((cy - ymin) / (ymax - ymin) * (h - 1))
                xi = np.clip(xi, 0, w - 1)
                yi = np.clip(yi, 0, h - 1)

                gx = grad_x[yi, xi]
                gy = grad_y[yi, xi]

                norm = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
                dx = gx / norm * 3.0
                dy = gy / norm * 3.0
                ax.arrow(cx, cy, dx, dy, head_width=1.0, head_length=1.5, fc='red', ec='red', linewidth=0.8)

        # 📌 绘制 Voronoi 区域轮廓和平滑边界
        if vor:
            for region_idx in vor.point_region:
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    continue

                polygon = [vor.vertices[i] for i in region]
                poly = np.array(polygon)
                if len(poly) < 3:
                    continue

                # 原始边界绘制（浅线）
                ax.fill(*zip(*poly), edgecolor='black', facecolor='none', linewidth=0.1)

                # 📌 平滑 + 收缩
                poly = np.vstack([poly, poly[0]])  # 闭合
                smooth_poly = chaikin_smoothing(poly, iterations=4)
                smooth_poly = shrink_polygon(smooth_poly, factor=0.7)

                # 📌 额外：沿密度梯度方向的垂直方向压缩 smooth_poly
                center = smooth_poly.mean(axis=0)
                xi = int((center[0] - xmin) / (xmax - xmin) * (w - 1))
                yi = int((center[1] - ymin) / (ymax - ymin) * (h - 1))
                xi = np.clip(xi, 0, w - 1)
                yi = np.clip(yi, 0, h - 1)

                gx = grad_x[yi, xi]
                gy = grad_y[yi, xi]
                norm = np.sqrt(gx**2 + gy**2) + 1e-8
                gx /= norm
                gy /= norm
                nx, ny = -gy, gx  # 垂直方向
                squash_factor = 0.7  # 压缩强度

                for i in range(len(smooth_poly)):
                    v = smooth_poly[i] - center
                    d = v[0] * nx + v[1] * ny
                    offset = np.array([nx, ny]) * d * (1 - squash_factor)
                    smooth_poly[i] = smooth_poly[i] - offset

                # 📌 绘制最终变形后的边界
                path = Path(poly)
                if np.all(path.contains_points(smooth_poly)):
                    ax.plot(smooth_poly[:, 0], smooth_poly[:, 1], color='black', linewidth=0.8)

                    # ✅ 第 1 帧导出一个 Voronoi cell
                    compound = TopoDS_Compound()
                    builder = BRep_Builder()
                    builder.MakeCompound(compound)

                from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
                from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

                if frame == 1:
                    height = 10.0
                    num_saved = 0
                    cell_shapes = []

                    for region_idx in vor.point_region:
                        region = vor.regions[region_idx]
                        if -1 in region or len(region) == 0:
                            continue

                        polygon = [vor.vertices[i] for i in region]
                        poly = np.array(polygon)
                        if len(poly) < 3:
                            continue

                        poly = np.vstack([poly, poly[0]])  # 闭合
                        smooth_poly = chaikin_smoothing(poly, iterations=4)
                        smooth_poly = shrink_polygon(smooth_poly, factor=0.7)

                        center = smooth_poly.mean(axis=0)
                        xi = int((center[0] - xmin) / (xmax - xmin) * (w - 1))
                        yi = int((center[1] - ymin) / (ymax - ymin) * (h - 1))
                        xi = np.clip(xi, 0, w - 1)
                        yi = np.clip(yi, 0, h - 1)
                        gx = grad_x[yi, xi]
                        gy = grad_y[yi, xi]
                        norm = np.sqrt(gx**2 + gy**2) + 1e-8
                        gx /= norm
                        gy /= norm
                        nx, ny = -gy, gx
                        squash_factor = 0.7
                        for i in range(len(smooth_poly)):
                            v = smooth_poly[i] - center
                            d = v[0] * nx + v[1] * ny
                            offset = np.array([nx, ny]) * d * (1 - squash_factor)
                            smooth_poly[i] -= offset

                        path = Path(poly)
                        if np.all(path.contains_points(smooth_poly)):
                            shape = occ_make_extruded_shape_bspl_closed(smooth_poly, height=height)
                            if shape:
                                cell_shapes.append(shape)
                                num_saved += 1

                    # ✅ 创建上下两个面板
                    panel_thickness = 2.0
                    panel_bottom = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), gp_Pnt(100, 100, panel_thickness)).Shape()
                    panel_top    = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, height), gp_Pnt(100, 100, height + panel_thickness)).Shape()

                    # # ✅ 把所有 cell 和底部面板融合
                    # fused_bottom = panel_bottom
                    # for shape in cell_shapes:
                    #     fused_bottom = BRepAlgoAPI_Fuse(fused_bottom, shape).Shape()

                    # # ✅ 再将顶部也融合（与同样的柱体）
                    # fused_full = BRepAlgoAPI_Fuse(fused_bottom, panel_top).Shape()

                    # ✅ 导出整体结构
                    export_to_step(panel_bottom, filename="box.stp")
                    print(f"✅ 导出整体结构，共包含 {num_saved} 个 Voronoi 柱体与上下面板融合")
                else:
                    ax.plot(poly[:, 0], poly[:, 1], color='gray', linewidth=0.5)
                    
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.show()


if __name__ == "__main__":
    main()
