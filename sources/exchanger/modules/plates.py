import os
import time

# ========= Geometric primitives (points, directions, axes, vectors, circles) ==========
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax2, gp_Circ, gp_Trsf       # Basic geometry types

# ========= Curve & interpolation tools ==========
from OCC.Core.BRepCheck import BRepCheck_Analyzer                     # Check shape validity
from OCC.Core.GC import GC_MakeCircle                                 # Create geometric circles
from OCC.Core.TColgp import TColgp_Array1OfPnt                        # Array of points for spline
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline                  # Interpolate B-spline curve
from OCC.Core.GeomAbs import GeomAbs_C2                               # Continuity type for splines

from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
# ========= Shape construction (edge, wire, face, solid) ==========
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,     # Create an edge from a curve
    BRepBuilderAPI_MakeWire,     # Build a wire from one or more edges
    BRepBuilderAPI_MakeFace,     # Create a planar face from a wire
    BRepBuilderAPI_MakeSolid,    # Build a solid from a shell
    BRepBuilderAPI_Transform,    # Transform a shape (translation, rotation)
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism                # Extrude a face along a direction

# ========= Boolean operations / thickening / filleting ==========
from OCC.Core.BRepAlgoAPI import (
    BRepAlgoAPI_Cut,             # Boolean subtraction
    BRepAlgoAPI_Fuse,            # Boolean fusion (union)
)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid, BRepOffsetAPI_MakeOffset       # Generate thick solids from shells
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet           # Create edge fillets
# ========= Topology definitions and shape casting ==========
from OCC.Core.TopoDS import (
    TopoDS_Shape,                # Base shape class
    TopoDS_Compound,             # Compound of multiple shapes
    TopoDS_Shell,                # Shell (connected faces)
    topods,                      # Casting functions (e.g., to Face, Edge)
)

# ========= Shape structure & builders ==========
from OCC.Core.BRep import BRep_Builder                                # Low-level shape construction tools
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Shell           # Fix shape issues (e.g., gaps, overlaps)

# ========= Topology traversal and classification ==========
from OCC.Core.TopExp import TopExp_Explorer                           # Topology explorer (iterate faces/edges)
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE     # Shape type enums: face/edge

# ========= Shape relationships & collections ==========
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,                        # Map shape → list of shapes
    TopTools_ListOfShape,                                             # Generic shape list
)

# ========= Bounding box and geometry bounds ==========
from OCC.Core.Bnd import Bnd_Box                                      # Axis-aligned bounding box
from OCC.Core.BRepBndLib import brepbndlib                            # Compute bounding box of shapes

# ========= STEP file export ==========
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs # STEP writer and mode
from OCC.Core.IFSelect import IFSelect_RetDone                        # Result status for export

from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely.wkt import dumps as wkt_dumps
from shapely.wkt import loads as wkt_loads

from modules.pattern import relaxed_voronoi
from modules.points import Points
from modules.unit import Units
from modules.maps import generate_density_with_mask
class Plate:
    '''
    Class to create a plate with holes and fillets.
            bottom
    | ╭────────────────╮
    | │ + hole       + │
    | │                │ height
    | │ +            + │
    | ╰────────────────╯fillet
    |       width
    | \________________/
    + -- -- -- -- -- -- ---

    '''
    def __init__(self,
                name=None,
                material=None,
                color=None,
                thickness=2.0,
                bottom_width=126.46,
                bottom_height=54.46,
                bottom_fillet=11.00,
                holes=[
                {"location": [11.23,  11.23], "radius": 5.75, "mod": "in",  "offset":2.00 ,"tag": "refrigerant"},
                {"location": [115.23, 43.23], "radius": 5.75, "mod": "out", "offset":2.00 ,"tag": "refrigerant"},
                {"location": [115.23, 11.23], "radius": 6.00, "mod": "in",  "offset":2.00 ,"tag": "water"},
                {"location": [11.23,  43.23], "radius": 6.00, "mod": "out", "offset":2.00 ,"tag": "water"},],
                outline_height=5.00,
                outline_angle=66.86,
                cell_height=1.5,
                cell_max_width=2.0,
                cell_min_width=1.0,
                cell_fillet=0.10,
                valid_angle=135.0,
                logger=None
        ):

        ''' Basic parameters '''
        self.name = name
        self.material = material
        self.color = color
        self.thickness = thickness

        ''' Plate bottom parameters '''
        self.bottom_width = bottom_width
        self.bottom_height = bottom_height
        self.bottom_fillet = bottom_fillet

        ''' Plate hole parameters '''
        self.holes = holes

        ''' Plate outline parameters '''
        self.outline_height = outline_height
        self.outline_angle = outline_angle
        self.outline_offset = outline_height/math.tan(self.outline_angle)

        ''' Plate cell parameters '''
        self.cells_A = None
        self.cells_B = None

        self.cell_height = cell_height
        self.cell_max_width = cell_max_width
        self.cell_min_width = cell_min_width
        self.cell_region_offset = 1.00
        self.cell_fillet = cell_fillet

        ''' Plate shapes '''
        self.cell_shaps = None
        self.bottom_face_U = None
        self.bottom_face_B = None           # Bottom face of the plate
        self.full_face = None               # Full face of the plate
        self.full_shell = None              # Full shell of the plate with thickness 

        self.valid_region_angle = valid_angle
        self.valid_region_mask = None       # Valid region of the plate bottom to generate cells
        self.valid_region_edge = None
        
        if self.valid_region_mask is None: 
            self._generate_bottom_region(edge_offset=self.cell_region_offset)

        self.logger = logger
        self.points = None

    def make(
            self, 
            map=None, 
            num_relaxations=6, 
            canvas=None, 
            point=None, 
            offset=False, 
            transform=None, 
            times=1,
            move = 1
        ):
        '''
        Generate all plate-related geometry including:
        - the bottom face with fillets and holes,
        - cut-out shapes from Voronoi units,
        - extrusion of cell geometry,
        - fusion with base face,
        - chamfering of cell top and bottom edges.
        '''

        bounds = [
            self.cell_region_offset, 
            self.bottom_width - self.cell_region_offset, 
            self.cell_region_offset, 
            self.bottom_height - self.cell_region_offset
        ]
        
        # 1. Generate the density map
        self.logger.warn("Generating density map...")
        density = self._generate_bottom_map_test()

        if point is None:
            # 2. Generate points
            self.points = Points(shape=self.valid_region_edge, spacing=3.0, offset_layers=1, logger=self.logger)

            # 3. Move points by density
            move_times = move
            self.logger.warn(f"Moving points by density: {move_times} times...")
            self.points.move(density, dt=0.1, mask_extent=(0, self.bottom_width, 0, self.bottom_height), iterations=move_times)

            # 4. Apply Lloyd relaxation
            self.logger.warn("Applying Lloyd relaxation...")
            self.points.relaxation(iterations=num_relaxations)
            points = self.points.get_points()
        else:
            points = point 

        
        # 5. Compute cells diagram
        self.logger.warn("Computing Voronoi diagram...")
        _, vor_a, _, points_b = relaxed_voronoi(
            points,
            bounds=bounds,
            iterations=10
        )

        # 6. Generate cells
        self.logger.warn("Generating Voronoi units...")
        self.cells_A = Units.generate_from(
            vor_a, 
            density=density, 
            bounds=bounds, 
            mask=self.valid_region_mask, 
            mask_extent=[0, self.bottom_width, 0, self.bottom_height]
        )

        # 7. Generate the cells all
        self.logger.warn("Generating cells...")
        self._generate_cells_solid()

        # 8. Generate the base face: rounded rectangle with circular holes
        self.logger.warn("Generating plate face...")
        faces = self. _generate_bottom_solid()

        # 9. Combine the plate with cells
        self.logger.warn("Combining plate and cells...")
        self._combine_plate_with_cells(faces=faces)

        # 10. draw the plate
        if canvas is not None:
            if offset:
                offset_value=(0.0, self.bottom_height + self.cell_region_offset)
            else:
                offset_value=(0.0, 0.0)
            self.logger.info("Drawing cad graphics...")
            self._draw_region(canvas, offset_value)
            self.cells_A.draw(
                canvas, 
                draw_index=False, 
                draw_shap=True, 
                fill_shap=False, 
                draw_vector=True, 
                vector_scale=3,
                density_map=density,
                bounds=[0, self.bottom_width, 0, self.bottom_height],
                offset=offset_value
            )

            canvas.set_xlim(bounds[0]-1, bounds[1]+1)
            canvas.set_ylim(bounds[2]-1, (bounds[3]+1)*2)
        def move_shape_z(shape, dz: float):
            """
            Move a TopoDS_Shape along the Z axis by dz.

            Parameters:
            - shape: TopoDS_Shape to move
            - dz: distance along the Z axis

            Returns:
            - Transformed TopoDS_Shape
            """
            if shape.IsNull():
                raise ValueError("Input shape is null and cannot be moved.")

            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(0, 0, dz))
            moved_shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
            return moved_shape 
        # 11. Move the plate to the correct position
        if transform:
            self.full_shell = move_shape_z(self.full_shell, (self.cell_height+self.thickness)*times)
        return self.full_shell, points_b

    def save(self, dirpath="plate_data"):
        '''
        Save plate parameters and generated data (mask, edge) to disk.
        '''
        os.makedirs(dirpath, exist_ok=True)

        # 1. Save config parameters
        config = {
            "name": self.name,
            "material": self.material,
            "color": self.color,
            "thickness": self.thickness,
            "bottom_width": self.bottom_width,
            "bottom_height": self.bottom_height,
            "bottom_fillet": self.bottom_fillet,
            "holes": self.holes,
            "outline_height": self.outline_height,
            "outline_angle": self.outline_angle,
            "cell_height": self.cell_height,
            "cell_max_width": self.cell_max_width,
            "cell_min_width": self.cell_min_width,
            "cell_fillet": self.cell_fillet,
            "valid_region_angle": self.valid_region_angle,
        }
        with open(os.path.join(dirpath, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # 2. Save valid region mask
        if hasattr(self, "valid_region_mask") and self.valid_region_mask is not None:
            np.savez_compressed(os.path.join(dirpath, "mask.npz"), mask=self.valid_region_mask)

        # 3. Save region edge (Polygon) as WKT
        if hasattr(self, "valid_region_edge") and self.valid_region_edge is not None:
            with open(os.path.join(dirpath, "region_edge.wkt"), "w") as f:
                f.write(wkt_dumps(self.valid_region_edge))

        self.logger.info(f"[✓] Plate saved to: {os.path.abspath(dirpath)}")

    @classmethod
    def load(cls, dirpath="plate_data", logger=None):
        '''
        Load a plate instance from saved config and data.
        '''
        if logger is None:
            raise ValueError("Logger is required for loading plate.")
        
        logger.info(f"Loading plate from: {os.path.abspath(dirpath)}")
        with open(os.path.join(dirpath, "config.json"), "r") as f:
            data = json.load(f)

        # void devide by zero
        outline_angle = data.get("outline_angle", 0.0)
        if outline_angle == 0:
            outline_angle = 1e-6
            logger.warn("outline_angle is 0, replaced with 1e-6 to avoid division by zero.")

        plate = cls(
            name=data.get("name"),
            material=data.get("material"),
            color=data.get("color"),
            thickness=data["thickness"],
            bottom_width=data["bottom_width"],
            bottom_height=data["bottom_height"],
            bottom_fillet=data["bottom_fillet"],
            holes=data["holes"],
            outline_height=data["outline_height"],
            outline_angle=data["outline_angle"],
            cell_height=data["cell_height"],
            cell_max_width=data["cell_max_width"],
            cell_min_width=data["cell_min_width"],
            cell_fillet=data["cell_fillet"],
            valid_angle=data["valid_region_angle"],
            logger=logger
        )

        # Load mask
        mask_path = os.path.join(dirpath, "mask.npz")
        if os.path.exists(mask_path):
            plate.valid_region_mask = np.load(mask_path)["mask"]

        # Load region edge
        edge_path = os.path.join(dirpath, "region_edge.wkt")
        if os.path.exists(edge_path):
            with open(edge_path, "r") as f:
                plate.valid_region_edge = wkt_loads(f.read())

        return plate

# *************PLATE*************
    def _generate_bottom_face(self):
        '''
        Generate the plate bottom face with holes and fillets
        '''
        # ------------------------------
        def offset_and_lift_wire(wire, offset_value, z_offset):
            """
            Offset a planar wire outward and lift it in the Z direction.

            Parameters:
            - wire: TopoDS_Wire
            - offset_value: float, outward offset distance
            - z_offset: float, shift in Z

            Returns:
            - TopoDS_Wire: the transformed wire
            """
            # Perform 2D offset on wire
            offset_builder = BRepOffsetAPI_MakeOffset()
            offset_builder.AddWire(wire)
            offset_builder.Perform(offset_value)
            offset_wire_shape = offset_builder.Shape()

            # Convert result back to TopoDS_Wire
            explorer = TopExp_Explorer(offset_wire_shape, TopAbs_WIRE)
            if not explorer.More():
                raise RuntimeError("Offset operation failed, no wire found")
            offset_wire = topods.Wire(explorer.Current())

            # Apply Z translation
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(0, 0, z_offset))
            transformed = BRepBuilderAPI_Transform(offset_wire, trsf, True).Shape()

            return topods.Wire(transformed)
        # ------------------------------
        # make holes
        def make_holes(face, z_offset: float = 0.0):
            '''
            Generate holes in the plate face by cutting out circles defined in self.holes.
            Each hole is defined by its center location and radius.

            Parameters:
            - face: TopoDS_Face to cut
            - z_offset: float, z-height where the hole should be placed
            '''
            wires = []
            for hole in self.holes:
                x, y = hole["location"]
                rad = hole["radius"]

                # Create a geometric circle in 3D
                geom_circle = GC_MakeCircle(gp_Pnt(x, y, z_offset), gp_Dir(0, 0, 1), rad)
                if not geom_circle.IsDone():
                    print(f"[Warning] Failed to create circle at ({x}, {y}) with radius {rad}")
                    continue

                circle = geom_circle.Value()

                # Build circular edge and wire
                edge = BRepBuilderAPI_MakeEdge(circle).Edge()
                wire = BRepBuilderAPI_MakeWire(edge).Wire()

                # Make a face from the circular wire (the hole)
                hole_face = BRepBuilderAPI_MakeFace(wire).Face()

                # Cut the hole from the main face
                face = BRepAlgoAPI_Cut(face, hole_face).Shape()
                wires.append(wire)

            return face, wires
        # ------------------------------

        w = self.bottom_width
        h = self.bottom_height
        r = min(self.bottom_fillet, w / 2.0, h / 2.0)

        # Key points (start/end of each segment)
        # Bottom edge
        p1 = gp_Pnt(r, 0, 0)
        p2 = gp_Pnt(w - r, 0, 0)

        # Bottom-right arc
        p3 = gp_Pnt(w, r, 0)

        # Right edge
        p4 = gp_Pnt(w, h - r, 0)

        # Top-right arc
        p5 = gp_Pnt(w - r, h, 0)

        # Top edge
        p6 = gp_Pnt(r, h, 0)

        # Top-left arc
        p7 = gp_Pnt(0, h - r, 0)

        # Left edge
        p8 = gp_Pnt(0, r, 0)

        # Bottom-left arc connects to p1

        # Arcs: use known centers + radius
        arc1 = BRepBuilderAPI_MakeEdge(
            gp_Circ(gp_Ax2(gp_Pnt(w - r, r, 0), gp_Dir(0, 0, 1)), r),
            p2, p3
        ).Edge()

        arc2 = BRepBuilderAPI_MakeEdge(
            gp_Circ(gp_Ax2(gp_Pnt(w - r, h - r, 0), gp_Dir(0, 0, 1)), r),
            p4, p5
        ).Edge()

        arc3 = BRepBuilderAPI_MakeEdge(
            gp_Circ(gp_Ax2(gp_Pnt(r, h - r, 0), gp_Dir(0, 0, 1)), r),
            p6, p7
        ).Edge()

        arc4 = BRepBuilderAPI_MakeEdge(
            gp_Circ(gp_Ax2(gp_Pnt(r, r, 0), gp_Dir(0, 0, 1)), r),
            p8, p1
        ).Edge()

        # Straight edges
        edge1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()  # bottom
        edge2 = BRepBuilderAPI_MakeEdge(p3, p4).Edge()  # right
        edge3 = BRepBuilderAPI_MakeEdge(p5, p6).Edge()  # top
        edge4 = BRepBuilderAPI_MakeEdge(p7, p8).Edge()  # left

        # Wire
        wire_maker = BRepBuilderAPI_MakeWire()
        for e in [edge1, arc1, edge2, arc2, edge3, arc3, edge4, arc4]:
            wire_maker.Add(e)

        if not wire_maker.IsDone():
            raise RuntimeError("Wire creation failed")

        offset_distance = self.thickness / math.tan(math.radians(self.outline_angle / 2))
        wire = wire_maker.Wire()
        wire_u = offset_and_lift_wire(wire, offset_distance, 0) 
        face_u = BRepBuilderAPI_MakeFace(wire_u).Face()
        
        face_u, inner_holes_u = make_holes(face_u, z_offset=0.0)
        self.bottom_face_U = face_u

        # ------------------------------
        # Create bottom_face_B (offset)
        # ------------------------------
        self.logger.info(f"Offset distance: {offset_distance:.2f} mm")
        wire_b = offset_and_lift_wire(wire, 0, -self.thickness)
        face_b = BRepBuilderAPI_MakeFace(wire_b).Face()
        face_b, inner_holes_b = make_holes(face_b, z_offset=-self.thickness)
        self.bottom_face_B = face_b

        # ------------------------------
        # Create outline face
        # ------------------------------
        offset_distance2 = self.outline_height / math.tan(math.radians(180 - self.outline_angle)) 
        wire_ut = offset_and_lift_wire(wire_u, offset_distance2, -self.outline_height)
        wire_bt = offset_and_lift_wire(wire_b, offset_distance2, -self.outline_height)

        faces = []
        faces.append(self._loft_between_wires(wire_u, wire_ut))
        faces.append(self._loft_between_wires(wire_b, wire_bt))
        faces.append(self._loft_between_wires(wire_ut, wire_bt))

        for i, wire_u in enumerate(inner_holes_u):
            wire_b = inner_holes_b[i]
            faces.append(self._loft_between_wires(wire_b, wire_u))


        return faces

    def _generate_bottom_holes(self):
        '''
        Cut 2D B-spline shapes from the plate bottom face (no extrusion).
        '''
        if self.bottom_face_U is None:
            raise RuntimeError("Bottom face not yet generated.")

        face = self.bottom_face_U
        shapes = []
        edges = []
        
        time.sleep(0.5)
        for cell in tqdm(
            self.cells_A.units, 
            desc="GCS_u", 
            bar_format="{desc}: |{bar} [{n_fmt:>05}/{total_fmt:>05}]"
        ):
            if not cell.valid or cell.curve_wire_bo is None:
                continue
            try:
                # Make Face from Wire
                hole_face = BRepBuilderAPI_MakeFace(cell.curve_wire_bo).Face()

                # Subtract from base face
                face = BRepAlgoAPI_Cut(face, hole_face).Shape()
                shapes.append(hole_face)

            except Exception as e:
                self.logger.warn(f"[Unit {cell.index}] spline cut failed: {e}")
        self.bottom_face_U = face

        face = self.bottom_face_B
        time.sleep(0.5)
        for cell in tqdm(
            self.cells_A.units, 
            desc="GCS_b", 
            bar_format="{desc}: |{bar} [{n_fmt:>05}/{total_fmt:>05}]"
        ):
            if not cell.valid or cell.curve_wire_bo is None:
                continue
            try:
                # Make Face from Wire
                hole_face = BRepBuilderAPI_MakeFace(cell.curve_wire_bi).Face()

                # Subtract from base face
                face = BRepAlgoAPI_Cut(face, hole_face).Shape()
                shapes.append(hole_face)

            except Exception as e:
                self.logger.warn(f"[Unit {cell.index}] spline cut failed: {e}")
        self.bottom_face_B = face

        return face, shapes, edges

    def _generate_bottom_region(self, resolution=300, edge_offset=0.0):
        '''
        Generate a boolean mask representing the valid plate area
        (rounded rectangle minus holes, with optional edge/inner thinning).
        
        Parameters:
            resolution (int): Pixel count along the shorter plate side.
            edge_offset (float): Shrink outer shape inward (in mm).
            valid_thick (float): Expand hole radius (in mm) to remove thin ring area.
        '''

        w, h, r = self.bottom_width, self.bottom_height, self.bottom_fillet
        r = min(r, w / 2, h / 2)

        # Step 1: Construct inner-shrunk rounded rectangle points
        outer_points = self._rounded_rect_points(
            cx=edge_offset,
            cy=edge_offset,
            w=w - 2 * edge_offset,
            h=h - 2 * edge_offset,
            r=max(0, r - edge_offset),
            segments=8
        )

        # Step 2: Holes (circular + corner sector cuts)
        hole_shapes = []

        for hole in self.holes:
            x, y = hole["location"]
            rad = hole["radius"] + hole.get("offset", 0.0)

            # Create circular hole
            circle = Point(x, y).buffer(rad, resolution=32)
            hole_shapes.append(circle)

            # Optional: create trimming sector
            if self.valid_region_angle > 0 and hole.get("valid", True):
                is_left = x < w / 2
                is_bottom = y < h / 2
                corner_x = 0 if is_left else w
                corner_y = 0 if is_bottom else h
                corner = np.array([corner_x, corner_y])

                vec = corner - np.array([x, y])
                base_angle = np.arctan2(vec[1], vec[0])

                angle_rad = np.radians(self.valid_region_angle)
                half = angle_rad / 2
                segments = 32
                angles = np.linspace(base_angle - half, base_angle + half, segments)

                radius = self.bottom_fillet + 5
                arc_pts = [(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles]
                sector = Polygon([(x, y)] + arc_pts + [(x, y)])
                hole_shapes.append(sector)

        # Step 3: Create polygon (outer rounded rect minus all holes & sectors)
        polygon = Polygon(shell=outer_points)
        full_holes = unary_union(hole_shapes)
        valid_area = polygon.difference(full_holes)
        prepared_polygon = prep(valid_area)

        # Step 4: Rasterize mask
        short = min(w, h)
        px_size = short / resolution
        nx = int(np.ceil(w / px_size))
        ny = int(np.ceil(h / px_size))

        xs = (np.arange(nx) + 0.5) * px_size
        ys = (np.arange(ny) + 0.5) * px_size
        X, Y = np.meshgrid(xs, ys)
        pts = np.vstack([X.ravel(), Y.ravel()]).T

        mask = np.array([prepared_polygon.contains(Point(x, y)) for x, y in pts], dtype=bool)
        mask = mask.reshape((ny, nx))

        self.valid_region_mask = mask
        self.valid_region_edge = valid_area 
        return mask, prepared_polygon

    def _generate_bottom_map_test(self, resolution=300):
        '''
        Generate and display a test density map using circle-based influence 
        and gradient, within the valid region mask.
        '''
        if self.valid_region_mask is None:
            self.logger.warn("No valid region found. Generating one...")
            self._generate_region(resolution=resolution)

        w, h = self.bottom_width, self.bottom_height
        valid_mask = self.valid_region_mask

        # example circles
        circles = [
            (11.23, 11.23, 6.00, 'out'),
            (115.23, 43.23, 5.75, 'in'),
        ]

        # Generate density map
        density = generate_density_with_mask(
            size=(w, h),
            valid_mask=valid_mask,
            circles=circles,
            gradient_direction=(-1, 0),
            gradient_strength=0.3,
            falloff=125.0
        )

        return density

    def _generate_bottom_solid(self):
        self.logger.info("Generating bottom face...")
        faces = self._generate_bottom_face()

        self.logger.info("Generating bottom holes...")
        self._generate_bottom_holes()

        return faces

# *************CELLS*************
    def _generate_cells_curve(self):
        def move_wire(wire, dx=0.0, dy=0.0, dz=0.0):
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(dx, dy, dz))
            transformer = BRepBuilderAPI_Transform(wire, trsf, True)  # True 表示 copy
            moved_wire = transformer.Shape()
            return moved_wire
        time.sleep(0.5)
        for cell in tqdm(
            self.cells_A.units, 
            desc="GCC", 
            bar_format="{desc}: |{bar} [{n_fmt:>05}/{total_fmt:>05}]"
        ):
            curve_spline_o = self._generate_curve_bspline(np.array(cell.curve_bezier))
            curve_spline_i = self._generate_curve_bspline(np.array(cell._offset_fixed(cell.curve_bezier, center=cell.center, offset_length=-self.thickness)))

            # bottom outer
            cell.curve_wire_bo = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(curve_spline_o).Edge()).Wire()
            # top outer
            cell.curve_wire_to = move_wire(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(curve_spline_o).Edge()).Wire(), dz=self.cell_height)
            # bottom inner
            cell.curve_wire_bi = move_wire(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(curve_spline_i).Edge()).Wire(), dz=-self.thickness)
            # top inner
            cell.curve_wire_ti = move_wire(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(curve_spline_i).Edge()).Wire(), dz=self.cell_height-self.thickness)

    def _generate_cells_convax(self):
        """
        Generate open-bottom cell shells by bridging inner and outer wires,
        and capping the top face.

        Each cell consists of:
        - outer wall (bottom_outer -> top_outer)
        - inner wall (top_inner -> bottom_inner)
        - top face (top_outer - top_inner)
        """
        height = self.cell_height
        thickness = self.thickness
        extruded = []

        time.sleep(0.5)
        for i, cell in tqdm(
            enumerate(self.cells_A.units),
            desc="ECC",
            total=len(self.cells_A.units),
            bar_format="{desc}: |{bar} [{n_fmt:>05}/{total_fmt:>05}]"
        ):
            if cell.valid:
                try:
                    wire_bo = cell.curve_wire_bo
                    wire_bi = cell.curve_wire_bi
                    wire_to = cell.curve_wire_to
                    wire_ti = cell.curve_wire_ti

                    # Skip if any wire is missing
                    if None in (wire_bo, wire_bi, wire_to, wire_ti):
                        self.logger.warn(f"[Cell {i}] missing wire(s), skipped.")
                        continue

                    # Build compound to hold wall and top face
                    builder = BRep_Builder()
                    compound = TopoDS_Compound()
                    builder.MakeCompound(compound)

                    # Outer wall: bottom outer -> top outer
                    wall_outer = self._loft_between_wires(wire_bo, wire_to)
                    builder.Add(compound, wall_outer)

                    # Inner wall: top inner -> bottom inner (reversed direction)
                    wall_inner = self._loft_between_wires(wire_ti, wire_bi, reversed=True)
                    builder.Add(compound, wall_inner)

                    # Top face: between top outer and top inner
                    top_face_outer = BRepBuilderAPI_MakeFace(wire_to).Face()
                    top_face_inner = BRepBuilderAPI_MakeFace(wire_ti).Face()

                    builder.Add(compound, top_face_outer)
                    builder.Add(compound, top_face_inner)

                    extruded.append(compound)

                except Exception as e:
                    self.logger.warn(f"[Cell {i}] extrusion failed: {e}")

        self.cell_shaps = extruded
        return extruded

    def _generate_curve_bspline(self, points, offset_z: float = 0.0):
        """
        Generate a closed B-spline curve from a list of 2D points.

        Parameters:
        - points: np.ndarray of shape (N, 2)

        Returns:
        - bspline curve or None if failed
        """
        for uint in self.cells_A.units:
            pass
        try:
            if not np.allclose(points[0], points[-1]):
                points = np.vstack([points, points[0]])

            n = len(points)
            array = TColgp_Array1OfPnt(1, n)
            for i in range(n):
                array.SetValue(i + 1, gp_Pnt(points[i][0], points[i][1], 0))
            builder = GeomAPI_PointsToBSpline(array, 3, 8, GeomAbs_C2, 1e-3)
            return builder.Curve()
        except Exception as e:
            self.logger.erro(f"❌ Failed to create B-spline curve: {e}")
            return None

    def _generate_cells_solid(self):
        self.logger.info("Generating cells' curves...")
        self._generate_cells_curve()

        self.logger.info("Generating cells' solids...")
        self._generate_cells_convax()

# *************COMB**************
    def _combine_plate_with_cells(self, faces: list = None):
        builder = BRep_Builder()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)

        def fix_face_orientation(face):
            fixer = ShapeFix_Face(face)
            fixer.FixOrientation()
            return fixer.Face()

        # Add upper base face(s)
        exp = TopExp_Explorer(self.bottom_face_U, TopAbs_FACE)
        face_count = 0
        while exp.More():
            face = topods.Face(exp.Current())
            face = fix_face_orientation(face)
            builder.Add(shell, face)
            face_count += 1
            exp.Next()
        self.logger.info(f"✅ Added {face_count} upper base face(s)")

        # Add lower base face(s)
        exp = TopExp_Explorer(self.bottom_face_B, TopAbs_FACE)
        face_count_b = 0
        while exp.More():
            face = topods.Face(exp.Current())
            face = fix_face_orientation(face)
            builder.Add(shell, face)
            face_count_b += 1
            exp.Next()
        self.logger.info(f"✅ Added {face_count_b} lower base face(s)")

        # Add all cell faces
        total_cell_faces = 0
        for i, compound in enumerate(self.cell_shaps):
            exp = TopExp_Explorer(compound, TopAbs_FACE)
            while exp.More():
                face = topods.Face(exp.Current())
                face = fix_face_orientation(face)
                builder.Add(shell, face)
                total_cell_faces += 1
                exp.Next()
        self.logger.info(f"✅ Added {total_cell_faces} cell faces")

        # Add faces from input list
        input_face_count = 0
        if faces:
            for i, input_shell in enumerate(faces):
                if input_shell.IsNull():
                    self.logger.warning(f"⚠️ Skipped null shell at index {i}")
                    continue

                exp = TopExp_Explorer(input_shell, TopAbs_FACE)
                while exp.More():
                    face = topods.Face(exp.Current())
                    face = fix_face_orientation(face)
                    builder.Add(shell, face)
                    input_face_count += 1
                    exp.Next()

            self.logger.info(f"✅ Added {input_face_count} face(s) from input TopoDS_Shell list")

        # Try to fix shell orientation (safe version)
        analyzer = BRepCheck_Analyzer(shell)
        if analyzer.IsValid():
            try:
                fixer = ShapeFix_Shell()
                fixer.Init(shell)
                fixer.Perform()
                shell = fixer.Shell()
                self.logger.info("🔧 Shell orientation fixed by ShapeFix_Shell.")
            except Exception as e:
                self.logger.warn(f"⚠️ ShapeFix_Shell.Perform() failed: {e}")
        else:
            self.logger.warn("⚠️ Skipping ShapeFix_Shell: shell is already invalid before fixing.")

        # Validate shell
        analyzer = BRepCheck_Analyzer(shell)
        if not analyzer.IsValid():
            self.logger.warning("⚠️ Warning: Combined shell is not geometrically valid (possibly open or flipped faces).")

        # Convert shell to solid
        self.full_shell = BRepBuilderAPI_MakeSolid(shell).Solid()
        if not self.full_shell.IsNull():
            self.logger.info("✅ Combined plate and cells into solid successfully.")
        else:
            self.logger.error("❌ Failed to combine plate and cells into solid.")

        return self.full_shell

# *************HELP************** 
    def _rounded_rect_points(self, cx=0, cy=0, w=None, h=None, r=None, segments=13):
        '''
        Return list of (x, y) points approximating the rounded rectangle border.
        
        Parameters:
            cx, cy: offset of bottom-left corner
            w, h: optional override of width/height (default from plate)
            r: optional override of corner radius (default from plate)
            segments: number of points per arc (1/4 circle)
        '''
        w = self.bottom_width if w is None else w
        h = self.bottom_height if h is None else h
        r = min(self.bottom_fillet, w / 2, h / 2) if r is None else r

        points = []
        corner_centers = [
            (cx + w - r, cy + r),        # Bottom-right
            (cx + w - r, cy + h - r),    # Top-right
            (cx + r,     cy + h - r),    # Top-left
            (cx + r,     cy + r),        # Bottom-left
        ]
        angles = [
            (270, 360),  # bottom-right
            (0, 90),     # top-right
            (90, 180),   # top-left
            (180, 270),  # bottom-left
        ]
        for (cx, cy), (start, end) in zip(corner_centers, angles):
            theta = np.linspace(np.radians(start), np.radians(end), segments)
            for t in theta:
                x = cx + r * np.cos(t)
                y = cy + r * np.sin(t)
                points.append((x, y))
        points.append(points[0])  # Close loop
        return points

    def _offset_cells(self):
        # Offset cells
        pass

    def _offset_thick(self):
        pass

    def _loft_between_wires(self, wire1, wire2, reversed=False):
        """
        Create a lofted shell surface between two wires.

        Parameters:
        - wire1, wire2: TopoDS_Wire objects
        - reversed: if True, wire order is reversed (useful for inner walls)

        Returns:
        - TopoDS_Shape: lofted shell surface
        """
        loft = BRepOffsetAPI_ThruSections(False, False)
        loft.CheckCompatibility(False)
        if reversed:
            loft.AddWire(wire2)
            loft.AddWire(wire1)
        else:
            loft.AddWire(wire1)
            loft.AddWire(wire2)
        loft.Build()
        if not loft.IsDone():
            self.logger.erro(f"loft failed")
        return loft.Shape()

    def _offset_and_lift_wire(self, wire, offset_value, z_offset):
        """
        Offset a planar wire outward and lift it in the Z direction.

        Parameters:
        - wire: TopoDS_Wire
        - offset_value: float, outward offset distance
        - z_offset: float, shift in Z

        Returns:
        - TopoDS_Wire: the transformed wire
        """
        # Perform 2D offset on wire
        offset_builder = BRepOffsetAPI_MakeOffset()
        offset_builder.AddWire(wire)
        offset_builder.Perform(offset_value)
        offset_wire_shape = offset_builder.Shape()

        # Convert result back to TopoDS_Wire
        explorer = TopExp_Explorer(offset_wire_shape, TopAbs_WIRE)
        if not explorer.More():
            raise RuntimeError("Offset operation failed, no wire found")
        offset_wire = topods.Wire(explorer.Current())

        # Apply Z translation
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(0, 0, z_offset))
        transformed = BRepBuilderAPI_Transform(offset_wire, trsf, True).Shape()

        return topods.Wire(transformed)

# *************DRAW**************
    def _show(self):
        ''' show the plate '''
        # display, start_display, _, _ = init_display()
        # if self.bottom_face_A:
        #     display.DisplayShape(self.bottom_face_A, update=True)
        #     display.FitAll()
        # start_display()

        self._show_region_mask()

    def _draw_region(self, ax, offset=(0.0, 0.0)):
        """
        Draw the valid region mask, holes, and outlines on the given matplotlib axis.

        Parameters:
        - ax: matplotlib Axes
        - offset: (x, y) tuple to shift the entire drawing
        """
        ox, oy = offset
        mask = self.valid_region_mask
        if mask is None:
            self.logger.warn("No valid region mask found. Generating a new one.")
            mask = self._generate_region(edge_offset=1.00)

        w, h = self.bottom_width, self.bottom_height
        extent = [ox, ox + w, oy, oy + h]  # <== 偏移应用到 extent

        ax.imshow(mask, cmap='Greys', origin='lower', extent=extent)
        ax.set_title("Valid Region Mask (White = Valid)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True)
        ax.set_aspect("equal")

        # Margin
        x0, x1, y0, y1 = extent
        x_margin = (x1 - x0) * 0.05
        y_margin = (y1 - y0) * 0.05
        ax.set_xlim(x0 - x_margin, x1 + x_margin)
        ax.set_ylim(y0 - y_margin, y1 + y_margin)

        # Draw hole (inner radius - red)
        for hole in self.holes:
            x, y = hole["location"]
            r = hole["radius"]
            circle = plt.Circle((x + ox, y + oy), r, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(circle)

        # Draw hole (outer offset radius - blue)
        for hole in self.holes:
            x, y = hole["location"]
            r = hole["radius"] + hole.get("offset", 0.0)
            circle = plt.Circle((x + ox, y + oy), r, edgecolor='blue', facecolor='none', linestyle='--')
            ax.add_patch(circle)

        # Rounded rectangle
        rounded = self._rounded_rect_points()
        xlist, ylist = zip(*rounded)
        xlist = [x + ox for x in xlist]
        ylist = [y + oy for y in ylist]
        ax.plot(xlist, ylist, color='red', linewidth=1.5, linestyle='--')

        # Draw valid region edge (if exists)
        if self.valid_region_edge:
            geoms = [self.valid_region_edge] if isinstance(self.valid_region_edge, Polygon) else self.valid_region_edge.geoms
            for geom in geoms:
                x, y = geom.exterior.xy
                x = [xi + ox for xi in x]
                y = [yi + oy for yi in y]
                ax.plot(x, y, color='yellow', linewidth=1.5, linestyle='-')
if __name__ == "__main__":
    path = "/Users/ein/EinDev/OcctStuff/.cache/test00" 
    # plate = Plate()
    # plate.make_all()
    # plate._show()
    # plate.save("/Users/ein/EinDev/OcctStuff/.cache/test00")

    loaded_plate = Plate.from_file(path)
    # loaded_plate._show_region_mask()
    # loaded_plate._generate_test_map(resolution=300)