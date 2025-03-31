#!/usr/bin/env python
# -*- coding: utf-8 -*-

# stl
import sys
import hashlib
import math
import datetime
import os
import threading
import queue
from loguru import logger as CORELOG
from collections import defaultdict
from itertools import combinations
# extern
from tqdm import tqdm
# multthread
import concurrent.futures

# OpenCASCADE
from OCC.Core.AIS import AIS_TextLabel, AIS_Point
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GC import GC_MakePlane
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.Geom import Geom_CartesianPoint
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepTools import breptools_Write


from OCC.Core.Quantity import (
    Quantity_NOC_RED, 
    Quantity_NOC_GREEN, 
    Quantity_NOC_YELLOW, 
    Quantity_NOC_PURPLE,
    Quantity_Color, 
    Quantity_TOC_RGB)

from OCC.Core.GeomAbs import (
    GeomAbs_Line, 
    GeomAbs_Circle, 
    GeomAbs_Ellipse,
    GeomAbs_Parabola, 
    GeomAbs_Hyperbola, 
    GeomAbs_BSplineCurve, 
    GeomAbs_BezierCurve, 
    GeomAbs_OffsetCurve,
    GeomAbs_OtherCurve)

# Curve type
CURVE_TYPE = {
    GeomAbs_Line: "Line",
    GeomAbs_Circle: "Circle",
    GeomAbs_Ellipse: "Ellipse",
    GeomAbs_Parabola: "Parabola",
    GeomAbs_Hyperbola: "Hyperbola",
    GeomAbs_BSplineCurve: "BSpline Curve",
    GeomAbs_BezierCurve: "Bezier Curve",
    GeomAbs_OffsetCurve: "Offset Curve",
    GeomAbs_OtherCurve :"Other"}

# Help data
CROSS_LINE = []
PASSED_LINE = []
OTHER_LINE = []
STUBBORN_LINE = []
EXCLUDE_GROUPS = []
CHECK_ONLY = False
PATH = ''
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

# LOG#############
class Log:
    class logdata:
        def __init__(self, data:str, tag:int) -> None:
            self.m_Data = data
            self.m_Tag = tag

    def __init__(self) -> None:
        self.m_DataQueue = queue.Queue()
        self.lock = threading.Lock()
        self.m_Thread = threading.Thread(target=self.Update)
        self.m_Thread.start()

    def Sync(self):
        self.m_Thread.join()

    def Update(self):
        while True:
            if not self.m_DataQueue.empty():
                with self.lock:
                    data = self.m_DataQueue.get()
                    time = datetime.datetime.now().time().strftime("%H:%M:%S")
                    
                    if data.m_Tag == 0:  # info
                        CORELOG(f"{GREEN}[{time}][INFO] {data.m_Data}{RESET}")

                    if data.m_Tag == 1:  # warningng
                        CORELOG(f"{YELLOW}[{time}][warning] {data.m_Data}{RESET}")

                    if data.m_Tag == 2:  # error
                        CORELOG(f"{RED}[{time}][ERROR] {data.m_Data}{RESET}")
            else:
                continue   

    def info(self, str):
        with self.lock:
            logData = self.logdata(str, 0)
            self.m_DataQueue.put(logData)

    def warning(self, str):
        with self.lock:
            logData = self.logdata(str, 1)
            self.m_DataQueue.put(logData)

    def error(self, str):
        with self.lock:
            logData = self.logdata(str, 2)
            self.m_DataQueue.put(logData)

CORELOG.configure(handlers=
    [{"sink": sys.stdout, 
        "level": "DEBUG", 
        "format": "<green>{time:YYYY-MM-DD HH:mm}</green> | <level>{message}</level>"}, 
        {"sink": "./Logcache/DemoLog{time:YYYY-MM}.log", 
        "level": "DEBUG", 
        "rotation": "10 MB", 
        "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"}])

def check_shape_validity(shape):
    """
    Check if a shape is valid.

    Parameters:
        shape (TopoDS_Shape): The shape to check.

    Returns:
        bool: True if the shape is valid, False otherwise.
    """
    analyzer = BRepCheck_Analyzer(shape)
    return analyzer.IsValid()

def ensure_normals_point_outward(shape):
    """
    Ensure that all face normals of a shape point outward.

    Parameters:
        shape (TopoDS_Shape): The shape to process.

    Returns:
        TopoDS_Shape: The corrected shape with outward-pointing normals.
    """
    # Step 1: Calculate the volume of the shape
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    volume = props.Mass()

    # Step 2: Check if the volume is negative
    if volume < 0:
        CORELOG("Shape has inverted normals. Correcting...")

        # Step 3: Reverse all faces
        reversed_faces = TopTools_ListOfShape()
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            reversed_face = BRep_Tool.Surface(face).Reversed()
            reversed_faces.Append(reversed_face)
            explorer.Next()

        # Step 4: Sew the reversed faces back into a solid
        sewing = BRepBuilderAPI_Sewing()
        for reversed_face in reversed_faces:
            sewing.Add(reversed_face)
        sewing.Perform()

        corrected_shape = sewing.SewedShape()
        return corrected_shape
    else:
        CORELOG.info("Shape normals are already correct.")

        return shape

def load_step_file(filepath):
    """
    Load a STEP file using STEPControl_Reader and return the TopoDS_Shape.
    
    Parameters:
        filepath (str): The path to the STEP file.
    
    Returns:
        TopoDS_Shape: The loaded shape.
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise ValueError(f"Error: Unable to read file {filepath}.")
    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    CORELOG.info("Check if a shape is valid...")
    if not CHECK_ONLY:
        ensure_normals_point_outward(shape)
        check_shape_validity(shape)
    CORELOG.info("Shape is valid!")
    return shape

def save_step_file(shape, path):
    """
    Save a TopoDS_Shape to a STEP file.

    Parameters:
        shape (TopoDS_Shape): The shape to save.
        path (str): The file path to save the shape.
    """
    try:
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)  # Transfer shape into the writer
        status = writer.Write(path)  # Save to the file
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to save STEP file: {path}")
        CORELOG.info(f"Model successfully saved to {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")
    
def build_edge_to_faces_map(shape):
    """
    Build a dictionary mapping each edge to its associated faces.
    
    Parameters:
        shape (TopoDS_Shape): The shape to explore.
    
    Returns:
        defaultdict: A dictionary where each key is an edge and the value is a list of associated faces.
    """
    edge_to_faces = defaultdict(list)
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge_to_faces[edge].append(face)
            edge_explorer.Next()
        face_explorer.Next()
    return edge_to_faces

def count_faces_per_edge(edge_to_faces_map):
    """
    Count the number of faces associated with each edge.
    
    Parameters:
        edge_to_faces_map (dict): A dictionary mapping edges to their associated faces.
    
    Returns:
        list: A list of tuples where each tuple contains an edge and its face count.
    """
    edge_face_counts = []
    for edge, faces in edge_to_faces_map.items():
        face_count = len(faces)
        edge_face_counts.append((edge, face_count))
    return edge_face_counts

def get_edge_key(edge, decimal=5):
    """
    Generate a unique SHA256 hash key based on the edge's vertices.
    
    Parameters:
        edge (TopoDS_Edge): The edge to generate the key for.
        decimal (int): Number of decimal places to round the coordinates.
        
    Returns:
        str: A hexadecimal SHA256 hash representing the edge.
    """
    vertices = []
    explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while explorer.More():
        vertex = explorer.Current()
        pnt = BRep_Tool.Pnt(vertex)
        vertices.append((round(pnt.X(), decimal), round(pnt.Y(), decimal), round(pnt.Z(), decimal)))
        explorer.Next()
    
    if len(vertices) != 2:
        return None
    
    v1, v2 = sorted(vertices)
    mid,_ = get_line_normal(edge)
    v3 = (round(mid.X(), decimal),round(mid.Y(), decimal), round(mid.Z(), decimal))
    key_string = f"{v1}-{v3}-{v2}"
    return hashlib.sha256(key_string.encode()).hexdigest()

def get_edge_key_two(edge, decimal=5):
    """
    Generate a unique SHA256 hash key based on the edge's vertices.
    
    Parameters:
        edge (TopoDS_Edge): The edge to generate the key for.
        decimal (int): Number of decimal places to round the coordinates.
        
    Returns:
        str: A hexadecimal SHA256 hash representing the edge.
    """
    vertices = []
    explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while explorer.More():
        vertex = explorer.Current()
        pnt = BRep_Tool.Pnt(vertex)
        vertices.append((round(pnt.X(), decimal), round(pnt.Y(), decimal), round(pnt.Z(), decimal)))
        explorer.Next()
    
    if len(vertices) != 2:
        return None
    
    v1, v2 = sorted(vertices)
    key_string = f"{v1}-{v2}"
    return hashlib.sha256(key_string.encode()).hexdigest()

def get_edge_key_sets(edge_to_faces_map):
    """
    Generate a list of (edge_key, edge, face) tuples.
    
    Parameters:
        edge_to_faces_map (dict): A dictionary mapping edges to their associated faces.
    
    Returns:
        list: A list of tuples containing edge keys, edges, and associated faces.
    """
    edge_face_pair = []
    for edge, faces in edge_to_faces_map.items():
        if CURVE_TYPE.get(BRepAdaptor_Curve(edge).GetType()) == CURVE_TYPE[GeomAbs_OtherCurve]:
            CORELOG.warning("Skip edges with GeomAbs_OtherCurve type")
            OTHER_LINE.append(edge)
            continue 
        key = get_edge_key(edge)
        if key is None:
            CORELOG.warning("Skip edges that don't have exactly two vertices!")
            continue
        for face in faces:
            edge_face_pair.append((key, edge, face))
    return edge_face_pair

def get_uv_bounds(face):
    """
    Get the U, V parameter bounds of a given face.

    Parameters:
        face (TopoDS_Face): The face to get the bounds for.

    Returns:
        tuple: (umin, umax, vmin, vmax) - the U, V parameter bounds.
    """
    surface_adaptor = BRepAdaptor_Surface(face)
    umin, umax, vmin, vmax = surface_adaptor.FirstUParameter(), surface_adaptor.LastUParameter(), surface_adaptor.FirstVParameter(), surface_adaptor.LastVParameter()
    return umin, umax, vmin, vmax

def group_coincident_edges(edge_face_pairs):
    """
    Group coincident edges based on their keys.
    
    Parameters:
        edge_face_pairs (list): A list of tuples containing edge keys, edges, and associated faces.
    
    Returns:
        list: A list of lists, where each inner list contains tuples of (edge, face) for coincident edges.
    """
    grouped = defaultdict(list)
    for key, edge, face in edge_face_pairs:
        grouped[key].append((edge, face))
    
    # Filter out groups with only one edge (no coincidence)
    coincident_groups = [group for group in grouped.values() if len(group) > 1]
    return coincident_groups

def get_line_normal(edge, pos = 0.5):
    """
    Get a point on an edge of type GeomAbs_OtherCurve using a parameter.
    Then, get normal at that point
    """

    type = CURVE_TYPE.get(BRepAdaptor_Curve(edge).GetType())
    if CURVE_TYPE.get(BRepAdaptor_Curve(edge).GetType()) == CURVE_TYPE[GeomAbs_OtherCurve]:
        CORELOG.warning("GeomAbs_OtherCurve should be passed before")
        raise ValueError("!")
    curve = BRepAdaptor_Curve(edge)
    first = curve.FirstParameter()
    last = curve.LastParameter()

    point = gp_Pnt()
    vec = gp_Vec()
    curve.D1((first + last)* pos, point, vec)

    return point, vec.Normalized()

def calculate_plane_face_intersection(face, midpoint, perpendicular_vec, normal_vec):
    """
    Calculate the intersection line between a plane and a face.

    Parameters:
        face (TopoDS_Face): The face to intersect with.
        midpoint (gp_Pnt): The midpoint of the edge.
        tangent_vec (gp_Vec): The tangent vector at the midpoint.
        normal_vec (gp_Vec): The normal vector at the midpoint.

    Returns:
        TopoDS_Edge: The intersection edge that contains the midpoint.
    """
    # Step 1: Create a plane using the midpoint, tangent vector, and normal vector
    point1 = midpoint
    point2 = gp_Pnt(midpoint.XYZ() + perpendicular_vec.XYZ())
    point3 = gp_Pnt(midpoint.XYZ() + normal_vec.XYZ())

    plane = GC_MakePlane(point1, point2, point3).Value()

    # Step 2: Calculate the intersection between the plane and the face
    section = BRepAlgoAPI_Section(face, plane, False)
    section.Approximation(True)
    section.Build()
    if not section.IsDone():
        raise RuntimeError("Intersection calculation failed.")

    # Step 3: Explore edges in the intersection shape
    explorer = TopExp_Explorer(section.Shape(), TopAbs_EDGE)

    # Check if there's only one edge in the intersection result
    edges = []
    while explorer.More():
        edge = explorer.Current()
        edges.append(edge)
        explorer.Next()

    # If only one edge is found, return it directly
    if len(edges) == 1:
        return edges[0]
    if len(edges) > 0:
        # Step 4: Otherwise, check if the midpoint lies on any edge
        for edge in edges:
            vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex = topods.Vertex(vertex_explorer.Current())
                point = BRep_Tool.Pnt(vertex)
                # Check if the vertex point matches the midpoint
                if midpoint.IsEqual(point, 1e-6):
                    return edge
                vertex_explorer.Next()
    else:
        return None

def find_nearby_interior_point(edge, target_point, offset_ratio=0.01, min_offset=0.001):
    """
    Find an interior point on the curve close to the given target point (start or end point of the curve).

    Parameters:
        edge (TopoDS_Edge): The edge representing the curve.
        target_point (gp_Pnt): The target point to start from (usually a vertex).
        offset (float): The parameter offset distance along the curve.

    Returns:
        gp_Pnt: A point on the curve near the given target point but not at the endpoint.

    BUG: Currently, all first_param < last_param, may cause bug if get a first_param > last_param
    """
    # Create a curve adaptor
    curve_adaptor = BRepAdaptor_Curve(edge)

    # Get the first and last parameters of the curve
    first_param = curve_adaptor.FirstParameter()
    last_param = curve_adaptor.LastParameter()

    # Calculate start and end points of the curve
    start_point = gp_Pnt()
    end_point = gp_Pnt()
    curve_adaptor.D0(first_param, start_point)
    curve_adaptor.D0(last_param, end_point)

    # Calculate the curve length
    curve_length = last_param - first_param
    
    # Determine which endpoint matches the target point
    if target_point.IsEqual(start_point, 1e-6):
        param = first_param + max(curve_length * offset_ratio, min_offset)
    elif target_point.IsEqual(end_point, 1e-6):
        param = last_param - max(curve_length * offset_ratio, min_offset)
    else:
        # If the target point is not an endpoint, find the nearest endpoint and offset from there
        distance_to_start = target_point.Distance(start_point)
        distance_to_end = target_point.Distance(end_point)

        if distance_to_start < distance_to_end:
            param = first_param + max(curve_length * offset_ratio, min_offset)
        else:
            param = last_param - max(curve_length * offset_ratio, min_offset)

            
    # Calculate the nearby point on the curve
    nearby_point = gp_Pnt()
    curve_adaptor.D0(param, nearby_point)

    # Ensure the nearby point is not equal to the target point
    if nearby_point.IsEqual(target_point, 1e-6):
        raise RuntimeError("Failed to find a distinct interior point on the curve.")

    return nearby_point

def calculate_angle_between_faces(lineAlpha, lineBeta, surfaceAlpha, surfaceBeta):
    '''
        Find 4 points:

        PointO: linestart
        PointA: linestart + delat

        pointB: point on surfaceAlpha
        PointC: point on suffaceBeta

        vecOA, vecOB -> normalA
        vecOA, vecOC -> normalB

        return: (normalAlpha, pointAlpha, pointAlpha2, normalBeta, pointBeta, pointBeta2, angle_deg)

    '''
    # Two normal with right hand rule
    pointAlpha, lineAlphaTangent = get_line_normal(lineAlpha)
    pointBeta, lineBetaTangent = get_line_normal(lineBeta)
    projectorAlpha = GeomAPI_ProjectPointOnSurf(pointAlpha, BRep_Tool.Surface(surfaceAlpha))
    projectorBeta = GeomAPI_ProjectPointOnSurf(pointBeta, BRep_Tool.Surface(surfaceBeta))
    
    if projectorAlpha.NbPoints() == 0 or projectorBeta.NbPoints() == 0:
        CORELOG.error("Projection failed! Point is not on the surface.")
        return None, None, None, None, None, None, 0
    
    ua, va = projectorAlpha.LowerDistanceParameters()
    ub, vb = projectorBeta.LowerDistanceParameters()

    props1 = GeomLProp_SLProps(BRep_Tool.Surface(surfaceAlpha), ua, va, 1, 1e-6)
    props2 = GeomLProp_SLProps(BRep_Tool.Surface(surfaceBeta), ub, vb, 1, 1e-6)
    if not props1.IsNormalDefined() or not props2.IsNormalDefined():
        print("\n", end="")
        CORELOG.warning("Normal vector is not defined at this point.")
        return None, None, None, None, None, None, 0
    
    normalAlpha = gp_Vec(props1.Normal())
    normalBeta = gp_Vec(props2.Normal())

    # crosslineAlpha = calculate_plane_face_intersection(surfaceAlpha, pointAlpha, lineAlphaTangent.Crossed(normalAlpha), normalAlpha)
    # crosslineBeta = calculate_plane_face_intersection(surfaceBeta, pointBeta, lineBetaTangent.Crossed(normalBeta), normalBeta)
    # Use ThreadPoolExecutor to run the intersection calculations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        future_alpha = executor.submit(calculate_plane_face_intersection, 
            surfaceAlpha, pointAlpha, lineAlphaTangent.Crossed(normalAlpha), normalAlpha)
        future_beta = executor.submit(calculate_plane_face_intersection, 
            surfaceBeta, pointBeta, lineBetaTangent.Crossed(normalBeta), normalBeta)
        
        # Wait for both tasks to complete
        done, not_done = concurrent.futures.wait([future_alpha, future_beta], 
            return_when=concurrent.futures.ALL_COMPLETED)
        
        # Get results after both are done
        crosslineAlpha = future_alpha.result()
        crosslineBeta = future_beta.result()

    if (not crosslineAlpha is None) and (not crosslineBeta is None):
        CROSS_LINE.append(crosslineAlpha)
        CROSS_LINE.append(crosslineBeta)
    else:
        CORELOG.warning("Temporily pass this case that cannot find two intersection line.")
        return None, None, None, None, None, None, 0

    pointAlpha2 = find_nearby_interior_point(crosslineAlpha, pointAlpha)
    pointBeta2 = find_nearby_interior_point(crosslineBeta, pointBeta)

    # # Get correct direction. Menuly handled the right hand rule:
    if (not pointAlpha2 is None) and (not pointBeta2 is None):
        vecAlpha = gp_Vec(pointAlpha, pointAlpha2).Normalized()
        vecBeta = gp_Vec(pointBeta, pointBeta2).Normalized()

        dirAlpha = vecAlpha.Crossed(lineAlphaTangent).Normalized()
        dirBeta = lineAlphaTangent.Crossed(vecBeta).Normalized()

        # Ensure dirAlpha and normalAlpha point in the same direction
        if normalAlpha.Dot(dirAlpha) < 0:
            normalAlpha.Reverse()

        # Ensure dirBeta and normalBeta point in the same direction
        if normalBeta.Dot(dirBeta) < 0:
            normalBeta.Reverse()
    else:
        CORELOG.warning("Temporily pass this case that connot find two point!")
        PASSED_LINE.append(crosslineAlpha)
        PASSED_LINE.append(crosslineBeta)
        return None, None, None, None, None, None, 0
    # Calculate the dot product
    dot_product = normalAlpha.Dot(normalBeta)

    # Clamp the dot product to the valid range [-1, 1]
    dot_product = max(min(dot_product, 1.0), -1.0)

    # Calculate the angle in radians and convert to degrees
    angle_rad = math.acos(dot_product)
    angle_deg = math.degrees(angle_rad)

    return normalAlpha, pointAlpha, pointAlpha2, normalBeta, pointBeta, pointBeta2, angle_deg

def process_group_range(groups, start_idx, end_idx, max_angle, normal):
    """Process a range of groups assigned to a specific thread"""
    sharp_edges = []
    exclude_groups = [] # Local storage for excluded groups
    
    # Add progress bar for each thread's work
    for i in tqdm(range(start_idx, end_idx), desc=f"Processing groups {start_idx}-{end_idx}"):
        group = groups[i]
        
        if len(group) == 2:
            lineAlpha, faceAlpha = group[0]
            lineBeta, faceBeta = group[1]
            
            # Get normal and angle
            result = calculate_angle_between_faces(lineAlpha, lineBeta, faceAlpha, faceBeta)
            if result[0] is None:  # Skip if calculation failed
                continue
                
            normalAlpha, pa, _, normalBeta, pb, pb2, angle = result
            
            if normal:
                if angle > max_angle:
                    sharp_edges.extend([
                        (lineAlpha, normalAlpha, pa, pb2, angle, True),
                        (lineBeta, normalBeta, pb, pb2, angle, True)
                    ])
                else:
                    sharp_edges.extend([
                        (lineAlpha, normalAlpha, pa, pb2, angle, False),
                        (lineBeta, normalBeta, pb, pb2, angle, False)
                    ])
            else:
                sharp_edges.extend([lineAlpha, lineBeta])
                
        elif len(group) > 2:
            exclude_groups.append(group)
            
    return sharp_edges, exclude_groups

def get_sharp_edges(coincident_edge_groups, max_angle = 120, normal = False):
    """Process groups using thread pool executor"""
    global EXCLUDE_GROUPS
    
    # Use up to 8 threads
    num_threads = min(8, len(coincident_edge_groups))
    CORELOG.info(f"Using {num_threads} threads to process {len(coincident_edge_groups)} groups")
    
    # Calculate range for each thread
    groups_per_thread = len(coincident_edge_groups) // num_threads
    remainder = len(coincident_edge_groups) % num_threads
    
    # Create ranges for each thread
    ranges = []
    start = 0
    for i in range(num_threads):
        size = groups_per_thread + (1 if i < remainder else 0)
        ranges.append((start, start + size))
        start += size
        
    # Process ranges in parallel using ThreadPoolExecutor
    all_sharp_edges = []
    all_exclude_groups = []
    
    # Add overall progress bar
    with tqdm(total=len(coincident_edge_groups), desc="Overall progress") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for start, end in ranges:
                future = executor.submit(
                    process_group_range,
                    coincident_edge_groups,
                    start,
                    end,
                    max_angle, 
                    normal
                )
                futures.append(future)
                
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    sharp_edges, exclude_groups = future.result()
                    all_sharp_edges.extend(sharp_edges)
                    all_exclude_groups.extend(exclude_groups)
                    pbar.update(groups_per_thread)
                except Exception as e:
                    CORELOG.error(f"Error processing group: {str(e)}")
                    continue
    
    # Update global variable once at the end
    EXCLUDE_GROUPS.extend(all_exclude_groups)
            
    return all_sharp_edges

def visualize_model_and_normal_with_groups(shape, groups, showhelp=False):
    """
    Visualize the entire shape with coincident edges highlighted in red and other edges in green.
    All faces are displayed in light green. Additionally, visualize normals as arrows in purple and yellow.

    Parameters:
        shape (TopoDS_Shape): The shape to visualize.
        groups (list): A list of lists containing [(edge, normal), (edge, normal)] pairs.
    """
    # Initialize the display
    display, start_display, add_menu, add_function_to_menu = init_display()
    
    # Define colors
    red_color = Quantity_NOC_RED       # Red for coincident edges
    purple_color = Quantity_NOC_PURPLE # Purple for the first normal vector
    yellow_color = Quantity_NOC_YELLOW # Yellow for the second normal vector
    green_color = Quantity_NOC_GREEN
    light_green_color = Quantity_Color(0.8, 1.0, 0.8, Quantity_TOC_RGB)  # Light green for faces

    # Display all faces in light green
    display.DisplayShape(shape, transparency=0.5, update=True)

    # Iterate over all edge groups
    group_id = 0
    for group in groups:
        group_id += 1
        for idx, (edge, normal, midPoint, nearPoint, angle, valid) in enumerate(group):
            if normal:
                # Create an arrow to visualize the normal
                midpoint, _ = get_line_normal(edge)  # Function to get the midpoint of the edge
                arrow_end = gp_Pnt(midpoint.XYZ() + normal.Scaled(10).XYZ())  # Scale the normal vector
                arrow_edge = BRepBuilderAPI_MakeEdge(midpoint, arrow_end).Edge()

                if nearPoint and showhelp:
                    # Create a vertex to visualize the point
                    vertex = BRepBuilderAPI_MakeVertex(nearPoint).Vertex()
                    # Convert TopoDS_Vertex to Geom_Point
                    geom_point = Geom_CartesianPoint(BRep_Tool.Pnt(vertex))

                    ais_point = AIS_Point(geom_point)
                    point_color = purple_color if idx == 0 else yellow_color
                    ais_point.SetColor(Quantity_Color(point_color))

                    # Display the point
                    display.Context.Display(ais_point, False)

                # Display the arrow in purple or yellow based on index
                arrow_color = purple_color if idx == 0 else yellow_color
                # Display coincident edges in red
                if valid:
                    display.DisplayShape(edge, color=red_color, update=False)
                    if showhelp:
                        display.DisplayShape(arrow_edge, color=arrow_color, update=False)
                else:
                    display.DisplayShape(edge, color=light_green_color, update=False)
                    if showhelp:
                        display.DisplayShape(arrow_edge, color=arrow_color, update=False)
                
                # Display the angle text
                if idx == 0 and showhelp:
                    text_label = AIS_TextLabel()
                    text_label.SetText(f"Angle: {angle:.2f}°-- pair: {group_id}")
                    if valid:
                        text_label.SetColor(Quantity_Color(Quantity_NOC_RED))
                    else:
                        text_label.SetColor(Quantity_Color(Quantity_NOC_GREEN))
                    text_label.SetHeight(24.0)
                    text_label.SetPosition(midpoint)
                    # Disable depth test for the text label
                    text_label.SynchronizeAspects()
                    text_label.SetAutoHilight(True)

                    display.Context.Display(text_label, False)
            else:
                if valid:
                    display.DisplayShape(edge, color=red_color, update=False)
                else:
                    display.DisplayShape(edge, color=light_green_color, update=False)

    if len(CROSS_LINE) != 0 and showhelp:
        for line in CROSS_LINE:
            display.DisplayShape(line, color=yellow_color, update=False)
    if len(PASSED_LINE) != 0 and showhelp:
        for line in PASSED_LINE:
            display.DisplayShape(line, color=purple_color, update=False)
    if len(STUBBORN_LINE) != 0:
        for line in STUBBORN_LINE:
            display.DisplayShape(line, color=purple_color, update=False)
    # Update the viewer
    display.Context.UpdateCurrentViewer()

    # Fit the view to the entire shape
    display.FitAll()

    # Optionally, set an isometric view
    display.View_Iso()

    # Start the display (this call is blocking)
    start_display()

def visualize_groups(groups):
    """
    Visualize each group in EXCLUDE_GROUPS in a separate window, one at a time.
    """

    red_color = Quantity_NOC_RED       # Red for coincident edges
    purple_color = Quantity_NOC_PURPLE # Purple for the first normal vector
    yellow_color = Quantity_NOC_YELLOW # Yellow for the second normal vector
    green_color = Quantity_NOC_GREEN

    display, start_display, add_menu, add_function_to_menu = init_display()

    for group_index, groups in enumerate(groups):
        print(f"Displaying Group {group_index + 1}/{len(groups)}")
        display.Context.RemoveAll(True)

        for idx, (edge, face) in enumerate(groups):
            if idx == 0:
                display.DisplayShape(edge, color=yellow_color, update=False)
                display.DisplayShape(face, color=purple_color, update=False)
            elif idx == 1:
                display.DisplayShape(edge, color=purple_color, update=False)
                display.DisplayShape(face, color=yellow_color, update=False)
            elif idx == 2:
                display.DisplayShape(edge, color=yellow_color, update=False)
                display.DisplayShape(face, color=green_color, update=False)
            elif idx == 3:
                display.DisplayShape(edge, color=yellow_color, update=False)
                display.DisplayShape(face, color=red_color, update=False)

        display.FitAll()
        display.Context.UpdateCurrentViewer()

        input(f"Press Enter to continue to the next group ({group_index + 1}/{len(groups)})...")

    print("All groups have been displayed.")
    start_display()

def visualize_lines(lines):
    """
    Visualize each line in lines in a separate window, one at a time.
    """
    display, start_display, add_menu, add_function_to_menu = init_display()

    for group_index, line in enumerate(lines):
        print(f"Displaying line {group_index + 1}")
        display.Context.RemoveAll(True)

        display.DisplayShape(line, color=Quantity_NOC_RED, update=False)

        display.FitAll()
        display.Context.UpdateCurrentViewer()

        input(f"Press Enter to continue to the next line {group_index + 1}...")

    print("All groups have been displayed.")
    start_display()

def main():
    global CROSS_LINE, PASSED_LINE, EXCLUDE_GROUPS, OTHER_LINE, STUBBORN_LINE, CHECK_ONLY, PATH
    
    # 1. Load the STEP file
    try:
        CORELOG.info("Loading STEP file")
        shape = load_step_file(PATH)
        CORELOG.info("STEP file successfully loaded.")
    except ValueError as e:
        CORELOG.error(e)
        sys.exit(1)

    # 2. Build the edge to face mapping with a progress bar
    CORELOG.info("Building edge to face mapping...")
    edge_to_faces_map = build_edge_to_faces_map(shape)
    CORELOG.info(f"Edge to face mapping completed. Total edges: {len(edge_to_faces_map)}")

    # 3. Count the number of faces associated with each edge
    CORELOG.info("Counting the number of faces associated with each edge...")
    edge_face_counts = count_faces_per_edge(edge_to_faces_map)
    CORELOG.warning(f"{max(pair[1] for pair in edge_face_counts)} face/s max per edge!")
    CORELOG.info("Face counting completed.")

    # 4. Generate edge-face pairs with unique keys
    CORELOG.info("Generating edge-face pairs with unique keys...")
    edge_face_pairs = get_edge_key_sets(edge_to_faces_map)
    CORELOG.info(f"Generated {len(edge_face_pairs)} edge-face pairs.")

    # 5. Group coincident edges
    CORELOG.info("Grouping coincident edges...")
    coincident_edge_groups = group_coincident_edges(edge_face_pairs)
    CORELOG.warning(f"{max(pair[1] for pair in edge_face_counts)} face/s max per edge!")
    CORELOG.info(f"Found {len(coincident_edge_groups)} groups of coincident edges.")

    if not coincident_edge_groups:
        CORELOG.error("No coincident edge groups found. Exiting visualization.")
        sys.exit(0)
    
    show_shape = shape
    # 6. Find all Sharp edge
    CORELOG.info("Finding all sharp edges and remove them...")
    sharp_edges = get_sharp_edges(coincident_edge_groups, 120, True)
    if not CHECK_ONLY:
        edges = []
        for group in sharp_edges:
            # find all circle with same start and end point
            for idx, (edge, normal, midPoint, nearPoint, angle, valid) in enumerate(group):
                if valid and idx == 0:
                    key = get_edge_key_two(edge)
                    edges.append((key, edge))

        groups = defaultdict(list)
        for key, edge in edges:
            groups[key].append(edge)
        
        chamfer_distance = 0.05
        count = 0
        show_shape = shape
        for key in groups:
            chamfer_maker = BRepFilletAPI_MakeChamfer(show_shape)
            edges = groups[key]
            for edge in edges:
                try:
                    chamfer_maker.Add(chamfer_distance, edge)
                except RuntimeError as e:
                    CORELOG.error(f"Failed to add chamfer for edge in group: {e}")

            try:
                show_shape = chamfer_maker.Shape()
            except RuntimeError as e:
                for edge in edges:
                    STUBBORN_LINE.append(edge)
                count += 1
                # CORELOG.error(f"Failed to get shape: {e}")

        CORELOG.warning(f"{count} edeges cannot be chamfered!")

    # 8. Visualize all model
    CORELOG.info("Test only")
    if len(EXCLUDE_GROUPS) != 0:
        visualize_groups(EXCLUDE_GROUPS)
    # visualize_lines(OTHER_LINE)
    visualize_model_and_normal_with_groups(show_shape, sharp_edges, showhelp=False)

    # 9. Saving your modle
    if not CHECK_ONLY:
        CORELOG.info("Saving model")
        base_name, ext = os.path.splitext(PATH)
        path = f"{base_name}_mod{ext}"
        save_step_file(show_shape, path)

if __name__ == "__main__":
    PATH = "/home/ein/Dev/TechDemo/sources/Occt/4207-2163-LS1-A00.STEP"
    CHECK_ONLY = True
    main()

'''
Exception has occurred: RuntimeError
StdFail_NotDone: BRep_API: command not done raised from method Shape of class BRepBuilderAPI_MakeShape
  File "/home/ein/Dev/TechDemo/sources/Occt/find_sharp_edge.py", line 779, in main
    chamfered_shape = chamfer_maker.Shape()
  File "/home/ein/Dev/TechDemo/sources/Occt/find_sharp_edge.py", line 792, in <module>
    main()
RuntimeError: StdFail_NotDone: BRep_API: command not done raised from method Shape of class BRepBuilderAPI_MakeShape
'''