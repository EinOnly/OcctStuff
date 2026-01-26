# -*- coding: utf-8 -*-
import sys
import os
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

from modules.show import MainWindow
from modules.plates import Plate

from utilities.log import CORELOG

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

def save_shapes(shapes, output_dir=".cache/output", filename="combined_shapes.step"):
    """
    Save a list of TopoDS_Shape objects to a STEP file.

    Parameters:
    - shapes: list of TopoDS_Shape objects
    - output_dir: directory to save the file
    - filename: name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Create a compound to hold all shapes
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for shape in shapes:
        if shape and not shape.IsNull():
            builder.Add(compound, shape)

    # Write to STEP file
    step_writer = STEPControl_Writer()
    step_writer.Transfer(compound, STEPControl_AsIs)
    status = step_writer.Write(filepath)

    if status == IFSelect_RetDone:
        CORELOG.info(f"[✓] Shapes saved to: {os.path.abspath(filepath)}")
        return filepath
    else:
        CORELOG.error(f"[✗] Failed to save shapes to: {filepath}")
        return None

def main():
    CORELOG.info("Starting application...")
    # Parameters
    path = "/Users/ein/EinDev/OcctStuff/.cache/test00"
    plateA = Plate.load(path, logger=CORELOG)
    plateB = Plate.load(path, logger=CORELOG)

    CORELOG.info("make all...")
    _, ax = plt.subplots(dpi=100)
    CORELOG.info("Gen shapes...")
    shapeA, points = plateA.make(canvas=ax, move=1)
    shapeB,_ = plateB.make(canvas=ax, point=points, offset=True, transform=True)

    # Save the generated shapes
    CORELOG.info("Saving shapes...")
    save_shapes([shapeA, shapeB], output_dir=".cache/output", filename="plates_combined.step")

    CORELOG.info("Starting GUI...")
    app = QApplication(sys.argv)
    window = MainWindow(ax, [shapeA, shapeB])
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()