# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

from modules.show import MainWindow
from modules.plates import Plate

from utilities.log import CORELOG

def main():
    CORELOG.info("Starting application...")
    # Parameters
    path = "/Users/ein/EinDev/OcctStuff/.cache/test00"
    plateA = Plate.load(path, logger=CORELOG)
    # plateB = Plate.load(path, logger=CORELOG)

    CORELOG.info("make all...")
    _, ax = plt.subplots(dpi=100)
    CORELOG.info("Gen shapes...")
    shapeA, points = plateA.make(canvas=ax)
    # shapeB,_ = plateB.make(canvas=ax, point=points, offset=True, transform=True)
    CORELOG.info("Starting GUI...")
    app = QApplication(sys.argv)
    window = MainWindow(ax, [shapeA])
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()