import sys

from PyQt5.QtWidgets import QApplication
# from log import CORELOG
from app import Application

def main():    
    # Create Qt application
    # CORELOG.info("Starting Qt Application")
    app = QApplication.instance()
    if app is None: app = QApplication(sys.argv)

    # CORELOG.info("Application initializing")
    core = Application()
    
    # CORELOG.info("Application started")
    core.render()
    
    # CORELOG.info("Application closed")
    core.close()
    # Run the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()