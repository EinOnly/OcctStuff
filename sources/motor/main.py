import sys
from PyQt5.QtWidgets import QApplication
from log import CORELOG

def main():    
    # Create Qt application
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Set up logging
    CORELOG.info("Application started")

    # Run the event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
