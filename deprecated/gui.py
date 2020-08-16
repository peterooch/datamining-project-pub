from MainWindow import MainWindow
from PyQt5.QtWidgets import QApplication

# GUI FrontEnd
if __name__ == "__main__":
    app = QApplication([])
    mainWnd = MainWindow(app)
    mainWnd.run()

