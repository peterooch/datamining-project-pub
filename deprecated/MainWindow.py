import sys
from PyQt5.QtWidgets import *
from OpenFileDialog import OpenFileDialog

class MainWindow(QWidget):
    def __init__(self, qApp):
        super(QWidget, self).__init__()
        self.app = qApp
        self.setWindowTitle("Datamining project GUI frontend")
        self.fileDiag = OpenFileDialog(self)

        #set Open file button
        self.OpenFileButton = QPushButton(self)
        self.OpenFileButton.setText("Open files")
        self.OpenFileButton.setGeometry(0,0,180,50)
        self.OpenFileButton.clicked.connect(self.fileDiag.exec)
        #set Change folder button (disable for now)
        #Save_diagrams_button = QtWidgets.QPushButton(w)
        #Save_diagrams_button.setText("Save diagrams")
        #Save_diagrams_button.setGeometry(180,0,180,50)

        rect = self.app.primaryScreen().availableGeometry()
        #set default label
        labelText = "Please click the \"Open files\" button to start working with this program"
        labelWidth = self.fontMetrics().horizontalAdvance(labelText)
        self.MainLabel = QLabel(self)
        self.MainLabel.setText(labelText)
        self.MainLabel.move(rect.width() / 2 - labelWidth / 2, rect.height() / 2)

        labelText = "Programmed by Baruch Rutman and Roi Amzallag"
        self.authLabel = QLabel(self)
        self.authLabel.setText(labelText)
        self.authLabel.move(10, rect.height() - self.fontMetrics().height() - 25)

    def run(self):
        self.showMaximized()
        self.app.exec()
