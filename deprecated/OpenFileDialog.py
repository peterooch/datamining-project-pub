from PyQt5.QtWidgets import *
import pathlib
from os import path

class OpenFileDialog(QDialog):
    def __init__(self, parent):
        super(QDialog, self).__init__(parent)
        self.parent = parent
        self.setWindowTitle("Open training file")
        self.resize(600,230)
        
        self.TrainFilePath = QLineEdit(self)
        self.TrainFilePath.setDisabled(True)
        self.TrainFilePath.setGeometry(10, 10, 400, 25)
        self.PickTrainFile = QPushButton("Choose training file", self)
        self.PickTrainFile.setGeometry(410, 10, 180, 25)
        self.PickTrainFile.clicked.connect(lambda x: self.fileDialog(x, self.TrainFilePath))

        self.StructFilePath = QLineEdit(self)
        self.StructFilePath.setDisabled(True)
        self.StructFilePath.setGeometry(10, 40, 400, 25)
        self.PickStructFile = QPushButton("Choose structure file", self)
        self.PickStructFile.setGeometry(410, 40, 180, 25)
        self.PickStructFile.clicked.connect(lambda x: self.fileDialog(x, self.StructFilePath))

        self.TestFilePath = QLineEdit(self)
        self.TestFilePath.setDisabled(True)
        self.TestFilePath.setGeometry(10, 70, 400, 25)
        self.PickTestFile = QPushButton("Choose test file", self)
        self.PickTestFile.setGeometry(410, 70, 180, 25)
        self.PickTestFile.clicked.connect(lambda x: self.fileDialog(x, self.TestFilePath))

        self.ResultDirPath = QLineEdit(self)
        self.ResultDirPath.setDisabled(True)
        self.ResultDirPath.setGeometry(10, 100, 400, 25)
        self.PickResultDir = QPushButton("Choose ouput folder", self)
        self.PickResultDir.setGeometry(410, 100, 180, 25)
        self.PickResultDir.clicked.connect(lambda x: self.fileDialog(x, self.ResultDirPath))

        self.BinLabel = QLabel("Bin count:", self)
        txwidth = self.fontMetrics().horizontalAdvance("Bin count:")
        self.BinLabel.setGeometry(10, 130, txwidth, 25)
        self.BinCount = QLineEdit("5", self)
        self.BinCount.setGeometry(10 + txwidth + 2, 130, 50, 25)

        self.OkButton = QPushButton("OK", self)
        self.OkButton.setGeometry(10,170,100,50)
        self.OkButton.clicked.connect(self.okClick)
        self.CancelButton = QPushButton("Cancel", self)
        self.CancelButton.setGeometry(110,170,100,50)
        self.CancelButton.clicked.connect(self.close)

    def fileDialog(self, *args):
        control = args[1]
        if control is self.ResultDirPath:
            # https://doc.qt.io/qt-5/qfiledialog.html#getExistingDirectory
            path = QFileDialog.getExistingDirectory(self, "Choose folder", str(pathlib.Path(__file__).parent.absolute()))
        else:
            file_filter = "CSV Files (*.csv)" if control is not self.StructFilePath else "Structure Files (*.txt)"
            path = QFileDialog.getOpenFileName(self, "Choose file", str(pathlib.Path(__file__).parent.absolute()), file_filter)[0]
        control.setText(path)

    def okClick(self):
        train_file = self.TrainFilePath.text()
        struct_file = self.StructFilePath.text()
        test_file = self.TestFilePath.text()
        output_dir = self.ResultDirPath.text()
        bin_count = self.BinCount.text()

        # Error checking
        try:
            if not path.isfile(train_file):
                raise ValueError("Training file does not exists.")
            elif struct_file and not path.isfile(struct_file):
                raise ValueError("Structure file does not exists.")
            elif test_file and not path.isfile(test_file):
                raise ValueError("Test file does not exists.")
            elif output_dir and not path.isdir(output_dir):
                raise ValueError("Output folder does not exists.")
            try:
                bin_count = int(bin_count)
            except: # Replace python-speak with custom message
                raise ValueError("Invalid bin count value.")
            if bin_count <= 0:
                raise ValueError("Bin count must be greater than 0.")
        except ValueError as ve:
            # https://doc.qt.io/qt-5/qmessagebox.html#information
            QMessageBox.information(self, "Error", str(ve))
            return

        # Everything is correct
        if struct_file == "":
            struct_file = None
        if test_file == "":
            test_file = None
        if output_dir == "":
            output_dir = None
        # Pass data to parent to build the models
        # parent.buildModel(train_file, struct_file, bin_count, test_file, output_dir)
        self.close()
