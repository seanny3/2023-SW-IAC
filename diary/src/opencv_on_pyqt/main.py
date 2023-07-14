import sys
 
from PyQt5.QtWidgets import QApplication, QWidget
 
 
class QtGUI(QWidget):
 
    def __init__(self):
 
        super().__init__()
 
        self.show()
 
 
if __name__ == '__main__':
 
    app = QApplication(sys.argv)
 
    ex = QtGUI()
 
    app.exec_()
