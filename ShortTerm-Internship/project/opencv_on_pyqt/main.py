import sys
import cv2, threading
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

form_class = uic.loadUiType("design.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        self.btn_camStart.clicked.connect(self.camStart)
        self.btn_camStop.clicked.connect(self.camStop)
        
        self.running = False
        
    def camRun(self):
        cap = cv2.VideoCapture(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.label_camView.resize(width, height)
        while self.running:
            ret, img = cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             
                
                h,w,c = img.shape
                qImg = QImage(img.data, w, h, w*c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.label_camView.setPixmap(pixmap)
            else:
                QMessageBox.about(QWidget(), "Error", "Cannot read frame.")
                print(f"[{sys._getframe().f_code.co_name}] cannot read frame.")
                break
            
        cap.release()
        print(f"[{sys._getframe().f_code.co_name}] Thread end.")

    def camStart(self) :
        self.running = True
        th = threading.Thread(target=self.camRun)
        th.start()
        print(f"[{sys._getframe().f_code.co_name}] started!")

    def camStop(self):
        self.running = False
        print(f"[{sys._getframe().f_code.co_name}] stopped!")
    
    def closeEvent(self, event):
        quit_msg = "종료하시겠습니까?"
        reply = QMessageBox.question(self, 'Message', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            
        print(f"[{sys._getframe().f_code.co_name}] Exit")
        self.camStop()
            
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())