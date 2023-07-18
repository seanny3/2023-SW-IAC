import sys
import cv2, threading
import numpy as np
from PIL import Image
from time import time
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO
from futils import add_salt_and_pepper_noise, add_gaussian_noise, bg_removal

form_class = uic.loadUiType("design.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        
        # 색 변환 버튼 그룹
        self.cvt_group_btn = QButtonGroup()
        self.cvt_group_btn.setExclusive(False)
        self.cvt_group_btn.addButton(self.cvt_gray_btn)
        
        # 노이즈 추가 버튼 그룹
        self.noise_group_btn = QButtonGroup()
        self.noise_group_btn.setExclusive(False)
        self.noise_group_btn.addButton(self.noise_salt_btn)
        self.noise_group_btn.addButton(self.noise_gaus_btn)
        
        # 필터링 버튼 그룹
        self.filter_group_btn = QButtonGroup()
        self.filter_group_btn.setExclusive(False)
        self.filter_group_btn.addButton(self.lpf_avg_btn)
        self.filter_group_btn.addButton(self.lpf_med_btn)
        self.filter_group_btn.addButton(self.lpf_gaus_btn)
        self.filter_group_btn.addButton(self.hpf_sharp1_btn)
        self.filter_group_btn.addButton(self.hpf_sharp2_btn)
        self.filter_group_btn.addButton(self.hpf_sharp3_btn)
        
        # 에지검출 버튼 그룹
        self.edge_group_btn = QButtonGroup()
        self.edge_group_btn.setExclusive(False)
        self.edge_group_btn.addButton(self.edge_sobel_btn)
        self.edge_group_btn.addButton(self.edge_prewitt_btn)
        self.edge_group_btn.addButton(self.edge_robert_btn)
        self.edge_group_btn.addButton(self.edge_canny_btn)
        self.edge_group_btn.addButton(self.edge_otsu_btn)
        self.edge_group_btn.addButton(self.edge_laplacian_btn)
        
        # 그룹별 버튼 토글링
        self.cvt_group_btn.buttonClicked.connect(self.cvt_RadioBtnToggle)
        self.noise_group_btn.buttonClicked.connect(self.noise_RadioBtnToggle)
        self.filter_group_btn.buttonClicked.connect(self.filter_RadioBtnToggle)
        self.edge_group_btn.buttonClicked.connect(self.edge_RadioBtnToggle)
        
        # menu bar events
        self.file_load_video.triggered.connect(self.loadVideo)
        self.file_load_model.triggered.connect(self.loadModel)
        self.file_run_webcam.triggered.connect(self.runWebcam)
        self.file_initAll.triggered.connect(self.initAll)
        
        # current file info
        self.loaded_file_info = {"video": "", "model": ""}
        
        self.th = None
        self.thread_is_running = False
        
    def camRun(self):
        if self.loaded_file_info["model"]:
            model = YOLO(self.loaded_file_info["model"])
        cap = cv2.VideoCapture(self.loaded_file_info["video"])
        
        size = (640,480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        
        fps = 0
        frame_count = 0
        start_time = time()
        while self.thread_is_running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h,w=frame.shape[:2]
                ash=size[1]/h
                asw=size[0]/w
                if asw<ash:
                    sizeas=(int(w*asw),int(h*asw))
                else:
                    sizeas=(int(w*ash),int(h*ash))
                frame = cv2.resize(frame, dsize=sizeas)
                
                # cvt gray-scale
                if self.cvt_gray_btn.isChecked():
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # make it noisy
                if self.noise_salt_btn.isChecked():
                    frame = add_salt_and_pepper_noise(frame, 0.05)
                if self.noise_gaus_btn.isChecked():
                    frame = add_gaussian_noise(frame, 20)
                    
                # filtering function
                if self.lpf_avg_btn.isChecked():
                    frame = cv2.blur(frame, (3,3))
                if self.lpf_med_btn.isChecked():
                    frame = cv2.medianBlur(frame, 3)
                if self.lpf_gaus_btn.isChecked():
                    frame = cv2.GaussianBlur(frame, (5,5), 1)
                if self.hpf_sharp1_btn.isChecked():
                    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                    frame = cv2.filter2D(frame, -1, kernel)
                if self.hpf_sharp2_btn.isChecked():
                    kernel = np.array([[1,1,1],[1,-7,1],[1,1,1]])
                    frame = cv2.filter2D(frame, -1, kernel)
                if self.hpf_sharp3_btn.isChecked():
                    kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
                    frame = cv2.filter2D(frame, -1, kernel)
                
                # edge detection
                if self.edge_sobel_btn.isChecked():     # sobel
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    x = cv2.Sobel(frame, -1, 1, 0, ksize=3, delta=0)
                    y = cv2.Sobel(frame, -1, 0, 1, ksize=3, delta=0)
                    frame = x+y
                if self.edge_prewitt_btn.isChecked():   # prewitt
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
                    gy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)

                    x = cv2.filter2D(frame, -1, gx, delta=0)
                    y = cv2.filter2D(frame, -1, gy, delta=0)
            
                    frame = x+y
                if self.edge_robert_btn.isChecked():    # robert cross
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gx = np.array([[-1, 0], [0, 1]], dtype=int)
                    gy = np.array([[0, -1], [1, 0]], dtype=int)
                    
                    x = cv2.filter2D(frame, -1, gx, delta=0)
                    y = cv2.filter2D(frame, -1, gy, delta=0)
                
                    frame = x+y
                if self.edge_canny_btn.isChecked():     # canny
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.Canny(frame,100,200)
                if self.edge_otsu_btn.isChecked():      # OTSU-canny
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    otsu_th, otsu_binary = cv2.threshold(frame, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    frame = cv2.Canny(frame, otsu_th, otsu_th)
                if self.edge_laplacian_btn.isChecked(): # laplacian
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.Laplacian(frame, -1)
                    
                if self.loaded_file_info["model"]:
                    # OpenCV 이미지를 PIL 이미지로 변환
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # 객체 인식
                    results = model(pil_image)
                    frame = results[0].plot()     # YOLOv8
                
                if len(frame.shape) < 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                base_frame = np.zeros((size[1],size[0],3), np.uint8)
                base_frame[
                    int(size[1]/2-sizeas[1]/2):int(size[1]/2+sizeas[1]/2),
                    int(size[0]/2-sizeas[0]/2):int(size[0]/2+sizeas[0]/2),
                    :
                ] = frame
                
                frame = base_frame
                
                # 1초당 몇 개 프레임인지
                frame_count += 1
                current_time = time()
                elapsed_time = current_time - start_time
                
                if elapsed_time > 1.0:
                    fps = frame_count
                    start_time = current_time
                    frame_count = 0
                            
                cv2.putText(frame, f"FPS: {fps}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 1)       
                
                h,w,c = frame.shape
                qImg = QImage(frame.data, w, h, w*c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.label_camView.setPixmap(pixmap)
                
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
        cap.release()
        self.label_camView.setPixmap(QPixmap.fromImage(QImage()))
        print(f"[{sys._getframe().f_code.co_name}] Thread end.")

    def camStart(self) :
        self.thread_is_running = True
        self.th = threading.Thread(target=self.camRun)
        self.th.start()

    def camStop(self):
        self.thread_is_running = False
    
    def closeEvent(self, event):
        quit_msg = "종료하시겠습니까?"
        reply = QMessageBox.question(self, '메시지', quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            print(f"[{sys._getframe().f_code.co_name}] Exit")
            self.camStop()
        else:
            event.ignore()
            
    def cvt_RadioBtnToggle(self, radioButton):
        for button in self.cvt_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
           
    def noise_RadioBtnToggle(self, radioButton):
        for button in self.noise_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
    
    def filter_RadioBtnToggle(self, radioButton):
        for button in self.filter_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
                
    def edge_RadioBtnToggle(self, radioButton):
        for button in self.edge_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
            
    def loadVideo(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video Files (*.avi *.mp4 *.mkv *.mov)")

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            if self.thread_is_running:
                self.camStop()
                self.th.join()
                
            self.loaded_file_info["video"] = file_path
            self.camStart()
    
    def loadModel(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Model Files (*.pt *.pth)")

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            if self.thread_is_running:
                self.camStop()
                self.th.join()
            
            self.loaded_file_info["model"] = file_path
            self.camStart()
    
    def runWebcam(self):
        if self.thread_is_running:
            self.camStop()
            self.th.join()
            
        self.loaded_file_info["video"] = 0
        self.camStart()

    def initAll(self):
        if self.thread_is_running:
            self.camStop()
            self.th.join()
        
        self.loaded_file_info["video"] = ""
        self.loaded_file_info["model"] = ""
        
        
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())