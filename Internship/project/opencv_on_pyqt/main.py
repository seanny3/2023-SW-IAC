import sys
import cv2, threading
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from time import time

from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from ultralytics import YOLO, RTDETR, NAS
from ultralytics.utils.plotting import Annotator

from futils import add_salt_and_pepper_noise, add_gaussian_noise, bg_removal
import range_slider

form_class = uic.loadUiType("design.ui")[0]

classes = ['0','1','2','3','4','5','6','7','8','9',
           '가','나','다','라','마','바','사','아','자','거',
           '너','더','러','머','버','서','어','저','고','노',
           '도','로','모','보','소','오','조','구','누','두',
           '루','무','부','수','우','주','배','하','허','호',
           '국','합','육','해','공','울','산','대','인','천',
           '전','광','제','경','기','강','원','충','남','북',
           '외','교','영','준','협','정','표','세','종','전기차'
           ]
fontpath = "fonts/MALGUN.TTF"
font = ImageFont.truetype(fontpath, 25)

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        
        # 히스토그램 캔버스
        self.histogram_widget.setVisible(False)
        
        self.histogram_layout = QVBoxLayout(self.histogram_widget)
        self.figure = plt.figure(figsize=(16,4))
        plt.rcParams.update({
            "figure.facecolor":  (0.0, 0.0, 0.0, 0),
            "axes.facecolor":    (0.0, 0.0, 0.0, 0),
            "text.color" : "red",
            "axes.labelcolor" : "red",
            "axes.edgecolor" : (0,0,0,0),
            "xtick.color": "red",
        })
        
        # plt.tick_params(axis='x', colors='red')
        self.figure.patch.set_facecolor("None")
        self.histogram_canvas = FigureCanvas(self.figure)
        self.histogram_canvas.setStyleSheet("background-color: transparent;")
        self.histogram_layout.addWidget(self.histogram_canvas)
        self.histogram_view_btn.clicked.connect(self.histogram_widgetToggle)
        
        # Histogram Stretching 범위 지정 슬라이더 생성
        self.hist_stretch_minVal = QLabel()
        self.hist_stretch_minVal.setObjectName(u"hist_stretch_minVal")
        self.hist_stretch_minVal.setText("0")
        self.hist_stretch_maxVal = QLabel()
        self.hist_stretch_maxVal.setObjectName(u"hist_stretch_maxVal")
        self.hist_stretch_maxVal.setText("255")
        
        self.hist_stretch_slider = range_slider.RangeSlider(Qt.Horizontal)
        self.hist_stretch_slider.setMinimumHeight(30)
        self.hist_stretch_slider.setMinimum(0)
        self.hist_stretch_slider.setMaximum(255)
        self.hist_stretch_slider.setLow(0)
        self.hist_stretch_slider.setHigh(255)
        self.hist_stretch_slider.setTickPosition(QSlider.TicksBelow)
        
        self.hist_stretch_slider_layout = QHBoxLayout()
        self.hist_stretch_slider_layout.setObjectName(u"hist_stretch_slider_layout")
        self.hist_stretch_slider_layout.addWidget(self.hist_stretch_minVal)
        self.hist_stretch_slider_layout.addWidget(self.hist_stretch_slider)
        self.hist_stretch_slider_layout.addWidget(self.hist_stretch_maxVal)
        self.hist_stretch_slider_layout.setStretch(0, 1)
        self.hist_stretch_slider_layout.setStretch(1, 8)
        self.hist_stretch_slider_layout.setStretch(2, 1)
        
        self.hist_stretch_slider.sliderMoved.connect(self.histStretch_sliderEvent)
        self.hist_verticalLayout.insertLayout(1, self.hist_stretch_slider_layout)
        
        # Canny 범위 지정 슬라이더 생성
        self.edge_canny_minVal = QLabel()
        self.edge_canny_minVal.setObjectName(u"edge_canny_minVal")
        self.edge_canny_minVal.setText("100")
        self.edge_canny_maxVal = QLabel()
        self.edge_canny_maxVal.setObjectName(u"edge_canny_maxVal")
        self.edge_canny_maxVal.setText("200")
        
        self.edge_canny_slider = range_slider.RangeSlider(Qt.Horizontal)
        self.edge_canny_slider.setMinimumHeight(30)
        self.edge_canny_slider.setMinimum(0)
        self.edge_canny_slider.setMaximum(255)
        self.edge_canny_slider.setLow(100)
        self.edge_canny_slider.setHigh(200)
        self.edge_canny_slider.setTickPosition(QSlider.TicksBelow)
        
        self.edge_canny_slider_layout = QHBoxLayout()
        self.edge_canny_slider_layout.setObjectName(u"edge_canny_slider_layout")
        self.edge_canny_slider_layout.addWidget(self.edge_canny_minVal)
        self.edge_canny_slider_layout.addWidget(self.edge_canny_slider)
        self.edge_canny_slider_layout.addWidget(self.edge_canny_maxVal)
        self.edge_canny_slider_layout.setStretch(0, 1)
        self.edge_canny_slider_layout.setStretch(1, 8)
        self.edge_canny_slider_layout.setStretch(2, 1)
        
        self.edge_canny_slider.sliderMoved.connect(self.edgeCanny_sliderEvent)
        self.edge_verticalLayout.insertLayout(7, self.edge_canny_slider_layout)
        
        # 전역 스레시홀딩 범위 지정 슬라이더 생성
        # range slider 생성
        self.binary_global_minVal = QLabel()
        self.binary_global_minVal.setObjectName(u"binary_global_minVal")
        self.binary_global_minVal.setText("100")
        self.binary_global_maxVal = QLabel()
        self.binary_global_maxVal.setObjectName(u"binary_global_maxVal")
        self.binary_global_maxVal.setText("255")
        
        self.binary_global_slider = range_slider.RangeSlider(Qt.Horizontal)
        self.binary_global_slider.setMinimumHeight(30)
        self.binary_global_slider.setMinimum(0)
        self.binary_global_slider.setMaximum(255)
        self.binary_global_slider.setLow(100)
        self.binary_global_slider.setHigh(255)
        self.binary_global_slider.setTickPosition(QSlider.TicksBelow)
        
        self.binary_global_slider_layout = QHBoxLayout()
        self.binary_global_slider_layout.setObjectName(u"binary_global_slider_layout")
        self.binary_global_slider_layout.addWidget(self.binary_global_minVal)
        self.binary_global_slider_layout.addWidget(self.binary_global_slider)
        self.binary_global_slider_layout.addWidget(self.binary_global_maxVal)
        self.binary_global_slider_layout.setStretch(0, 1)
        self.binary_global_slider_layout.setStretch(1, 8)
        self.binary_global_slider_layout.setStretch(2, 1)
        
        self.binary_global_slider.sliderMoved.connect(self.binaryGlobal_sliderEvent)
        self.binary_verticalLayout.insertLayout(1, self.binary_global_slider_layout)
        
        
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
        
        # 히스토그램
        self.hist_group_btn = QButtonGroup()
        self.hist_group_btn.setExclusive(False)
        self.hist_group_btn.addButton(self.hist_stretch_btn)
        self.hist_group_btn.addButton(self.hist_equal_btn)
        
        # 에지검출 버튼 그룹 + 이진화 버튼 그룹
        self.edgeNbinary_group_btn = QButtonGroup()
        self.edgeNbinary_group_btn.setExclusive(False)
        self.edgeNbinary_group_btn.addButton(self.edge_sobel_btn)
        self.edgeNbinary_group_btn.addButton(self.edge_prewitt_btn)
        self.edgeNbinary_group_btn.addButton(self.edge_robert_btn)
        self.edgeNbinary_group_btn.addButton(self.edge_canny_btn)
        self.edgeNbinary_group_btn.addButton(self.edge_otsu_btn)
        self.edgeNbinary_group_btn.addButton(self.edge_laplacian_btn)
        
        self.edgeNbinary_group_btn.addButton(self.binary_global_btn)
        self.edgeNbinary_group_btn.addButton(self.binary_adaptive_btn1)
        self.edgeNbinary_group_btn.addButton(self.binary_adaptive_btn2)
        self.edgeNbinary_group_btn.addButton(self.binary_otsu_btn)
        self.edgeNbinary_group_btn.addButton(self.binary_triangle_btn)
        
        # 그룹별 버튼 토글링
        self.cvt_group_btn.buttonClicked.connect(self.cvt_RadioBtnToggle)
        self.noise_group_btn.buttonClicked.connect(self.noise_RadioBtnToggle)
        self.filter_group_btn.buttonClicked.connect(self.filter_RadioBtnToggle)
        self.edgeNbinary_group_btn.buttonClicked.connect(self.edgeNbinary_RadioBtnToggle)
        self.hist_group_btn.buttonClicked.connect(self.hist_RadioBtnToggle)
        
        # menu bar events
        self.file_load_video.triggered.connect(self.loadVideo)
        self.file_load_model.triggered.connect(self.loadModel)
        self.file_run_webcam.triggered.connect(self.runWebcam)
        self.file_initAll.triggered.connect(self.initAll)
        
        # slider value changed event
        self.noise_salt_slider.valueChanged.connect(self.noiseSalt_sliderEvent)
        self.noise_gaus_slider.valueChanged.connect(self.noiseGaus_sliderEvent)
        self.lpf_avg_slider.valueChanged.connect(self.lpfAvg_sliderEvent)
        self.lpf_med_slider.valueChanged.connect(self.lpfMed_sliderEvent)
        self.lpf_gaus_kslider.valueChanged.connect(self.lpfGaus_ksliderEvent)
        self.lpf_gaus_sslider.valueChanged.connect(self.lpfGaus_ssliderEvent)
        self.binary_adaptive_bslider1.valueChanged.connect(self.binaryAdaptive_bslider1Event)
        self.binary_adaptive_cslider1.valueChanged.connect(self.binaryAdaptive_cslider1Event)
        self.binary_adaptive_bslider2.valueChanged.connect(self.binaryAdaptive_bslider2Event)
        self.binary_adaptive_cslider2.valueChanged.connect(self.binaryAdaptive_cslider2Event)
        
        # current file info
        self.loaded_file_info = {"video": "", "model": ""}
        
        self.th = None
        self.thread_is_running = False
        
        # opencv params
        self.noise_salt_param = 0.01
        self.noise_gaus_param = 1
        self.lpf_avg_ksize = 1
        self.lpf_med_ksize = 3
        self.lpf_gaus_ksize = 1
        self.lpf_gaus_sig = 1
        self.hist_stretch_minThreshold = 0
        self.hist_stretch_maxThreshold = 255
        self.edge_canny_minThreshold = 100
        self.edge_canny_maxThreshold = 200
        self.binary_global_minThreshold = 100
        self.binary_global_maxThreshold = 255
        self.binary_adaptive_mean_b = 3
        self.binary_adaptive_mean_c = 1
        self.binary_adaptive_gaus_b = 3
        self.binary_adaptive_gaus_c = 1
        
        self.current_xval = 0
        
        self.car_number = ""
        
    def camRun(self):
        if self.loaded_file_info["model"]:
            model = YOLO(self.loaded_file_info["model"])
            # model = NAS(self.loaded_file_info["model"])
            # model = RTDETR(self.loaded_file_info["model"])
        
        path = self.loaded_file_info["video"]
        ext = 0
        if path != 0:
            ext = path.split(".")[-1]
        
        size = (640,480)
        
        isVideo = False
        
        if ext in [0, "avi", "mp4", "mkv", "mov"]:
            cap = cv2.VideoCapture(self.loaded_file_info["video"])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        
            fps = 0
            start_time = time()
            isVideo = True
        else:
            img = cv2.imread(self.loaded_file_info["video"])
                    
        
        while self.thread_is_running:
            if isVideo:
                ret, img = cap.read()
            else:
                ret = True
                
            if ret:
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                h,w=frame.shape[:2]
                ash=size[1]/h
                asw=size[0]/w
                if asw<ash:
                    sizeas=(int(w*asw),int(h*asw))
                else:
                    sizeas=(int(w*ash),int(h*ash))
                    
                frame = cv2.resize(frame, dsize=sizeas, interpolation=cv2.INTER_AREA if w > size[0] else cv2.INTER_LANCZOS4)
                
                # make it noisy
                if self.noise_salt_btn.isChecked():
                    frame = add_salt_and_pepper_noise(frame, self.noise_salt_param)
                if self.noise_gaus_btn.isChecked():
                    frame = add_gaussian_noise(frame, self.noise_gaus_param)
                    
                # cvt gray-scale
                if self.cvt_gray_btn.isChecked():
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # filtering function
                if self.lpf_avg_btn.isChecked():
                    frame = cv2.blur(frame, (self.lpf_avg_ksize,self.lpf_avg_ksize))
                if self.lpf_med_btn.isChecked():
                    frame = cv2.medianBlur(frame, self.lpf_med_ksize)
                if self.lpf_gaus_btn.isChecked():
                    frame = cv2.GaussianBlur(frame, (self.lpf_gaus_ksize,self.lpf_gaus_ksize), self.lpf_gaus_sig)
                if self.hpf_sharp1_btn.isChecked():
                    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                    frame = cv2.filter2D(frame, -1, kernel)
                if self.hpf_sharp2_btn.isChecked():
                    kernel = np.array([[1,1,1],[1,-7,1],[1,1,1]])
                    frame = cv2.filter2D(frame, -1, kernel)
                if self.hpf_sharp3_btn.isChecked():
                    kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
                    frame = cv2.filter2D(frame, -1, kernel)
                
                # histogram processing
                if self.hist_stretch_btn.isChecked():
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame = cv2.normalize(frame, None, self.hist_stretch_minThreshold, self.hist_stretch_maxThreshold, cv2.NORM_MINMAX)
                if self.hist_equal_btn.isChecked():
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame = cv2.equalizeHist(frame)
                
                
                # show histogram
                if self.histogram_view_btn.isChecked():
                    self.figure.clear()
                    
                    hist = cv2.calcHist([frame],[0],None,[256],[0,256])
                    plt.plot(hist, alpha=0.6)
                    plt.yticks([])
                    plt.axvline(self.current_xval, 0, 1, color='red', linestyle='-', linewidth=1)
                    
                    self.histogram_canvas.draw()
                
                # edge detection
                if self.edge_sobel_btn.isChecked():     # sobel
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    x = cv2.Sobel(frame, -1, 1, 0, ksize=3, delta=0)
                    y = cv2.Sobel(frame, -1, 0, 1, ksize=3, delta=0)
                    frame = x+y
                if self.edge_prewitt_btn.isChecked():   # prewitt
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
                    gy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)

                    x = cv2.filter2D(frame, -1, gx, delta=0)
                    y = cv2.filter2D(frame, -1, gy, delta=0)
            
                    frame = x+y
                if self.edge_robert_btn.isChecked():    # robert cross
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    gx = np.array([[-1, 0], [0, 1]], dtype=int)
                    gy = np.array([[0, -1], [1, 0]], dtype=int)
                    
                    x = cv2.filter2D(frame, -1, gx, delta=0)
                    y = cv2.filter2D(frame, -1, gy, delta=0)
                
                    frame = x+y
                if self.edge_canny_btn.isChecked():     # canny
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame = cv2.Canny(frame,self.edge_canny_minThreshold,self.edge_canny_maxThreshold)
                if self.edge_otsu_btn.isChecked():      # OTSU-canny
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    otsu_th, otsu_binary = cv2.threshold(frame, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    frame = cv2.Canny(frame, otsu_th, otsu_th)
                if self.edge_laplacian_btn.isChecked(): # laplacian
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    frame = cv2.Laplacian(frame, -1)
                
                # Binarization
                if self.binary_global_btn.isChecked():
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, frame = cv2.threshold(frame, self.binary_global_minThreshold, self.binary_global_maxThreshold, cv2.THRESH_BINARY)
                if self.binary_adaptive_btn1.isChecked():
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.binary_adaptive_mean_b, self.binary_adaptive_mean_c)
                if self.binary_adaptive_btn2.isChecked():
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.binary_adaptive_gaus_b, self.binary_adaptive_gaus_c)
                if self.binary_otsu_btn.isChecked():
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    th, frame = cv2.threshold(frame, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)                
                    self.current_xval = th
                if self.binary_triangle_btn.isChecked():
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    th, frame = cv2.threshold(frame, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)                
                    self.current_xval = th
                    
                if self.loaded_file_info["model"]:
                    # 객체 인식
                    results = model.predict(frame)
                    
                    # 바운딩 박스                 
                    for r in results:
                        annotator = Annotator(frame)
                        
                        boxes = r.boxes
                        for box in boxes:
                            
                            b = box.xyxy[0]
                            c = box.cls
                            annotator.box_label(b)
                    
                    frame = annotator.result()
                    
                    # 차량번호 정렬
                    data = results[0].boxes.cpu().numpy().data
                    sorted_data = data[data[:,0].argsort()]
                    sorted_clsIdx = [int(row[5]) for row in sorted_data]
                    
                    # 차량번호 결과
                    self.car_number = ''.join(classes[idx] for idx in sorted_clsIdx)
                    
                    # OpenCV 이미지를 PIL 이미지로 변환
                    pil_image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((sorted_data[0][0], sorted_data[0][1]-40),  self.car_number, font=font, fill=(255,0,0))
                    
                    frame = np.array(pil_image)
                
                if len(frame.shape) < 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                base_frame = np.zeros((size[1],size[0],3), np.uint8)
                base_frame[
                    int(size[1]/2-sizeas[1]/2):int(size[1]/2+sizeas[1]/2),
                    int(size[0]/2-sizeas[0]/2):int(size[0]/2+sizeas[0]/2),
                    :
                ] = frame
                
                frame = base_frame
                
                if isVideo:
                    # 1초당 몇 개 프레임인지
                    current_time = time()
                    elapsed_time = current_time - start_time
                    
                    if elapsed_time > 0.3:
                        fps = 1/(current_time - start_time)
                        start_time = current_time
                                
                    cv2.putText(frame, f"FPS: {fps:.3f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 1)   
                
                h,w,c = frame.shape
                qImg = QImage(frame.data, w, h, w*c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.label_camView.setPixmap(pixmap)
                
                if not isVideo: cv2.waitKey(1000)
                
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        if isVideo: 
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
        self.current_xval = 0
        for button in self.cvt_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
           
    def noise_RadioBtnToggle(self, radioButton):
        self.current_xval = 0
        for button in self.noise_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
    
    def filter_RadioBtnToggle(self, radioButton):
        self.current_xval = 0
        for button in self.filter_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
                
    def edgeNbinary_RadioBtnToggle(self, radioButton):
        self.current_xval = 0
        for button in self.edgeNbinary_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
                
    def hist_RadioBtnToggle(self, radioButton):
        self.current_xval = 0
        for button in self.hist_group_btn.buttons():
            if button is not radioButton:
                button.setChecked(False)
            
    def loadVideo(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("All(*.avi *.mp4 *.mkv *.mov *.jpeg *.jpg *.gif *.bmp *.png *.jfif)")

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
    
    def noiseSalt_sliderEvent(self, value):
        param = round(value*0.01, 2)
        self.noise_salt_value.setText(str(param))
        self.noise_salt_param = float(param)
        
    def noiseGaus_sliderEvent(self, value):
        self.noise_gaus_value.setText(str(value))
        self.noise_gaus_param = int(value)
    
    def lpfAvg_sliderEvent(self, value):
        value -= (1 - value % 2)
        self.lpf_avg_value.setText(f"{value}x{value}")
        self.lpf_avg_ksize = int(value)
    
    def lpfMed_sliderEvent(self, value):
        value -= (1 - value % 2)
        self.lpf_med_value.setText(f"{value}x{value}")
        self.lpf_med_ksize = int(value)
    
    def lpfGaus_ksliderEvent(self, value):
        value -= (1 - value % 2)
        self.lpf_gaus_kvalue.setText(f"{value}x{value}")
        self.lpf_gaus_ksize = int(value)
        
    def lpfGaus_ssliderEvent(self, value):
        self.lpf_gaus_svalue.setText(f"sig={value}")
        self.lpf_gaus_sig = int(value)
    
    def histStretch_sliderEvent(self, min, max):
        self.hist_stretch_minVal.setText(str(min))
        self.hist_stretch_maxVal.setText(str(max))
        self.hist_stretch_minThreshold = min
        self.hist_stretch_maxThreshold = max
        
    def edgeCanny_sliderEvent(self, min, max):
        self.edge_canny_minVal.setText(str(min))
        self.edge_canny_maxVal.setText(str(max))
        self.edge_canny_minThreshold = min
        self.edge_canny_maxThreshold = max
        self.current_xval = min
    
    def binaryGlobal_sliderEvent(self, min, max):
        self.binary_global_minVal.setText(str(min))
        self.binary_global_maxVal.setText(str(max))
        self.binary_global_minThreshold = min
        self.binary_global_maxThreshold = max
        self.current_xval = min
        
    def binaryAdaptive_bslider1Event(self, value):
        value -= (1 - value % 2)
        self.binary_adaptive_bvalue1.setText(f"size={value}")
        self.binary_adaptive_mean_b = int(value)
    
    def binaryAdaptive_cslider1Event(self, value):
        self.binary_adaptive_cvalue1.setText(f"C={value}")
        self.binary_adaptive_mean_c = int(value)
        
    def binaryAdaptive_bslider2Event(self, value):
        value -= (1 - value % 2)
        self.binary_adaptive_bvalue2.setText(f"size={value}")
        self.binary_adaptive_gaus_b = int(value)
    
    def binaryAdaptive_cslider2Event(self, value):
        self.binary_adaptive_cvalue2.setText(f"C={value}")
        self.binary_adaptive_gaus_c = int(value)
    
    def histogram_widgetToggle(self):
        # 위젯의 표시 여부를 토글
        current_state = self.histogram_widget.isVisible()
        self.histogram_widget.setVisible(not current_state)
        
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())