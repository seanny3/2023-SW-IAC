import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageViewerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        self.step = []

    def init_ui(self):
        self.setWindowTitle("Image Viewer")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)  # Track mouse movement even without clicking
        self.image_label.setStyleSheet("border: 1px solid black;")

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.bounding_box = None
        self.dragging = False

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)

        self.input_text = QLineEdit()

        layout = QVBoxLayout()
        layout.addWidget(self.load_button, 1)
        layout.addWidget(self.image_label, 3)
        layout.addWidget(self.input_text, 1)
        layout.addWidget(self.result_label, 1)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, '이미지 불러오기', '', 'Images (*.jpeg *.jpg *.bmp *.png *.jfif)', options=options)
        
        if file_name:
            self.image_pixmap = QPixmap(file_name)
            # self.image_label.setPixmap(self.image_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            desired_width = 2100
            scaled_pixmap = self.image_pixmap.scaledToWidth(desired_width, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.mousePressEvent = self.getMousePos

    def getMousePos(self, event):
        y = event.pos().y()
        self.step.append(y)
        print(f"{len(self.step)}: y={y}")
        
        if len(self.step) == 3:
            car_height = abs(self.step[0] - self.step[1])
            driver_height = abs(self.step[1] - self.step[2])
            result = (int(self.input_text.text())*driver_height)/car_height
            self.result_label.setText(f"result:{result}")
            print(f"result:{result}")
            self.step = []
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewerApp()
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())
