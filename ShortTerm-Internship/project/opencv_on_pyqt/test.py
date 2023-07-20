import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # 메인 윈도우 설정
        self.setWindowTitle('PyQt5 Matplotlib Example')
        self.setGeometry(100, 100, 800, 600)

        # 그래프를 표시할 위젯 생성
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        central_widget.setStyleSheet("background-color: gray;")

        # Matplotlib Figure 생성
        self.figure = plt.figure()
        self.figure.patch.set_facecolor("None")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # 배경색 설정 (예: 회색 배경색)
        self.canvas.setStyleSheet("background-color: gray;")

        # 그래프를 다시 그리도록 업데이트
        self.plot_graph()

    def plot_graph(self):
        # 기존 그래프를 지우기 위해 clear() 메서드 호출
        self.figure.clear()
        plt.rcParams.update({
                        "figure.facecolor":  (0.0, 0.0, 0.0, 0),
                        "axes.facecolor":    (0.0, 0.0, 0.0, 0),
                        "savefig.facecolor": (0.0, 0.0, 1.0, 0.2),
                        "text.color" : "red",
                        "axes.labelcolor" : "red",
                        "axes.edgecolor" : (0,0,0,0)
                    })
        # 예시로 간단한 선 그래프를 그립니다.
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 1, 6, 3]
        plt.plot(x, y, color='red', linewidth=3)

        # 그래프 속성 설정 (선택 사항)
        plt.title('Sample Plot')
        plt.xlabel('X-axis')
        plt.tick_params(axis='x', colors='red')
        plt.grid()

        # y 축 눈금을 숨깁니다.
        plt.yticks([])

        # 그래프를 다시 그리도록 업데이트
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
