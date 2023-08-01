import argparse
import cv2
from LPRNet import LPRNet
from time import time

class LPRStream:
    def __init__(self, source, weights, device):
        self.lpr_net = LPRNet(weights, device)
        self.cap = cv2.VideoCapture(source)

    def __del__(self):
        self.cap.release()

    def run(self):
        fps = 0
        start_time = time()
            
        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            self.stop()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            input = frame.copy()
            output = self.lpr_net.detect(input)[0]
            frame = self.lpr_net.draw_results(output, input)
            
            current_time = time()
            elapsed_time = current_time - start_time
            print(f"{elapsed_time*1000:.2f}ms")
            
            cv2.imshow("LPR", frame)

            start_time = current_time
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='0:webcam or others:video path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--device', type=str, default='cpu', help='none or cpu or gpu')
    opt = parser.parse_args()
    
    lpr = LPRStream(
        source=opt.source,            # webcam is 0
        weights=opt.weights,          # supports only PyTorch. (*.pt)
        device=opt.device.upper()     # using openVINO
    )
    lpr.run()