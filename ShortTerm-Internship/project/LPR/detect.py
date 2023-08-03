import argparse
import cv2
import yaml
import numpy as np
from LPRNet import LPRNet
from time import time

class LPRStream:
    def __init__(self, opt):
        self.lpr_net = LPRNet(opt)
        self.cap = cv2.VideoCapture(int(opt.source) if opt.source == "0" else opt.source)
       
        with open(opt.data, 'r') as file:
            yaml_content = yaml.safe_load(file)
        self.classes = yaml_content["names"]
        
    def __del__(self):
        self.cap.release()

    def run(self):
        if not self.cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            self.stop()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            start_time = time()
            
            # object detection
            frame = cv2.resize(frame, (640, 480))
            
            input = frame.copy()
            output = self.lpr_net.detect(input)[0]
            frame = self.lpr_net.draw_results(output, input)
            
            end_time = time()
            
            # detected results
            det = np.array(output['det'])
            
            if len(det) > 0:      
                sorted_det = det[det[:, 0].argsort()]
                # print(sorted_det)
                sorted_clsIdx = [int(row[-1]) for row in sorted_det]
                nums = ''.join(self.classes[idx] for idx in sorted_clsIdx)
                
                cv2.putText(frame, nums, (10,75), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), thickness=2)
            
            # inference speed, fps
            speed = end_time - start_time    # speed (sec)
            fps = int(1./speed)
            # print(f"inference: {speed*1000:.2f}ms, fps: {fps}")
            
            cv2.putText(frame, f'FPS: {fps}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), thickness=2)
            cv2.imshow("LPR", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=0, help='0:webcam or others:video path')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='.pt or .onnx')
    parser.add_argument('--data', type=str, default='data.yaml', help='data.yaml')
    parser.add_argument('--device', type=str, default='cpu', help='none or cpu or gpu')
    
    lpr = LPRStream(parser.parse_args())
    
    lpr.run()