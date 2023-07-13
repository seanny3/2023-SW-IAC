# app.py
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
import torch
from PIL import Image
import os
import numpy as np
from time import time
from ultralytics import YOLO
from futils import add_salt_and_pepper_noise, add_gaussian_noise, bg_removal

app = Flask(__name__)
global cv_noise, cv_filter, cv_edge, cv_cvt, cv_tf
cv_noise = ''
cv_filter = ''
cv_edge = ''
cv_cvt = ''
cv_tf = []

global angle
angle = 90

# init_file_names = os.listdir('./resources/video')
# for file_name in init_file_names:
#     os.remove('./resources/video/' + file_name)
    
# init_model_names = os.listdir('./resources/model')
# for model_name in init_model_names:
#     os.remove('./resources/model/' + model_name)
    
@app.route('/')
def video_show():
    return render_template('index.html')

@app.route('/upload/video', methods=['POST'])
def upload_video():
    file_names = os.listdir('./resources/video')
    
    for file_name in file_names:
        os.remove('./resources/video/' + file_name)
    
    file = request.files['file']
    ext = file.filename.split(".")[-1]
    file_path = './resources/video/uploaded.' + ext
    file.save(file_path)
    return redirect("/")

@app.route('/upload/model', methods=['POST'])
def upload_model():
    file_names = os.listdir('./resources/model')
    
    for file_name in file_names:
        os.remove('./resources/model/' + file_name)
    
    file = request.files['file']
    ext = file.filename.split(".")[-1]
    file_path = './resources/model/uploaded.' + ext
    file.save(file_path)
    return redirect("/")

@app.route('/video')
def video():
    file_name = os.listdir('./resources/video')
    if file_name:
        return Response(gen_frames('./resources/video/' + file_name[0]), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("none")

@app.route('/cam')
def cam():
    return Response(gen_frames(str(0)), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(file):
    global cv_noise, cv_filter, cv_edge, cv_cvt, cv_tf
    cv_noise = ''
    cv_filter = ''
    cv_edge = ''
    cv_cvt = ''
    cv_tf = []
    global angle
    angle = 90
    
    model_name = os.listdir("./resources/model/")
    if model_name:
        # model = torch.hub.load('../../../../yolov7', 'custom',  './resources/model/uploaded.pt', source='local')
        model = torch.hub.load('../../../../yolov5', 'custom',  '../../../../yolov5/yolov5s.pt', source='local')
        # model = YOLO('./resources/model/uploaded.pt')
    
    ext = file.split(".")[-1]
    
    if ext in ["mp4", "avi", "0"]:
        cap = cv2.VideoCapture(0 if ext == "0" else file)
        
        while True:
            start_time = time()
            
            ret, frame = cap.read()
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            else:
                # 이미지 변환
                if 'rotate90' in cv_tf:
                    height, width = frame.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1.0)
                    output_width = int(height * (width / height))
                    frame = cv2.warpAffine(frame, rotation_matrix, (output_width, height))
                if 'leftright' in cv_tf:
                    frame = cv2.flip(frame, 1)
                if 'updown' in cv_tf:
                    frame = cv2.flip(frame, 0)
                
                # 노이즈 추가 기능
                if cv_noise == 'salt':
                    frame = add_salt_and_pepper_noise(frame, 0.05)
                elif cv_noise == 'gaus':
                    frame = add_gaussian_noise(frame, 20)
                
                # 그레이스케일 변환
                if cv_cvt == 'gray':
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 필터링 기능
                if cv_filter == 'avg':
                    frame = cv2.blur(frame, (3,3))
                elif cv_filter == 'med':
                    frame = cv2.medianBlur(frame, 3)
                elif cv_filter == 'gaus':
                    frame = cv2.GaussianBlur(frame, (5,5), 1)
                elif cv_filter == 'sharp1':
                    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
                    frame = cv2.filter2D(frame, -1, kernel)
                elif cv_filter == 'sharp2':
                    kernel = np.array([[1,1,1],[1,-7,1],[1,1,1]])
                    frame = cv2.filter2D(frame, -1, kernel)
                elif cv_filter == 'sharp3':
                    kernel = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
                    frame = cv2.filter2D(frame, -1, kernel)
                
                # 윤곽선 검출 기능 - 그레이스케일 필수
                if cv_edge:
                    if len(frame.shape) == 3:  # 이미지가 컬러 이미지인 경우
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                if cv_edge == 'robert':          
                    # 로버트 크로스 필터
                    gx = np.array([[-1, 0], [0, 1]], dtype=int)
                    gy = np.array([[0, -1], [1, 0]], dtype=int)

                    # 로버트 크로스 컨벌루션
                    x = cv2.filter2D(frame, -1, gx, delta=0)
                    y = cv2.filter2D(frame, -1, gy, delta=0)
                
                    frame = x+y
                    
                elif cv_edge == 'sobel':
                    # Sobel operator
                    x = cv2.Sobel(frame, -1, 1, 0, ksize=3, delta=0)
                    y = cv2.Sobel(frame, -1, 0, 1, ksize=3, delta=0)
                    frame = x+y
                    
                elif cv_edge == 'prewitt':
                    # 프르윗 필터
                    gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
                    gy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)

                    # 프르윗 필터 컨벌루션
                    x = cv2.filter2D(frame, -1, gx, delta=0)
                    y = cv2.filter2D(frame, -1, gy, delta=0)
                
                    frame = x+y
                    
                elif cv_edge == 'canny':
                    frame = cv2.Canny(frame,100,200)
                
                elif cv_edge == 'otsucanny':
                    # 2. Global OTSU-Canny
                    otsu_th, otsu_binary = cv2.threshold(frame, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

                    # 임계값 수동 설정의 번거로움 극복, 불필요한 윤곽선
                    frame = cv2.Canny(frame, otsu_th, otsu_th)
                    
                elif cv_edge == 'laplacian':
                    frame = cv2.Laplacian(frame, -1)
                
                if model_name:
                    # OpenCV 이미지를 PIL 이미지로 변환
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # 객체 인식
                    results = model(pil_image)
                    # frame = results[0].plot()     # YOLOv8
                    # bounding box 처리
                    boxes = results.xyxy[0]  # (x1, y1, x2, y2) 형식의 bounding box 좌표
                    confidences = results.xyxy[0][:, 4]  # bounding box의 신뢰도

                    for box, confidence in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box[:4])
                        label = f'{results.names[int(box[5])]} {confidence:.2f}'  # 객체 클래스와 신뢰도

                        # bounding box 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)               
                
                end_time = time()
                fps = 1/np.round(end_time - start_time, 3)
                cv2.putText(frame, f"FPS: {fps:.3f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    elif ext in ["jpg", "png", "jpeg", "gif"]:
        frame = cv2.imread(file)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    
@app.route('/cv', methods=['POST'])
def cv():
    global cv_noise, cv_filter, cv_edge, cv_cvt, cv_tf, angle
    
    payload = request.get_json()["func"].split("-")
        
    if payload[1] == 'init':
        if payload[0] == 'noise':
            cv_noise = ''
        elif payload[0] == 'filter':
            cv_filter = ''
        elif payload[0] == 'edge':
            cv_edge = ''
        elif payload[0] == 'cvt':
            cv_cvt = ''
    
    else:
        if payload[0] == 'noise':
            cv_noise = payload[1]
        elif payload[0] == 'filter':
            cv_filter = payload[1]
        elif payload[0] == 'edge':
            cv_edge = payload[1]
        elif payload[0] == 'cvt':
            cv_cvt = payload[1]
        elif payload[0] == 'tf':
            if payload[1] in cv_tf:
                if payload[1] == 'rotate90':
                    angle += 90
                    if angle == 360:
                        angle = 0
                else:
                    cv_tf.remove(payload[1])
            else:
                cv_tf.append(payload[1])
                
                    
        
    return jsonify(success=True)
    


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")