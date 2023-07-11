# app.py
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import torch
import os

app = Flask(__name__)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/')
def video_show():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file_names = os.listdir('./resources')
    
    for file_name in file_names:
        os.remove('./resources/' + file_name)
    
    file = request.files['file']
    ext = file.filename.split(".")[-1]
    file_path = './resources/uploaded.' + ext
    file.save(file_path)
    return redirect("/")

@app.route('/video')
def video():
    file_name = os.listdir('./resources')
    if file_name:
        return Response(gen_frames('./resources/' + file_name[0]), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("none")


@app.route('/cam')
def cam():
    return Response(gen_frames(str(0)), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(file):
    ext = file.split(".")[-1]
    
    if ext in ["mp4", "avi", "0"]:
        cap = cv2.VideoCapture(0 if ext == "0" else file)
        
        while True:
            _, frame = cap.read()
            if not _:
                break
            else:
                # results = model(frame)
                # annotated_frame = results.render()
                
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
    

if __name__ == "__main__":
    app.run(debug=True)