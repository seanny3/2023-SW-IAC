# YOLOv8 with [openVINO](https://docs.openvino.ai/2023.0/notebooks/230-yolov8-optimization-with-output.html)
- openVINO는 intel에서 주도적으로 진행 중인 프로젝트이다.
- 다양한 딥러닝 프레임워크의 모델들을 openVINO 모델로 변환하여 intel device에 최적화된 inference를 할 수 있도록 해준다.
- Nvidia 제품을 사용하지 않는 PC 환경에서 openVINO를 사용하면 latency 성능이 좋아지는 이점이 있다.
- 이 프로젝트는 실시간 영상에 대한 object detection만을 지원한다.
- 이 프로젝트는 YOLOv8로 학습된 *.pt, *.onnx 만을 지원한다.
- [호환 디바이스 확인하기](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)

### 🚀 Speed
- YOLOv8m 모델 구조를 사용하여 학습한 커스텀 가중치 파일
- OpenVINO-CPU를 사용할 때, 사용하지 않았을 때보다 24.7% 정도 속도가 향상되었다.
- OpenVINO-GPU를 사용할 때, 사용하지 않았을 때보다 66.2% 정도 속도가 향상되었다.
- OpenVINO-GPU를 사용할 때, OpenVINO-CPU 보다 55.7% 정도 속도가 향상되었다.

<div align="center">

||Original CPU|OpenVINO CPU|OpenVINO GPU|
|:---:|:---:|:---:|:---:|
|**평균(ms)**|582ms|438ms|196ms|

</div>
&nbsp;

### 🛠️ YOLOv8 → ONNX → openVINO

```bash
pip install -q 'openvino-dev>=2023.0.0' 'nncf>=2.5.0'
pip install -q 'ultralytics==8.0.43' onnx
```
```python
from openvino.runtime import Core
```

- weights_name_ex: yolov8n.pt
- weights_name: yolov8n
- openvino_path: openvino weights dir (.xml)
```python
save_dir = Path('./weights')
pt_model = YOLO(save_dir / f'{weights_name_ex}')
openvino_path = save_dir / f'{weights_name}_openvino_model/{weights_name}.xml'
```

- 다행히도 ultralytics 라이브러리에서 openVINO 변환 API를 제공해주고 있다.
```python
if not openvino_path.exists():
    pt_model.export(format="openvino", dynamic=True, half=False)
```

### 🔎 Object detection
```python
core = Core()
ov_model = core.read_model(openvino_path)

if device != "CPU":
    ov_model.reshape({0: [1, 3, 640, 640]})
    
model = core.compile_model(ov_model, device)
```

## 🏃🏻‍♂️ Run
```bash
python main.py --source my/video.mp4 --weights my/weights.pt --device gpu
```

- source = ***0*** &nbsp; is webcam
- source = ***my/video.mp4*** &nbsp; is video file
- weights = ***my/weights.pt***
- device = ***cpu*** or ***gpu***
