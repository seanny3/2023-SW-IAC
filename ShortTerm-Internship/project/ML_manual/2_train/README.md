# YOLOv8로 데이터 학습하기
학습은 Goolge Colab GPU 환경에서 진행한다.

<img src="https://img.shields.io/badge/Colab-F9AB00?style=flat&logo=GoogleColab&logoColor=white"/>

### 라이브러리 설치
```
!pip install ultralytics
```
```
import ultralytics
ultralytics.checks()
```

### 커스텀 데이터셋 업로드
라벨링 과정에서 생성된 ```output.zip```을 실행 환경에 압축 해제한다.
```
!mkdir my/dataset
%cd my/dataset
!unzip -qq /content/drive/MyDrive/output.zip
```

### 모델 불러오기
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
# model = YOLO('my/dataset/runs/detect/train/weights/last.pt')

print(type(model.names), len(model.names))
print(model.names)
```

### 학습
```python
model.train(data="my/dataset/data.yaml", batch=32, epochs=30)
# model.train(resume=True)
```

### 테스트
```python
results = model.predict(source='my/dataset/test/images', save=True)
# source=[directory|image|video]
```

### 메모리 정리
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```