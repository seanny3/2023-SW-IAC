# ✏️ [labelImg](https://github.com/HumanSignal/labelImg)

### 🌐 Evironments
```python
# python==3.8.17
# ultralytics==8.0.149
# split-folders==0.5.1

conda create -n autolabel python=3.8
conda activate autolabel
pip install ultralytics
pip install split-folders
```

### 🔗 이미지 업로드 &nbsp; ```./data/img```
- 라벨링되어 있지 않은 이미지 혹은 라벨링되어 있는 이미지 업로드
- 모든 이미지는 *.jpg 으로 전환하여 업로드해야 한다.

### 🛠️ classes.txt 수정 &nbsp; ```./classes.txt, ./data/img/classes.txt```
```bash
# example)
0
1
2
...
se
jong
E
```

### 🏃🏻‍♂️ labelImg.bat 실행 &nbsp; ```./labelImg.bat```
```python
# labelImg.bat
# labelImg.exe [이미지 경로] [클래스 경로]
labelImg.exe data/img classes.txt
```

&nbsp;
# 🚀 Auto labeling
- 라벨링 작업을 수월하게 하기 위해서 라벨링 자동화 프로그램을 사용할 수 있다.
- 여기서 사용되는 프로그램은 학습된 가중치 파일을 사용하기 때문에 적어도 1000개 가량의 데이터를 수작업으로 라벨링하는 과정을 거쳐야 의미있는 작업이 수행된다.
- 영어가 아닌 파일명은 오류가 발생할 수 있기 때문에 새로운 이름으로 바꾸는 로직이 추가되어 있다. 기존 파일명을 되찾을 수 없기 때문에 원본은 따로 저장한 후 진행하길 바란다.
- labelImg 프로그램에 필요한 classes.txt를 model.names 정보를 참고하여 자동 생성해준다.

### 🛠️ ```config.ini``` 수정
```ini
# config.ini

[autolabel]
weights = data/best.pt  # 가중치 파일 저장 경로
target = data/img       # 대상 이미지 경로
output = data/img       # 결과 저장 경로

[split-folders]
target = data/img
ratio = .7,.2,.1        # train/valid/test 비율
```

### 🏃🏻‍♂️ Run


```python
python autolabel.py
```

&nbsp;
# 🧩 [split-folders](https://pypi.org/project/split-folders/)
- 모든 라벨링 작업이 끝났다면 YOLO에서 정해준 폴더 구조에 맞게 분배해야한다.
- 실행 결과로 output.zip 파일이 생성되면 data.yaml 이 자동 생성되어 추출된다.
- data.yaml 내용은 classes.txt의 클래스 정보를 기반으로 자동 생성되어 압축된다.

### 🏃🏻‍♂️ Run
```
python split-folders.py
```
