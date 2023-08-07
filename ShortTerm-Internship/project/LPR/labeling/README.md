# ✏️ Labeling
라벨링 툴은 [labelImg](https://github.com/HumanSignal/labelImg) 프로그램을 사용한다.


### 1. 이미지 업로드 &nbsp; ```./data/img```
- 라벨링되어 있지 않은 이미지 혹은 라벨링되어 있는 이미지 업로드

### 2. classes.txt 수정 &nbsp; ```./classes.txt```
```bash
0
1
2
...
se
jong
E
```

### 3. labelImg.bat 실행 &nbsp; ```./laelImg.bat```
```python
# labelImg.bat
# labelImg.exe [이미지 경로] [클래스 경로]
labelImg.exe data/img classes.txt
```

&nbsp;
## 🚀 Auto labeling
라벨링 작업을 수월하게 하기 위해서 라벨링 자동화 프로그램을 사용할 수 있다.

여기서 사용되는 프로그램은 학습된 가중치 파일을 사용하기 때문에 적어도 1000개 가량의 데이터를 수작업으로 라벨링하는 과정을 거쳐야 한다.

### 🏃🏻‍♂️ Run
target 폴더와 output 폴더는 ```./data/img``` 로 기본 값으로 설정되어 있다.
```python
python autolabel.py --weights my/weights.pt --target my/imgs --output my/labels
```
- --target: 타겟 이미지 경로
- --output: 생성된 라벨 데이터 저장 경로

### 🛠️ Rename
영어가 아닌 파일명은 오류가 발생할 수 있기 때문에 새로운 이름으로 바꿔서 이를 해결할 수 있다.

이 프로그램을 실행하면 기존 파일명을 되찾을 수 없기 때문에 원본은 따로 저장한 후 진행하길 바란다.
```python
python rename.py --target my/imgs --rename [바꿀 이름]
```

&nbsp;
## 🧩 [split-folders](https://pypi.org/project/split-folders/)
모든 라벨링 작업이 끝났다면 YOLO에서 정해준 폴더 구조에 맞게 분배해야한다.
```bash
pip install split-folders
```

### 🏃🏻‍♂️ Run
실행 결과로 output.zip 파일이 생성되면 data.yaml 이 자동 생성되어 추출된다.

data.yaml 내용은 classes.txt의 클래스 정보를 기반으로 자동 생성된다.
```
python split-folders.py
```
