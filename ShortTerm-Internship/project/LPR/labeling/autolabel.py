import argparse
from glob import glob
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

def main(opt):
    pt_model = YOLO(opt.weights)
    img_list = glob(f"{opt.target}/*.jpg")
    
    for img in tqdm(img_list, ncols = 80, desc="labeling"):
        results = pt_model.predict(source=img, verbose=False)
        for r in results:
            table = []
            boxes = r.boxes
            for box in boxes:
                row = []
                row.append(int(box.cls))
                for xywh in box.xywhn.squeeze().tolist():
                    row.append(xywh)
                table.append(row)
                
            file = Path(r.path).with_suffix('.txt')
            output = Path(opt.output) / file.name
            with open(output, 'w') as f:
                # print(f"target: {file}")
                for row in table:
                    row_data = ' '.join(map(str, row))
                    f.write(row_data + '\n')
                    # print(row_data)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='only yolov8 weights')
    parser.add_argument('--target', type=str, default='./data/img', help='target img dir')
    parser.add_argument('--output', type=str, default='./data/img', help='output dir')
    
    main(parser.parse_args())