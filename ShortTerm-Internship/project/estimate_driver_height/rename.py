import glob
import os
import argparse
from pathlib import Path

def main(opt):
    target = Path(opt.target)
    cnt = 0
    for img in glob.glob(f'{target}/*.jpg'):
        cnt += 1
        file = Path(img)
        os.rename(file, f"{target}/{opt.rename}{cnt}{file.suffix}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='./target', help='target img dir')
    parser.add_argument('--rename', type=str, required=True, help='name to change')
    
    main(parser.parse_args())