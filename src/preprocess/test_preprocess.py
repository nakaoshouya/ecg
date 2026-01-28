# src/afterecg/preprocess/preprocess.py
import cv2
from pathlib import Path

def preprocess_test_dir(in_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in Path(in_dir).glob("*.png"):
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1024, 768))
        cv2.imwrite(str(out_dir / p.name), img)
