# scripts/preprocess_test.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from afterecg.preprocess.test_preprocess import preprocess_test_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    preprocess_test_dir(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()
