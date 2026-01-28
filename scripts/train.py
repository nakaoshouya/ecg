# scripts/train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from afterecg.pipeline.train_loop import run_train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--masks-dir", type=str, required=True)
    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", choices=["auto","off"], default="auto")
    ap.add_argument("--channels-last", action="store_true")
    args = ap.parse_args()

    run_train(vars(args))

if __name__ == "__main__":
    main()
