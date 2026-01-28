# scripts/infer.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from afterecg.pipeline.infer_loop import run_infer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=str, required=True)     # 前処理済み
    ap.add_argument("--ckpt-dir", type=str, required=True)     # best.ptやepoch_*.ptがある
    ap.add_argument("--out-dir", type=str, required=True)      # 予測保存先
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", choices=["auto","off"], default="auto")
    ap.add_argument("--channels-last", action="store_true")
    args = ap.parse_args()

    run_infer(vars(args))

if __name__ == "__main__":
    main()
