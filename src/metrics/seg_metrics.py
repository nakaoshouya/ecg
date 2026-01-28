# src/afterecg/metrics/seg_metrics.py
import torch
import numpy as np

def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    preds: (B,H,W) argmax済み
    targets: (B,H,W)
    """
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(1, num_classes):  # 背景0は無視
        p = preds == cls
        t = targets == cls
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        ious.append(1.0 if union == 0 else inter / union)

    return float(sum(ious) / len(ious))
