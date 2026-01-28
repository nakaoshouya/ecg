# src/afterecg/pipeline/infer_loop.py
import torch
from pathlib import Path
from torchvision.utils import save_image
from afterecg.models.unet import UNet

def run_infer(model, loader, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for imgs, names in loader:
            logits = model(imgs)
            preds = logits.argmax(1, keepdim=True).float()
            for i, name in enumerate(names):
                save_image(preds[i], out_dir / name)
