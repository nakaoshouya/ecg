# src/afterecg/pipeline/train_loop.py
import torch
from torch.utils.data import DataLoader
from afterecg.models.unet import UNet
from afterecg.metrics.seg_metrics import compute_miou

def run_train(cfg):
    dataset = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(**model_cfg).to(device)

    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"])
    optim = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        for imgs, masks, _ in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optim.step()
            optim.zero_grad()
