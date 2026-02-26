import torch
import random
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image

def dice_loss_multi_class(pred, target, epsilon=1e-6):
    """
    pred: (B, C, H, W)  -- softmax確率
    target: (B, H, W)    -- クラスラベル
    """
    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes).permute(0,3,1,2).float()  # (B,C,H,W)

    dice = 0
    for c in range(1, num_classes):
        p = pred[:, c]
        t = target_one_hot[:, c]
        intersection = torch.sum(p * t)
        union = torch.sum(p) + torch.sum(t)
        dice += 1 - (2 * intersection + epsilon) / (union + epsilon)

    return dice / (num_classes - 1)

def focal_loss_multi_class(logits, target, gamma=2.0, alpha=None):
    """
    logits: (B, C, H, W)  -- 未正規化スコア
    target: (B, H, W)      -- クラスラベル
    alpha: (C,) tensor or None -- クラス重み
    """
    ce_loss = F.cross_entropy(logits, target, weight=alpha, reduction='none')  # (B,H,W)
    pt = torch.exp(-ce_loss)  # 正解クラスの確率
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

def compute_miou(preds, targets, num_classes):
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(1, num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(1.0)  # そのクラスが存在しない場合は IoU=1
        else:
            ious.append(intersection / union)
    return sum(ious) / (num_classes - 1)

@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_bg=False, amp_device="cpu", scaler=None):
    model.eval()
    total_miou = 0.0
    total = 0
    ig = 0 if ignore_bg else None
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, enabled=(scaler is not None)):
            logits = model(imgs)
        preds = logits.argmax(dim=1)
        miou = compute_miou(preds, masks, num_classes=num_classes)
        total_miou += miou * imgs.size(0)
        total += imgs.size(0)
    return total_miou / max(total, 1)


@torch.no_grad()
def save_predictions(model, loader, device, out_dir, num_classes, max_batches=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batches = 0
    for imgs, masks, stems in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1, keepdim=True).float()
        for i in range(imgs.size(0)):
            save_image((imgs[i] * 0.5 + 0.5).clamp(0,1), out_dir / f"{stems[i]}_img.png")
            pred_vis = (preds[i] / max(num_classes - 1, 1)).clamp(0,1)
            mask_vis = (masks[i].unsqueeze(0).float() / max(num_classes - 1, 1)).clamp(0,1)
            save_image(pred_vis, out_dir / f"{stems[i]}_pred.png")
            save_image(mask_vis, out_dir / f"{stems[i]}_gt.png")
        batches += 1
        if batches >= max_batches:
            break

def set_seed(seed: int, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def train_one_epoch(model, loader, optimizer, scaler, device, num_classes, amp_device,
                    accum_steps=1, use_dice=True, use_focal=True, focal_gamma=2.0, focal_alpha=None):
    model.train()
    total_loss = 0.0
    step = 0
    total_miou = 0.0
    total_samples = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=amp_device, enabled=(scaler is not None)):
            logits = model(imgs)

            # softmax 確率
            probs = F.softmax(logits, dim=1)

            loss = 0.0
            if use_focal:
                loss += focal_loss_multi_class(logits, masks, gamma=focal_gamma, alpha=focal_alpha)
            if use_dice:
                loss += dice_loss_multi_class(probs, masks)

            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size * accum_steps

        # train_mIoU を計算（表示用）
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_miou = compute_miou(preds, masks, num_classes)
            total_miou += batch_miou * batch_size
            total_samples += batch_size

        step += 1

    avg_loss = total_loss / total_samples
    avg_miou = total_miou / total_samples
    print(f"Train mIoU: {avg_miou:.4f}")

    return avg_loss