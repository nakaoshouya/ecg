import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

# もしまだ import していなければ
from src.preprocess import preprocess_test_image
from src.dataset import SegDataset, TestImageDataset
from src.model import UNet
from src.utils import set_seed , train_one_epoch, evaluate, save_predictions, focal_loss_multi_class, dice_loss_multi_class, compute_miou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    #ここがエポック数だよ！！！
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--save-dir', type=str, default='./outputs')
    parser.add_argument(
        '--amp',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Automatic Mixed Precision setting'
    )
    parser.add_argument(
        '--accum-steps',
        type=int,
        default=1,
        help='Gradient accumulation steps'
    )
    args = parser.parse_args()

    set_seed(42)
    data_root = Path(args.data_root)
    images_dir = data_root / "image"
    masks_dir = data_root / "mask"

    dataset = SegDataset(images_dir, masks_dir, num_classes=args.num_classes)
    val_len = int(len(dataset) * 0.1)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # デバイス設定 & AMP
    cuda_ok = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_ok else 'cpu')
    amp_device = 'cuda' if (args.amp == 'auto' and cuda_ok) else 'cpu'
    scaler = torch.amp.GradScaler() if amp_device != 'cpu' else None

    model = UNet(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            num_classes=args.num_classes,
            amp_device=amp_device,
            accum_steps=args.accum_steps
        )
        val_miou = evaluate(model, val_loader, device, args.num_classes)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_mIoU={val_miou:.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(model, val_loader, device, save_dir, args.num_classes)


if __name__ == "__main__":
    main()
