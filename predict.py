# predict.py
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image

from src.model import UNet
from src.dataset import TestImageDataset

# --- 設定 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
batch_size = 4

# 保存済みモデルパス
model_path = Path("trained_model.pth")  # プロジェクト直下

# 予測対象画像のディレクトリ
images_dir = Path(r"data\test\image")

# 出力ディレクトリ
save_dir = Path(r"outputs\test_pred")
save_dir.mkdir(parents=True, exist_ok=True)

# --- データセット & ローダー ---
dataset = TestImageDataset(images_dir)  # mask は不要
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- モデル初期化 & 読み込み ---
model = UNet(num_classes=num_classes).to(device)
# 安全にモデルの重みだけをロード
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# --- 予測 & 保存 ---
with torch.no_grad():
    for i, images in enumerate(loader):
        # Dataset が (image,) のタプルを返す場合に対応
        if isinstance(images, (list, tuple)):
            images = images[0]

        images = images.to(device)
        outputs = model(images)  # [B, C, H, W]
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]

        for j in range(preds.shape[0]):
            pred_img = preds[j].cpu().numpy().astype('uint8') * 255  # 0/1 → 0/255
            # 元画像名取得
            img_name = dataset.image_paths[i * batch_size + j].name
            save_path = save_dir / f"pred_{img_name}"  # 上書き防止
            Image.fromarray(pred_img).save(save_path)
            print(f"保存しました: {save_path}")