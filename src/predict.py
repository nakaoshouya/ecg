import torch
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image

from src.model import UNet
from src.dataset import TestImageDataset

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader

def predict(file_path, model_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    batch_size = 4

    images_dir = file_path
    save_dir = Path(__file__).resolve().parent.parent / "outputs/test_pred"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TestImageDataset(images_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = UNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    saved_paths = []

    with torch.no_grad():
        for i, images in enumerate(loader):

            if isinstance(images, (list, tuple)):
                images = images[0]

            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for j in range(preds.shape[0]):
                pred_img = preds[j].cpu().numpy().astype('uint8') * 255

                img_name = dataset.image_paths[i * batch_size + j].name
                save_path = save_dir / f"pred_{img_name}"

                Image.fromarray(pred_img).save(save_path)
                print(f"保存しました: {save_path}")

                saved_paths.append(str(save_path))

    #パスのリストを返す
    return saved_paths