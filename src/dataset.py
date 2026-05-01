import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np


class SegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, num_classes=2, overfit_one=False, repeat_factor=1):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_classes = num_classes
        self.image_paths = sorted(self.images_dir.iterdir())
        self.mask_paths = sorted(self.masks_dir.iterdir())
        self.overfit_one = overfit_one
        self.repeat_factor = repeat_factor

    def __len__(self):
        if self.overfit_one:
            return self.repeat_factor
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.overfit_one:
            idx = 0

        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 画像読み込み
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        # マスク読み込み＆0/1 に正規化
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = torch.tensor(mask // 255, dtype=torch.long) 
        stem = img_path.stem

        return img, mask, stem


class TestImageDataset(Dataset):
    def __init__(self, test_dir, height=768, width=1024, normalize=True):
        self.test_dir = Path(test_dir)
        self.height = height
        self.width = width
        self.normalize = normalize
        self.image_paths = sorted(self.test_dir.iterdir())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        if self.normalize:
            img /= 255.0
        return img, str(path)
