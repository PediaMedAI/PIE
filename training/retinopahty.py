import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Retinopathy(Dataset):
    def __init__(self, path, split="train", transform=None):
        super().__init__()
        if split == "train":
            csvpath = os.path.join(path, 'trainLabels.csv')
            self.csvpath = csvpath
            self.imgpath = os.path.join(path, 'train')
        
        self.transform = transform
        self.csv = pd.read_csv(self.csvpath)
        self.pathologies = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    def __len__(self):
        return len(self.csv.index)

    def __getitem__(self, idx):
        imgid = self.csv['image'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid+'.jpeg')
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = self.csv['level'].iloc[idx]
        if target == 0:
            metadata = {'img_path': img_path, "pathologies": "healthy"}
        else:
            metadata = {'img_path': img_path, "pathologies": "diabetic"}
        return img, target, metadata