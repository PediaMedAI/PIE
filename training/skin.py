import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

def cohen_aug(img):
    # Follow https://arxiv.org/pdf/2002.02497.pdf, page 4
    # "Data augmentation was used to improve generalization.  According to best results inCohen et al. (2019) (and replicated by us)
    # each image was rotated up to 45 degrees, translatedup to 15% and scaled larger of smaller up to 10%"
    aug_ = A.Compose([
        A.ShiftScaleRotate(p=1.0, shift_limit=0.25, rotate_limit=45, scale_limit=0.1),
        A.HorizontalFlip(p=0.5),
    ])
    return aug_(image=img[0])["image"].reshape(img.shape)


class ISIC(Dataset):
    def __init__(self, path, aug=None, transform=None):
        super().__init__()
        self.data_path1 = os.path.join(path, "ham10000_images_part_1")
        self.data_path2 = os.path.join(path, "ham10000_images_part_2")
        self.csvpath = os.path.join(path, "HAM10000_metadata.csv")
        self.data_aug = aug
        self.MAXVAL = 255

        self.pathologies = {
            "bkl": "benign keratosis-like lesions", 
            "mel": "melanoma", 
            "nv": "melanocytic nevi", 
            "akiec": "actinic keratoses and intraepithelial carcinoma", 
            "bcc": "basal cell carcinoma", 
            "vasc": "vascular lesions",
            "df": "dermatofibroma" 
        }
        self.imgpath = os.path.dirname(path)
        self.transform = transform
        self.data_aug = aug
        self.csv = pd.read_csv(self.csvpath)

    def __len__(self):
        return len(self.csv["image_id"])

    def __getitem__(self, idx):
        imgid = self.csv['image_id'].iloc[idx]
        img_path = os.path.join(self.data_path1, str(imgid)+".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_path2, str(imgid)+".jpg")
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.data_aug is not None:
            img = self.data_aug(img)
        metadata = {'img_path': img_path, "target": self.csv['dx'].iloc[idx] ,"pathologies": self.pathologies[self.csv['dx'].iloc[idx]]}
        return img, None, metadata