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


class ChexPert(Dataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
    Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, 
    Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, 
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, 
    Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. https://arxiv.org/abs/1901.07031
    
    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    # def __init__(self, imgpath, csvpath, views=["PA"], transform=None, data_aug=None,
    # flat_dir=True, seed=0, unique_patients=True):
    def __init__(self, path, split="train", aug=None, transform=None, views=["AP", "PA"], unique_patients=False):
        super().__init__()
        if split == "train":
            csvpath = os.path.join(path, 'train.csv')
        elif split == "valid":
            csvpath = os.path.join(path, 'valid.csv')
        else:
            raise ValueError(csvpath)
        self.data_aug = aug
        # np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255

        self.pathologies = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
            "Support Devices"
        ]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = os.path.dirname(path)
        self.transform = transform
        self.data_aug = aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views

        # To list
        if type(self.views) is not list:
            views = [views]
            self.views = views

        self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv[
            "AP/PA"]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"})  # Rename Lateral with L
        self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat='(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        print(self.pathologies)

        ########## add consistent csv values

        # offset_day_int

        # patientid
        if 'train' in csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'valid' in csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplemented

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")
        self.csv["patientid"] = patientid

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(
            len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['Path'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        target = self.labels[idx]
        pathologies = []
        for i in range(len(target)):
            if target[i] == 1:
                pathologies.append(self.pathologies[i])
        if len(pathologies) > 0:
            pathologies = ",".join(pathologies)
        else:
            pathologies = ""
        target = torch.from_numpy(target).float()
        metadata = {'img_path': img_path, "pathologies": pathologies}
        return img, target, metadata