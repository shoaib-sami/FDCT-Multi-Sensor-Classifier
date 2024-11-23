import torch, torchvision

from torchvision.transforms import ToTensor, ToPILImage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
import PIL
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import random
import torch.nn.functional as F
import tarfile
import io
import os
import pandas as pd
import cv2

from torch.utils.data import Dataset
import torch


class YourDataset(Dataset):
    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        #self.transform = transform
        self.labels = sorted(os.listdir(self.img_dir), key=lambda x: int(x))

        lb = [int(l) - 1 for l in self.labels]
        self.labels_ohe = lb
        # self.labels_ohe = F.one_hot(torch.as_tensor(lb), num_classes=11) # I have changed from 11 to 5 sami

        self.img_lists = []
        self.all_class_dirs = [os.path.join(self.img_dir, label) for label in self.labels]

        for class_dir in self.all_class_dirs[:10]:  # For 5 classes
            self.img_lists += os.listdir(class_dir)

        self.transform = transforms.Compose([
            transforms.Resize((68, 68)),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225] )
        ])

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, index):
        all_img_abs_dir = []
        for class_dir in self.all_class_dirs[:10]:  # for 5 classes
            all_img_abs_dir += [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]

        image_abs_dir = all_img_abs_dir[index]
        label = int(image_abs_dir.split("/")[-2])
        #print(label)

        try:
            img = Image.open(image_abs_dir) #.convert("L")
            img = self.transform(img)
            x = img
            #x.unsqueeze_(0) # done august 16 2022 by Shoaib
            x = x.repeat(1, 3, 1, 1)   # done august 16 2022
            x = x.view(-1, 68, 68)
            # vis = np.concatenate((img, img, img), axis=1)
            return x, self.labels_ohe[label - 1]

        except PIL.UnidentifiedImageError as ui:
            print(image_abs_dir)
            return None, None