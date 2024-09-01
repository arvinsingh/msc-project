import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class LandmarksDataset(Dataset):
    def __init__(self, data, landmarks, transform=None):
        self.data = data
        self.landmarks = landmarks
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name)
        landmarks = self.landmarks.iloc[idx, 1:].values.astype('float')
        landmarks = landmarks.reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample