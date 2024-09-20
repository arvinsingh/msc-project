import os
import pandas as pd

from src.features import mfcc
from src.features import landmarks

import torch
from torch.utils.data import Dataset


import logging

# configure logging
logging.basicConfig(
    filename='dataset_creation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class MyDataset(Dataset):


    def __init__(self, data=[], labels=[], location=""):
        self.location = location
        self.data_path = self.location + "dataset\\"
        self.general_labels = pd.read_csv(self.location + "labels.csv", index_col=0)
        self.data = data
        self.labels = labels


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


    def create_dataset(self):
        _data = list()
        dirs = os.listdir(self.data_path)
        for dir in dirs:
            files = os.listdir(self.data_path + dir)
            for i in range(1, 11):
                # check if [1-10].wav file exists
                if str(i) + ".wav" in files:
                    try:
                        # generate MFCCs for .wav file
                        frame = mfcc.process_audio_file(self.data_path + dir + "\\" + str(i) + ".wav")
                        frame = torch.squeeze(frame)
                        # open .7z file and extract all .ljson files to create a list of landmarks
                        sequence = landmarks.get_mouth_sequences(self.data_path + dir + "\\" + str(i) + ".7z")
                        # finally save
                        # append label to labels list
                        _data.append([frame, sequence])
                        self.labels.append(self.general_labels.loc[dir][0])
                    except Exception as e:
                        # log error if any
                        logging.error(f"Error processing {dir}\\{i}: {e}")

        self.data = _data

    
    def save_dataset(self, save_path):
        torch.save(self.data, save_path + "data.pt")
        torch.save(self.labels, save_path + "labels.pt")
    

    def load_dataset(self, load_path):
        self.data = torch.load(load_path + "data.pt")
        self.labels = torch.load(load_path + "labels.pt")


class AudioLandmarkTripletDataset(Dataset):


    def __init__(self, dataset, triplets):
        self.dataset = dataset
        self.triplets = triplets


    def __len__(self):
        return len(self.triplets)


    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        anchor = self.dataset[anchor_idx]
        positive = self.dataset[positive_idx]
        negative = self.dataset[negative_idx]
        return anchor, positive, negative


class AudioTripletDataset(Dataset):


    def __init__(self, dataset, triplets):
        self.dataset = dataset
        self.triplets = triplets


    def __len__(self):
        return len(self.triplets)


    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        anchor = self.dataset[anchor_idx][0]
        positive = self.dataset[positive_idx][0]
        negative = self.dataset[negative_idx][0]
        return anchor, positive, negative


class LandmarkTripletDataset(Dataset):


    def __init__(self, dataset, triplets):
        self.dataset = dataset
        self.triplets = triplets


    def __len__(self):
        return len(self.triplets)


    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        anchor = self.dataset[anchor_idx][1]
        positive = self.dataset[positive_idx][1]
        negative = self.dataset[negative_idx][1]
        return anchor, positive, negative