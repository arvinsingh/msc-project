import os
import numpy as np
import pandas as pd

from src.features import mfcc
from src.features import landmarks


class MyDataset():


    def __init__(self, location):
        self.location = location
        self.data_path = self.location + "dataset\\"
        self.general_labels = pd.read_csv(self.location + "labels.csv", index_col=0)
        self.labels = list()
        self.data = list()


    def create_dataset(self):
        dirs = os.listdir(self.data_path)
        for dir in dirs:
            files = os.listdir(self.data_path + dir)
            for i in range(1, 11):
                # check if [1-10].wav file exists
                if str(i) + ".wav" in files:
                    # append label to labels list
                    self.labels.append(self.general_labels.loc[dir][0])
                    # generate MFCCs for .wav file
                    frame = mfcc.process_audio_file(self.data_path + dir + "\\" + str(i) + ".wav")
                    # open .7z file and extract all .ljson files to create a list of landmarks
                    sequence = landmarks.get_mouth_sequences(self.data_path + dir + "\\" + str(i) + ".7z")
                    # finally save
                    self.data.append([frame, sequence])

    
    def save_dataset(self, save_path):
        np.save(save_path + "data.npy", self.data)
        np.save(save_path + "labels.npy", self.labels)
    

    def load_dataset(self, load_path):
        self.data = np.load(load_path + "data.npy", allow_pickle=True)
        self.labels = np.load(load_path + "labels.npy", allow_pickle=True)