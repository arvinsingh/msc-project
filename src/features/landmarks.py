import os
import numpy as np
import torch
from torch.utils.data import Dataset
import py7zr
import json


class LandmarksDataset(Dataset):
    def __init__(self, data, transform=None):
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
    
def get_landmarks_dir(data_path):
    '''
    Get the list of directories containing the landmarks data
    :param data_path: path to the data directory
    :return: list of directories containing the landmarks data
    '''

    dir_list = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".7z"):
                dir_list.append(os.path.join(root, file))
    return dir_list

def extract_mouth_contour_points(json_file):
    mouth_contour_points = np.zeros((20, 3))
    with open(json_file, 'r') as f:
        data = json.load(f)
        landmarks = data['landmarks']['points']
        # points 48 to 67 correspond to the mouth contour
        mouth_contour_points = landmarks[48: 68]
    return mouth_contour_points

def get_mouth_sequences(landmarks_path, max_frames=150):
    sequences = np.zeros((3, max_frames, 20, 1))
    with py7zr.SevenZipFile(landmarks_path, mode='r') as archive:
        archive.extractall()
        for i, name in enumerate(archive.getnames()):
            if i >= max_frames:
                break
            if name.endswith(".ljson"):
                data = extract_mouth_contour_points(name)
                sequences[:, i, :, 0] = data.T
    return sequences


def load_landmarks(dir_list, max_frames=150):
    landmarks = []
    for file in dir_list:
        landmarks.append(get_mouth_sequences(file, max_frames))
    return landmarks

def pad_sequences(sequences, max_frames=150):
    padded_sequences = np.zeros((len(sequences), 3, max_frames, 20, 1))
    for i, sequence in enumerate(sequences):
        padded_sequences[i, :, :sequence.shape[1], :, :] = sequence
    return padded_sequences

def create_landmarks_dataset(data_path, max_frames=150):
    dataset = []
    landmarks_dir = get_landmarks_dir(data_path)
    dataset = load_landmarks(landmarks_dir, max_frames)
    dataset = pad_sequences(dataset, max_frames)
    return dataset

def save_landmarks(dataset, save_path):
    np.save(save_path, dataset)