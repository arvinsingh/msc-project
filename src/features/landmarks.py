import os
import numpy as np
import py7zr
import json
import shutil

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
        # points 48 to 68 correspond to the mouth contour
        mouth_contour_points = landmarks[48: 68]
    return mouth_contour_points

def get_mouth_sequences(landmarks_path, max_frames=250):
    sequences = np.zeros((3, max_frames, 20, 1))
    with py7zr.SevenZipFile(landmarks_path, mode='r') as archive:
        archive.extractall('temp')
        for i, name in enumerate(archive.getnames()):
            if i >= max_frames:
                break
            if name.endswith(".ljson"):
                data = extract_mouth_contour_points('temp\\' + name)
                sequences[:, i, :, 0] = np.transpose(data)
        shutil.rmtree('temp')
    return sequences

def load_landmarks(dir_list, max_frames=150):
    landmarks = []
    for file in dir_list:
        landmarks.append(get_mouth_sequences(file, max_frames))
    return landmarks

def create_landmarks_dataset(data_path, max_frames=150):
    dataset = []
    landmarks_dir = get_landmarks_dir(data_path)
    dataset = load_landmarks(landmarks_dir, max_frames)
    final_dataset = np.stack(dataset, axis=-1)
    final_dataset = np.transpose(final_dataset, (4, 0, 1, 2, 3))
    return final_dataset

def save_landmark_dataset(dataset, save_path, filename):
    np.save(save_path + filename, dataset)

def load_landmark_dataset(data_path):
    return np.load(data_path)
