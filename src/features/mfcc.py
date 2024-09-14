import os
import numpy as np
import torchaudio
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset

def create_file_list(root_directory):
    """
    Create a list of audio file paths.
    Structure: root_directory/Fluent/XXX/X.wav
    Structure: root_directory/Nonfluent/XXX/X.wav
    """
    file_list = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".wav"):
                file_list.append(os.path.join(root, file))
    return file_list


def process_frames(frames, max_columns):
    """
    Process frames to ensure consistent number of columns (features).
    """
    if frames.shape[2] < max_columns:
        padding = torch.zeros(1, frames.shape[1], max_columns - frames.shape[2])
        frames = torch.cat((frames, padding), dim=2)
    elif frames.shape[2] > max_columns:
        frames = frames[:, :, :max_columns]
    return frames


def process_waveforms(frames, max_frames):
    """
    Process waveforms to ensure consistent length.
    """
    if frames.shape[1] < max_frames:
        padding = torch.zeros(1, max_frames - frames.shape[1], frames.shape[2])
        frames = torch.cat((frames, padding), dim=1)
    elif frames.shape[1] > max_frames:
        frames = frames[:, :max_frames, :]
    return frames



def process_audio_file(audio_file, max_frames=250, max_columns=400):
    """
    Process an audio file and return MFCC frames and label index.
    """
    waveform, sample_rate = torchaudio.load(audio_file)

    # Apply transform to ensure consistent frame size
    transform = T.MFCC(sample_rate=16000, n_mfcc=13)
    frames = transform(waveform)

    # Process frames
    frames = process_frames(frames, max_columns)

    # Process waveforms
    frames = process_waveforms(frames, max_frames)

    return frames


def AudioFrameDataset(root_directory, max_frames=250, max_columns=400):
    """
    A function that loads audio files and returns MFCC frames.
    """
    file_list = create_file_list(root_directory)

    def __len__():
        return len(file_list)

    def __getitem__(idx):
        audio_file = file_list[idx]
        frames = process_audio_file(audio_file, max_frames, max_columns)
        return frames

    return __len__, __getitem__


def MFCC_collate_fn(batch):
    """
    Custom collate_fn to apply padding to the variable length sequences.
    """
    # Unzip the batch
    frames, labels = zip(*batch)

    # Stack the frames & labels
    frames = torch.stack(frames, dim=0)
    labels = torch.tensor(labels)

    return frames, labels   

def get_dataloader(root_directory, max_frames=250, max_columns=400, batch_size=32):
    """
    Create a DataLoader for the AudioFrameDataset.
    """
    dataset = AudioFrameDataset(root_directory, max_frames, max_columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=MFCC_collate_fn)
    return dataloader

def save_mfcc_features(dataloader, output_dir):
    """
    save all the mfcc features to a pickle file.
    """
    features = []
    labels = []
    for frames, label in dataloader:
        features.append(frames)
        labels.append(label)
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    np.save(output_dir+"mfcc_features.npy", features)
    np.save(output_dir+"mfcc_labels.npy", labels)

def load_mfcc_features(input_dir):
    """
    Load the mfcc features from a pickle file.
    """
    features = np.load(os.path.join(input_dir, "mfcc_features.npy"))
    labels = np.load(os.path.join(input_dir, "mfcc_labels.npy"))
    return features, labels
