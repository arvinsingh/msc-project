import os
import numpy as np
import torchaudio
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.nn.functional import pad

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

def truncate_or_pad_mfcc(mfcc, target_frames=250, target_columns=400):
    # Get the current shape of the MFCC tensor
    current_frames, current_columns = mfcc.shape[-2], mfcc.shape[-1]
    
    # Truncate or pad frames
    if current_frames > target_frames:
        mfcc = mfcc[..., :target_frames]
    else:
        pad_frames = target_frames - current_frames
        mfcc = pad(mfcc, (0, 0, 0, pad_frames), mode='constant', value=0)
    
    # Truncate or pad columns
    if current_columns > target_columns:
        mfcc = mfcc[..., :target_columns]
    else:
        pad_columns = target_columns - current_columns
        mfcc = pad(mfcc, (0, pad_columns, 0, 0), mode='constant', value=0)
    
    return mfcc



def process_audio_file(dir_path, target_sample_rate=16000, n_mfcc=13, max_frames=250, max_columns=400):
    """
    Process an audio file and return MFCC frames and label index.
    :param dir_path: path to the audio files
    :param target_sample_rate: target sample rate
    :param n_mfcc: number of MFCC features - 12-13 for speech recognition tasks
    :param max_frames: maximum number of frames
    :param max_columns: maximum number of columns (features) per frame

    :return: frames
    """

    waveform, original_sample_rate = torchaudio.load(dir_path)
    # Apply transform to ensure consistent sample rate & frame size
    resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    mfcc_transform = T.MFCC(sample_rate=target_sample_rate, n_mfcc=n_mfcc)
    waveform = resampler(waveform)
    frames = mfcc_transform(waveform)
    frames = truncate_or_pad_mfcc(frames, max_frames, max_columns)

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
