import os
import torchaudio
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import librosa

class AudioFrameDataset(Dataset):
    """
    A PyTorch Dataset that loads audio files and returns MFCC frames.
    """
    def __init__(self, root_directory, label_to_index, max_frames=250, max_columns=400):
        self.root_directory = root_directory
        self.label_to_index = label_to_index
        self.max_frames = max_frames
        self.max_columns = max_columns
        self.transform = T.MFCC(sample_rate=16000, n_mfcc=13)

        # Create a list of audio file paths
        self.file_list = []
        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.endswith(".wav"):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_file = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(audio_file)

        # Apply transform to ensure consistent frame size
        frames = self.transform(waveform)

        # Ensure all frames have the same number of columns (features)
        if frames.shape[2] < self.max_columns:
            padding = torch.zeros(1, frames.shape[1], self.max_columns - frames.shape[2])
            frames = torch.cat((frames, padding), dim=2)
        elif frames.shape[2] > self.max_columns:
            frames = frames[:, :, :self.max_columns]

        # Ensure all waveforms have the same length
        if frames.shape[1] < self.max_frames:
            padding = torch.zeros(1, self.max_frames - frames.shape[1], frames.shape[2])
            frames = torch.cat((frames, padding), dim=1)
        elif frames.shape[1] > self.max_frames:
            frames = frames[:, :self.max_frames, :]

        # Determine the label based on the higher-level directory structure
        label = os.path.basename(os.path.dirname(os.path.dirname(audio_file)))  # Get the parent directory name
        label_index = self.label_to_index[label]

        return frames, label_index


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

def get_dataloader(root_directory, label_to_index, max_frames=250, max_columns=400, batch_size=32):
    """
    Create a DataLoader for the AudioFrameDataset.
    """
    dataset = AudioFrameDataset(root_directory, label_to_index, max_frames, max_columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=MFCC_collate_fn)
    return dataloader

def save_mfcc_features(dataloader, output_file):
    """
    save all the mfcc features to a pickle file.
    """
    features = []
    labels = []
    for frames, label in dataloader:
        features.append(frames)
        labels.append(label)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    torch.save((features, labels), output_file)

def load_mfcc_features(input_file):
    """
    Load the mfcc features from a pickle file.
    """
    features, labels = torch.load(input_file)
    return features, labels

def visualize_mfcc_feature(audio_file, title, sample_rate=16000, n_mfcc=13):
    """
    Visualize the MFCC feature for a given label and index.
    """
    # load MFCC features for audio
    mfcc_features = librosa.feature.mfcc(y=librosa.load(audio_file, sr=sample_rate)[0], sr=sample_rate, n_mfcc=n_mfcc)
    # plot the MFCC features
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mfcc_features, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.show()