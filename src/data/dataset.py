import numpy as np

class MyDataset():
    def __init__(self, target_loc, data_type):
        if data_type == "train":
            self.data = np.load(target_loc + "train_triplets.npy")
            self.target = np.load(target_loc + "train_labels.npy")
        elif data_type == "test":
            self.data = np.load(target_loc + "test_triplets.npy")
            self.target = np.load(target_loc + "test_labels.npy")
        elif data_type == "val":
            self.data = np.load(target_loc + "val_triplets.npy")
            self.target = np.load(target_loc + "val_labels.npy")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

