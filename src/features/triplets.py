import random
import torch


class TripletGenerator:
    def __init__(self, dataset=None, num_triplets=1000, load=False, root_path=None, prefix=""):
        self.prefix = prefix
        if load:
            self.triplets = self._load_triplets(root_path)
        else:
            self.triplets = self._create_triplets(dataset, num_triplets)
        

    def _create_triplets(self, dataset, num_triplets=1000):
        triplets = []
        if isinstance(dataset, torch.utils.data.Subset):
            indices = dataset.indices
            labels = torch.tensor([dataset.dataset.labels[i] for i in indices])
        else:
            indices = list(range(len(dataset)))
            labels = torch.tensor(dataset.labels)

        label_to_indices = {label.item(): (labels == label).nonzero(as_tuple=True)[0].tolist() for label in set(labels.numpy())}
        for _ in range(num_triplets):
            anchor_label = random.choice(list(label_to_indices.keys()))
            anchor_idx = random.choice(label_to_indices[anchor_label])
            positive_idx = random.choice(label_to_indices[anchor_label])
            while positive_idx == anchor_idx:
                positive_idx = random.choice(label_to_indices[anchor_label])

            negative_label = random.choice(list(label_to_indices.keys()))
            while negative_label == anchor_label:
                negative_label = random.choice(list(label_to_indices.keys()))
            negative_idx = random.choice(label_to_indices[negative_label])

            triplets.append((anchor_idx, positive_idx, negative_idx, anchor_label))

        return triplets


    def save_triplets(self, root_path):
        torch.save(self.triplets, root_path + self.prefix + "triplets.pt")


    def _load_triplets(self, root_path):
        return torch.load(root_path + self.prefix + "triplets.pt")