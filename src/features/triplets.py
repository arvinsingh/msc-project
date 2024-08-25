import numpy as np
import os


def generate_triplets(anchor_features, anchor_labels, num_triplets=1000):
    triplets = []
    labels = []
    for _ in range(num_triplets):
        # Randomly select anchor sample and its label
        anchor_index = np.random.randint(0, len(anchor_features))
        anchor_label = anchor_labels[anchor_index]

        # Find positive samples - the same label as anchor
        positive_indices = np.where(anchor_labels == anchor_label)[0]
        positive_index = np.random.choice(positive_indices)

        # Find negative samples - different label from anchor
        negative_indices = np.where(anchor_labels != anchor_label)[0]
        negative_index = np.random.choice(negative_indices)

        anchor = anchor_features[anchor_index]
        positive = anchor_features[positive_index]
        negative = anchor_features[negative_index]

        # Include anchor_label in triplet
        triplet = (anchor, positive, negative)
        label = anchor_label

        triplets.append(triplet)
        labels.append(label)

    return np.array(triplets), np.array(labels)


def save_triplets(triplets, target_dir):
    np.save(os.path.join(target_dir, 'train_triplets.npy'), triplets[0])
    np.save(os.path.join(target_dir, 'train_labels.npy'), triplets[1])

    np.save(os.path.join(target_dir, 'val_triplets.npy'), triplets[2])
    np.save(os.path.join(target_dir, 'val_labels.npy'), triplets[3])

    np.save(os.path.join(target_dir, 'test_triplets.npy'), triplets[4])
    np.save(os.path.join(target_dir, 'test_labels.npy'), triplets[5])

    print("Triplets and labels saved to:", target_dir)

def load_triplets(target_dir):
    train_triplets = np.load(os.path.join(target_dir, 'train_triplets.npy'))
    train_labels = np.load(os.path.join(target_dir, 'train_labels.npy'))

    val_triplets = np.load(os.path.join(target_dir, 'val_triplets.npy'))
    val_labels = np.load(os.path.join(target_dir, 'val_labels.npy'))

    test_triplets = np.load(os.path.join(target_dir, 'test_triplets.npy'))
    test_labels = np.load(os.path.join(target_dir, 'test_labels.npy'))

    return (train_triplets, train_labels, val_triplets, val_labels, test_triplets, test_labels)