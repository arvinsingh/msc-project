
# To be used for testing the model on the test dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TrainingConfig


def test(
    train_config: TrainingConfig,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    ):

    model.eval()
    correct = 0
    total = 0
    device = train_config.device


    with torch.no_grad():
        for data in test_loader:
            anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)

            # Generate embeddings
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            # Calculate distances
            positive_distance = F.pairwise_distance(anchor_output, positive_output, p=2)
            negative_distance = F.pairwise_distance(anchor_output, negative_output, p=2)

            # Determine correct predictions
            threshold = train_config.threshold
            pos_correct = (positive_distance < threshold).sum().item()
            neg_correct = (negative_distance > threshold).sum().item()

            correct += pos_correct + neg_correct
            total += 2 * anchor.size(0)  # Two comparisons per triplet

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    test()
