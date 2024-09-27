import torch
from torch import nn
import torch.nn.functional as F


def triplet_loss(positive_similarity, negative_similarity, margin=1.0):
    """
    Compute the triplet loss based on the positive and negative similarities.

    Parameters:
    - positive_similarity: Similarity scores between anchor and positive samples.
    - negative_similarity: Similarity scores between anchor and negative samples.
    - margin: Margin value for the triplet loss.

    Returns:
    - loss: Computed triplet loss.
    """
    loss = F.relu(positive_similarity - negative_similarity + margin)
    
    return loss.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.relu(positive_distance - negative_distance + self.margin)
        return loss.mean()