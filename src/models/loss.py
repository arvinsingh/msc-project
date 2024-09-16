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