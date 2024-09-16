import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.config import TrainingConfig
from .loss import triplet_loss


def validate(
    train_config: TrainingConfig, 
    model: nn.Module, 
    valid_loader: torch.utils.data.DataLoader,
    epoch_idx: int, 
    total_epochs: int
) -> tuple[float, float]:

    # Change model to evaluation mode.
    model.eval()

    mean_metric = MeanMetric()
    correct = 0
    total = 0

    device = train_config.device

    status = f"Valid:\tEpoch: {epoch_idx}/{total_epochs}"

    prog_bar = tqdm(valid_loader, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    prog_bar.set_description(status)

    for data in prog_bar:
        # Send data to appropriate device.
        anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)

        # Get the model's predicted logits.
        with torch.no_grad():
            positive_similarity, negative_similarity = model(anchor, positive, negative)


        # Triplet loss
        loss = triplet_loss(positive_similarity, negative_similarity)

        # Batch validation loss.
        mean_metric.update(loss.item(), weight=anchor.size(0))

        # Apply threshold to determine correct predictions
        threshold = train_config.threshold
        pos_correct = (positive_similarity < threshold).sum().item()
        neg_correct = (negative_similarity > threshold).sum().item()

        correct += pos_correct + neg_correct
        total += 2 * anchor.size(0)  # Two comparisons per triplet

        # Update progress bar description.
        step_status = status + f" Valid Loss: {mean_metric.compute():.4f}, Valid Acc: {correct / total:.4f}"
        prog_bar.set_description(step_status)

    valid_loss = mean_metric.compute()
    valid_acc = correct / total

    prog_bar.close()

    return valid_loss, valid_acc
