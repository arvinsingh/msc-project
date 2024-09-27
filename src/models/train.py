import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.config import TrainingConfig

def train(
    train_config: TrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    total_epochs: int,
    criterion: nn.Module,
) -> tuple[float, float]:
    
    # change model to training mode.
    model.train()

    mean_metric = MeanMetric()
    correct = 0
    total = 0

    device = train_config.device

    status = f"Train:\tEpoch: {epoch_idx}/{total_epochs}"

    prog_bar = tqdm(train_loader, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    prog_bar.set_description(status)

    for data in prog_bar:
        # send data to appropriate device.
        anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)

        # reset parameters gradient to zero.
        optimizer.zero_grad()

        # forward pass to the model.
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        # triplet loss
        loss = criterion(anchor_output, positive_output, negative_output)

        # find gradients w.r.t training parameters.
        loss.backward()

        # update parameters using gradients.
        optimizer.step()

        # batch Loss.
        mean_metric.update(loss.item(), weight=anchor.size(0))

        # calculate similarities
        positive_similarity = F.pairwise_distance(anchor_output, positive_output, p=2)
        negative_similarity = F.pairwise_distance(anchor_output, negative_output, p=2)

        # apply threshold to determine correct predictions
        threshold = train_config.threshold
        pos_correct = (positive_similarity < threshold).sum().item()
        neg_correct = (negative_similarity > threshold).sum().item()

        correct += pos_correct + neg_correct
        total += 2 * anchor.size(0)  # Two comparisons per triplet

        # update progress bar description.
        step_status = status + f" Train Loss: {mean_metric.compute():.4f}, Train Acc: {correct / total:.4f}"
        prog_bar.set_description(step_status)

    epoch_loss = mean_metric.compute()
    epoch_acc = correct / total

    prog_bar.close()

    return epoch_loss, epoch_acc
