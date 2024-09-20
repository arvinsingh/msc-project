import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.config import TrainingConfig
from .loss import triplet_loss


def train(
    train_config: TrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    total_epochs: int,
    adj: torch.Tensor = None
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
        positive_similarity, negative_similarity = model(anchor, positive, negative, adj=adj)

        # triplet loss
        loss = triplet_loss(positive_similarity, negative_similarity)

        # find gradients w.r.t training parameters.
        loss.backward()

        # update parameters using gradients.
        optimizer.step()

        # batch Loss.
        mean_metric.update(loss.item(), weight=anchor.size(0))

        # Aapply threshold to determine correct predictions
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
