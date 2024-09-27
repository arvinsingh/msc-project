import os
import time
import numpy as np
import torch
from torch import optim

from src.config import SystemConfig, TrainingConfig, setup_system

from .loss import TripletLoss
from .train import train
from .validate import validate


def main(model,
         data_loader,
         summary_writer, 
         scheduler=None, 
         system_config=SystemConfig(), 
         training_config=TrainingConfig(), 
         data_augmentation=False,
         ):
    
    # system config
    setup_system(system_config)

    # data loader
    train_loader, valid_loader = data_loader

    NUM_EPOCHS = training_config.epochs_count

    # acceleration device.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # send model to device (GPU/CPU)
    model.to(device)

    # optimizer.
    optimizer = optim.Adam(model.parameters(), lr=training_config.init_learning_rate, weight_decay=training_config.weight_decay)

    criterion = TripletLoss()

    best_loss = torch.tensor(np.inf)

    # epoch train & valid loss accumulator.
    epoch_train_loss = []
    epoch_valid_loss = []

    # epoch train & valid accuracy accumulator.
    epoch_train_acc = []
    epoch_valid_acc = []

    # trainig time measurement
    t_begin = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(training_config, model, optimizer, train_loader, epoch + 1, NUM_EPOCHS, criterion)
        val_loss, val_accuracy = validate(training_config, model, valid_loader, epoch + 1, NUM_EPOCHS, criterion)

        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)

        epoch_valid_loss.append(val_loss)
        epoch_valid_acc.append(val_accuracy)

        summary_writer.add_scalar("Loss/Train", train_loss, epoch)
        summary_writer.add_scalar("Accuracy/Train", train_acc, epoch)

        summary_writer.add_scalar("Loss/Validation", val_loss, epoch)
        summary_writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"\nModel Improved... Saving Model ... ", end="")
            try:
                torch.save(model.state_dict(), os.path.join(training_config.checkpoint_dir, training_config.save_audio_model_name))
                print("Done.\n")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        if scheduler:
            scheduler.step(val_loss)

        print(f"{'='*72}\n")
        

    print(f"Total time: {(time.time() - t_begin):.2f}s, Best Loss: {best_loss:.3f}")

    return epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc


if __name__ == "__main__":
    pass