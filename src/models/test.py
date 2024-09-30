
# To be used for testing the model on the test dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import TrainingConfig


def test(
    audio_model: nn.Module,
    landmark_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    train_config: TrainingConfig,
):
    device = train_config.device
    audio_model.to(device).eval()
    landmark_model.to(device).eval()
    correct = 0
    total = 0

    test_loader = DataLoader(test_loader, batch_size=TrainingConfig.batch_size, shuffle=True)

    

    with torch.no_grad():
        for data in test_loader:
            anchor, positive, negative = data[0], data[1], data[2]

            # generate embeddings for audio data
            anchor_audio_output = audio_model(anchor[0].to(device))
            positive_audio_output = audio_model(positive[0].to(device))
            negative_audio_output = audio_model(negative[0].to(device))
            
            # generate embeddings for facial landmark data
            anchor_facial_output = landmark_model(anchor[1].to(device))
            positive_facial_output = landmark_model(positive[1].to(device))
            negative_facial_output = landmark_model(negative[1].to(device))

            # combine the outputs (concatenating both models' outputs)
            anchor_output = torch.cat((anchor_audio_output, anchor_facial_output), dim=1)
            positive_output = torch.cat((positive_audio_output, positive_facial_output), dim=1)
            negative_output = torch.cat((negative_audio_output, negative_facial_output), dim=1)

            positive_distance = F.pairwise_distance(anchor_output, positive_output, p=2)
            negative_distance = F.pairwise_distance(anchor_output, negative_output, p=2)

            threshold = train_config.threshold
            pos_correct = (positive_distance < threshold).sum().item()
            neg_correct = (negative_distance > threshold).sum().item()

            correct += pos_correct + neg_correct
            total += 2 * anchor.size(0)

    accuracy = (correct / total) * 100
    print(f"Accuracy on test set: {accuracy:.2f}")



if __name__ == "__main__":

    from src.models import test, load_model
    from src.data import MyDataset, AudioLandmarkTripletDataset
    
    # paths
    raw_data_path = "..\\data\\raw\\"
    processed_data_path = "..\\data\\processed\\"

    # load the test dataset
    my_dataset = MyDataset(location=raw_data_path)
    my_dataset.load_dataset(processed_data_path)
    test_AL_triplets_dataset = AudioLandmarkTripletDataset(my_dataset.data, test_triplets.triplets)

    # load the models
    audio_model = load_model("audio_model.pt")
    landmark_model = load_model("landmark_model.pt")

    test(
        audio_model,
        landmark_model,
        test_AL_triplets_dataset,
    )
