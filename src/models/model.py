import os
import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):

    def __init__(self, input_shape, lstm_units=64):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=lstm_units, num_layers=1, batch_first=True)
        self.attention = nn.Linear(lstm_units, 1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(lstm_units, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)

    def forward(self, x):
        x, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.normalize(self.dense3(x), p=2, dim=1)
        return x


class SiameseModel(torch.nn.Module):

    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network

    def forward(self, anchor, positive, negative):
        embedding_anchor = self.siamese_network(anchor)
        embedding_positive = self.siamese_network(positive)
        embedding_negative = self.siamese_network(negative)
        positive_similarity = euclidean_distance(embedding_anchor, embedding_positive)
        negative_similarity = euclidean_distance(embedding_anchor, embedding_negative)
        return positive_similarity, negative_similarity

    
def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two tensors.
    Simplified version of the pairwise distance function.
    """
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))


def euclidean_distance_original(vects):
    """
    Compute the Euclidean distance between two tensors.
    Includes a small epsilon for numerical stability.
    """
    x, y = vects
    sum_square = torch.sum(torch.square(x - y), dim=1, keepdim=True)
    return torch.sqrt(torch.maximum(sum_square, torch.tensor(1e-07)))


def save_model(model, device, model_dir="models", model_file_name="AV_classifier.pt"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # Make sure you transfer the model to cpu.
    if device == "cuda":
        model.to("cpu")

    # Save the 'state_dict'
    torch.save(model.state_dict(), model_path)

    if device == "cuda":
        model.to("cuda")

    return


def load_model(model, model_dir="models", model_file_name="AV_classifier.pt", device=torch.device("cpu")):
    model_path = os.path.join(model_dir, model_file_name)

    # Load model parameters by using 'load_state_dict'.
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def MyModel(input_shape):
    """
    A wrapper function for Siamese LSTM.
    """
    model = LSTM(input_shape=input_shape)
    model = SiameseModel(model)
    return model