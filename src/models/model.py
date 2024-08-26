import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(self, input_shape, lstm_units=64):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=lstm_units, num_layers=1, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(lstm_units, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))

        return x


class SiameseModel(torch.nn.Module):
    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network

    def forward(self, anchor, positive, negative):
        embedding_anchor = self.siamese_network(anchor)
        embedding_positive = self.siamese_network(positive)
        embedding_negative = self.siamese_network(negative)
        positive_similarity = euclidean_distance((embedding_anchor, embedding_positive))
        negative_similarity = euclidean_distance((embedding_anchor, embedding_negative))
        return positive_similarity, negative_similarity
    

class TripletLoss(torch.nn.Module):
    def __init__(self, alpha=0.3):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        anchor_positive_distance = torch.sqrt(torch.sum(torch.square(anchor - positive), dim=-1))
        anchor_negative_distance = torch.sqrt(torch.sum(torch.square(anchor - negative), dim=-1))
        loss = torch.maximum(0.0, anchor_positive_distance - anchor_negative_distance + self.alpha)
        return torch.mean(loss)
    

def euclidean_distance(vects):
    x, y = vects
    sum_square = torch.sum(torch.square(x - y), dim=1, keepdim=True)
    return torch.sqrt(torch.maximum(sum_square, torch.tensor(1e-07)))