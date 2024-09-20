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
    

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, num_frames, temporal_kernel_size=3):
        super(STGCN, self).__init__()
        self.temporal_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(temporal_kernel_size, 1))
        self.graph_conv1 = GraphConvLayer(64, 32)
        self.temporal_conv2 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(temporal_kernel_size, 1))
        self.num_nodes = num_nodes # why? Do I need this?
        self.num_frames = num_frames

    def forward(self, x, adj):
        x = self.temporal_conv1(x)
        x = F.relu(x)
        x = x.permute(0, 3, 1, 2)  # rearrange dimensions for graph convolution
        x = self.graph_conv1(x, adj)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)  # rearrange dimensions back
        x = self.temporal_conv2(x)
        return x


class SiameseModel(torch.nn.Module):

    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network

    def forward(self, anchor, positive, negative, adj=None):
        embedding_anchor = self.siamese_network(anchor)
        embedding_positive = self.siamese_network(positive)
        embedding_negative = self.siamese_network(negative)
        positive_similarity = euclidean_distance(embedding_anchor, embedding_positive)
        negative_similarity = euclidean_distance(embedding_anchor, embedding_negative)
        return positive_similarity, negative_similarity


class CombinedSiameseNetwork(nn.Module):
    def __init__(self, audio_network, landmarks_network):
        super(CombinedSiameseNetwork, self).__init__()
        self.audio_network = audio_network
        self.landmarks_network = landmarks_network
        self.fc = nn.Linear(64, 32)  # Combine embeddings

    def forward(self, anchor, positive, negative, adj):
        anchor_audio, anchor_landmarks = anchor[0], anchor[1]
        positive_audio, positive_landmarks = positive[0], positive[1]
        negative_audio, negative_landmarks = negative[0], negative[1]

        # process audio data
        anchor_audio_embedding = self.audio_network(anchor_audio)
        positive_audio_embedding = self.audio_network(positive_audio)
        negative_audio_embedding = self.audio_network(negative_audio)

        # process landmarks data
        anchor_landmarks_embedding = self.landmarks_network(anchor_landmarks, adj)
        positive_landmarks_embedding = self.landmarks_network(positive_landmarks, adj)
        negative_landmarks_embedding = self.landmarks_network(negative_landmarks, adj)

        # combine embeddings
        anchor_embedding = torch.cat((anchor_audio_embedding, anchor_landmarks_embedding), dim=1)
        positive_embedding = torch.cat((positive_audio_embedding, positive_landmarks_embedding), dim=1)
        negative_embedding = torch.cat((negative_audio_embedding, negative_landmarks_embedding), dim=1)

        # further process combined embeddings
        anchor_embedding = F.relu(self.fc(anchor_embedding))
        positive_embedding = F.relu(self.fc(positive_embedding))
        negative_embedding = F.relu(self.fc(negative_embedding))

        # calculate similarities
        positive_similarity = euclidean_distance(anchor_embedding, positive_embedding)
        negative_similarity = euclidean_distance(anchor_embedding, negative_embedding)

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

    # transfer the model to cpu.
    if device == "cuda":
        model.to("cpu")

    # Save the 'state_dict'
    torch.save(model.state_dict(), model_path)

    if device == "cuda":
        model.to("cuda")

    return


def load_model(model, model_dir="models", model_file_name="AV_classifier.pt", device=torch.device("cpu")):
    model_path = os.path.join(model_dir, model_file_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def audio_model(input_shape):
    """
    A wrapper function for Siamese LSTM.
    """
    model = LSTM(input_shape=input_shape)
    model = SiameseModel(model)
    return model

def landmark_model():
    """
    A wrapper function for Siamese STGCN.
    """
    model = STGCN(num_nodes=20, in_channels=3, out_channels=64, num_frames=250)
    model = SiameseModel(model)
    return model

def combined_model(audio_input_shape):
    """
    A wrapper function for CombinedSiameseNetwork.
    """
    audio_network = LSTM(input_shape=audio_input_shape)
    landmarks_network = STGCN(num_nodes=20, in_channels=3, out_channels=64, num_frames=250)
    model = CombinedSiameseNetwork(audio_network, landmarks_network)
    return model