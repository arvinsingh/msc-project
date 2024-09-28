import os
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.features import Graph
from src.config import TrainingConfig


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


class LSTM_II(nn.Module):

    def __init__(self, input_shape, lstm_units=128, num_layers=2):
        super(LSTM_II, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=lstm_units, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * lstm_units, 1)  # bidirectional LSTM doubles the units
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2 * lstm_units, 256)  # increased dimensions
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(p=0.5) 

    def forward(self, x):
        x, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = F.relu(self.dense3(x))
        x = F.normalize(self.dense4(x), p=2, dim=1)
        return x


class LSTM_III(nn.Module):
    def __init__(self, input_shape, lstm_units=64):
        super(LSTM_III, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=lstm_units, batch_first=True)

        # batch norma layer after LSTM
        self.batch_norm_lstm = nn.BatchNorm1d(input_shape[0])

        # fully connected layers after flattening the LSTM output
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(lstm_units * input_shape[0], 128)  # input_shape[0] = sequence length
        self.batch_norm_dense1 = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(128, 64)
        self.batch_norm_dense2 = nn.BatchNorm1d(64)
        self.dense3 = nn.Linear(64, 32)

    def forward(self, x):
        # LSTM layer
        x, _ = self.lstm(x)
        
        # batchNorm after LSTM and activation (ReLU)
        x = self.batch_norm_lstm(x)  # permute to apply BatchNorm over the right dimension
        x = F.relu(x)  # permute back to original shape

        # flatten
        x = self.flatten(x)

        # dense layers
        x = F.relu(self.batch_norm_dense1(self.dense1(x)))
        x = F.relu(self.batch_norm_dense2(self.dense2(x)))

        # sigmoid activation for the last layer
        x = torch.sigmoid(self.dense3(x))

        return x

    

class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=2)  # 2D convolution
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm for faster convergence
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.flatten = nn.Flatten()

        # fully connected layers
        conv_output_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)

    def _get_conv_output(self, shape):
        """
        Helper function to calculate the output size after the convolutions
        """
        x = torch.rand(1, 1, *shape)  # 1 sample, 1 channel, height = 250, width = 400
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x.numel()  # Flattened size

    def forward(self, x):
        # add a channel dimension for CNN (batch_size, 1, 250, 400)
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.flatten(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.normalize(self.fc4(x), p=2, dim=1)
        
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
        self.graph_args = {
            'max_hop': 1,
            'dilation': 1
        }
        self.graph = Graph(**self.graph_args)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32).to(TrainingConfig.device)
        self.temporal_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(temporal_kernel_size, 1))
        self.graph_conv1 = GraphConvLayer(64, 32)
        self.temporal_conv2 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(temporal_kernel_size, 1), padding=(1, 0))
        self.num_nodes = num_nodes # why? Do I need this?
        self.num_frames = num_frames

    def forward(self, x):
        # x is of shape [batch_size, num_frames, channel, num_nodes]
        
        # temporal convolution
        x = self.temporal_conv1(x)  # output shape: [batch_size, 64, num_frames, num_nodes]
        x = F.relu(x)

        # permute x to [batch_size, num_nodes, num_frames, channels] for graph convolution
        x = x.permute(0, 3, 2, 1)  # Shape: [batch_size, num_nodes, num_frames, 64]

        # collapse the temporal dimension to prepare for graph convolution
        batch_size, num_nodes, num_frames, num_channels = x.size()
        x = x.view(batch_size * num_frames, num_nodes, num_channels)  # Shape: [batch_size*num_frames, num_nodes, 64]

        # graph convolution
        x = self.graph_conv1(x, self.A)  # operates on [num_nodes, features]
        x = F.relu(x)

        # reshape back to [batch_size, num_nodes, num_frames, 32]
        x = x.view(batch_size, num_frames, num_nodes, -1)

        # permute x back to [batch_size, channels, num_frames, num_nodes]
        x = x.permute(0, 3, 1, 2)

        # second temporal convolution
        x = self.temporal_conv2(x)

        return x
    

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



class SiameseModel(torch.nn.Module):

    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network

    def forward(self, anchor, positive, negative, adj=None):
        embedding_anchor = self.siamese_network(anchor)
        embedding_positive = self.siamese_network(positive)
        embedding_negative = self.siamese_network(negative)
        # cosine embeddings loss - investigate
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