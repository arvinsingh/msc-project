
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fluency(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Fluency, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def init_fluency(input_size, hidden_size, num_layers, dropout):
    return Fluency(input_size, hidden_size, num_layers, dropout)

def train_fluency(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, (x, y) in enumerate(val_loader):
                y_hat = model(x)
                val_loss += criterion(y_hat, y)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

def predict_fluency(model, x):
    
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
    return y_hat
