import torch
from torch.utils.data import DataLoader
from model import TripletLoss, SiameseModel, LSTM

def train(train_dataset, num_epochs=10, input_shape=(250, 400)):

    
    lstm_network = LSTM(input_shape)
    model = SiameseModel(lstm_network)

    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    train()
