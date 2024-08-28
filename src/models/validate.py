import torch
from torch.utils.data import DataLoader
from model import MyModel
from src.data.dataset import MyDataset

def validate():
    model = MyModel()
    val_dataset = MyDataset(train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            print(f'Validation Loss: {loss.item():.4f}')

if __name__ == "__main__":
    validate()
