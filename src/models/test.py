import torch
from torch.utils.data import DataLoader
from src.models import load_model

def test(test_dataset, test_model):
    model = load_model(test_model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            print(f'Test Outputs: {outputs}')

if __name__ == "__main__":
    test()
