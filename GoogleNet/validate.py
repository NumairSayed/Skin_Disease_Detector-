import torch
import numpy as np
from dataloader import get_dataloaders
from models import GoogLeNet

device = "cuda" if torch.cuda.is_available() else "cpu"
num_diseases = 485

model = GoogLeNet(num_classes=num_diseases).to(device)
_, test_dataloader = get_dataloaders('/img')

def validate(dataloader, model, loss_fn):
    model.eval()
    correct, total_loss = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    total_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    print(f"Validation: Loss {total_loss:.4f}, Accuracy {accuracy:.4f}")

loss_fn = torch.nn.CrossEntropyLoss()
validate(test_dataloader, model, loss_fn)
