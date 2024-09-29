import torch
import numpy as np
from dataloader import get_dataloaders
from models import GoogLeNet

device = "cuda" if torch.cuda.is_available() else "cpu"
num_diseases = 485

model = GoogLeNet(num_classes=num_diseases).to(device)
_, test_dataloader = get_dataloaders('/img')

def test_model(dataloader, model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    # Compute evaluation metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.hstack(all_labels)
    
    top1_accuracy = np.mean(np.any(all_preds.argsort(axis=1)[:, -1:] == all_labels[:, None], axis=1))
    top5_accuracy = np.mean(np.any(all_preds.argsort(axis=1)[:, -5:] == all_labels[:, None], axis=1))
    
    print(f"Top-1 Accuracy: {top1_accuracy}")
    print(f"Top-5 Accuracy: {top5_accuracy}")

test_model(test_dataloader, model)
