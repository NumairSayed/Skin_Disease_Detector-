import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import get_dataloaders
from models import GoogLeNet

device = "cuda" if torch.cuda.is_available() else "cpu"
num_diseases = 485

# Initialize the model
model = GoogLeNet(num_classes=num_diseases).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Get the dataloaders
train_dataloader, _ = get_dataloaders('/img')

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"Loss: {loss.item()}")

# Training loop
epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)

print("Training complete")
