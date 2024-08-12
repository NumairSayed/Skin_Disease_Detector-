import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image, ImageOps
from sklearn.metrics import f1_score, recall_score
import numpy as np  
from tqdm import tqdm as tqdm

num_diseases = 485
root_dir = '/home/numair/Desktop/Codes/Skin_Disease/Augmented'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Custom class to pad with a specific value and recenter the image
class ResizeWithPadAndCenter(object):
    def __init__(self, size, padding_value=0):
        self.size = size
        self.padding_value = padding_value

    def __call__(self, img):
        # Calculate padding
        delta_w = max(0, self.size[0] - img.size[0])
        delta_h = max(0, self.size[1] - img.size[1])
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        # Apply padding
        img = ImageOps.expand(img, padding, fill=self.padding_value)
        # Resize to the target size
        img = img.resize(self.size, Image.BILINEAR)
        return img


# mean and std of all images (This needs to be changed as soon as new data is added and merged)

mean= [0.57802826, 0.29917458, 0.26115456]
std= [0.18442076, 0.28176323, 0.25507942]

# Create the transformation pipeline
transform = transforms.Compose([
    ResizeWithPadAndCenter((224, 224), padding_value=128),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


# Create the dataset
dataset = datasets.ImageFolder(root=root_dir, transform=transform)

# Define the split ratio
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# Create the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


class DenseNet201(nn.Module):
    def __init__(self, num_classes=num_diseases):
        super(DenseNet201, self).__init__()

        self.densenet201 = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        
        num_ftrs = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Linear(num_ftrs, num_classes)

        # Turn off autograd for all layers except the classifier 
        for param in self.densenet201.parameters():
            param.requires_grad = True
        #for param in self.densenet201.classifier.parameters():
        #   param.requires_grad = True

    def forward(self, x):
        x = self.densenet201(x)
        return x

model = DenseNet201()
# Move the model to the specified device
model.to(device)

# Define the loss function and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

# Function to check the dimensions and types of images in the dataloader
def check_dataloader(dataloader):
    for batch, (X, y) in enumerate(dataloader):
        for i in range(X.size(0)):
            img = X[i]
            print(f"Image {i} in batch {batch}: size {img.size()}, dtype {img.dtype}")
            if img.size() != (3, 224, 224):
                print(f"Unexpected image size: {img.size()} in batch {batch} index {i}")
            if not torch.is_floating_point(img):
                print(f"Unexpected image type: {img.dtype} in batch {batch} index {i}")

# Function to train the model
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

# Function to test the model
def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    top1_accuracy = np.mean(np.any(all_preds.argsort(axis=1)[:, -1:] == all_labels[:, None], axis=1))
    top2_accuracy = np.mean(np.any(all_preds.argsort(axis=1)[:, -2:] == all_labels[:, None], axis=1))
    top5_accuracy = np.mean(np.any(all_preds.argsort(axis=1)[:, -5:] == all_labels[:, None], axis=1))
    
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Top-1 Accuracy: {top1_accuracy:>0.1f}")
    print(f"Top-2 Accuracy: {top2_accuracy:>0.1f}")
    print(f"Top-5 Accuracy: {top5_accuracy:>0.1f}")

#Check the dataloader for any issues
#check_dataloader(train_dataloader)
#check_dataloader(test_dataloader)

# Training loop
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Training Complete")
