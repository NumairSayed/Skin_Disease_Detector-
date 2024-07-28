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

pretrained_weights = models.GoogLeNet_Weights.IMAGENET1K_V1

# mean and std of Imagenet 1k dataset for the RGB channels respectively
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create the transformation pipeline
transform = transforms.Compose([
    ResizeWithPadAndCenter((224, 224), padding_value=128),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

root_dir = '/home/numair/Desktop/Codes/Skin_Disease/zzzzzzzz'

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

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Load the pretrained GoogLeNet model
pretrained_model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

# Define the new model with the number of classes for your task
model = GoogLeNet(num_classes=num_diseases)

# Load pretrained weights, excluding the fully connected layer
pretrained_dict = pretrained_model.state_dict()
model_dict = model.state_dict()

# Update the dictionary nomenclature
updated_pretrained_dict = {}
for key in pretrained_dict.keys():
    new_key = key.replace("inception3a", "inception3a").replace("inception3b", "inception3b") \
                 .replace("inception4a", "inception4a").replace("inception4b", "inception4b") \
                 .replace("inception4c", "inception4c").replace("inception4d", "inception4d") \
                 .replace("inception4e", "inception4e").replace("inception5a", "inception5a") \
                 .replace("inception5b", "inception5b")
    updated_pretrained_dict[new_key] = pretrained_dict[key]

for name, param in updated_pretrained_dict.items():
    if name in model_dict and param.shape == model_dict[name].shape:
        model_dict[name].copy_(param)

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.reset_parameters()

# Freeze all layers except the new fully connected layer
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

# Move the model to the specified device
model.to(device)

# Define the loss function and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
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

# Check the dataloader for any issues
check_dataloader(train_dataloader)
check_dataloader(test_dataloader)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Training Complete")
