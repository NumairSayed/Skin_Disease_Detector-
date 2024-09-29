import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image, ImageOps

# Custom class to pad with a specific value and recenter the image
class ResizeWithPadAndCenter(object):
    def __init__(self, size, padding_value=0):
        self.size = size
        self.padding_value = padding_value

    def __call__(self, img):
        delta_w = max(0, self.size[0] - img.size[0])
        delta_h = max(0, self.size[1] - img.size[1])
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding, fill=self.padding_value)
        img = img.resize(self.size, Image.BILINEAR)
        return img

def get_dataloaders(root_dir, batch_size=32, train_ratio=0.8):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        ResizeWithPadAndCenter((224, 224), padding_value=128),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader

