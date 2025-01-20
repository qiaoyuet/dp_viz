import os,sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

import gc # For the memory management (Garbagge collector)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available device : {device}")

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],      # There are three values because
    std=[0.2, 0.2, 0.2],       #   there are 3 channels R, G, B
)

# define transforms
transform = transforms.Compose([
        transforms.Resize((224,224)), # Same size as the paper
        transforms.ToTensor(),
        normalize,
])

# User parameters
data_dir = "/home/qiaoyuet/project/data"
valid_size = 0.1  # ratio between train and valid data
batch_size = 64   # like in the paper

##### Train and valid datasets
train_dataset = datasets.CIFAR10(
    root=data_dir, train=True,
    download=True, transform=transform,
)

valid_dataset = datasets.CIFAR10(
    root=data_dir, train=True,
    download=True, transform=transform,
)

# Split the data between train and valid

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

# Shuffling the data
np.random.seed(42)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, sampler=valid_sampler)

##### Test dataset
dataset = datasets.CIFAR10(
    root=data_dir, train=False,
    download=True, transform=transform, )

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    # Constructor
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # 7x7 conv, 64, /2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7,
                      stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # pool, /2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        # Average pooling
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Fully connected layer
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

num_epochs = 30
layers = [3, 4, 6, 3]

num_classes = 10

batch_size = 16
learning_rate = 0.005

model = ResNet(Bottleneck, layers).to(device) # ResNet18 [in + 16 + out]

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                            weight_decay=0.001, momentum=0.9)


total_step = len(train_loader)

loss_list = []
acc_list = []

for epoch in range(num_epochs):
    # for i, (images, labels) in enumerate(train_loader):
    for i, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del images, labels, outputs
        torch.cuda.empty_cache()
        gc.collect()

    print('Epoch [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))
    loss_list.append(loss.item())

    # # Validation
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in valid_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         del images, labels, outputs
    #     acc_list.append(100 * correct / total)
    #     print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    # Test
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        acc_list.append(100 * correct / total)
        print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
