import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, SubsetRandomSampler
import copy
import numpy as np
import torchvision.models as models
import os
import time
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return running_loss / total, correct / total


def federated_averaging(global_model, client_models, client_weights):
    for global_param, client_params in zip(global_model.parameters(), zip(*[client.parameters() for client in client_models])):
        global_param.data = torch.sum(torch.stack([client_weights[i] * client_params[i].data for i in range(len(client_models))]), dim=0)
    return global_model


def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


model = models.mobilenet_v3_small(num_classes=10)

device_ids = [0]
model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

indices = np.random.choice(len(train_dataset), len(train_dataset), replace=False)

client_num = 5
client_indices = np.array_split(indices, client_num)

client_dataloaders = []
for idx in client_indices:
    sampler = SubsetRandomSampler(idx)
    dataloader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    client_dataloaders.append(dataloader)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

rounds = 100
client_weights = [1 / client_num for _ in range(client_num)]

global_model = model
local_models = []

epoch = 2

test_indices = np.random.choice(len(test_dataset), len(test_dataset), replace=False)
test_sampler = SubsetRandomSampler(test_indices)
test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler)

for r in range(rounds):
    local_accuracies = []
    for i, dataloader in enumerate(client_dataloaders):
        print(f"Training client {i + 1}/{client_num} in round {r + 1}/{rounds}")
        if r % epoch == 0:
            if i == 0:
                local_models = []
            local_model = copy.deepcopy(global_model)
            local_models.append(local_model)
        local_optimizer = torch.optim.SGD(local_models[i].parameters(), lr=0.01, momentum=0.9)
        start = time.time()
        loss, accuracy = train(local_models[i], dataloader, criterion, local_optimizer)
        end = time.time()
        local_accuracies.append(accuracy)
        print(f"Client {i + 1} accuracy: {accuracy:.2%}, time: {end - start}")
    if (r + 1) % epoch == 0:
        global_model = federated_averaging(global_model, local_models, client_weights)
        acc = test(global_model, test_dataloader)
        torch.save(global_model, f"./vit/net{r}_{acc}.pth")

accuracy = test(global_model, test_dataloader)
print(f"Ensemble model accuracy: {accuracy:.2%}")
