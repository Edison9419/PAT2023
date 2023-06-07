# coding=gbk

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import datetime
import math

import matplotlib.pyplot as plt
import gc
import numpy
import torch
import torch.nn as nn
import torchvision
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from util.rank_moudle import CBAM, rank, privacy_entropy
import numpy as np
import os.path as osp
import time
import os
from PIL import ImageFile
from tqdm import *


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
train_path = '/home/edison/imagenet1k/train/'
val_path = '/home/edison/imagenet1k/val/'


def topk_accuracy(output, targets, k):
    values, indices = torch.topk(output, k=k, sorted=True)
    targets = torch.reshape(targets, [-1, 1])
    correct = (targets == indices) * 1
    top_k_accuracy = torch.sum(torch.max(correct, dim=1)[0])
    return top_k_accuracy


def accuracy(output, targets):
    cm = np.zeros((6, 6))
    targets = targets.numpy()
    for i in range(output.shape[0]):
        pos = np.unravel_index(np.argmax(output[i].cpu().numpy()), output.shape)
        pre_label = pos[1]
        cm[targets[i]][pre_label] = cm[targets[i]][pre_label] + 1
    return cm


def train(train_epoch,  learning_rate_start):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False).to(device)
    model.load_state_dict(torch.load("./weight/imagenet1k/res_.pth"))

    student_model = models.mobilenet_v3_small(pretrained=False).to(device)

    rank_moudle = CBAM(3, 3).to(device)
    rank_moudle.load_state_dict(torch.load('./weight/imagenet1k/res_rank.pth'))

    train_dataset = torchvision.datasets.ImageFolder('/home/edison/imagenet1k/train/',
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                     ]))

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=128,
                                              shuffle=True,
                                              pin_memory=False,
                                              num_workers=8)

    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate_start)
    schedule = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=0.0001)
    criterion = nn.CrossEntropyLoss()

    temperature = 2.0
    alpha = 0.9

    for epoch in range(train_epoch):
        model.eval()
        student_model.train()
        rank_moudle.train()
        loop = tqdm((trainloader), total=len(trainloader))
        for inputs, target in loop:
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            teacher_output = model(inputs)
            student_output = student_model(inputs)
            loss_hard = criterion(student_output, target)
            loss_soft = nn.KLDivLoss()(nn.functional.log_softmax(student_output/temperature, dim=1), nn.functional.softmax(teacher_output/temperature, dim=1))
            loss = alpha * loss_hard + (1. - alpha) * loss_soft
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedule.step()
        torch.save(student_model.state_dict(), f"./weight/imagenet1k/kd/res_{epoch}.pth")
        torch.save(rank_moudle.state_dict(), f"./weight/imagenet1k/kd/res_rank_{epoch}.pth")


if __name__ == '__main__':
    train(100, 0.01)
