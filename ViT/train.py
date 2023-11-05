import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from argparse import argparse
import torchvision
from ViT import ViT
import random
from EarlyStopping import EarlyStopping
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
def train():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )

    args=argparse()
    model = ViT(image_size=args.image_size,patch_size=args.patch_size,num_classes=args.num_classes,
                dim=args.dim,depth=args.depth,heads=args.heads,
                mlp_dim=args.mlp_dim).to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    #开始训练
    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
            data_x = data_x.to(torch.float32).to(args.device)
            data_y = data_y.to(torch.float32).to(args.device)
            data_y=data_y.view(data_y.shape[0],1)
            outputs = model(data_x)
            optimizer.zero_grad()
            loss = criterion(outputs, data_y)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, args.epochs, idx, len(train_dataloader), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
        # ====================adjust lr========================
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    #画图
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    train()