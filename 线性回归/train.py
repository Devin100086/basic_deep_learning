import torch
import torch.nn as nn
from dataset import *
from torch.optim import SGD
from model import mymodel
def train(epoch=10):
    X=torch.randn(10000,1)
    dataloader=dataset(X)
    loss_function=nn.MSELoss()
    model=mymodel()
    optimizer=SGD(model.parameters(),lr=0.01)
    Min_loss=1e9
    for feature,label in dataloader:
        result=model(feature)
        loss=loss_function(result,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Min_loss=min(Min_loss,loss)
    print(Min_loss)
    return model