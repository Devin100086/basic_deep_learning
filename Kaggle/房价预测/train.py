import numpy as np
import torch
import torchvision
import torch.nn as nn
from model import mymodel
from torch.optim import Adam
from tqdm import tqdm
from dataset import dataset
from matplotlib import pyplot as plt
def train(epoch=30):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function=nn.MSELoss().to(device)
    model=mymodel()
    optimizer=Adam(model.parameters(),lr=0.01)
    epochs=range(epoch)
    dataloader=dataset()
    X=np.linspace(1,epoch,epoch)
    Loss=[]
    for epoch in tqdm(epochs,desc="循环次数"):
        all_loss=0
        for i,(feature,label) in enumerate(dataloader):
            output=model(feature)
            loss=loss_function(output,label)
            all_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss.append(all_loss)
    torch.save(model.state_dict(),'model.pt')
    plt.figure()
    plt.plot(X,Loss,'r',label='loss_change')
    plt.legend()
    plt.show()
