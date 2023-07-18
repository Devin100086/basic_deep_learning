import torch
import torch.nn as nn
import torchvision
from model import mymodel
from tqdm import tqdm
from torch.optim import SGD
from dataset import get_dataset
def train(epoch=10):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs=range(epoch)
    model=mymodel().to(device)
    dataloader=get_dataset()
    optimizer=SGD(model.parameters(),lr=0.01)
    loss_function=nn.NLLLoss().to(device)
    for epoch in tqdm(epochs,desc='迭代次数',ncols=100):
        for i,data in enumerate(dataloader):
            feature=data[0].to(device)
            label=data[1].to(device)
            ouput=model(feature)
            loss=loss_function(ouput,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(),'model.pt')
if __name__ == '__main__':
    train()