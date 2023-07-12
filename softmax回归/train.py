from dataset import *
from model import mymodel
import torch
import torch.nn as nn
from torch.optim import SGD
def train(epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader=get_dataset()
    loss_function=nn.NLLLoss().to(device)
    model=mymodel().to(device)
    optimizer=SGD(model.parameters(),lr=0.01)
    for epoch in range(epochs):
        all_loss=0
        for i,inputs in enumerate(dataloader):
            feature=inputs[0].to(device)
            label=inputs[1].to(device)
            result=model(feature).float()
            loss=loss_function(result,label)
            all_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0:
                print('第{}迭代的损失量'.format(i),loss)
        print('第{}轮的损失量'.format(epoch),all_loss)
    torch.save(model.state_dict(),'model.pt')
if __name__ == '__main__':
    X=torch.randn(2,2,2)
    print(X)
    X=torch.reshape(X,[2,4])
    print(X)