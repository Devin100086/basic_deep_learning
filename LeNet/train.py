import numpy
import torch.cuda
from dataset import get_dataset
from model import mymodel
import torch.nn as nn
from torch.optim import Adam,SGD
from tqdm import tqdm
from matplotlib import pyplot as plt
def train(epochs=30):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset=get_dataset()
    model=mymodel().to(device)
    loss_function=nn.NLLLoss().to(device)
    optimizer=SGD(model.parameters(),lr=0.01)
    epochs=range(epochs)
    Loss=[]
    for epoch in tqdm(epochs):
        all_loss=0
        for i,inputs in enumerate(dataset):
            feature=inputs[0].to(device)
            label=inputs[1].to(device)
            output=model(feature)
            loss=abs(loss_function(output,label))
            all_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss.append(all_loss)
        print('第{}轮的损失函数为{}'.format(epoch,all_loss))
    X=numpy.linspace(1,len(Loss),len(Loss))
    torch.save(model.state_dict(),'model.pt')
    plt.figure()
    plt.plot(X,Loss, 'r','loss')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    train()