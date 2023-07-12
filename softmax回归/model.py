import torch
import torch.nn as nn
import torchvision
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(28*28,15*15),
            nn.Linear(15*15,10*10),
            nn.Linear(10*10,10)
        )
    def forward(self,x):
        x=torch.reshape(x,[32,28*28*1])
        x=self.layer1(x)
        x=torch.softmax(x,dim=1)
        return x
if __name__ == '__main__':
    X=torch.randn(32,28,28,1)
    mymodel=mymodel()
    print(mymodel(X).shape)