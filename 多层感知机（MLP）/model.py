import torch
import torch.nn as nn
import torchvision
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,250),
            nn.ReLU(),
            nn.Linear(250,10)
        )
    def forward(self,x):
        x=self.layer1(x)
        x=torch.softmax(x,dim=1)
        return x
if __name__ == '__main__':
    X=torch.randn(32,1,28,28)
    model=mymodel()
    X=model(X)
    print(X.shape)
