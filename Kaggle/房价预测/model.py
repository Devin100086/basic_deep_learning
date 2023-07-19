import torch.nn as nn
import torch
import torchvision
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1=nn.Sequential(
            nn.Flatten(),
            nn.Linear(331,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,1)
        )
    def forward(self,x):
        x=self.layer1(x)
        return x
