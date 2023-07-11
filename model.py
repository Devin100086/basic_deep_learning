import torch
import torch.nn as nn
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(1,1)
        )
    def forward(self,x):
        x=self.layer1(x)
        return x