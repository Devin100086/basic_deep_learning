import torch
import torch.nn as nn
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    def forward(self,x):
        x=self.layer1(x)
        x=torch.softmax(x,dim=1)
        return x
if __name__ == '__main__':
    X=torch.rand([32,1,28,28])
    model=mymodel()
    print(model(X).shape)