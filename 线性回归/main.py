import torch

from train import *
if __name__ == '__main__':
    model=train(500)
    X=torch.tensor([1]).float()
    print(model(X))