import torch
import torch.nn as nn
from model import mymodel
if __name__ == '__main__':
    model=mymodel()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    ###这里输入你要预测的图片