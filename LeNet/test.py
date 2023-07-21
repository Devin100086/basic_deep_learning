import torch
from dataset import *
from model import mymodel
if __name__ == '__main__':
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=mymodel()
    model.load_state_dict(torch.load('model.pt'))
    mnist_test = torchvision.datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor(),
                                                   download=True)
    dataloader = data.DataLoader(mnist_test, batch_size=32, shuffle=True)
    feature=next(iter(dataloader))[0][0]
    feature=torch.unsqueeze(feature,dim=0)
    index=model(feature)
    print("该类别是{}".format(get_label(torch.argmax(index).item())))