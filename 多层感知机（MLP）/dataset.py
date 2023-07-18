import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
def get_label(index):
    label=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return label[index]
def get_dataset():
    mnist_train = torchvision.datasets.FashionMNIST('../data', train=True, transform=transforms.ToTensor(),
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor(),
                                                   download=True)
    dataloader = data.DataLoader(mnist_train, batch_size=32, shuffle=True)
    return dataloader
if __name__ == '__main__':
    data=get_dataset()
    print(next(iter(data))[0].size())