import torch
import torch.utils.data as data
def dataset(X):
    w=torch.tensor([1.2])
    b=torch.tensor([5])
    label=X*w+b
    features=X
    dataset=data.TensorDataset(*(features,label))
    dataloader=data.DataLoader(dataset,2,shuffle=True)
    return dataloader
if __name__ == '__main__':
    X=torch.randn(6,1)+10
    dataloader=dataset(X)
    for feature,label in dataloader:
        print('feature:{}'.format(feature),'label:{}'.format(label))
    print(next(iter(dataloader)))