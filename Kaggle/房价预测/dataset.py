from read_data import read_data
import torch
import torch.nn as nn
from torch.utils import data as Data
def dataset():
    data=read_data()
    feature=data.loc[:, ~data.columns.str.contains('SalePrice')]
    label=data['SalePrice']
    feature=torch.from_numpy(feature.values)
    feature=feature.to(torch.float32)
    label=torch.from_numpy(label.values)
    label=label.to(torch.float32)
    dataset=Data.TensorDataset(*(feature,label))
    dataloader=Data.DataLoader(dataset,batch_size=20,shuffle=True)
    return dataloader
if __name__ == '__main__':
    dataset()