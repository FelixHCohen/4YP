import os
import pandas as pd
import torch
from torch.utils.data import Dataset

path = '/Users/felixcohen/Downloads/archive (5)'
train_df = pd.read_csv(f'{path}/mnist_train.csv')

images = (train_df.copy()).drop('label',axis=1)
labels = train_df.loc[:,'label']

image = torch.Tensor((images.iloc[4]).values,dtype=torch.float32)
image = torch.reshape(image,(28,28))
label = torch.Tensor(labels.iloc[4].values,dtype=torch.float32)
label = torch.reshape(label,(1))

print(image)
print(label)