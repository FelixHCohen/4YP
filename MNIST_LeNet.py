import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
import tqdm
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')
device = 'mps'
path = '/Users/felixcohen/Downloads/archive (5)'
train_df = pd.read_csv(f'{path}/mnist_train.csv')
test_df = pd.read_csv(f'{path}/mnist_test.csv')


transform = A.Compose([
    A.ElasticTransform(alpha=10, sigma=25, alpha_affine=0, p=0.4), # elastically distort the image
    A.GaussNoise(var_limit=(0.01), p=0.4), # add Gaussian noise to the image
    A.Rotate(limit=(-15, 15), p=0.4) # randomly rotate the image within +-5 degrees (small rotation as 180 degrees changes semantics of image) e.g. 6->9 similarly cant flip
])
class MNISTDataset(Dataset):
    def __init__(self, df,transform=None):
        self.labels = df.loc[:,'label']
        self.images = df.copy().drop('label',axis=1) # need to check if df.loc makes a copy, if so then copy() is not needed as we can alter original df
        self.images = self.images.apply(lambda x: x/255)
        self.transform = transform
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images.iloc[idx].values
        image = np.reshape(image, (28, 28))
        image = image[:,:,np.newaxis]
        if self.transform != None:
            # create HxWx3 array for albumentations input
            image = np.repeat(image,3,axis=-1)
            image = self.transform(image=image)['image']
            #squeeze back to 1 channel
            image = np.mean(image,axis=-1)
            image = image[np.newaxis,:,:]
        image = torch.as_tensor(image, dtype=torch.float32)
        if self.transform ==None:
            #move channel axis to front CxHxW
            image = image[:,:,0]
            image = image[np.newaxis,:,:]
        number = self.labels.iloc[idx]
        label = torch.zeros(10,dtype=torch.float32)
        label[number] = 1
        label = torch.squeeze(label)

        return image,label


class LeNet(nn.Module):
    def __init__(self, kernels,output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=kernels[0],
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=kernels[0],
                               out_channels=kernels[1],
                               kernel_size=5)

        self.fc_1 = nn.Linear(kernels[1] * 4 * 4, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        # x = [batch size, 1, 28, 28]
        # print(f'initial: {x.shape}')
        x = self.conv1(x)
        # x = [batch size, 6, 24, 24]
        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 6, 12, 12]
        x = F.relu(x)
        # print(f'conv + pool 1: {x.shape}')
        x = self.conv2(x)
        # x = [batch size, 16, 8, 8]
        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 16, 4, 4]
        x = F.relu(x)
        # print(f'conv + pool 2: {x.shape}')
        x = x.view(x.shape[0],-1,)
        # x = [batch size, 16*4*4 = 256]
        x = self.fc_1(x)
        # x = [batch size, 120]
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc_2(x)
        # x = batch size, 84]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc_3(x)
        # x = [batch size, output dim]
        return x

def train_batch(images,labels,model,optimizer,criterion):
    images,labels = images.to(device),labels.to(device)
    outputs = model(images)
    loss = criterion(outputs,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

def train(model, loader,test_loader, criterion, optimizer, config):

    wandb.watch(model,criterion,log='all',log_freq=50) #this is freq of gradient recordings
    model.train()
    example_ct = 0
    batch_ct = 0
    for epoch in range(config.epochs):
        for _,(images, labels) in enumerate(loader):

            loss = train_batch(images,labels,model,optimizer,criterion)
            example_ct += len(images)
            batch_ct +=1

            if ((batch_ct+1)%50)==0:
                train_log(loss,example_ct,epoch)

            if ((batch_ct + 1) % 150) == 0:
                test(model,test_loader,criterion)




def test(model,test_loader,criterion):
    model.eval()

    with torch.no_grad():
        correct,total = 0,0
        total_loss = 0
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            total_loss += loss
            _,predicted = torch.max(outputs.data,1)
            _,label_to_num = torch.max(labels,1)
            total += labels.size(0)
            correct += (predicted == label_to_num).sum().item()

        print(f"Accuracy of the model on the {total} " +
                  f"test images: {correct / total:%}")
        wandb.log({"test_accuracy": correct / total,"avg_test_loss": total_loss/total})

            # Save the model in the exchangeable ONNX format
        torch.onnx.export(model, images, "model.onnx")
        wandb.save("model.onnx")

def make_loader(dataset,batch_size):
    loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,)
    return loader

def get_data(train):
    if train == True:
        return MNISTDataset(train_df,transform=None)
    else:
        return MNISTDataset(test_df,transform=None) # no need to transform for testing set
def make(config):

    train,test = get_data(train=True),get_data(train=False)
    train_loader = make_loader(train,batch_size=config.batch_size)
    test_loader = make_loader(test,batch_size=config.batch_size)
    criterion = nn.CrossEntropyLoss()
    model = LeNet(config.kernels,config.classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = config.learning_rate)

    return model,train_loader,test_loader,criterion,optimizer
def model_pipeline(hyperparameters):
    with wandb.init(project="LeNet-demo",config=hyperparameters):
        config = wandb.config

        model,train_loader,test_loader,criterion,optimizer = make(config)
        print(model)

        train(model,train_loader,test_loader,criterion,optimizer,config)

    return model

config = dict(epochs=50,classes=10,kernels = [6,16],batch_size=256,learning_rate=0.001,dataset="MNIST",architecture="LeNet")
model = model_pipeline(config)
# train =  MNISTDataset(train_df.iloc[4:9],transform=transform)
# train_loader = make_loader(train,batch_size=1)
# img_list,label_list = [],[]
# for image,label in train_loader:
#     img_list.append(image)
#     label_list.append(label)
#
# fig, axes = plt.subplots(2, 2)
# # display each array as an image on each axis
# axes[0, 0].imshow(img_list[0][0,0,:,:])
# axes[0, 1].imshow(img_list[1][0,0,:,:])
# axes[1, 0].imshow(img_list[2][0,0,:,:])
# axes[1, 1].imshow(img_list[3][0,0,:,:])
#
# # show the plot
# plt.show()