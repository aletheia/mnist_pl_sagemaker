import os
import math
import random as rn
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T, datasets
import pytorch_lightning as pl



class MNISTClassifier(pl.LightningModule):
    def __init__(self, train_data_dir,batch_size=128,test_data_dir=None, num_workers=4):
        super(MNISTClassifier, self).__init__()

        self.batch_size = batch_size
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.num_workers = num_workers

        self.conv_layer_1 = torch.nn.Sequential(
        torch.nn.Conv2d(3,28, kernel_size=5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2))
        self.conv_layer_2 = torch.nn.Sequential(
        torch.nn.Conv2d(28,10, kernel_size=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2))
        self.dropout1=torch.nn.Dropout(0.25)
        self.fully_connected_1=torch.nn.Linear(250,18)
        self.dropout2=torch.nn.Dropout(0.08)
        self.fully_connected_2=torch.nn.Linear(18,10)

    def load_split_train_test(self, valid_size = .2):
        num_workers = self.num_workers
        train_transforms = T.Compose([T.RandomHorizontalFlip(),                                       
                                           T.ToTensor(),
                                           T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        train_data = datasets.ImageFolder(self.train_data_dir, transform=train_transforms)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)        

        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(train_data, sampler=val_sampler, batch_size=self.batch_size, num_workers=num_workers)

        test_loader = None
        if self.test_data_dir is not None:
            test_transforms = T.Compose([T.ToTensor(),T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
            test_data = datasets.ImageFolder(self.test_data_dir, transform=test_transforms)
            test_loader = torch.utils.data.DataLoader(train_data,batch_size=self.batch_size, num_workers=num_workers)
        return train_loader, val_loader, test_loader
    
    def prepare_data(self):
        self.train_loader, self.val_loader, self.test_loader  = self.load_split_train_test()
        
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
#    def test_dataloader(self):
#        return DataLoader(MNIST(os.getcwd(), train=False, download=False, transform=transform.ToTensor()), batch_size=128)
    
    def forward(self,x):
        x=self.conv_layer_1(x)
        x=self.conv_layer_2(x)
        x=self.dropout1(x)
        x=torch.relu(self.fully_connected_1(x.view(x.size(0),-1)))
        x=F.leaky_relu(self.dropout2(x))
        return F.softmax(self.fully_connected_2(x), dim=1)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def training_step(self, batch, batch_idx):
        
        # Get input and output from batch
        x, labels = batch
        
        # Compute prediction through the network
        prediction = self.forward(x)
        
        loss = F.nll_loss(prediction, labels)
        
        # Logs training loss
        logs={'train_loss':loss}
        
        output = {
            # This is required in training to be used by backpropagation
            'loss':loss,
            # This is optional for logging pourposes
            'log':logs
        }
        
        return output
    
    def validation_step(self, batch, batch_idx):
        x, labels = batch
        prediction = self.forward(x)
        return {
            'val_loss': F.cross_entropy(prediction, labels)
        }
    
    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Average training loss: '+str(avg_loss.item()))
        logs = {'val_loss':avg_loss}
        return {
            'avg_val_loss':avg_loss,
            'log':logs
        }