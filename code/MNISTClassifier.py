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
    def __init__(self, train_data_dir, batch_size=128, test_data_dir=None, num_workers=4):
        '''Constructor method 

        Parameters:
        train_data_dir (string): path of training dataset to be used either for training and validation
        batch_size (int): number of images per batch. Defaults to 128.
        test_data_dir (string): path of testing dataset to be used after training. Optional.
        num_workers (int): number of processes used by data loader. Defaults to 4.

        '''

        # Invoke constructor
        super(MNISTClassifier, self).__init__()

        # Set up class attributes
        self.batch_size = batch_size
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.num_workers = num_workers

        # Define network layers as class attributes to be used
        self.conv_layer_1 = torch.nn.Sequential(
            # The first block is made of a convolutional layer (3 channels, 28x28 images and a kernel mask of 5),
            torch.nn.Conv2d(3, 28, kernel_size=5),
            # a non linear activation function
            torch.nn.ReLU(),
            # a maximization layer, with mask of size 2
            torch.nn.MaxPool2d(kernel_size=2))

        # A second block is equal to the first, except for input size which is different
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(28, 10, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))

        # A dropout layer, useful to reduce network overfitting
        self.dropout1 = torch.nn.Dropout(0.25)

        # A fully connected layer to reduce dimensionality
        self.fully_connected_1 = torch.nn.Linear(250, 18)

        # Another fine tuning dropout layer to make network fine tune
        self.dropout2 = torch.nn.Dropout(0.08)

        # The final fully connected layer wich output maps to the number of desired classes
        self.fully_connected_2 = torch.nn.Linear(18, 10)

    def load_split_train_test(self, valid_size=.2):
        '''Loads data and builds training/validation dataset with provided split size

        Parameters:
        valid_size (float): the percentage of data reserved to validation

        Returns:
        (torch.utils.data.DataLoader): Training data loader
        (torch.utils.data.DataLoader): Validation data loader
        (torch.utils.data.DataLoader): Test data loader

        '''

        num_workers = self.num_workers

        # Create transforms for data augmentation. Since we don't care wheter numbers are upside-down, we add a horizontal flip,
        # then normalized data to PyTorch defaults
        train_transforms = T.Compose([T.RandomHorizontalFlip(),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Use ImageFolder to load data from main folder. Images are contained in subfolders wich name represents their label. I.e.
        # training
        #   |--> 0
        #   |    |--> image023.png
        #   |    |--> image024.png
        #   |    ...
        #   |--> 1
        #   |    |--> image032.png
        #   |    |--> image0433.png
        #   |    ...
        #   ...
        train_data = datasets.ImageFolder(
            self.train_data_dir, transform=train_transforms)

        # loads image indexes within dataset, then computes split and shuffles images to add randomness
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)

        # extracts indexes for train and validation, then builds a random sampler
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # which is passed to data loader to perform image sampling when loading data
        train_loader = torch.utils.data.DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(
            train_data, sampler=val_sampler, batch_size=self.batch_size, num_workers=num_workers)

        # if testing dataset is defined, we build its data loader as well
        test_loader = None
        if self.test_data_dir is not None:
            test_transforms = T.Compose([T.ToTensor(), T.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            test_data = datasets.ImageFolder(
                self.test_data_dir, transform=test_transforms)
            test_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def prepare_data(self):
        '''Prepares datasets. Called once per training execution
        '''
        self.train_loader, self.val_loader, self.test_loader = self.load_split_train_test()

    def train_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Training set data loader
        '''
        return self.train_loader

    def val_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Validation set data loader
        '''
        return self.val_loader

    def test_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Testing set data loader
        '''
        return DataLoader(MNIST(os.getcwd(), train=False, download=False, transform=transform.ToTensor()), batch_size=128)

    def forward(self, x):
        '''Forward pass, it is equal to PyTorch forward method. Here network computational graph is built

        Parameters:
        x (Tensor): A Tensor containing the input batch of the network

        Returns: 
        An one dimensional Tensor with probability array for each input image
        '''
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.dropout1(x)
        x = torch.relu(self.fully_connected_1(x.view(x.size(0), -1)))
        x = F.leaky_relu(self.dropout2(x))
        return F.softmax(self.fully_connected_2(x), dim=1)

    def configure_optimizers(self):
        '''
        Returns:
        (Optimizer): Adam optimizer tuned wit model parameters
        '''
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        '''Called for every training step, uses NLL Loss to compute training loss, then logs and sends back 
        logs parameter to Trainer to perform backpropagation

        '''

        # Get input and output from batch
        x, labels = batch

        # Compute prediction through the network
        prediction = self.forward(x)

        loss = F.nll_loss(prediction, labels)

        # Logs training loss
        logs = {'train_loss': loss}

        output = {
            # This is required in training to be used by backpropagation
            'loss': loss,
            # This is optional for logging pourposes
            'log': logs
        }

        return output

    def test_step(self, batch, batch_idx):
        '''Called for every testing step, uses NLL Loss to compute testing loss
        '''
        # Get input and output from batch
        x, labels = batch

        # Compute prediction through the network
        prediction = self.forward(x)

        loss = F.nll_loss(prediction, labels)

        # Logs training loss
        logs = {'train_loss': loss}

        output = {
            # This is required in training to be used by backpropagation
            'loss': loss,
            # This is optional for logging pourposes
            'log': logs
        }

        return output

    def validation_step(self, batch, batch_idx):
        ''' Prforms model validation computing cross entropy for predictions and labels
        '''
        x, labels = batch
        prediction = self.forward(x)
        return {
            'val_loss': F.cross_entropy(prediction, labels)
        }

    def validation_epoch_end(self, outputs):
        '''Called after every epoch, stacks validation loss
        '''
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    def validation_end(self, outputs):
        '''Called after validation completes. Stacks all testing loss and computes average.
        '''
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('Average training loss: '+str(avg_loss.item()))
        logs = {'val_loss': avg_loss}
        return {
            'avg_val_loss': avg_loss,
            'log': logs
        }

    def testing_step(self, batch, batch_idx):
        ''' Prforms model testing, computing cross entropy for predictions and labels
        '''
        x, labels = batch
        prediction = self.forward(x)
        return {
            'test_loss': F.cross_entropy(prediction, labels)
        }

    def testing_epoch_end(self, outputs):
        '''Called after every epoch, stacks testing loss
        '''
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def testing_end(self, outputs):
        '''Called after testing completes. Stacks all testing loss and computes average.
        '''
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        print('Average testing loss: '+str(avg_loss.item()))
        logs = {'test_loss': avg_loss}
        return {
            'avg_test_loss': avg_loss,
            'log': logs
        }
