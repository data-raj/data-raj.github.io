## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(32,48, 5)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.drop2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv2d(48,64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout(p=0.2)
        self.conv4 = nn.Conv2d(64, 96, 5)
        self.conv4_bn = nn.BatchNorm2d(96)
        self.drop4 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(9600, 2400)
        self.fc1_bn = nn.BatchNorm1d(2400)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(2400, 612)
        self.fc2_bn = nn.BatchNorm2d(612)
        self.fc2_drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(612, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        I.xavier_uniform(self.conv1.weight, gain=I.calculate_gain('relu'))
        I.xavier_uniform(self.conv2.weight, gain=I.calculate_gain('relu'))
        I.xavier_uniform(self.conv3.weight, gain=I.calculate_gain('relu'))
        I.xavier_uniform(self.conv4.weight, gain=I.calculate_gain('relu'))
        I.xavier_uniform(self.fc1.weight, gain=I.calculate_gain('relu'))
        I.xavier_uniform(self.fc2.weight, gain=I.calculate_gain('relu'))
        I.xavier_uniform(self.fc3.weight)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.drop4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
