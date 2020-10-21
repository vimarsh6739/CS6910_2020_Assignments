# Defining neural net base models here

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                                                stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                                stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                                                stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=5)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2,self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, 1, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4*4*64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(self.conv_block3(x))
        x = self.pool(self.conv_block4(x))

        x = x.view(x.shape[0],-1)
        x = self.linear_block(x)

        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3,self).__init__()

        self.conv1 = nn.Conv2d(3,16,4,2,1)
        self.conv2 = nn.Conv2d(16,32,3,1,1)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 256, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1,1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear_block = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2*2*256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(self.conv_block3(x))
        x = self.pool(self.conv_block4(x))

        x = x.view(x.shape[0],-1)
        x = self.linear_block(x)
        
        return x

class Net4(nn.Module):
    def __init__(self):
        super(Net4,self).__init__()

        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.conv2 = nn.Conv2d(16,32,3,1,1)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear_block = nn.Sequential(
            nn.Linear(2*2*256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,5)
        )


    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(self.conv_block3(x))
        x = self.pool(self.conv_block4(x))
        x = self.pool(self.conv_block5(x))

        x = x.view(x.shape[0],-1)
        x = self.linear_block(x)
        
        return x


