# mount the google drive to Colab
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

# Clone https://drive.google.com/drive/folders/1piu6jMrcu__BMJ1Wknm7aFkRyYQS5huV?usp=sharing to your google drive
# put the path of the data split assigned to you here
datasetpath = "/content/drive/My Drive/CS6910_PA1/2"  # e.g., /content/drive/My Drive/CS6910_PA1/1

# Imports
import torch, os, os.path as osp
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

# API to load the saved dataset file
def get_dataloader_from_pth(path, batch_size=4):
    print('loading {}'.format(path))
    contents = torch.load(path)
    print('data split: {}, classes: {}, {} data points'.format(contents['split'], contents['classes'], len(contents['x'])))

    # create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(contents['x'], contents['y'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    
    return dataloader, contents['classes']

# file paths
train_pth = osp.join(datasetpath, 'train.pth')
val_pth = osp.join(datasetpath, 'val.pth')
test_pth = osp.join(datasetpath, 'test.pth')

# create dataloaders
trainloader, classes = get_dataloader_from_pth(train_pth, batch_size=32)
valloader, _ = get_dataloader_from_pth(val_pth, batch_size=32)
testloader, _ = get_dataloader_from_pth(test_pth, batch_size=32)

########################################################################
# Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

# Loss function, learning rate and decays
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

decays = [0.0005, 0.005, 0.05, 0.1 ]
nets = [Net4() for i in range(len(decays))]
optimizers = [optim.SGD(nets[i].parameters(), lr=learning_rate, momentum=0.9, weight_decay=wd) for i,wd in enumerate(decays,0)]

if torch.cuda.is_available():
    for net in nets:
        net = net.cuda()

def train(net, epoch, trainloader, optimizer, criterion):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('\nepoch %d training loss: %.3f' %
            (epoch + 1, running_loss / (len(trainloader))))

########################################################################
# Let us look at how the network performs on the test dataset.

def test(testloader, model, criterion, set_name):
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    running_loss /= len(testloader)
    accuracy = 100 * correct / total
    #print('Accuracy of the network on the %s images: %d %%' % (set_name, accuracy))
    return accuracy, running_loss

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

########################################################################
# Plot loss and accuracy curves for model

def plotCurve(plot_name, training_loss, training_acc, validation_loss, validation_acc, wd):

    # Plot loss curve and accuracy curve in 2 subplots for 
    # both training and validation
    ar = 1 + np.arange(len(training_loss))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    ax1.plot(ar, training_loss, label = 'train loss')
    ax1.plot(ar, validation_loss, label = 'test loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('error')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title('Loss Curve for weight decay = %.4f ' %(wd))

    ax2.plot(ar, training_acc, label = 'training accuracy')
    ax2.plot(ar, validation_acc, label = 'test accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title('Accuracy plot, weight decay = %.4f ' %(wd))

    plt.savefig('images/'+plot_name+'.png',bbox_inches='tight')

#train all models for 25 epochs
num_epochs = 25
for i, net in enumerate(nets,0):
    print('Start Training model with lr=%f, decay=%f'%(learning_rate, decays[i]))
    trloss, tracc, tstloss, tstacc = [],[],[],[] 
    for epoch in range(num_epochs):
        print('Epoch: ', epoch+1)
        train(net, epoch, trainloader, optimizers[i], criterion)
        acc_train, loss_train = test(trainloader, net, criterion, set_name='train')
        acc_test, loss_test = test(testloader, net, criterion, set_name='test')
        trloss.append(loss_train)
        tracc.append(acc_train)
        tstloss.append(loss_test)
        tstacc.append(acc_test)
    plot_name = 'decay' + str(decays[i])
    plotCurve(plot_name, trloss, tracc, tstloss, tstacc, decays[i])
print('Finished training all nets')

from tabulate import tabulate
rows = []
for i, net in enumerate(nets, 0):
    acc_train, loss_train = test(trainloader, net, criterion, set_name='train')
    acc_test, loss_test = test(testloader, net, criterion, set_name='test')
    rows.append([decays[i], loss_train, acc_train, acc_test])

t = tabulate(rows, headers=['Weight decay', 'Train loss', 'Train accuracy(%)', 'Test accuracy(%)'], tablefmt='orgtbl')
print(t)
