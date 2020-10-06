import torch, os
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

########################################################################
# Load and transform datasets

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])
print(transform)


train_data_dir = '../dataset/train' # put path of training dataset
val_data_dir = '../dataset/val' # put path of validation dataset
test_data_dir = '../dataset/test' # put path of test dataset

trainset = torchvision.datasets.ImageFolder(root= train_data_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.ImageFolder(root= val_data_dir, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=32,
                                         shuffle=False, num_workers=2)

testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

#########################################################################
# get details of classes and class to index mapping in a directory
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def classwise_test(testloader, net):
########################################################################
# class-wise accuracy

    classes, _ = find_classes(train_data_dir)
    n_class = len(classes) # number of classes

    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_class):
        print('Accuracy of %10s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

########################################################################
# Train the network for 1 epoch
# ^^^^^^^^^^^^^^^^^^^^

def train(epoch, trainloader, net, optimizer, criterion):
    running_loss = 0.0
    correct=0
    total=0
    for i, data in enumerate(tqdm(trainloader), 0):
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

        # calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    running_loss = running_loss / (len(trainloader))
    running_acc = (100 * correct / total)
    print('epoch %d training loss: %.3f, acc: %d' %
            (epoch + 1, running_loss, running_acc))
    return running_loss, running_acc

########################################################################
# Test dataset performance

def test(testloader, net, criterion):
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            loss = criterion(outputs, labels)

            #calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    running_loss = running_loss / (len(testloader))
    running_acc = (100 * correct / total)
    print('test error: %.3f, accuracy: %d' % (running_loss, running_acc))
    return running_loss, running_acc

########################################################################
# Plot loss and accuracy curves for model

def plotCurve(model_name, training_loss, training_acc, validation_loss, validation_acc):

    # Plot loss curve and accuracy curve in 2 subplots for 
    # both training and validation and save to folder
    ar = 1 + np.arange(len(training_loss))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    ax1.plot(ar, training_loss, label = "training loss")
    ax1.plot(ar, validation_loss, label = "validation loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("error")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title("Loss Curve %s " %(model_name))

    ax2.plot(ar, training_acc, label = "training accuracy")
    ax2.plot(ar, validation_acc, label = "validation accuracy")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("Accuracy Curve %s " %(model_name))

    plt.savefig('../images/'+model_name+'.png')

########################################################################
# Train the entire model and save weights every 10 epochs

def trainModel(model_name, net, num_epochs, optimizer, criterion):
    
    print("Started Training %s" % (model_name))
    
    num_params = np.sum([p.nelement() for p in net.parameters()])
    print(num_params, ' parameters')
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('epoch ', epoch + 1)
        epoch_train_loss, epoch_train_acc = train(epoch, trainloader, net, optimizer, criterion)
        epoch_val_loss, epoch_val_acc = test(valloader, net, criterion)
        classwise_test(valloader, net)

        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        
        # Checkpoint model every 10 epochs
        if ((epoch + 1) % 5 == 0) :
            model_path = '../models/'+str(model_name)+'_'+str(epoch+1)+'.pth'
            torch.save({'epoch':epoch,
                        'model_state_dict':net.state_dict() }, model_path)
    
    # Plot and save loss curve and accuracy curve

    print("Saving loss curves")
    plotCurve(model_name, train_loss, train_acc, val_loss, val_acc)
    
    print('Performing Test')
    test(testloader, net, criterion)
    classwise_test(testloader, net)

    print('Finished Training %s' % (model_name))


os.makedirs('../models', exist_ok=True)
os.makedirs('../images', exist_ok=True)

num_epochs = 50                     # desired number of training epochs.
learning_rate = 0.001               #desired learning rate
criterion = nn.CrossEntropyLoss()   #loss function

########################################################################
# Import and run all neural nets

from classifiers import Net1, Net2, Net3, Net4

models = [Net1(), Net2(), Net3(), Net4()]
optimizers = [optim.SGD(p.parameters(),lr=learning_rate,momentum=0.9, weight_decay=5e-4) for p in models]

for i in range(len(models)):
    net = models[i]
    optimizer = optimizers[i]

    model_name = 'Net' + str(i+1)
    print( 'On %s' %(model_name) )

    num_params = np.sum([p.nelement() for p in net.parameters()])
    print(num_params, ' parameters')
    
    if torch.cuda.is_available():
        net = net.cuda()
    
    trainModel(model_name, net, num_epochs, optimizer, criterion)
