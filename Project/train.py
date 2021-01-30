import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
from SCNet import scnet

def get_loaders(root_dir, transform = {'train':None,'val':None}, batch_size=32, num_workers=2):
    # Get training data
    train_dir = os.path.join(root_dir,'train')
    trainset = ImageFolder(root=train_dir, transform=transform['train'])
    print(f'Length of training dataset = {len(trainset)}')
    trainloader = torch.utils.data.DataLoader(trainset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,
                            drop_last=True)
    # Get validation data
    val_dir = os.path.join(root_dir,'val')
    valset = ImageFolder(root=val_dir, transform=transform['val'])
    print(f'Length of validation dataset = {len(valset)}')
    valloader = torch.utils.data.DataLoader(valset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)
    
    return trainloader, valloader, len(trainset), len(valset)

## Recommended transform for Imagenet images
IMAGE_SIZE=224

TRANSFORMS = {
    'train': T.Compose([
        T.RandomResizedCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train_model(model, data_dir, num_epochs, optimizer, criterion,save_path=None):
  
    train_dl, val_dl, train_len, val_len = get_loaders(data_dir, transform=TRANSFORMS)
    tr_loss = []
    val_loss = []
    tr_acc = []
    val_acc = []
    best_val_acc = 0.0
    since = time.time()
    for epoch in range(num_epochs):
        # Train data
        tr_epoch_loss = 0.0
        tr_epoch_acc = 0.0
        for data in tqdm(train_dl):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            tr_epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data,1)
            tr_epoch_acc += (predicted == labels).sum().item()
        
        tr_epoch_loss /= len(train_dl)
        tr_epoch_acc = 100 * tr_epoch_acc / train_len
        tr_loss.append(tr_epoch_loss)
        tr_acc.append(tr_epoch_acc)

        # Validate data
        model.eval() # Set model in evaluation mode
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        with torch.no_grad(): # disable gradient computation
            for data in tqdm(val_dl):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_epoch_acc += (predicted == labels).sum().item()
        
        val_epoch_loss /= len(val_dl)
        val_epoch_acc = 100 * val_epoch_acc / val_len
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)

        # Print Statistics
        print("\nEpoch: [{}/{} ({:.0f}%)]\n train Loss: {:.6f} Acc: {:.3f}\n Val Loss: {:.6f} Acc: {:.3f}".
                format(epoch+1, num_epochs, 
                        100. * (epoch+1) / num_epochs, 
                        tr_epoch_loss,
                        tr_epoch_acc,
                        val_epoch_loss,
                        val_epoch_acc))

        # Overwrite best model 
        model.train()  #Set model in training mode
        if(val_acc[-1] > best_val_acc ):
            best_val_acc = val_epoch_acc
            if(save_path != None):
                print('!Updating best model to current model!')
                torch.save({'epoch':epoch+1,
                        'tr_loss':tr_epoch_loss,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},save_path)
        print('-' * 10)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_val_acc))

    return tr_loss, tr_acc, val_loss, val_acc

def get_model(data_dir, model_type='resnet50'):
    if model_type == 'scnet50':
        model = scnet.scnet50(pretrained=True)
    elif model_type == 'scnet101':
        model = scnet.scnet101(pretrained=True)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)

    # Freeze all convolution layers
    for param in model.parameters():
        param.requires_grad = False

    # Redefine final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(os.path.join(data_dir,'train')))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    return model, optimizer, criterion

os.makedirs('./models',exist_ok=True)

model_types = ['resnet50', 'scnet50', 'resnet101', 'scnet101']
datasets = {'imagewoof': './data/imagewoof2-160' , 'imagenette':'./data/imagenette2-160'}

for dset in datasets.keys():
    d_path = datasets[dset]
    print('Training Models for ', dset)
    print('-'*20)
    for model_type in model_types:
        print('Training ', model_type)
        print('-'*15)

        model_path = './models/'+model_type+'_'+dset+'.pth'

        model, optimizer, criterion = get_model(data_dir=d_path, model_type=model_type)
        tr_loss, tr_acc, val_loss, val_acc = train_model(model = model,
                            data_dir = d_path,
                            num_epochs=15,
                            optimizer=optimizer,
                            criterion=criterion,
                            save_path=model_path)
        
        csv_path = './models/'+model_type+'_'+dset+'.csv'
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(tr_loss, tr_acc, val_loss, val_acc))
        print('Saved loss values to ', os.path.basename(csv_path))
