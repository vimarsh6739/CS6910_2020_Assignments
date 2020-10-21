import torch, os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

#########################################################################
# get details of classes and class to index mapping in a directory
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

########################################################################
# load-model - returns DataLoader for image dataset
def get_loader(data_dir, transform=None,batch_size=1,shuffle=False):
    dataset = torchvision.datasets.ImageFolder(root= data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=2)
    return dataloader


########################################################################
# display image tensor with 3 channels
def imshow(img, ax, title=None):
     
    npimg = img.numpy()
  
    #plot the image as numpy array
    ax.axis("off")
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    if title!=None:
        ax.set_title(title)

########################################################################
# class-wise accuracy
def classwise_test(testloader, net, classes):

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
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_class):
        print('Accuracy of %10s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

########################################################################
# bitwise accuracy mapping for all images in testloader
def find_accuracymap(net, testloader):
    c = None
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs,1)
            c_batch = (predicted==labels).squeeze().int()
            if c==None:
                c=c_batch
            else:
                c=torch.cat((c,c_batch),dim=0)
    return c
