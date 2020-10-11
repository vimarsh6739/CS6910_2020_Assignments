import torch, os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from classifiers import Net1, Net2, Net3, Net4
from tqdm import tqdm

# Load data

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

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

########################################################################
# Test dataset performance

def test(testloader, net, criterion):
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
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
    #print('test error: %.3f, accuracy: %d' % (running_loss, running_acc))
    return running_loss, running_acc


model_dir = '../models'
criterion = nn.CrossEntropyLoss()
#Benchmark all params
print("%15s %15s %15s %15s" %("Model Name", "Epochs", "Val accuracy", "Test accuracy"))

model_list = os.listdir(model_dir)
model_list = sorted(model_list)

for m in model_list:
    model_path = model_dir + "/" + m
    checkpoint = torch.load(f=model_path,map_location=torch.device('cpu'))
    epoch = checkpoint['epoch'] + 1
    if (m.split('_')[0]=='Net4') :
        if (m.split('_')[0] == 'Net1'):
            net = Net1()
        else:
            if(m.split('_')[0] == 'Net2'):
                net = Net2()
            else:
                if(m.split('_')[0] == 'Net3'):
                    net = Net3()
                else:
                    net = Net4()
        
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        v_loss, v_acc = test(valloader,net, criterion)
        t_loss, t_acc = test(testloader,net,criterion)
        print("%15s %15s %15s %15s" %(m.split('_')[0], str(epoch), str(v_acc), str(t_acc)))



        

