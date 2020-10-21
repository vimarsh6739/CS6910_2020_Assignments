import torch
from classifiers import Net4

########################################################################
# Import best performing model
model_path = '../models/Net4_30.pth'

if torch.cuda.is_available():
    checkpoint = torch.load(f=model_path)
else:
    checkpoint = torch.load(f=model_path, map_location=torch.device('cpu'))

net = Net4()
if torch.cuda.is_available():
    net = net.cuda()

net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
epoch = checkpoint['epoch'] + 1

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

########################################################################
# Load test images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

test_data_dir = '../dataset/test' 
testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)
evalloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=2, shuffle=False)

########################################################################
# Define hooks and compute activations - Begin Part 2.2.1
activation = {}
def get_activation(name):
    def hook_fn(model, inp, out):
        activation[name] = out.detach()
    return hook_fn

net.conv1.register_forward_hook(get_activation('conv1'))
net.conv2.register_forward_hook(get_activation('conv2'))
net.conv_block3[0].register_forward_hook(get_activation('conv3'))
net.conv_block4[0].register_forward_hook(get_activation('conv4'))
net.conv_block5[0].register_forward_hook(get_activation('conv5'))

#Compute and store activations for the filter in the specified layer
def find_filtermaps(net, testloader, filter_id, layer_name):
    act = None
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            output = net(images)
            act_b = activation[layer_name][:,filter_id]
            if act == None:
                act = act_b
            else:
                act = torch.cat((act, act_b),dim=0)
    return act

from utils import *
import matplotlib.pyplot as plt

classes, _  = find_classes(test_data_dir)

# Plot all maximally activated filtermaps
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
filter_idx = {'conv1' : [2,13], 'conv2' : [11,23], 'conv3' : [9,49], 'conv4' : [54,98], 'conv5':[5,171]}

for l in layer_names:
    for f in filter_idx[l]:
        act = find_filtermaps(net, evalloader, f, l)
        act_idx = torch.argsort(torch.Tensor([act[i].norm() for i in range(act.shape[0])]),descending=False)
        fig,axes = plt.subplots(nrows=2, ncols=5, figsize=(15,6))
        for i in range(5):
            imshow((act[act_idx[i]])[None,:,:],axes[0,i])
            img, label = evalloader.dataset[act_idx[i]]
            imshow(img,axes[1,i],classes[label])
        plot_name = l + '_filter' + str(f)
        plt.suptitle(plot_name)
        #plt.savefig('../images/Filter_'+plot_name+'.png')
        plt.show()

########################################################################
# Part 2.2.2 - Switching off filter weights and visualize

acc_on = find_accuracymap(net,evalloader)

#Switch off filter weights
net.conv1.weight[2]=0
net.conv1.weight[13]=0

net.conv2.weight[11]=0
net.conv2.weight[23]=0

net.conv_block3[0].weight[9]=0
net.conv_block3[0].weight[49]=0

net.conv_block4[0].weight[54]=0
net.conv_block4[0].weight[98]=0

net.conv_block5[0].weight[5]=0
net.conv_block5[0].weight[171]=0

acc_off = find_accuracymap(net,evalloader)

cnt=0
img_idx = []
for i in range(len(evalloader.dataset)):
    if acc_on[i]==1 and acc_off[i]==0:
        cnt+=1
        img_idx.append(i)

## Classwise count of misclassified images
n_class=5
classwise_error = list(0. for i in range(n_class))
for idx in img_idx:
    classwise_error[evalloader.dataset[idx][1]]+=1
    
for i in range(n_class):
    print("%15s : %d images"%(classes[i],classwise_error[i]))

## Display 5 random misclassified images
subset_idx = np.random.randint(cnt,size=5)
fig,axes = plt.subplots(nrows=1, ncols=5, figsize=(15,3))
i=0
for index in subset_idx:
    image, label = evalloader.dataset[img_idx[index]]
    outputs = net(image[None,:,:,:])
    _, predicted = torch.max(outputs, 1)
    tit = classes[label] + '==>'+ classes[predicted]
    imshow(image, axes[i],tit)
    i+=1
#plt.savefig('../images/misclassified_switchoff.png', bbox_inches='tight')
plt.show()