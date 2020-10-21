# code for Part-B 2.1 - Occlusion sensitivity

import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

########################################################################
# Finds and returns probablity heatmap for grayscaling heatmap
def occlusion(net, img, label, kernel_size):
    
    width, height = img.shape[-1], img.shape[-2]
    
    heatmap = torch.zeros((height-kernel_size+1, width-kernel_size+1))
    
    for i in tqdm(range(height-kernel_size+1)):
        bottom = i+kernel_size
        for j in range(width-kernel_size+1):
            right = j+kernel_size
            
            #grayscale img tensor in required window
            img_clone = img.clone().detach()
            
            img_clone[:, :, i:bottom, j:right] = 0.5

            #run inference on modified image
            output = net(img_clone)

            #convert loss to probability
            output = F.softmax(output, dim=1)

            heatmap[i, j] = output[0][label]
    
    return heatmap


from utils import *
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########################################################################
# Select images for occlusion
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

test_data_dir = '../dataset/test' 
testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)

subset_indices = np.random.randint(2500,size=5) #select 5 images
subset = torch.utils.data.Subset(testset, subset_indices)

evalloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
evaliter = iter(evalloader)
classes, _ = find_classes(test_data_dir)

########################################################################
# Plot heatmaps as subplots
kernel_size = [10, 20]

cols = ['Kernel size = {}'.format(k) for k in kernel_size]
fig_fname = '../images/Occlusion_heatmaps.png'
fig, axes = plt.subplots(nrows = len(subset), ncols=3, figsize=(9,3*len(subset)))

for i in range(len(kernel_size)):
    axes[0,(i+1)].set_title(cols[i])
    
idx=0
with torch.no_grad():   
    for data in evalloader:
        image, label = data
        
        imshow(image[0],axes[idx,0],classes[label])
        
        idy=1
        for k in kernel_size:
            
            heatmap = occlusion(net, image, label, k)
            sns.heatmap(heatmap, vmin=0, vmax=1,xticklabels=False,yticklabels=False,
                        ax=axes[idx,idy], square=True, cmap='coolwarm')
            idy+=1
            
        idx+=1  

#plt.savefig(fig_fname, bbox_inches='tight')
plt.show()