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

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
########################################################################
# Load test images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

test_data_dir = '../dataset/test' 
testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, num_workers=2, shuffle=False)

classes, _ = find_classes(test_data_dir)

acc_map = find_accuracymap(net, testloader)

## Count misclassified images and index of them
cnt=0
img_idx = []
for i in range(len(testloader.dataset)):
    if acc_map[i]==0:
        cnt+=1
        img_idx.append(i)

import matplotlib.pyplot as plt
## Display some 5 misclassified images
subset_idx = np.random.randint(cnt,size=5)
fig,axes = plt.subplots(nrows=1, ncols=5, figsize=(15,3))
i=0
for index in subset_idx:
    image, label = testloader.dataset[img_idx[index]]
    outputs = net(image[None,:,:,:])
    _, predicted = torch.max(outputs, 1)
    tit = 'expected '+classes[label] + ' not '+ classes[predicted]
    imshow(image, axes[i],tit)
    i+=1
#plt.savefig('../images/misclassified_imgs.png')
plt.show()