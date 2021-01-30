import torch, os 
from torchvision import models
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from grad_cam_pytorch.grad_cam import grad_cam
from SCNet import scnet

def visualize_gradCAM(img_paths, dsets, dset_dict, model_types, model_base_path, save_path=None):
  
    fig, axs = plt.subplots(len(model_types),len(dsets),figsize=(15,12))
    # fig.suptitle('Attention Map Visualization for all models')

    for j, dset in enumerate(dsets,0):
        for i, model_type in enumerate(model_types,0):
            model_path = model_base_path + '/' + model_type + '_' + dset + '.pth'
            
            # Load torch model
            if model_type == 'scnet50':
                model = scnet.scnet50(pretrained=False)
            elif model_type == 'scnet101':
                model = scnet.scnet101(pretrained=False)
            elif model_type == 'resnet101':
                model = models.resnet101(pretrained=False)
            else:
                model = models.resnet50(pretrained=False)
            
            model.eval()
            
            # Set the heatmap layer as the final conv layer
            if model_type == 'scnet50' or model_type=='scnet101':
                heatmap_layer = model.layer4[2].k1
            else:
                heatmap_layer = model.layer4[2].conv3

            # Select image to show
            img = Image.open(img_paths[j])
            input_tensor = T.Compose([T.Resize(256),
                                        T.CenterCrop(224),
                                        T.ToTensor(),
                                        T.Normalize(mean = [0.485, 0.456, 0.406], 
                                                    std = [0.229, 0.224, 0.225])])(img)

            final_img = grad_cam(model, input_tensor, heatmap_layer)
            axs[i,j].imshow(final_img)
            axs[i,j].tick_params(axis='both', which='both', length=0)
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
    
    # Label y axis
    for j, model_type in enumerate(model_types,0):
        axs[j,0].set_ylabel(model_type, fontsize=18)
    
    # Set no ticks
    plt.axis('off')
    if save_path!=None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

img_paths = ['./data/imagewoof2-160/train/n02086240/n02086240_2019.JPEG',
          './data/imagewoof2-160/train/n02115641/n02115641_10798.JPEG',
          './data/imagenette2-160/train/n02979186/n02979186_1043.JPEG',
          './data/imagenette2-160/train/n03417042/n03417042_12244.JPEG',
          './data/imagenette2-160/train/n03028079/n03028079_118756.JPEG']
dsets = ['imagewoof','imagewoof','imagenette','imagenette','imagenette']
datasets = {'imagewoof': './data/imagewoof2-160' , 'imagenette':'./data/imagenette2-160'}
model_types = ['resnet50', 'scnet50', 'resnet101', 'scnet101']

os.makedirs(os.path.join('./drive/MyDrive/CS6910_Project','images'),exist_ok=True)

visualize_gradCAM(img_paths, 
                  dsets, 
                  datasets,
                  model_types, 
                  model_base_path = './drive/MyDrive/CS6910_Project/models',
                  save_path='./drive/MyDrive/CS6910_Project/images/gradCAM_viz.png')