# CS6910: Deep Learning Course Project
#### Vimarsh Sathia (CS17B046), Sarvesh Sakhare (CE17B055)
The aim of this project is to understand and re-create some of the results on self-calibrated convolutions present in the CVPR 2020 paper ["Improving Convolutional Networks with Self-Calibrated Convolutions"](http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf) on the
Imagenette and Imagewoof datasets, which can be accessed [here](https://github.com/fastai/imagenette).  
We try to compare the performance of pretrained **SCNet-50** and **SCNet-101** with that of pretrained **ResNet-50** and **ResNet-101** by adapting the models for the Imagenette and Imagewoof datasets 
using transfer learning. For a given set of fixed images, we also try to visualize the attention maps for each network using **Grad-CAM**. 

- [Report](https://github.com/vimarsh6739/CS6910_2020_Assignments/tree/main/Project)
- [Trained Models](https://drive.google.com/drive/folders/1GCazJEuNHWF0kwyAGZ5qV3WFPfoHH9xU?usp=sharing)

## Contents
### `download.py` 
* Usage `python download.py`
* This script contains some handy code to download the Imagenette and Imagewoof datasets to a local `data/` folder. The resolution of the images is set to the minimum possible (160px)

### `train.py`
* Usage: `python train.py`
* This script trains 4 models for both the Imagenet and the Imagewoof datasets, using transfer learning. This method of training is adopted as both the datasets in question are 
subsets of the Imagenet dataset, on which the original models are trained.

### `visualize.py`
* Usage: `python visualize.py`
* This script visualizes the attention maps generated for all 4 models in the last layer, for 5 handpicked images.
* It uses the implementation of Grad-CAM defined [here](https://github.com/tanjimin/grad-cam-pytorch-light) 

All of the code is also available in the interactive python notebook present in the folder.
