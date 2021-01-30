import os
import torch
from torchvision.datasets.utils import download_url
import tarfile

imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
imagewoof_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"

# Download dataset tars
download_url(imagenette_url,'.')
download_url(imagewoof_url, '.')

# Extract imagewoof
with tarfile.open('./imagewoof2-160.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')

# Extract imagenette
with tarfile.open('./imagenette2-160.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')

# Look into the data directory contents
data_dirs = ['./data/imagewoof2-160','./data/imagenette2-160']
for dset in data_dirs:
    print('|=====>',dset)
    print(os.listdir(dset))
    classes = os.listdir(dset + "/train")
    print(classes)