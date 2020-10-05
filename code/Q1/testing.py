import torch.nn as nn
import torch
import torch.nn.functional as F

x = torch.randn((32, 64, 8, 8))
print(x.view(x.shape[0],-1).shape)
#print('After flatten : ', x.view(x.shape[0],-1).shape)