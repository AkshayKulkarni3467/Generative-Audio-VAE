import os
import numpy as np


import torch
from torch import nn

x = torch.randn(4,256,64,1)
x = torch.permute(x,[3,0,1,2])
layer = nn.Flatten(start_dim = 0, end_dim = 3)
x = layer(x)
print(x.shape)