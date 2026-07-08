import torch
import torch.nn as nn
import os, sys
from config import RegularConfigTorch

class FFNClass(nn.Module):
   def __init__(self):
      super().__init__()
      self.module = sys.modules[__name__]
      self.config = RegularConfigTorch()
      self.ffn = getattr(self.module, self.config.ffn_type, None)()
   
   def forward(self, X):
      return self.ffn(X)

class RegularFFN(nn.Module):
   def __init__(self):
      super().__init__()
      self.config = RegularConfigTorch()
      self.layers = nn.ModuleList([Layer() for i in range(self.config.num_layers-1)])
      self.layers.append(Layer(False))
   
   def forward(self, X):
      for i in range(len(self.layers)):
         X = self.layers[i](X)
      return X

class Layer(nn.Module):
   def __init__(self, activ=True):
      super().__init__()
      self.linear = nn.Linear() #actually, I don't think I can do this... or I need some dimensional stuff to be done
