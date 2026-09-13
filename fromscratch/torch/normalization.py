import torch
import torch.nn as nn
from ..config import RegularConfig
import sys, os

class NormClass(nn.Module):
   def __init__(self, lnum):
         super().__init__()
         self.c = RegularConfig()
         self.module = sys.modules[__name__]
         self.norm = getattr(self.module, self.c.norm_type)(lnum)
   
   def forward(self, X):
      return self.norm(X)

   def save_weights(self):
      self.norm.save_weights()


class LayerNorm(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.lnum = lnum
      self.c = RegularConfig()
      self.gamma = nn.Parameter(torch.ones(self.c.d_model))
      self.beta = nn.Parameter(torch.zeros(self.c.d_model))

   def forward(self, X):
      var = torch.var(X, dim=-1, keepdim=True, unbiased=False)
      mean = torch.mean(X, dim=-1, keepdim=True)
      X = (X - mean) / torch.sqrt(var + self.c.epsn)
      X = self.gamma * X + self.beta
      return X

   def save_weights(self):
      ln_path = os.path.join(self.c.weights_base_path, f"{self.lnum}/layernorm/")
      if not os.path.exists(ln_path):
         os.makedirs(ln_path)
      torch.save(self.gamma.detach(), os.path.join(ln_path, "gamma.pt"))
      torch.save(self.beta.detach(), os.path.join(ln_path, "beta.pt"))


class RMSNorm(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.lnum = lnum
      self.c = RegularConfig()
      self.gamma = nn.Parameter(torch.ones(self.c.d_model))

   def forward(self, X):
      mean = torch.mean(X**2, dim=-1, keepdim=True)
      X = X / torch.sqrt(mean + self.c.epsn)
      X = self.gamma * X 
      return X

   def save_weights(self):
      ln_path = os.path.join(self.c.weights_base_path, f"{self.lnum}/rmsnorm/")
      if not os.path.exists(ln_path):
         os.makedirs(ln_path)
      torch.save(self.gamma.detach(), os.path.join(ln_path, "gamma.pt"))
