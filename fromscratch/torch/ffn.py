import torch
import torch.nn as nn
import os, sys
from ..config import RegularConfig

class FFNClass(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.module = sys.modules[__name__]
      self.c = RegularConfig()
      self.ffn = getattr(self.module, self.c.ffn_type)(lnum)
   
   def forward(self, X):
      return self.ffn(X)

   def save_weights(self):
      self.ffn.save_weights()

# Training 
class RegularFFN(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.lnum = lnum
      self.upproj = nn.Linear(self.c.d_model, self.c.d_ff, bias=False)
      self.activ = nn.ReLU()
      self.downproj = nn.Linear(self.c.d_ff, self.c.d_model, bias=False)

   def forward(self, X):
      X = self.upproj(X)
      X = self.activ(X)
      X = self.downproj(X)
      return X

   def save_weights(self):
      # nuclear detach option; do not use for intermediate saving 
      ffn_path = os.path.join(self.c.weights_base_path, f"{self.lnum}/ffn/")
      if not os.path.exists(ffn_path):
         os.makedirs(ffn_path)
      torch.save(self.upproj.weight.T.detach().contiguous(), os.path.join(ffn_path, "upproj.pt"))
      torch.save(self.downproj.weight.T.detach().contiguous(), os.path.join(ffn_path, "downproj.pt"))



class FFNInference(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      ffn_path = os.path.join(self.c.weights_base_path, f"{lnum}/ffn/")
      self.upproj_weights = torch.load(os.path.join(ffn_path, "upproj.pt"))
      self.downproj_weights = torch.load(os.path.join(ffn_path, "downproj.pt"))
      self.activ = nn.ReLU()

   def forward(self, X):
      X = X @ self.upproj_weights # (B,s,d) @ (d,d_ff)
      X = self.activ(X)
      X = X @ self.downproj_weights # (B,s,d_ff) @ (d_ff, d)
      return X

   def save_weights(self):
      pass
