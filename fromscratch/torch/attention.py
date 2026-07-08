import torch
import torch.nn as nn
from config import RegularConfigTorch
import os, sys

class AttentionClass:
   def __init__(self):
      self.config = RegularConfigTorch()
      self.module = sys.modules[__name__]
      self.attn = getattr(self.module, self.config.attention_type, None)()

   def forward(self, X):
      return self.attn(X)

class MultiHeadAttention(nn.Module):
   def __init__(self, *args, **kwargs):
      super().__init__()
      self.attnheads = nn.ModuleList([AttentionHead() for head in self.config.num_heads])
      self.MHA = nn.Linear(self.config.d_model, self.config.d_model, bias=False)

   def forward(self, X):
      headoutputs = [head(X) for head in self.attnheads]
      MHA = torch.cat(headoutputs, dim=-1)
      return self.Wo(MHA)

class AttentionHead(nn.Module):
   def __init__(self, *args, **kwargs):
      super().__init__()
      self.Wq = nn.Linear(self.config.d_model, self.config.d_head, bias=False)
      self.Wk = nn.Linear(self.config.d_model, self.config.d_head, bias=False)
      self.Wv = nn.Linear(self.config.d_model, self.config.d_head, bias=False)
   
   def forward(self, X):
      Q = X @ self.Wq
      K = X @ self.Wk
      V = X @ self.Wv

      scores = Q @ K.T
      scores /= (self.config.d_head ** 0.5)
      mask = torch.tril(torch.ones(X.shape[0], X.shape[0], device=X.device))
      scores = scores.masked_fill(mask == 0, float('-inf'))
      attention = torch.softmax(scores, dim=-1)
      return attention @ V

