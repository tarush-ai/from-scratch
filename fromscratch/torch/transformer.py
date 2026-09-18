import torch
import torch.nn as nn
import os, sys
from ..config import RegularConfig
from .attention import AttentionClassTrain
from .ffn import FFNClassTrain
from .normalization import NormClass

class Transformer(nn.Module):
   def __init__(self, tokenizer):
      super().__init__()
      self.tokenizer = tokenizer
      self.c = RegularConfig()
      self.module = sys.modules[__name__]
      self.transformer = getattr(self.module, self.c.transformer_type, None)(self.tokenizer)
   
   def forward(self, X):
      return self.transformer(X)

   def save_weights(self):
      self.transformer.save_weights()
   
class StandardPreNormTransformer(nn.Module):
   def __init__(self, tokenizer):
      super().__init__()
      self.c = RegularConfig()
      mask = torch.triu(torch.full((self.c.max_seq_length, self.c.max_seq_length),float('-inf')),diagonal=1)
      self.register_buffer("mask", mask, persistent=False)
      self.tokenizer = tokenizer
      self.embed = nn.Embedding(self.c.vocab_length, self.c.d_model)
      self.pos = nn.Embedding(self.c.max_seq_len, self.c.d_model)
      nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
      nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
      self.transformerblocks = nn.ModuleList([PreNormTransformerBlock(i) for i in range(self.c.num_blocks)])
      self.final_ln = NormClass(self.c.num_blocks, 1)
   
   def forward(self, X):
      embedded = self.embed(X)
      positional = self.pos(torch.arange(X.shape[1], device=X.device))
      X = embedded + positional
      for block in self.transformerblocks:
         X = block(X, self.mask)
      X = self.final_ln(X)
      return X @ self.embed.weight.T

   def save_weights(self):
      embeddings_path = os.path.join(self.c.weights_base_path, f"embeddings/")
      os.makedirs(embeddings_path, exist_ok=True)
      torch.save(self.embed.weight.detach().contiguous(), os.path.join(embeddings_path, "embed.pt"))
      torch.save(self.pos.weight.detach().contiguous(), os.path.join(embeddings_path, "pos.pt"))
      for block in self.transformerblocks:
         block.save_weights()
      self.final_ln.save_weights()


class PreNormTransformerBlock(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.ln1 = NormClass(lnum,1)
      self.attn = AttentionClassTrain(lnum)
      self.ln2 = NormClass(lnum,2)
      self.ffn = FFNClassTrain(lnum)

   def forward(self, X, mask):
      X = X + self.attn(self.ln1(X), mask)
      X = X + self.ffn(self.ln2(X))
      return X

   def save_weights(self):
      self.ln1.save_weights()
      self.attn.save_weights()
      self.ln2.save_weights()
      self.ffn.save_weights()
      