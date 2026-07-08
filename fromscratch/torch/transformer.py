import torch
import torch.nn as nn
import os, sys
from config import RegularConfigTorch
from embeddings import EmbeddingTorch
from attention import AttentionClass
from ffn import FFNClass
from normalization import NormalizationClass

class Transformer(nn.Module):
   def __init__(self, tokenizer):
      super().__init__()
      self.tokenizer = tokenizer
      self.config = RegularConfigTorch()
      self.module = sys.modules[__name__]
      self.transformer = getattr(self.module, self.config.transformer_type, None)(self.tokenizer)
   
   def forward(self, X):
      return self.transformer(X)
   
class PreNormTransformer(nn.Module):
   def __init__(self, tokenizer):
      super().__init__()
      self.tokenizer = tokenizer
      self.embedding = EmbeddingTorch(self.tokenizer)
      self.config = RegularConfigTorch()
      self.transformerblocks = nn.ModuleList([PreNormTransformerBlock() for block in self.config.num_blocks])
   
   def forward(self, X):
      embedded = self.embedding.embed(X)
      positional = self.embedding.positional(X)
      X = embedded + positional
      for block in self.trasnformerblocks:
         X = block(X)
      return X

class PreNormTransformerNoEndNorm(PreNormTransformer, nn.Module):
   def __init__(self, tokenizer):
      super().__init__(tokenizer)
      self.Wo = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
   
   def forward(self, X):
      X = PreNormTransformer(X)
      X = self.Wo(X)
      return X

class PreNormTransformerEndNorm(PreNormTransformer, nn.Module):
   def __init__(self, tokenizer):
      super().__init__(tokenizer)
      self.endnorm = NormalizationClass()
      self.Wo = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

   def forward(self, X):
      X = PreNormTransformer(X)
      X = self.endnorm(X)
      X = self.Wo(X)
      return X
   
class PreNormTransformerBlock(nn.Module):
   def __init__(self):
      super().__init__()
      self.norm1 = NormalizationClass()
      self.mha = AttentionClass()
      self.norm2 = NormalizationClass()
      self.ffn = FFNClass()

   def foward(self, X):
      saver = X
      X = self.mha(X)
      X = self.norm1(X)
      X += saver
      saver2 = X
      X = self.ffn(X)
      X = self.norm2(X)
      X += saver2
      return X
      # X2 = X + LN(MHA(X)); return X=X2 + LN2(FFN(X2))

class PostNormTransformer(nn.Module): #looks the same as PreNormTransformer with one tweak
   def __init__(self, tokenizer):
      super().__init__()
      self.tokenizer = tokenizer
      self.embedding = EmbeddingTorch(self.tokenizer)
      self.config = RegularConfigTorch()
      self.transformerblocks = nn.ModuleList([PostNormTransformerBlock() for block in self.config.num_blocks])
   
   def forward(self, X):
      embedded = self.embedding.embed(X)
      positional = self.embedding.positional(X)
      X = embedded + positional
      for block in self.trasnformerblocks:
         X = block(X)
      return X

class PostNormTransformerNoEndNorm(PostNormTransformer, nn.Module):
   def __init__(self, tokenizer):
      super().__init__(tokenizer)
      self.Wo = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
   
   def forward(self, X):
      X = PreNormTransformer(X)
      X = self.Wo(X)
      return X

class PostNormTransformerEndNorm(PostNormTransformer, nn.Module):
   def __init__(self, tokenizer):
      super().__init__(tokenizer)
      self.endnorm = NormalizationClass()
      self.Wo = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

   def forward(self, X):
      X = PreNormTransformer(X)
      X = self.endnorm(X)
      X = self.Wo(X)
      return X
   
class PostNormTransformerBlock(nn.Module):
   def __init__(self):
      super().__init__()
      self.norm1 = NormalizationClass()
      self.mha = AttentionClass()
      self.norm2 = NormalizationClass()
      self.ffn = FFNClass()

   def foward(self, X):
      saver = X
      X = self.mha(X)
      X += saver
      X = self.norm1(X)
      
      saver2 = X
      X = self.ffn(X)
      X += saver2
      X = self.norm2(X)

      return X
      # X2 = LN(X+MHA(X)); return X=LN2(X2+FFN(X2))
