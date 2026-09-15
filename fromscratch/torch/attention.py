import torch
import torch.nn as nn
from ..config import RegularConfig
import os, sys

class AttentionClassTrain(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.module = sys.modules[__name__]
      self.attn = getattr(self.module, self.c.attention_type)(lnum)

   def forward(self, X, mask=None):
      return self.attn(X, mask)

   def save_weights(self):
      self.attn.save_weights()

class AttentionClassInference(nn.Module):
   def __init__(self, lnum):
         super().__init__()
         self.c = RegularConfig()
         self.module = sys.modules[__name__]
         self.attn = getattr(self.module, self.c.attention_type)(lnum)
   
   def forward(self, X, mask=None, KV=None, prevlen=None):
      return self.attn(X, mask, KV, prevlen)

# Base MultiHeadAttention: 
# Training
class OptimizedMHATrain(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.lnum = lnum
      self.h, self.d_k = self.c.num_heads, self.c.d_model // self.c.num_heads
      self.Wqkv = nn.Linear(self.c.d_model, 3*self.c.d_model, bias=False)
      self.Wo = nn.Linear(self.c.d_model, self.c.d_model, bias=False)

   def forward(self, X, mask):
      B, s, d = tuple(X.shape)
      Q, K, V = self.Wqkv(X).split(d, dim=-1)
      Q, K, V = [i.contiguous().view(B,s,self.h,self.d_k).permute(0,2,1,3) for i in (Q, K, V)]
      scores = (Q @ K.permute(0,1,3,2)) / (self.d_k ** 0.5)
      scores = scores + mask[:s,:s]
      scores = scores.softmax(dim=-1)
      A = (scores @ V).permute(0,2,1,3).contiguous().view(B,s,d) # (B,h,s,s) @ (B,h,s,d_k) = (B,h,s,d_k) -> (B,s,h,d_k) -> (B,s,d)
      return self.Wo(A)

   def save_weights(self):
      # nuclear: detaches from autograd, please only use when you want to save weights, not for intermediate weight caching. 
      attn_path = os.path.join(self.c.weights_base_path, f"{self.lnum}/attn/")
      os.makedirs(attn_path, exist_ok=True)
      torch.save(self.Wqkv.weight.T.detach().contiguous(), os.path.join(attn_path, "Wqkv.pt"))
      torch.save(self.Wo.weight.T.detach().contiguous(), os.path.join(attn_path, "Wo.pt"))


class StandardMHATrain(nn.Module):
   # LEGACY: Unoptimized, disjoint Q, K, V instantiation, device brittle
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.lnum = lnum
      self.Wq = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.Wk = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.Wv = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.Wo = nn.Linear(self.c.d_model, self.c.d_model, bias=False)

   def forward(self, X, mask):
      assert tuple(X.shape) == (self.c.batch_size, self.c.max_seq_len, self.c.d_model)
      d_k = self.c.d_model // self.c.num_heads # note: for single head attention transformers, d_k = d_model
      # Take a tensor of (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k) for matmuls
      Q = self.Wq(X).reshape(self.c.batch_size, self.c.max_seq_len, self.c.num_heads, d_k).transpose(1,2)
      K = self.Wk(X).reshape(self.c.batch_size, self.c.max_seq_len, self.c.num_heads, d_k).transpose(1,2)
      V = self.Wv(X).reshape(self.c.batch_size, self.c.max_seq_len, self.c.num_heads, d_k).transpose(1,2)

      scores = Q @ K.permute(0,1,3,2) # (B,h,s,d_k) @ (B,h,d_k, s) = (B,h,s,s) 
      scores = scores / (d_k ** 0.5)
      scores = scores + mask
      scores = scores.softmax(dim=-1) # row-wise softmax
      A = scores @ V # (B,h,s,s) @ (B,h,s,d_k)
      A = A.permute(0,2,1,3).reshape(self.c.batch_size, self.c.max_seq_len, self.c.d_model) # (B,h,s,d_k) -> (B,s,h,d_k) -> (B,s,d)
      return self.Wo(A) # (B,s,d) @ (d,d) = (B,s,d); this is okay since it is a nn.Linear here

   def save_weights(self):
      # nuclear: detaches from autograd, please only use when you want to save weights, not for intermediate weight caching. 
      attn_path = os.path.join(self.c.weights_base_path, f"{self.lnum}/attn/")
      os.makedirs(attn_path, exist_ok=True)
      torch.save(self.Wq.weight.T.detach().contiguous(), os.path.join(attn_path, "Wq.pt")) # Storing as (in, out) instead of (out, in)
      torch.save(self.Wk.weight.T.detach().contiguous(), os.path.join(attn_path, "Wk.pt"))
      torch.save(self.Wv.weight.T.detach().contiguous(), os.path.join(attn_path, "Wv.pt"))
      torch.save(self.Wo.weight.T.detach().contiguous(), os.path.join(attn_path, "Wo.pt"))

#Inference
class MHAKVOptim(nn.Module):
   # Inference with KV caching, optimized.
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.h, self.d_k = self.c.num_heads, self.c.d_model // self.c.num_heads
      self.Wqkv = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wqkv.pt"))
      self.Wo = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wo.pt"))
      assert tuple(self.Wqkv.shape) == (self.c.d_model, 3*self.c.d_model)
      assert tuple(self.Wo.shape) == (self.c.d_model, self.c.d_model)

   def forward(self, X, mask, KV=None, prevlen=0):
      B, s, d = tuple(X.shape)
      Q, K, V = (X @ self.Wqkv).split(d, dim=-1) 
      Q, K, V = [i.contiguous().view(B,i.shape[1],self.h,self.d_k).permute(0,2,1,3) for i in (Q, K, V)] # (B,s,h,d_k) -> (B,h,s,d_k)
      check = KV is not None
      currlen = prevlen + s
      if check: 
         Kc, Vc = KV 
      else: 
         Kc, Vc = tuple([torch.zeros(B, self.h, self.c.max_seq_len, self.d_k) for i in range(2)])
      Kc[:, :, prevlen:currlen, :] = K
      Vc[:, :, prevlen:currlen, :] = V
      KV = (Kc, Vc)
      
      scores = (Q @ Kc[:, :, :currlen, :].permute(0,1,3,2)) / (self.d_k ** 0.5) # (B,h,s,d_k) (B,h,d_k,currlen)
      scores = scores + mask[prevlen:currlen, :currlen]
      A = (scores.softmax(dim=-1) @ Vc[:, :, :currlen, :]).permute(0,2,1,3).contiguous().view(B,s,d) # (B,h,s,currlen) @ (B,h,currlen,d_k) -> (B,h,s,d_k) -> (B,s,h,d_k) -> (B,s,d)
      return (A @ self.Wo), KV, currlen

class MHAInferenceNoKV(nn.Module):
   # LEGACY: Zero KV optimizations whatsoever.
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.Wq = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wq.pt"))
      self.Wk = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wk.pt"))
      self.Wv = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wv.pt"))
      self.Wo = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wo.pt"))
      self.mask = torch.triu(torch.full((self.c.max_seq_len, self.c.max_seq_len), float('-inf')),diagonal=1)

   def forward(self, X, KV=None, prev_seq=0):
      seq_len = X.shape[1]
      d_k = self.c.d_model // self.c.num_heads # note: for single head attention transformers, d_k = d_model
      # Take a tensor of (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k) for matmuls
      Q = (X @ self.Wq).reshape(self.c.batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2)
      K = (X @ self.Wk).reshape(self.c.batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2)
      V = (X @ self.Wv).reshape(self.c.batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2)

      scores = Q @ K.transpose(-2,-1) # (B,h,s,d_k) @ (B,h,d_k, s) = (B,h,s,s) 
      scores = scores / (d_k ** 0.5)
      scores = scores + self.mask[:seq_len, :seq_len]
      scores = scores.softmax(dim=-1) # row-wise softmax
      A = scores @ V # (B,h,s,s) @ (B,h,s,d_k)
      A = A.transpose(1,2).reshape(self.c.batch_size, seq_len, self.c.d_model) # (B,h,s,d_k) -> (B,s,h,d_k) -> (B,s,d)
      return A @ self.Wo # (B,s,d) @ (d,d) = (B,s,d)
   # Works for both prefill and decoding.


# Grouped Query Attention
# Training
class GQATrain(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.lnum = lnum
      self.h_q, self.d_q = self.c.h_q, self.c.d_model // self.c.h_q
      self.h_kv = self.c.h_kv
      self.dim_kv = self.d_q * self.h_kv
      self.Wqkv = nn.Linear(self.c.d_model, self.c.d_model + 2*self.dim_kv, bias=False)
      self.Wo = nn.Linear(self.c.d_model, self.c.d_model, bias=False)

   def forward(self, X, mask=None):
      B, s, d = tuple(X.shape)
      rep = self.h_q // self.h_kv
      Q, K, V = self.Wqkv(X).split([self.c.d_model, self.dim_kv, self.dim_kv], dim=-1) 
      Q = Q.contiguous().view(B, s, self.h_q, self.d_q).permute(0,2,1,3)
      K, V = [i.contiguous().view(B,s,self.h_kv, self.d_q).permute(0,2,1,3).repeat_interleave(rep, dim=1) for i in (K,V)]
      # (B,s,dim_kv) -> (B,s,h_kv,d_q) -> (B,h_kv,s,d_q) -> (B,h_q, s, d_q) # repeated h_q // h_kv times
      scores = Q @ K.permute(0,1,3,2) # (B,h_q,s,d_q) @ (B,h_q,d_q, s) -> (B,h_q, s,s)
      scores = (scores / (self.d_q ** 0.5)) + mask
      A = (scores.softmax(dim=-1) @ V).permute(0,2,1,3).contiguous().view(B,s,d) # (B,h_q,s,s) (B,h_q,s,d_q) -> (B,h_q,s,d_q) 
      return self.Wo(A)

   def save_weights(self):
      attnpath = os.path.join(self.c.weights_base_path, f"{self.lnum}/attn")
      os.makedirs(attnpath, exist_ok=True)
      torch.save(self.Wqkv.weight.detach().T.contiguous(), os.path.join(attnpath, "Wqkv_gqa.pt"))
      torch.save(self.Wo.weight.detach().T.contiguous(), os.path.join(attnpath, "Wo_gqa.pt"))

# Inference
class GQAInferenceKV(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.h_q, self.d_q = self.c.h_q, self.c.d_model // self.c.h_q
      self.h_kv = self.c.h_kv
      self.dim_kv = self.d_q * self.h_kv
      self.Wqkv = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wqkv_gqa.pt"))
      self.Wo = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wo_gqa.pt"))

   def forward(self, X, mask=None, KV=None, prevseq=0):
      B, s, d = tuple(X.shape)
      Q, K, V = (X @ self.Wqkv).split([self.c.d_model, self.dim_kv, self.dim_kv], dim=-1) 
      Q = Q.contiguous().view(B, s, self.h_q, self.d_q).permute(0,2,1,3)
      K, V = [i.contiguous().view(B,s,self.h_kv, self.d_q).permute(0,2,1,3) for i in (K,V)]
      check = KV is not None
      currseq = prevseq + s
      if check: 
         Kcache, Vcache = KV
      else:
         Kcache, Vcache = [torch.zeros(B,self.h_kv,self.c.max_seq_length,self.d_q) for i in range(2)]
      Kcache[:, :, prevseq:currseq, :] = K
      Vcache[:, :, prevseq:currseq, :] = V
      KV = (Kcache, Vcache)

      rep = self.h_q // self.h_kv
      Kexp = Kcache[:, :, :currseq, :].repeat_interleave(rep, dim=1)
      Vexp = Vcache[:, :, :currseq, :].repeat_interleave(rep, dim=1)
      # (B,s,dim_kv) -> (B,s,h_kv,d_q) -> (B,h_kv,s,d_q) -> (B,h_q, s, d_q) # repeated h_q // h_kv times
      scores = Q @ Kexp.permute(0,1,3,2) # (B,h_q,s,d_q) @ (B,h_q,d_q, s) -> (B,h_q, s,s)
      scores = (scores / (self.d_q ** 0.5)) + mask[prevseq:currseq, :currseq]
      A = (scores.softmax(dim=-1) @ Vexp).permute(0,2,1,3).contiguous().view(B,s,d) # (B,h_q,s,s) (B,h_q,s,d_q) -> (B,h_q,s,d_q) 
      return (A @ self.Wo), KV, currseq