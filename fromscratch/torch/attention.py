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
   
   def forward(self, X, mask=None, KV=None, prevlen=0):
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

class MHATrain(nn.Module):
   # LEGACY: Reimplemented old MHA training class (this one is effectively the same thing, but is not as visible with view() usage but functionally the same thing).
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.Wqkv = nn.Linear(self.c.d_model, 3*self.c.d_model, bias=False)
      self.Wo = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.lnum = lnum

   def forward(self, X, mask=None):
      B, s, d = X.shape
      d_k = self.c.d_model // self.c.num_heads
      Q, K, V = self.Wqkv(X).split(self.c.d_model, dim=-1)
      Q, K, V = [i.reshape(B, s, self.c.num_heads, d_k).permute(0,2,1,3) for i in (Q,K,V)]
      scores = (Q @ K.permute(0,1,3,2) / d_k ** 0.5) + mask[:s,:s]
      A = (scores.softmax(dim=-1) @ V).permute(0,2,1,3).reshape(B,s,d)
      return self.Wo(A)

   def save_weights(self):
      attnpath = os.path.join(self.c.weights_base_path, f"{self.lnum}/attn")
      os.makedirs(attnpath, exists_ok=True)
      torch.save(self.Wqkv.weight.T.detach().contiguous(), os.path.join(attnpath, "Wqkv.pt"))
      torch.save(self.Wo.weight.T.detach().contiguous(), os.path.join(attnpath, "Wo.pt"))

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

class MHAInfNoKV(nn.Module):
   # LEGACY: Reimplemented no KV caching for MHA. 
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.Wqkv = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wqkv.pt"))
      self.Wo = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wo.pt"))

   def forward(self, X, mask=None, KV=None, prevseq=0):
      B, s, d = X.shape
      d_k = self.c.d_model // self.c.num_heads
      Q, K, V = (X @ self.Wqkv).split(self.c.d_model, dim=-1)
      Q, K, V = [i.reshape(B, s, self.c.num_heads, d_k).permute(0,2,1,3) for i in (Q,K,V)]
      scores = (Q @ K.permute(0,1,3,2) / d_k ** 0.5) + mask
      A = (scores.softmax(dim=-1) @ V).permute(0,2,1,3).reshape(B,s,d)
      return A @ self.Wo, None, None


# Grouped Query Attention
# Technically, this can handle MQA based on config
# Training
class GQATrainRepeatInterleaving(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.lnum = lnum
      self.h_q, self.d_q = self.c.h_q, self.c.d_model // self.c.h_q
      self.h_kv = self.c.h_kv
      self.rel = self.h_kv / self.h_q
      self.d_kv = int(self.c.d_model * self.rel)
      self.qkvdim = self.c.d_model + 2*self.d_kv
      self.Wqkv = nn.Linear(self.c.d_model, self.qkvdim, bias=False)
      self.Wo = nn.Linear(self.c.d_model, self.c.d_model, bias=False)

   def forward(self, X, mask=None):
      B, s, d = X.shape
      rellocal = int(1/self.rel) # (h_q/h_kv)
      # d_kv = d * (h_kv/h_q)
      # (B,s,d) @ (d, d+ 2*d_kv) -> (B,s,d + 2*d_kv)
      Q, K, V = self.Wqkv(X).split([self.c.d_model, self.d_kv, self.d_kv], dim=-1)
      # (B,s,d+2*d_kv) -> Q (B,s,d); K/V (B,s,d_kv)
      Q = Q.view(B, s, self.h_kv, rellocal, self.d_q).permute(0,2,3,1,4)
      # Q (B,s,d) -> (B,s,h_kv,rel,d_q) -> (B,h_kv,rel,s,d_q)
      K, V = [i.view(B,s,self.h_kv,1,self.d_q).permute(0,2,3,1,4) for i in (K,V)]
      # K,V (B,s,d_kv) -> (B,s,h_kv,1,d_q) -> (B,h_kv,1,s,d_q)
      scores = (Q @ K.permute(0,1,2,4,3) / self.d_q ** 0.5) + mask[:s,:s] 
      # Q (B,h_kv,rel,s,d_q) @ (B,h_kv,1,d_q,s) -> (B,h_kv,rel,s,s)
      A = (scores.softmax(dim=-1) @ V).permute(0,3,1,2,4).contiguous().view(B,s,d)
      # scores (B,h_kv,rel,s,s) @ (B,h_kv,1,s,d_q) -> (B,h_kv,rel,s,d_q)
      # (B,h_kv,rel,s,d_q) -> (B,s,h_kv,rel,d_q)
      # (B,s,d) -> d = h_kv*rel*d_q where rel=h_q/h_kv; d=h_q*d_q=d correct
      return self.Wo(A) # (B,s,d) @ (d,d) = (B,s,d)

   def save_weights(self):
      attnpath = os.path.join(self.c.weights_base_path, f"{self.lnum}/attn")
      os.makedirs(attnpath, exist_ok=True)
      torch.save(self.Wqkv.weight.T.detach().contiguous(), os.path.join(attnpath, "Wqkv_gqa.pt"))
      torch.save(self.Wo.weight.T.detach().contiguous(), os.path.join(attnpath, "Wo_gqa.pt"))

class GQAInferenceKV(nn.Module):
   def __init__(self, lnum):
      super().__init__()
      self.c = RegularConfig()
      self.Wqkv = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wqkv_gqa.pt"))
      self.Wo = torch.load(os.path.join(self.c.weights_base_path, f"{lnum}/attn/Wo_gqa.pt"))
      self.h_q, self.d_q = self.c.h_q, self.c.d_model // self.c.h_q
      self.h_kv = self.c.h_kv
      self.d_kv = self.h_kv * self.d_q
      self.rel = int(self.h_q / self.h_kv)


   def forward(self, X, mask=None, KV=None, prevseq=0):
      B,s,d = X.shape
      Q, K, V = (X @ self.Wqkv).split([d, self.d_kv, self.d_kv], dim=-1)
      Q = Q.view(B,s,self.h_kv,self.rel,self.d_q).permute(0,2,3,1,4)
      K, V = [i.view(B,s,self.h_kv,1,self.d_q).permute(0,2,3,1,4) for i in (K,V)]
      # (B,h_kv,1,s,d_q)
      check = KV is not None
      currseq = prevseq + s
      if check:
         Kcache, Vcache = KV
      else:
         Kcache, Vcache = tuple([torch.zeros(B,self.h_kv,1,self.c.max_seq_length,self.d_q) for i in range(2)])
      Kcache[:, :, :, prevseq:currseq, :] = K
      Vcache[:, :, :, prevseq:currseq, :] = V
      KV = (Kcache, Vcache)
      scores = (Q @ Kcache[:,:,:,:currseq,:].permute(0,1,2,4,3) / self.d_q ** 0.5)
      # (B,h_kv,rel,s, d_q) @ (B,h_kv,1,d_q,s) -> (B,h_kv,rel,s,s)
      scores = scores + mask[prevseq:currseq, :currseq] 
      A = (scores.softmax(dim=-1) @ Vcache[:,:,:,:currseq,:]).permute(0,3,1,2,4).contiguous().view(B,s,d)
      # (B,h_kv,rel,s,s) @ (B,h_kv,1,s,d_q) -> (B,h_kv,rel,s,d_q) -> (B,s,h_kv,rel,d_q) -> (B,s,d)
      return A @ self.Wo, KV, currseq

      



'''
How might a 5D tensor work? Brainstorming.

The problem is h_dk vs h_q. From my understanding, Q is:
(B,self.h_q,1,s,d_q) after permuting and everything
Is K and V supposed to be 
(B,self.h_q,rel,s,d_q)? Is that what it is? I don't understand why though.
Is it because when we matmul by permute(0,1,2,4,3) for K...
it becomes (B,self.h_q,rel,s,s)?
How does reshaping work? I guess I'm super confused...
Please help me understand it a bit better. Trying my best but kind of guessing. 
Clearly I get the alg- I think there's struggling with tensor bookkeeping dimensions. 
The actual matmuls are correct; it's more so tensor manipulation for 5d tensors, where I've worked with 
max 4d in the past. Can you give me tensor manipulation skills that go up to Nd? do we have to deal with >5D tensors in infra work? 
:( Idk if I'm doing well or nah
'''

'''
Revised note given Claude's teaching:
Q (B,s,d) -> (B,s,g,m,d_q) where g is h_kv; -> (B,g,m,s,d_q)
which is here equal to (B,h_kv,rel,s,d_q) where rel is h_q/h_kv
do the math: h_kv*rel*d_q = h_q*d_q = d_model

K, V (B,s,d_kv) -> (B,s,g,m,d_q) where g is h_kv -> (B,g,1,s,d_q)
which is here equal to (B,h_kv,1,s,d_q) because h_dk*d_q is d_kv

What I don't get is the MATMUL itself, and the reshape operation.

Q@K.permute(0,1,2,4,3) = (B,h_kv,rel,s,d_q) @ (B,h_kv,rel,d_q,s) -> (B,h_kv,__,s,s)... does the rel multiply through? 
I think it does right? Does that mean there is a 3d matmul of some kind? Not really sure how multiplying two third order tensors works...
The reason I think rel carries through is because we need to reshape into size d_model, NOT size d_kv.

Also, how do we KNOW to reshape in that way? Like, those 3 nums could have come out of the final dim, or they could have come out of -2,-1...?
'''