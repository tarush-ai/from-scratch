import torch
import torch.nn as nn
from config import RegularConfig
import os, sys

class AttentionClass(nn.Module):
   def __init__(self):
      super().__init__()
      self.c = RegularConfig()
      self.module = sys.modules[__name__]
      self.attn = getattr(self.module, self.c.attention_type)()

   def forward(self, X):
      return self.attn(X)

class StandardMHATrain(nn.Module):
   def __init__(self, *args, **kwargs):
      super().__init__()
      self.c = RegularConfig()
      self.Wq = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.Wk = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.Wv = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.Wo = nn.Linear(self.c.d_model, self.c.d_model, bias=False)
      self.mask = torch.triu(torch.full((self.c.max_seq_len, self.c.max_seq_len), float('-inf')),diagonal=1)

   def forward(self, X):
      assert tuple(X.shape) == (self.c.batch_size, self.c.max_seq_len, self.c.d_model)
      d_k = self.c.d_model // self.c.num_heads # note: for single head attention transformers, d_k = d_model
      # Take a tensor of (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k) for matmuls
      Q = self.Wq(X).reshape(self.c.batch_size, self.c.max_seq_len, self.c.num_heads, d_k).transpose(1,2)
      K = self.Wk(X).reshape(self.c.batch_size, self.c.max_seq_len, self.c.num_heads, d_k).transpose(1,2)
      V = self.Wv(X).reshape(self.c.batch_size, self.c.max_seq_len, self.c.num_heads, d_k).transpose(1,2)

      scores = Q @ K.transpose(-2,-1) # (B,h,s,d_k) @ (B,h,d_k, s) = (B,h,s,s) 
      scores = scores / (d_k ** 0.5)
      scores = scores + self.mask
      scores = scores.softmax(dim=-1) # row-wise softmax
      A = scores @ V # (B,h,s,s) @ (B,h,s,d_k)
      A = A.transpose(1,2).reshape(self.c.batch_size, self.c.max_seq_len, self.c.d_model) # (B,h,s,d_k) -> (B,s,h,d_k) -> (B,s,d)
      return self.Wo(A) # (B,s,d) @ (d,d) = (B,s,d)

class MHAInferenceNoKV(nn.Module):
   def __init__(self, *args, **kwargs):
      super().__init__()
      self.c = RegularConfig()
      self.Wq = torch.load(self.c.wqpath)
      self.Wk = torch.load(self.c.wkpath)
      self.Wv = torch.load(self.c.wvpath)
      self.Wo = torch.load(self.c.wopath)
      self.mask = torch.triu(torch.full((self.c.max_seq_len, self.c.max_seq_len), float('-inf')),diagonal=1)

   def forward(self, X):
      seq_len = X.shape[1]
      d_k = self.c.d_model // self.c.num_heads # note: for single head attention transformers, d_k = d_model
      # Take a tensor of (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k) for matmuls
      Q = self.Wq(X).reshape(self.c.batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2)
      K = self.Wk(X).reshape(self.c.batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2)
      V = self.Wv(X).reshape(self.c.batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2)

      scores = Q @ K.transpose(-2,-1) # (B,h,s,d_k) @ (B,h,d_k, s) = (B,h,s,s) 
      scores = scores / (d_k ** 0.5)
      scores = scores + self.mask[:seq_len, :seq_len]
      scores = scores.softmax(dim=-1) # row-wise softmax
      A = scores @ V # (B,h,s,s) @ (B,h,s,d_k)
      A = A.transpose(1,2).reshape(self.c.batch_size, seq_len, self.c.d_model) # (B,h,s,d_k) -> (B,s,h,d_k) -> (B,s,d)
      return self.Wo(A) # (B,s,d) @ (d,d) = (B,s,d)
   # Works for both prefill and decoding.


class MHAInferenceKVNoBatching(nn.Module):
   def __init__(self, *args, **kwargs):
      super().__init__()
      self.c = RegularConfig()
      self.Wq = torch.load(self.c.wqpath) 
      self.Wk = torch.load(self.c.wkpath)
      self.Wv = torch.load(self.c.wvpath)
      self.Wo = torch.load(self.c.wopath)

   def forward(self, X, prefill=False):
      #assumptions made:
      # 2d tensor (seq_len, d_model); NO batching (don't want to think about static vs dynamic and how to not waste)
      # path implementation (since it is per generation, needs some unique handling over here with datetime or something but focusing on the algorithm for this iteration.

      seq_len = X.shape[0] 
      if not prefill:
         assert seq_len == 1

      d_k = self.c.d_model // self.c.num_heads

      Q = self.Wq(X).reshape(seq_len, self.c.num_heads, d_k).transpose(0,1) # (1, d_model) @ (d_model, d_model) = (1, d_model) -> (1, num_heads, d_k) -> (num_heads, 1, d_k)
      K = self.Wk(X).reshape(seq_len, self.c.num_heads, d_k).transpose(0,1) # (1, d_model) @ (d_model, d_model) = (1, d_model) -> (1, num_heads, d_k) -> (num_heads, 1, d_k)
      V = self.Wv(X).reshape(seq_len, self.c.num_heads, d_k).transpose(0,1) # (1, d_model) @ (d_model, d_model) = (1, d_model) -> (1, num_heads, d_k) -> (num_heads, 1, d_k)

      if not prefill:
         Kcache = torch.load('kcache_example.pt')
         Vcache = torch.load('vcache_example.pt')
         prev_seq = Kcache.shape[0]
         assert prev_seq == Vcache.shape[0] # should be the same, or there is a problem
         Kcache = Kcache.reshape(prev_seq, self.c.num_heads, d_k).transpose(0,1)
         Vcache = Vcache.reshape(prev_seq, self.c.num_heads, d_k).transpose(0,1) # (prev_seq, d_model) -> (prev_seq, num_heads, d_k) -> (num_heads, prev_seq, d_k)
         Kcache = torch.concat([Kcache, K], dim=1)
         Vcache = torch.concat([Vcache, V], dim=1)
         Ksaver = Kcache.transpose(0,1).reshape(prev_seq+1, self.c.d_model)
         Vsaver = Vcache.transpose(0,1).reshape(prev_seq+1, self.c.d_model)
         torch.save(Ksaver, 'kcache_example.pt')
         torch.save(Vsaver, 'vcache_example.pt')
      else:
         Kcache = K
         Vcache = V
         Ksave = K.transpose(0,1).reshape(seq_len, self.c.d_model)
         Vsave = V.transpose(0,1).reshape(seq_len, self.c.d_model)
         torch.save(Ksave, 'kcache_example.pt')
         torch.save(Vsave, 'vcache_example.pt')

      scores = Q @ Kcache.transpose(-2,-1) # (num_heads, 1, d_k) @ (num_heads, d_k, seq_len) = (num_heads, 1, seq_len)
      scores = scores / (d_k ** 0.5)
      if prefill: scores = scores + torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
      scores = scores.softmax(dim=-1)
      A = scores @ Vcache # (num_heads, 1, seq_len) @ (num_heads, seq_len, d_k) = (num_heads, 1, d_k)
      A = A.transpose(0,1).reshape(seq_len, self.c.d_model) # (num_heads, 1, d_k) -> (1, num_heads, d_k) -> (1, d_model)
      return self.Wo(A) # (1, d_model) @ (d_model, d_model) = (1, d_model)

class MHAInferenceKVBatching(nn.Module):
   def __init__(self):
      super().__init__()
      self.c = RegularConfig()
      self.Wq = torch.load(self.c.wqpath)
      self.Wk = torch.load(self.c.wkpath)
      self.Wv = torch.load(self.c.wvpath)
      self.Wo = torch.load(self.c.wopath)


   def forward(self, X, prefill=False):
      batch_size, seq_len, d_model = tuple(X.shape) 
      assert d_model == self.c.d_model
      if not prefill: assert seq_len == 1
      d_k = d_model // self.c.num_heads
      Q = self.Wq(X).reshape(batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2) # (B,s,d) -> (B,s,h,d_k) -> (B,h,s,d_k)
      K = self.Wk(X).reshape(batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2) # (B,s,d) -> (B,s,h,d_k) -> (B,h,s,d_k)
      V = self.Wv(X).reshape(batch_size, seq_len, self.c.num_heads, d_k).transpose(1,2) # (B,s,d) -> (B,s,h,d_k) -> (B,h,s,d_k)

      if not prefill:
         Kcache = torch.load('kcache_example.pt')
         Vcache = torch.load('vcache_example.pt')
         prev_seq = Kcache.shape[1]
         assert prev_seq == Vcache.shape[1]
         Kcache = Kcache.reshape(batch_size, prev_seq, self.c.num_heads, d_k).transpose(1,2) # (B,ps,d) -> (B,ps,h,d_k) -> (B,h,ps,d_k) 
         Vcache = Vcache.reshape(batch_size, prev_seq, self.c.num_heads, d_k).transpose(1,2) # (B,ps,d) -> (B,ps,h,d_k) -> (B,h,ps,d_k) 
         K = torch.concat([Kcache, K], dim=2)
         V = torch.concat([Vcache, V], dim=2)
         Ksave = K.transpose(1,2).reshape(batch_size, prev_seq+1, d_model)
         Vsave = V.transpose(1,2).reshape(batch_size, prev_seq+1, d_model)
         torch.save(Ksave, 'kcache_example.pt')
         torch.save(Vsave, 'vcache_example.pt')

      else:
         Ksave = K.transpose(1,2).reshape(batch_size, seq_len, d_model)
         Vsave = V.transpose(1,2).reshape(batch_size, seq_len, d_model)
         torch.save(Ksave, 'kcache_example.pt')
         torch.save(Vsave, 'vcache_example.pt')

      scores = Q @ K.transpose(-2, -1)
      scores = scores / (d_k ** 0.5)
      if prefill: scores = scores + torch.triu(torch.full((seq_len,seq_len), float('-inf')), diagonal=1)
      scores = scores.softmax(dim=-1)
      A = scores @ V # (B,h,s,s) @ (B,h,s,d_k)
      A = A.transpose(1,2).reshape(batch_size, seq_len, d_model)
      return self.Wo(A)


# For a GPU optimized implementation, leverage: (B, h, s, d_k) -> Permute to (B, s, h, d_k) -> force contiguity -> view flat
# A = (scores @ V).permute(0, 2, 1, 3).contiguous().view(X.shape[0], X.shape[1], -1)


'''
Implemented:
MHA (training, no KV cache)
MHA (inference (decode), no KV cache)
MHA (inference (decode), KV cache, no batching)

To implement:
MHA (inference (decode), KV cache, batching)
Optimizations of MHA tensors (contiguous caching, predecessor to PagedAttention, CPU vs GPU)
MHA (inference (prefill), no KV cache)
MHA (inference (prefill), KV cache)
PagedAttention
GQA (Grouped Query Attention)
MLA (Multihead Latent Attention)
GQLA (Grouped Query Latent Attention)
MoBA (Mixture of Block Attention)


Consolidated roadmap (dependency-ordered; each = one unit, naive form first, fast form second,
test that the two agree before moving on)

--- FOUNDATIONS (the correctness spine) ---
1.  MHA training (full sequence, causal mask)                              [done]
2.  MHA decode, no KV cache                                                [done]
3.  MHA decode, single-request KV cache                                    [done]
4.  Reference prefill, no KV cache                     (whole prompt in one masked forward)
5.  Prefill WITH KV cache                              (prefill is what fills the cache decode reads)
    >> TEST: decode-with-cache == full masked forward, token by token, to 1e-5
             (this is the gate; nothing past here is trustworthy until it passes)

--- POSITIONS (what bounds the context window) ---
6.  Sinusoidal positional encoding                    (the original; you've done it in AureliusGPT)
7.  Learned positional embeddings                     (GPT-2 style; what you have now)
8.  RoPE (rotary)                                      (rotate q,k by position; relative pos falls out)
9.  ALiBi                                              (linear position bias; no embedding at all)
10. Context extension: position interpolation / RoPE scaling / YaRN
                                                       (why max_seq_len is not a hard wall)

--- BATCHING & VARIABLE LENGTH (the thing that actually gives pause) ---
11. Batched decode, uniform length                    (B independent requests, walled-off batch axis)
12. Batched decode, variable length: right-pad + key-padding mask
13. Batched decode, variable length: LEFT-pad         (newest token always at last index; what engines do)
14. Batched prefill / packed sequences                (multiple prompts in one flat tensor + block-diag mask)

--- KV-CACHE SIZE ATTACKS (each cuts the I/O cost you can already feel) ---
15. MQA (Multi-Query Attention)                        (all query heads share ONE K,V head)
16. GQA (Grouped Query Attention), variable group counts   (g query heads per K,V head; g=1 -> MHA)
17. Sliding-window attention                           (mask keys older than W; cache bounded to W)
18. KV-cache quantization (int8/fp8)                   (fewer bytes per cached token)
19. MLA (Multihead Latent Attention)                   (cache a low-rank latent, up-project on read)
20. GQLA (Grouped Query Latent Attention)              (GQA + MLA combined)

--- MEMORY MANAGEMENT (systems layer on top of the math) ---
21. Contiguous cache preallocation                     (ring buffer; predecessor to paging; CPU vs GPU placement)
22. PagedAttention                                     (fixed blocks + page table; kills padding waste)
23. Prefix caching / cache reuse                       (share blocks across requests with a common prompt)
24. KV-cache offloading                                (spill cold cache to CPU/NVMe; the memory hierarchy from the filing-cabinet talk)

--- SCHEDULING (throughput, not correctness) ---
25. Continuous / in-flight batching                    (evict finished, admit waiting mid-flight; kills straggler waste)
26. Chunked prefill                                    (interleave prefill chunks with decode so long prompts don't stall the batch)

--- KERNELS (the compute axis; where "how fast" lives) ---
27. Tiled attention (the FlashAttention idea, in PyTorch first)  (never materialize the s x s scores)
28. FlashAttention forward in Triton                   (one launch, Q/K/V tiles -> SRAM -> out; the ~80-line rite of passage)
29. PyTorch SDPA backend comparison                    (F.scaled_dot_product_attention: know when it's silently used, and how close yours is)

--- DECODE-TIME COMPUTE ATTACKS ---
30. Speculative decoding                               (draft proposes k tokens, target verifies in one forward)
31. Medusa / EAGLE                                     (make the drafter a head on the target model)

--- SPARSE & BLOCK ATTENTION ---
32. Cross-attention                                    (encoder-decoder; Q from one stream, K,V from another)
33. Block-sparse attention                             (structured sparsity over the s x s grid)
34. MoBA (Mixture of Block Attention)                  (route each query to a subset of key-blocks)

--- STATE-BASED / SUB-QUADRATIC (the Baseten article's spine; a different regime: state, not cache) ---
35. Linear attention                                   (drop softmax, fold history into fixed state S; recurrent == chunked)
36. DeltaNet                                           (delta rule: read-before-write; exact associative recall)
37. Gated DeltaNet                                     (scalar decay -> state can forget)
38. KDA (Kimi Delta Attention)                         (per-channel decay)
39. Hybrid stack (Kimi-K3 style)                       (interleave KDA:MLA layers + AttnRes; needle-in-haystack vs pure-linear)

--- MoE (not attention, but the other half of a modern block; separate ladder) ---
40. Top-k router + expert FFNs + load-balance loss     (the routing layer is one matmul + top-k)


Cross-cutting things to internalize as you go (not separate files):
- Arithmetic intensity per step: decode is memory-bound, prefill is compute-bound (measure it, don't take my word)
- FLOP + byte accounting per variant (this is your transformer-accounting repo's real job)
- Numerical: online/streaming softmax (the max-subtraction trick that makes FlashAttention and linear attention stable)
- Every rung ships with its equality test; naive form is the oracle the fast form is checked against
'''