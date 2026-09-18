import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_add_1d_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
   pid = tl.program_id(0)
   offset = pid * BLOCK + tl.arange(0,BLOCK)
   mask = offset < n
   x = tl.load(x_ptr + offset, mask=mask)
   y = tl.load(y_ptr + offset, mask=mask)
   out = x + y
   tl.store(out_ptr + offset, out, mask=mask)

def elementwise_add_1d(x, y):
   out = torch.empty_like(x)
   n = x.numel()
   BLOCK = 1024
   grid = (triton.cdiv(n,BLOCK),)
   elementwise_add_1d_kernel[grid](x, y, out, n, BLOCK)
   return out 

