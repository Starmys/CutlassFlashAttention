## Prerequsites
- PyTorch
- NVCC >= 11.3
- CUDA Compute Capacity >= 7.0

## Installation
```bash
pip install git+https://github.com/Starmys/CutlassFlashAttention.git
```

## Quick Start
```python
import torch
from cutlass_flash_attention import FlashMultiHeadAttention

BATCH, N_CTX, N_HEADS, D_HEAD = 2, 1024, 32, 128
dtype = torch.float32
device = 'cuda'

q = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
scale = D_HEAD ** -0.5

fmha = FlashMultiHeadAttention(training=True, causal=True)

o = fmha(q, k, v, scale)
```
