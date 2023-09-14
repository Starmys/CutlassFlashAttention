import time
import torch
from cutlass_flash_attention import FlashMultiHeadAttention


BATCH, N_HEADS, N_CTX, D_HEAD = 16, 64, 1024, 64


def attention_forward_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    p = torch.einsum(f'bmhk, bnhk -> bhmn', query, key) * sm_scale
    for b, (q_seqlen, kv_seqlen) in enumerate(zip(q_seqlens, kv_seqlens)):
        p[b, :, q_seqlen:, :] = -torch.inf
        p[b, :, :, kv_seqlen:] = -torch.inf
    # s = torch.softmax(p.float(), dim=-1).half()
    p_max = p.max(-1, keepdim=True).values
    p_max = torch.where(p_max < 0, 0.0, p_max)
    p_exp = torch.exp(p - p_max)
    s = p_exp / (p_exp.sum(-1, keepdim=True) + 1e-6)
    out = torch.einsum(f'bhmn, bnhk -> bmhk', s, value)
    return out


def profile(fn, mask_nnz, mode='fwd', warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    end = time.time
    latency = (end - start) * 1e3
    flops_per_matmul = 2. * N_HEADS * mask_nnz * D_HEAD
    total_flops = 2 * flops_per_matmul
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    gflops = total_flops / latency * 1e-9
    print(f'{mode}: {latency:.3f} ms | {gflops:.3f} GFLOP/s')


def test_flash_attention(dtype=torch.float16, device="cuda"):
    q = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, N_CTX, N_HEADS, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    seqlens = torch.zeros((BATCH, ), dtype=torch.int32, device=device, requires_grad=False) + N_CTX
    # seqlens = torch.randint(N_CTX // 2, N_CTX, (BATCH, ), dtype=torch.int32, device=device, requires_grad=False)
    sm_scale = D_HEAD ** -0.5

    cutlass_fmha = FlashMultiHeadAttention(training=True)

    ref_o = attention_forward_reference(q, k, v, seqlens, seqlens, sm_scale)
    do = torch.randn_like(ref_o)
    ref_o.backward(do)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    forward_fn = lambda: cutlass_fmha(q, k, v, sm_scale)
    o = forward_fn()
    backward_fn = lambda: o.backward(do, retain_graph=True)
    backward_fn()
    dv, v.grad = v.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dq, q.grad = q.grad.clone(), None

    import ipdb; ipdb.set_trace()
    torch.testing.assert_close(o, ref_o, atol=1e-2, rtol=0)
    torch.testing.assert_close(dq, ref_dq, atol=1e-2, rtol=0)
    torch.testing.assert_close(dk, ref_dk, atol=1e-2, rtol=0)
    torch.testing.assert_close(dv, ref_dv, atol=1e-2, rtol=0)

    mask_nnz = seqlens.square().sum().item()
    forward_flops = profile(forward_fn, mask_nnz, 'fwd')
    backward_flops = profile(backward_fn, mask_nnz, 'bwd')
    return forward_flops, backward_flops


torch.manual_seed(2023)
forward_flops, backward_flops = test_flash_attention(torch.float16)
