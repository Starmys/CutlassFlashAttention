import torch

import cutlass_fmha_cpp


class _FMHA(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        causal: bool,
    ):
        batch_size, seq_len, head_num, head_dim = query.shape
        out = torch.empty_like(query)
        lse = torch.empty(size=(batch_size, head_num, seq_len), dtype=torch.float32, device=query.device)
        cutlass_fmha_cpp.forward_training(query, key, value, out, lse, scale, causal)
        ctx.save_for_backward(query, key, value, out, lse)
        ctx.scale = scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ):
        query, key, value, out, lse = ctx.saved_tensors
        delta = (grad.float() * out.float()).sum(-1).transpose(-2, -1).contiguous()
        grad_q = torch.empty_like(query)
        grad_k = torch.empty_like(key)
        grad_v = torch.empty_like(value)
        cutlass_fmha_cpp.backward(
            query, key, value, out,
            grad_q, grad_k, grad_v, grad,
            lse, delta, ctx.scale, ctx.causal,
        )
        return grad_q, grad_k, grad_v, None, None


class FlashMultiHeadAttention(torch.nn.Module):

    def __init__(self, training: bool = True, causal: bool = True):
        super().__init__()
        self._forward_func = self._forward_train if training else self._forward_test
        self._causal = causal

    def _forward_train(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
        return _FMHA.apply(query, key, value, scale, self._causal)

    def _forward_test(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
        out = torch.empty_like(query)
        cutlass_fmha_cpp.forward_inference(query, key, value, out, scale, self._causal)
        return out

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
        return self._forward_func(query, key, value, scale)
