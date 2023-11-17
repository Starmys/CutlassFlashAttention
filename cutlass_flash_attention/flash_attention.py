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
        out, lse = cutlass_fmha_cpp.forward(query, key, value, scale, True, causal)
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
        grad_q, grad_k, grad_v = cutlass_fmha_cpp.backward(
            query, key, value, out,
            grad, lse, delta,
            ctx.scale, ctx.causal,
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
        return cutlass_fmha_cpp.forward(query, key, value, scale, False, self._causal)[0]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
        return self._forward_func(query, key, value, scale)
