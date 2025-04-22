from typing import Literal, Optional

import numpy as np
import torch as th
from jaxtyping import Float, Int
from fancy_einsum import einsum

from .discrete import Discrete

# Code structure modeled off transformer_lens.components (MIT License)


class Embed(th.nn.Module):
    def __init__(self, d_vocab: int, d_model: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.W_E = th.nn.Parameter(th.empty(d_vocab, d_model, **factory_kwargs))

    def forward(
        self, tokens: Int[th.Tensor, "batch pos"]
    ) -> Float[th.Tensor, "batch pos d_model"]:
        return self.W_E[tokens, :]



class Unembed(th.nn.Module):
    def __init__(self, d_model: int, d_vocab: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.W_U = th.nn.Parameter(th.empty(d_model, d_vocab, **factory_kwargs))

    def forward(
        self, x: Float[th.Tensor, "batch pos d_model"]
    ) -> Float[th.Tensor, "batch pos d_vocab"]:
        return x @ self.W_U[None]


class FinalSoftmax(th.nn.Module):
    def __init__(self, d_model: int, d_vocab: int, n_quad: int = 65):
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.n_quad = n_quad

    def forward(
        self,
        logits: Float[th.Tensor, "batch pos d_vocab"],
        *,
        temperature: float = 1.0,
    ) -> Float[th.Tensor, "batch pos d_vocab"]:
        if temperature == 0.0:
            argmax = th.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1])
            return argmax.to(logits.dtype)
        else:
            return th.nn.functional.softmax(logits / temperature, -1)


class PosEmbed(th.nn.Module):
    def __init__(self, n_ctx: int, d_model: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.W_pos = th.nn.Parameter(th.empty(n_ctx, d_model, **factory_kwargs))

    def forward(
        self, tokens: Int[th.Tensor, "batch pos"]
    ) -> Float[th.Tensor, "batch pos d_model"]:
        return self.W_pos[None, : tokens.shape[-1]].repeat(tokens.shape[0], 1, 1)


class LayerNorm(th.nn.Module):
    def __init__(self, d_model: int, ln_eps: float, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.n_quad = d_model + 1
        self.ln_eps = ln_eps
        self.w: Float[th.Tensor, "d_model"] = th.nn.Parameter(
            th.ones(d_model, **factory_kwargs)
        )
        self.b: Float[th.Tensor, "d_model"] = th.nn.Parameter(
            th.zeros(d_model, **factory_kwargs)
        )

    def forward(
        self,
        x: Float[th.Tensor, "batch pos d_model"],
    ) -> Float[th.Tensor, "batch pos d_model"]:
        x = x - x.mean(-1, keepdim=True)
        x = x / ((x**2).mean(-1, keepdim=True) + self.ln_eps) ** 0.5
        x = x * self.w[None, None] + self.b[None, None]
        return x


def masked_softmax(logits, mask):
    mask = mask[(None,) * (logits.ndim - mask.ndim)]
    mask = mask[..., : logits.shape[-2], : logits.shape[-1]]
    logits = logits.masked_fill(~mask, float("-inf"))
    logits = (
        logits - logits.max(dim=-1, keepdim=True)[0]
    )  # important that this is done after masking to avoid division by 0
    probs = th.exp(logits)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


class MLP(th.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        act_fn: Literal["relu", "gelu"],
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.W_in = th.nn.Parameter(th.empty(d_model, d_mlp, **factory_kwargs))
        self.b_in = th.nn.Parameter(th.empty(d_mlp, **factory_kwargs))
        self.W_out = th.nn.Parameter(th.empty(d_mlp, d_model, **factory_kwargs))
        self.b_out = th.nn.Parameter(th.empty(d_model, **factory_kwargs))
        self.act_fn = {"relu": th.nn.functional.relu, "gelu": th.nn.functional.gelu}[
            act_fn
        ]

    def forward(
        self,
        x: Float[th.Tensor, "batch pos d_model"],
    ) -> Float[th.Tensor, "batch pos d_model"]:
        x = x @ self.W_in + self.b_in
        x = self.act_fn(x)
        x = x @ self.W_out + self.b_out
        return x


class Attention(th.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_head: int,
        n_ctx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.W_Q = th.nn.Parameter(th.empty(n_heads, d_model, d_head, **factory_kwargs))
        self.W_K = th.nn.Parameter(th.empty(n_heads, d_model, d_head, **factory_kwargs))
        self.W_V = th.nn.Parameter(th.empty(n_heads, d_model, d_head, **factory_kwargs))
        self.W_O = th.nn.Parameter(th.empty(n_heads, d_head, d_model, **factory_kwargs))
        self.b_Q = th.nn.Parameter(th.zeros(n_heads, d_head, **factory_kwargs))
        self.b_K = th.nn.Parameter(th.zeros(n_heads, d_head, **factory_kwargs))
        self.b_V = th.nn.Parameter(th.zeros(n_heads, d_head, **factory_kwargs))
        self.b_O = th.nn.Parameter(th.zeros(d_model, **factory_kwargs))
        causal_mask = th.tril(th.ones(n_ctx, n_ctx, **factory_kwargs)).bool()
        self.register_buffer("mask", causal_mask)
        self.register_buffer("IGNORE", th.tensor(float("-inf")))

    def forward(
        self,
        x: Float[th.Tensor, "batch pos d_model"],
    ) -> Float[th.Tensor, "batch pos d_model"]:
        x: Float[th.Tensor, "batch head pos d_model"] = x[:, None]
        Q = x @ self.W_Q[None] + self.b_Q[None, :, None]
        K = x @ self.W_K[None] + self.b_K[None, :, None]
        V = x @ self.W_V[None] + self.b_V[None, :, None]
        L: Float[th.Tensor, "batch head pos pos"] = Q @ K.mT
        P: Float[th.Tensor, "batch head pos pos"] = masked_softmax(
            L / self.d_head**0.5, self.mask
        )
        Z: Float[th.Tensor, "batch head pos d_head"] = P @ V
        result = (
            einsum(
                "batch head pos d_head, head d_head d_model -> batch pos d_model",
                Z,
                self.W_O,
            )
            + self.b_O[None, None]
        )
        return result

class TransformerBlock(th.nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_head: int,
        n_ctx: int,
        ln_eps: float,
        attn_only: bool,
        act_fn: Optional[Literal["gelu", "relu"]] = None,
        d_mlp: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.ln1 = LayerNorm(d_model=d_model, ln_eps=ln_eps, **factory_kwargs)
        self.attn = Attention(
            n_heads=n_heads,
            d_model=d_model,
            d_head=d_head,
            n_ctx=n_ctx,
            **factory_kwargs,
        )
        self.attn_only = attn_only
        if not attn_only:
            assert d_mlp is not None, "d_mlp must be provided for non-attention-only"
            self.ln2 = LayerNorm(d_model=d_model, ln_eps=ln_eps, **factory_kwargs)
            self.mlp = MLP(
                d_model=d_model,
                d_mlp=d_mlp,
                act_fn=act_fn,
                **factory_kwargs,
            )

    def forward(
        self,
        x: Float[th.Tensor, "batch pos d_model"],
    ) -> Float[th.Tensor, "batch pos d_model"]:
        res = x
        x = res + self.attn(self.ln1(x))
        if not self.attn_only:
            res2 = x
            x = res2 + self.mlp(self.ln2(x))
        return x