import json
import warnings
from contextlib import contextmanager
from functools import cache
from typing import List, Optional

import blobfile as bf
import torch as th
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int
from transformers import AutoTokenizer

from .components import (
    Embed,
    FinalSoftmax,
    LayerNorm,
    PosEmbed,
    TransformerBlock,
    Unembed,
)
from .discrete import Discrete

MODEL_REPO_NAMES = {
    "attn-only-1l": "NeelNanda/Attn_Only_1L512W_C4_Code",
    "attn-only-2l": "NeelNanda/Attn_Only_2L512W_C4_Code",
    "attn-only-3l": "NeelNanda/Attn_Only_3L512W_C4_Code",
    "attn-only-4l": "NeelNanda/Attn_Only_4L512W_C4_Code",
    "gelu-1l": "NeelNanda/GELU_1L512W_C4_Code",
    "gelu-2l": "NeelNanda/GELU_2L512W_C4_Code",
    "gelu-3l": "NeelNanda/GELU_3L512W_C4_Code",
    "gelu-4l": "NeelNanda/GELU_4L512W_C4_Code",
}

TOKENIZER_REPO_NAME = "NeelNanda/gpt-neox-tokenizer-digits"


class Transformer(th.nn.Module):
    def __init__(self, cfg: dict, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.cfg = cfg

        norm = cfg.get("normalization", cfg.get("normalization_type", "LN"))
        assert norm == "LN", f"{norm} normalization not yet implemented"
        assert cfg.get("use_attn_scale", True), "attention is always scaled"
        assert not cfg.get("use_local_attn", False), "local attention not implemented"
        assert (
            cfg.get("attention_dir", "causal") == "causal"
        ), "bidirectional attention not implemented"
        pos_embed = cfg.get(
            "positional_embedding_type",
            cfg.get("shortformer_pos", "standard") or "standard",
        )
        assert (
            pos_embed == "standard"
        ), "non-standard positional embeddings not implemented"

        self.embed = Embed(
            d_vocab=cfg["d_vocab"], d_model=cfg["d_model"], **factory_kwargs
        )
        self.pos_embed = PosEmbed(
            n_ctx=cfg["n_ctx"], d_model=cfg["d_model"], **factory_kwargs
        )
        if not cfg.get("attn_only", False):
            mlp_cfg = {"d_mlp": cfg["d_mlp"], "act_fn": cfg["act_fn"]}
        else:
            mlp_cfg = {}

        self.blocks = th.nn.ModuleList(
            [
                TransformerBlock(
                    n_heads=cfg["n_heads"],
                    d_model=cfg["d_model"],
                    d_head=cfg["d_head"],
                    n_ctx=cfg["n_ctx"],
                    ln_eps=cfg["ln_eps"],
                    attn_only=cfg.get("attn_only", False),
                    **mlp_cfg,
                    **factory_kwargs,
                )
                for _ in range(cfg["n_layers"])
            ]
        )
        self.ln_final = LayerNorm(
            d_model=cfg["d_model"], ln_eps=cfg["ln_eps"], **factory_kwargs
        )
        self.unembed = Unembed(
            d_model=cfg["d_model"], d_vocab=cfg["d_vocab"], **factory_kwargs
        )
        self.final_softmax = FinalSoftmax(
            d_model=cfg["d_model"], d_vocab=cfg["d_vocab"]
        )

    @property
    def device(self):
        return self.embed.W_E.device

    @property
    def dtype(self):
        return self.embed.W_E.dtype

    @contextmanager
    def to_temp(self, dtype=None, device=None):
        orig_dtype = self.dtype
        orig_device = self.device
        if orig_dtype.is_floating_point and dtype is not None:
            if (not dtype.is_floating_point) or (
                th.finfo(dtype).resolution > th.finfo(orig_dtype).resolution
            ):
                warnings.warn("Converting to lower-precision dtype")
        try:
            self.to(dtype=dtype, device=device)
            yield
        finally:
            self.to(dtype=orig_dtype, device=orig_device)

    @classmethod
    def create_empty(cls, model_name_or_path: str):
        if model_name_or_path.startswith("gs://"):
            cfg_path = bf.join(bf.dirname(model_name_or_path), "config.json")
        else:
            repo_name = MODEL_REPO_NAMES[model_name_or_path]
            cfg_path = hf_hub_download(repo_name, "config.json")

        with bf.BlobFile(cfg_path, "r") as cfg_fh:
            cfg = json.load(cfg_fh)
        model = cls(cfg)

        model.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
        assert (
            len(model.tokenizer.encode("")) == 0
        ), "tokenizer should not prepend or append anything"

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        checkpoint_path: Optional[str] = None,
        **blobfile_kwargs,
    ):
        model = cls.create_empty(model_name_or_path)

        if model_name_or_path.startswith("gs://"):
            state_dict_path = model_name_or_path
            if checkpoint_path is not None:
                raise ValueError("checkpoint_path is only for use with named models")
        else:
            if checkpoint_path is None:
                checkpoint_path = "model_final.pth"
            repo_name = MODEL_REPO_NAMES[model_name_or_path]
            state_dict_path = hf_hub_download(repo_name, checkpoint_path)

        with bf.BlobFile(state_dict_path, "rb", **blobfile_kwargs) as state_dict_fh:
            state_dict = th.load(state_dict_fh, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        return model

    def init_weights(self, scale_mult: float = 1.0, seed: Optional[int] = None):
        if seed is None:
            generator = None
        else:
            generator = th.Generator(device=self.device).manual_seed(seed)
        for name, param in self.named_parameters():
            rel_name = name.split(".")[-1]
            if rel_name.startswith("W_"):
                inv_var = {
                    "W_E": self.cfg["d_model"],
                    "W_pos": self.cfg["d_model"] * 4,
                    "W_Q": self.cfg["d_model"],
                    "W_K": self.cfg["d_model"],
                    "W_V": self.cfg["d_model"],
                    "W_O": self.cfg["n_heads"]
                    * self.cfg["d_head"]
                    * self.cfg["n_layers"],
                    "W_in": self.cfg["d_model"],
                    "W_out": self.cfg["d_mlp"] * self.cfg["n_layers"],
                    "W_U": self.cfg["d_model"],
                }[rel_name]
                scale = scale_mult * inv_var ** (-0.5)
                th.nn.init.normal_(param, std=scale, generator=generator)

    def encode(self, strings: List[str]) -> Int[th.Tensor, "batch pos"]:
        tokens = [self.tokenizer.encode(string) for string in strings]
        maxlen = max(len(t) for t in tokens)
        tokens = [t + [self.tokenizer.pad_token_id] * (maxlen - len(t)) for t in tokens]
        return th.tensor(tokens, device=self.device)

    def logits(
        self,
        tokens: Int[th.Tensor, "batch pos"],
    ) -> Float[th.Tensor, "batch pos d_vocab"]:
        x = self.embed(tokens)
        x = x + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.unembed(x)

    def sample(
        self,
        tokens: Int[th.Tensor, "batch pos"],
        completion_length: int = 1,
        *,
        temperature: float = 1.0,
    ):
        pad_token_id = self.tokenizer.pad_token_id
        last_non_pad_index = (tokens != pad_token_id).long().cumsum(1).max(1).indices
        pad_token = th.tensor(
            [[pad_token_id]] * tokens.shape[0], dtype=tokens.dtype, device=tokens.device
        )
        completion = th.zeros(
            tokens.shape[0], 0, dtype=tokens.dtype, device=tokens.device
        )
        for _ in range(completion_length):
            logits = self.logits(tokens)[th.arange(tokens.shape[0]), last_non_pad_index]
            if temperature == 0.0:
                next_token = logits.argmax(-1)
            else:
                logits = logits / temperature
                next_token = th.distributions.Categorical(logits=logits).sample()
            completion = th.cat([completion, next_token[:, None]], dim=1)
            last_non_pad_index += 1
            tokens = th.cat([tokens, pad_token], dim=1)
            tokens[th.arange(tokens.shape[0]), last_non_pad_index] = next_token
        return completion

    def probs(
        self,
        tokens: Int[th.Tensor, "batch pos"],
        *,
        temperature: float = 1.0
    ) -> Float[th.Tensor, "batch pos d_vocab"]:
        logits = self.logits(tokens)
        return self.final_softmax(logits, temperature=temperature)

@cache
def token_string_dict(tokenizer):
    """
    Returns the inverse of the tokenizer's vocab dict, but decodes each token
    explicitly to get rid of any encoded characters.
    """
    return {token: tokenizer.decode([token]) for token in tokenizer.vocab.values()}