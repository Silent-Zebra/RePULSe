from dataclasses import dataclass
from inspect import signature
from typing import Callable, Optional

import torch as th
from jaxtyping import Int

from .discrete import Discrete, constant_dist


@dataclass
class Behavior:
    """
    The behavior encodes the question: "What's the probability the model outputs a token in `output_tokens` given that the input tokens are sampled from the input distribution?"

    At the moment, the family of token input distribution must have the following form.
    - `input_dists_distinct` is a list of Discrete distributions. Say it's [D_0, D_1, \\dots, D_{n-1}].
    - `rep_range` is a tuple (a, b) where a and b are indices in [n] (if a<0 or b<=0, they are interpreted as indexing from back).
    - Then, the input distribution induced by `self.input_dists(l)` is the (independent) product of
        `D_0 x D_1 x ... x \\underbrace{D_a x ... x D_{b-1} x D_{b-1}}_{length l} x D_b x ... x D_{n-1}`
        (note that D_{b-1} is repeated l+1-(b-a) times)
    """

    input_dists_distinct: list[Discrete]
    rep_range: tuple[int, int]
    output_tokens: Int[th.Tensor, "tokens"]

    @property
    def rep_start(self):
        rep_start = self.rep_range[0]
        if rep_start < 0:
            rep_start = rep_start + len(self.input_dists_distinct)
        return rep_start

    @property
    def rep_end(self):
        rep_end = self.rep_range[1]
        if rep_end <= 0:
            rep_end = rep_end + len(self.input_dists_distinct)
        return rep_end

    @property
    def n_nonreps(self):
        return len(self.input_dists_distinct) - (self.rep_end - self.rep_start)

    def input_dists(self, n_reps: int) -> list[Discrete]:
        rep_dists = self.input_dists_distinct[self.rep_start : self.rep_end][:n_reps]
        rep_dists = rep_dists + rep_dists[-1:] * (n_reps - len(rep_dists))
        return (
            self.input_dists_distinct[: self.rep_start]
            + rep_dists
            + self.input_dists_distinct[self.rep_end :]
        )

    def sample(self, n_reps: int, *, n_samples: int) -> th.Tensor:
        return th.stack(
            [dist.sample((n_samples,)) for dist in self.input_dists(n_reps)], dim=1
        )

def constant_dist_maker(tokenizer, dtype=None, device=None):
    def make_constant_dist(token_or_string):
        if isinstance(token_or_string, str):
            [token] = tokenizer.encode(token_or_string)
        else:
            token = token_or_string
        return constant_dist(token, dtype=dtype, device=device)

    return make_constant_dist


class NoRepsBehavior(Behavior):
    def input_dists(self, n_reps: int) -> list[Discrete]:
        if n_reps == 1:
            return super().input_dists(n_reps)
        else:
            raise ValueError("No reps allowed for this behavior")
