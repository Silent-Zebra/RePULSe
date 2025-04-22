import torch as th
from torch.distributions import Categorical, constraints

from .datasets import get_token_dataset


class Discrete(Categorical):
    """
    A categorical distribution over a specific list of values.
    """

    def __init__(self, probs=None, logits=None, values=None, validate_args=None):
        super().__init__(probs, logits, validate_args)
        if values is None:
            values = th.arange(
                self._num_events, dtype=th.long, device=self._param.device
            )
        self.orig_values = values
        self.values = values.expand(self._param.shape)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        lower_bound = self.orig_values.min()
        upper_bound = self.orig_values.max()
        if th.is_floating_point(self.orig_values):
            support = constraints.interval(lower_bound, upper_bound)
            support.is_discrete = True
            return support
        else:
            return constraints.integer_interval(lower_bound, upper_bound)

    @property
    def mean(self):
        return (self.probs * self.values).sum(-1)

    @property
    def mode(self):
        lookup = self.values
        return th.gather(lookup, -1, self.probs.argmax(-1, keepdim=True)).squeeze(-1)

    @property
    def variance(self):
        return (self.probs * self.values**2).sum(-1) - self.mean**2

    def sample(self, sample_shape=th.Size()):
        samples = super().sample(sample_shape)
        lookup = self.values.expand(samples.shape + self.values.shape[-1:])
        return th.gather(lookup, -1, samples.unsqueeze(-1)).squeeze(-1)

    def boltzmann_distribution(self, scores, temperature=1.0):
        """Returns a new Discrete distribution where probabilities are re-weighted proportionally to exp(scores)"""
        assert scores.shape == (self._num_events,)
        scores = (scores - scores.max()) / temperature
        new_probs = self.probs * th.exp(scores)
        return Discrete(probs=new_probs, values=self.orig_values)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.unsqueeze(-1)
        value, lookup, log_pmf = th.broadcast_tensors(value, self.values, self.logits)
        masked_log_pmf = log_pmf.masked_fill(value != lookup, float("-inf"))
        return th.logsumexp(masked_log_pmf, -1)

    def enumerate_support(self, expand=True):
        if expand:
            support = self.values
        else:
            support = self.orig_values.view(
                (1,) * (self.values.ndim - self.orig_values.ndim)
                + self.orig_values.shape
            )
        return support.permute((support.ndim - 1,) + tuple(range(0, support.ndim - 1)))


def filter_dist(dist, allowed_values):
    assert dist.orig_values.ndim == 1
    mask = th.isin(dist.orig_values, allowed_values)
    return Discrete(probs=dist.probs[..., mask], values=dist.orig_values[..., mask])


def top_k_dist(dist, k):
    indices = dist.probs.argsort(-1, descending=True)[..., :k].sort(-1).values
    return Discrete(
        probs=th.gather(dist.probs, -1, indices),
        values=th.gather(dist.values, -1, indices),
    )


def mixture_dist(dists):
    assert all(dist.orig_values.ndim == 1 for dist in dists)
    assert all(dist.probs.shape[:-1] == dists[0].probs.shape[:-1] for dist in dists)
    values = th.unique(th.cat([dist.orig_values for dist in dists]))
    dist_probs = []
    for dist in dists:
        probs = dist.probs.unsqueeze(-1).expand(dist.probs.shape + values.shape)
        mask = dist.orig_values[:, None] != values[None]
        dist_probs.append(probs.masked_fill(mask, 0.0).sum(-2))
    return Discrete(probs=sum(dist_probs), values=values)


def constant_dist(value, dtype=None, device=None):
    return Discrete(
        logits=th.zeros((1,), dtype=dtype, device=device),
        values=th.tensor([value], device=device),
    )


def dataset_dist(
    dataset_name,
    tokenizer,
    *,
    n_items,
    truncate_tokens=2**14,
    dtype=None,
    device=None
):
    dataset = get_token_dataset(dataset_name, tokenizer, n_items=n_items)
    tokens = [item["tokens"][:truncate_tokens] for item in dataset]
    tokens = th.tensor([token for item in tokens for token in item])
    values, counts = th.unique(tokens, sorted=True, return_counts=True)
    counts = counts.to(dtype=dtype, device=device)
    values = values.to(device=device)
    return Discrete(probs=counts, values=values)


def plot_token_dist(dist, tokenizer, *, ax, cumulative=False):
    assert dist.probs.ndim == 1
    indices = dist.probs.argsort(descending=True)
    if cumulative:
        ax.plot(th.arange(len(indices)), dist.probs[indices].cumsum(-1))
        ax.set_ylabel("cumulative frequency")
    else:
        ax.plot(th.arange(len(indices)), dist.probs[indices])
        ax.set_yscale("log")
        ax.set_ylabel("frequency")
    ax.set_xticks(
        th.arange(len(indices)),
        [tokenizer.decode([token]) for token in dist.values[indices]],
        rotation="vertical",
        fontname="monospace",
    )
    ax.set_xlabel("token")
