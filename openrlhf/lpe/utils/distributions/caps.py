import torch as th

from ..behavior import Behavior, constant_dist_maker
from ..discrete import Discrete, dataset_dist, filter_dist, top_k_dist
from ..transformer import token_string_dict


def is_all_caps(string):
    return len(string) > 1 and all(
        char.isupper() or char in ",.?!:;" for char in string
    )


def caps_behavior(tokenizer, *, n_items=1024, n_values=1024, dtype=None, device=None):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)
    caps_tokens = [
        token
        for token, string in sorted(token_string_dict(tokenizer).items())
        if is_all_caps(string)
    ]
    nlp_dist = dataset_dist(
        "c4", tokenizer, n_items=n_items, dtype=dtype, device=device
    )
    caps_dist = filter_dist(nlp_dist, th.tensor(caps_tokens, device=device))
    caps_dist = top_k_dist(caps_dist, n_values)

    [he_token] = tokenizer.encode("He")
    [she_token] = tokenizer.encode("She")
    heshe_distribution = Discrete(
        probs=th.tensor([0.5, 0.5], dtype=dtype, device=device),
        values=th.tensor([he_token, she_token], device=device),
    )

    middle_tokens = th.tensor(tokenizer.encode(' screamed: "'))
    middle_dists = [constant(x) for x in middle_tokens]

    return Behavior(
        input_dists_distinct=[
            constant(tokenizer.bos_token_id),
            heshe_distribution,
        ]
        + middle_dists
        + [
            caps_dist,
        ],
        rep_range=(-1, 0),
        output_tokens=th.tensor(caps_tokens),
    )
