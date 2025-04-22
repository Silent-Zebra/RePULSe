import torch as th

from ..behavior import Behavior, constant_dist_maker
from ..discrete import dataset_dist, filter_dist, top_k_dist
from ..transformer import token_string_dict


def spanish_behavior(
    tokenizer, *, n_items=2**14, n_values=1024, dtype=None, device=None
):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)
    spanish_dist = dataset_dist(
        "c4_es", tokenizer, n_items=n_items, dtype=dtype, device=device
    )

    alpha_tokens = [
        token
        for token, string in sorted(token_string_dict(tokenizer).items())
        if all(char.isalpha() or char.isspace() for char in string)
    ]

    spanish_dist = filter_dist(spanish_dist, th.tensor(alpha_tokens, device=device))
    spanish_dist = top_k_dist(spanish_dist, n_values)

    return Behavior(
        input_dists_distinct=[
            constant(tokenizer.bos_token_id),
            spanish_dist,
        ],
        rep_range=(-1, 0),
        output_tokens=th.tensor(alpha_tokens),
    )
