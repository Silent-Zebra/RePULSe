import itertools
import re
from collections import Counter

import torch as th

from ..behavior import Behavior, constant_dist_maker
from ..datasets import get_dataset
from ..discrete import dataset_dist, filter_dist
from ..transformer import token_string_dict


def is_camel(string):
    return (
        len(string) > 1
        and string[0].isupper()
        and string[1:].islower()
        and " " not in string
        and string not in ["None", "True", "False"]
    )


def camel_behavior(tokenizer, *, n_items=1024, dtype=None, device=None):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)
    camel_tokens = [
        token
        for token, string in sorted(token_string_dict(tokenizer).items())
        if is_camel(string)
    ]
    code_dist = dataset_dist(
        "code", tokenizer, n_items=n_items, dtype=dtype, device=device
    )
    camel_dist = filter_dist(code_dist, th.tensor(camel_tokens, device=device))
    return Behavior(
        input_dists_distinct=[constant(tokenizer.bos_token_id), camel_dist],
        rep_range=(-1, 0),
        output_tokens=camel_dist.values,
    )


def camel_clean(tokenizer, *, n_items=8192):
    strings = [item["text"] for item in itertools.islice(get_dataset("code"), n_items)]
    camel_pattern = r"\b[A-Z](?=[A-Za-z]*[A-Z])(?=[A-Za-z]*[a-z])[A-Za-z]*\b"
    camels = [
        camel_string
        for string in strings
        for camel_string in re.findall(camel_pattern, string)
    ]
    camel_counts = sorted(Counter(camels).items(), key=lambda p: (-p[1], p[0]))
    split_camel_counts = []
    for string, count in camel_counts:
        tokens = tokenizer.encode(string)
        indices = [
            i
            for i, token in enumerate(tokens)
            if tokenizer.decode([token])[0].isupper()
        ] + [len(tokens)]
        split_string = tuple(
            [
                tokenizer.decode(tokens[indices[i] : indices[i + 1]])
                for i in range(len(indices) - 1)
            ]
        )
        split_camel_counts.append((split_string, count))
    return split_camel_counts
