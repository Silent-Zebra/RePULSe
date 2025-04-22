import itertools
from collections import Counter

import torch as th

from ..behavior import Behavior, constant_dist_maker
from ..datasets import get_dataset
from ..discrete import dataset_dist, top_k_dist
from ..transformer import token_string_dict


def is_indent(string):
    return (
        len(string) > 1
        and string[0] == "\n"
        and all(char == " " for char in string[1:])
    )


def indent_behavior(tokenizer, *, n_items=1024, n_values=1024, dtype=None, device=None):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)
    code_dist = dataset_dist(
        "code", tokenizer, n_items=n_items, dtype=dtype, device=device
    )
    code_dist = top_k_dist(code_dist, n_values)
    indent_tokens = [
        token
        for token, string in sorted(token_string_dict(tokenizer).items())
        if is_indent(string)
    ]
    return Behavior(
        input_dists_distinct=[
            constant(tokenizer.bos_token_id),
            code_dist,
            constant(":"),
        ],
        rep_range=(1, 2),
        output_tokens=th.tensor(indent_tokens),
    )


def indent_clean(tokenizer, *, n_items=8192):
    strings = [item["text"] for item in itertools.islice(get_dataset("code"), n_items)]
    line_pairs = [
        line_pair
        for string in strings
        for line_pair in list(zip(string.split("\n")[:-1], string.split("\n")[1:]))
    ]
    colon_strings = [
        (f"\n{line1[:-1]}", ":", tokenizer.decode(tokenizer.encode("\n" + line2)[:1]))
        for line1, line2 in line_pairs
        if len(line1) > 0
        and line1[-1] == ":"
        and tokenizer.encode(line1)[-1:] == tokenizer.encode(":")
    ]
    for string1, string2, string3 in colon_strings:
        assert [
            tokenizer.encode(f"{string1}{string2}{string3}")[-2]
        ] == tokenizer.encode(":")
    return sorted(Counter(colon_strings).items(), key=lambda p: (-p[1], p[0]))
