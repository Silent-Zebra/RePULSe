import ast
import itertools
import re
from collections import Counter

from ..behavior import Behavior, constant_dist_maker
from ..datasets import get_dataset
from ..discrete import dataset_dist, top_k_dist


def ifelse_behavior(tokenizer, *, n_items=1024, n_values=1024, dtype=None, device=None):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)
    code_dist = dataset_dist(
        "code", tokenizer, n_items=n_items, dtype=dtype, device=device
    )
    code_dist = top_k_dist(code_dist, n_values)
    return Behavior(
        input_dists_distinct=[
            constant(tokenizer.bos_token_id),
            constant(" if"),
            code_dist,
        ],
        rep_range=(-1, 0),
        output_tokens=constant(" else").values,
    )


def ifelse_clean(tokenizer, *, n_items=8192):
    strings = [item["text"] for item in itertools.islice(get_dataset("code"), n_items)]
    exprs = [
        expr for string in strings for expr in re.findall(r" if(.*?) else", string)
    ]
    valid_exprs = []
    for expr in exprs:
        try:
            ast.parse(expr.strip(), mode="eval")
            valid_exprs.append(expr)
        except SyntaxError:
            pass
    for expr in valid_exprs:
        tokens = tokenizer.encode(f" if{expr} else")
        assert tokens[:1] == tokenizer.encode(" if")
        assert tokens[-1:] == tokenizer.encode(" else")
    valid_counts = sorted(Counter(valid_exprs).items(), key=lambda p: (-p[1], p[0]))
    return [((" if", expr, " else"), count) for expr, count in valid_counts]
