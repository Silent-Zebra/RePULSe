import itertools
import re
from collections import Counter
from functools import cache

import torch as th

from ..behavior import Behavior, constant_dist_maker
from ..datasets import get_dataset
from ..discrete import Discrete
from ..transformer import token_string_dict

HEX_CHARS = "".join(f"{i:x}" for i in range(16))


def is_hex(string):
    return len(string) > 0 and all(char in HEX_CHARS for char in string)


@cache
def get_approx_long_run_freqs(strings, alphabet_size):
    n_strings = len(strings)
    probs = th.zeros((n_strings, n_strings))
    for prev_i, prev_string in enumerate(strings):
        for next_i, next_string in enumerate(strings):
            matches = []
            for other_string in strings:
                if other_string == next_string:
                    continue
                if other_string.startswith(next_string):
                    matches.append(other_string[len(next_string) :])
                if other_string.startswith(prev_string + next_string):
                    matches.append(other_string[len(prev_string) + len(next_string) :])
            matches = [
                match
                for match in matches
                if not any(
                    match.startswith(other_match)
                    for other_match in matches
                    if other_match != match
                )
            ]
            probs[prev_i, next_i] = alphabet_size ** (-len(next_string)) * (
                1 - sum(alphabet_size ** (-len(match)) for match in matches)
            )
    probs = probs / probs.sum(1, keepdim=True)
    A = th.eye(n_strings) - probs.T
    A[-1, :] = 1
    b = th.zeros(n_strings)
    b[-1] = 1
    return th.linalg.solve(A, b)


def hex_behavior(tokenizer, *, dtype=None, device=None):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)
    hex_tokens = [
        token
        for token, string in sorted(token_string_dict(tokenizer).items())
        if is_hex(string)
    ]
    hex_strings = tuple([tokenizer.decode([token]) for token in hex_tokens])
    hex_dist = Discrete(
        probs=get_approx_long_run_freqs(hex_strings, len(HEX_CHARS)).to(
            dtype=dtype, device=device
        ),
        values=th.tensor(hex_tokens, device=device),
    )
    return Behavior(
        input_dists_distinct=[constant(tokenizer.bos_token_id), hex_dist],
        rep_range=(-1, 0),
        output_tokens=hex_dist.values,
    )


def hex_clean(tokenizer, *, n_items=8192):
    strings = [item["text"] for item in itertools.islice(get_dataset("code"), n_items)]
    hex_strings = []
    for string in strings:
        for line in string.split("\n"):
            spans = []
            for m in re.finditer(r"[0-9a-f]{8,}", line):
                if (
                    len([char for char in m[0] if char in HEX_CHARS[:10]]) >= 2
                    and len([char for char in m[0] if char in HEX_CHARS[10:]]) >= 2
                ):
                    start, end = m.span()
                    if tokenizer.encode(line) == tokenizer.encode(
                        line[:start]
                    ) + tokenizer.encode(line[start:end]) + tokenizer.encode(
                        line[end:]
                    ):
                        spans.append((start, end))
            if len(spans) > 0:
                parts = []
                prev_end = None
                for start, end in spans:
                    if prev_end is None:
                        parts.append(
                            tokenizer.decode(tokenizer.encode("\n" + line[:start])[1:])
                        )
                    else:
                        parts.append(line[prev_end:start])
                    parts.append(line[start:end])
                    prev_end = end
                hex_strings.append(tuple(parts))
    return sorted(Counter(hex_strings).items(), key=lambda p: (-p[1], p[0]))


# Older version of hex_clean, may not be needed anymore
def hex_clean_synthetic(*, n_strings=2**13, n_chars=2**6, seed=0):
    generator = None if seed is None else th.Generator().manual_seed(seed)
    n_chars_total = n_strings * n_chars
    all_chars = "".join(
        [HEX_CHARS[i] for i in th.randint(0, 16, (n_chars_total,), generator=generator)]
    )
    strings = [all_chars[i : i + n_chars] for i in range(0, n_chars_total, n_chars)]
    assert len(strings) == len(set(strings))
    return [((string,), 1) for string in strings]
