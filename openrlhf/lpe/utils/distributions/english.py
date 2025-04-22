import torch as th

from ..behavior import Behavior, constant_dist_maker
from ..discrete import dataset_dist, filter_dist, top_k_dist
from ..transformer import token_string_dict


def is_middle_word_or_punc(word):
    return (
        len(word) > 1
        and all(char.isalpha() or char in ",.?!:;" for char in word[1:])
        and word[0] == " "
    ) or (word in [",", ".", "!", "?", ";"])


def is_starting_word(word):
    return all(char.isalpha() for char in word)


def english_behavior(
    tokenizer, *, n_items=2**14, n_values=2**14, dtype=None, device=None
):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)
    english_dist = dataset_dist(
        "c4", tokenizer, n_items=n_items, dtype=dtype, device=device
    )

    starting_word_tokens = [
        token
        for token, string in sorted(token_string_dict(tokenizer).items())
        if is_starting_word(string)
    ]
    middle_word_tokens = [
        token
        for token, string in sorted(token_string_dict(tokenizer).items())
        if is_middle_word_or_punc(string)
    ]

    starting_word_english_dist = filter_dist(
        english_dist, th.tensor(starting_word_tokens, device=device)
    )
    starting_word_english_dist = top_k_dist(starting_word_english_dist, n_values)
    middle_word_english_dist = filter_dist(
        english_dist, th.tensor(middle_word_tokens, device=device)
    )
    middle_word_english_dist = top_k_dist(middle_word_english_dist, n_values)

    return Behavior(
        input_dists_distinct=[
            constant(tokenizer.bos_token_id),
            starting_word_english_dist,
            middle_word_english_dist,
        ],
        rep_range=(-1, 0),
        output_tokens=th.tensor(middle_word_tokens),
    )
