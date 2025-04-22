from string import ascii_uppercase

import torch as th

from ..behavior import Behavior, NoRepsBehavior, constant_dist_maker
from ..discrete import Discrete, dataset_dist, filter_dist
from ..transformer import token_string_dict


def icl_behavior(tokenizer, *, n_items=2**14, dtype=None, device=None):
    constant = constant_dist_maker(tokenizer, dtype=dtype, device=device)

    for_dist = constant(" for")

    IN_CONTEXT = "ARCTHEORY"

    letter_tokens = []
    for letter in ascii_uppercase:
        [letter_token] = tokenizer.encode(" " + letter)
        letter_tokens.append(letter_token)

    c4_dist = dataset_dist("c4", tokenizer, n_items=n_items, dtype=dtype, device=device)

    distribution_list = [constant(tokenizer.bos_token_id)]
    for i, letter in enumerate(IN_CONTEXT):
        if i == 0:
            distribution_list.append(constant(letter))
        else:
            distribution_list.append(constant(" " + letter))

        distribution_list.append(for_dist)

        starting_letter_tokens = [
            token
            for token, string in sorted(token_string_dict(tokenizer).items())
            if len(string) >= 5
            and string[0] == " "
            and string[1].upper() == letter
            and string[1:].isalpha()
        ]
        distribution_list.append(
            filter_dist(c4_dist, th.tensor(starting_letter_tokens, device=device))
        )

    query_dist = Discrete(
        logits=th.zeros(len(letter_tokens), dtype=dtype, device=device),
        values=th.tensor(letter_tokens, device=device),
    )
    distribution_list.append(query_dist)
    distribution_list.append(for_dist)

    return NoRepsBehavior(
        input_dists_distinct=distribution_list,
        # we won't actually ever use rep_range or output_tokens
        rep_range=(-1, 0),
        output_tokens=th.tensor(letter_tokens),
    )
