"""Lightweight diversity metrics for token sequences.

Only depends on standard library (collections.Counter, math).
"""

import math
from collections import Counter


def _extract_ngrams(tokens, n):
    """Extract n-grams from a list of token IDs.

    Args:
        tokens: List of token IDs (ints).
        n: n-gram order (e.g. 1 for unigrams, 2 for bigrams).

    Returns:
        List of n-gram tuples.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_distinct_n(sequences, n):
    """Compute distinct-n: #unique n-grams / #total n-grams, pooled across sequences.

    Args:
        sequences: List of token ID lists.
        n: n-gram order.

    Returns:
        Float in [0, 1]. Returns 0.0 if there are no n-grams.
    """
    all_ngrams = []
    for seq in sequences:
        all_ngrams.extend(_extract_ngrams(seq, n))
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_self_bleu(sequences, max_n=4):
    """Compute self-BLEU: average pairwise BLEU similarity among sequences.

    For each sequence, treats it as the "candidate" and all other sequences as
    "references", computes a smoothed BLEU score, then averages over all sequences.
    High self-BLEU (~1) means the sequences are very similar to each other (low diversity).
    Low self-BLEU (~0) means the sequences are very different (high diversity).

    Uses modified precision with add-1 (Laplace) smoothing to avoid zero precision
    when n-gram matches are sparse:
        precision_n = (clipped_matches + 1) / (num_candidate_ngrams + 1)
    Brevity penalty uses closest reference length (standard BLEU).
    BLEU = brevity_penalty * exp(mean of log precisions for n=1..max_n).

    Args:
        sequences: List of token ID lists.
        max_n: Maximum n-gram order (e.g. 2 for self-BLEU-2, 4 for self-BLEU-4).

    Returns:
        Float in [0, 1]. Returns 0.0 if fewer than 2 sequences.
    """
    if len(sequences) < 2:
        return 0.0

    bleu_scores = []
    for i in range(len(sequences)):
        candidate = sequences[i]
        references = [sequences[j] for j in range(len(sequences)) if j != i]
        score = _sentence_bleu_smoothed(candidate, references, max_n)
        bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores)


def _sentence_bleu_smoothed(candidate, references, max_n):
    """Compute smoothed BLEU score for one candidate against multiple references.

    Args:
        candidate: List of token IDs.
        references: List of token ID lists.
        max_n: Maximum n-gram order.

    Returns:
        Float BLEU score.
    """
    cand_len = len(candidate)
    if cand_len == 0:
        return 0.0

    # Brevity penalty (BP): penalizes candidates shorter than the closest reference.
    # BP = exp(1 - closest_ref_len / cand_len) if cand_len <= closest_ref_len, else 1.0
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - cand_len), r))
    if cand_len <= closest_ref_len:
        brevity_penalty = math.exp(1.0 - closest_ref_len / cand_len) if cand_len > 0 else 0.0
    else:
        brevity_penalty = 1.0

    # Compute modified precision for each n-gram order 1..max_n
    log_precisions = []
    for n in range(1, max_n + 1):
        candidate_ngrams = _extract_ngrams(candidate, n)
        if len(candidate_ngrams) == 0:
            # No n-grams of this order (candidate too short); smoothed precision = 1/(0+1) = 1
            log_precisions.append(math.log(1.0 / (0 + 1)))
            continue

        candidate_ngram_counts = Counter(candidate_ngrams)

        # For each n-gram, find the maximum count across all references (standard BLEU clipping)
        max_ref_ngram_counts = Counter()
        for ref in references:
            ref_ngram_counts = Counter(_extract_ngrams(ref, n))
            for ngram, count in ref_ngram_counts.items():
                max_ref_ngram_counts[ngram] = max(max_ref_ngram_counts[ngram], count)

        # Clipped matches: for each candidate n-gram, count up to the max reference count
        clipped_match_count = 0
        for ngram, count in candidate_ngram_counts.items():
            clipped_match_count += min(count, max_ref_ngram_counts.get(ngram, 0))

        # Add-1 (Laplace) smoothing avoids log(0) when there are no matches
        smoothed_precision = (clipped_match_count + 1) / (len(candidate_ngrams) + 1)
        log_precisions.append(math.log(smoothed_precision))

    # BLEU = BP * exp(average of log precisions across n-gram orders)
    avg_log_precision = sum(log_precisions) / len(log_precisions)
    return brevity_penalty * math.exp(avg_log_precision)
