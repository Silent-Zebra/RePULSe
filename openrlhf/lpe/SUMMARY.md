# Evaluation Summary

This document provides a summary of the scripts available in the `openrlhf/lpe` directory.

## `load_lpe_distributions.py`

### Purpose

This script provides utility functions to create token-factorized input probability distributions, primarily intended for use with Latent Prompt Engineering (LPE) methods. It generates a list of discrete probability distributions, one for each token position in a sequence.

### Functions

*   **`get_hex_token_ids(tokenizer: PreTrainedTokenizer) -> Optional[torch.Tensor]`**:
    *   **Description:** Identifies and returns the unique token IDs corresponding to hexadecimal characters (`0`-`9`, `a`-`f`, `A`-`F`) within the vocabulary of the provided tokenizer. It handles potential variations in tokenizer encoding (e.g., prefixes like ' ').
    *   **Returns:** A `torch.Tensor` containing the unique hex token IDs, or `None` if none are found.

*   **`get_token_factorized_distribution(distribution_name: str, sequence_length: int, tokenizer: PreTrainedTokenizer, device: torch.device) -> List[Discrete]`**:
    *   **Description:** Creates and returns a list of `Discrete` probability distribution objects (one for each sequence position) based on the specified `distribution_name`.
    *   **Arguments:**
        *   `distribution_name`: Specifies the type of distribution ("uniform", "hex").
        *   `sequence_length`: The length of the token sequence for which to generate distributions.
        *   `tokenizer`: The model's tokenizer.
        *   `device`: The torch device for tensor creation.
    *   **Returns:** A list of `openrlhf.lpe.utils.Discrete` objects.
    *   **Workflow/Logic:**
        *   If `distribution_name` is "uniform", it creates uniform distributions over the entire vocabulary.
        *   If `distribution_name` is "hex", it uses `get_hex_token_ids` to find hex tokens and creates uniform distributions over only those tokens. Raises `ValueError` if no hex tokens are found.
        *   Raises `ValueError` for unknown distribution names.

### Dependencies

*   `torch`
*   `transformers.PreTrainedTokenizer`
*   `openrlhf.lpe.utils.Discrete` (conditionally imported, with a warning and dummy class if unavailable)

### Use Case

This script serves as a utility for setting up input distributions required by LPE algorithms or evaluations. It allows defining how initial latent prompt tokens should be sampled, such as uniformly across all possibilities or restricted to specific subsets like hexadecimal characters.

## `harmfulness_metrics.py`

### Purpose

This script provides basic functions to assess the potential "harmfulness" or undesirability of generated token sequences. Currently, its primary method involves checking for the presence of a specific, user-defined "bad" token ID within a sequence.

### Configuration

*   **`TARGET_BAD_TOKEN_ID` (Global Variable):** This integer variable must be manually set by the user to the specific token ID that should be considered "bad" or undesirable. It defaults to `-1` (placeholder), and the script will issue a warning if it's used without being properly set.

### Functions

*   **`check_bad_token(output_sequence: torch.Tensor, tokenizer: PreTrainedTokenizer, target_token_id: int = TARGET_BAD_TOKEN_ID) -> bool`**:
    *   **Description:** Checks if the specified `target_token_id` exists anywhere within the input `output_sequence` tensor.
    *   **Arguments:**
        *   `output_sequence`: The generated sequence of token IDs (as a `torch.Tensor`).
        *   `tokenizer`: The model's tokenizer (currently unused in the function logic but included in the signature).
        *   `target_token_id`: The integer token ID to search for. Defaults to the global `TARGET_BAD_TOKEN_ID`.
    *   **Returns:** `True` if the `target_token_id` is found in the sequence, `False` otherwise. Issues a warning if the default `TARGET_BAD_TOKEN_ID` is used.

*   **`check_bad` (Alias):**
    *   **Description:** Currently, this is an alias directly pointing to the `check_bad_token` function. Commented-out code suggests it could potentially be expanded into a dispatcher calling multiple different check functions (e.g., token-based, regex-based).

### Dependencies

*   `torch`
*   `transformers.PreTrainedTokenizer`

### Use Case

This script provides a simple metric for evaluating generated sequences, specifically focusing on detecting undesirable content represented by a single token ID. It's likely intended for use in evaluation scripts or within algorithms (like LPE) that require a function to identify "bad" outputs based on simple criteria. The effectiveness depends on the user correctly identifying and setting the `TARGET_BAD_TOKEN_ID`.
