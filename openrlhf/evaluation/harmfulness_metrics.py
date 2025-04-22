import torch
from transformers import PreTrainedTokenizer

# Placeholder for a specific "bad" token ID. Replace with actual ID later.
# Example: Find the ID for an undesirable word using tokenizer.encode(" undesirable_word")[0]
TARGET_BAD_TOKEN_ID = -1 # Replace this

def check_bad_token(output_sequence: torch.Tensor, tokenizer: PreTrainedTokenizer, target_token_id: int = TARGET_BAD_TOKEN_ID) -> bool:
    """
    Checks if a specific target 'bad' token ID appears in the output sequence.

    Args:
        output_sequence: A tensor of token IDs representing the generated sequence (e.g., shape [seq_len] or [batch, seq_len]).
        tokenizer: The tokenizer used for the model (unused in this simple version but kept for interface consistency).
        target_token_id: The integer ID of the token considered 'bad'.

    Returns:
        True if the target bad token ID is found in the sequence, False otherwise.
    """
    if target_token_id == -1:
        # Remind the user to set the actual target token ID
        print("Warning: TARGET_BAD_TOKEN_ID is not set in harmfulness_metrics.py. Please replace the placeholder.")
        # Defaulting to False to avoid accidental positives during setup
        return False

    # Ensure we are working with CPU tensor for comparison
    if output_sequence.is_cuda:
        output_sequence = output_sequence.cpu()

    # Check if the target token ID exists anywhere in the sequence
    found = torch.any(output_sequence == target_token_id).item()
    return found

# Example of how to potentially add more complex checks later:
# def check_bad_regex(output_sequence: torch.Tensor, tokenizer: PreTrainedTokenizer, pattern: str) -> bool:
#     decoded_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
#     import re
#     return bool(re.search(pattern, decoded_text))

# You can create a main check_bad function that calls specific checks:
# def check_bad(output_sequence: torch.Tensor, tokenizer: PreTrainedTokenizer) -> bool:
#     if check_bad_token(output_sequence, tokenizer):
#         return True
#     # if check_bad_regex(output_sequence, tokenizer, "some_regex_pattern"):
#     #     return True
#     # Add calls to other checks as needed
#     return False

# For now, let's alias the simple check as the main one for LPE integration
check_bad = check_bad_token 