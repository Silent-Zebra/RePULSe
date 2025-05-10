import torch
from transformers import PreTrainedTokenizer
from typing import List, Optional
import warnings

# Attempt to import from lpe, assuming it's installed or in the path
try:
    from openrlhf.lpe.utils import Discrete
except ImportError:
    warnings.warn(
        "Could not import 'Discrete' from 'openrlhf.lpe.utils'. "
        "Ensure the 'lpe' directory is inside 'openrlhf' and contains an __init__.py."
    )
    # Define a dummy class to avoid errors during initial setup, users must fix the import
    class Discrete:
        def __init__(self, probs=None, logits=None, values=None, validate_args=None):
            print("Warning: Using dummy Discrete class. LPE functionality will not work.")
            pass


def get_hex_token_ids(tokenizer: PreTrainedTokenizer) -> Optional[torch.Tensor]:
    """Finds the token IDs corresponding to hexadecimal characters."""
    hex_chars = '0123456789abcdefABCDEF'
    hex_token_ids = []
    for char in hex_chars:
        # Tokenizers might handle single characters differently (e.g., adding prefixes like ' ')
        # Try encoding common variations
        encodings = [char, f" {char}"] # Add more variations if needed based on tokenizer behavior
        for enc in encodings:
            tokens = tokenizer.encode(enc, add_special_tokens=False)
            # Sometimes encoding a single char results in multiple tokens, ignore those complex cases for now
            # Also, ensure the token is not a special token (like UNK, PAD, etc.)
            if len(tokens) == 1 and tokens[0] < tokenizer.vocab_size:
                 hex_token_ids.append(tokens[0])

    if not hex_token_ids:
        warnings.warn("Could not find any token IDs for hex characters. 'hex' distribution will be empty.")
        return None

    # Get unique IDs and sort them
    unique_ids = torch.tensor(list(set(hex_token_ids)), dtype=torch.long)
    return unique_ids

def get_token_factorized_distribution(
    distribution_name: str,
    sequence_length: int,
    tokenizer: PreTrainedTokenizer,
    device: torch.device = torch.device("cpu")
) -> List[Discrete]:
    """
    Creates a token-factorized input distribution for LPE methods.

    Args:
        distribution_name: The name of the distribution ('uniform', 'hex', etc.).
        sequence_length: The desired length of the input sequences.
        tokenizer: The tokenizer used for the model.
        device: The torch device for the tensors.

    Returns:
        A list of lpe.utils.Discrete objects, one for each position.
    """
    vocab_size = tokenizer.vocab_size
    all_token_ids = torch.arange(vocab_size, dtype=torch.long, device=device)

    distributions = []
    if distribution_name == "uniform":
        # Uniform probability over all tokens
        probs = torch.ones(vocab_size, device=device) / vocab_size
        for _ in range(sequence_length):
            distributions.append(Discrete(probs=probs, values=all_token_ids))

    elif distribution_name == "hex":
        hex_ids = get_hex_token_ids(tokenizer)
        if hex_ids is None or len(hex_ids) == 0:
            raise ValueError("Cannot create 'hex' distribution: no hex token IDs found.")
        hex_ids = hex_ids.to(device)
        num_hex_ids = len(hex_ids)
        # Uniform probability over only hex tokens
        probs = torch.ones(num_hex_ids, device=device) / num_hex_ids
        for _ in range(sequence_length):
            distributions.append(Discrete(probs=probs, values=hex_ids))

    # elif distribution_name == "english_freq":
    #     # Placeholder: Implement logic to load frequencies from a corpus
    #     # Example: use lpe.utils.dataset_dist if applicable
    #     warnings.warn("'english_freq' distribution not implemented, falling back to uniform.")
    #     probs = torch.ones(vocab_size, device=device) / vocab_size
    #     for _ in range(sequence_length):
    #         distributions.append(Discrete(probs=probs, values=all_token_ids))

    else:
        raise ValueError(f"Unknown distribution name: {distribution_name}")

    return distributions 