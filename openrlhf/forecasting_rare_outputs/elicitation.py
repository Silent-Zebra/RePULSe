import math
import torch
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Callable, Dict, Any
from openrlhf.models import Actor, ActorCustom
from .behavior_defs import Behavior

def estimate_keyword_probability(
    model: Actor,
    tokenizer: PreTrainedTokenizer,
    query: str,
    keyword: str,
    k_samples: int = 100, # Use sampling as default estimation method
    **generation_kwargs
) -> float:
    """Estimates the probability of the model generating the keyword in response to the query.
    
    NOTE: This is a complex estimation problem. This implementation uses a basic
    Monte Carlo approach (repeated sampling) as a default. More sophisticated methods
    (e.g., marginalizing over positions, beam search analysis) could be implemented but
    are significantly more complex.
    """
    print(f"Warning: estimate_keyword_probability using repeated sampling (k={k_samples}). This can be slow and approximate.")

    # Create a temporary behavior definition for the keyword check
    # temp_behavior_def = { # Old dictionary approach
    #     'type': 'keyword',
    #     'target': keyword
    # }
    # New approach: Create a Behavior object
    temp_behavior_obj = Behavior(
        id='temp_keyword_check', 
        name='Temporary Keyword Check',
        description='Internal object for estimate_keyword_probability',
        type='keyword',
        target=keyword, # estimate_keyword_probability designed for single keyword string
        goal='Temporary goal for internal check' # Placeholder
        # default_elicitation_method can be None or omitted if not needed by get_behavior_check_function's path for 'keyword'
    )
    keyword_check_fn = get_behavior_check_function(temp_behavior_obj) # Pass the object

    # Generate samples
    outputs = generate_output(model, tokenizer, query, k_samples, **generation_kwargs)

    if not outputs: # Handle cases where no outputs were generated
        return 0.0

    # # Old approach to Count successes
    # success_count = 0
    # for output in outputs:
    #     if keyword_check_fn(output):
    #         success_count += 1

    # Pandas vectorized approach:
    output_series = pd.Series(outputs)
    keyword_lower = keyword.lower() # The keyword to check, lowercased once

    # .str.lower() converts all outputs to lowercase
    # .str.contains(keyword_lower, regex=False) checks for substring presence
    # .sum() counts boolean True values (which convert to 1)
    success_count = output_series.str.lower().str.contains(keyword_lower, regex=False).sum()

    if k_samples == 0: # Should ideally not happen if outputs is not empty, but good for safety
        return 0.0
    return success_count / k_samples


def generate_output(model: Actor, tokenizer: PreTrainedTokenizer, query: str, k_samples: int, **generation_kwargs) -> list[str]:
    """Generates k_samples outputs for the given query using the Actor model."""
    model.eval()

    # Prepare input: Tokenize query. Actor.generate expects input_ids.
    # Chat templating should be handled by the user if desired, by formatting `query` before passing it here.
    input_tokens = tokenizer(query, return_tensors="pt", padding=False, truncation=True, max_length=tokenizer.model_max_length - generation_kwargs.get("max_new_tokens", 256) -5) # Reserve space for new tokens and special tokens
    
    try:
        device = model.model.device if hasattr(model, 'model') and hasattr(model.model, 'device') else next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_ids = input_tokens["input_ids"].to(device)
    attention_mask = input_tokens["attention_mask"].to(device) # Actor.generate can infer this

    # Default generation args (can be overridden by **generation_kwargs)
    # Actor's generate will use its own defaults if not specified.
    default_gen_kwargs = {
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_p": 0.9,
        "do_sample": True,
        "num_return_sequences": k_samples,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    final_gen_kwargs = {**default_gen_kwargs, **generation_kwargs}
    
    # Ensure pad_token_id is set if generation needs it (e.g. for batching or beam search)
    if final_gen_kwargs.get("pad_token_id") is None:
        final_gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
        # print("generate_output: Setting pad_token_id to eos_token_id as it was None.")


    generated_outputs_decoded = []
    with torch.no_grad():
        try:
            # Actor.generate returns: sequences, attention_mask, action_mask
            # sequences are (batch_size * num_return_sequences, seq_len)
            # We are providing batch_size = 1 for input_ids
            output_sequences, _, _ = model.generate(
                input_ids,
                attention_mask=attention_mask, # provided explicitly because of some issue inferring the attention mask
                **final_gen_kwargs
            )
            
            input_length = input_ids.shape[1]
            # Decode only the newly generated part
            generated_tokens_only = output_sequences[:, input_length:]
            generated_outputs_decoded = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)

        except Exception as e:
            # print(f"Error during model generation for query: {query[:100]}...\nError: {e}")
            # import traceback
            # traceback.print_exc()
            return [] # Return empty list on error

    return generated_outputs_decoded[:k_samples]


def get_behavior_check_function(behavior_definition: Behavior) -> Callable[[str], bool]:
    """Returns a function `check(output) -> bool` implementing the B(o)=1 check."""
    behavior_type = behavior_definition.type
    target = behavior_definition.target

    if behavior_type == 'specific_output':
        if not isinstance(target, str):
            raise ValueError("Target must be a string for 'specific_output' type.")
        # Normalize whitespace for comparison
        target_normalized = " ".join(target.strip().split())
        return lambda output: " ".join(output.strip().split()) == target_normalized

    elif behavior_type == 'keyword':
        if isinstance(target, str):
            targets = [target]
        elif isinstance(target, list) and all(isinstance(t, str) for t in target):
            targets = target
        else:
            raise ValueError("Target must be a string or list of strings for 'keyword' type.")
        
        # Case-insensitive check for any of the keywords
        targets_lower = [t.lower() for t in targets]
        return lambda output: any(t_lower in output.lower() for t_lower in targets_lower)

    elif behavior_type == 'custom_function':
        # Assumes 'target' is the actual callable function
        if not callable(target):
             raise ValueError("Target must be a callable function for 'custom_function' type.")
        return target

    # TODO: Add cases for classifier or other complex checks if needed.
    else:
        raise NotImplementedError(f"Behavior check type '{behavior_type}' not implemented.")


def estimate_p_elicit(model: Actor, tokenizer: PreTrainedTokenizer, query: str, behavior_definition: Behavior, method: str, k_samples: int = 100, generation_kwargs: Dict[str, Any] = None) -> float:
    """Estimates the elicitation probability (p_elicit) for a given query and behavior definition.

    Args:
        model: The language model (OpenRLHF Actor).
        tokenizer: The tokenizer.
        query: The input query string.
        behavior_definition: A Behavior object defining the harmful behavior B.
        method: The estimation method ('logprob_target_sequence', 'logprob_target_keyword_in_target_seq', 'repeated_sampling').
        k_samples: Number of samples for 'repeated_sampling'.
        generation_kwargs: Additional arguments for model.generate, used in 'repeated_sampling'.

    Returns:
        Estimated elicitation probability (float between 0 and 1).
    """
    if generation_kwargs is None:
        generation_kwargs = {}

    p_elicit_val = 0.0

    if method == 'logprob_target_sequence':
        if not behavior_definition.target_sequence:
            raise ValueError("Method 'logprob_target_sequence' requires behavior_definition.target_sequence to be set.")
        
        target_output = behavior_definition.target_sequence
        # Ensure target_output is a string, though type hint on Behavior.target_sequence should cover this.
        if not isinstance(target_output, str):
            raise ValueError("behavior_definition.target_sequence must be a string for 'logprob_target_sequence' method.")
        
        log_prob = calculate_sequence_logprob(model, tokenizer, query, target_output)
        
        if log_prob > 0: # Log probs should be <= 0
            log_prob = 0.0 
        
        log_prob_clamp = max(log_prob, -700.0)
        p_elicit_val = math.exp(log_prob_clamp)

    elif method == 'logprob_target_keyword_in_target_seq':
        if not behavior_definition.target_sequence:
            raise ValueError("Method 'logprob_target_keyword_in_target_seq' requires behavior_definition.target_sequence to be set as the reference output.")
        if not behavior_definition.target_keywords:
            raise ValueError("Method 'logprob_target_keyword_in_target_seq' requires behavior_definition.target_keywords to be a non-empty list.")

        reference_output_text = behavior_definition.target_sequence
        if not isinstance(reference_output_text, str) or not reference_output_text:
            # This should be caught by the first check, but kept for robustness
            raise ValueError("'target_sequence' must be a non-empty string to serve as reference output.")

        keywords_to_find = behavior_definition.target_keywords
        if not isinstance(keywords_to_find, list) or not all(isinstance(k, str) for k in keywords_to_find) or not keywords_to_find:
            raise ValueError("'target_keywords' must be a non-empty list of strings.")

        model.eval()

        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        reference_output_tokens_ids = tokenizer.encode(reference_output_text, add_special_tokens=False)

        if not reference_output_tokens_ids:
            # print(f"Warning: Reference output (from target_sequence) '{reference_output_text[:100]}...' tokenized to empty. Returning 0.0")
            return 0.0

        input_ids_list = []
        if tokenizer.bos_token_id is not None:
            input_ids_list.append(tokenizer.bos_token_id)
        input_ids_list.extend(query_tokens)
        input_ids_list.extend(reference_output_tokens_ids)
        
        input_ids_tensor = torch.tensor([input_ids_list], dtype=torch.long)
        num_actions = len(reference_output_tokens_ids) # Logprobs for each token in the reference_output_text

        try:
            device = model.model.device if hasattr(model, 'model') and hasattr(model.model, 'device') else next(model.parameters()).device
        except StopIteration: # Model has no parameters (e.g. on meta device)
            # print("Warning: Could not determine model device from parameters, attempting to use cuda if available, else cpu.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids_tensor = input_ids_tensor.to(device)

        total_probability_sum = 0.0

        with torch.no_grad():
            try:
                # Actor.forward returns log P(action_i | context_i) for each action in num_actions
                # Here, actions are the tokens of reference_output_text
                all_action_log_probs = model.forward(input_ids_tensor, num_actions=num_actions, attention_mask=None)

                if all_action_log_probs is None or all_action_log_probs.ndim == 0 or all_action_log_probs.shape[-1] != num_actions:
                    # print(f"Warning: Invalid action_log_probs received from model.forward. Shape: {all_action_log_probs.shape if all_action_log_probs is not None else 'None'}. Expected num_actions: {num_actions}")
                    return 0.0 # Cannot proceed if log_probs are not as expected
                
                for keyword_str in keywords_to_find:
                    if not keyword_str: continue # Should be caught by earlier validation

                    keyword_tokens_ids = tokenizer.encode(keyword_str, add_special_tokens=False)
                    if not keyword_tokens_ids:
                        # print(f"Warning: Keyword '{keyword_str}' tokenized to empty sequence. Skipping.")
                        continue
                    
                    len_keyword_tokens = len(keyword_tokens_ids)

                    # Find occurrences of keyword_tokens_ids within reference_output_tokens_ids
                    for i in range(len(reference_output_tokens_ids) - len_keyword_tokens + 1):
                        reference_slice = reference_output_tokens_ids[i : i + len_keyword_tokens]
                        if reference_slice == keyword_tokens_ids:
                            # This is an occurrence. Sum the log_probs of its constituent tokens.
                            log_prob_of_this_keyword_occurrence = 0.0
                            for k_idx in range(len_keyword_tokens):
                                # all_action_log_probs[0, j] is log P(reference_output_tokens_ids[j] | query, reference_output_tokens_ids[<j])
                                log_prob_of_this_keyword_occurrence += all_action_log_probs[0, i + k_idx].item()
                            
                            # Clamp and convert to probability
                            prob_of_this_occurrence = math.exp(max(log_prob_of_this_keyword_occurrence, -700.0))
                            total_probability_sum += prob_of_this_occurrence
            except Exception as e:
                # print(f"Error during 'logprob_target_keyword_in_target_seq' (model.forward or token processing) for query: '{query[:100]}...' Sequence: '{reference_output_text[:100]}...' Keywords: {keywords_to_find}\nError: {e}")
                # import traceback
                # traceback.print_exc()
                return 0.0 # Return 0 probability on error during calculation
        p_elicit_val = total_probability_sum
        
    elif method == 'repeated_sampling':
        success_count = 0
        check_function = get_behavior_check_function(behavior_definition)
        outputs = generate_output(model, tokenizer, query, k_samples, **generation_kwargs)
        for output_text in outputs:
            if check_function(output_text):
                success_count += 1
        if k_samples == 0: p_elicit_val = 0.0
        else: p_elicit_val = success_count / k_samples
    else:
        raise ValueError(f"Unknown elicitation method: {method}")

    return max(0.0, min(p_elicit_val, 1.0)) # Ensure result is strictly in [0, 1]

# Placeholder: Implement these helper functions based on your model framework
# These will need access to the specific model and tokenizer objects used in OpenRLHF
def calculate_sequence_logprob(model: Actor, tokenizer: PreTrainedTokenizer, query: str, target_output: str) -> float:
    """Calculates the log probability log P(target_output | query) using the Actor model's forward pass.
    """
    model.eval() # Ensure model is in evaluation mode
    # TODO: Check the correctness of this implementation. We should be able to do this in a single forward pass of the model.
    
    # 1. Tokenize query and target_output.
    # The Actor.forward method expects the full sequence that includes the "actions" (target tokens).
    # Typically, for P(Y|X), the input to the model should represent X and then Y.
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    target_tokens = tokenizer.encode(target_output, add_special_tokens=False)

    if not target_tokens:
        # print(f"Warning: Target output '{target_output[:100]}...' tokenized to an empty sequence. Returning -inf logprob.")
        return -float('inf')

    # 2. Construct the full input_ids sequence for the model.
    # Based on Actor.forward typical usage for log P(Y|X) calculation:
    # Sequence: [BOS] + query_tokens + target_tokens
    # The model then calculates log P(target_token_i | BOS, query_tokens, target_tokens_<i)
    # The EOS token is usually handled by the generation process or sequence length,
    # for explicit logprob calculation, we care about the probability of the target_tokens themselves.
    
    input_ids_list = []
    if tokenizer.bos_token_id is not None:
        input_ids_list.append(tokenizer.bos_token_id)
    
    input_ids_list.extend(query_tokens)
    input_ids_list.extend(target_tokens) # These are the "actions"

    # If the model implicitly adds EOS or expects it for full sequence processing,
    # and if num_actions correctly slices *before* any such auto-added EOS, this is fine.
    # Let's assume for now that num_actions targets exactly the target_tokens as appended.

    input_ids_tensor = torch.tensor([input_ids_list], dtype=torch.long)
    
    num_actions = len(target_tokens)

    try:
        device = model.model.device if hasattr(model, 'model') and hasattr(model.model, 'device') else next(model.parameters()).device
    except StopIteration: 
        # print("Warning: Could not determine model device, attempting to use cuda if available, else cpu.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # If the model is on meta, it needs to be moved.
        # The experiment_runner should handle placing the model on a device.
        # model.to(device) # This might be problematic if device_map was used.

    input_ids_tensor = input_ids_tensor.to(device)

    # 3. Call model.forward() with the full sequence and num_actions.
    # Actor.forward is expected to return log_probs of shape (batch_size, num_actions)
    # where batch_size is 1 here.
    with torch.no_grad():
        try:
            # The Actor's forward pass with num_actions should give us the logprobs for the target sequence.
            action_log_probs = model.forward(input_ids_tensor, num_actions=num_actions, attention_mask=None) # Let Actor handle attention_mask

            if action_log_probs is None or action_log_probs.ndim == 0 : # Check if it's a scalar or None
                # print(f"Warning: action_log_probs from model.forward is None or scalar. Value: {action_log_probs}")
                return -float('inf')
            if action_log_probs.shape[-1] != num_actions:
                # print(f"Warning: action_log_probs from model.forward has unexpected shape. Shape: {action_log_probs.shape}. Expected num_actions: {num_actions}")
                return -float('inf')

            # 4. Sum the log probabilities for the target sequence.
            sum_log_probs = action_log_probs.sum().item()
            
            return sum_log_probs

        except Exception as e:
            # print(f"Error during logprob calculation (Actor.forward) for query: '{query[:100]}...' Target: '{target_output[:100]}...'\nError: {e}")
            # import traceback
            # traceback.print_exc()
            return -float('inf') 