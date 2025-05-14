import math
import torch
import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Callable, Dict, Any
from openrlhf.models import Actor, ActorCustom
from .behavior_defs import Behavior, TARGET_SEQUENCES

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
        target_keywords=[keyword], # Use target_keywords as it's a list
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

    # Base generation args (can be overridden by **generation_kwargs)
    # num_return_sequences will be set per batch.
    default_gen_params = {
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_p": 0.9,
        "do_sample": True,
        # "num_return_sequences": k_samples, # Removed: will be controlled per batch
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    # These parameters are common to all batches, after merging with user's kwargs.
    base_batch_gen_kwargs = {**default_gen_params, **generation_kwargs}
    
    # Ensure pad_token_id is set if generation needs it (e.g. for batching or beam search)
    if base_batch_gen_kwargs.get("pad_token_id") is None:
        base_batch_gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
        # print("generate_output: Setting pad_token_id to eos_token_id as it was None.")

    all_generated_outputs_decoded = []
    samples_generated_so_far = 0
    
    # Tune this batch size based on your GPU memory.
    # Starting with a smaller value like 1000 or 5000 might be safer, then increase if possible.
    # User suggested 10000 for 100k total -> 10 batches.
    generation_batch_size = k_samples

    with torch.no_grad(): # Single no_grad context for all generation batches
        while samples_generated_so_far < k_samples:
            num_to_generate_this_batch = min(generation_batch_size, k_samples - samples_generated_so_far)

            if num_to_generate_this_batch <= 0:
                break  # Should not be strictly necessary due to while condition but safe

            current_batch_gen_kwargs = base_batch_gen_kwargs.copy()
            current_batch_gen_kwargs["num_return_sequences"] = num_to_generate_this_batch
            
            # For debugging, you might want to uncomment this:
            # print(f"DEBUG: Generating batch of {num_to_generate_this_batch}. Args: {current_batch_gen_kwargs}")

            try:
                # Actor.generate returns: sequences, attention_mask, action_mask
                # sequences are (num_to_generate_this_batch, seq_len)
                # We are providing batch_size = 1 for input_ids
                output_sequences, _, _ = model.generate(
                    input_ids,
                    attention_mask=attention_mask, # provided explicitly
                    **current_batch_gen_kwargs
                )
                # print(f"output_sequences.shape for batch: {output_sequences.shape}")
                input_length = input_ids.shape[1]
                # Decode only the newly generated part
                generated_tokens_only = output_sequences[:, input_length:]
                decoded_batch_outputs = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)

                all_generated_outputs_decoded.extend(decoded_batch_outputs)
                
                actual_generated_in_batch = len(decoded_batch_outputs)
                samples_generated_so_far += actual_generated_in_batch

                # If the model returns fewer sequences than requested in a batch (without an error)
                if actual_generated_in_batch < num_to_generate_this_batch:
                    print(f"Warning: Model generated {actual_generated_in_batch} sequences in current batch, but {num_to_generate_this_batch} were requested. Stopping generation for this query.")
                    break # Exit the while loop for this query; will return what's collected so far.

            except Exception as e:
                print(f"Error during model generation batch for query: {query[:100]}... (requested {num_to_generate_this_batch} for this batch)\nError: {e}")
                # import traceback
                # traceback.print_exc()
                return [] # Return empty list on error for the entire query's generation attempt

    return all_generated_outputs_decoded[:k_samples] # Ensure we don't return more than k_samples


def get_behavior_check_function(behavior_definition: Behavior) -> Callable[[str], bool]:
    """Returns a function `check(output) -> bool` implementing the B(o)=1 check."""
    behavior_type = behavior_definition.type
    # target = behavior_definition.target # This attribute no longer exists on Behavior

    if behavior_type == 'specific_output':
        # This type is not currently defined or used in behavior_defs.py with the current Behavior structure.
        # If it were to be used, target_sequence would be the likely field.
        # For now, let's assume if this type is passed, target_sequence should be checked.
        target_output_seq = behavior_definition.target_sequence
        if not isinstance(target_output_seq, str):
            raise ValueError("Target sequence must be a string for 'specific_output' type if target_sequence is used.")
        # Normalize whitespace for comparison
        target_normalized = " ".join(target_output_seq.strip().split())
        return lambda output: " ".join(output.strip().split()) == target_normalized

    elif behavior_type == 'keyword':
        # For 'keyword' type, we now use target_keywords attribute from Behavior
        targets = behavior_definition.target_keywords
        if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets) or not targets:
            raise ValueError("Target_keywords must be a non-empty list of strings for 'keyword' type.")
        
        # Case-insensitive check for any of the keywords
        targets_lower = [t.lower() for t in targets]
        return lambda output: any(t_lower in output.lower() for t_lower in targets_lower)

    # TODO: Add cases for classifier or other complex checks if needed.
    else:
        raise NotImplementedError(f"Behavior check type '{behavior_type}' not implemented.")


def estimate_p_elicit(model: Actor, tokenizer: PreTrainedTokenizer, query: str, behavior_definition: Behavior, method: str, k_samples: int = 100, generation_kwargs: Dict[str, Any] = None, elicitation_processing_batch_size: int = 1000) -> float:
    """Estimates the elicitation probability (p_elicit) for a given query and behavior definition.

    Args:
        model: The language model (OpenRLHF Actor).
        tokenizer: The tokenizer.
        query: The input query string.
        behavior_definition: A Behavior object defining the harmful behavior B.
        method: The estimation method ('logprob_target_seq', 'logprob_target_keyword_in_target_seq', 'repeated_sampling').
        k_samples: Number of samples for 'repeated_sampling' (total samples to acquire).
        generation_kwargs: Additional arguments for model.generate, used in 'repeated_sampling'.
        elicitation_processing_batch_size: For 'repeated_sampling', the number of samples to generate and process in each internal batch. Default is 1000.

    Returns:
        Estimated elicitation probability (float between 0 and 1).
    """
    if generation_kwargs is None:
        generation_kwargs = {}

    p_elicit_val = 0.0

    if method == 'logprob_target_seq':
        if not behavior_definition.target_sequence:
            raise ValueError("Method 'logprob_target_seq' requires behavior_definition.target_sequence to be set.")
        # Behavior-specific target_sequence is no longer used for this method.
        # We iterate through the global TARGET_SEQUENCES.
        
        if not TARGET_SEQUENCES:
            # print("Warning: Global TARGET_SEQUENCES is empty. Returning 0.0 for 'logprob_target_seq'.")
            return 0.0

        current_total_prob = 0.0
        for target_output in TARGET_SEQUENCES:
            if not isinstance(target_output, str) or not target_output:
                # print(f"Warning: Invalid entry in TARGET_SEQUENCES: '{target_output}'. Skipping.")
                continue

            log_prob = calculate_sequence_logprob(model, tokenizer, query, target_output)
            
            # Clamp log_prob: should be <= 0. If positive, suggests an issue or perfect P=1.0.
            # Also clamp at a very low value to prevent underflow with math.exp.
            clamped_log_prob = max(min(log_prob, 0.0), -700.0) 
            
            prob = math.exp(clamped_log_prob)
            current_total_prob += prob
        
        p_elicit_val = current_total_prob
        # The final result will be clamped to [0, 1] at the end of the function.

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
                # --- MODIFIED CALL --- 
                # Get the full model output, which includes logits
                _, model_output = model.forward(
                    input_ids_tensor,
                    num_actions=None, # Requesting full output, num_actions slicing happens later
                    attention_mask=None, # Let Actor handle attention_mask
                    return_output=True # Request the full output object
                    # return_type="all_vocab" # Removed: Not supported
                )
                
                # Extract logits [batch_size=1, seq_len, vocab_size]
                all_logits = model_output['logits'].to(torch.float32)
                
                # Calculate log probabilities across the full vocabulary
                # Apply log_softmax to the vocabulary dimension
                all_vocab_log_probs = torch.log_softmax(all_logits, dim=-1)
                
                # The logits correspond to predicting the *next* token. 
                # So, all_logits[:, i, :] contains the distribution for predicting token i+1 given tokens up to i.
                # We want the logprobs for predicting the tokens in reference_output_tokens_ids.
                # The input was [BOS] + query_tokens + reference_output_tokens_ids
                # The logits for predicting the first reference token are at index len(query_tokens) (if BOS exists) or len(query_tokens)-1 (if no BOS).
                # Let's determine the start index precisely.
                query_start_index = 1 if tokenizer.bos_token_id is not None else 0
                num_query_tokens = len(query_tokens)
                # Logits relevant for predicting the reference sequence start at index: query_start_index + num_query_tokens -1
                # The sequence length dimension of all_vocab_log_probs is len(input_ids_list)
                # We need the log_probs for the last num_actions tokens predicted. 
                # These correspond to the *output* logits from position query_len-1 up to seq_len-2
                start_logit_index = len(input_ids_list) - num_actions -1 
                end_logit_index = len(input_ids_list) -1 # Exclusive, so up to seq_len-2
                
                # Slice the log_probs tensor to get the distributions corresponding to the target sequence positions
                # Shape should be [1, num_actions, vocab_size]
                all_vocab_log_probs_for_target_sequence = all_vocab_log_probs[:, start_logit_index:end_logit_index, :]
                # --- END MODIFIED CALL --- 
                
                print(f"Returned all_vocab_log_probs_for_target_sequence.shape: {all_vocab_log_probs_for_target_sequence.shape}", flush=True)

                # Shape of all_vocab_log_probs_for_target_sequence should be (batch_size=1, num_actions, vocab_size)
                if all_vocab_log_probs_for_target_sequence is None or \
                   all_vocab_log_probs_for_target_sequence.ndim != 3 or \
                   all_vocab_log_probs_for_target_sequence.shape[0] != 1 or \
                   all_vocab_log_probs_for_target_sequence.shape[1] != num_actions:
                    print(f"Warning: Invalid all_vocab_log_probs_for_target_sequence received. Shape: {all_vocab_log_probs_for_target_sequence.shape if all_vocab_log_probs_for_target_sequence is not None else 'None'}. Expected: (1, {num_actions}, vocab_size)")
                    return 0.0 # Cannot proceed if log_probs are not as expected

                # --- ADDED PART: Get all unique token IDs for target keywords --- 
                target_keyword_token_ids = set()
                for keyword_str in keywords_to_find:
                    if not keyword_str: continue
                    # Tokenize keyword alone
                    tokens_alone = tokenizer.encode(keyword_str, add_special_tokens=False)
                    if tokens_alone:
                        for token_id in tokens_alone:
                            target_keyword_token_ids.add(token_id)
                    # Tokenize keyword with preceding space (common variation)
                    tokens_with_space = tokenizer.encode(" " + keyword_str, add_special_tokens=False)
                    # Sometimes tokenizing " word" results in the same token ID for "word" plus a space marker token,
                    # sometimes it's a completely different token ID. Add all possibilities.
                    if tokens_with_space:
                        for token_id in tokens_with_space:
                            target_keyword_token_ids.add(token_id)
                # --- END ADDED PART ---

                # --- DEBUG PRINT 1 ---
                print(f"DEBUG: Target Keyword Token IDs: {target_keyword_token_ids}", flush=True)
                # --- END DEBUG PRINT 1 ---

                # --- REVISED LOGIC: Sum probabilities of target keyword tokens at each step ---
                vocab_size = all_vocab_log_probs_for_target_sequence.shape[2]
                print(f"DEBUG: Vocab size: {vocab_size}", flush=True) # --- DEBUG PRINT Vocab Size ---
                print(f"DEBUG: num_actions (steps in target_sequence): {num_actions}", flush=True) # --- DEBUG PRINT num_actions ---

                for j in range(num_actions): # Iterate through each token position in the target sequence
                    log_probs_at_step_j = all_vocab_log_probs_for_target_sequence[0, j, :] # Shape: (vocab_size)
                    # --- DEBUG PRINT for first step (j=0) ---
                    if j == 0:
                        print(f"DEBUG: Processing step j={j} in target_sequence", flush=True)
                    # --- END DEBUG PRINT ---
                    for token_id in target_keyword_token_ids:
                        if 0 <= token_id < vocab_size:
                            log_prob_keyword_token = log_probs_at_step_j[token_id].item()
                            # --- DEBUG PRINT for first step (j=0) ---
                            if j == 0:
                                print(f"DEBUG:   Keyword Token ID: {token_id}, Log Prob: {log_prob_keyword_token}", flush=True)
                            # --- END DEBUG PRINT ---
                            prob_keyword_token = math.exp(max(log_prob_keyword_token, -700.0))
                            # --- DEBUG PRINT for first step (j=0) ---
                            if j == 0:
                                print(f"DEBUG:     Prob for Token ID {token_id}: {prob_keyword_token}", flush=True)
                            # --- END DEBUG PRINT ---
                            total_probability_sum += prob_keyword_token
                        # else: # Optional: Warn if a keyword token ID is out of bounds
                        #     print(f"Warning: Keyword token ID {token_id} is out of vocabulary range [0, {vocab_size-1}]. Skipping.")
                # --- END REVISED LOGIC ---

                # --- REMOVED OLD LOGIC --- 
                # for keyword_str in keywords_to_find:
                #     if not keyword_str: continue # Should be caught by earlier validation
                #
                #     keyword_tokens_ids = tokenizer.encode(keyword_str, add_special_tokens=False)
                #     if not keyword_tokens_ids:
                #         # print(f"Warning: Keyword '{keyword_str}' tokenized to empty sequence. Skipping.")
                #         continue
                #     
                #     len_keyword_tokens = len(keyword_tokens_ids)
                #
                #     # Find occurrences of keyword_tokens_ids within reference_output_tokens_ids
                #     for i in range(len(reference_output_tokens_ids) - len_keyword_tokens + 1):
                #         reference_slice = reference_output_tokens_ids[i : i + len_keyword_tokens]
                #         if reference_slice == keyword_tokens_ids:
                #             # This is an occurrence. Sum the log_probs of its constituent tokens.
                #             log_prob_of_this_keyword_occurrence = 0.0
                #             for k_idx in range(len_keyword_tokens):
                #                 # all_action_log_probs[0, j] is log P(reference_output_tokens_ids[j] | query, reference_output_tokens_ids[<j])
                #                 # THIS PART WILL BE REPLACED IN THE NEXT STEP
                #                 # log_prob_of_this_keyword_occurrence += all_vocab_log_probs_for_target_sequence[0, i + k_idx].item() # Placeholder, needs fix
                #                 # --- Corrected logic to use appropriate log probs --- 
                #                 # We should actually sum the logprobs for the specific keyword tokens *at their position* 
                #                 # This old logic sums the logprob of whatever token *was* in the target sequence at that point.
                #                 # The revised approach below handles this better.
                #                 pass # Logic removed, handled by new loops
                #             
                #             # Clamp and convert to probability
                #             # prob_of_this_occurrence = math.exp(max(log_prob_of_this_keyword_occurrence, -700.0))
                #             # total_probability_sum += prob_of_this_occurrence
                #             # --- This calculation is replaced by the new loop structure ---
                #             pass # Logic removed
                # --- END REMOVED OLD LOGIC --- 
            except Exception as e:
                # print(f"Error during 'logprob_target_keyword_in_target_seq' (model.forward or token processing) for query: '{query[:100]}...' Sequence: '{reference_output_text[:100]}...' Keywords: {keywords_to_find}\\nError: {e}")
                print(f"Error during 'logprob_target_keyword_in_target_seq' calculation: {e}", flush=True) # Added flush=True
                # import traceback
                # traceback.print_exc()
                return 0.0 # Return 0 probability on error during calculation
        p_elicit_val = total_probability_sum
        # --- DEBUG PRINT FINAL SUM ---
        print(f"DEBUG: Final total_probability_sum for query '{query[:50]}...': {p_elicit_val}", flush=True)
        # --- END DEBUG PRINT FINAL SUM ---

    elif method == 'repeated_sampling':
        success_count = 0
        total_samples_processed = 0
        check_function = get_behavior_check_function(behavior_definition)

        # k_samples is the total number of samples to acquire for this query.
        # elicitation_processing_batch_size is how many we ask generate_output for at a time.
        # This is now a parameter to the function, with a default.

        if k_samples == 0:
            p_elicit_val = 0.0
        else:
            while total_samples_processed < k_samples:
                num_to_generate_this_iteration = min(elicitation_processing_batch_size, k_samples - total_samples_processed)

                if num_to_generate_this_iteration <= 0:
                    break

                outputs = generate_output(model, tokenizer, query, num_to_generate_this_iteration, **generation_kwargs)
                
                if not outputs and num_to_generate_this_iteration > 0:
                    print(f"Warning: generate_output returned no samples for query '{query[:100]}...' when {num_to_generate_this_iteration} were requested. Stopping for this query.")
                    break

                batch_success_count = 0
                if outputs: # Process only if there are outputs
                    output_series = pd.Series(outputs)
                    
                    if behavior_definition.type == 'keyword':
                        # Vectorized check for keyword presence (any of the keywords)
                        targets_lower = [t.lower() for t in behavior_definition.target_keywords]
                        if not output_series.empty:
                            output_series_lower = output_series.str.lower()
                            # Initialize a mask for recording if any keyword is found in an output
                            combined_keyword_mask = pd.Series(False, index=output_series_lower.index, dtype=bool)
                            for t_lower in targets_lower:
                                combined_keyword_mask |= output_series_lower.str.contains(t_lower, regex=False, na=False)
                            batch_success_count = combined_keyword_mask.sum()
                    else:
                        # For 'specific_output' or other types, use the pre-defined check_function.
                        # Series.apply is a vectorized way to apply the function to each element.
                        if not output_series.empty:
                            batch_success_count = output_series.apply(check_function).sum()
                
                success_count += batch_success_count
                
                total_samples_processed += len(outputs)
                print(f"DEBUG: Processed {total_samples_processed}/{k_samples} samples for query '{query[:50]}...'")

            if total_samples_processed == 0:
                p_elicit_val = 0.0
            else:
                p_elicit_val = success_count / total_samples_processed 
                if total_samples_processed < k_samples:
                    print(f"Warning: Elicitation probability for query '{query[:100]}...' calculated over {total_samples_processed} samples instead of the requested {k_samples} due to generation issues.")
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