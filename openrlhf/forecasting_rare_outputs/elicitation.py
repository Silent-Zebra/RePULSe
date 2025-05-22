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
    k_samples: int = 100, # Number of samples to generate for repeated sampling.
    **generation_kwargs
) -> float:
    """Estimates the probability of the model generating the keyword in response to the query.

    """
    print(f"Estimate keyword probability using repeated sampling (k={k_samples}). This can be slow and approximate.")

    outputs = generate_output(model, tokenizer, query, k_samples, **generation_kwargs)

    if not outputs:
        return 0.0

    output_series = pd.Series(outputs)
    keyword_lower = keyword.lower()
    success_count = output_series.str.lower().str.contains(keyword_lower, regex=False).sum()

    if k_samples == 0:
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
    attention_mask = input_tokens["attention_mask"].to(device)

    # Base generation args (can be overridden by **generation_kwargs)
    default_gen_params = {
        "max_new_tokens": 50,
        "temperature": 1.0,
        "top_p": 0.9,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    # These parameters are common to all batches, after merging with user's kwargs.
    base_batch_gen_kwargs = {**default_gen_params, **generation_kwargs}
    
    if base_batch_gen_kwargs.get("pad_token_id") is None:
        base_batch_gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    all_generated_outputs_decoded = []
    samples_generated_so_far = 0
    
    # Tune this batch size based on your GPU memory.
    # Starting with a smaller value like 500 or 1000 might be safer, then increase if possible.
    generation_batch_size = k_samples

    with torch.no_grad():
        while samples_generated_so_far < k_samples:
            num_to_generate_this_batch = min(generation_batch_size, k_samples - samples_generated_so_far)

            if num_to_generate_this_batch <= 0:
                break

            current_batch_gen_kwargs = base_batch_gen_kwargs.copy()
            current_batch_gen_kwargs["num_return_sequences"] = num_to_generate_this_batch
            
            try:
                output_sequences, _, _ = model.generate(
                    input_ids,
                    attention_mask=attention_mask, # provided explicitly
                    **current_batch_gen_kwargs
                )
                input_length = input_ids.shape[1]
                generated_tokens_only = output_sequences[:, input_length:]
                decoded_batch_outputs = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)

                all_generated_outputs_decoded.extend(decoded_batch_outputs)
                
                actual_generated_in_batch = len(decoded_batch_outputs)
                samples_generated_so_far += actual_generated_in_batch

                if actual_generated_in_batch < num_to_generate_this_batch:
                    print(f"Warning: Model generated {actual_generated_in_batch} sequences in current batch, but {num_to_generate_this_batch} were requested. Stopping generation for this query.")
                    break

            except Exception as e:
                print(f"Error during model generation batch for query: {query[:100]}... (requested {num_to_generate_this_batch} for this batch)\nError: {e}")
                return []

    return all_generated_outputs_decoded[:k_samples]


def get_behavior_check_function(behavior_definition: Behavior) -> Callable[[str], bool]:
    """Returns a function `check(output) -> bool` implementing the B(o)=1 check."""
    behavior_type = behavior_definition.type

    if behavior_type == 'specific_output':
        target_output_seq = behavior_definition.target_sequence
        if not isinstance(target_output_seq, str):
            raise ValueError("Target sequence must be a string for 'specific_output' type if target_sequence is used.")
        target_normalized = " ".join(target_output_seq.strip().split())
        return lambda output: " ".join(output.strip().split()) == target_normalized

    elif behavior_type == 'keyword':
        targets = behavior_definition.target_keywords
        if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets) or not targets:
            raise ValueError("Target_keywords must be a non-empty list of strings for 'keyword' type.")
        
        targets_lower = [t.lower() for t in targets]
        return lambda output: any(t_lower in output.lower() for t_lower in targets_lower)

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
        target_sequences_list = behavior_definition.target_sequence

        # Expect target_sequence to be a list of strings for this method.
        if not isinstance(target_sequences_list, list) or not target_sequences_list:
            print(f"Warning: For 'logprob_target_seq', behavior_definition.target_sequence is not a non-empty list for behavior '{behavior_definition.id}'. Value: {target_sequences_list}. Returning 0.0.")
            return 0.0

        current_total_prob = 0.0
        for target_output in target_sequences_list:
            if not isinstance(target_output, str) or not target_output:
                print(f"Warning: Invalid item in behavior_definition.target_sequence (expected non-empty string): '{target_output}' for behavior '{behavior_definition.id}'. Skipping.")
                continue

            log_prob = calculate_sequence_logprob(model, tokenizer, query, target_output)
            prob = math.exp(log_prob)
            current_total_prob += prob
        
        p_elicit_val = current_total_prob / len(target_sequences_list)

    elif method == 'logprob_target_keyword_in_target_seq':
        target_sequences_list = behavior_definition.target_sequence # Expect a list of strings

        if not isinstance(target_sequences_list, list) or not target_sequences_list:
            print(f"Warning: For 'logprob_target_keyword_in_target_seq', behavior_definition.target_sequence is not a non-empty list for behavior '{behavior_definition.id}'. Value: {target_sequences_list}. Returning 0.0.")
            return 0.0

        if not behavior_definition.target_keywords:
            raise ValueError("Method 'logprob_target_keyword_in_target_seq' requires behavior_definition.target_keywords to be a non-empty list.")
        
        keywords_to_find = behavior_definition.target_keywords
        if not isinstance(keywords_to_find, list) or not all(isinstance(k, str) for k in keywords_to_find) or not keywords_to_find:
            raise ValueError("'target_keywords' must be a non-empty list of strings.")

        model.eval()

        # --- Get all unique token IDs for target keywords (once) ---
        target_keyword_token_ids = set()
        for keyword_str in keywords_to_find:
            if not keyword_str: continue
            tokens_alone = tokenizer.encode(keyword_str, add_special_tokens=False)
            if tokens_alone:
                for token_id in tokens_alone:
                    target_keyword_token_ids.add(token_id)
            tokens_with_space = tokenizer.encode(" " + keyword_str, add_special_tokens=False)
            if tokens_with_space:
                for token_id in tokens_with_space:
                    target_keyword_token_ids.add(token_id)
        
        if not target_keyword_token_ids:
            print(f"Warning: No valid token IDs found for any target_keywords: {keywords_to_find} for behavior '{behavior_definition.id}'. Returning 0.0.")
            return 0.0

        final_total_probability_sum_across_sequences = 0.0
        total_num_sequences = len(target_sequences_list)
        for seq_idx, reference_output_text in enumerate(target_sequences_list):
            if not isinstance(reference_output_text, str) or not reference_output_text:
                print(f"Warning: Invalid item #{seq_idx} in behavior_definition.target_sequence (expected non-empty string): '{reference_output_text}' for behavior '{behavior_definition.id}' in 'logprob_target_keyword_in_target_seq'. Skipping this item.")
                continue

            current_sequence_probability_sum = 0.0
            try:
                query_tokens = tokenizer.encode(query, add_special_tokens=False)
                reference_output_tokens_ids = tokenizer.encode(reference_output_text, add_special_tokens=False)

                if not reference_output_tokens_ids:
                    print(f"Warning: Reference output (item #{seq_idx} from target_sequence list) '{reference_output_text[:100]}...' tokenized to empty. Skipping this item for behavior '{behavior_definition.id}'.")
                    continue

                input_ids_list = []
                if tokenizer.bos_token_id is not None:
                    input_ids_list.append(tokenizer.bos_token_id)
                input_ids_list.extend(query_tokens)
                input_ids_list.extend(reference_output_tokens_ids)
                
                input_ids_tensor = torch.tensor([input_ids_list], dtype=torch.long)
                num_actions = len(reference_output_tokens_ids)

                try:
                    device = model.model.device if hasattr(model, 'model') and hasattr(model.model, 'device') else next(model.parameters()).device
                except StopIteration:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_ids_tensor = input_ids_tensor.to(device)
                attention_mask_tensor = torch.ones_like(input_ids_tensor, device=device) # Create explicit attention mask

                with torch.no_grad():
                    model_output = model.forward(
                        input_ids_tensor,
                        num_actions=None,
                        attention_mask=attention_mask_tensor,
                        return_output=True
                    )
                    
                    all_logits = model_output['logits'].to(torch.float32)
                    all_vocab_log_probs = torch.log_softmax(all_logits, dim=-1)
                    
                    start_logit_index = len(input_ids_list) - num_actions - 1 
                    end_logit_index = len(input_ids_list) - 1
                    
                    all_vocab_log_probs_for_target_sequence = all_vocab_log_probs[:, start_logit_index:end_logit_index, :]
                    
                    if all_vocab_log_probs_for_target_sequence is None or \
                       all_vocab_log_probs_for_target_sequence.ndim != 3 or \
                       all_vocab_log_probs_for_target_sequence.shape[0] != 1 or \
                       all_vocab_log_probs_for_target_sequence.shape[1] != num_actions:
                        print(f"Warning: Invalid all_vocab_log_probs_for_target_sequence received for seq_idx {seq_idx} ('{reference_output_text[:50]}...'). Shape: {all_vocab_log_probs_for_target_sequence.shape if all_vocab_log_probs_for_target_sequence is not None else 'None'}. Expected: (1, {num_actions}, vocab_size). Skipping this item.")
                        continue

                    vocab_size = all_vocab_log_probs_for_target_sequence.shape[2]

                    for j in range(num_actions):
                        log_probs_at_step_j = all_vocab_log_probs_for_target_sequence[0, j, :]
                        for token_id in target_keyword_token_ids:
                            if 0 <= token_id < vocab_size:
                                log_prob_keyword_token = log_probs_at_step_j[token_id].item()
                                prob_keyword_token = math.exp(log_prob_keyword_token)
                                current_sequence_probability_sum += prob_keyword_token
                
                final_total_probability_sum_across_sequences += current_sequence_probability_sum

            except Exception as e:
                print(f"Error during 'logprob_target_keyword_in_target_seq' for query: '{query[:100]}...' with reference_output_text (item #{seq_idx}): '{reference_output_text[:100]}...' for behavior '{behavior_definition.id}'. Error: {e}", flush=True)
        
        p_elicit_val = final_total_probability_sum_across_sequences / total_num_sequences

    elif method == 'repeated_sampling':
        success_count = 0
        total_samples_processed = 0
        check_function = get_behavior_check_function(behavior_definition)

        # k_samples is the total number of samples to acquire for this query.
        # elicitation_processing_batch_size is how many we ask generate_output for at a time.
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
                if outputs:
                    output_series = pd.Series(outputs)
                    
                    if behavior_definition.type == 'keyword':
                        targets_lower = [t.lower() for t in behavior_definition.target_keywords]
                        if not output_series.empty:
                            output_series_lower = output_series.str.lower()
                            combined_keyword_mask = pd.Series(False, index=output_series_lower.index, dtype=bool)
                            for t_lower in targets_lower:
                                combined_keyword_mask |= output_series_lower.str.contains(t_lower, regex=False, na=False)
                            batch_success_count = combined_keyword_mask.sum()
                    else:
                        if not output_series.empty:
                            batch_success_count = output_series.apply(check_function).sum()
                
                success_count += batch_success_count
                
                total_samples_processed += len(outputs)

            if total_samples_processed == 0:
                p_elicit_val = 0.0
            else:
                p_elicit_val = success_count / total_samples_processed 
                if total_samples_processed < k_samples:
                    print(f"Warning: Elicitation probability for query '{query[:100]}...' calculated over {total_samples_processed} samples instead of the requested {k_samples} due to generation issues.")
    else:
        raise ValueError(f"Unknown elicitation method: {method}")

    return max(0.0, min(p_elicit_val, 1.0))


def calculate_sequence_logprob(model: Actor, tokenizer: PreTrainedTokenizer, query: str, target_output: str) -> float:
    """Calculates the log probability log P(target_output | query) using the Actor model's forward pass.
    """
    model.eval()
    
    # 1. Tokenize query and target_output.
    # The Actor.forward method expects the full sequence that includes the "actions" (target tokens).
    # Typically, for P(Y|X), the input to the model should represent X and then Y.
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    target_tokens = tokenizer.encode(target_output, add_special_tokens=False)

    if not target_tokens:
        print(f"[DEBUG calculate_sequence_logprob] Target output '{target_output[:100]}...' tokenized to an empty sequence. Returning -inf logprob.")
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
        print("[DEBUG calculate_sequence_logprob] Warning: Could not determine model device from parameters, attempting to use cuda if available, else cpu.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids_tensor = input_ids_tensor.to(device)
    attention_mask_tensor = torch.ones_like(input_ids_tensor, device=device)

    # 3. Call model.forward() with the full sequence and num_actions.
    # Actor.forward is expected to return log_probs of shape (batch_size, num_actions)
    # where batch_size is 1 here.
    with torch.no_grad():
        try:
            # The Actor's forward pass with num_actions should give us the logprobs for the target sequence.
            action_log_probs = model.forward(input_ids_tensor, num_actions=num_actions, attention_mask=attention_mask_tensor) # Pass the created attention_mask
            
            if action_log_probs is None or action_log_probs.ndim == 0 :
                print(f"[DEBUG calculate_sequence_logprob] Condition met: action_log_probs is None or scalar. Value: {action_log_probs}. Returning -float('inf').")
                return -float('inf')
            if action_log_probs.shape[-1] != num_actions:
                print(f"[DEBUG calculate_sequence_logprob] Condition met: action_log_probs shape[-1] ({action_log_probs.shape[-1]}) != num_actions ({num_actions}). Shape: {action_log_probs.shape}. Returning -float('inf').")
                return -float('inf')

            # 4. Sum the log probabilities for the target sequence.
            sum_log_probs = action_log_probs.sum().item()
            
            return sum_log_probs

        except Exception as e:
            print(f"[DEBUG calculate_sequence_logprob] Error during logprob calculation (Actor.forward or summation) for query: '{query[:100]}...' Target: '{target_output[:100]}...'")
            import traceback
            traceback.print_exc()
            return -float('inf') 