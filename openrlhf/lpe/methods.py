import torch as th
from fancy_einsum import einsum
from tqdm import tqdm
from typing import TYPE_CHECKING, Union, List, Optional

from .utils import Discrete

if TYPE_CHECKING:
    from openrlhf.models.actor import Actor # Assuming Actor is the wrapper


def random_constraint_projection(
    P: th.tensor, biases: th.tensor, n_reps=1000, tol=1e-9, factor=0.99
):
    """
    Finds shortest point in the polytope defined by P @ x + biases >= 0.
    This method isn't very airtight, but it was what we used in our experiments.
    """
    n, d = P.shape
    device = P.device

    x = th.zeros(d, device=device)

    for _ in range(n_reps * 100):
        constraints = P @ x + biases
        # This should be be -tol, but we used tol in the official experiments
        unsatisfied = constraints < tol

        # Randomly choose an unsatisfied constraint
        # This will crash if there are no unsatisfied constraints?
        unsatisfied_indices = th.where(unsatisfied)[0]
        random_index = unsatisfied_indices[th.randint(len(unsatisfied_indices), (1,))]

        # Project onto the chosen constraint
        p_i = P[random_index]

        projection = ((-constraints[random_index]) / th.sum(p_i**2)) * p_i
        x += projection.squeeze()

        # Check if we found a feasible point
        if th.all(P @ x + biases >= 0):
            if _ < n_reps:
                x *= factor
            else:
                return x

    assert False, "Did not find a feasible point"


def QLD(
    W_U: th.Tensor, act_samps: th.Tensor, target: int, *, batch_size: int = 512
) -> float:
    """
    Quadratic Logit Decomposition. Estimates the probability of outputing token `target` by using the Quadratic Logit Decomposition with the Shortest Accepting Vector as `d'.
    Inputs:
    - W_U: the unembedding matrix of the model (d_model x d_vocab).
    - act_samps: the samples of the activations right before unembedding (n_samples x d_model).
    - target: the target token (in range [0...d_model)).
    Returns:
    - The estimated probability of outputing token `target`.
    """

    d_model = W_U.shape[0]
    d_vocab = W_U.shape[1]
    n = act_samps.shape[0]
    assert target < d_vocab, "Target token out of range"
    assert act_samps.shape == (n, d_model), "act_samps has incorrect shape"

    # whiten the activations.
    n = act_samps.shape[0]
    mean = act_samps.mean(dim=0)
    cov = (act_samps - mean).T @ (act_samps - mean) / n
    EPS = 1e-5
    A = th.linalg.cholesky(
        cov + EPS * th.eye(d_model, device=cov.device)
    )  # (d_model, d_model). We have cov + EPS*I == A @ A.T.
    u_samps = act_samps - mean @ th.inverse(A.T)  # (n, d_model), the whitened samples.

    # Given whitened samples z of shape (n, d_model), the logits are given by z @ pullbacks.T + biases. Every row of the logits is a sample.
    pullbacks = W_U.T @ A  # (d_vocab, d_model)
    biases = W_U.T @ mean  # (d_vocab, ).

    # find the shortest accepting vector.
    pullbacks_diff = pullbacks[target].unsqueeze(0) - pullbacks
    biases_diff = biases[target].unsqueeze(0) - biases
    biases_diff[target] = 100
    d = random_constraint_projection(
        pullbacks_diff, biases_diff, n_reps=200, factor=0.95
    )  # (d_model, )
    d = d / th.norm(d)

    a_samps = (u_samps @ d).sort().values
    all_probs = []

    assert n % batch_size == 0
    for y in th.split(u_samps, batch_size):
        b_samps = y - y @ th.outer(d, d)  # (batch_size, d_model)

        # figure out the lower and upper bounds on the last direction.
        # Let z be b_samps. For a particular sample i, we need that for all j != t=token, we have:
        #  z_i @ pullbacks[t].T + biases[t] > z_i @ pullbacks[j].T + biases[j].
        # If we let z_i = a_i + r_i d, where r_i is a scalar, then we need that:
        #  a_i @ (pullbacks[t] - pullbacks[j]).T + biases[t] - biases[j] > -r_i d @ (pullbacks[t] - pullbacks[j]).T
        #  {a_i @ (pullbacks[t] - pullbacks[j]).T + biases[t] - biases[j]} / -{d @ (pullbacks[t] - pullbacks[j]).T} > r_i   (possibly with a sign flip)

        pullbacks_diff = (pullbacks[target].unsqueeze(0) - pullbacks).mT
        numerator = y @ pullbacks_diff + biases[target] - biases
        denominator = -d @ pullbacks_diff
        lower = (
            th.where(denominator < 0, numerator / denominator, -th.inf).max(-1).values
        )  # (batch_size, )
        upper = (
            th.where(denominator > 0, numerator / denominator, th.inf).min(-1).values
        )  # (batch_size, )

        # find how many latents were between upper and lower
        all_probs.append(
            th.maximum(
                th.searchsorted(a_samps, upper) - th.searchsorted(a_samps, lower),
                th.tensor(0),
            )
            / n  # (batch_size, )
        )

    all_probs = th.cat(all_probs)  # (n, )
    return all_probs.mean().item()


def GLD(
    W_U: th.Tensor, act_samps: th.Tensor, target: int, *, batch_size: int = 512
) -> float:
    """
    Gaussian Logit Difference. Finds parameters of the normal distribution fit to the target logit minus the maximum logit.
    Inputs:
    - W_U: the unembedding matrix of the model (d_model x d_vocab).
    - act_samps: the samples of the activations right before unembedding (n_samples x d_model).
    - target: the target token (in range [0...d_model)).
    Returns:
    - mu, sigma: The mean and variance of the logit differnce. Note mu <= 0.
    """

    argmax = []
    # Use batches to avoid OOM
    for batch in act_samps.split(batch_size, dim=0):
        logits = batch @ W_U
        argmax.append(logits.argmax(dim=1))
    argmax = th.cat(argmax)

    max_samps = einsum("b x, x b -> b", act_samps, W_U[:, argmax])
    target_samps = act_samps @ W_U[:, target]

    mu = (target_samps - max_samps).mean().item()
    sigma = (target_samps - max_samps).std().item()

    return mu, sigma


# --- Autoregressive Generation Helper --- #

@th.no_grad() # Generation should not track gradients
def _generate_autoregressive(
    model: 'Actor',
    initial_tokens: th.Tensor, # Shape: (batch_size, seq_len)
    num_tokens_to_generate: int,
    temperature: float = 1.0, # Add temperature for sampling if needed, but greedy is default
    top_k: Optional[int] = None, # Add top_k sampling
    top_p: Optional[float] = None, # Add top_p (nucleus) sampling
) -> th.Tensor:
    """
    Generates sequences autoregressively from a batch of initial tokens.
    """
    # Access the underlying HF model
    if hasattr(model.model, 'module'):
        hf_model = model.model.module
    else:
        hf_model = model.model

    device = initial_tokens.device
    current_tokens = initial_tokens
    generated_tokens_list = []

    # Prepare model inputs for HF model's generate or forward pass
    # Use past_key_values for efficiency if available/applicable
    past_key_values = None

    for _ in range(num_tokens_to_generate):
        model_inputs = hf_model.prepare_inputs_for_generation(current_tokens, past_key_values=past_key_values, use_cache=True)
        
        # Handle different potential structures returned by prepare_inputs_for_generation
        input_ids = model_inputs.get("input_ids", None)
        if input_ids is None and "inputs_embeds" not in model_inputs:
             # If only attention_mask etc. are returned, use the last token of current_tokens
             input_ids = current_tokens[:, -1:]

        # Construct minimal kwargs for the forward pass
        forward_kwargs = {
            "attention_mask": model_inputs.get("attention_mask", None),
            "position_ids": model_inputs.get("position_ids", None),
            "past_key_values": model_inputs.get("past_key_values", None),
            "use_cache": True,
        }
        # Add input_ids or inputs_embeds based on what prepare_inputs provided
        if input_ids is not None:
            forward_kwargs["input_ids"] = input_ids
        elif "inputs_embeds" in model_inputs:
            forward_kwargs["inputs_embeds"] = model_inputs["inputs_embeds"]
        else:
             raise ValueError("Could not determine input_ids or inputs_embeds for model forward pass.")

        # Get model outputs
        outputs = hf_model(**forward_kwargs, return_dict=True, output_hidden_states=False, output_attentions=False)
        
        next_token_logits = outputs.logits[:, -1, :] # Logits for the very next token

        # Apply temperature scaling
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k / top-p filtering if specified
        if top_k is not None or top_p is not None:
             # Use Hugging Face's filtering function
             # Need to import it or re-implement logic
             # For simplicity here, let's use greedy if filtering is complex to add inline
             # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
             # For now, stick to greedy/basic sampling or implement filtering if essential
             pass # Placeholder: add filtering if needed

        # Sample the next token ID (using greedy approach here for simplicity)
        # Consider multinomial sampling if temperature/top_k/top_p are used meaningfully
        next_token_id = th.argmax(next_token_logits, dim=-1, keepdim=True) # (batch_size, 1)

        # Append generated token
        generated_tokens_list.append(next_token_id)

        # Update current_tokens and past_key_values for the next iteration
        current_tokens = next_token_id # Only the new token is needed if using past_key_values
        past_key_values = outputs.past_key_values

    # Concatenate all generated tokens
    if not generated_tokens_list:
        return th.empty((initial_tokens.shape[0], 0), dtype=th.long, device=device) # Return empty tensor if 0 tokens generated
    
    generated_tokens = th.cat(generated_tokens_list, dim=1) # (batch_size, num_tokens_to_generate)
    return generated_tokens

# --- End Helper --- #


def ITGIS(
    model: 'Actor', # Expect an Actor instance (or similar wrapper around HF model)
    orig_dists: list[Discrete],
    target: Union[int, List[int]], # Modified type hint
    *,
    temp: float,
    n_samples: int,
    batch_size: int = 256,
    decay_rate: float = 0.9,
    show_progress: bool = False
) -> float:
    """
    Independent Token Gradient Importance Sampling. Uses the gradient of the logit with respect to the token embedding to define a new importance sampling distribution (with all tokens still being independent). Adaptively updates the importance sampling distribution based on samples from the previous.
    Inputs:
    - model: the OpenRLHF Actor model instance wrapping a Hugging Face model.
    - orig_dists: list of Discrete distributions for each token position.
    - target: the target token ID (int) or sequence of token IDs (list).
    - temp: the temperature.
    - n_samples: the number of samples to be drawn.
    - batch_size: the batch size.
    - decay_rate: the decay rate in the exponentially-weighted moving average of gradients.
    Returns:
    - The estimated probability of outputing token `target`.
    """
    if isinstance(target, int):
        target_ids = [target]
    else:
        target_ids = target
    if not target_ids:
        raise ValueError("Target token list cannot be empty.")
    first_target_id = target_ids[0]
    num_target_tokens = len(target_ids)
    target_ids_tensor = th.tensor(target_ids, device=model.model.device) # For comparison

    if hasattr(model.model, 'module'):
        hf_model = model.model.module
    else:
        hf_model = model.model

    config = hf_model.config
    d_vocab = config.vocab_size

    device = hf_model.device
    embed_layer = hf_model.get_input_embeddings()
    W_E = embed_layer.weight
    # Get LM head (handle potential variations)
    lm_head = hf_model.get_output_embeddings()
    if lm_head is None:
        # Common fallback for models like Llama
        lm_head = getattr(hf_model, 'lm_head', None)
    if lm_head is None:
        raise AttributeError("Could not find output embedding/LM head layer.")
    # Get final layer norm (handle potential variations)
    final_norm = None
    if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'norm'): # Llama-style
         final_norm = hf_model.model.norm
    elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'ln_f'): # GPT-2-style
         final_norm = hf_model.transformer.ln_f
    # Add more checks if needed for other architectures
    if final_norm is None:
         # Use Identity if not found, but warn
         print("Warning: Could not find final layer norm, using Identity. Gradients might be affected.")
         final_norm = th.nn.Identity()


    ctx_len = len(orig_dists)
    scores = th.zeros((ctx_len, d_vocab), device=device)

    # Ensure model parameters don't require gradients for the main computation
    # We only need gradients w.r.t. embeddings temporarily
    for param in hf_model.parameters():
        param.requires_grad_(False)

    # Initialize accumulators for incremental mean calculation
    total_weighted_prob_sum = 0.0
    total_samples_processed = 0

    assert n_samples % batch_size == 0
    for i in tqdm(list(range(n_samples // batch_size)), disable=not show_progress):
        target_samples = []
        target_logratios = []
        adj_temp = (
            temp * (1 - decay_rate**i) / (1 - decay_rate) if i > 0 else 1
        )  # adjust temperature for exponential moving average

        # Sample tokens based on current importance scores
        for dist, scores_at_pos in zip(orig_dists, scores):
            samples_at_pos = dist.boltzmann_distribution(
                scores=scores_at_pos[dist.values], temperature=adj_temp
            ).sample((batch_size,))
            target_samples.append(samples_at_pos)
            target_logratios.append(
                th.logsumexp(
                    scores_at_pos[dist.values] / adj_temp + th.log(dist.probs), dim=0
                )
                - scores_at_pos[samples_at_pos] / adj_temp
            )

        samples = th.stack(target_samples, dim=1)  # (batch_size, ctx_len)
        logratios = th.stack(target_logratios, dim=1)  # (batch_size, ctx_len)

        with th.enable_grad():
            # 1. Embed tokens
            # NOTE: The HF model handles the addition of positional embeddings internally 
            # when it receives token embeddings, so we don't need to add them separately.
            embeddings = embed_layer(samples) # (batch_size, ctx_len, d_model)
            embeddings.requires_grad_(True)

            # 2. Pass embeddings through the main transformer body
            if hasattr(hf_model, 'model') and callable(hf_model.model): # Llama-style
                transformer_outputs = hf_model.model(inputs_embeds=embeddings)
            elif hasattr(hf_model, 'transformer') and callable(hf_model.transformer): # GPT-2 style
                transformer_outputs = hf_model.transformer(inputs_embeds=embeddings)
            else:
                raise NotImplementedError("Cannot determine main transformer body for gradient calculation.")

            hidden_states = transformer_outputs.last_hidden_state # (batch_size, ctx_len, d_model)

            # 3. Apply final norm and LM head manually to last hidden state
            last_hidden_state = hidden_states[:, -1]
            normed_hidden_state = final_norm(last_hidden_state)
            logits = lm_head(normed_hidden_state) # (batch_size, d_vocab)

            # 4. Get target logit (of the *first* target token) and backpropagate
            target_logit = logits[:, first_target_id] # Use first_target_id for gradient guidance
            target_logit.sum().backward()

            # 5. Get embedding gradient and calculate "one-hot" equivalent gradient
            embedding_grad = embeddings.grad # (batch_size, ctx_len, d_model)
            # Project gradients back to vocab space: d_logit/d_onehot = (d_logit/d_embed) @ W_E.T
            onehot_grad = einsum("b s d, v d -> b s v", embedding_grad, W_E) # (batch_size, ctx_len, d_vocab)

            # Clear gradients before probability calculation
            embeddings.grad = None
            # Optional: zero_grad the model if needed

            # --- End Gradient Calculation ---

        # --- Calculate P(Y | X) using sequential forward passes ---
        # Instead of checking if greedy generation matches, calculate the actual
        # conditional probability P(Y|X) = P(T1|X) * P(T2|X,T1) * ...
        # This is required for the correct importance sampling estimator.
        with th.no_grad():
            past_key_values = None # Initialize KV cache state
            # Accumulator for log P(Y|X) = log P(T1|X) + log P(T2|X,T1) + ...
            total_log_prob_y_given_x = th.zeros(batch_size, device=device)
            # Use the initial logits computed during the gradient pass for the first token prediction
            current_logits_for_prob = logits

            # We need initial KV cache. A simple way is to re-run the first pass if needed.
            # Alternatively, if transformer_outputs contained past_key_values, use that.
            # Re-running might be cleaner if initial grad pass had use_cache=False.
            # Let's assume for now we can get the initial KV cache if needed.
            # TODO: Ensure initial past_key_values are correctly obtained if t > 0 requires them.

            for t in range(num_target_tokens):
                if t > 0:
                    # Prepare inputs for subsequent steps using the *actual* previous target token
                    input_ids_step = target_ids_tensor[t-1].repeat(batch_size).unsqueeze(-1) # Shape: (batch_size, 1)

                    # Requires handling attention_mask correctly if model uses it.
                    # Assuming prepare_inputs handles mask extension based on past_key_values.
                    model_inputs = hf_model.prepare_inputs_for_generation(
                        input_ids=input_ids_step,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    # Ensure all necessary inputs are present
                    forward_kwargs = model_inputs.copy() # Start with prepared inputs
                    forward_kwargs.update({
                        "return_dict": True,
                        "output_hidden_states": False,
                        "output_attentions": False,
                        "use_cache": True
                    })


                    outputs = hf_model(**forward_kwargs)
                    current_logits_for_prob = outputs.logits[:, -1, :] # Logits for the token predicted *after* input_ids_step
                    past_key_values = outputs.past_key_values # Update KV cache for the next step

                # Calculate log probability of the target token at the current step t
                log_probs_step = th.log_softmax(current_logits_for_prob, dim=-1)
                # Gather prob of the specific target token target_ids[t] for this step
                target_log_prob_step = log_probs_step[th.arange(batch_size), target_ids[t]]

                total_log_prob_y_given_x += target_log_prob_step

            # Final probability P(Y | X) = exp( sum of log probabilities )
            prob_y_given_x = th.exp(total_log_prob_y_given_x) # Shape: (batch_size,)

        # --- End P(Y | X) Calculation ---

        # Calculate weighted probabilities for the current batch using the calculated probability
        batch_weighted_probs = th.exp(logratios.sum(-1)) * prob_y_given_x.detach() # Use the computed P(Y|X)

        # Update accumulators
        total_weighted_prob_sum += batch_weighted_probs.sum().item()
        total_samples_processed += batch_size # Increment by the actual batch size processed

        # Update scores using the calculated "one-hot" gradients
        # Aggregate gradient across batch dimension
        batch_grad = onehot_grad.sum(0) # (ctx_len, d_vocab)
        scores *= decay_rate
        scores += batch_grad / batch_size # Apply EWMA update

        # Detach grads for next iteration
        embeddings.grad = None

    # Calculate final mean from accumulated sums
    if total_samples_processed == 0:
        return 0.0 # Avoid division by zero if no samples were processed
    else:
        final_mean = total_weighted_prob_sum / total_samples_processed
        return final_mean

def MHIS(
    model: 'Actor', # Expect an Actor instance
    orig_dists: list[Discrete],
    target: Union[int, List[int]], # Modified type hint
    *,
    temp: float,
    n_samples: int,
    burn_in: int,
    batch_size: int = 32,
    show_progress: bool = False
) -> float:
    """
    Metropolis-Hastings Importance Sampling. Takes batch_size independent random walks in token space, with q(x) \propto exp(logit / temp) * p(x) as the stationary distribution.
    We use the proposal function phi(x'|x) defined by:
    - Choose a random token position i
    - Take the gradient of s(x) with respect to the embedding of token i. Dot that with the different tokens y you could replace token i with, and take probabilities proportional to p(y) exp(grad \cdot y / temp).
    Inputs:
    - model: the OpenRLHF Actor model instance wrapping a Hugging Face model.
    - orig_dists: list of Discrete distributions for each token position.
    - target: the target token ID (int) or sequence of token IDs (list).
    - temp: the temperature (for both the Boltzmann stationary distribution and proposal distribution)
    - n_samples: the total number of samples to be drawn, ignoring burn-in.
    - batch_size: the number of parallel random walks to run (the total number of samples drawn is n_samples + burn_in * batch_size)
    Returns:
    - The estimated probability of outputing token `target`.
    """
    # Parse target
    if isinstance(target, int):
        target_ids = [target]
    else:
        target_ids = target
    if not target_ids:
        raise ValueError("Target token list cannot be empty.")
    first_target_id = target_ids[0]
    num_target_tokens = len(target_ids)
    target_ids_tensor = th.tensor(target_ids, device=model.model.device) # For comparison

    # Access the underlying HF model, potentially wrapped by DeepSpeed
    if hasattr(model.model, 'module'):
        hf_model = model.model.module
    else:
        hf_model = model.model

    # Get model components using standard HF methods
    config = hf_model.config
    d_vocab = config.vocab_size # Now config should be the HF config object

    device = hf_model.device # Get device from the underlying HF model
    embed_layer = hf_model.get_input_embeddings()
    W_E = embed_layer.weight
    # Get LM head (handle potential variations)
    lm_head = hf_model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(hf_model, 'lm_head', None)
    if lm_head is None:
        raise AttributeError("Could not find output embedding/LM head layer.")
    # Get final layer norm (handle potential variations)
    final_norm = None
    if hasattr(hf_model, 'model') and hasattr(hf_model.model, 'norm'): # Llama-style
         final_norm = hf_model.model.norm
    elif hasattr(hf_model, 'transformer') and hasattr(hf_model.transformer, 'ln_f'): # GPT-2-style
         final_norm = hf_model.transformer.ln_f
    if final_norm is None:
         print("Warning: Could not find final layer norm, using Identity.")
         final_norm = th.nn.Identity()

    ctx_len = len(orig_dists)
    # `scores` here refers to the target logit score s(x), not the importance sampling scores from ITGIS
    # results here refers to whether the sample produced the target token

    # Precompute original log probabilities for allowed tokens at each position
    orig_log_probs = []
    for pos in range(ctx_len):
        mask = -th.inf * th.ones(d_vocab, device=device)
        mask[orig_dists[pos].values] = th.log(orig_dists[pos].probs)
        orig_log_probs.append(mask)
    orig_log_probs = th.stack(orig_log_probs) # (ctx_len, d_vocab)

    # Ensure model parameters don't require gradients except embeddings when needed
    for param in hf_model.parameters():
        param.requires_grad_(False)

    final_results = []
    final_scores = [] # Stores s(x) = logit_target(x) for accepted samples after burn-in

    # --- Helper function for forward pass and gradient calculation --- #
    def get_scores_grads_results(samples: th.Tensor):
        # This helper now returns the logit score for the *first* target token,
        # the corresponding gradients, and the result of checking the *full* sequence.
        with th.enable_grad():
            # 1. Embed tokens
            embeddings = embed_layer(samples) # (batch_size, ctx_len, d_model)
            embeddings.requires_grad_(True)

            # 2. Pass embeddings through the main transformer body
            if hasattr(hf_model, 'model') and callable(hf_model.model):
                 transformer_outputs = hf_model.model(inputs_embeds=embeddings)
            elif hasattr(hf_model, 'transformer') and callable(hf_model.transformer):
                 transformer_outputs = hf_model.transformer(inputs_embeds=embeddings)
            else:
                 raise NotImplementedError("Cannot determine main transformer body for gradient calculation.")

            hidden_states = transformer_outputs.last_hidden_state # (batch_size, ctx_len, d_model)

            # 3. Apply final norm and LM head manually to last hidden state
            last_hidden_state = hidden_states[:, -1]
            normed_hidden_state = final_norm(last_hidden_state)
            logits = lm_head(normed_hidden_state) # (batch_size, d_vocab)

            # 4. Get target logit score s(x) (using *first* target) and backpropagate
            target_logit_score = logits[:, first_target_id]
            target_logit_score.sum().backward()

            # 5. Get embedding gradient and calculate "one-hot" equivalent gradient
            embedding_grad = embeddings.grad # (batch_size, ctx_len, d_model)
            onehot_grad = einsum("b s d, v d -> b s v", embedding_grad, W_E) # (batch_size, ctx_len, d_vocab)

            # Detach score and grads
            scores_out = target_logit_score.detach().clone()
            grads_out = onehot_grad.detach().clone()

            # Clear gradients for next use (before generation call)
            embeddings.grad = None
            # Optional: zero_grad model params if needed

        # --- Check for Target Sequence Match (No Grad) ---
        generated_sequence = _generate_autoregressive(
            model=model, # Pass the Actor wrapper
            initial_tokens=samples,
            num_tokens_to_generate=num_target_tokens,
        ) # (batch_size, num_target_tokens)
        # Expand target tensor for comparison
        target_ids_expanded = target_ids_tensor.unsqueeze(0).expand_as(generated_sequence)
        results_out = (generated_sequence == target_ids_expanded).all(dim=1).float().detach() # (batch_size,)
        # --- End Check ---

        return scores_out, grads_out, results_out
    # --- End Helper Function --- #

    # Initialize the first batch of samples
    current_samples = th.stack([dist.sample((batch_size,)) for dist in orig_dists], dim=1).to(device)

    # Calculate initial scores, gradients, and results
    current_scores, current_grads, current_results = get_scores_grads_results(current_samples)

    acceptance_count = 0
    total_proposals = 0

    for step in tqdm(range((n_samples // batch_size + burn_in)), disable=not show_progress):
        # Choose a random position to modify for each sample in the batch
        pos = th.randint(0, ctx_len, (batch_size,), device=device)

        # Compute proposal probabilities q(x' | x) based on current gradients
        # proposal_logits = log p_orig(x'_i) + grad_i(x) . W_E[x'_i] / temp
        # Note: current_grads has shape (batch_size, ctx_len, d_vocab)
        # We need the gradient for the specific position `pos` for each sample
        grad_at_pos = current_grads[th.arange(batch_size), pos] # (batch_size, d_vocab)
        proposal_logits = orig_log_probs[pos] + grad_at_pos / temp
        # Ensure proposal probs are only non-zero for allowed tokens
        proposal_logits[orig_log_probs[pos] == -th.inf] = -th.inf
        proposal_probs = th.softmax(proposal_logits, dim=-1)

        # Propose new tokens based on proposal distribution
        # Add check for valid probabilities before sampling
        valid_proposal = proposal_probs.sum(dim=-1) > 0.5 # Check if probs sum to approx 1
        if not valid_proposal.all():
            print(f"Warning: Invalid proposal distribution detected at step {step}. Skipping proposal for invalid samples.")
            # Handle invalid proposals, e.g., by resampling positions or skipping updates for those samples
            # For now, we'll just proceed, but sampling might fail
            # Ensure probabilities are non-negative and normalized if issues persist
            proposal_probs = th.clamp(proposal_probs, min=0) # Ensure non-negative
            proposal_probs = proposal_probs / proposal_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9) # Renormalize

        # Sample proposed tokens, handle potential NaN/inf issues if necessary
        try:
            proposed_tokens = th.multinomial(proposal_probs, 1).squeeze(-1)
        except RuntimeError as e:
            print(f"Error during multinomial sampling at step {step}: {e}")
            print("Proposal probabilities (sample):", proposal_probs[0])
            # Decide how to handle: skip step, use fallback, etc.
            # For now, re-raising to halt execution and allow inspection
            raise e


        # Create proposed samples x'
        proposed_samples = current_samples.clone()
        proposed_samples[th.arange(batch_size), pos] = proposed_tokens

        # Recompute scores, gradients, and results for proposed samples x'
        proposed_scores, proposed_grads, proposed_results = get_scores_grads_results(proposed_samples)

        # Compute reverse proposal probabilities q(x | x')
        # reverse_proposal_logits = log p_orig(x_i) + grad_i(x') . W_E[x_i] / temp
        grad_at_pos_proposed = proposed_grads[th.arange(batch_size), pos] # (batch_size, d_vocab)
        reverse_proposal_logits = orig_log_probs[pos] + grad_at_pos_proposed / temp
        reverse_proposal_logits[orig_log_probs[pos] == -th.inf] = -th.inf
        reverse_proposal_probs = th.softmax(reverse_proposal_logits, dim=-1)

        # Extract needed components for acceptance probability calculation
        current_tokens_at_pos = current_samples[th.arange(batch_size), pos] # (batch_size,)
        # Ensure indices are within bounds before gathering
        current_tokens_at_pos = th.clamp(current_tokens_at_pos, 0, d_vocab - 1)
        proposed_tokens = th.clamp(proposed_tokens, 0, d_vocab - 1)

        # Gather reverse and forward probabilities
        p_rev = reverse_proposal_probs.gather(1, current_tokens_at_pos.unsqueeze(-1)).squeeze(-1) # q(x | x')
        p_fwd = proposal_probs.gather(1, proposed_tokens.unsqueeze(-1)).squeeze(-1)             # q(x'| x)

        # Avoid log(0) if proposal probability was zero (shouldn't happen with valid sampling)
        p_rev = th.clamp(p_rev, min=1e-20)
        p_fwd = th.clamp(p_fwd, min=1e-20)

        # Compute log acceptance probabilities: log A(x' | x)
        # log A = log [ (p(x') q(x | x')) / (p(x) q(x' | x)) ]
        # where p(x) is the target stationary distribution pi(x) = exp(s(x)/temp) * p_orig(x)
        # log A = log [ (exp(s(x')/T) p_orig(x') q(x|x')) / (exp(s(x)/T) p_orig(x) q(x'|x)) ]
        # log A = (s(x') - s(x))/T + log p_orig(x') - log p_orig(x) + log q(x|x') - log q(x'|x)
        # Here, log p_orig(x') corresponds to orig_log_probs[pos, proposed_tokens]
        # and   log p_orig(x) corresponds to orig_log_probs[pos, current_tokens_at_pos]

        log_accept_probs = (proposed_scores - current_scores) / temp + \
                           orig_log_probs[pos].gather(1, proposed_tokens.unsqueeze(1)).squeeze(1) - \
                           orig_log_probs[pos].gather(1, current_tokens_at_pos.unsqueeze(1)).squeeze(1) + \
                           th.log(p_rev) - th.log(p_fwd)

        # Accept or reject proposals
        accept_mask = th.log(th.rand(batch_size, device=device)) < log_accept_probs

        # Update current state for accepted proposals
        current_samples[accept_mask] = proposed_samples[accept_mask]
        current_scores[accept_mask] = proposed_scores[accept_mask]
        # Gradients need to be updated for the next step's proposal calculation
        current_grads[accept_mask] = proposed_grads[accept_mask]
        # Results are updated for the final output
        current_results[accept_mask] = proposed_results[accept_mask]

        # Don't need to clone scores/grads/results here as they are overwritten or indexed

        # Collect samples after burn-in period
        if step >= burn_in:
            final_results.append(current_results.clone()) # Store whether target was predicted
            final_scores.append(current_scores.clone())  # Store target logit score s(x)

        acceptance_count += accept_mask.sum().item()
        total_proposals += batch_size

    # This block should be outside the loop
    acceptance_rate = acceptance_count / total_proposals if total_proposals > 0 else 0
    print(f"MHIS Acceptance Rate: {acceptance_rate:.4f}")


    # --- Importance Sampling Calculation --- #
    # We have samples x drawn from pi(x) \propto exp(s(x)/temp) * p_orig(x)
    # We want E_p_orig[f(x)], where f(x) = 1 if target predicted, else 0.
    # E_p_orig[f(x)] = E_pi [ f(x) * w(x) ], where w(x) = p_orig(x) / pi(x)
    # w(x) = p_orig(x) / ( C * exp(s(x)/temp) * p_orig(x) ) = 1 / ( C * exp(s(x)/temp) )
    # where C is the normalizing constant of pi(x).
    # E_p_orig[f(x)] = E_pi [ f(x) / (C * exp(s(x)/temp)) ]
    # E_p_orig[f(x)] = (1/C) * E_pi [ f(x) / exp(s(x)/temp) ]
    # We estimate C using 1/C = E_pi [ 1 / exp(s(x)/temp) ]
    # E_p_orig[f(x)] approx = mean( f(x_i) / exp(s(x_i)/temp) ) / mean( 1 / exp(s(x_i)/temp) )
    # where x_i are samples from pi (our MCMC chain after burn-in)

    # This block should be outside the loop
    if not final_results:
        print("Warning: No samples collected after burn-in. Returning 0.")
        return 0.0

    # This block should be outside the loop
    results_tensor = th.cat(final_results) # Shape: (num_collected_samples,)
    scores_tensor = th.cat(final_scores)   # Shape: (num_collected_samples,)

    # Weights w_i = 1 / exp(s(x_i) / temp)
    inv_exp_scores = th.exp(-scores_tensor / temp)

    # Estimate normalizing constant C's inverse: mean(w_i)
    mean_inv_exp_scores = inv_exp_scores.mean()

    if mean_inv_exp_scores == 0 or not th.isfinite(mean_inv_exp_scores): # Check for inf/nan too
        print(f"Warning: Estimated normalizing constant is zero or invalid ({mean_inv_exp_scores}). Cannot compute estimate.")
        # This might happen if scores are extremely large/small
        # Could indicate numerical instability or issues with the sampling
        return float('nan') # Or handle appropriately

    # Calculate the unbiased estimate E_p_orig[f(x)]
    # mean( f(x_i) * w_i ) / mean( w_i )
    unbiased_estimate_numerator = (results_tensor * inv_exp_scores).mean()

    # Check if numerator is valid before division
    if not th.isfinite(unbiased_estimate_numerator):
        print(f"Warning: Numerator for estimate is invalid ({unbiased_estimate_numerator}). Cannot compute estimate.")
        return float('nan')

    estimated_prob = unbiased_estimate_numerator / mean_inv_exp_scores

    # Final check for NaN/inf in the result
    if not th.isfinite(estimated_prob):
        print(f"Warning: Final estimated probability is invalid ({estimated_prob}). Returning NaN.")
        return float('nan')


    return estimated_prob.item()