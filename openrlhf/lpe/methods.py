import torch as th
import torch.nn.functional as F # Add import for padding
from fancy_einsum import einsum
from tqdm import tqdm
from typing import TYPE_CHECKING, Union, List, Optional
import warnings
import inspect

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




def ITGIS(
    model: 'Actor', # Expect an Actor instance (or similar wrapper around HF model)
    orig_dists: list[Discrete],
    target: Union[int, List[int]], # Modified type hint
    *,
    temp: float,
    n_samples: int,
    batch_size: int = 256,
    decay_rate: float = 0.9,
    show_progress: bool = False,
    use_argmax: bool = False # Added flag
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
    - use_argmax: If True, estimate probability based on whether the target sequence is generated greedily (argmax). If False (default), use softmax probability.
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

        # --- Calculate P(target | input) using a standard Actor forward pass ---
        # NOTE: This simplification removes the 'use_argmax' functionality previously
        # present, which allowed estimating probability based on whether greedy generation matching the target.
        # The 'prob_or_match' variable will now always represent exp(log P(target | input)).
        with th.no_grad():
            # 1. Prepare combined input sequences (input + target)
            expanded_target_ids = target_ids_tensor.unsqueeze(0).expand(batch_size, -1) # (batch_size, num_target_tokens)
            combined_sequences = th.cat([samples, expanded_target_ids], dim=1) # (batch_size, ctx_len + num_target_tokens)

            # 2. Create an attention mask for the combined sequence.
            # This assumes that 'samples' and 'target_ids' do not contain internal padding
            # and represent contiguous blocks of tokens. If 'samples' had its own attention_mask,
            # it should be used for the first part of combined_attention_mask.
            combined_attention_mask = th.ones_like(combined_sequences, device=device, dtype=th.long)
            
            # 3. Call Actor's forward method to get log probabilities for the target tokens (Y)
            log_probs_y_given_x = model.forward(
                sequences=combined_sequences,
                num_actions=num_target_tokens,
                attention_mask=combined_attention_mask,
                return_type='p' # Ensures log P(token | context) is returned for the action sequence
            ) # Expected shape: (batch_size, num_target_tokens)

            # 4. Sum log probabilities along the sequence dimension to get total log P(target|input) for each item in the batch
            total_log_prob_y_given_x = log_probs_y_given_x.sum(dim=-1) # Shape: (batch_size,)

            # 5. Calculate P(target|input) = exp(log P(target|input))
            # As noted above, the 'use_argmax' conditional path and its associated logic
            # (all_steps_match_argmax) are removed in this simplified version.
            prob_or_match = th.exp(total_log_prob_y_given_x) # Shape: (batch_size,)

        # --- End P(target | input) Calculation ---

        # Calculate weighted probabilities for the current batch using the calculated probability or match result
        batch_weighted_probs = th.exp(logratios.sum(-1)) * prob_or_match.detach()

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
    show_progress: bool = False,
    use_argmax: bool = False # Added flag
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
    - use_argmax: If True, estimate probability based on whether the target sequence is generated greedily (argmax). If False (default), use softmax probability.
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

    # Access the underlying HF model, potentially wrapped by DeepSpeed
    if hasattr(model.model, 'module'):
        hf_model = model.model.module
    else:
        hf_model = model.model

    config = hf_model.config
    d_vocab = config.vocab_size

    device = hf_model.device
    embed_layer = hf_model.get_input_embeddings()
    W_E = embed_layer.weight

    lm_head = hf_model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(hf_model, 'lm_head', None)
    if lm_head is None:
        raise AttributeError("Could not find output embedding/LM head layer.")

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

    for param in hf_model.parameters():
        param.requires_grad_(False)

    # --- Helper function for forward pass and gradient calculation --- #
    def get_scores_grads_and_prob(samples: th.Tensor):
        # Stores results
        scores_out = None
        grads_out = None
        prob_y_given_x_out = None
        argmax_matches_out = None # Added output for argmax match
        _batch_size, _seq_len = samples.shape

        # --- Calculate Score s(x) and Gradient d(s)/d(onehot) --- #
        with th.enable_grad():
            initial_attention_mask_grad = th.ones_like(samples, device=device)
            embeddings = embed_layer(samples)
            embeddings.requires_grad_(True)

            forward_kwargs_grad = {
                "inputs_embeds": embeddings,
                "attention_mask": initial_attention_mask_grad,
                "use_cache": False, # No need for KV cache from this specific pass
                "return_dict": True,
                "output_hidden_states": False,
                "output_attentions": False
            }
            try:
                if hasattr(hf_model, 'model') and callable(hf_model.model):
                    transformer_outputs = hf_model.model(**forward_kwargs_grad)
                elif hasattr(hf_model, 'transformer') and callable(hf_model.transformer):
                    transformer_outputs = hf_model.transformer(**forward_kwargs_grad)
                else:
                    # Basic fallback, might ignore mask
                    sig = inspect.signature(hf_model.forward)
                    if "attention_mask" not in sig.parameters:
                         forward_kwargs_grad.pop("attention_mask", None)
                    transformer_outputs = hf_model(**forward_kwargs_grad)
            except Exception as e:
                print(f"Error during gradient forward pass: {e}")
                raise NotImplementedError("Cannot determine main transformer body or handle forward pass correctly for grad.")

            hidden_states = transformer_outputs.last_hidden_state

            # 3. Apply final norm and LM head manually to last hidden state
            last_hidden_state = hidden_states[:, -1]
            normed_hidden_state = final_norm(last_hidden_state)
            logits = lm_head(normed_hidden_state)

            # 4. Get target logit score s(x) (using *first* target) and backpropagate
            target_logit_score = logits[:, first_target_id]
            target_logit_score.sum().backward()

            embedding_grad = embeddings.grad
            if embedding_grad is None:
                grads_out = th.zeros((_batch_size, ctx_len, d_vocab), device=device)
                warnings.warn("Embedding gradient is None, returning zero gradient.")
            else:
                onehot_grad = einsum("b s d, v d -> b s v", embedding_grad, W_E)
                grads_out = onehot_grad.detach().clone()

            scores_out = target_logit_score.detach().clone()
            embeddings.grad = None # Clear grad

        # --- Calculate P(target seq | input seq) using a standard Actor forward pass (No Grad) --- #
        with th.no_grad():
            # 1. Prepare combined input sequences (input seq + target seq)
            expanded_target_ids = target_ids_tensor.unsqueeze(0).expand(_batch_size, -1) # Shape: (_batch_size, num_target_tokens)
            combined_sequences = th.cat([samples, expanded_target_ids], dim=1) # Shape: (_batch_size, _seq_len + num_target_tokens)

            # 2. Create an attention mask for the combined sequence
            combined_attention_mask = th.ones_like(combined_sequences, device=device, dtype=th.long)
            
            # 3. Call Actor's forward method to get log probabilities for the target tokens
            log_probs_y_given_x = model.forward(
                sequences=combined_sequences,
                num_actions=num_target_tokens,
                attention_mask=combined_attention_mask,
                return_type='p' # Ensures log P(token | context) is returned
            ) # Expected shape: (_batch_size, num_target_tokens)

            # 4. Sum log probabilities to get total log P(target seq | input seq) for each item in the batch
            total_log_prob_y_given_x = log_probs_y_given_x.sum(dim=-1) # Shape: (_batch_size,)

            # 5. Calculate P(target seq | input seq) = exp(log P(target seq | input seq))
            # Both outputs related to P(target seq | input seq) are now set to this probability.
            prob_y_given_x_out = th.exp(total_log_prob_y_given_x) 
            argmax_matches_out = prob_y_given_x_out.clone()

        return scores_out, grads_out, prob_y_given_x_out, argmax_matches_out
    # --- End Helper Function --- #

    # Initialize the first batch of samples
    current_samples = th.stack([dist.sample((batch_size,)) for dist in orig_dists], dim=1).to(device)

    # Calculate initial scores, gradients, and probabilities P(Y|X) / Argmax Match
    current_scores, current_grads, current_probs, current_argmax_matches = get_scores_grads_and_prob(current_samples)

    acceptance_count = 0
    total_proposals = 0

    # Lists to store results after burn-in
    final_results_list = [] # Stores either P(Y|X) or argmax match result
    final_scores_list = [] # Store s(x)

    for step in tqdm(range((n_samples // batch_size + burn_in)), disable=not show_progress):
        # Choose a random position to modify for each sample in the batch
        pos = th.randint(0, ctx_len, (batch_size,), device=device)

        # Compute proposal probabilities q(x' | x) based on current gradients
        # proposal_logits = log p_orig(x'_i) + grad_i(x) . W_E[x'_i] / temp
        # Note: current_grads has shape (batch_size, ctx_len, d_vocab)
        # We need the gradient for the specific position `pos` for each sample
        grad_at_pos = current_grads[th.arange(batch_size), pos] # (batch_size, d_vocab)
        proposal_logits = orig_log_probs[pos] + grad_at_pos / temp
        proposal_probs = th.softmax(proposal_logits, dim=-1)

        # Make sure that the proposal probabilities are valid
        valid_proposal = proposal_probs.sum(dim=-1) > 0.5
        if not valid_proposal.all():
            print(f"Warning: Invalid proposal distribution detected at step {step}.")
            proposal_probs = th.clamp(proposal_probs, min=0)
            proposal_probs = proposal_probs / proposal_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Propose new tokens based on proposal distribution
        try:
            proposed_tokens = th.multinomial(proposal_probs, 1).squeeze(-1)
        except RuntimeError as e:
            print(f"Error during multinomial sampling at step {step}: {e}")
            print("Proposal probabilities (sample):", proposal_probs[0])
            raise e

        # Create proposed samples x'
        proposed_samples = current_samples.clone()
        proposed_samples[th.arange(batch_size), pos] = proposed_tokens

        # Recompute scores, gradients, and probabilities for proposed samples x'
        proposed_scores, proposed_grads, proposed_probs, proposed_argmax_matches = get_scores_grads_and_prob(proposed_samples)

        # Compute reverse proposal probabilities q(x | x')
        # reverse_proposal_logits = log p_orig(x_i) + grad_i(x') . W_E[x_i] / temp
        grad_at_pos_proposed = proposed_grads[th.arange(batch_size), pos] # (batch_size, d_vocab)
        reverse_proposal_logits = orig_log_probs[pos] + grad_at_pos_proposed / temp
        reverse_proposal_probs = th.softmax(reverse_proposal_logits, dim=-1)

        # Extract needed components for acceptance probability calculation
        current_tokens_at_pos = current_samples[th.arange(batch_size), pos]
        # current_tokens_at_pos = th.clamp(current_tokens_at_pos, 0, d_vocab - 1)
        # proposed_tokens = th.clamp(proposed_tokens, 0, d_vocab - 1)

        # Gather reverse and forward probabilities
        p_rev = reverse_proposal_probs.gather(1, current_tokens_at_pos.unsqueeze(-1)).squeeze(-1)
        p_fwd = proposal_probs.gather(1, proposed_tokens.unsqueeze(-1)).squeeze(-1)
        # p_rev = th.clamp(p_rev, min=1e-20)
        # p_fwd = th.clamp(p_fwd, min=1e-20)

        # Compute log acceptance probabilities: log A(x' | x)
        log_accept_probs = (proposed_scores - current_scores) / temp + \
                           orig_log_probs[pos].gather(1, proposed_tokens.unsqueeze(1)).squeeze(1) - \
                           orig_log_probs[pos].gather(1, current_tokens_at_pos.unsqueeze(1)).squeeze(1) + \
                           th.log(p_rev) - th.log(p_fwd)

        # Accept or reject proposals
        accept_mask = th.log(th.rand(batch_size, device=device)) < log_accept_probs

        # Update current state for accepted proposals
        current_samples[accept_mask] = proposed_samples[accept_mask]
        current_scores[accept_mask] = proposed_scores[accept_mask]
        current_grads[accept_mask] = proposed_grads[accept_mask]
        # Update the correct result based on use_argmax flag
        if use_argmax:
            current_argmax_matches[accept_mask] = proposed_argmax_matches[accept_mask]
        else:
            current_probs[accept_mask] = proposed_probs[accept_mask]

        # Collect samples after burn-in period
        if step >= burn_in:
            if use_argmax:
                final_results_list.append(current_argmax_matches.clone())
            else:
                final_results_list.append(current_probs.clone()) # Store computed P(Y|X)
            final_scores_list.append(current_scores.clone())  # Store target logit score s(x)

        acceptance_count += accept_mask.sum().item()
        total_proposals += batch_size

    # --- End MCMC Loop --- #

    acceptance_rate = acceptance_count / total_proposals if total_proposals > 0 else 0
    print(f"MHIS Acceptance Rate: {acceptance_rate:.4f}")

    # --- Importance Sampling Calculation --- #
    # We have samples x ~ pi(x) \propto exp(s(x)/temp) * p_orig(x)
    # We want E_p_orig[P(Y|X)]
    # E_p_orig[P(Y|X)] = E_pi [ P(Y|X) * w(x) ], where w(x) = p_orig(x) / pi(x)
    # w(x) = 1 / ( C * exp(s(x)/temp) )
    # E_p_orig[P(Y|X)] approx = mean( P(Y|x_i) / exp(s(x_i)/temp) ) / mean( 1 / exp(s(x_i)/temp) )

    if not final_results_list:
        print("Warning: No samples collected after burn-in. Returning 0.")
        return 0.0

    results_tensor = th.cat(final_results_list) # Shape: (num_collected_samples,) - Contains probs or matches
    scores_tensor = th.cat(final_scores_list)   # Shape: (num_collected_samples,)

    # Weights w_i' = 1 / exp(s(x_i) / temp)
    inv_exp_scores = th.exp(-scores_tensor / temp)

    # Estimate 1/C = mean(w_i')
    mean_inv_exp_scores = inv_exp_scores.mean()

    if mean_inv_exp_scores == 0 or not th.isfinite(mean_inv_exp_scores):
        print(f"Warning: Estimated normalizing constant is zero or invalid ({mean_inv_exp_scores}). Cannot compute estimate.")
        return float('nan')

    # Calculate the numerator: mean( Result_i * w_i' ) - Result is P(Y|x_i) or ArgmaxMatch(x_i)
    unbiased_estimate_numerator = (results_tensor * inv_exp_scores).mean()

    if not th.isfinite(unbiased_estimate_numerator):
        print(f"Warning: Numerator for estimate is invalid ({unbiased_estimate_numerator}). Cannot compute estimate.")
        return float('nan')

    estimated_prob = unbiased_estimate_numerator / mean_inv_exp_scores

    if not th.isfinite(estimated_prob):
        print(f"Warning: Final estimated probability is invalid ({estimated_prob}). Returning NaN.")
        return float('nan')

    return estimated_prob.item()