import torch as th
from fancy_einsum import einsum
from tqdm import tqdm
from typing import TYPE_CHECKING

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
    target: int,
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
    - target: the target token (in range [0...d_vocab)).
    - temp: the temperature.
    - n_samples: the number of samples to be drawn.
    - batch_size: the batch size.
    - decay_rate: the decay rate in the exponentially-weighted moving average of gradients.
    Returns:
    - The estimated probability of outputing token `target`.
    """
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

    imp_samp_probs = []
    assert n_samples % batch_size == 0
    for i in tqdm(list(range(n_samples // batch_size)), disable=not show_progress):
        target_samples = []
        target_logratios = []
        adj_temp = (
            temp * (1 - decay_rate**i) / (1 - decay_rate) if i > 0 else 1
        )  # adjust temperature for exponential moving average

        # Sample tokens based on current importance scores
        for dist, scores_at_pos in zip(orig_dists, scores):
            # Ensure scores_at_pos[dist.values] is 1D
            current_scores = scores_at_pos[dist.values].squeeze()
            if current_scores.dim() == 0: # Handle scalar case if dist.values has one element
                current_scores = current_scores.unsqueeze(0)

            # Compute Boltzmann distribution based on scores for allowed tokens
            boltzmann_dist = dist.boltzmann_distribution(
                scores=current_scores, temperature=adj_temp
            )
            samples_at_pos = boltzmann_dist.sample((batch_size,))
            target_samples.append(samples_at_pos)

            # Calculate log ratio: log(p_orig(x) / p_importance(x))
            # log p_importance(x) = scores[x]/temp - log Z(temp)
            # log Z(temp) = logsumexp(scores[allowed]/temp)
            # log p_orig(x) = log(dist.probs[x]) # Need to map sampled value back to index in dist.probs
            # Assuming boltzmann_dist.probs corresponds to dist.values
            log_p_importance = boltzmann_dist.log_prob(samples_at_pos)

            # Map sampled values back to their original probabilities in orig_dist
            value_to_prob_map = {val.item(): prob for val, prob in zip(dist.values, dist.probs)}
            log_p_orig = th.tensor([th.log(value_to_prob_map[samp.item()]) for samp in samples_at_pos], device=device)

            # log ratio = log_p_orig - log_p_importance
            target_logratios.append(log_p_orig - log_p_importance)


        samples = th.stack(target_samples, dim=1)  # (batch_size, ctx_len)
        logratios = th.stack(target_logratios, dim=1)  # (batch_size, ctx_len)

        # --- Calculate Gradients using HF Model ---
        with th.enable_grad():
            # 1. Embed tokens
            embeddings = embed_layer(samples) # (batch_size, ctx_len, d_model)
            embeddings.requires_grad_(True)

            # 2. Pass embeddings through the main transformer body
            # Use attention_mask if needed by the model (Llama doesn't strictly need it here)
            # Assuming the model's main body is accessible (e.g., hf_model.model for Llama)
            if hasattr(hf_model, 'model') and callable(hf_model.model):
                 transformer_outputs = hf_model.model(inputs_embeds=embeddings)
            elif hasattr(hf_model, 'transformer') and callable(hf_model.transformer): # GPT-2 style
                 transformer_outputs = hf_model.transformer(inputs_embeds=embeddings)
            else:
                 # Fallback: Pass directly to the model, hoping it handles inputs_embeds
                 # This might double-apply embeddings if not careful! Requires checking model type.
                 # For now, assume Llama-like structure hf_model.model exists.
                 raise NotImplementedError("Cannot determine main transformer body for gradient calculation.")

            hidden_states = transformer_outputs.last_hidden_state # (batch_size, ctx_len, d_model)

            # 3. Apply final norm and LM head manually to last hidden state
            last_hidden_state = hidden_states[:, -1]
            normed_hidden_state = final_norm(last_hidden_state)
            logits = lm_head(normed_hidden_state) # (batch_size, d_vocab)

            # 4. Get target logit and backpropagate
            target_logit = logits[:, target]
            target_logit.sum().backward()

            # 5. Get embedding gradient and calculate "one-hot" equivalent gradient
            embedding_grad = embeddings.grad # (batch_size, ctx_len, d_model)
            # Project gradients back to vocab space: d_logit/d_onehot = (d_logit/d_embed) @ W_E.T
            onehot_grad = einsum("b s d, v d -> b s v", embedding_grad, W_E) # (batch_size, ctx_len, d_vocab)

            # --- End Gradient Calculation ---

            # Calculate probability of target token based on computed logits
            # Note: Use the *actual* model output logits here
            probs = (logits.argmax(-1) == target).float().detach()
            imp_samp_probs.append((th.exp(logratios.sum(-1)) * probs))

            # Update scores using the calculated "one-hot" gradients
            # Aggregate gradient across batch dimension
            batch_grad = onehot_grad.sum(0) # (ctx_len, d_vocab)
            scores *= decay_rate
            scores += batch_grad / batch_size # Apply EWMA update

            # Detach grads for next iteration
            embeddings.grad = None


    imp_samp_probs = th.cat(imp_samp_probs, dim=0)
    return imp_samp_probs.mean().item()

def MHIS(
    model: 'Actor', # Expect an Actor instance
    orig_dists: list[Discrete],
    target: int,
    *,\
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
    - target: the target token (in range [0...d_vocab)).
    - temp: the temperature (for both the Boltzmann stationary distribution and proposal distribution)
    - n_samples: the total number of samples to be drawn, ignoring burn-in.
    - batch_size: the number of parallel random walks to run (the total number of samples drawn is n_samples + burn_in * batch_size)
    Returns:
    - The estimated probability of outputing token `target`.
    """
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

            # 4. Get target logit score s(x) and backpropagate
            target_logit_score = logits[:, target]
            target_logit_score.sum().backward()

            # 5. Get embedding gradient and calculate "one-hot" equivalent gradient
            embedding_grad = embeddings.grad # (batch_size, ctx_len, d_model)
            onehot_grad = einsum("b s d, v d -> b s v", embedding_grad, W_E) # (batch_size, ctx_len, d_vocab)

            # Detach results
            scores_out = target_logit_score.detach().clone()
            grads_out = onehot_grad.detach().clone()
            results_out = (logits.argmax(dim=-1) == target).float().detach().clone()

            # Clear gradients for next use
            embeddings.grad = None
            # Zero grads on the model parameters just in case, although they shouldn't accumulate
            # hf_model.zero_grad(set_to_none=True)
            # It might be safer to manually zero potentially affected params if needed
            # but disabling requires_grad initially should prevent accumulation.

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
        proposed_tokens = th.multinomial(proposal_probs, 1).squeeze(-1)

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
        p_rev = reverse_proposal_probs[th.arange(batch_size), current_tokens_at_pos] # q(x | x')
        p_fwd = proposal_probs[th.arange(batch_size), proposed_tokens]             # q(x'| x)

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
                           orig_log_probs[pos, proposed_tokens] - orig_log_probs[pos, current_tokens_at_pos] + \
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

    if mean_inv_exp_scores == 0:
        print("Warning: Estimated normalizing constant is zero. Cannot compute estimate.")
        # This might happen if scores are extremely large
        # Could indicate numerical instability or issues with the sampling
        return float('nan') # Or handle appropriately

    # Calculate the unbiased estimate E_p_orig[f(x)]
    # mean( f(x_i) * w_i ) / mean( w_i )
    unbiased_estimate_numerator = (results_tensor * inv_exp_scores).mean()
    estimated_prob = unbiased_estimate_numerator / mean_inv_exp_scores

    return estimated_prob.item()