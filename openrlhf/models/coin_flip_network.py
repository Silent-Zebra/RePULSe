from typing import Optional
import copy

import torch
import torch.nn as nn
from transformers import AutoConfig

from .utils import masked_mean, reset_position_ids




class CoinFlipNetwork(nn.Module):
    """
    Coin Flip Network for intrinsic exploration bonuses.
    
    This network learns to estimate state visitation counts by minimizing MSE loss
    against randomly generated coin flip vectors. The intrinsic reward is computed as:
    r_I(x) = sqrt((1/d) * ||f_φ(x)||^2) ≈ 1/sqrt(n_x)
    
    Args:
        base_model: The base transformer model (typically from Actor)
        coin_flip_dim: Dimension d for coin flip vectors (default: 64)
        normalization_momentum: Momentum for exponential moving average of running statistics
            used to normalize the exploration bonus. If None, normalization is disabled (default: None)
        head_init_std: Standard deviation for initializing the coin flip head weights (default: 0.001)
        frozen_prior_init_std: Standard deviation for initializing the frozen prior network weights (default: 0.1)
        coin_flip_linear_bias: If True, adds bias to the linear head for the trainable coin flip network (default: False)
        base_actor_learning_rate: Learning rate of the base actor. If provided and != 0, raises
            NotImplementedError as the random prior structure should be reviewed when base_model is trainable (default: None)
    """
    
    def __init__(
        self, 
        base_model: nn.Module, 
        coin_flip_dim: int = 64, 
        normalization_momentum: Optional[float] = None,
        head_init_std: float = 0.001,
        frozen_prior_init_std: float = 0.1,
        coin_flip_linear_bias: bool = False,
        base_actor_learning_rate: Optional[float] = None,
    ):
        super().__init__()
        self.coin_flip_dim = coin_flip_dim
        
        # TODO: Review random prior structure when base_model becomes trainable (base actor LR != 0)
        # Currently assumes base_model is frozen. When training the base actor, the random prior
        # approach may need adjustment.
        if base_actor_learning_rate is not None and abs(base_actor_learning_rate) > 1e-10:
            raise NotImplementedError(
                "Coin flip network with random prior is not yet implemented for trainable base_model. "
                "The random prior structure should be reviewed when base_actor_learning_rate != 0."
            )
        
        # Get the base model (unwrap if it's an Actor)
        if hasattr(base_model, 'model'):
            # It's an Actor wrapper
            unwrapped_model = base_model.model
        else:
            # It's already the base transformer
            unwrapped_model = base_model
        
        # Unwrap DeepSpeed engine if present to get the actual PyTorch model
        try:
            import deepspeed
            if isinstance(unwrapped_model, deepspeed.DeepSpeedEngine):
                # Get the actual module from DeepSpeed engine
                unwrapped_model = unwrapped_model.module
        except (ImportError, AttributeError):
            pass
        
        # Create a deep copy of the base model to ensure complete separation
        # This ensures the coin flip network is entirely independent from the
        # base/sampling actors and won't be interfered with by their training
        self.base_model = copy.deepcopy(unwrapped_model)
        
        # Get hidden size from config or model architecture
        hidden_size = None
        
        # First, try to get from config
        if hasattr(self.base_model, 'config'):
            config = self.base_model.config
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            elif hasattr(config, 'd_model'):  # Some models use d_model instead
                hidden_size = config.d_model
            elif hasattr(config, 'n_embd'):  # GPT-2 style models
                hidden_size = config.n_embd
        
        # If not found, try from original base_model config
        if hidden_size is None and hasattr(base_model, 'config'):
            config = base_model.config
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            elif hasattr(config, 'd_model'):
                hidden_size = config.d_model
            elif hasattr(config, 'n_embd'):
                hidden_size = config.n_embd
        
        # If still not found, try to infer from model architecture
        if hidden_size is None:
            # Try to find the LM head (input dimension = hidden_size)
            if hasattr(self.base_model, 'lm_head') and hasattr(self.base_model.lm_head, 'in_features'):
                hidden_size = self.base_model.lm_head.in_features
            
            # Try to find from transformer structure (for distilgpt2 and similar models)
            if hidden_size is None and hasattr(self.base_model, 'transformer'):
                transformer = self.base_model.transformer
                # Check final layer norm (distilgpt2, GPT-2 style)
                if hasattr(transformer, 'ln_f') and hasattr(transformer.ln_f, 'normalized_shape'):
                    hidden_size = transformer.ln_f.normalized_shape[0]
                # Check transformer blocks
                elif hasattr(transformer, 'h') and len(transformer.h) > 0:
                    last_block = transformer.h[-1]
                    if hasattr(last_block, 'ln_2') and hasattr(last_block.ln_2, 'normalized_shape'):
                        hidden_size = last_block.ln_2.normalized_shape[0]
            
            # Try using base_model_prefix approach
            if hidden_size is None and hasattr(self.base_model, 'base_model_prefix'):
                base_model_prefix = self.base_model.base_model_prefix
                base = getattr(self.base_model, base_model_prefix, None)
                if base is not None:
                    # Try to find the output dimension of the transformer layers
                    # Look for the last layer norm or the last transformer block
                    if hasattr(base, 'ln_f') and hasattr(base.ln_f, 'normalized_shape'):
                        # LayerNorm normalized_shape is a tuple, take the first element
                        hidden_size = base.ln_f.normalized_shape[0]
                    elif hasattr(base, 'layer_norm') and hasattr(base.layer_norm, 'normalized_shape'):
                        hidden_size = base.layer_norm.normalized_shape[0]
                    # Try to find from transformer blocks
                    if hidden_size is None and hasattr(base, 'h') and len(base.h) > 0:
                        # GPT-2 style: check the last transformer block
                        last_block = base.h[-1]
                        if hasattr(last_block, 'ln_2') and hasattr(last_block.ln_2, 'normalized_shape'):
                            hidden_size = last_block.ln_2.normalized_shape[0]
                    elif hidden_size is None and hasattr(base, 'layers') and len(base.layers) > 0:
                        # Other architectures: check the last layer
                        last_layer = base.layers[-1]
                        if hasattr(last_layer, 'norm') and hasattr(last_layer.norm, 'normalized_shape'):
                            hidden_size = last_layer.norm.normalized_shape[0]
        
        # If still not found, do a test forward pass to infer the dimension
        if hidden_size is None:
            try:
                # Get device from base_model, prioritizing GPU/cuda
                base_model_device = None
                for param in self.base_model.parameters():
                    base_model_device = param.device
                    break
                
                # Fallback: use cuda if available, else cpu
                if base_model_device is None:
                    base_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Create a dummy input to infer the hidden size, on the same device as base_model
                dummy_input = torch.zeros(1, 1, dtype=torch.long, device=base_model_device)
                
                with torch.no_grad():
                    outputs = self.base_model(
                        dummy_input,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    if "hidden_states" in outputs and len(outputs["hidden_states"]) > 0:
                        hidden_size = outputs["hidden_states"][-1].shape[-1]
                    elif "last_hidden_state" in outputs:
                        hidden_size = outputs["last_hidden_state"].shape[-1]
            except Exception:
                pass
        
        if hidden_size is None:
            raise ValueError("Could not determine hidden_size for CoinFlipNetwork. "
                           "Tried config, model architecture, and test forward pass.")
        else:
            print("Determined hidden_size for CoinFlipNetwork:", hidden_size)
        
        # Create coin flip head: maps hidden_size -> coin_flip_dim
        self.coin_flip_head = nn.Linear(hidden_size, coin_flip_dim, bias=coin_flip_linear_bias)
        
        # Move coin_flip_head to the same device as base_model
        # Get device from base_model parameters, prioritizing GPU/cuda
        base_model_device = None
        for param in self.base_model.parameters():
            base_model_device = param.device
            break
        
        # If no parameters found, use cuda if available, else cpu
        if base_model_device is None:
            base_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.coin_flip_head = self.coin_flip_head.to(base_model_device)
        
        # Reinitialize coin flip head with custom standard deviation for smaller initial outputs
        nn.init.normal_(self.coin_flip_head.weight, mean=0.0, std=head_init_std)
        if coin_flip_linear_bias and self.coin_flip_head.bias is not None:
            nn.init.zeros_(self.coin_flip_head.bias)
        
        # Create random prior head: frozen linear layer with same architecture
        # This ensures new states have ~1 pseudocount at initialization
        self.random_prior_head = nn.Linear(hidden_size, coin_flip_dim, bias=False)
        self.random_prior_head = self.random_prior_head.to(base_model_device)
        nn.init.normal_(self.random_prior_head.weight, mean=0.0, std=frozen_prior_init_std)

        # Freeze the random prior head - it should never be trained
        for param in self.random_prior_head.parameters():
            param.requires_grad = False
        
        # Freeze the base model - we only train the coin_flip_head
        # This avoids DeepSpeed ZeRO hook conflicts and keeps training simple
        # The optimizer will automatically exclude frozen parameters (requires_grad=False)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Support gradient checkpointing if base model does
        self.supports_gradient_checkpointing = getattr(self.base_model, 'supports_gradient_checkpointing', False)
        
        # Running statistics for normalization of exploration bonus
        # Using exponential moving average with momentum
        self.normalization_momentum = normalization_momentum
        if normalization_momentum is not None:
            raise NotImplementedError("Need to double check this, including initializations")
            self.momentum = normalization_momentum
            self.register_buffer('running_mean', torch.zeros(1, device=base_model_device))
            self.register_buffer('running_var', torch.zeros(1, device=base_model_device))
            self.register_buffer('num_updates', torch.zeros(1, dtype=torch.long, device=base_model_device))
        else:
            self.momentum = None
        
        # Running statistics for normalization of random prior outputs (per dimension)
        # These ensure the random prior contributes ~1 pseudocount
        # Normalize each of the d dimensions to have mean 0, std 1
        # Uses Welford's online algorithm (same as CFN implementation)
        # Track statistics per dimension: shape (coin_flip_dim,)
        self.register_buffer('prior_running_mean', torch.zeros(coin_flip_dim, device=base_model_device))
        self.register_buffer('prior_running_var', torch.zeros(coin_flip_dim, device=base_model_device))
        self.register_buffer('prior_num_updates', torch.zeros(1, dtype=torch.long, device=base_model_device))
    
    def _get_device(self):
        """Get the device of the base model, prioritizing GPU/cuda."""
        # Try to get device from base_model parameters
        for param in self.base_model.parameters():
            return param.device
        
        # Fallback: use cuda if available, else cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_final_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_outputs: bool = False,
    ):
        """
        Extract final hidden states from sequences using the same logic as forward.
        
        This is a helper method that can be used by training code to get final hidden states
        for saving to replay buffers, without computing the full forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            return_outputs: If True, also return base model outputs
            
        Returns:
            Final hidden states, shape (batch_size, hidden_size)
            If return_outputs=True, also returns base model outputs
        """
        # Compute position_ids
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None
        
        # Forward through base model
        # Note: No torch.no_grad() here - let caller decide if gradients are needed
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get last hidden state (all sequence positions)
        # This contains hidden states for all positions in the sequence
        if "hidden_states" in outputs:
            hidden_states = outputs["hidden_states"][-1]  # (batch_size, seq_len, hidden_size)
        elif "last_hidden_state" in outputs:
            hidden_states = outputs["last_hidden_state"]  # (batch_size, seq_len, hidden_size)
        else:
            raise ValueError("Model outputs must contain either 'hidden_states' or 'last_hidden_state'")
        
        # Ensure hidden_states and coin_flip_head are on the same device
        coin_flip_head_device = next(self.coin_flip_head.parameters()).device
        if hidden_states.device != coin_flip_head_device:
            hidden_states = hidden_states.to(coin_flip_head_device)
        
        # Extract final token hidden states (last valid position for each sequence)
        # Only final states are used in reward computation and training
        if attention_mask is not None:
            # Find the last valid position for each sequence
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1, keepdim=True)
            # Use advanced indexing to extract final hidden states: (batch_size, hidden_size)
            batch_size = hidden_states.size(0)
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            final_hidden_states = hidden_states[batch_indices, eos_indices.squeeze(1), :]  # (batch_size, hidden_size)
        else:
            # Use last position
            final_hidden_states = hidden_states[:, -1, :]  # (batch_size, hidden_size)
        
        if return_outputs:
            return final_hidden_states, outputs
        return final_hidden_states
    
    def _predict_from_embeddings(
        self,
        final_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined predictions from final hidden states.
        
        This method takes already-extracted final hidden states and applies the coin flip head,
        random prior head, and normalization to produce combined predictions. This is used
        for training on embeddings from the replay buffer, ensuring consistency with the
        forward method.
        
        Args:
            final_hidden_states: Final hidden states, shape (batch_size, hidden_size)
            
        Returns:
            Combined predictions, shape (batch_size, coin_flip_dim)
        """
        # Apply coin flip head (trainable) only to final states
        coin_flip_predictions = self.coin_flip_head(final_hidden_states)  # (batch_size, coin_flip_dim)
        
        # Apply random prior head (frozen) only to final states
        # This ensures new states have ~1 pseudocount at initialization
        random_prior_final = self.random_prior_head(final_hidden_states)  # (batch_size, coin_flip_dim)
        
        # Normalize random prior outputs dimension-wise to have mean 0, std 1
        # This ensures sqrt((1/d) * ||normalized_prior||^2) has expectation 1
        # Statistics are computed only on final states using Welford's algorithm
        normalized_random_prior_final = self._normalize_with_welford_per_dim(
            random_prior_final,
            self.prior_running_mean,
            self.prior_running_var,
            self.prior_num_updates
        )  # (batch_size, coin_flip_dim)
        
        # Combine main predictions with normalized random prior predictions
        combined_predictions = coin_flip_predictions + normalized_random_prior_final  # (batch_size, coin_flip_dim)
        
        return combined_predictions
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the coin flip network.
        
        Only computes predictions for final states (last valid token per sequence),
        as only final states are used in reward computation and training.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            return_output: If True, also return base model outputs
            
        Returns:
            Coin flip predictions for final states, shape (batch_size, coin_flip_dim)
            If return_output=True, also returns base model outputs
        """
        # Get final hidden states using the shared helper method
        # Also get outputs if needed to avoid recomputation
        if return_output:
            final_hidden_states, outputs = self._get_final_hidden_states(
                input_ids, attention_mask, return_outputs=True
            )
        else:
            final_hidden_states = self._get_final_hidden_states(input_ids, attention_mask)
            outputs = None
        
        # Compute combined predictions from final hidden states
        combined_predictions = self._predict_from_embeddings(final_hidden_states)
        
        # Combined bonus (for statistics)
        combined_norm_squared = (combined_predictions ** 2).sum(dim=-1)  # (B,)
        combined_bonus = torch.sqrt(combined_norm_squared / self.coin_flip_dim)  # (B,)
        
        # Get individual components for printing statistics
        coin_flip_predictions = self.coin_flip_head(final_hidden_states)
        random_prior_final = self.random_prior_head(final_hidden_states)
        normalized_random_prior_final = self._normalize_with_welford_per_dim(
            random_prior_final,
            self.prior_running_mean,
            self.prior_running_var,
            self.prior_num_updates
        )
        
        # Print statistics
        print(f"[Coin Flip Network] Predictions: {coin_flip_predictions}")
        print(f"[Random Prior] Values: {random_prior_final}, ")
        print(f"[Random Prior] Values (normalized): {normalized_random_prior_final}, ")
        print(f"[Combined] Values: {combined_predictions}, ")
        print(f"[Combined] Bonus - Mean: {combined_bonus.mean().item():.6f}, "
              f"Min: {combined_bonus.min().item():.6f}, Max: {combined_bonus.max().item():.6f}")
        print(f"[Random Prior Running Stats] Mean: {self.prior_running_mean.mean().item():.6f} "
              f"(per-dim range: [{self.prior_running_mean.min().item():.6f}, {self.prior_running_mean.max().item():.6f}]), "
              f"Var: {self.prior_running_var.mean().item():.6f} "
              f"(per-dim range: [{self.prior_running_var.min().item():.6f}, {self.prior_running_var.max().item():.6f}])")
        
        if return_output:
            return combined_predictions, outputs
        return combined_predictions
    
    def compute_intrinsic_reward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        bonus_alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward based on coin flip network output.
        
        The intrinsic reward is: r_I(x) = bonus_alpha * sqrt((1/d) * ||f_φ(x)||^2)
        which approximates 1/sqrt(n_x) where n_x is the visitation count.
        
        Uses only the final token output (final state) since reward is computed
        over the full sequence.
        
        Args:
            sequences: Input sequences, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            bonus_alpha: Scaling factor for the intrinsic reward
            
        Returns:
            Intrinsic rewards per sequence, shape (batch_size,)
        """
        # Get coin flip predictions (already for final states only)
        final_predictions = self.forward(sequences, attention_mask)  # (B, d)
        
        # Compute ||f_φ(x)||^2 for final token: sum over coin_flip_dim dimension
        norm_squared = (final_predictions ** 2).sum(dim=-1)  # (B,)
        
        # Compute intrinsic reward: sqrt((1/d) * ||f_φ(x)||^2)
        intrinsic_reward = torch.sqrt(norm_squared / self.coin_flip_dim)
        
        # Normalize the exploration bonus using running mean and variance (if enabled)
        if self.normalization_momentum is not None:
            intrinsic_reward = self._normalize_bonus(intrinsic_reward)
        
        # Apply correction when adjust_reward is True (train_coin_flip_before mode)
        # This corrects from 1/sqrt(n+1) to 1/sqrt(n) by removing the +1 pseudocount
        # from the fixed random prior
        if getattr(self, 'adjust_reward', False):
            raise NotImplementedError("Need to check this first")
            # Correction: invert, square, subtract 1, square root, invert again
            # This transforms 1/sqrt(n+1) to 1/sqrt(n)
            # Add small epsilon to avoid numerical issues when intrinsic_reward is very small
            inv_squared = (1.0 / (intrinsic_reward + 1e-8)) ** 2
            # Clamp to ensure we don't take sqrt of negative values
            sqrt_arg = torch.clamp(inv_squared - 1.0, min=1e-8)
            intrinsic_reward = 1.0 / torch.sqrt(sqrt_arg)

        intrinsic_reward *= bonus_alpha
        
        return intrinsic_reward
    
    def _update_running_stats(
        self, 
        values: torch.Tensor, 
        running_mean: torch.Tensor, 
        running_var: torch.Tensor, 
        num_updates: torch.Tensor,
        momentum: float
    ) -> None:
        """
        Update running statistics using exponential moving average.
        
        For the first update, initializes with batch statistics. Subsequent updates
        use exponential moving average.
        
        Args:
            values: Batch of values to compute statistics for, shape (batch_size,)
            running_mean: Buffer storing running mean
            running_var: Buffer storing running variance
            num_updates: Buffer storing number of updates
            momentum: Momentum for exponential moving average
        """
        # Compute batch statistics
        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)  # Use biased variance for consistency
        
        # Update running statistics using exponential moving average
        # For the first update, initialize with batch statistics
        if num_updates.item() == 0:
            running_mean.data = batch_mean
            running_var.data = batch_var
        else:
            # Exponential moving average update
            running_mean.data = momentum * running_mean + (1 - momentum) * batch_mean
            running_var.data = momentum * running_var + (1 - momentum) * batch_var
        
        num_updates.data += 1
    
    def _normalize_values(
        self, 
        values: torch.Tensor, 
        running_mean: torch.Tensor, 
        running_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize values by subtracting running mean and dividing by running standard deviation.
        
        Args:
            values: Values to normalize, shape (batch_size,)
            running_mean: Running mean
            running_var: Running variance
            
        Returns:
            Normalized values, shape (batch_size,)
        """
        # Normalize using current running statistics
        # Add small epsilon to avoid division by zero
        normalized = (values - running_mean) / (torch.sqrt(running_var) + 1e-8)
        return normalized
    
    def _normalize_with_stats_update(
        self,
        values: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        num_updates: torch.Tensor,
        momentum: float
    ) -> torch.Tensor:
        """
        Update running statistics and normalize values in one step.
        
        This is a convenience method that combines _update_running_stats and _normalize_values
        to avoid duplicate code.
        
        Args:
            values: Values to normalize, shape (batch_size,)
            running_mean: Buffer storing running mean, shape (1,)
            running_var: Buffer storing running variance, shape (1,)
            num_updates: Buffer storing number of updates
            momentum: Momentum for exponential moving average
            
        Returns:
            Normalized values with mean ~0, std ~1, shape (batch_size,)
        """
        # Update running statistics
        self._update_running_stats(values, running_mean, running_var, num_updates, momentum)
        
        # Normalize using updated running statistics
        return self._normalize_values(values, running_mean, running_var)
    
    def _normalize_with_welford_per_dim(
        self,
        values: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        num_updates: torch.Tensor
    ) -> torch.Tensor:
        """
        Update running statistics using Welford's online algorithm and normalize values per dimension.
        
        This uses Welford's algorithm (same as CFN implementation) which computes true running
        statistics rather than exponential moving average. This is better for the fixed prior network.
        
        The CFN code processes one sample at a time. For batches, we process each sample sequentially
        (vectorized) to maintain the same update formula.
        
        Args:
            values: Values to normalize, shape (batch_size, num_dims)
            running_mean: Buffer storing running mean per dimension, shape (num_dims,)
            running_var: Buffer storing running variance per dimension, shape (num_dims,)
            num_updates: Buffer storing number of updates (will be incremented by batch_size)
            
        Returns:
            Normalized values with mean ~0, std ~1 per dimension, shape (batch_size, num_dims)
        """
        batch_size = values.shape[0]
        
        # Process each sample in the batch sequentially (vectorized where possible)
        # This matches the CFN implementation which processes one sample at a time
        for i in range(batch_size):
            value = values[i]  # (num_dims,)
            effective_iter = num_updates.item() + 1  # +1 because we're about to update, and start at 0

            # Ok so let's define n as the number of updates we've done so far
            # n starts at 0 and increments by 1 each time we update
            
            # Welford's algorithm
            # First let's update the mean:
            # mean_new = (mean_old * n + value) / (n+1)
            # = (mean_old * n + mean_old + value - mean_old) / (n+1)
            # With delta := value - mean_old
            delta = value - running_mean
            # mean_new = (mean_old * (n+1) + delta) / (n+1)
            # = mean_old + delta / (n+1)
            running_mean.data = running_mean + delta / effective_iter
            # So if the code we uses divides by effective_iter, then effective_iter = n+1 by necessity; incrementing must be done first
            
            # Now to update the variance:
            squared_delta = delta ** 2
            # Variance = sum of squared deviations from the mean / (number of samples) (no correction for bias here, since we're correcting the same set of samples)
            # running_var * n = sum of squared deviations from previous mean
            # Let the sum of squared deviations from the previous mean be M^2_n
            # and let the sum of squared deviations from the new mean be M^2_{n+1} 
            # and let x_i denote the i-th value
            # M^2_{n+1} = sum_{i=1}^{n+1} (x_i - mean_new)^2 = sum_{i=1}^{n} (x_i - mean_new)^2 + (x_{n+1} - mean_new)^2
            # Then since x_i - mean_new = (x_i - mean_old) + (mean_old - mean_new)
            # squaring and summing both sides, the cross term will disappear since sum of (x_i - mean_old) is 0
            # Then we get that sum_{i=1}^{n} (x_i - mean_new)^2 = sum_{i=1}^{n} (x_i - mean_old)^2 + sum_{i=1}^{n} (mean_old - mean_new)^2
            # = M^2_n + n * (mean_old - mean_new)^2
            # So M^2_{n+1} = M^2_n + n * (mean_old - mean_new)^2 + (x_{n+1} - mean_new)^2
            # Now recall that delta = x_{n+1} - mean_old, and mean_new = mean_old + delta / (n+1)
            # So mean_old - mean_new = - delta / (n+1)
            # Also note that delta = x_{n+1} - mean_old = x_{n+1} - mean_old + mean_new - mean_new
            # So x_{n+1} - mean_new = delta + mean_old - mean_new 
            # = delta - (delta / (n+1)) = ((n+1) - 1) * delta / (n+1)
            # So M^2_{n+1} = M^2_n + n * (- delta / (n+1))^2 + (((n+1) - 1) * delta / (n+1))^2
            # = M^2_n + n * (- delta / (n+1))^2 + (n * delta / (n+1))^2
            # = M^2_n + delta^2 (n + n^2) / (n+1)^2
            # = M^2_n + delta^2 n(n+1) / (n+1)^2
            # = M^2_n + delta^2 n / (n+1)
            # Then since effective_iter = n+1, we get that:
            # M^2_{n+1} = M^2_n + delta^2 * (effective_iter - 1) / effective_iter
            # Finally, to get the new variance, we need
            # variance = M^2_{n+1} / (n+1)
            # = M^2_{n+1} / effective_iter

            # Then add the new squared deviation to get the new sum of squared deviations
            # From the derivation: M^2_{n+1} = M^2_n + delta^2 * (effective_iter - 1) / effective_iter
            # where n = effective_iter - 1, so M^2_n = running_var * n = running_var * (effective_iter - 1)
            # Then variance_new = M^2_{n+1} / effective_iter
            n = effective_iter - 1
            
            # M^2_n = running_var * n (sum of squared deviations from previous mean)
            M_squared_n = running_var * n
            # M^2_{n+1} = M^2_n + delta^2 * n / effective_iter
            M_squared_new = M_squared_n + squared_delta * n / effective_iter
            # variance_new = M^2_{n+1} / effective_iter
            running_var.data = M_squared_new / effective_iter
            # Note: variance should indeed be 0 when n=0 (first update), so no need to handle separate cases
            
            # Increment update counter
            num_updates.data += 1
        
        # Normalize all values using the final updated statistics
        # Add small epsilon to avoid division by zero
        normalized = (values - running_mean.unsqueeze(0)) / (torch.sqrt(running_var.unsqueeze(0)) + 1e-8)
        return normalized
    
    def _normalize_with_stats_update_per_dim(
        self,
        values: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        num_updates: torch.Tensor,
        momentum: float
    ) -> torch.Tensor:
        """
        Update running statistics and normalize values per dimension in one step.
        
        This normalizes each dimension independently to have mean 0, std 1.
        Uses exponential moving average (for non-prior statistics).
        
        Args:
            values: Values to normalize, shape (batch_size, num_dims)
            running_mean: Buffer storing running mean per dimension, shape (num_dims,)
            running_var: Buffer storing running variance per dimension, shape (num_dims,)
            num_updates: Buffer storing number of updates
            momentum: Momentum for exponential moving average
            
        Returns:
            Normalized values with mean ~0, std ~1 per dimension, shape (batch_size, num_dims)
        """
        # Compute batch statistics per dimension
        # Mean and var across first dimension (batch), keeping feature dimensions
        batch_mean = values.mean(dim=0)  # (num_dims,)
        batch_var = values.var(dim=0, unbiased=False)  # (num_dims,)
        
        # Update running statistics using exponential moving average
        # For the first update, initialize with batch statistics
        if num_updates.item() == 0:
            running_mean.data = batch_mean
            running_var.data = batch_var
        else:
            # Exponential moving average update per dimension
            running_mean.data = momentum * running_mean + (1 - momentum) * batch_mean
            running_var.data = momentum * running_var + (1 - momentum) * batch_var
        
        num_updates.data += 1
        
        # Normalize per dimension using updated running statistics
        # Add small epsilon to avoid division by zero
        normalized = (values - running_mean) / (torch.sqrt(running_var) + 1e-8)
        return normalized
    
    def _normalize_bonus(self, bonus: torch.Tensor) -> torch.Tensor:
        """
        Normalize the exploration bonus by subtracting running mean and dividing by running variance.
        
        Uses exponential moving average to update running statistics. Statistics are updated
        even when the model is in eval mode, since we want to track the distribution of bonuses
        during experience collection.
        
        Args:
            bonus: Exploration bonus tensor, shape (batch_size,)
            
        Returns:
            Normalized bonus tensor, shape (batch_size,)
        """
        if self.momentum is None:
            # Normalization is disabled, return bonus as-is
            return bonus
        
        # Update running statistics and normalize
        return self._normalize_with_stats_update(
            bonus,
            self.running_mean,
            self.running_var,
            self.num_updates,
            self.momentum
        )
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Backward pass for the coin flip network.
        
        This method is called by the training strategy (e.g., DeepSpeedStrategy).
        Since base_model is frozen, we only backpropagate through coin_flip_head,
        avoiding DeepSpeed ZeRO hook conflicts.
        
        Args:
            loss: The loss tensor to backpropagate
        """
        # Standard PyTorch backward pass
        # Since base_model is frozen (requires_grad=False), gradients won't flow
        # through it, so DeepSpeed hooks on base_model won't interfere
        loss.backward()
    
    def step(self) -> None:
        """
        Optimizer step for the coin flip network.
        
        This method is called by the training strategy (e.g., DeepSpeedStrategy).
        If the model is wrapped by DeepSpeed, the wrapper's step method will be used instead,
        which handles optimizer stepping internally.
        Otherwise, this method is a no-op (optimizer stepping is handled in training code).
        """
        # No-op for non-DeepSpeed models
        # If this model is wrapped by DeepSpeed, the wrapper's step will be called instead
        # Otherwise, optimizer.step() is called manually in the training code
        pass
    
    def get_trainable_parameters(self):
        """
        Get only the trainable parameters (coin_flip_head only, base_model and random_prior_head are frozen).
        
        Returns:
            Iterator over trainable parameters
        """
        return self.coin_flip_head.parameters()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        """Enable gradient checkpointing if supported."""
        if self.supports_gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if self.supports_gradient_checkpointing:
            self.base_model.gradient_checkpointing_disable()

