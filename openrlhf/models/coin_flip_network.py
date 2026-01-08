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
    """
    
    def __init__(self, base_model: nn.Module, coin_flip_dim: int = 64):
        super().__init__()
        self.coin_flip_dim = coin_flip_dim
        
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
        self.coin_flip_head = nn.Linear(hidden_size, coin_flip_dim, bias=False)
        
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
        
        # Freeze the base model - we only train the coin_flip_head
        # This avoids DeepSpeed ZeRO hook conflicts and keeps training simple
        # The optimizer will automatically exclude frozen parameters (requires_grad=False)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Support gradient checkpointing if base model does
        self.supports_gradient_checkpointing = getattr(self.base_model, 'supports_gradient_checkpointing', False)
        
        # Running statistics for normalization of exploration bonus
        # Using exponential moving average with momentum
        self.momentum = 0.99  # EMA momentum for running statistics
        self.register_buffer('running_mean', torch.zeros(1, device=base_model_device))
        self.register_buffer('running_var', torch.ones(1, device=base_model_device))
        self.register_buffer('num_updates', torch.zeros(1, dtype=torch.long, device=base_model_device))
    
    def _get_device(self):
        """Get the device of the base model, prioritizing GPU/cuda."""
        # Try to get device from base_model parameters
        for param in self.base_model.parameters():
            return param.device
        
        # Fallback: use cuda if available, else cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the coin flip network.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            return_output: If True, also return base model outputs
            
        Returns:
            Coin flip predictions, shape (batch_size, seq_len, coin_flip_dim)
            If return_output=True, also returns base model outputs
        """
        # Compute position_ids
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None
        
        # Forward through base model
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get hidden states (last hidden state)
        # hidden_states is a tuple, extract the last one
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
        
        # Apply coin flip head
        coin_flip_predictions = self.coin_flip_head(hidden_states)  # (batch_size, seq_len, coin_flip_dim)
        
        if return_output:
            return coin_flip_predictions, outputs
        return coin_flip_predictions
    
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
        # Get coin flip predictions
        coin_flip_predictions = self.forward(sequences, attention_mask)  # (B, S, d)
        
        # Extract final token predictions (last valid position for each sequence)
        if attention_mask is not None:
            # Find the last valid position for each sequence (same as reward model does)
            # fliplr() is deprecated, use flip() instead
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1, keepdim=True)
            # Use advanced indexing to extract final predictions: (B, d)
            batch_indices = torch.arange(coin_flip_predictions.size(0), device=coin_flip_predictions.device)
            final_predictions = coin_flip_predictions[batch_indices, eos_indices.squeeze(1), :]  # (B, d)
        else:
            # Use last position
            final_predictions = coin_flip_predictions[:, -1, :]  # (B, d)
        
        # Compute ||f_φ(x)||^2 for final token: sum over coin_flip_dim dimension
        norm_squared = (final_predictions ** 2).sum(dim=-1)  # (B,)
        
        # Compute intrinsic reward: sqrt((1/d) * ||f_φ(x)||^2)
        intrinsic_reward = torch.sqrt(norm_squared / self.coin_flip_dim)
        
        # Normalize the exploration bonus using running mean and variance
        intrinsic_reward = self._normalize_bonus(intrinsic_reward)
        intrinsic_reward *= bonus_alpha
        
        return intrinsic_reward
    
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
        # Compute batch statistics
        batch_mean = bonus.mean()
        batch_var = bonus.var(unbiased=False)  # Use biased variance for consistency
        
        # Update running statistics using exponential moving average
        # For the first update, initialize with batch statistics
        if self.num_updates.item() == 0:
            self.running_mean.data = batch_mean
            self.running_var.data = batch_var
        else:
            # Exponential moving average update
            self.running_mean.data = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var.data = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        
        self.num_updates.data += 1
        
        # Normalize using current running statistics
        # Add small epsilon to avoid division by zero
        normalized = (bonus - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        return normalized
    
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
        Get only the trainable parameters (coin_flip_head only, base_model is frozen).
        
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

