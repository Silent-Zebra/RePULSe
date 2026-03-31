from typing import Optional, Tuple
import copy

import torch
import torch.nn as nn
from transformers import AutoConfig

from .utils import masked_mean, reset_position_ids

# Architecture constants
STATIC_ARCHITECTURES = ["linear_head_on_static_initial_base"]
LEARNING_ARCHITECTURES = ["linear_head_on_learning_base", "linear_head_on_learning_proposal"]
TOKEN_STORAGE_ARCHITECTURES = ["separate_nn"] + LEARNING_ARCHITECTURES


def _get_param_dtype(module: nn.Module) -> torch.dtype:
    """Return the dtype of the first parameter of the module (e.g. backbone output dtype)."""
    for param in module.parameters():
        return param.dtype
    return torch.float32  # fallback if module has no parameters


def _compute_position_ids(attention_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """
    Compute position_ids from attention_mask.

    Args:
        attention_mask: Attention mask, shape (batch_size, seq_len)

    Returns:
        Position IDs, shape (batch_size, seq_len), or None if attention_mask is None
    """
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids
    return None


def _extract_hidden_states_from_outputs(outputs: dict) -> torch.Tensor:
    """
    Extract last-layer hidden states from model outputs dictionary.

    Args:
        outputs: Model outputs dictionary

    Returns:
        Hidden states tensor, shape (batch_size, seq_len, hidden_size)
    """
    if "hidden_states" in outputs:
        return outputs["hidden_states"][-1]
    elif "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    else:
        raise ValueError("Model outputs must contain either 'hidden_states' or 'last_hidden_state'")


def _extract_final_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Extract final token hidden states from sequence hidden states.

    Args:
        hidden_states: Hidden states for all positions, shape (batch_size, seq_len, hidden_size)
        attention_mask: Attention mask, shape (batch_size, seq_len)

    Returns:
        Final hidden states, shape (batch_size, hidden_size)
    """
    if attention_mask is not None:
        # Find the last valid position for each sequence
        assert attention_mask.any(dim=1).all(), (
            "attention_mask has all-zero rows — no valid tokens. "
            "This would cause _extract_final_hidden_states to return hidden states at invalid positions."
        )
        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1, keepdim=True)
        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        final_hidden_states = hidden_states[batch_indices, eos_indices.squeeze(1), :]  # (batch_size, hidden_size)
    else:
        # Use last position
        final_hidden_states = hidden_states[:, -1, :]  # (batch_size, hidden_size)

    return final_hidden_states


class CoinFlipTrainableModule(nn.Module):
    """
    Wraps a transformer backbone + trainable coin flip head into a single nn.Module,
    suitable for DeepSpeed wrapping via strategy.prepare().

    The coin flip head is applied INSIDE forward() so it is unambiguously part of
    the DeepSpeed engine's computation graph, ensuring proper gradient synchronization
    across ranks in multi-GPU training.

    For the separate_nn architecture the frozen prior network is kept separate
    (it goes through strategy.prepare() as a bare arg, using eval-mode DeepSpeed)
    so that it remains independent of the trainable backbone throughout training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        coin_flip_dim: int = 64,
        head_init_std: float = 0.001,
        coin_flip_linear_bias: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.coin_flip_dim = coin_flip_dim

        # Determine hidden_size from backbone config
        hidden_size = None
        if hasattr(backbone, 'config'):
            config = backbone.config
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            elif hasattr(config, 'd_model'):
                hidden_size = config.d_model
            elif hasattr(config, 'n_embd'):
                hidden_size = config.n_embd
        if hidden_size is None and hasattr(backbone, 'lm_head') and hasattr(backbone.lm_head, 'in_features'):
            hidden_size = backbone.lm_head.in_features
        if hidden_size is None:
            raise ValueError(
                "Could not determine hidden_size for CoinFlipTrainableModule. "
                "Ensure the backbone has a 'config' with 'hidden_size', 'd_model', or 'n_embd'."
            )
        self.hidden_size = hidden_size

        # Get device and dtype from backbone parameters
        device = None
        dtype = torch.float32
        for param in backbone.parameters():
            device = param.device
            dtype = param.dtype
            break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Trainable coin flip head: maps last hidden state → coin flip predictions
        self.coin_flip_head = nn.Linear(hidden_size, coin_flip_dim, bias=coin_flip_linear_bias).to(device=device, dtype=dtype)
        nn.init.normal_(self.coin_flip_head.weight, mean=0.0, std=head_init_std)
        if coin_flip_linear_bias and self.coin_flip_head.bias is not None:
            nn.init.zeros_(self.coin_flip_head.bias)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run backbone + apply trainable coin flip head.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)

        Returns:
            Trainable coin flip predictions for final tokens, shape (batch_size, coin_flip_dim)
        """
        position_ids = _compute_position_ids(attention_mask)

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = _extract_hidden_states_from_outputs(outputs)
        final_hidden_states = _extract_final_hidden_states(hidden_states, attention_mask)

        return self.coin_flip_head(final_hidden_states)  # (B, coin_flip_dim)


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
        coin_flip_architecture: Architecture type: "linear_head_on_static_initial_base" (linear head on frozen base copy),
            "linear_head_on_learning_base" (linear head on live base_actor), "linear_head_on_learning_proposal"
            (linear head on live sampling_actor), or "separate_nn" (separate trainable and frozen networks)
            (default: "linear_head_on_static_initial_base")
        warmup_steps: Number of calls to compute_intrinsic_reward before returning non-zero bonuses.
            During warmup, Welford stats are still updated but bonus is returned as 0. (default: 0)
    """
    
    def __init__(
        self, 
        base_model: nn.Module, 
        coin_flip_dim: int = 64, 
        normalization_momentum: Optional[float] = None,
        head_init_std: float = 0.001,
        frozen_prior_init_std: float = 0.1,
        coin_flip_linear_bias: bool = False,
        coin_flip_architecture: str = "linear_head_on_static_initial_base",
        trainable_network: Optional[nn.Module] = None,
        frozen_prior_network: Optional[nn.Module] = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.coin_flip_dim = coin_flip_dim
        self.coin_flip_architecture = coin_flip_architecture
        
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
        
        # Create frozen_prior_model only for learning architectures, where the backbone
        # model changes during training and we need a separate frozen copy for the random
        # prior head. For static architectures, base_model is already frozen and serves
        # the same role. For separate_nn, frozen_prior_network is used instead.
        if coin_flip_architecture in LEARNING_ARCHITECTURES:
            self.frozen_prior_model = copy.deepcopy(unwrapped_model)
            for param in self.frozen_prior_model.parameters():
                param.requires_grad = False
        else:
            self.frozen_prior_model = None
        
        # Determine architecture type and set up models accordingly
        if coin_flip_architecture == "separate_nn":
            # trainable_network: DeepSpeedEngine wrapping CoinFlipTrainableModule (trainable end-to-end;
            #   coin_flip_head is already inside CoinFlipTrainableModule).
            # frozen_prior_network: Actor with eval-mode DeepSpeed engine (completely frozen;
            #   a separate frozen coin_flip_head is added below).
            assert trainable_network is not None and frozen_prior_network is not None, (
                "separate_nn requires trainable_network (DeepSpeedEngine wrapping CoinFlipTrainableModule) "
                "and frozen_prior_network (Actor with eval-mode DeepSpeed) to be provided."
            )
            self.trainable_engine = trainable_network   # DeepSpeedEngine wrapping CoinFlipTrainableModule
            self.frozen_prior_network = frozen_prior_network  # Actor (frozen coin_flip_head added below)
            self.trainable_network = None  # unused after refactor; replaced by trainable_engine

            # Freeze the frozen prior network completely
            for param in self.frozen_prior_network.parameters():
                param.requires_grad = False

            # For separate_nn mode, we don't use base_model or backbone_model
            self.base_model = None
            self.backbone_model = None
            self.use_learning_backbone = False
        elif coin_flip_architecture in LEARNING_ARCHITECTURES:
            # Learning architectures: use live model reference, no copy
            # Store reference to the live model (will be set by caller)
            self.backbone_model = base_model  # Store the original Actor reference
            self.use_learning_backbone = True
            
            # For learning architectures, we don't use base_model (frozen copy)
            self.base_model = None
            
            # Only for separate_nn mode. Here, these will be None as they are not used.
            self.trainable_network = None
            self.frozen_prior_network = None
        elif coin_flip_architecture in STATIC_ARCHITECTURES:
            # Static architecture: create frozen copy (existing behavior)
            # Create a deep copy of the base model to ensure complete separation
            # This ensures the coin flip network is entirely independent from the
            # base/sampling actors and won't be interfered with by their training
            self.base_model = copy.deepcopy(unwrapped_model)
            
            # For static architecture, we don't use backbone_model
            self.backbone_model = None
            self.use_learning_backbone = False
            
            # Only for separate_nn mode. Here, these will be None as they are not used.
            self.trainable_network = None
            self.frozen_prior_network = None
        else:
            raise ValueError(f"Unknown coin_flip_architecture: {coin_flip_architecture}. "
                           f"Must be one of: {['separate_nn'] + LEARNING_ARCHITECTURES + STATIC_ARCHITECTURES}")
        
        # Get hidden size from config or model architecture
        # Helper function to get hidden size from a model
        def get_hidden_size_from_model(model):
            """Extract hidden_size from a model (Actor or raw transformer)."""
            hidden_size = None
            
            # Get the actual transformer model (unwrap Actor if needed)
            if hasattr(model, 'model'):
                actual_model = model.model
            else:
                actual_model = model
            
            # Try to get from config
            if hasattr(actual_model, 'config'):
                config = actual_model.config
                if hasattr(config, 'hidden_size'):
                    hidden_size = config.hidden_size
                elif hasattr(config, 'd_model'):
                    hidden_size = config.d_model
                elif hasattr(config, 'n_embd'):
                    hidden_size = config.n_embd
            
            # If not found, try from model architecture
            if hidden_size is None:
                if hasattr(actual_model, 'lm_head') and hasattr(actual_model.lm_head, 'in_features'):
                    hidden_size = actual_model.lm_head.in_features
                elif hasattr(actual_model, 'transformer'):
                    transformer = actual_model.transformer
                    if hasattr(transformer, 'ln_f') and hasattr(transformer.ln_f, 'normalized_shape'):
                        hidden_size = transformer.ln_f.normalized_shape[0]
                    elif hasattr(transformer, 'h') and len(transformer.h) > 0:
                        last_block = transformer.h[-1]
                        if hasattr(last_block, 'ln_2') and hasattr(last_block.ln_2, 'normalized_shape'):
                            hidden_size = last_block.ln_2.normalized_shape[0]
            
            # If still not found, try test forward pass
            if hidden_size is None:
                try:
                    device = next(actual_model.parameters()).device if list(actual_model.parameters()) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    dummy_input = torch.zeros(1, 1, dtype=torch.long, device=device)
                    with torch.no_grad():
                        outputs = actual_model(dummy_input, output_hidden_states=True, return_dict=True)
                        if "hidden_states" in outputs and len(outputs["hidden_states"]) > 0:
                            hidden_size = outputs["hidden_states"][-1].shape[-1]
                        elif "last_hidden_state" in outputs:
                            hidden_size = outputs["last_hidden_state"].shape[-1]
                except Exception:
                    pass
            
            return hidden_size
        
        # Get hidden size based on architecture
        if coin_flip_architecture == "separate_nn":
            # trainable_engine.module is CoinFlipTrainableModule, which stores hidden_size
            hidden_size = self.trainable_engine.module.hidden_size
        elif coin_flip_architecture in LEARNING_ARCHITECTURES:
            # For learning architectures, get hidden size from backbone model
            hidden_size = get_hidden_size_from_model(self.backbone_model)
        else:
            # Static architecture
            hidden_size = get_hidden_size_from_model(self.base_model)
        
        # Fallback to original base_model if still None
        if hidden_size is None:
            hidden_size = get_hidden_size_from_model(base_model)
        
        if hidden_size is None:
            raise ValueError("Could not determine hidden_size for CoinFlipNetwork. "
                           "Tried config, model architecture, and test forward pass.")
        else:
            print("Determined hidden_size for CoinFlipNetwork:", hidden_size)
        
        # Get device for initializing heads
        if coin_flip_architecture == "separate_nn":
            # Get device from trainable_engine (DeepSpeedEngine)
            base_model_device = None
            for param in self.trainable_engine.parameters():
                base_model_device = param.device
                break
        elif coin_flip_architecture in LEARNING_ARCHITECTURES:
            # Get device from backbone_model
            base_model_device = None
            for param in self.backbone_model.parameters():
                base_model_device = param.device
                break
        else:
            # Get device from base_model
            base_model_device = None
            for param in self.base_model.parameters():
                base_model_device = param.device
                break
        
        # If no parameters found, use cuda if available, else cpu
        if base_model_device is None:
            base_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get feature dtype from backbone so coin flip heads match (avoids bf16 vs float32 mismatch)
        if coin_flip_architecture == "separate_nn":
            head_dtype = _get_param_dtype(self.trainable_engine)
        elif coin_flip_architecture in LEARNING_ARCHITECTURES:
            head_dtype = _get_param_dtype(self.backbone_model)
        else:
            head_dtype = _get_param_dtype(self.base_model)
        
        if coin_flip_architecture == "separate_nn":
            # The trainable coin flip head is already inside CoinFlipTrainableModule
            # (accessible as trainable_engine.module.coin_flip_head). Only add the frozen
            # prior head to frozen_prior_network here.
            self.frozen_prior_network.coin_flip_head = nn.Linear(hidden_size, coin_flip_dim, bias=False)
            self.frozen_prior_network.coin_flip_head = self.frozen_prior_network.coin_flip_head.to(device=base_model_device, dtype=head_dtype)
            nn.init.normal_(self.frozen_prior_network.coin_flip_head.weight, mean=0.0, std=frozen_prior_init_std)
            for param in self.frozen_prior_network.coin_flip_head.parameters():
                param.requires_grad = False

            # CoinFlipNetwork itself does not hold these heads directly for separate_nn
            self.coin_flip_head = None
            self.random_prior_head = None
        else:
            # linear_head_on_* architectures: create heads
            # Create coin flip head: maps hidden_size -> coin_flip_dim
            self.coin_flip_head = nn.Linear(hidden_size, coin_flip_dim, bias=coin_flip_linear_bias)
            self.coin_flip_head = self.coin_flip_head.to(device=base_model_device, dtype=head_dtype)
            
            # Reinitialize coin flip head with custom standard deviation for smaller initial outputs
            nn.init.normal_(self.coin_flip_head.weight, mean=0.0, std=head_init_std)
            if coin_flip_linear_bias and self.coin_flip_head.bias is not None:
                nn.init.zeros_(self.coin_flip_head.bias)
            
            # Create random prior head: frozen linear layer with same architecture
            # This ensures new states have ~1 pseudocount at initialization
            self.random_prior_head = nn.Linear(hidden_size, coin_flip_dim, bias=False)
            self.random_prior_head = self.random_prior_head.to(device=base_model_device, dtype=head_dtype)
            nn.init.normal_(self.random_prior_head.weight, mean=0.0, std=frozen_prior_init_std)

            # Freeze the random prior head - it should never be trained
            for param in self.random_prior_head.parameters():
                param.requires_grad = False
            
            # For static architecture, freeze the base model
            # For learning architectures, backbone_model is not frozen (it's trainable)
            if coin_flip_architecture in STATIC_ARCHITECTURES:
                # Freeze the base model - we only train the coin_flip_head
                # This avoids DeepSpeed ZeRO hook conflicts and keeps training simple
                # The optimizer will automatically exclude frozen parameters (requires_grad=False)
                for param in self.base_model.parameters():
                    param.requires_grad = False
        
        # Support gradient checkpointing if base model does
        if coin_flip_architecture == "separate_nn":
            # trainable_engine.module = CoinFlipTrainableModule; .backbone = HF model
            self.supports_gradient_checkpointing = getattr(
                self.trainable_engine.module.backbone, 'supports_gradient_checkpointing', False
            )
        elif coin_flip_architecture in LEARNING_ARCHITECTURES:
            # For learning architectures, check backbone_model
            if hasattr(self.backbone_model, 'model'):
                self.supports_gradient_checkpointing = getattr(self.backbone_model.model, 'supports_gradient_checkpointing', False)
            else:
                self.supports_gradient_checkpointing = getattr(self.backbone_model, 'supports_gradient_checkpointing', False)
        else:
            # Static architecture
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
        # Uses Chan's parallel Welford algorithm for batched updates and distributed sync
        # Track statistics per dimension: shape (coin_flip_dim,)
        # prior_running_M2 stores the sum of squared deviations (M2), NOT the variance.
        # Variance is computed on the fly as M2 / count when needed for normalization.
        self.register_buffer('prior_running_mean', torch.zeros(coin_flip_dim, device=base_model_device))
        self.register_buffer('prior_running_M2', torch.zeros(coin_flip_dim, device=base_model_device))
        self.register_buffer('prior_num_updates', torch.zeros(1, dtype=torch.long, device=base_model_device))

        # Warmup: return bonus = 0 for the first warmup_steps calls to compute_intrinsic_reward,
        # while still updating Welford stats so they stabilize before bonuses are used.
        #
        # Why returning 0 during warmup is fine:
        # The target distribution for CTL is sigma(x) ∝ p(x) * e^{beta * reward(x) + bonus(x)}.
        # With SIS, the importance weights w_i = sigma(x_i)/q(x_i) are self-normalized:
        # w̃_i = w_i / sum_j w_j. If bonus(x) = c for all x (any constant, whether 0 or 1),
        # the e^c factor cancels in normalization. So a constant bonus has no effect on CTL
        # learning — only relative differences between samples matter. This means:
        # (1) returning bonus = 0 during warmup is equivalent to returning bonus = 1 everywhere,
        # (2) after warmup, when bonuses become non-constant, the transition is seamless because
        #     the pre-warmup constant bonus was already having no effect on the target distribution.
        self.warmup_steps = warmup_steps
        self.register_buffer('warmup_counter', torch.zeros(1, dtype=torch.long, device=base_model_device))

        # In distributed settings, synchronize frozen prior head weights and check that the
        # architecture supports multi-GPU training.
        #
        # For separate_nn: the trainable engine goes through strategy.prepare() which broadcasts
        # weights from rank 0 and all-reduces gradients — fully synced. But the frozen prior head
        # (frozen_prior_network.coin_flip_head) is initialized independently per rank with
        # nn.init.normal_(), so we broadcast it from rank 0 here.
        #
        # For non-separate_nn architectures: neither the trainable coin_flip_head weights nor its
        # gradients are synced across ranks (it uses a plain optimizer with manual backward/step,
        # not DeepSpeed). This means the trainable head diverges across ranks from the start.
        # Until this is fixed (e.g., by all-reducing gradients or wrapping in DDP), we fail noisily
        # rather than silently producing incorrect results.
        self._sync_frozen_prior_weights_or_fail()

    def _sync_frozen_prior_weights_or_fail(self):
        """
        In distributed settings: broadcast frozen prior head weights from rank 0, or raise
        NotImplementedError if the architecture doesn't support multi-GPU training.
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
            return

        if self.coin_flip_architecture == "separate_nn":
            # frozen_prior_network.coin_flip_head is the frozen head — broadcast from rank 0
            for param in self.frozen_prior_network.coin_flip_head.parameters():
                torch.distributed.broadcast(param.data, src=0)
        else:
            raise NotImplementedError(
                f"CoinFlipNetwork with architecture '{self.coin_flip_architecture}' does not support "
                f"distributed/multi-GPU training. The trainable coin_flip_head uses a plain optimizer "
                f"with manual backward/step (no gradient all-reduce across ranks), so weights diverge. "
                f"Use coin_flip_architecture='separate_nn' for multi-GPU, or implement gradient "
                f"synchronization for this architecture."
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Backward compatibility: convert old prior_running_var to prior_running_M2."""
        old_key = prefix + 'prior_running_var'
        new_key = prefix + 'prior_running_M2'
        count_key = prefix + 'prior_num_updates'
        if old_key in state_dict and new_key not in state_dict:
            # Old format stored variance = M2 / count. Convert to M2 = variance * count.
            count = state_dict.get(count_key, torch.zeros(1, dtype=torch.long))
            state_dict[new_key] = state_dict.pop(old_key) * count.item()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _get_device(self):
        """Get the device of the model, prioritizing GPU/cuda."""
        # Try to get device from model parameters
        if self.coin_flip_architecture == "separate_nn":
            for param in self.trainable_engine.parameters():
                return param.device
        elif self.use_learning_backbone:
            for param in self.backbone_model.parameters():
                return param.device
        else:
            for param in self.base_model.parameters():
                return param.device
        
        # Fallback: use cuda if available, else cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _forward_through_model(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        apply_no_grad: bool = False,
    ) -> dict:
        """
        Forward pass through a model to get hidden states.

        Args:
            model: The model to forward through (can be Actor or raw transformer)
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            position_ids: Position IDs, shape (batch_size, seq_len). If None, computed from attention_mask
            apply_no_grad: If True, wrap forward pass in torch.no_grad()

        Returns:
            Model outputs dictionary containing hidden_states or last_hidden_state
        """
        # Compute position_ids if not provided
        if position_ids is None:
            position_ids = _compute_position_ids(attention_mask)

        # Get the actual model (unwrap Actor if needed)
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model

        # Forward pass
        if apply_no_grad:
            with torch.no_grad():
                outputs = actual_model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            outputs = actual_model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        return outputs

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
        Works for all linear_head architectures (static and learning).
        
        For non-separate_nn architectures, applies torch.no_grad() to prevent gradients
        from flowing through the backbone/base model.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            return_outputs: If True, also return base model outputs
            
        Returns:
            Final hidden states, shape (batch_size, hidden_size)
            If return_outputs=True, also returns base model outputs
        """
        assert self.coin_flip_architecture in STATIC_ARCHITECTURES + LEARNING_ARCHITECTURES, \
            f"_get_final_hidden_states only works for linear_head architectures, got {self.coin_flip_architecture}"

        # Determine which model to use
        if self.coin_flip_architecture in LEARNING_ARCHITECTURES:
            model = self.backbone_model
        else:
            model = self.base_model
        
        # For non-separate_nn architectures, always use torch.no_grad() to prevent gradients
        # from flowing through backbone/base model (only coin_flip_head should receive gradients)
        outputs = self._forward_through_model(
            model, input_ids, attention_mask, apply_no_grad=True
        )
        
        # Extract hidden states using helper method
        hidden_states = _extract_hidden_states_from_outputs(outputs)
        
        # Ensure hidden_states and coin_flip_head are on the same device
        coin_flip_head_device = next(self.coin_flip_head.parameters()).device
        if hidden_states.device != coin_flip_head_device:
            hidden_states = hidden_states.to(coin_flip_head_device)
        
        # Extract final token hidden states using helper method
        final_hidden_states = _extract_final_hidden_states(hidden_states, attention_mask)
        
        if return_outputs:
            return final_hidden_states, outputs
        return final_hidden_states
    
    def _get_linear_head_components(
        self,
        final_hidden_states: torch.Tensor,
        random_prior_final_hidden_states: Optional[torch.Tensor] = None,
        update_prior_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all components from linear_head forward pass.

        This helper method centralizes the linear_head forward pass logic to avoid duplication
        between _predict_from_embeddings() and forward().

        For static architectures, both heads use the same final_hidden_states.
        For learning architectures, random_prior_final_hidden_states should be provided separately.

        Args:
            final_hidden_states: Final hidden states for trainable head, shape (batch_size, hidden_size)
            random_prior_final_hidden_states: Final hidden states for random prior head, shape (batch_size, hidden_size).
                If None, uses final_hidden_states (for static architectures).
            update_prior_stats: If True, update Welford running statistics for the random prior
                normalization. Should be True for new states (bonus computation) and False for
                replayed states (training). Defaults to True.

        Returns:
            Tuple of (combined_predictions, coin_flip_predictions, random_prior_final, normalized_random_prior_final)
            All have shape (batch_size, coin_flip_dim)
        """
        # Apply coin flip head (trainable) only to final states
        coin_flip_predictions = self.coin_flip_head(final_hidden_states)  # (batch_size, coin_flip_dim)

        # Apply random prior head (frozen) to final states, ensuring ~1 pseudocount at initialization
        # For static architectures, use same embeddings; for learning, use separate frozen prior embeddings
        if random_prior_final_hidden_states is None:
            random_prior_final_hidden_states = final_hidden_states

        random_prior_final = self.random_prior_head(random_prior_final_hidden_states)  # (batch_size, coin_flip_dim)

        # Normalize random prior outputs dimension-wise to have mean 0, std 1
        # This ensures sqrt((1/d) * ||normalized_prior||^2) has expectation 1
        normalized_random_prior_final = self._normalize_with_welford_per_dim(
            random_prior_final,
            self.prior_running_mean,
            self.prior_running_M2,
            self.prior_num_updates,
            update_stats=update_prior_stats,
        )  # (batch_size, coin_flip_dim)

        # Combine main predictions with normalized random prior predictions
        combined_predictions = coin_flip_predictions + normalized_random_prior_final  # (batch_size, coin_flip_dim)

        return combined_predictions, coin_flip_predictions, random_prior_final, normalized_random_prior_final
    
    def _predict_from_embeddings(
        self,
        final_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined predictions from final hidden states.

        This method takes already-extracted final hidden states and applies the coin flip head,
        random prior head, and normalization to produce combined predictions. This is used
        for training on embeddings from the replay buffer.

        Does NOT update Welford stats (training path — stats should only be updated on new states).

        Args:
            final_hidden_states: Final hidden states, shape (batch_size, hidden_size)

        Returns:
            Combined predictions, shape (batch_size, coin_flip_dim)
        """
        # Use helper method to get all components, return only combined predictions
        # update_prior_stats=False: training path, don't update Welford stats
        combined_predictions, _, _, _ = self._get_linear_head_components(
            final_hidden_states, update_prior_stats=False
        )
        return combined_predictions
    
    def _separate_nn_combined_predictions(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_prior_stats: bool = True,
    ) -> torch.Tensor:
        """
        Compute combined predictions for the separate_nn architecture.

        Calls trainable_engine (DeepSpeedEngine wrapping CoinFlipTrainableModule) for the
        trainable component, and frozen_prior_network for the frozen prior component.
        The frozen prior is always run under torch.no_grad().

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            update_prior_stats: If True, update Welford running statistics for the frozen prior
                normalization. Should be True for new states (bonus computation) and False for
                replayed states (training). Defaults to True.

        Returns:
            Combined predictions, shape (batch_size, coin_flip_dim)
        """
        # Trainable predictions: forward through DeepSpeedEngine wrapping CoinFlipTrainableModule.
        # Gradients flow here during training; caller wraps in torch.no_grad() when not needed.
        trainable_predictions = self.trainable_engine(input_ids, attention_mask)  # (B, coin_flip_dim)

        # Frozen prior forward pass (always run; no caching).
        position_ids = _compute_position_ids(attention_mask)
        frozen_outputs = self._forward_through_model(
            self.frozen_prior_network, input_ids, attention_mask, position_ids, apply_no_grad=True
        )
        frozen_hidden_states = _extract_hidden_states_from_outputs(frozen_outputs)
        frozen_final = _extract_final_hidden_states(frozen_hidden_states, attention_mask)

        # Move to the frozen prior head's device if needed
        coin_flip_head_device = next(self.frozen_prior_network.coin_flip_head.parameters()).device
        if frozen_final.device != coin_flip_head_device:
            frozen_final = frozen_final.to(coin_flip_head_device)

        with torch.no_grad():
            frozen_predictions = self.frozen_prior_network.coin_flip_head(frozen_final)  # (B, coin_flip_dim)

        # Normalize frozen prior outputs using Chan's parallel Welford algorithm.
        # update_prior_stats controls whether running stats are updated:
        # - False during training (replayed states should not shift the normalization)
        # - True during bonus computation (new states should update the normalization)
        normalized_frozen = self._normalize_with_welford_per_dim(
            frozen_predictions,
            self.prior_running_mean,
            self.prior_running_M2,
            self.prior_num_updates,
            update_stats=update_prior_stats,
        )

        return trainable_predictions + normalized_frozen
    
    def _get_learning_backbone_components(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_prior_stats: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all components from learning backbone forward pass.

        This method handles learning architectures where the trainable head uses a live backbone model
        and the random prior head uses the frozen_prior_model.

        Always applies torch.no_grad() to backbone model to prevent gradients from flowing through it
        (only coin_flip_head should receive gradients).

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            update_prior_stats: If True, update Welford running statistics for the random prior
                normalization. Should be True for new states (bonus computation) and False for
                replayed states (training). Defaults to True.

        Returns:
            Tuple of (combined_predictions, coin_flip_predictions, random_prior_final, normalized_random_prior_final)
            All have shape (batch_size, coin_flip_dim)
        """
        # Compute position_ids using helper method
        position_ids = _compute_position_ids(attention_mask)
        
        # Forward through backbone model (for trainable head)
        # Always use torch.no_grad() to prevent gradients from flowing through backbone
        backbone_outputs = self._forward_through_model(
            self.backbone_model, input_ids, attention_mask, position_ids, apply_no_grad=True
        )
        
        # Forward through frozen prior model (for random prior head, always with no_grad)
        frozen_prior_outputs = self._forward_through_model(
            self.frozen_prior_model, input_ids, attention_mask, position_ids, apply_no_grad=True
        )
        
        # Extract hidden states using helper method
        backbone_hidden_states = _extract_hidden_states_from_outputs(backbone_outputs)
        frozen_prior_hidden_states = _extract_hidden_states_from_outputs(frozen_prior_outputs)
        
        # Extract final token hidden states using helper method
        backbone_final = _extract_final_hidden_states(backbone_hidden_states, attention_mask)
        frozen_prior_final = _extract_final_hidden_states(frozen_prior_hidden_states, attention_mask)
        
        # Ensure tensors are on the same device as coin_flip_head
        coin_flip_head_device = next(self.coin_flip_head.parameters()).device
        if backbone_final.device != coin_flip_head_device:
            backbone_final = backbone_final.to(coin_flip_head_device)
        if frozen_prior_final.device != coin_flip_head_device:
            frozen_prior_final = frozen_prior_final.to(coin_flip_head_device)
        
        # Apply coin flip head (trainable) to backbone embeddings
        coin_flip_predictions = self.coin_flip_head(backbone_final)  # (batch_size, coin_flip_dim)
        
        # Apply random prior head (frozen) to frozen prior embeddings
        random_prior_final = self.random_prior_head(frozen_prior_final)  # (batch_size, coin_flip_dim)
        
        # Normalize random prior outputs dimension-wise to have mean 0, std 1
        # This ensures sqrt((1/d) * ||normalized_prior||^2) has expectation 1
        normalized_random_prior_final = self._normalize_with_welford_per_dim(
            random_prior_final,
            self.prior_running_mean,
            self.prior_running_M2,
            self.prior_num_updates,
            update_stats=update_prior_stats,
        )  # (batch_size, coin_flip_dim)

        # Combine main predictions with normalized random prior predictions
        combined_predictions = coin_flip_predictions + normalized_random_prior_final  # (batch_size, coin_flip_dim)

        return combined_predictions, coin_flip_predictions, random_prior_final, normalized_random_prior_final

    def _predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined predictions from inputs (training path).

        Does NOT update Welford stats — stats should only be updated on new states
        during bonus computation (via forward()), not on replayed/training states.

        This method handles all architectures:
        - For "linear_head_on_static_initial_base": extracts embeddings and calls _predict_from_embeddings()
        - For "linear_head_on_learning_base" or "linear_head_on_learning_proposal": uses learning backbone
        - For "separate_nn": calls _separate_nn_combined_predictions() and returns combined predictions

        For non-separate_nn architectures, applies torch.no_grad() to prevent gradients
        from flowing through the backbone/base model.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)

        Returns:
            Combined predictions, shape (batch_size, coin_flip_dim)
        """
        if self.coin_flip_architecture in STATIC_ARCHITECTURES:
            # Get final hidden states and use _predict_from_embeddings
            # _get_final_hidden_states already applies torch.no_grad() internally
            final_hidden_states = self._get_final_hidden_states(input_ids, attention_mask)
            return self._predict_from_embeddings(final_hidden_states)
        elif self.coin_flip_architecture in LEARNING_ARCHITECTURES:
            # Use helper method to get all components, return only combined predictions
            # _get_learning_backbone_components already applies torch.no_grad() internally
            # update_prior_stats=False: training path, don't update Welford stats
            combined_predictions, _, _, _ = self._get_learning_backbone_components(
                input_ids, attention_mask, update_prior_stats=False
            )
            return combined_predictions
        elif self.coin_flip_architecture == "separate_nn":
            # update_prior_stats=False: training path, don't update Welford stats
            return self._separate_nn_combined_predictions(input_ids, attention_mask, update_prior_stats=False)
        else:
            raise ValueError(f"Unknown coin flip architecture: {self.coin_flip_architecture}")

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
        
        For non-separate_nn architectures, applies torch.no_grad() to prevent gradients
        from flowing through the backbone/base model.
        
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            return_output: If True, also return base model outputs
            
        Returns:
            Coin flip predictions for final states, shape (batch_size, coin_flip_dim)
            If return_output=True, also returns base model outputs
        """
        if self.coin_flip_architecture in STATIC_ARCHITECTURES:
            # Get final hidden states from base_model for coin_flip_head
            # Also get outputs if needed to avoid recomputation
            # _get_final_hidden_states already applies torch.no_grad() internally
            if return_output:
                final_hidden_states, outputs = self._get_final_hidden_states(
                    input_ids, attention_mask, return_outputs=True
                )
            else:
                final_hidden_states = self._get_final_hidden_states(
                    input_ids, attention_mask
                )
                outputs = None

            # For static architectures, base_model is frozen, so its embeddings are identical
            # to what frozen_prior_model would produce. Use the same embeddings for both heads
            # to be consistent with _predict_from_embeddings() (used during training).
            combined_predictions, coin_flip_predictions, random_prior_final, normalized_random_prior_final = \
                self._get_linear_head_components(final_hidden_states)
        elif self.coin_flip_architecture in LEARNING_ARCHITECTURES:
            # Learning architectures: use learning backbone components
            # _get_learning_backbone_components already applies torch.no_grad() internally
            combined_predictions, coin_flip_predictions, random_prior_final, normalized_random_prior_final = \
                self._get_learning_backbone_components(input_ids, attention_mask)
            outputs = None
            if return_output:
                # For learning architectures, return_output is not fully supported yet
                # Could be added later if needed
                raise NotImplementedError("return_output is not fully supported yet for learning architectures")
        elif self.coin_flip_architecture == "separate_nn":
            combined_predictions = self._separate_nn_combined_predictions(
                input_ids, attention_mask, update_prior_stats=True
            )
            outputs = None
            if return_output:
                raise NotImplementedError("return_output is not fully supported yet for separate_nn mode")
        else:
            raise ValueError(f"Unknown coin flip architecture: {self.coin_flip_architecture}")

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
        
        For non-separate_nn architectures, applies torch.no_grad() to prevent gradients
        from flowing through the backbone/base model.
        
        Args:
            sequences: Input sequences, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            bonus_alpha: Scaling factor for the intrinsic reward
            
        Returns:
            Intrinsic rewards per sequence, shape (batch_size,)
        """
        # Get coin flip predictions (already for final states only)
        # Use torch.no_grad() since we only need the value, not gradients — the result is
        # .detach()'ed anyway. Without this, the separate_nn architecture would build a full
        # computation graph through the trainable network that is immediately discarded.
        # This also updates Welford stats for the random prior normalization.
        with torch.no_grad():
            final_predictions = self.forward(sequences, attention_mask)  # (B, d)

        # Increment warmup counter (counts number of calls, i.e., batches of sequences seen)
        self.warmup_counter.data += 1

        # During warmup, return 0 bonus. The Welford stats are still being updated above
        # (via forward()), so they stabilize before bonuses are actually used.
        if self.warmup_counter.item() <= self.warmup_steps:
            batch_size = sequences.shape[0]
            print(f"[Coin Flip Warmup] Step {self.warmup_counter.item()}/{self.warmup_steps}, returning bonus = 0")
            return torch.zeros(batch_size, device=sequences.device)

        # Compute ||f_φ(x)||^2 for final token: sum over coin_flip_dim dimension
        norm_squared = (final_predictions ** 2).sum(dim=-1)  # (B,)

        # Compute intrinsic reward: sqrt((1/d) * ||f_φ(x)||^2)
        intrinsic_reward = torch.sqrt(norm_squared / self.coin_flip_dim)

        # Normalize the exploration bonus using running mean and variance (if enabled)
        if self.normalization_momentum is not None:
            intrinsic_reward = self._normalize_bonus(intrinsic_reward)

        intrinsic_reward *= bonus_alpha

        return intrinsic_reward.detach()
    
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
    
    def _merge_batch_stats_across_ranks(
        self,
        batch_mean: torch.Tensor,
        batch_M2: torch.Tensor,
        batch_count: int,
    ) -> tuple:
        """
        Merge batch statistics across distributed ranks using Chan's parallel merge formula.

        Each rank has computed (batch_mean, batch_M2, batch_count) from its local subset of the
        batch. This method all-gathers these stats and merges them into a single global batch
        statistic, as if all samples had been on one rank.

        No-ops (returns inputs unchanged) when not in a distributed setting or world_size <= 1.

        Args:
            batch_mean: Mean of local batch per dimension, shape (num_dims,)
            batch_M2: Sum of squared deviations from batch_mean in local batch, shape (num_dims,)
            batch_count: Number of samples in local batch

        Returns:
            (global_mean, global_M2, global_count) — merged statistics across all ranks
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
            return batch_mean, batch_M2, batch_count

        world_size = torch.distributed.get_world_size()

        # All-gather batch stats from all ranks
        all_means = [torch.zeros_like(batch_mean) for _ in range(world_size)]
        all_M2s = [torch.zeros_like(batch_M2) for _ in range(world_size)]
        # Use a tensor for batch_count so we can all-gather it
        count_tensor = torch.tensor([batch_count], dtype=torch.long, device=batch_mean.device)
        all_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]

        torch.distributed.all_gather(all_means, batch_mean.contiguous())
        torch.distributed.all_gather(all_M2s, batch_M2.contiguous())
        torch.distributed.all_gather(all_counts, count_tensor)

        # Sequential merge using Chan's formula (world_size is typically 2-8, so this is fast)
        # Chan's merge for two datasets A and B:
        #   delta = mean_B - mean_A
        #   count_AB = count_A + count_B
        #   mean_AB = mean_A + delta * count_B / count_AB
        #   M2_AB = M2_A + M2_B + delta^2 * count_A * count_B / count_AB
        merged_mean = all_means[0].clone()
        merged_M2 = all_M2s[0].clone()
        merged_count = all_counts[0].item()

        for i in range(1, world_size):
            other_count = all_counts[i].item()
            if other_count == 0:
                continue
            if merged_count == 0:
                merged_mean = all_means[i].clone()
                merged_M2 = all_M2s[i].clone()
                merged_count = other_count
                continue

            delta = all_means[i] - merged_mean
            new_count = merged_count + other_count
            merged_mean = merged_mean + delta * other_count / new_count
            merged_M2 = merged_M2 + all_M2s[i] + delta ** 2 * merged_count * other_count / new_count
            merged_count = new_count

        return merged_mean, merged_M2, merged_count

    def _normalize_with_welford_per_dim(
        self,
        values: torch.Tensor,
        running_mean: torch.Tensor,
        running_M2: torch.Tensor,
        num_updates: torch.Tensor,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """
        Normalize values per dimension, optionally updating running statistics using Chan's
        parallel Welford algorithm.

        When update_stats=True, incorporates the current batch into the running statistics
        before normalizing (so batch T is normalized using stats from batches 1,...,T).
        In distributed settings, batch stats are merged across ranks before updating running
        stats, ensuring all ranks maintain identical statistics.
        When update_stats=False, normalizes using existing stats without modifying them.

        Stats should only be updated on genuinely new states (during bonus computation),
        not on replayed states (during training), to avoid double-counting.

        Args:
            values: Values to normalize, shape (batch_size, num_dims)
            running_mean: Buffer storing running mean per dimension, shape (num_dims,)
            running_M2: Buffer storing sum of squared deviations (M2) per dimension, shape (num_dims,)
            num_updates: Buffer storing total number of samples seen (will be incremented by
                batch_count if update_stats=True). This counts individual samples, not batches.
            update_stats: If True, update running statistics with the current batch before normalizing.
                If False, only normalize using existing statistics. Defaults to True.

        Returns:
            Normalized values with mean ~0, std ~1 per dimension, shape (batch_size, num_dims)
        """
        if update_stats:
            batch_count = values.shape[0]
            assert batch_count > 0, "Cannot update Welford stats with an empty batch"

            # Phase 1: Compute local batch statistics (vectorized, no running state mutation yet)
            batch_mean = values.mean(dim=0)  # (num_dims,)
            # batch_M2 = sum of squared deviations from batch_mean within this batch
            batch_M2 = ((values - batch_mean.unsqueeze(0)) ** 2).sum(dim=0)  # (num_dims,)

            # Phase 2: In distributed, merge batch stats across ranks into a single global batch
            # stat. All ranks start from identical running stats (from the previous call's merge),
            # and after this step they all have the same global batch stat, so the running stats
            # update below produces identical results on every rank.
            batch_mean, batch_M2, batch_count = self._merge_batch_stats_across_ranks(
                batch_mean, batch_M2, batch_count
            )

            # Phase 3: Merge the (global) batch stat into running stats using Chan's parallel
            # merge formula.
            #
            # Chan's merge for combining existing stats (A) with new batch (B):
            #   delta = mean_B - mean_A
            #   count_AB = count_A + count_B
            #   mean_AB = mean_A + delta * count_B / count_AB
            #   M2_AB = M2_A + M2_B + delta^2 * count_A * count_B / count_AB
            #
            # Derivation of the M2 update (using the parallel axis / bias-variance decomposition):
            #   For data x_1,...,x_m with mean mu, and any constant c:
            #     sum_i (x_i - c)^2 = sum_i (x_i - mu)^2 + m*(mu - c)^2
            #   Apply with all n_AB samples, mu = mean_AB, c = mean_A:
            #     sum_i (x_i - mean_A)^2 = M2_AB + n_AB * (mean_AB - mean_A)^2
            #   The left side splits into group A and group B contributions:
            #     M2_A + M2_B + n_B * (mean_B - mean_A)^2 [by bias-variance on each group]
            #   Wait — more precisely, for group A: sum_{i in A} (x_i - mean_A)^2 = M2_A
            #   For group B: sum_{i in B} (x_i - mean_A)^2 = M2_B + n_B * (mean_B - mean_A)^2
            #   So the left side = M2_A + M2_B + n_B * delta^2
            #   Therefore: M2_AB = M2_A + M2_B + n_B * delta^2 - n_AB * (mean_AB - mean_A)^2
            #   Since mean_AB - mean_A = delta * n_B / n_AB:
            #     n_AB * (delta * n_B / n_AB)^2 = delta^2 * n_B^2 / n_AB
            #   So: M2_AB = M2_A + M2_B + n_B * delta^2 - n_B^2 * delta^2 / n_AB
            #            = M2_A + M2_B + delta^2 * n_B * (1 - n_B / n_AB)
            #            = M2_A + M2_B + delta^2 * n_B * n_A / n_AB
            #            = M2_A + M2_B + delta^2 * n_A * n_B / n_AB
            #
            # When n_B = 1 (single-sample Welford), batch_M2 = 0 and the formula reduces to:
            #   M2_new = M2_old + delta^2 * n / (n+1), matching the original Welford update.
            old_count = num_updates.item()

            if old_count == 0:
                # First batch: initialize directly from batch statistics
                running_mean.data = batch_mean
                running_M2.data = batch_M2
            else:
                delta = batch_mean - running_mean
                new_count = old_count + batch_count
                running_mean.data = running_mean + delta * batch_count / new_count
                running_M2.data = running_M2 + batch_M2 + delta ** 2 * old_count * batch_count / new_count

            num_updates.data += batch_count

        # Normalize all values using the (possibly updated) running statistics.
        # If no stats have been collected yet (num_updates == 0), skip normalization and return
        # raw values. This avoids dividing by ~1e-8 (since M2 is 0), which would scale
        # values by ~1e8 and cause a massive gradient step on the first training iteration
        # (relevant when train_coin_flip_before=True, where _predict is called before forward).
        if num_updates.item() == 0:
            return values
        variance = running_M2 / num_updates.item()  # (num_dims,)
        normalized = (values - running_mean.unsqueeze(0)) / (torch.sqrt(variance.unsqueeze(0)) + 1e-8)
        return normalized
    
    # def _normalize_with_stats_update_per_dim(
    #     self,
    #     values: torch.Tensor,
    #     running_mean: torch.Tensor,
    #     running_var: torch.Tensor,
    #     num_updates: torch.Tensor,
    #     momentum: float
    # ) -> torch.Tensor:
    #     """
    #     Update running statistics and normalize values per dimension in one step.
    #
    #     This normalizes each dimension independently to have mean 0, std 1.
    #     Uses exponential moving average (for non-prior statistics).
    #
    #     Args:
    #         values: Values to normalize, shape (batch_size, num_dims)
    #         running_mean: Buffer storing running mean per dimension, shape (num_dims,)
    #         running_var: Buffer storing running variance per dimension, shape (num_dims,)
    #         num_updates: Buffer storing number of updates
    #         momentum: Momentum for exponential moving average
    #
    #     Returns:
    #         Normalized values with mean ~0, std ~1 per dimension, shape (batch_size, num_dims)
    #     """
    #     # Compute batch statistics per dimension
    #     # Mean and var across first dimension (batch), keeping feature dimensions
    #     batch_mean = values.mean(dim=0)  # (num_dims,)
    #     batch_var = values.var(dim=0, unbiased=False)  # (num_dims,)
    #
    #     # Update running statistics using exponential moving average
    #     # For the first update, initialize with batch statistics
    #     if num_updates.item() == 0:
    #         running_mean.data = batch_mean
    #         running_var.data = batch_var
    #     else:
    #         # Exponential moving average update per dimension
    #         running_mean.data = momentum * running_mean + (1 - momentum) * batch_mean
    #         running_var.data = momentum * running_var + (1 - momentum) * batch_var
    #
    #     num_updates.data += 1
    #
    #     # Normalize per dimension using updated running statistics
    #     # Add small epsilon to avoid division by zero
    #     normalized = (values - running_mean) / (torch.sqrt(running_var) + 1e-8)
    #     return normalized
    
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
    
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing if supported."""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        if self.supports_gradient_checkpointing:
            if self.coin_flip_architecture == "separate_nn":
                # trainable_engine.module = CoinFlipTrainableModule; .backbone = HF model
                self.trainable_engine.module.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            elif self.use_learning_backbone:
                if hasattr(self.backbone_model, 'model'):
                    self.backbone_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                else:
                    self.backbone_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            else:
                self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if self.supports_gradient_checkpointing:
            if self.coin_flip_architecture == "separate_nn":
                self.trainable_engine.module.backbone.gradient_checkpointing_disable()
            elif self.use_learning_backbone:
                if hasattr(self.backbone_model, 'model'):
                    self.backbone_model.model.gradient_checkpointing_disable()
                else:
                    self.backbone_model.gradient_checkpointing_disable()
            else:
                self.base_model.gradient_checkpointing_disable()

