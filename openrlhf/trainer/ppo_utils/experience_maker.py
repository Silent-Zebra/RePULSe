import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple, Union, Set
from torch.profiler import profile, record_function, ProfilerActivity
import random

import ray
import torch
import torch.distributed as dist
import torch.nn as nn

from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, masked_sum, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
from openrlhf.utils.utils import tile_prompts
from openrlhf.models.model import INDICATOR_REWARD_EPS

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


def generate_coin_flip_vectors(batch_size: int, d: int, device: torch.device) -> torch.Tensor:
    """
    Generate random coin flip vectors c ~ {-1, 1}^d for the final state of each sequence.
    
    Args:
        batch_size: Number of sequences
        d: Dimension of coin flip vectors
        device: Device to create tensors on
    
    Returns:
        Tensor of shape (batch_size, d) with values in {-1, 1}
    """
    # Generate random binary values {0, 1} and convert to {-1, 1}
    coin_flips = torch.randint(0, 2, (batch_size, d), device=device, dtype=torch.float32)
    coin_flips = coin_flips * 2 - 1  # Convert {0, 1} -> {-1, 1}
    return coin_flips


class CoinFlipReplayBuffer:
    """
    Replay buffer for coin flip network training.
    Stores final embeddings from the base model (states) and associated coin flip vectors.
    
    This follows a similar pattern to NaiveReplayBuffer but is optimized for storing
    just embeddings and coin flip vectors rather than full Experience objects.
    Stores 1D tensors directly for efficient sampling.
    """
    
    def __init__(self, limit: int = 0, cpu_offload: bool = True):
        """
        Initialize the replay buffer.
        
        Args:
            limit: Maximum number of samples in the buffer. A number <= 0 means unlimited. Defaults to 0.
            cpu_offload: Whether to offload data to CPU to save GPU memory. Defaults to True.
        """
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        # Store 1D tensors directly: each element is (hidden_size,) and (coin_flip_dim,)
        self.embeddings: List[torch.Tensor] = []
        self.coin_flip_vectors: List[torch.Tensor] = []
        self.size = 0
    
    @torch.no_grad()
    def add(self, embeddings: torch.Tensor, coin_flip_vectors: torch.Tensor):
        """
        Add embeddings and coin flip vectors to the buffer.
        
        Args:
            embeddings: Final hidden states, shape (batch_size, hidden_size)
            coin_flip_vectors: Coin flip targets, shape (batch_size, coin_flip_dim)
        """
        batch_size = embeddings.shape[0]
        
        # Move to CPU if cpu_offload is enabled
        if self.cpu_offload:
            embeddings = embeddings.detach().cpu()
            coin_flip_vectors = coin_flip_vectors.detach().cpu()
        else:
            embeddings = embeddings.detach()
            coin_flip_vectors = coin_flip_vectors.detach()
        
        # Convert 2D tensors to list of 1D tensors using unbind, then extend
        self.embeddings.extend(torch.unbind(embeddings, dim=0))
        self.coin_flip_vectors.extend(torch.unbind(coin_flip_vectors, dim=0))
        self.size += batch_size
        
        # If limit is set and we exceed it, remove oldest entries
        if self.limit > 0:
            while self.size > self.limit:
                self.embeddings.pop(0)
                self.coin_flip_vectors.pop(0)
                self.size -= 1
    
    @torch.no_grad()
    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch of embeddings and coin flip vectors uniformly at random.
        
        Args:
            batch_size: Number of samples to return
            device: Device to move tensors to. If None, uses target_device. Defaults to None.
        
        Returns:
            Tuple of (embeddings, coin_flip_vectors), both shape (batch_size, ...)
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty replay buffer")
        
        if device is None:
            device = self.target_device
        
        # Sample random indices (using random.sample for consistency with NaiveReplayBuffer)
        num_samples = min(batch_size, self.size)
        indices = random.sample(range(self.size), num_samples)
        
        # Collect sampled tensors and stack them
        sampled_embeddings = [self.embeddings[i].to(device) for i in indices]
        sampled_coin_flips = [self.coin_flip_vectors[i].to(device) for i in indices]
        
        # Stack into batch tensors
        sampled_embeddings = torch.stack(sampled_embeddings, dim=0)  # (batch_size, hidden_size)
        sampled_coin_flips = torch.stack(sampled_coin_flips, dim=0)  # (batch_size, coin_flip_dim)
        
        return sampled_embeddings, sampled_coin_flips
    
    def clear(self):
        """Clear the replay buffer."""
        self.embeddings = []
        self.coin_flip_vectors = []
        self.size = 0
    
    def __len__(self) -> int:
        """Return the number of samples in the buffer."""
        return self.size


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        if self.advantages is not None:
            self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    pad_len: Optional[int]


class BaseExperienceMaker(ABC):
    """
    Base experience maker that only handles initialization.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
        shared_actorcritic=False,
        threshold=-5.,
        reward_cap=4.5,
        target_dist_beta=1.,
        alpha=0.,
        rm_type=None,
        actor_loss_type=None,
        max_new_tokens=None,
        save_negdata=False,
        save_negdata_threshold=-10000,
        neg_data: Optional[Set[str]] = None,
        reward_transform: Optional[str] = None,
        reward_transform_beta: Optional[float] = None,
        bad_word_tokens_ids: Optional[List[int]] = None,
        reward_pretrain: Optional[str] = None,
        exploration_bonus: Optional[str] = None,
        bonus_alpha: float = 1.0,
        coin_flip_network: Optional[nn.Module] = None,
        coin_flip_dim: int = 64,
        coin_flip_optim: Optional[torch.optim.Optimizer] = None,
        coin_flip_scheduler: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.shared_actorcritic = shared_actorcritic
        self.threshold = threshold
        self.reward_cap = reward_cap
        self.target_dist_beta = target_dist_beta
        self.alpha = alpha
        self.rm_type = rm_type
        self.actor_loss_type = actor_loss_type
        self.max_new_tokens = max_new_tokens
        self.reward_transform = reward_transform
        self.reward_transform_beta = reward_transform_beta

        self.perf_stats = {}
        self.advantage_estimator = strategy.args.advantage_estimator
        self.ring_rank0_group = None

        assert actor_loss_type is not None

        if self.actor_loss_type == "ppo" or actor_loss_type == "reinforce":
            self.multiply_by_beta = False
        else:
            self.multiply_by_beta = True

        self.save_negdata = save_negdata
        self.save_negdata_threshold = save_negdata_threshold
        if self.save_negdata:
            assert neg_data is not None
        self.neg_data = neg_data
        self.bad_word_tokens_ids = bad_word_tokens_ids
        self.reward_pretrain = reward_pretrain
        self.exploration_bonus = exploration_bonus
        self.bonus_alpha = bonus_alpha
        self.coin_flip_network = coin_flip_network
        self.coin_flip_dim = coin_flip_dim
        self.coin_flip_optim = coin_flip_optim
        self.coin_flip_scheduler = coin_flip_scheduler
        
        # Initialize coin flip replay buffer if using coin_flip exploration bonus
        # Follows same pattern as NaiveReplayBuffer: limit=0 means unlimited, cpu_offload=True saves GPU memory
        if self.exploration_bonus == "coin_flip":
            buffer_limit = getattr(strategy.args, 'coin_flip_replay_buffer_limit', 0) if strategy else 0
            self.coin_flip_replay_buffer = CoinFlipReplayBuffer(limit=buffer_limit, cpu_offload=True)
        else:
            self.coin_flip_replay_buffer = None
        
        # Initialize state visitation count tensor for t=0 tokens (vocab size = 50257)
        # Only for exact_count type
        if self.exploration_bonus == "exact_count":
            # Start with 0 for all tokens (will be incremented to 1 on first visit)
            self.state_visitation_counts = torch.zeros(50257, dtype=torch.long)
        else:
            self.state_visitation_counts = None

    def _calculate_exploration_bonus(
        self, 
        sequences: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        track_both_positions: bool = False
    ) -> torch.Tensor:
        """
        Calculate exploration bonus based on exploration_bonus type.
        
        Args:
            sequences: Tensor of shape (B, S) containing sequences
            attention_mask: Attention mask, shape (B, S)
            track_both_positions: If True, track both t=0 and t=1 tokens and average bonuses.
                                 If False, only track t=0 tokens. (Only used for exact_count)
        
        Returns:
            Tensor of shape (B,) containing exploration bonuses for each sequence
        """
        if self.exploration_bonus is None:
            device = sequences.device
            batch_size = sequences.shape[0]
            return torch.zeros(batch_size, device=device, dtype=torch.float32)
        
        if self.exploration_bonus == "exact_count":
            return self._calculate_exact_count_bonus(sequences, track_both_positions)
        elif self.exploration_bonus == "coin_flip":
            return self._calculate_coin_flip_bonus(sequences, attention_mask)
        else:
            raise ValueError(f"Unknown exploration_bonus type: {self.exploration_bonus}")
    
    def _calculate_exact_count_bonus(
        self,
        sequences: torch.Tensor,
        track_both_positions: bool = False
    ) -> torch.Tensor:
        """
        Calculate exploration bonus using exact state visitation counts.
        
        Args:
            sequences: Tensor of shape (B, S) containing sequences
            track_both_positions: If True, track both t=0 and t=1 tokens and average bonuses.
        
        Returns:
            Tensor of shape (B,) containing exploration bonuses for each sequence
        """
        assert self.max_new_tokens is not None, "max_new_tokens must be set for exploration_bonus='exact_count'"
        if track_both_positions:
            assert self.max_new_tokens >= 2, "exploration_bonus='exact_count' requires max_new_tokens >= 2 to track both t=0 and t=1"
        else:
            assert self.max_new_tokens >= 1, "exploration_bonus='exact_count' requires max_new_tokens >= 1"
        
        device = sequences.device
        
        # Move state_visitation_counts to device if needed
        if self.state_visitation_counts.device != device:
            self.state_visitation_counts = self.state_visitation_counts.to(device)
        
        # Extract tokens from response (last max_new_tokens tokens)
        response_tokens = sequences[:, -self.max_new_tokens:]  # Shape: (B, max_new_tokens)
        t0_tokens = response_tokens[:, 0]  # Shape: (B,)
        t0_tokens_long = t0_tokens.long()  # Ensure integer type
        
        # Update state visitation counts for t=0 tokens (vectorized)
        updates_t0 = torch.ones_like(t0_tokens_long, dtype=self.state_visitation_counts.dtype)  # (B,)
        self.state_visitation_counts.scatter_add_(0, t0_tokens_long, updates_t0)
        
        # Get counts for each t=0 token (after update)
        counts_t0 = self.state_visitation_counts[t0_tokens_long]  # (B,)
        
        # Calculate bonus for t=0: bonus_alpha * (1/sqrt(N(x)))
        bonus_t0 = self.bonus_alpha * (1.0 / torch.sqrt(counts_t0.float()))  # (B,)
        
        if track_both_positions:
            # Also track t=1 tokens
            t1_tokens = response_tokens[:, 1]  # Shape: (B,)
            t1_tokens_long = t1_tokens.long()  # Ensure integer type
            
            # Update state visitation counts for t=1 tokens (vectorized)
            updates_t1 = torch.ones_like(t1_tokens_long, dtype=self.state_visitation_counts.dtype)  # (B,)
            self.state_visitation_counts.scatter_add_(0, t1_tokens_long, updates_t1)
            
            # Get counts for each t=1 token (after update)
            counts_t1 = self.state_visitation_counts[t1_tokens_long]  # (B,)
            
            # Calculate bonus for t=1: bonus_alpha * (1/sqrt(N(x)))
            bonus_t1 = self.bonus_alpha * (1.0 / torch.sqrt(counts_t1.float()))  # (B,)
            
            # Average the bonuses from t=0 and t=1
            exploration_bonus = (bonus_t0 + bonus_t1) / 2.0  # (B,)
        else:
            exploration_bonus = bonus_t0
        
        return exploration_bonus
    
    def _train_coin_flip_network(self, sequences: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Train the coin flip network on the given sequences.
        
        For each sequence:
        1. Forward pass through base model to get hidden states (done once, reused)
        2. Extract final token hidden states (done once, reused)
        3. Generate random coin flip targets c ~ {-1, 1}^d for final state only
        4. Save embeddings and coin flip vectors to replay buffer
        5. For each update step:
           a. Sample a batch from replay buffer (uniformly at random)
           b. Forward pass through coin_flip_head to get predictions f_φ(x_final)
           c. Compute MSE loss: L(x, c) = ||f_φ(x_final) - c||^2
           d. Backward pass and optimizer step
        """
        if self.coin_flip_network is None or self.coin_flip_optim is None:
            return
        
        # Get update steps from args
        update_steps = getattr(self.strategy.args, 'coin_flip_update_steps', 1) if self.strategy else 1

        # Keep network in eval mode - only the head is trained, base model is frozen
        # This ensures consistent outputs (no dropout/stochasticity from base model)
        # The head (Linear layer) doesn't have dropout/batch_norm, so eval mode is fine
        self.coin_flip_network.eval()
        
        batch_size = sequences.shape[0]
        coin_flip_dim = self.coin_flip_network.coin_flip_dim
        device = sequences.device

        # Step 1: Forward pass through base model to get hidden states (expensive, done once)
        # Compute position_ids
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None
        
        # Forward through base model (frozen, no gradients needed)
        with torch.no_grad():
            outputs = self.coin_flip_network.base_model(
                sequences,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get hidden states (last hidden state)
        if "hidden_states" in outputs:
            hidden_states = outputs["hidden_states"][-1]  # (batch_size, seq_len, hidden_size)
        elif "last_hidden_state" in outputs:
            hidden_states = outputs["last_hidden_state"]  # (batch_size, seq_len, hidden_size)
        else:
            raise ValueError("Model outputs must contain either 'hidden_states' or 'last_hidden_state'")
        
        # Ensure hidden_states and coin_flip_head are on the same device
        coin_flip_head_device = next(self.coin_flip_network.coin_flip_head.parameters()).device
        if hidden_states.device != coin_flip_head_device:
            hidden_states = hidden_states.to(coin_flip_head_device)
        
        # Step 2: Extract final token hidden states (done once, reused for all update steps)
        if attention_mask is not None:
            # Find the last valid position for each sequence (same as reward model does)
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1, keepdim=True)
            # Use advanced indexing to extract final hidden states: (B, hidden_size)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            final_hidden_states = hidden_states[batch_indices, eos_indices.squeeze(1), :]  # (B, hidden_size)
        else:
            # Use last position
            final_hidden_states = hidden_states[:, -1, :]  # (B, hidden_size)
        
        assert self.replay_buffer is not None, "for now, replay_buffer must be provided when exploration_bonus='coin_flip'. Fail noisily for now"
        
        # Step 3: Generate coin flip targets: c ~ {-1, 1}^d for final state only
        coin_flip_targets = generate_coin_flip_vectors(
            batch_size, coin_flip_dim, device
        )  # (B, d)
        
        # Step 4: Save embeddings and coin flip vectors to replay buffer
        if self.coin_flip_replay_buffer is not None:
            self.coin_flip_replay_buffer.add(final_hidden_states, coin_flip_targets)
        
        # Step 5: Get replay buffer batch size (defaults to train_batch_size if None)
        if self.coin_flip_replay_buffer is not None and self.coin_flip_replay_buffer.size > 0:
            # Use replay buffer for training
            replay_buffer_batch_size = getattr(self.strategy.args, 'coin_flip_replay_buffer_batch_size', None) if self.strategy else None
            if replay_buffer_batch_size is None:
                # Default to train_batch_size
                replay_buffer_batch_size = getattr(self.strategy.args, 'train_batch_size', batch_size) if self.strategy else batch_size
        else:
            # No replay buffer yet, use current batch
            replay_buffer_batch_size = batch_size
        
        # Check if model is wrapped by DeepSpeed
        try:
            import deepspeed
            is_deepspeed_wrapped = isinstance(self.coin_flip_network, deepspeed.DeepSpeedEngine)
        except (ImportError, AttributeError):
            is_deepspeed_wrapped = False
        
        # Step 6: Loop over update steps
        for update_step in range(update_steps):
            # Sample from replay buffer if available, otherwise use current batch
            if self.coin_flip_replay_buffer is not None and self.coin_flip_replay_buffer.size > 0:
                # Sample from replay buffer
                sampled_embeddings, sampled_coin_flips = self.coin_flip_replay_buffer.sample(
                    replay_buffer_batch_size, coin_flip_head_device
                )
            else:
                # Use current batch (first few iterations before buffer has data)
                sampled_embeddings = final_hidden_states
                sampled_coin_flips = coin_flip_targets
            
            # Forward pass through coin_flip_head only (cheap, recomputed each step)
            final_predictions = self.coin_flip_network.coin_flip_head(sampled_embeddings)  # (B, d)
            
            # Compute MSE loss: L(x, c) = ||f_φ(x_final) - c||^2
            # Average over coin_flip_dim and batch
            loss = ((final_predictions - sampled_coin_flips) ** 2).mean()

            print_info = False # True
            if print_info:
                # Compute differences for inspection
                differences = final_predictions - sampled_coin_flips  # (B, d)
                abs_differences = differences.abs()  # (B, d)
                abs_differences_flat = abs_differences.flatten()  # (B*d,)
                max_diff, max_diff_flat_idx = abs_differences_flat.max(dim=0)  # scalar, flat index tensor
                max_diff_flat_idx = max_diff_flat_idx.item()  # convert to Python int
                max_diff_batch_idx = max_diff_flat_idx // coin_flip_dim
                max_diff_dim_idx = max_diff_flat_idx % coin_flip_dim
                
                # Compute bonus statistics (same computation as compute_intrinsic_reward)
                with torch.no_grad():
                    # Compute ||f_φ(x)||^2 for final token: sum over coin_flip_dim dimension
                    norm_squared = (final_predictions ** 2).sum(dim=-1)  # (B,)
                    
                    # Compute intrinsic reward: sqrt((1/d) * ||f_φ(x)||^2)
                    intrinsic_reward = torch.sqrt(norm_squared / coin_flip_dim)  # (B,)
                    
                    # Apply normalization if enabled (same as in compute_intrinsic_reward)
                    if self.coin_flip_network.normalization_momentum is not None:
                        intrinsic_reward = self.coin_flip_network._normalize_bonus(intrinsic_reward)
                    
                    # Multiply by bonus_alpha
                    bonus = intrinsic_reward * self.bonus_alpha  # (B,)
                    
                    # Compute statistics
                    mean_bonus = bonus.mean().item()
                    min_bonus = bonus.min().item()
                    max_bonus = bonus.max().item()
                
                # Print inspection information
                if self.strategy and self.strategy.is_rank_0():
                    print(f"\n[Coin Flip Network Update Step {update_step + 1}/{update_steps}]")
                    print(f"  Replay buffer size: {self.coin_flip_replay_buffer.size if self.coin_flip_replay_buffer is not None else 0}")
                    print(f"  Target (first sample, first 10 dims): {sampled_coin_flips[0, :10].cpu().tolist()}")
                    print(f"  Prediction (first sample, first 10 dims): {final_predictions[0, :10].detach().cpu().tolist()}")
                    print(f"  Loss: {loss.item():.6f}")
                    print(f"  Max absolute difference: {max_diff.item():.6f}")
                    print(f"  Max diff location: batch_idx={max_diff_batch_idx}, dim_idx={max_diff_dim_idx}")
                    print(f"  Max diff target value: {sampled_coin_flips[max_diff_batch_idx, max_diff_dim_idx].item():.6f}")
                    print(f"  Max diff prediction value: {final_predictions[max_diff_batch_idx, max_diff_dim_idx].detach().item():.6f}")
                    print(f"  Mean bonus: {mean_bonus:.6f}")
                    print(f"  Min bonus: {min_bonus:.6f}")
                    print(f"  Max bonus: {max_bonus:.6f}")
            
            # Backward pass and optimizer step
            # Note: Network stays in eval mode - only the head is trained, base model is frozen
            if self.strategy:
                self.strategy.backward(loss, self.coin_flip_network, self.coin_flip_optim)
                
                if is_deepspeed_wrapped:
                    # DeepSpeed handles optimizer step internally
                    self.strategy.optimizer_step(
                        self.coin_flip_optim,
                        self.coin_flip_network,
                        self.coin_flip_scheduler,
                        name="coin_flip_network"
                    )
                else:
                    # Not wrapped by DeepSpeed - manually step the optimizer
                    self.coin_flip_optim.step()
                    if self.coin_flip_scheduler is not None:
                        self.coin_flip_scheduler.step()
                    self.coin_flip_optim.zero_grad()
            else:
                # Fallback if no strategy (shouldn't happen in practice)
                loss.backward()
                self.coin_flip_optim.step()
                if self.coin_flip_scheduler is not None:
                    self.coin_flip_scheduler.step()
                self.coin_flip_optim.zero_grad()
    
    def _calculate_coin_flip_bonus(
        self,
        sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate exploration bonus using coin flip network.
        
        Args:
            sequences: Tensor of shape (B, S) containing full sequences (prompt + response)
            attention_mask: Attention mask, shape (B, S)
        
        Returns:
            Tensor of shape (B,) containing exploration bonuses for each sequence
        """
        if self.coin_flip_network is None:
            raise ValueError("coin_flip_network must be provided when exploration_bonus='coin_flip'")
        
        # Set network to eval mode for consistent reward computation
        # This ensures dropout and batch norm behave consistently
        self.coin_flip_network.eval()
        
        # Compute intrinsic reward using coin flip network
        # The network expects full sequences and computes r_I(x) = sqrt((1/d) * ||f_φ(x)||^2)
        intrinsic_reward = self.coin_flip_network.compute_intrinsic_reward(
            sequences,
            attention_mask,
            bonus_alpha=self.bonus_alpha,
        )
        
        return intrinsic_reward

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}



    @torch.no_grad()
    def make_experience(
        self, 
        prompts: Union[str, List[str]], 
        samples_per_prompt: int = 1,
        sequences: Optional[torch.Tensor] = None,
        action_log_probs: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_actions: Optional[int] = None,
        value: Optional[torch.Tensor] = None,
        **generate_kwargs
    ) -> Experience:
        print(f"Current target_dist_beta: {self.target_dist_beta}")
        
        # If sequences are provided, use them; otherwise generate
        if sequences is not None:
            # Use pre-generated sequences and related data
            if action_log_probs is None or action_mask is None or attention_mask is None or num_actions is None:
                raise ValueError("If sequences are provided, action_log_probs, action_mask, attention_mask, and num_actions must also be provided")
            
            # For shared_actorcritic case, value should be provided if critic exists
            if self.shared_actorcritic:
                if value is None and self.critic is not None:
                    raise ValueError("For shared_actorcritic, value must be provided when sequences are pre-generated")
            else:
                # For non-shared case, compute value if critic exists and value not provided
                if self.critic is not None and value is None:
                    value = self.critic(sequences, action_mask, attention_mask)
                elif self.critic is None:
                    value = None
        else:
            # Generate sequences as before (backward compatibility)
            expanded_prompts = tile_prompts(prompts, samples_per_prompt)
            action_log_probs, action_mask, attention_mask, num_actions, sequences, value = self.generate_seqs_and_get_all_data(
                expanded_prompts, **generate_kwargs)

        # init log probs
        with torch.no_grad():
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        r, untransformed_reward = self.compute_reward_no_kl(sequences, attention_mask, multiply_by_beta=self.multiply_by_beta)

        rewards, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )

        if value is None:
            advantages = torch.zeros_like(rewards)
            returns = action_mask * rewards
            value = torch.zeros_like(rewards)
        else:
            advantages, returns = self.get_advantages_and_returns(
                value,
                rewards,
                action_mask,
                generate_kwargs["gamma"],
                generate_kwargs["lambd"],
            )

        log_q = (action_log_probs.float() * action_mask).sum(dim=-1)
        log_phi = r
        if not self.multiply_by_beta: # If didn't already multiply by beta, need to do it for log_phi
            log_phi = r * self.target_dist_beta # Otherwise log phi will be wrong. Log phi needs to have beta in it. In the PPO formulation, yes, the reward should not be modified, KL should be (although you could also keep KL penalty coef at 1 and just do beta times reward), but for the calculation of the target/potential you need this beta. Assumes of course that we have potential of the form of e^{beta r} which in log space is beta r
            # Avoid *=, otherwise that would modify r as well
        log_p = (base_action_log_probs.float() * action_mask).sum(dim=-1)
        log_tilde_sigma = log_p + log_phi
        f_q = log_tilde_sigma - log_q


        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": rewards.sum(dim=-1),
            "return2": returns.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
            "f_q": f_q,
            "entropy": -log_q,
            "untransformed_reward": untransformed_reward,
            "untransformed_ret": untransformed_reward - self.kl_ctl.value * masked_sum(kl, action_mask, dim=-1) # includes KL but uses untransformed reward
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            returns,
            advantages,
            attention_mask,
            action_mask,
            info,
            kl
        )



    def compute_reward_no_kl(
        self, sequences, attention_mask, class_num=0, multiply_by_beta=False,
    ):
        # rewards
        if self.reward_pretrain == "indicator_bad_token":
            # Hard-coded reward function: -1 if output contains any bad token, otherwise 0
            # For this toy experiment, we only generate 2 tokens, so we check the last 2 positions
            assert self.bad_word_tokens_ids is not None, "bad_word_tokens_ids must be provided for indicator_bad_token reward_pretrain"
            assert self.max_new_tokens == 2, "indicator_bad_token reward_pretrain currently only supports generate_max_len == 2"
            
            batch_size = sequences.shape[0]
            device = sequences.device
            
            # Convert bad_word_tokens_ids to tensor
            bad_tokens = torch.tensor(self.bad_word_tokens_ids, device=device, dtype=torch.long)
            
            # Extract the last 2 tokens (positions -2 and -1) which are the generated response tokens
            response_tokens = sequences[:, -2:]  # Shape: (B, 2)
            
            # Check if any response token is in bad_word_tokens_ids
            # For each sequence, check if any token in response_tokens matches any bad token
            # Shape: (B, 2) -> check each position
            assert response_tokens.shape == (batch_size, 2), "response_tokens should have shape (B, 2)"
            assert len(bad_tokens.shape) == 1, "bad_tokens should have shape (n,)"
            bad_token_mask = (response_tokens[:, :, None] == bad_tokens[None, None, :]).any(dim=-1)  
            # Check if any position in the response has a bad token
            has_bad_token = bad_token_mask.any(dim=-1)  # (B,)
            
            # Reward: -1 if has bad token, 0 otherwise
            r = torch.where(has_bad_token, torch.tensor(-1.0, device=device), torch.tensor(0.0, device=device))
           
            print("Num satisfying indicator function for bad tokens:")
            print(-r.sum().item())
            
            # Count occurrences of each bad token in responses (vectorized)
            # Flatten response_tokens and use broadcasting to count all bad tokens at once
            flat_response = response_tokens.flatten()  # (B*2,)
            counts = (flat_response[:, None] == bad_tokens[None, :]).sum(dim=0)  # (n,)
            bad_token_counts = dict(zip(self.bad_word_tokens_ids, counts.cpu().tolist()))
            print("Bad token counts in responses:", bad_token_counts)
            
            # Count occurrences of each bad token at t=0 only (vectorized)
            t0_tokens = response_tokens[:, 0]  # (B,)
            t0_counts = (t0_tokens[:, None] == bad_tokens[None, :]).sum(dim=0)  # (n,)
            t0_bad_token_counts = dict(zip(self.bad_word_tokens_ids, t0_counts.cpu().tolist()))
            print("Bad token counts at t=0:", t0_bad_token_counts)
            
            # Apply exploration bonus if enabled
            if self.exploration_bonus:
                raise NotImplementedError("Check that exploration bonus is applied correctly for p vs q")
                exploration_bonuses = self._calculate_exploration_bonus(sequences, attention_mask, track_both_positions=True)
                # Add exploration bonus to base reward
                r = r + exploration_bonuses
                
                print(f"Exploration bonus applied. Mean bonus: {exploration_bonuses.mean().item():.4f}, "
                      f"Min bonus: {exploration_bonuses.min().item():.4f}, "
                      f"Max bonus: {exploration_bonuses.max().item():.4f}")

        elif self.remote_rm_url is not None:
            # TODO not yet supported/checked with custom_single_prompt

            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(),
                                                  skip_special_tokens=False)
            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(
                device=attention_mask.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)
        
        untransformed_reward = r


        if self.save_negdata:

            print("savenegdata")
            print(len(self.neg_data))
            print(sequences.shape)
            # print(sequences)
            # print(r < self.save_negdata_threshold)
            # print(sequences[r < self.save_negdata_threshold])
            seqs_below_threshold = sequences[r < self.save_negdata_threshold]
            print(seqs_below_threshold.shape)

            queries = self.tokenizer.batch_decode(seqs_below_threshold.cpu(),
                                                  skip_special_tokens=False)
            self.neg_data.update(queries) # keep only unique samples
            print(len(self.neg_data))
        if self.reward_transform == "minus_alpha_exp_beta_r":
            print("REWARD TRANSFORM INSPECTION")
            print(self.alpha)
            print(self.reward_transform_beta)
            transformed_reward = r - self.alpha * torch.exp(self.reward_transform_beta * r)
            print(r)
            print(transformed_reward)
            r = transformed_reward
        elif self.reward_transform == "minus_alpha_ind":
            print("REWARD TRANSFORM INSPECTION")
            print(self.alpha)
            print(self.threshold)
            transformed_reward = r - self.alpha * (r < self.threshold)
            print(r)
            print(transformed_reward)
            r = transformed_reward

        else:
            assert self.reward_transform is None  # Others not yet implemented

        # Initialize exploration_bonus (will be set for specific rm_types)
        exploration_bonus = None

        if self.rm_type == "exp_beta_toxicity_class_logprob":
            if self.exploration_bonus:
                raise NotImplementedError("exploration_bonus not yet implemented for rm_type='exp_beta_toxicity_class_logprob'")
            print("WARNING: only set up for toxicity so far") # TODO later make more flexible, for different reward models. Also, be careful that this is the only place the reward model is used
            score = r
            nontoxic_class_logprob = torch.nn.functional.logsigmoid(score)

            if class_num == 1:
                log_prob_of_class = nontoxic_class_logprob
            else:
                assert class_num == 0
                # toxic_class_prob = 1 - torch.exp(nontoxic_class_logprob)
                # log_prob_of_class = torch.log(toxic_class_prob)

                # NUMERICAL STABILITY
                # Note that we want to calculate: log prob of class 0
                # but we have a logit for class 1, call it c
                # So naively, we would take log(1 - sigmoid(c))
                # This is equivalent to log(1 - 1/(1+e^-c))
                # = log(e^-c/(1+e^-c))
                # = -c - log(1+e^-c)
                # Softplus function is log(1 + e^x)
                # So -score - softplus(-score) is one way of doing this
                # Alternatively, you can write out:
                # log(1 - sigmoid(c)) = log(1 - e^c/(1+e^c))
                # = log(1/(1+e^c))
                # = - log(1 + e^c)
                # = - softplus(score)

                log_prob_of_class = -torch.nn.functional.softplus(score)
                # print("INSPECTING REWARDS: log_prob_of_class (softplus)")
                # print(log_prob_of_class)

            final_reward = log_prob_of_class
            # Because remember r_u = 1/beta log phi is the right way to set up the unregularized reward for equivalence between standard RL formulation and our setup
            # BUT remember that phi = p(class | s)^\beta right? So log phi is beta * p(class | s). But anyway, my experiments just use beta = 1 here...
        elif self.rm_type == "indicator_below_threshold": # works for any arbitrary indicator function on checking if score is less than threshold
            eps = INDICATOR_REWARD_EPS
            score = r
            # print("score")
            # print(score)
            
            # Calculate exploration bonus if enabled
            exploration_bonus = self._calculate_exploration_bonus(sequences, attention_mask, track_both_positions=True)
            
            if self.exploration_bonus:
                raise NotImplementedError("Check sign on bonus")
                print(f"Exploration bonus applied for indicator_below_threshold. Mean bonus: {exploration_bonus.mean().item():.4f}, "
                      f"Min bonus: {exploration_bonus.min().item():.4f}, "
                      f"Max bonus: {exploration_bonus.max().item():.4f}")
            
            # Add exploration bonus to the log argument: log((score < threshold) + eps + bonus)
            final_reward = torch.log((score < self.threshold) + eps + exploration_bonus)            
        elif self.rm_type == "rlhf":
            score = r
            
            # Calculate exploration bonus if enabled (will be added after transformations)
            if self.exploration_bonus:
                exploration_bonus = self._calculate_exploration_bonus(sequences, attention_mask, track_both_positions=False)
                print(f"Exploration bonus calculated for rlhf. Mean bonus: {exploration_bonus.mean().item():.4f}, "
                      f"Min bonus: {exploration_bonus.min().item():.4f}, "
                      f"Max bonus: {exploration_bonus.max().item():.4f}")
            else:
                exploration_bonus = None
            
            capped_reward = torch.minimum(score, self.reward_cap * torch.ones_like(score))

            # print(score)
            # print(capped_reward)
            # print(self.target_dist_beta) # Debug only

            final_reward = capped_reward # Here, 1/beta log phi = 1/beta log e^beta (capped r) = capped r.

        else:
            raise NotImplementedError

        if multiply_by_beta: # Use for twist formulation # For twists: target potential phi is e^{beta r}, so log potential is beta r
            result = final_reward * self.target_dist_beta
            if exploration_bonus is not None:
                result = result + exploration_bonus
            return result, untransformed_reward
        else: # Use for PPO formulation # For PPO, e.g. see the RL with KL penalties is better viewed as Bayesian inference paper, we have that reward - 1/beta (KL to prior) is equivalent to targeting base e^{beta r}
            if self.target_dist_beta < 0:
                result = -final_reward
            else:
                result = final_reward
            if exploration_bonus is not None:
                result = result + exploration_bonus
            return result, untransformed_reward

    def set_all_eval(self):
        self.actor.eval()
        if self.critic is not None:
            self.critic.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

    def set_all_policies_train(self):
        # not currently training reward model
        self.actor.train()
        if self.critic is not None:
            self.critic.train()
        self.initial_model.train()


    def generate_seqs_and_get_logprobs(self, prompts, **generate_kwargs):
        self.set_all_eval()

        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")

        sequences, attention_mask, action_mask = self.actor.generate(**inputs,
                                                                     **generate_kwargs)

        num_actions = action_mask.size(1)

        if self.shared_actorcritic:

            self.set_all_policies_train()

            action_log_probs, values = self.actor(sequences, num_actions, attention_mask)
            return action_log_probs, action_mask, attention_mask, num_actions, sequences, values
        else:
            self.set_all_policies_train()

            action_log_probs = self.actor(sequences, num_actions, attention_mask)

            return action_log_probs, action_mask, attention_mask, num_actions, sequences

    @torch.no_grad()
    def generate_seqs_and_get_all_data(self, prompts, **generate_kwargs):
        """
        Generate sequences and compute all necessary data (logprobs, masks, values).
        This is a convenience wrapper that handles both shared_actorcritic and non-shared cases.
        
        Args:
            prompts: Prompts to generate sequences from
            **generate_kwargs: Additional generation arguments
            
        Returns:
            Tuple of (action_log_probs, action_mask, attention_mask, num_actions, sequences, value)
        """
        if self.shared_actorcritic:
            action_log_probs, action_mask, attention_mask, num_actions, sequences, value = self.generate_seqs_and_get_logprobs(
                prompts, **generate_kwargs)
        else:
            action_log_probs, action_mask, attention_mask, num_actions, sequences = self.generate_seqs_and_get_logprobs(
                prompts, **generate_kwargs)
            if self.critic is not None:
                value = self.critic(sequences, action_mask, attention_mask)
            else:
                value = None
        
        return action_log_probs, action_mask, attention_mask, num_actions, sequences, value

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns




class RemoteExperienceMaker(BaseExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        raise NotImplementedError # Check the latest version of OpenRLHF repo


    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()

        # generate sequence
        start = time.time()
        sequences, attention_mask, action_mask = (
            self._generate_local(prompts, **generate_kwargs)
            if self.vllm_engines is None
            else self._generate_vllm(prompts, **generate_kwargs)
        )
        generate_time = time.time() - start

        num_actions = action_mask.size(1)
        sequences_cpu, attention_mask_cpu, action_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
            action_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        if self.shared_actorcritic:
            raise NotImplementedError # Below stuff not implemented for this yet
        # values
        value_ref = self.critic.forward.remote(sequences_cpu, action_mask_cpu, attention_mask_cpu)

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward:
            ray.get([value_ref])
            ray.get([self.critic.empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))
        else:
            # remote RM
            for rm in self.remote_rm_url:
                queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
                r = remote_rm_fn_ray.remote(rm, queries=queries)
                r_refs.append(r)

        # log probs
        start = time.time()
        action_log_probs = self.actor(sequences, num_actions, attention_mask)
        actor_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        # TODO check whether you want clamping in the below
        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }

        if self.strategy.args.perf:
            batch_size = 1 if isinstance(prompts, str) else len(prompts)
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # send experience to critic
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")
        self._ref = self.critic.append.remote(experience_cpu)

        self.actor.train()  # reset model state
        return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 0), # Use 0 instead of 1; because with 0, if you force first token to not be EOS, this screws up the KL div if the policy places any reasonably high amount of probability mass on the EOS token such that the EOS token would appear in the first token generation. I'd rather have this fail noisily (e.g. if reward model needs non-empty generation) than subtly wrong KL values.
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # TODO: can't pass `max_length` to vLLM's tokenizer for input truncation, remove this once it is supported.
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)
        prompt_token_ids = []
        for i, pad_index in enumerate(pad_indices.numpy()):
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        for output in outputs:
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for output in outputs:
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

            if output_ids[output_len - 1] != eos_token_id:
                raise Exception # This is likely doing the wrong thing, e.g. see https://github.com/OpenRLHF/OpenRLHF/issues/238
                output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        sequences, attention_mask, action_mask = self.actor.process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None

