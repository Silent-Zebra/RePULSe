import math
import os
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union, Set
from openrlhf.models.loss import get_positive_weights_detached, get_normalized_positive_weights_detached

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

import torch.nn.functional as F

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.coin_flip_network import CoinFlipNetwork
from openrlhf.models.loss import REINFORCELoss, NegTrainingLoss, NegREINFORCELoss, CTLLoss, DPGLoss
from openrlhf.models.utils import masked_mean, compute_approx_kl, compute_reward
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.utils import (
    get_info_name_str,
    tile_prompts,
    inspect_rewards_list,
    log_sequence_for_negatives,
    get_custom_prompt_with_chat_template,
    swap_actor,
)

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveReplayBuffer
from openrlhf.trainer.ppo_utils.experience_maker import BaseExperienceMaker, generate_coin_flip_vectors

from openrlhf.models.model import INDICATOR_REWARD_EPS



class CombinedHarmlessnessTrainer(ABC):
    """
        Trainer for Harmlessness training algorithm.

    Args:
        base_actor (Actor): the actor model to undergo harmlessness training
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        static_initial_model (Actor): the very first initial model, frozen, not trained, only used for metrics (and KL penalty to prior, if implemented in future e.g. using PPO on base_actor)
        base_actor_optim (Optimizer): the optimizer to use for base_actor model
        sampling_actor (Actor): the twisted proposal model used for generating approximate target sigma samples (sigma propto base e^{-beta r})
        base_actor_optim (Optimizer): the optimizer to use for sampling_actor model
    """

    def __init__(
        self,
        sampling_target_updated_base: bool,
        strategy,
        base_actor: Actor,
        sampling_actor: Actor,
        base_critic: nn.Module,
        sampling_critic: nn.Module,
        reward_model: nn.Module,
        static_initial_model: Actor,
        ema_model: Actor,
        base_actor_optim: Optimizer,
        sampling_actor_optim: Optimizer,
        base_critic_optim: Optimizer,
        sampling_critic_optim: Optimizer,
        base_actor_scheduler,
        sampling_actor_scheduler,
        base_critic_scheduler,
        sampling_critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        shared_actorcritic: bool = False,
        vf_coef: float = 0.1,
        model_eval: bool = False,
        threshold: float = -5.,
        reward_clamp: Optional[float] = None,
        reward_cap: Optional[float] = None,
        target_dist_beta: float = 1,
        rm_type: str = '',
        bc_coef: float = 0,
        bc_steps: int = -1,
        base_actor_loss_type: str = 'reinforce',
        base_critic_loss_type: str = 'mse',
        sampling_actor_loss_type: str = 'ctl',
        sampling_critic_loss_type: str = 'mse',
        alpha: float = 0.5,
        parameterization: str = '',
        save_negdata=False,
        save_negdata_threshold=-10000,
        neg_data: Optional[Set[str]] = None,
        baseline_type: Optional[str] = None,
        hardcoded_baseline: Optional[float] = None,
        baseline_type_neg: Optional[str] = None,
        hardcoded_baseline_neg: Optional[float] = None,
        reward_transform: Optional[str] = None,
        rew_trans_alpha: Optional[float] = None,
        rew_trans_beta: Optional[float] = None,
        use_base_as_proposal: bool = False,
        separate_reweighting_beta: Optional[float] = None,
        uniform_reweight: bool = False,
        bad_word_tokens_ids: Optional[List[int]] = None,
        train_coin_flip_before: bool = False,
        coin_flip_first_online: bool = False,
        coin_flip_trainable_network: Optional[Actor] = None,
        coin_flip_frozen_prior_network: Optional[Actor] = None,
        q_best_model: Optional[Actor] = None,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()

        self.sampling_target_updated_base = sampling_target_updated_base
        
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn
        self.reward_transform = reward_transform
        self.rew_trans_alpha = rew_trans_alpha
        self.rew_trans_beta = rew_trans_beta

        self.neg_data = neg_data

        self.base_actor = base_actor
        self.base_critic = base_critic
        self.sampling_actor = sampling_actor
        self.sampling_critic = sampling_critic

        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.static_initial_model = static_initial_model
        self.ema_model = ema_model
        self.base_actor_optim = base_actor_optim
        self.base_critic_optim = base_critic_optim
        self.base_actor_scheduler = base_actor_scheduler
        self.base_critic_scheduler = base_critic_scheduler

        self.sampling_actor_optim = sampling_actor_optim
        self.sampling_critic_optim = sampling_critic_optim
        self.sampling_actor_scheduler = sampling_actor_scheduler
        self.sampling_critic_scheduler = sampling_critic_scheduler

        assert parameterization != ""
        self.parameterization = parameterization

        self.rm_type = rm_type
        self.threshold = threshold

        self.use_base_as_proposal = use_base_as_proposal

        self.separate_reweighting_beta = separate_reweighting_beta
        self.uniform_reweight = uniform_reweight
        self.train_coin_flip_before = train_coin_flip_before
        self.coin_flip_first_online = coin_flip_first_online

        # Mixture proposal state
        self.q_best_model = q_best_model
        self.mixture_proposal = q_best_model is not None
        if self.mixture_proposal:
            D = self.args.duplicate_rollout_batch_by
            self.n_current = D // 2
            self.n_best = D - self.n_current
            self.w_current = self.n_current / D
            self.w_best = self.n_best / D
            self.log_w_current = math.log(self.w_current)
            self.log_w_best = math.log(self.w_best)
            self.best_g_q = float('inf')
            self.mixture_optimization = getattr(self.args, 'mixture_optimization', 'mixture')
            self.mixture_psi_use_mix = getattr(self.args, 'mixture_psi_use_mix', False)
            self._q_ind_data = None  # populated per step for q_independent mode
            if sampling_actor_loss_type == "ctl_nosecondterm":
                assert self.mixture_optimization == "mixture", (
                    f"ctl_nosecondterm drops the negative term, so q_half/q_independent modes "
                    f"(which only differ in how they handle the negative term) are wasteful. "
                    f"Use mixture_optimization='mixture' instead, got '{self.mixture_optimization}'."
                )

        self.base_actor_loss_type = base_actor_loss_type
        self.alpha = alpha
        self.hardcoded_baseline = hardcoded_baseline
        self.baseline_type = baseline_type
        self.baseline_type_neg = baseline_type_neg
        self.hardcoded_baseline_neg = hardcoded_baseline_neg

        self.base_actor_loss_fn = self.get_base_actor_loss_fn()

        self.sampling_actor_loss_type = sampling_actor_loss_type
        # if self.sampling_actor_loss_type == "ppo":
        #     self.sampling_actor_loss_fn = PolicyLoss(eps_clip)
        if self.sampling_actor_loss_type == "ctl":
            self.sampling_actor_loss_fn = CTLLoss()
        elif self.sampling_actor_loss_type == "ctl_nosecondterm":
            self.sampling_actor_loss_fn = CTLLoss(no_second_term=True)
        elif self.sampling_actor_loss_type == "dpg":
            self.sampling_actor_loss_fn = DPGLoss()
        else:
            raise NotImplementedError # others not yet tested

        if self.sampling_actor_loss_type == "ppo":
            self.sampling_shuffle_replay_buffer_sample = True
        else:
            self.sampling_shuffle_replay_buffer_sample = False

        self.base_shuffle_replay_buffer_sample = False

        self.base_critic_loss_type = base_critic_loss_type


        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        self.vf_coef = vf_coef
        self.bc_coef = bc_coef

        self.bc_steps = bc_steps

        self.model_eval = model_eval

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            raise NotImplementedError
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        assert not shared_actorcritic # Not yet implemented/tested here
        self.shared_actorcritic = shared_actorcritic

        assert base_critic is None # Not yet implemented/tested

        self.separate_neg_samples = True
        if self.base_actor_loss_type == "reinforce" or self.use_base_as_proposal:
            self.separate_neg_samples = False

        assert not (self.mixture_proposal and not self.separate_neg_samples), (
            "mixture_proposal requires separate_neg_samples=True (mixture sampling code is inside the "
            "separate_neg_samples branch). This is incompatible with base_actor_loss_type='reinforce' "
            "or use_base_as_proposal=True, which set separate_neg_samples=False."
        )

        base_rm_type = "rlhf" # Use this to ensure the standard reward formulation for the base actor
        # Keep this even in case of the indicator_bad_token
        # Because this gives the right base reward. The sampling actor will then use the indicator with specific threshold

        # Check exploration bonus flags
        exploration_bonus_base_actor = getattr(strategy.args, 'exploration_bonus_base_actor', None)
        exploration_bonus_sampling_actor = getattr(strategy.args, 'exploration_bonus_sampling_actor', None)
        
        # Raise NotImplementedError for base_actor exploration bonus
        if exploration_bonus_base_actor is not None:
            raise NotImplementedError(f"exploration_bonus_base_actor='{exploration_bonus_base_actor}' is not yet supported. Only sampling_actor exploration bonus is currently implemented.")
        
        # Initialize coin flip network for sampling_actor if needed
        self.coin_flip_network = None
        self.coin_flip_optim = None
        self.coin_flip_scheduler = None
        
        if exploration_bonus_sampling_actor == "coin_flip":
            coin_flip_dim = getattr(strategy.args, 'coin_flip_dim', 64)
            normalization_momentum = getattr(strategy.args, 'coin_flip_normalization_momentum', None)
            head_init_std = getattr(strategy.args, 'coin_flip_head_init_std', 0.001)
            frozen_prior_init_std = getattr(strategy.args, 'frozen_prior_init_std', 0.1)
            coin_flip_linear_bias = getattr(strategy.args, 'coin_flip_linear_bias', False)
            base_actor_lr = getattr(strategy.args, 'base_actor_learning_rate', None)
            coin_flip_architecture = getattr(strategy.args, 'coin_flip_architecture', 'linear_head_on_static_initial_base')
            warmup_steps = getattr(strategy.args, 'coin_flip_warmup_steps', 0)

            # Determine which model to use as base_model for CoinFlipNetwork initialization
            # For learning architectures, we pass the appropriate model (base_actor or sampling_actor)
            # The network will store it as backbone_model
            from openrlhf.models.coin_flip_network import LEARNING_ARCHITECTURES, STATIC_ARCHITECTURES
            
            if coin_flip_architecture == "linear_head_on_learning_base":
                # Use base_actor as the backbone model
                coin_flip_base_model = base_actor
            elif coin_flip_architecture == "linear_head_on_learning_proposal":
                # Use sampling_actor as the backbone model
                coin_flip_base_model = sampling_actor
            else:
                # For static architecture or separate_nn, use sampling_actor (will create copy for static)
                coin_flip_base_model = sampling_actor
            
            # Initialize coin flip network
            # If pre-initialized networks are provided (for separate_nn mode), use them
            self.coin_flip_network = CoinFlipNetwork(
                coin_flip_base_model,
                coin_flip_dim=coin_flip_dim,
                normalization_momentum=normalization_momentum,
                head_init_std=head_init_std,
                frozen_prior_init_std=frozen_prior_init_std,
                coin_flip_linear_bias=coin_flip_linear_bias,
                coin_flip_architecture=coin_flip_architecture,
                trainable_network=coin_flip_trainable_network,
                frozen_prior_network=coin_flip_frozen_prior_network,
                warmup_steps=warmup_steps,
            )
            
            # Keep network in eval mode always - only the head is trained, base model is frozen
            # This ensures consistent outputs (no dropout/stochasticity from base model)
            self.coin_flip_network.eval()
            
            # Create optimizer and scheduler (defaulting to sampling_actor's)
            coin_flip_lr = getattr(strategy.args, 'coin_flip_lr', None)
            if coin_flip_lr is None:
                # Default to same LR as actor (avoid get_last_lr() which warns before first step)
                coin_flip_lr = getattr(strategy.args, 'actor_learning_rate', 1e-5)
            
            # Use strategy's create_optimizer method (same as sampling_actor)
            # Extract optimizer parameters from args (same as sampling_actor uses)
            adam_betas = getattr(strategy.args, 'adam_betas', (0.9, 0.95))
            l2 = getattr(strategy.args, 'l2', 0.0)
            self.coin_flip_optim = strategy.create_optimizer(
                self.coin_flip_network,
                lr=coin_flip_lr,
                betas=adam_betas,
                weight_decay=l2
            )
            
            # Use same scheduler type as sampling_actor_scheduler
            if sampling_actor_scheduler is not None:
                from transformers.trainer import get_scheduler
                scheduler_type = getattr(strategy.args, 'lr_scheduler', 'constant')
                num_training_steps = getattr(strategy.args, 'num_training_steps', 1000)
                num_warmup_steps = getattr(strategy.args, 'num_warmup_steps', 0)
                self.coin_flip_scheduler = get_scheduler(
                    scheduler_type,
                    optimizer=self.coin_flip_optim,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            else:
                self.coin_flip_scheduler = None

        # Base actor experience maker (for standard reinforce)
        self.base_experience_maker = BaseExperienceMaker(
            base_actor,
            base_critic,
            reward_model,
            static_initial_model, # Not really used right now except for diagnostics; later if using PPO with KL penalty, would need.
            tokenizer,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
            shared_actorcritic,
            threshold,
            reward_clamp,
            reward_cap,
            1, # target_dist_beta 1 here, because this is just going to need regular rewards for REINFORCE
            self.rew_trans_alpha,
            base_rm_type, 
            base_actor_loss_type, # Does not matter, when the target_dist_beta is 1
            self.generate_kwargs['max_new_tokens'],
            save_negdata=save_negdata,
            save_negdata_threshold=save_negdata_threshold,
            neg_data=self.neg_data,
            reward_transform=self.reward_transform,
            reward_transform_beta=self.rew_trans_beta,
            bad_word_tokens_ids=bad_word_tokens_ids,
            reward_pretrain=getattr(strategy.args, 'reward_pretrain', None),
            exploration_bonus=exploration_bonus_base_actor,
            bonus_alpha=getattr(strategy.args, 'bonus_alpha', 1.0),
            coin_flip_use_prioritization=getattr(strategy.args, 'coin_flip_use_prioritization', False)
        )

        self.sampling_experience_maker_neg = None
        # Below is needed for base proposal... cannot just make it None always
        # Sampling actor experience maker (for approximate sigma samples)
        # This one needs SMC (or SIS) sampling from the approx target so we need the target_dist_beta here
        self.sampling_experience_maker_neg = BaseExperienceMaker(
            sampling_actor,
            sampling_critic,
            reward_model,
            base_actor, # use base_actor here as the initial model. But should not matter except for f_q calculation, and for the KL reward, which if I'm not using PPO, would not matter
            tokenizer,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
            shared_actorcritic,
            threshold,
            reward_clamp,
            reward_cap,
            target_dist_beta,
            self.rew_trans_alpha,
            rm_type,
            sampling_actor_loss_type,
            self.generate_kwargs['max_new_tokens'],
            save_negdata=save_negdata,
            save_negdata_threshold=save_negdata_threshold,
            neg_data=self.neg_data,
            # reward_transform=self.reward_transform # Don't use reward transform on the SMC part. Of course this is a choice, you could if you wanted to, but I think let's avoid this for now to keep things simpler.
            bad_word_tokens_ids=bad_word_tokens_ids,
            reward_pretrain=getattr(strategy.args, 'reward_pretrain', None),
            exploration_bonus=exploration_bonus_sampling_actor,
            bonus_alpha=getattr(strategy.args, 'bonus_alpha', 1.0),
            coin_flip_network=self.coin_flip_network,
            coin_flip_dim=getattr(strategy.args, 'coin_flip_dim', 64),
            coin_flip_optim=self.coin_flip_optim,
            coin_flip_scheduler=self.coin_flip_scheduler,
            coin_flip_first_online=self.coin_flip_first_online,
            coin_flip_use_prioritization=getattr(strategy.args, 'coin_flip_use_prioritization', False),
            coin_flip_architecture=getattr(strategy.args, 'coin_flip_architecture', 'linear_head_on_static_initial_base'),
            base_actor=base_actor,
            sampling_actor=sampling_actor
        )

        self.base_replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)
        self.sampling_replay_buffer_neg = None
        if self.separate_neg_samples:
            self.sampling_replay_buffer_neg = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        self.total_steps = 0

    def get_base_actor_loss_fn(self):
        if self.base_actor_loss_type == "reinforce":
            base_actor_loss_fn = REINFORCELoss(baseline_type=self.baseline_type,
                                                    hardcoded_baseline=self.hardcoded_baseline)  # PolicyLoss(eps_clip)
        elif self.base_actor_loss_type == "neg_training":
            base_actor_loss_fn = NegTrainingLoss(alpha=self.alpha, baseline_type=self.baseline_type,
                                                      hardcoded_baseline=self.hardcoded_baseline)
        elif self.base_actor_loss_type == "neg_reinforce":
            base_actor_loss_fn = NegREINFORCELoss(
                alpha=self.alpha, baseline_type=self.baseline_type, hardcoded_baseline=self.hardcoded_baseline,
                baseline_type_neg=self.baseline_type_neg, hardcoded_baseline_neg=self.hardcoded_baseline_neg,
            )
        else:
            raise NotImplementedError

        return base_actor_loss_fn

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
        is_first_fit_step=False,
        rewards_list=None,
        kl_vals_list=None,
        entropy_list=None,
        untrans_ret_list=None,
        rewards_list_sampling=None,
        untrans_ret_list_sampling=None,
        bonus_vals_list_sampling=None,
        mid_fit_callback=None,
    ) -> (List, List, List, List):

        # Extract prompt_text for new_custom_single_prompt case
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes * args.train_batch_size // args.max_epochs // args.rollout_batch_size
        )
        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)
        if update_timesteps != 1:
            raise NotImplementedError(
                f"update_timesteps={update_timesteps} != 1 is not supported "
                f"(rollout_batch_size={args.rollout_batch_size}, world_size={self.strategy.world_size}, "
                f"micro_rollout_batch_size={self.micro_rollout_batch_size}). "
                f"The steps variable in make_experience_and_do_update is not propagated back to the caller, "
                f"so training would be silently skipped when update_timesteps > 1."
            )

        print("UPDATE TIMESTEPS")
        print(update_timesteps)

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps_harmless == -1:
            args.save_steps_harmless = float("inf")  # do not save ckpt
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch

        # if args.num_episodes > 1:
        #     raise NotImplementedError # Later: can create an additional outer loop to allow for more proposal/twist updates per harmlessness update. But 1 is a decent baseline, to keep overhead to a minimum and learn fast...

        # print("INSPECT_HARMLESS")
        # print(num_update_steps_per_episodes)
        # print(args.train_batch_size)
        # print(args.max_epochs)
        # print(args.rollout_batch_size)
        # print(args.train_batch_size // args.max_epochs // args.rollout_batch_size)
        #
        # print(consumed_samples)
        # print(args.rollout_batch_size)
        # print(num_rollouts_per_episodes)


        steps = consumed_samples // args.rollout_batch_size * update_timesteps + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        # print("INSPECT_HARMLESS2")
        # print(steps)
        # print(start_episode)
        # print(consumed_samples)

        if consumed_samples > 0:
            raise NotImplementedError # Should check that this all works correctly after I modified it.

        # Initialize lists if not provided (for backward compatibility)
        if rewards_list is None:
            rewards_list = []
        if kl_vals_list is None:
            kl_vals_list = []  # NOTE: Currently never populated; placeholder for future KL-to-prior tracking
        if entropy_list is None:
            entropy_list = []  # NOTE: Currently never populated; placeholder for future entropy tracking
        if untrans_ret_list is None:
            untrans_ret_list = []
        if rewards_list_sampling is None:
            rewards_list_sampling = []
        if untrans_ret_list_sampling is None:
            untrans_ret_list_sampling = []
        if bonus_vals_list_sampling is None:
            bonus_vals_list_sampling = []  # TODO: Add support for base_actor bonus tracking

        # estimates_list contains all non-f_q_g_q/iwae metrics
        estimates_list = (rewards_list, kl_vals_list, entropy_list, untrans_ret_list, rewards_list_sampling, untrans_ret_list_sampling, bonus_vals_list_sampling)

        custom_prompt = None

        assert start_episode < args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop # Otherwise no updates done; this might be ok depending on setup, but for now this would be unexpected behaviour.

        # total_update_steps should match the total number of times self.total_steps is incremented
        # (once per dataloader batch, across all episodes and fit_steps loops).
        # If this doesn't match, schedule indexing with self.total_steps will go out of bounds.
        total_update_steps = self.prompts_dataloader.__len__() * args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop * args.fit_steps

        # --- Trajectory recording: metadata save and max_ckpt_num validation ---
        if getattr(args, 'save_trajectory_metadata', False):
            if args.save_steps_harmless != float("inf"):
                n_checkpoints_to_save = total_update_steps // int(args.save_steps_harmless)
                if not getattr(args, 'no_save_optim', False):
                    # save_ckpt prunes — max_ckpt_num must be sufficient
                    assert args.max_ckpt_num >= n_checkpoints_to_save, (
                        f"max_ckpt_num={args.max_ckpt_num} < expected checkpoints ({n_checkpoints_to_save}). "
                        f"Old checkpoints will be pruned during trajectory recording. "
                        f"Set --max_ckpt_num >= {n_checkpoints_to_save}, or use --no_save_optim "
                        f"(which doesn't prune)."
                    )

            info_name_str = get_info_name_str(args)
            metadata = {
                "version": 1,
                "custom_prompt": args.custom_prompt if args.new_custom_single_prompt else None,
                "save_steps_harmless": args.save_steps_harmless,
                "target_dist_beta": args.target_dist_beta,
                "reward_clamp": args.reward_clamp,
                "generate_max_len": args.generate_max_len,
                "prompt_max_len": args.prompt_max_len,
                "pretrain": args.pretrain,
                "reward_pretrain": args.reward_pretrain,
                "seed": args.seed,
                "no_save_optim": getattr(args, 'no_save_optim', False),
                "total_update_steps": total_update_steps,
            }
            metadata_path = os.path.join(args.ckpt_path, f"{info_name_str}_trajectory_metadata.pt")
            torch.save(metadata, metadata_path)
            print(f"Saved trajectory metadata to {metadata_path}", flush=True)
            self._trajectory_metadata_path = metadata_path
            self._trajectory_steps_with_rejection_samples = []

        beta_schedule = None
        if args.anneal_target_dist_beta:
            beta_schedule = log_sequence_for_negatives(args.start_target_dist_beta, args.target_dist_beta, total_update_steps)
            print("BETA SCHEDULE:")
            print(beta_schedule)

        alpha_schedule = None
        if args.start_alpha is not None:
            alpha_schedule = log_sequence_for_negatives(args.start_alpha, args.alpha, total_update_steps)
            print("ALPHA SCHEDULE:")
            print(alpha_schedule)

        # Extract prompt_text for new_custom_single_prompt case
        prompt_text = None
        if args.new_custom_single_prompt:
            prompt_text = get_custom_prompt_with_chat_template(
                self.tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), self.strategy
            )

        # --- Trajectory replay setup ---
        if getattr(args, 'load_base_actor_trajectory', None):
            if not args.new_custom_single_prompt:
                raise NotImplementedError(
                    "Trajectory replay is currently only supported for single-prompt mode "
                    "(--new_custom_single_prompt). Multi-prompt support requires: "
                    "(1) saving/replaying prompt order alongside trajectory, "
                    "(2) per-prompt rejection samples, "
                    "(3) per-prompt eval_target_samples_fixed update."
                )

            from openrlhf.utils.utils import discover_trajectory_checkpoints
            self._trajectory_dir = args.load_base_actor_trajectory
            self.trajectory_checkpoints = discover_trajectory_checkpoints(args.load_base_actor_trajectory)
            assert len(self.trajectory_checkpoints) > 0, \
                f"No checkpoints found in trajectory directory: {args.load_base_actor_trajectory}"
            print(f"Found {len(self.trajectory_checkpoints)} trajectory checkpoints: "
                  f"{[tag for _, tag in self.trajectory_checkpoints]}", flush=True)

            # Determine steps between checkpoint loads
            if args.trajectory_steps_per_ckpt is not None:
                self.trajectory_steps_per_ckpt = args.trajectory_steps_per_ckpt
            else:
                steps_list = [s for s, _ in self.trajectory_checkpoints]
                intervals = [steps_list[i+1] - steps_list[i] for i in range(len(steps_list)-1)]
                assert len(intervals) > 0, (
                    "Only one trajectory checkpoint found; cannot infer interval. "
                    "Please specify --trajectory_steps_per_ckpt explicitly."
                )
                assert len(set(intervals)) == 1, (
                    f"Checkpoint intervals are not uniform: {intervals}. "
                    f"Please specify --trajectory_steps_per_ckpt explicitly."
                )
                self.trajectory_steps_per_ckpt = intervals[0]
            print(f"Trajectory steps per checkpoint: {self.trajectory_steps_per_ckpt}", flush=True)

            self.trajectory_ckpt_index = 0

            # Detect checkpoint format (HF vs DeepSpeed)
            first_tag = self.trajectory_checkpoints[0][1]
            first_dir = os.path.join(args.load_base_actor_trajectory, first_tag)
            self.trajectory_is_hf_format = os.path.exists(os.path.join(first_dir, "config.json"))
            print(f"Trajectory checkpoint format: {'HuggingFace' if self.trajectory_is_hf_format else 'DeepSpeed'}", flush=True)

            # Load first checkpoint
            self._load_trajectory_checkpoint(0)

            # Load rejection samples if available
            self.trajectory_rejection_samples = {}
            rejection_dir = args.load_base_actor_trajectory.replace("_harml_actor", "_rejection_samples")
            if os.path.isdir(rejection_dir):
                for f in sorted(os.listdir(rejection_dir)):
                    if f.endswith('.pt'):
                        tag = f.replace('.pt', '')
                        self.trajectory_rejection_samples[tag] = torch.load(
                            os.path.join(rejection_dir, f), map_location='cpu'
                        )
                print(f"Loaded rejection samples for {len(self.trajectory_rejection_samples)} trajectory steps", flush=True)

            # Set initial rejection samples for evaluation
            first_tag = self.trajectory_checkpoints[0][1]
            self.current_trajectory_rejection_samples = self.trajectory_rejection_samples.get(first_tag, None)

        from openrlhf.utils.utils import print_timestamp
        for episode in range(start_episode, args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop): # Actually with this current setup is kind of redundant to have these 2 hyperparameters, loops here or in the outer loop, just pick one, doesn't really matter with 1 update each...
            print_timestamp(f"training - episode {episode}/{args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop}: start")
            print(f"HARMLESSNESS TRAINING EPISODE {episode}", flush=True)
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                if args.anneal_target_dist_beta:
                    assert self.total_steps < len(beta_schedule), (
                        f"Schedule index out of bounds: total_steps={self.total_steps} >= "
                        f"len(beta_schedule)={len(beta_schedule)}. "
                        f"total_update_steps computation may not match actual iteration count."
                    )
                    new_beta = beta_schedule[self.total_steps]
                    self.sampling_experience_maker_neg.target_dist_beta = new_beta
                    print(f"Using new beta: {new_beta}")
                if args.start_alpha is not None:
                    assert self.total_steps < len(alpha_schedule), (
                        f"Schedule index out of bounds: total_steps={self.total_steps} >= "
                        f"len(alpha_schedule)={len(alpha_schedule)}. "
                        f"total_update_steps computation may not match actual iteration count."
                    )
                    new_alpha = alpha_schedule[self.total_steps]
                    self.alpha = new_alpha
                    self.base_actor_loss_fn = self.get_base_actor_loss_fn()
                    print(f"Using new alpha: {new_alpha}")

                if args.new_custom_single_prompt:
                    rand_prompts = [prompt_text]

                # Load next trajectory checkpoint if in replay mode
                if getattr(args, 'load_base_actor_trajectory', None):
                    self._maybe_load_next_trajectory_checkpoint()

                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #              profile_memory=True, record_shapes=True) as prof:

                if args.num_episodes > 1:
                    for q_train_step in range(args.num_episodes - 1):
                        print(f"q train step: {q_train_step}")
                        self.make_experience_and_do_update(args, custom_prompt, pbar, rand_prompts, rewards_list, steps,
                                                           untrans_ret_list, update_timesteps, neg_sample_only=True,
                                                           rewards_list_sampling=rewards_list_sampling, untrans_ret_list_sampling=untrans_ret_list_sampling, bonus_vals_list_sampling=bonus_vals_list_sampling)

                # If base_actor learning rate is 0, only sample from sampling_actor (q)
                # Use flag from args if set, otherwise check learning rate
                neg_sample_only = getattr(args, 'neg_sample_only', False) or abs(getattr(args, 'base_actor_learning_rate', 0)) < 1e-10
                self.make_experience_and_do_update(args, custom_prompt, pbar, rand_prompts, rewards_list, steps,
                                                   untrans_ret_list, update_timesteps, neg_sample_only=neg_sample_only,
                                                   rewards_list_sampling=rewards_list_sampling, untrans_ret_list_sampling=untrans_ret_list_sampling, bonus_vals_list_sampling=bonus_vals_list_sampling)

                if mid_fit_callback is not None:
                    mid_fit_callback(len(rand_prompts))

        # Update trajectory metadata with rejection sample steps (at end of training)
        if getattr(args, 'save_trajectory_metadata', False) and hasattr(self, '_trajectory_metadata_path'):
            metadata = torch.load(self._trajectory_metadata_path, map_location='cpu')
            metadata["steps_with_rejection_samples"] = getattr(self, '_trajectory_steps_with_rejection_samples', [])
            torch.save(metadata, self._trajectory_metadata_path)
            print(f"Updated trajectory metadata with {len(metadata['steps_with_rejection_samples'])} "
                  f"rejection sample steps", flush=True)

        # train_ppo now owns f_q/g_q evaluation: calls f_q_g_q_evaluation at initial and after each fit step
        return estimates_list

    def maybe_update_q_best(self, current_g_q: float):
        """Update q_best model if current q achieves better (lower) g_q."""
        assert self.mixture_proposal, "maybe_update_q_best called but mixture_proposal is False"
        if current_g_q < self.best_g_q:
            old_best = self.best_g_q
            self.best_g_q = current_g_q
            # Unwrap DeepSpeed engines if needed
            q_current_unwrapped = self.sampling_actor.module if hasattr(self.sampling_actor, 'module') else self.sampling_actor
            q_best_unwrapped = self.q_best_model.module if hasattr(self.q_best_model, 'module') else self.q_best_model
            q_best_unwrapped.load_state_dict(q_current_unwrapped.state_dict())
            print(f"[Mixture] Updated q_best: g_q improved from {old_best:.4f} to {current_g_q:.4f}")
        else:
            print(f"[Mixture] No q_best update: current g_q={current_g_q:.4f} >= best g_q={self.best_g_q:.4f}")

    def _pad_generation_outputs(self, sequences, action_log_probs, action_mask, attention_mask,
                                value, num_actions, target_num_actions):
        """Right-pad generation outputs so that num_actions matches target_num_actions.

        When two generation passes (e.g. q_current and q_best) produce different
        sequence lengths due to early EOS stopping, this pads the shorter batch so
        both can be interleaved into a single tensor.
        """
        if num_actions == target_num_actions:
            return sequences, action_log_probs, action_mask, attention_mask, value

        assert num_actions < target_num_actions, (
            f"num_actions ({num_actions}) > target ({target_num_actions})")
        pad_len = target_num_actions - num_actions
        batch_size = sequences.shape[0]
        device = sequences.device

        # Sequences: pad with pad_token_id on right
        seq_pad = torch.full((batch_size, pad_len), self.tokenizer.pad_token_id,
                             dtype=sequences.dtype, device=device)
        sequences = torch.cat([sequences, seq_pad], dim=1)

        # Attention mask: pad with 0 on right
        atmask_pad = torch.zeros((batch_size, pad_len), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, atmask_pad], dim=1)

        # Action mask: pad with 0 on right
        amask_pad = torch.zeros((batch_size, pad_len), dtype=action_mask.dtype, device=device)
        action_mask = torch.cat([action_mask, amask_pad], dim=1)

        # Action log probs: pad with 0 on right (will be recomputed after interleaving anyway)
        alp_pad = torch.zeros((batch_size, pad_len), dtype=action_log_probs.dtype, device=device)
        action_log_probs = torch.cat([action_log_probs, alp_pad], dim=1)

        # Value: pad with 0 if not None
        if value is not None:
            val_pad = torch.zeros((batch_size, pad_len), dtype=value.dtype, device=device)
            value = torch.cat([value, val_pad], dim=1)

        return sequences, action_log_probs, action_mask, attention_mask, value

    @staticmethod
    def _interleave_by_prompt(tensor_a, tensor_b, n_prompts, n_a, n_b):
        """Interleave two tensors by prompt: reshape to (P, n_per, ...), cat on dim=1, flatten back.

        Args:
            tensor_a: (n_prompts * n_a, ...) — e.g. samples from q_current
            tensor_b: (n_prompts * n_b, ...) — e.g. samples from q_best
            n_prompts: number of prompts
            n_a: samples per prompt in tensor_a
            n_b: samples per prompt in tensor_b

        Returns:
            (n_prompts * (n_a + n_b), ...) with samples interleaved per prompt:
            [prompt0_a0..a_{n_a-1}, prompt0_b0..b_{n_b-1}, prompt1_a0..., ...]
        """
        extra_dims = tensor_a.shape[1:]
        a = tensor_a.view(n_prompts, n_a, *extra_dims)
        b = tensor_b.view(n_prompts, n_b, *extra_dims)
        merged = torch.cat([a, b], dim=1)  # (P, n_a + n_b, ...)
        return merged.view(n_prompts * (n_a + n_b), *extra_dims)

    def _compute_mixture_log_probs(self, q_current_alp, q_best_alp, action_mask):
        """Compute sequence-level and partial-sequence-level mixture log probs.

        All inputs are (P, n, A) shaped (already reshaped by prompt).
        Mixture: q_mix(x) = w_current * q_current(x) + w_best * q_best(x)
        In log space: log q_mix = logsumexp(log_w_current + log_q_current, log_w_best + log_q_best)

        Returns:
            seq_log_probs: (P, n) — sequence-level log q_mix
            partial_seq_log_probs: (P, n, A) — partial-sequence-level log q_mix(s_{1:t})
        """
        # Mask log probs (padding -> 0)
        q_curr_masked = q_current_alp * action_mask
        q_best_masked = q_best_alp * action_mask

        # Sequence-level: sum over action dim, then logsumexp over components
        q_curr_seq = q_curr_masked.sum(dim=-1)  # (P, n)
        q_best_seq = q_best_masked.sum(dim=-1)  # (P, n)
        seq_log_probs = torch.logaddexp(
            self.log_w_current + q_curr_seq,
            self.log_w_best + q_best_seq
        )  # (P, n)

        # Partial-sequence-level: cumsum over action dim, then logsumexp
        q_curr_partial = q_curr_masked.cumsum(dim=-1)  # (P, n, A)
        q_best_partial = q_best_masked.cumsum(dim=-1)  # (P, n, A)
        partial_seq_log_probs = torch.logaddexp(
            self.log_w_current + q_curr_partial,
            self.log_w_best + q_best_partial
        )  # (P, n, A)

        return seq_log_probs, partial_seq_log_probs

    def make_experience_and_do_update(self, args, custom_prompt, pbar, rand_prompts, rewards_list, steps,
                                      untrans_ret_list, update_timesteps, neg_sample_only=False,
                                      rewards_list_sampling=None, untrans_ret_list_sampling=None, bonus_vals_list_sampling=None):
        if not neg_sample_only:
            print("Making experience: standard sampling")
            experience = self.base_experience_maker.make_experience(
                rand_prompts,
                samples_per_prompt=args.duplicate_rollout_batch_by,
                **self.generate_kwargs
            )
            self.base_replay_buffer.append(experience)

            # # print prompt/answer in each update step
            # if steps % update_timesteps == 0:
            #     output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)
            #     self.strategy.print(output[0])

        from openrlhf.utils.utils import print_timestamp
        if self.separate_neg_samples:
            print("Making experience: neg sampling")

            if self.mixture_proposal:
                # ---- Mixture proposal: two-pass sampling ----
                D = args.duplicate_rollout_batch_by
                n_prompts = len(rand_prompts) if isinstance(rand_prompts, list) else 1
                print(f"[Mixture] Generating {self.n_current} from q_current + {self.n_best} from q_best per prompt")

                # Pass 1: Generate n_current samples from q_current
                print_timestamp("training - sampling: start generate from q_current")
                expanded_current = tile_prompts(rand_prompts, self.n_current)
                alp_curr, amask_curr, atmask_curr, nact_curr, seq_curr, val_curr = \
                    self.sampling_experience_maker_neg.generate_seqs_and_get_all_data(
                        expanded_current, **self.generate_kwargs)
                print_timestamp("training - sampling: end q_current, start generate from q_best")

                # Pass 2: Generate n_best samples from q_best
                with swap_actor(self.sampling_experience_maker_neg, self.q_best_model):
                    expanded_best = tile_prompts(rand_prompts, self.n_best)
                    alp_best, amask_best, atmask_best, nact_best, seq_best, val_best = \
                        self.sampling_experience_maker_neg.generate_seqs_and_get_all_data(
                            expanded_best, **self.generate_kwargs)
                print_timestamp("training - sampling: end q_best generation")

                # Pad shorter batch to match longer one (different lengths due to early EOS stopping)
                max_nact = max(nact_curr, nact_best)
                if nact_curr != nact_best:
                    print(f"[Mixture] Padding generation outputs: nact_curr={nact_curr}, nact_best={nact_best} -> {max_nact}")
                seq_curr, alp_curr, amask_curr, atmask_curr, val_curr = \
                    self._pad_generation_outputs(seq_curr, alp_curr, amask_curr, atmask_curr, val_curr, nact_curr, max_nact)
                seq_best, alp_best, amask_best, atmask_best, val_best = \
                    self._pad_generation_outputs(seq_best, alp_best, amask_best, atmask_best, val_best, nact_best, max_nact)
                num_actions = max_nact

                # Interleave by prompt: [curr_for_prompt0, best_for_prompt0, curr_for_prompt1, ...]
                sequences = self._interleave_by_prompt(seq_curr, seq_best, n_prompts, self.n_current, self.n_best)
                action_mask = self._interleave_by_prompt(amask_curr, amask_best, n_prompts, self.n_current, self.n_best)
                attention_mask = self._interleave_by_prompt(atmask_curr, atmask_best, n_prompts, self.n_current, self.n_best)
                if val_curr is not None and val_best is not None:
                    value = self._interleave_by_prompt(val_curr, val_best, n_prompts, self.n_current, self.n_best)
                else:
                    value = None

                # Recompute log probs for ALL merged samples from BOTH models
                print_timestamp("training - sampling: start recompute log probs (q_current + q_best)")
                with torch.no_grad():
                    q_current_action_log_probs = self.sampling_experience_maker_neg.actor(
                        sequences, num_actions, attention_mask)
                    q_best_action_log_probs = self.q_best_model(
                        sequences, num_actions, attention_mask)
                print_timestamp("training - sampling: end recompute log probs, start make_experience")

                # Use q_current's log probs as action_log_probs (for gradient tracking in make_experience)
                action_log_probs = q_current_action_log_probs

                # make_experience with q_current's log probs
                experience_neg_sampling = self.sampling_experience_maker_neg.make_experience(
                    rand_prompts,
                    samples_per_prompt=D,
                    sequences=sequences,
                    action_log_probs=action_log_probs,
                    action_mask=action_mask,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    value=value,
                    **self.generate_kwargs
                )

                print_timestamp("training - sampling: end make_experience (mixture path)")
                # NOTE: We do NOT store q_best_action_log_probs in experience.info because the
                # replay buffer's split_experience_batch only supports scalar info values.
                # Instead, q_best log probs are recomputed in get_sampling_actor_loss via
                # a forward pass through self.q_best_model (frozen, no_grad).

                # For "q_independent" mode: generate D additional samples from q_current
                # and store on self (not in experience.info) for the same replay buffer reason.
                if self.mixture_optimization == "q_independent":
                    print(f"[Mixture q_independent] Generating {D} additional samples from q_current")
                    expanded_ind = tile_prompts(rand_prompts, D)
                    alp_ind, amask_ind, atmask_ind, nact_ind, seq_ind, val_ind = \
                        self.sampling_experience_maker_neg.generate_seqs_and_get_all_data(
                            expanded_ind, **self.generate_kwargs)
                    self._q_ind_data = {
                        "sequences": seq_ind,
                        "action_log_probs": alp_ind,
                        "action_mask": amask_ind,
                        "attention_mask": atmask_ind,
                    }

                self.sampling_replay_buffer_neg.append(experience_neg_sampling)

            else:
                # ---- Original (non-mixture) path ----
                # Generate sequences once (with no_grad since generation doesn't need gradients)
                print_timestamp("training - sampling: start generate (non-mixture)")
                expanded_prompts = tile_prompts(rand_prompts, args.duplicate_rollout_batch_by)
                action_log_probs, action_mask, attention_mask, num_actions, sequences, value = self.sampling_experience_maker_neg.generate_seqs_and_get_all_data(
                    expanded_prompts, **self.generate_kwargs)
                print_timestamp("training - sampling: end generate, start make_experience (non-mixture)")

                # Update exact_count visits if enabled (before make_experience)
                if self.sampling_experience_maker_neg.exploration_bonus == "exact_count":
                    track_both = (self.sampling_experience_maker_neg.rm_type == "indicator_below_threshold")
                    self.sampling_experience_maker_neg._update_exact_count_visits(sequences, track_both_positions=track_both)

                if self.train_coin_flip_before:
                    if self.sampling_experience_maker_neg.coin_flip_network is not None and self.sampling_experience_maker_neg.coin_flip_optim is not None:
                        self.sampling_experience_maker_neg._train_coin_flip_network(sequences, attention_mask)
                        torch.cuda.empty_cache()

                # Pass pre-generated sequences to make_experience to avoid duplicate generation
                # Exploration bonus is calculated inside make_experience
                experience_neg_sampling = self.sampling_experience_maker_neg.make_experience(
                    rand_prompts,
                    samples_per_prompt=args.duplicate_rollout_batch_by,
                    sequences=sequences,
                    action_log_probs=action_log_probs,
                    action_mask=action_mask,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    value=value,
                    **self.generate_kwargs
                )

                if not self.train_coin_flip_before:
                    # Train coin flip network AFTER exploration bonus calculation
                    # This ensures pseudocounts are correctly initialized near 1 for new states
                    if self.sampling_experience_maker_neg.coin_flip_network is not None and self.sampling_experience_maker_neg.coin_flip_optim is not None:
                        self.sampling_experience_maker_neg._train_coin_flip_network(sequences, attention_mask)
                        torch.cuda.empty_cache()

                print_timestamp("training - sampling: end make_experience (non-mixture)")
                self.sampling_replay_buffer_neg.append(experience_neg_sampling)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #              profile_memory=True, record_shapes=True) as prof:

        print_timestamp("training - end sampling phase")
        self.total_steps += 1  # do this update before the save_steps, so that saving does happen e.g. if you do 4 save_steps, then on the 4th step, saving will actually happen
        # so far I modified self.save_logs_and_checkpoints, this should be the only place using self.total_steps

        if steps % update_timesteps == 0:
            global_steps = steps // update_timesteps

            torch.cuda.empty_cache()
            if not neg_sample_only:
                self.base_replay_buffer.normalize(self.strategy, "advantages")
            if self.separate_neg_samples:
                self.sampling_replay_buffer_neg.normalize(self.strategy, "advantages")

            assert custom_prompt is None
            print_timestamp("training - start backprop (train())")
            status = self.train(global_steps, custom_prompt=custom_prompt, neg_sample_only=neg_sample_only)
            print_timestamp("training - end backprop (train())")

            if not neg_sample_only:
                self.base_replay_buffer.clear()
            if self.separate_neg_samples:
                self.sampling_replay_buffer_neg.clear()
            torch.cuda.empty_cache()

            if "kl" in status:
                self.kl_ctl.update(status["kl"], args.rollout_batch_size)
            pbar.set_postfix(status)

            # logs/checkpoints
            client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
            self.save_logs_and_checkpoints(args, global_steps, pbar, status, client_states)
        # print("PROFILE2")
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
        pbar.update()
        steps = steps + 1
        if not neg_sample_only:
            rewards_list.append(experience.info["untransformed_reward"].mean().item())
            untrans_ret_list.append(experience.info["untransformed_ret"].mean().item())
            inspect_rewards_list(rewards_list)
            inspect_rewards_list(untrans_ret_list)
        
        if self.separate_neg_samples and experience_neg_sampling is not None and rewards_list_sampling is not None and untrans_ret_list_sampling is not None:
            rewards_list_sampling.append(experience_neg_sampling.info["reward"].mean().item())
            untrans_ret_list_sampling.append(experience_neg_sampling.info["untransformed_reward"].mean().item())
            # Diagnostic: within-prompt reward std (untransformed, i.e. raw RM output)
            samples_per_prompt = args.duplicate_rollout_batch_by
            untrans_rew = experience_neg_sampling.info["untransformed_reward"]
            num_prompts_rew = untrans_rew.shape[0] // samples_per_prompt
            if num_prompts_rew > 1 and samples_per_prompt > 1:
                per_prompt_rew = untrans_rew.view(num_prompts_rew, samples_per_prompt)
                # Per-prompt reward std (distribution over prompts)
                per_prompt_rew_stds = per_prompt_rew.std(dim=1)  # (num_prompts,)
                # Max-to-second-max gap in β·r (the quantity driving softmax concentration)
                # Sort rewards per prompt; with β < 0, lowest reward gets highest β·r
                sorted_rew, _ = per_prompt_rew.sort(dim=1)  # ascending
                # Gap between 2nd-lowest and lowest reward (= gap between max and 2nd-max β·r)
                rew_gaps = sorted_rew[:, 1] - sorted_rew[:, 0]  # (num_prompts,)
                beta_r_gaps = abs(args.target_dist_beta) * rew_gaps  # (num_prompts,)
                gap_over_std_mean = rew_gaps.mean().item() / (per_prompt_rew_stds.mean().item() + 1e-8)
                def _stats(t):
                    return (f"mean={t.mean().item():.4f}, med={t.median().item():.4f}, "
                            f"min={t.min().item():.4f}, max={t.max().item():.4f}")
                print(f"[Reward Diagnostic] within-prompt reward std: {_stats(per_prompt_rew_stds)}")
                print(f"[Reward Diagnostic] max-to-2nd reward gap: {_stats(rew_gaps)} "
                      f"(ratio to std: {gap_over_std_mean:.4f}, normal theory: ~0.67)")
                print(f"[Reward Diagnostic] |beta|*gap: {_stats(beta_r_gaps)}")
            # Extract exploration bonus if available (only for sampling actor for now)
            # TODO: Add support for base_actor bonus tracking
            if bonus_vals_list_sampling is not None and "exploration_bonus" in experience_neg_sampling.info:
                exploration_bonus = experience_neg_sampling.info["exploration_bonus"]
                if exploration_bonus is not None:
                    bonus_vals_list_sampling.append(exploration_bonus.mean().item())
                    # Diagnostic: within-prompt vs. across-prompt bonus variance
                    samples_per_prompt = args.duplicate_rollout_batch_by
                    num_prompts = exploration_bonus.shape[0] // samples_per_prompt
                    if num_prompts > 1 and samples_per_prompt > 1:
                        per_prompt_bonus = exploration_bonus.view(num_prompts, samples_per_prompt)
                        # Per-prompt bonus stats (distributions over prompts)
                        bonus_stds = per_prompt_bonus.std(dim=1)  # (num_prompts,)
                        across_prompt_std = per_prompt_bonus.mean(dim=1).std().item()
                        bonus_ranges = per_prompt_bonus.max(dim=1).values - per_prompt_bonus.min(dim=1).values  # (num_prompts,)
                        bonus_alpha = getattr(args, 'bonus_alpha', 1.0)
                        delta_raws = bonus_ranges / bonus_alpha if bonus_alpha > 0 else bonus_ranges * float('inf')
                        def _stats(t):
                            return (f"mean={t.mean().item():.4f}, med={t.median().item():.4f}, "
                                    f"min={t.min().item():.4f}, max={t.max().item():.4f}")
                        print(f"[Bonus Diagnostic] within-prompt std: {_stats(bonus_stds)}")
                        print(f"[Bonus Diagnostic] across-prompt std: {across_prompt_std:.6f}, "
                              f"ratio (within/across): {bonus_stds.mean().item() / (across_prompt_std + 1e-8):.4f}")
                        print(f"[Bonus Diagnostic] within-prompt range: {_stats(bonus_ranges)}")
                        print(f"[Bonus Diagnostic] delta_raw (range/alpha): {_stats(delta_raws)}")
                else:
                    bonus_vals_list_sampling.append(0.0)  # No bonus when not enabled

    def train(self, global_steps=0, custom_prompt=None, neg_sample_only=False):
        if not neg_sample_only:
            # replay buffer may be empty at first, we should rebuild at each training
            dataloader = DataLoader(
                self.base_replay_buffer,
                batch_size=self.base_replay_buffer.sample_batch_size,
                shuffle=self.base_shuffle_replay_buffer_sample,
                drop_last=True,
                pin_memory=self.dataloader_pin_memory,
                collate_fn=self.base_replay_buffer.collate_fn,
            )
        dataloader_neg = None
        if self.separate_neg_samples:
            dataloader_neg = DataLoader(
                self.sampling_replay_buffer_neg,
                batch_size=self.sampling_replay_buffer_neg.sample_batch_size,
                shuffle=self.sampling_shuffle_replay_buffer_sample,
                drop_last=True,
                pin_memory=self.dataloader_pin_memory,
                collate_fn=self.sampling_replay_buffer_neg.collate_fn,
            )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            if self.separate_neg_samples:
                if neg_sample_only:
                    pbar = tqdm(
                        dataloader_neg,
                        desc=f"Train epoch (neg only) [{epoch + 1}/{self.max_epochs}]",
                        disable=not self.strategy.is_rank_0(),
                    )
                    for experience_neg_sampling in pbar:
                        self.train_on_experiences(custom_prompt, device, None, experience_neg_sampling, global_steps,
                                                  pbar, status_list, neg_sampling_train_only=True)
                else:
                    # do combined train of p and q
                    assert len(dataloader) == len(dataloader_neg)
                    pbar = tqdm(
                        zip(dataloader, dataloader_neg),  # Zip both dataloaders
                        desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                        disable=not self.strategy.is_rank_0(),
                        total=min(len(dataloader), len(dataloader_neg))  # Ensure tqdm gets a proper length
                    )
                    for experience, experience_neg_sampling in pbar:
                        self.train_on_experiences(custom_prompt, device, experience, experience_neg_sampling, global_steps,
                                                  pbar, status_list)
            else:
                # When separate_neg_samples=False (e.g. base_actor_loss_type="reinforce" or
                # use_base_as_proposal=True), p and q samples come from the same dataloader.
                # neg_sample_only=True means we have no base dataloader, so there's nothing to iterate.
                if neg_sample_only:
                    raise ValueError(
                        "neg_sample_only=True is incompatible with separate_neg_samples=False: "
                        "no base dataloader was created, but the non-separate path requires one. "
                        "This can happen with base_actor_loss_type='reinforce' and base_actor_learning_rate=0."
                    )
                pbar = tqdm(
                    dataloader,
                    desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                    disable=not self.strategy.is_rank_0(),
                )
                for experience in pbar:
                    self.train_on_experiences(custom_prompt, device, experience, experience, global_steps,
                                              pbar, status_list)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def train_on_experiences(self, custom_prompt, device, experience, experience_neg_sampling, global_steps, pbar,
                             status_list, neg_sampling_train_only=False):
        experience_neg_sampling.to_device(device)

        if neg_sampling_train_only:
            status = self.training_step_sampling_actor(experience_neg_sampling, custom_prompt=custom_prompt)
        else:
            experience.to_device(device)
            status = self.training_step(experience, experience_neg_sampling, global_steps, custom_prompt=custom_prompt)

        # for DP
        # weighted mean for kl
        for x in ["sampling", "base"]:
            if f"{x}_kl" in status:
                status[f"{x}_kl"] *= status[f"{x}_response_length"]
        status = self.strategy.all_reduce(status)
        for x in ["sampling", "base"]:
            if f"{x}_kl" in status:
                status[f"{x}_kl"] /= status[f"{x}_response_length"]
        if not neg_sampling_train_only:
            short_status = {
                "bpg": status["base_policy_loss"],
                "brm": status["base_reward"],
                "bret": status["base_return"],
                "bglen": status["base_response_length"],
                "btlen": status["base_total_length"],
                "bkl": status["base_kl"],
                "bact_lr": status["base_actor_lr"],
            }

        if "sampling_reward" in status:
            sampling_short_status = {
                "srm": status["sampling_reward"],
                "sret": status["sampling_return"],
                "sglen": status["sampling_response_length"],
                "stlen": status["sampling_total_length"],
                "skl": status["sampling_kl"],
                "sact_lr": status["sampling_actor_lr"],
            }
            if "sampling_f_q" in status:
                sampling_short_status["sf_q"] = status["sampling_f_q"]
            if "sampling_policy_loss" in status:
                sampling_short_status["spg"] = status["sampling_policy_loss"]
            status.update(sampling_short_status)


        if "critic_loss" in status:
            raise NotImplementedError
            # short_status["cri"] = status["critic_loss"]
            # short_status["vals"] = status["values"]
            # if "critic_lr" in status:
            #     short_status["cri_lr"] = status["critic_lr"]
        if "ptx_loss" in status:
            raise NotImplementedError
            # short_status["ptx"] = status["ptx_loss"]
        status_list.append(status)
        if not neg_sampling_train_only:
            pbar.set_postfix(short_status)

    def training_step(self, experience: Experience, experience_neg_sampling: Experience, global_steps, custom_prompt=None) -> Dict[str, float]:
        status = {}
        if self.shared_actorcritic:
            raise NotImplementedError
        else:
            if global_steps > self.freezing_actor_steps:
                if self.sampling_target_updated_base:
                    # Do the base model update first, and do the twist/proposal learning based on sigma which is based on the updated base model
                    status = self.training_step_base_actor(experience, experience_neg_sampling, custom_prompt=custom_prompt)
                    if self.separate_neg_samples:
                        status_sampling = self.training_step_sampling_actor(experience_neg_sampling, custom_prompt=custom_prompt)
                else:
                    # Do the twist/proposal learning based on sigma which is based on the base model before its update
                    if self.separate_neg_samples:
                        status_sampling = self.training_step_sampling_actor(experience_neg_sampling, custom_prompt=custom_prompt)
                    status = self.training_step_base_actor(experience, experience_neg_sampling, custom_prompt=custom_prompt)
                # Note that the updates to the sampling actor don't change the experience_maker_neg_sampling log probs that were already stored there
                # So the only direction of effect is that changing the base model can change the sampling actor target.
                # But in later iterations, the sampling actor should be different, which means the neg train loss should be different...
                if self.separate_neg_samples:
                    status.update(status_sampling)

            if self.base_critic is not None:
                raise NotImplementedError
                status.update(self.training_step_critic(experience, custom_prompt=custom_prompt))

            if self.sampling_critic is not None:
                raise NotImplementedError
                # status.update(self.training_step_sampling_critic(experience, custom_prompt=custom_prompt))

        return status

    def training_step_base_actor(self, experience: Experience, experience_neg_sampling: Experience, custom_prompt=None) -> Dict[str, float]:
        if self.model_eval:
            self.base_actor.eval()
        else:
            self.base_actor.train()

        actor_loss = self.get_base_actor_loss(experience, experience_neg_sampling, custom_prompt)

        # mixtral
        if self.aux_loss:
            raise NotImplementedError
            # aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef

        if self.bc_coef > 0:
            raise NotImplementedError
            print("DOING BEHAVIOUR CLONING")

        self.strategy.backward(loss, self.base_actor, self.base_actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            raise NotImplementedError # not yet checked/fixed
            # data = next(self.pretrain_dataloader)
            # inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            # attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            # label = torch.where(
            #     attention_mask.bool(),
            #     inputs,
            #     self.ptx_loss_fn.IGNORE_INDEX,
            # )
            #
            # output = self.base_actor(inputs, attention_mask=attention_mask, return_output=True)
            # ptx_log_probs = output["logits"]
            #
            # # loss function
            # ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # # mixtral
            # if self.aux_loss:
            #     aux_loss = output.aux_loss
            # else:
            #     aux_loss = 0
            # loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            # self.strategy.backward(self.ptx_coef * loss, self.base_actor, self.base_actor_optim)


        self.strategy.optimizer_step(self.base_actor_optim, self.base_actor, self.base_actor_scheduler, name="base_actor") # this name doesn't appear to do anything though
        if self.ema_model:
            raise NotImplementedError # not tested
            self.strategy.moving_average(self.base_actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {"base_policy_loss": actor_loss.item(), "base_actor_lr": self.base_actor_scheduler.get_last_lr()[0]}
        # if self.pretrain_dataloader is not None:
        #     status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[f"base_{k}"] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[f"base_{k}"] = v.mean().item()
        return status

    def training_step_sampling_actor(self, experience_neg_sampling: Experience, custom_prompt=None) -> Dict[str, float]:
        from openrlhf.utils.utils import print_timestamp
        if self.model_eval:
            self.sampling_actor.eval()
        else:
            self.sampling_actor.train()

        print_timestamp("training - backprop: start loss computation")
        sampling_actor_loss = self.get_sampling_actor_loss(experience_neg_sampling, custom_prompt)

        # mixtral
        if self.aux_loss:
            raise NotImplementedError
            # aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = sampling_actor_loss + aux_loss * self.args.aux_loss_coef

        if self.bc_coef > 0:
            raise NotImplementedError
        # ptx loss
        if self.pretrain_dataloader is not None:
            raise NotImplementedError # not yet checked/fixed

        print_timestamp("training - backprop: start backward pass")
        self.strategy.backward(loss, self.sampling_actor, self.sampling_actor_optim)

        print_timestamp("training - backprop: start optimizer step")
        self.strategy.optimizer_step(self.sampling_actor_optim, self.sampling_actor, self.sampling_actor_scheduler, name="actor")
        print_timestamp("training - backprop: end optimizer step")

        if self.ema_model:
            raise NotImplementedError
            # self.strategy.moving_average(self.sampling_actor, self.ema_model, self.ema_beta, "cpu")

        # Note: Coin flip network training now happens in make_experience before exploration bonus is calculated

        # status
        status = {"sampling_policy_loss": sampling_actor_loss.item(), "sampling_actor_lr": self.sampling_actor_scheduler.get_last_lr()[0]}
        # if self.pretrain_dataloader is not None:
        #     status["ptx_loss"] = ptx_loss.item()
        for k, v in experience_neg_sampling.info.items():
            if k == "kl":
                status[f"sampling_{k}"] = (
                    (v * experience_neg_sampling.info["response_length"]).sum() / experience_neg_sampling.info["response_length"].sum()
                ).item()
            else:
                status[f"sampling_{k}"] = v.mean().item()
        return status


    def get_base_actor_loss(self, experience: Experience, experience_neg_sampling: Experience, custom_prompt=None):

        batch_size = experience.sequences.size(0)
        samples_per_prompt = self.args.duplicate_rollout_batch_by
        num_prompts = batch_size // samples_per_prompt

        if self.base_actor_loss_type == "reinforce":
            action_log_probs = self.base_actor(
                experience.sequences, experience.action_mask.size(1),
                attention_mask=experience.attention_mask, return_output=False
            )

            action_log_probs = action_log_probs.view(num_prompts, samples_per_prompt, -1)

            # final_reward = experience.info["reward"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)
            final_reward_including_kl = experience.info["return"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)
            exper_action_mask = experience.action_mask.view(num_prompts, samples_per_prompt, -1)

            actor_loss = self.base_actor_loss_fn(
                action_log_probs,
                final_reward_including_kl,
                action_mask=exper_action_mask,
            )

        elif self.base_actor_loss_type == "neg_training":
            action_log_probs = self.base_actor(
                experience.sequences, experience.action_mask.size(1),
                attention_mask=experience.attention_mask, return_output=False
            )

            if self.separate_neg_samples:

                action_log_probs_neg = self.base_actor(
                    experience_neg_sampling.sequences, experience_neg_sampling.action_mask.size(1),
                    attention_mask=experience_neg_sampling.attention_mask, return_output=False
                )
            else:
                action_log_probs_neg = action_log_probs


            action_log_probs = action_log_probs.view(num_prompts, samples_per_prompt, -1)
            action_log_probs_neg = action_log_probs_neg.view(num_prompts, samples_per_prompt, -1)

            # final_reward = experience.info["reward"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)
            final_reward_including_kl = experience.info["return"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)

            # Part 3 (base actor SIS): use reward without exploration bonus for reweighting q-samples.
            # Use stored reward_no_bonus when available to avoid a duplicate reward model pass.
            reward_no_bonus = experience_neg_sampling.info.get("reward_no_bonus")
            if reward_no_bonus is not None:
                final_reward_neg = reward_no_bonus.view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device)
            else:
                log_phi, _, _, _ = self.sampling_experience_maker_neg.compute_reward_no_kl(
                    experience_neg_sampling.sequences, experience_neg_sampling.attention_mask,
                    multiply_by_beta=True, force_no_exploration_bonus=True
                )
                final_reward_neg = log_phi.view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device)

            exper_action_mask = experience.action_mask.view(num_prompts, samples_per_prompt, -1)
            exper_neg_action_mask = experience_neg_sampling.action_mask.view(num_prompts, samples_per_prompt, -1)

            if self.uniform_reweight:
                log_w_t_approx_sigma_samples = torch.zeros((num_prompts, samples_per_prompt)).to(action_log_probs.device)
                normalized_w_t_approx_sigma_samples = F.softmax(log_w_t_approx_sigma_samples, dim=-1)
            elif self.separate_reweighting_beta is not None:
                # Just use untransformed reward * the sampling beta. Keep the target_dist_beta as the one for training
                # And use the separate beta for the reweighting of samples for the base actor loss
                normalized_w_t_approx_sigma_samples = get_normalized_positive_weights_detached(
                    action_log_probs_neg,
                    experience_neg_sampling.action_log_probs.view(num_prompts, samples_per_prompt, -1),
                    experience_neg_sampling.info["untransformed_reward"].view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device) * self.separate_reweighting_beta
                )
            else:
                normalized_w_t_approx_sigma_samples = get_normalized_positive_weights_detached(
                    action_log_probs_neg,
                    experience_neg_sampling.action_log_probs.view(num_prompts, samples_per_prompt, -1),
                    final_reward_neg
                )

            print("NORMALIZED POSITIVE WEIGHTS")
            print(normalized_w_t_approx_sigma_samples)

            if self.rm_type == "indicator_below_threshold":
                # Only have any weight (do the negative training/gradient ascent/-SFT) on any samples that satisfy the indicator function
                normalized_w_t_approx_sigma_samples = normalized_w_t_approx_sigma_samples * (torch.exp(final_reward_neg) > INDICATOR_REWARD_EPS * 2) # Assign 0 weights to all samples that do not satisfy the indicator. This really only makes a difference if all the samples do not satisfy the indicator, in which case this ensures no negative training update is applied, otherwise all samples would get equal weights and pushed down equally even if none satisfy the indicator, which is probably not what we want (we don't want to just randomly push down on a bunch of samples that aren't from the target)


            actor_loss = self.base_actor_loss_fn(
                action_log_probs,
                action_log_probs_neg,
                final_reward_including_kl,
                normalized_w_t_approx_sigma_samples=normalized_w_t_approx_sigma_samples, # TODO fill in with maybe the log p phi / q calculation. p has to be using what, using the base_actor I guess, whereas q is the proposal or sampling actor now.
                action_mask=exper_action_mask,
                action_mask_neg=exper_neg_action_mask,
            )
        elif self.base_actor_loss_type == "neg_reinforce":
            action_log_probs = self.base_actor(
                experience.sequences, experience.action_mask.size(1),
                attention_mask=experience.attention_mask, return_output=False
            )

            action_log_probs_neg = self.base_actor(
                experience_neg_sampling.sequences, experience_neg_sampling.action_mask.size(1),
                attention_mask=experience_neg_sampling.attention_mask, return_output=False
            )

            with torch.no_grad():
                base_action_log_probs_neg = self.static_initial_model(
                    experience_neg_sampling.sequences, experience_neg_sampling.action_mask.size(1),
                    attention_mask=experience_neg_sampling.attention_mask, return_output=False
                )

            # Part 3 (base actor): use reward without exploration bonus for SIS and for compute_reward.
            # Use stored reward_no_bonus when available to avoid a duplicate reward model pass.
            reward_neg_no_bonus = experience_neg_sampling.info.get("reward_no_bonus")
            if reward_neg_no_bonus is None:
                reward_neg_no_bonus, _, _, _ = self.sampling_experience_maker_neg.compute_reward_no_kl(
                    experience_neg_sampling.sequences, experience_neg_sampling.attention_mask,
                    multiply_by_beta=self.sampling_experience_maker_neg.multiply_by_beta,
                    force_no_exploration_bonus=True
                )
            reward_neg, _ = compute_reward(
                reward_neg_no_bonus,
                self.kl_ctl.value,
                action_log_probs_neg.detach(),
                base_action_log_probs_neg,
                action_mask=experience_neg_sampling.action_mask,
            )

            action_log_probs = action_log_probs.view(num_prompts, samples_per_prompt, -1)
            action_log_probs_neg = action_log_probs_neg.view(num_prompts, samples_per_prompt, -1)
            # base_action_log_probs_neg = base_action_log_probs_neg.view(num_prompts, samples_per_prompt, -1)

            final_reward_no_kl = experience.info["reward"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)
            final_reward_including_kl = experience.info["return"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)

            final_reward_neg = reward_neg_no_bonus.view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device)
            # untransformed_rewards_neg = experience_neg_sampling.info["untransformed_reward"].view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device)


            exper_action_mask = experience.action_mask.view(num_prompts, samples_per_prompt, -1)
            exper_neg_action_mask = experience_neg_sampling.action_mask.view(num_prompts, samples_per_prompt, -1)

            if self.uniform_reweight:
                log_w_t_approx_sigma_samples = torch.zeros((num_prompts, samples_per_prompt)).to(action_log_probs.device)
                normalized_w_t_approx_sigma_samples = F.softmax(log_w_t_approx_sigma_samples, dim=-1)
            elif self.separate_reweighting_beta is not None:
                # Just use untransformed reward * the sampling beta. Keep the target_dist_beta as the one for training
                # And use the separate beta for the reweighting of samples for the base actor loss
                normalized_w_t_approx_sigma_samples = get_normalized_positive_weights_detached(
                    action_log_probs_neg,
                    experience_neg_sampling.action_log_probs.view(num_prompts, samples_per_prompt, -1),
                    experience_neg_sampling.info["untransformed_reward"].view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device) * self.separate_reweighting_beta
                )
            else:
                normalized_w_t_approx_sigma_samples = get_normalized_positive_weights_detached(
                    action_log_probs_neg,
                    experience_neg_sampling.action_log_probs.view(num_prompts, samples_per_prompt, -1),
                    final_reward_neg
                )

            print("NORMALIZED POSITIVE WEIGHTS")
            print(normalized_w_t_approx_sigma_samples)

            # Zero out IS weights for samples that don't satisfy the indicator function,
            # consistent with the neg_training path above. Without this, all samples get
            # equal weights even if none satisfy the indicator, causing undesired gradient updates.
            if self.rm_type == "indicator_below_threshold":
                normalized_w_t_approx_sigma_samples = normalized_w_t_approx_sigma_samples * (torch.exp(final_reward_neg) > INDICATOR_REWARD_EPS * 2)

            actor_loss = self.base_actor_loss_fn(
                action_log_probs,
                action_log_probs_neg,
                final_reward_including_kl,
                rewards_neg=reward_neg.sum(dim=-1).view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device),
                normalized_w_t_approx_sigma_samples=normalized_w_t_approx_sigma_samples, # TODO fill in with maybe the log p phi / q calculation. p has to be using what, using the base_actor I guess, whereas q is the proposal or sampling actor now.
                action_mask=exper_action_mask,
                action_mask_neg=exper_neg_action_mask,
                # standard_final_reward_no_kl=final_reward_no_kl
            )

        else:
            raise NotImplementedError

        return actor_loss



    def get_sampling_actor_loss(self, experience, custom_prompt=None):

        batch_size = experience.sequences.size(0)
        samples_per_prompt = self.args.duplicate_rollout_batch_by
        num_prompts = batch_size // samples_per_prompt

        if self.sampling_actor_loss_type in ["ctl", "ctl_nosecondterm"]:
            # Right now by using experience_maker sequences, this is essentially just twisted proposal samples
            # And we do CTL by reweighting those according to the twist values and tilde sigma values.

            with torch.no_grad():
                base_action_log_probs = self.base_actor(
                    experience.sequences, experience.action_mask.size(1),
                    experience.attention_mask)
                # log_phi = self.base_experience_maker.compute_reward_no_kl(
                #     experience.sequences, experience.attention_mask, multiply_by_beta=True # beta multiplied for non-PPO formulations
                # )
                log_phi = experience.info["reward"].to(base_action_log_probs.device)

            # print("REWARD COMPARISON")
            # print(experience.returns[:, -1] - log_phi) # same
            if "policy" in self.parameterization:
                log_psi = self.get_log_psi_policy_parameterization(self.sampling_actor, base_action_log_probs, experience, experience.action_mask.size(1), self.parameterization)
            else:
                log_psi = self.sampling_actor(experience.sequences, experience.action_mask.size(1), experience.attention_mask,
                                                      return_only_modulation=True)

            # Reshape tensors to group samples by prompt
            log_psi = log_psi.view(num_prompts, samples_per_prompt, -1)
            log_phi = log_phi.view(num_prompts, samples_per_prompt)
            exper_action_mask = experience.action_mask.view(num_prompts, samples_per_prompt, -1)
            exper_action_log_probs = experience.action_log_probs.view(num_prompts, samples_per_prompt, -1)
            base_action_log_probs = base_action_log_probs.view(num_prompts, samples_per_prompt, -1)

            # --- Mixture proposal modifications ---
            mixture_kwargs = {}
            if self.mixture_proposal:
                device = log_psi.device

                # Recompute q_best log probs on current sequences (frozen model, no_grad).
                # We recompute here rather than storing in experience.info because the replay
                # buffer's split_experience_batch only supports scalar info values.
                with torch.no_grad():
                    q_best_alp = self.q_best_model(
                        experience.sequences, experience.action_mask.size(1),
                        experience.attention_mask)
                q_best_alp = q_best_alp.view(num_prompts, samples_per_prompt, -1)

                mixture_seq_lp, mixture_partial_seq_lp = self._compute_mixture_log_probs(
                    exper_action_log_probs, q_best_alp, exper_action_mask)

                if self.mixture_psi_use_mix:
                    # Use q_mix (instead of q_current) for log_psi.
                    # log_psi_mix_t = log q_mix(s_t | s_{1:t-1}) - log p(s_t | s_{1:t-1})
                    #
                    # The gradient through logaddexp w.r.t. q_current's params includes
                    # the responsibility factor r(x) = w * q_current(x) / q_mix(x),
                    # which down-weights gradients for samples dominated by q_best.
                    assert "policy" in self.parameterization, (
                        "mixture_psi_use_mix requires policy parameterization")
                    assert self.parameterization == "policy_psi_q_p_s_t", (
                        f"mixture_psi_use_mix only implemented for policy_psi_q_p_s_t, "
                        f"got {self.parameterization}")

                    # Recover per-token log q_current WITH gradient.
                    # log_psi = log_q_current - log_p (per token), so log_q = log_psi + log_p.
                    # base_action_log_probs is no_grad; gradient flows through log_psi.
                    log_q_current_with_grad = log_psi + base_action_log_probs  # (P, n, A)

                    # Compute mixture partial-sequence log probs with gradient through q_current.
                    q_curr_masked_g = log_q_current_with_grad * exper_action_mask
                    q_best_masked_g = q_best_alp * exper_action_mask  # no grad (frozen)
                    mixture_partial_g = torch.logaddexp(
                        self.log_w_current + q_curr_masked_g.cumsum(dim=-1),
                        self.log_w_best + q_best_masked_g.cumsum(dim=-1),
                    )  # (P, n, A) with gradient through q_current

                    # Per-token mixture log prob: log q_mix(s_t | s_{1:t-1}) = diff of partial
                    log_q_mix_token = torch.zeros_like(mixture_partial_g)
                    log_q_mix_token[:, :, 0] = mixture_partial_g[:, :, 0]
                    log_q_mix_token[:, :, 1:] = (
                        mixture_partial_g[:, :, 1:] - mixture_partial_g[:, :, :-1]
                    )

                    # Replace log_psi with mixture version
                    log_psi = log_q_mix_token - base_action_log_probs

                if self.mixture_optimization == "mixture":
                    # All n samples, q_mix weights everywhere
                    mixture_kwargs["mixture_seq_log_probs"] = mixture_seq_lp
                    mixture_kwargs["mixture_partial_seq_log_probs"] = mixture_partial_seq_lp

                elif self.mixture_optimization == "q_half":
                    # Positive: all n samples with mixture weights
                    # Negative: first n_current samples (from q_current) with q_current's weights
                    nc = self.n_current
                    mixture_kwargs["mixture_seq_log_probs"] = mixture_seq_lp
                    # Note on log_psi: We use q_current (not q_mix) for log_psi.
                    #
                    # With policy parameterization, log_psi = log q_theta - log p, where theta are
                    # q_current's parameters.
                    #
                    # Case 1 (what we do): log_psi = log q_theta(x) - log p(x)
                    #   grad_theta log_psi(x) = grad_theta log q_theta(x)
                    #   Every sample gets the full gradient signal.
                    #
                    # Case 2 (hypothetical): log_psi_mix = log q_mix(x) - log p(x)
                    #   where q_mix(x) = w * q_theta(x) + (1-w) * q_best(x).
                    #   grad_theta log q_mix(x) = (w * q_theta(x)) / q_mix(x) * grad_theta log q_theta(x)
                    #                           = r(x) * grad_theta log q_theta(x)
                    #   where r(x) = w * q_theta(x) / (w * q_theta(x) + (1-w) * q_best(x)) in [0, 1]
                    #   is the "responsibility" of q_current for sample x in the mixture.
                    #
                    # The problem: r(x) -> 0 when q_best(x) >> q_theta(x). But those are precisely
                    # the samples where the mixture adds value (regions q_current hasn't learned yet).
                    # So the gradient for the most informative mixture samples gets suppressed.
                    #
                    # Conceptually, psi approximates the optimal twist sigma(x)/p(x), and is separate
                    # from the proposal distribution. The SIS importance weights already handle the
                    # "where did this sample come from" correction (dividing by q_mix). Making log_psi
                    # also encode mixture composition would double-count the proposal correction.
                    mixture_kwargs["neg_values"] = log_psi[:, :nc, :]
                    mixture_kwargs["neg_action_mask"] = exper_action_mask[:, :nc, :]
                    mixture_kwargs["neg_curr_log_probs"] = exper_action_log_probs[:, :nc, :]
                    mixture_kwargs["neg_base_action_log_probs"] = base_action_log_probs[:, :nc, :]

                elif self.mixture_optimization == "q_independent":
                    # Positive: all n mixture samples with mixture weights
                    # Negative: D independently-generated q_current samples
                    mixture_kwargs["mixture_seq_log_probs"] = mixture_seq_lp
                    D = self.args.duplicate_rollout_batch_by

                    # Retrieve independent samples stored on self (not in experience.info,
                    # since the replay buffer only supports scalar info values).
                    assert hasattr(self, '_q_ind_data') and self._q_ind_data is not None, (
                        "q_independent mode requires _q_ind_data; was make_experience_and_do_update called?")
                    ind_sequences = self._q_ind_data["sequences"].to(device)
                    ind_action_mask = self._q_ind_data["action_mask"].to(device)
                    ind_attention_mask = self._q_ind_data["attention_mask"].to(device)
                    ind_action_log_probs = self._q_ind_data["action_log_probs"].to(device)

                    with torch.no_grad():
                        ind_base_alp = self.base_actor(
                            ind_sequences, ind_action_mask.size(1), ind_attention_mask)

                    # Forward pass through sampling_actor for log_psi on independent samples.
                    # get_log_psi_policy_parameterization uses experience.sequences/attention_mask
                    # for the forward pass; we need ind_sequences instead, so use a temporary object.
                    class _TempExperience:
                        pass
                    temp_exp = _TempExperience()
                    temp_exp.sequences = ind_sequences
                    temp_exp.attention_mask = ind_attention_mask
                    temp_exp.action_mask = ind_action_mask

                    if "policy" in self.parameterization:
                        ind_log_psi = self.get_log_psi_policy_parameterization(
                            self.sampling_actor, ind_base_alp, temp_exp, ind_action_mask.size(1),
                            self.parameterization)
                    else:
                        ind_log_psi = self.sampling_actor(
                            ind_sequences, ind_action_mask.size(1), ind_attention_mask,
                            return_only_modulation=True)

                    # Reshape to (P, D, A)
                    ind_log_psi = ind_log_psi.view(num_prompts, D, -1)
                    ind_action_mask_r = ind_action_mask.view(num_prompts, D, -1)
                    ind_action_log_probs_r = ind_action_log_probs.view(num_prompts, D, -1)
                    ind_base_alp_r = ind_base_alp.view(num_prompts, D, -1)

                    mixture_kwargs["neg_values"] = ind_log_psi
                    mixture_kwargs["neg_action_mask"] = ind_action_mask_r
                    mixture_kwargs["neg_curr_log_probs"] = ind_action_log_probs_r
                    mixture_kwargs["neg_base_action_log_probs"] = ind_base_alp_r

            # Calculate loss for all groups at once
            sampling_actor_loss = self.sampling_actor_loss_fn(
                log_psi,  # shape: [num_prompts, samples_per_prompt, num_actions]
                log_phi,  # shape: [num_prompts, samples_per_prompt]
                exper_action_mask,
                exper_action_log_probs,
                base_action_log_probs,
                **mixture_kwargs,
            )
        elif self.sampling_actor_loss_type in ["dpg"]:
            with torch.no_grad():
                base_action_log_probs_all_vocab, base_action_log_probs = self.base_actor(
                    experience.sequences, experience.action_mask.size(1),
                    experience.attention_mask, return_type="both")
                # log_phi = self.experience_maker.compute_reward_no_kl(
                #     experience.sequences, experience.attention_mask, multiply_by_beta=True
                #     # beta multiplied for non-PPO formulations
                # )
                log_phi = experience.info["reward"].to(base_action_log_probs.device)

            if "policy" in self.parameterization:
                # call actor with return_all_vocab=True
                log_psi_all_vocab, log_psi = self.get_log_psi_policy_parameterization(self.sampling_actor, base_action_log_probs, experience, experience.action_mask.size(1), self.parameterization, return_type="both", base_action_log_probs_all=base_action_log_probs_all_vocab)
            else:
                log_psi_all_vocab, log_psi = self.sampling_experience_maker_neg.actor(experience.sequences, experience.action_mask.size(1), experience.attention_mask,
                                                      return_only_modulation=True, return_type="both")

            # Reshape tensors to group samples by prompt
            log_psi = log_psi.view(num_prompts, samples_per_prompt, -1)
            log_phi = log_phi.view(num_prompts, samples_per_prompt)
            exper_action_mask = experience.action_mask.view(num_prompts, samples_per_prompt, -1)
            exper_action_log_probs = experience.action_log_probs.view(num_prompts, samples_per_prompt, -1)
            base_action_log_probs = base_action_log_probs.view(num_prompts, samples_per_prompt, -1)

            log_psi_all_vocab = log_psi_all_vocab.view(num_prompts, samples_per_prompt, log_psi_all_vocab.shape[1], log_psi_all_vocab.shape[2])
            base_action_log_probs_all_vocab = base_action_log_probs_all_vocab.view(num_prompts, samples_per_prompt, base_action_log_probs_all_vocab.shape[1], base_action_log_probs_all_vocab.shape[2])

            # Calculate loss for all groups at once
            sampling_actor_loss = self.sampling_actor_loss_fn(
                log_psi,  # shape: [num_prompts, samples_per_prompt, num_actions]
                log_phi,  # shape: [num_prompts, samples_per_prompt]
                exper_action_mask,
                exper_action_log_probs,
                base_action_log_probs,
                # reduce_mean_per_prompt=True
                log_psi_all_vocab,
                base_action_log_probs_all_vocab,
            )

        else:
            raise NotImplementedError

        return sampling_actor_loss

    def get_log_psi_policy_parameterization(self, actor, base_action_log_probs, experience, num_actions, parameterization, return_type: str = 'p', base_action_log_probs_all=None):

        if return_type == "both":

            assert base_action_log_probs_all is not None

            if parameterization == "policy_psi_unnorm":
            # if log_psi_parameterization_type == "unnormalized_q_s_t_logits_minus_log_p_s_t":
                log_p_psi_all, log_p_psi = actor(experience.sequences, num_actions,
                                                                       experience.attention_mask,
                                                                       return_type=return_type,
                                                                       return_unnormalized=True)
            elif parameterization in ["policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"]:
            # elif log_psi_parameterization_type in ["log_q_s_t_minus_log_p_s_t", "log_q_s_1_to_t_minus_log_p_s_1_to_t"]:
                log_p_psi_all, log_p_psi = actor(experience.sequences, num_actions,
                                                                       experience.attention_mask,
                                                                       return_type=return_type)
            else:
                raise NotImplementedError

            if parameterization == "policy_psi_q_p_s_1_to_t":
            # if log_psi_parameterization_type == "log_q_s_1_to_t_minus_log_p_s_1_to_t":
                log_p_psi_all = torch.cumsum(log_p_psi_all, dim=1)
                log_p_psi = torch.cumsum(log_p_psi, dim=1)
                base_action_log_probs = torch.cumsum(base_action_log_probs, dim=1)
                base_action_log_probs_all = torch.cumsum(base_action_log_probs_all, dim=1)

            log_psi = log_p_psi - base_action_log_probs.detach()
            log_psi_all = log_p_psi_all - base_action_log_probs_all.detach()

            return log_psi_all, log_psi

        if parameterization == "policy_psi_unnorm":
        # if log_psi_parameterization_type == "unnormalized_q_s_t_logits_minus_log_p_s_t":
            log_p_psi = actor(experience.sequences, num_actions,
                                                                   experience.attention_mask,
                                                                   return_type=return_type,
                                                                   return_unnormalized=True)
        elif parameterization in ["policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"]:
        # elif log_psi_parameterization_type in ["log_q_s_t_minus_log_p_s_t", "log_q_s_1_to_t_minus_log_p_s_1_to_t"]:
            log_p_psi = actor(experience.sequences, num_actions,
                                                                   experience.attention_mask,
                                                                   return_type=return_type)
        else:
            raise NotImplementedError

        if parameterization == "policy_psi_q_p_s_1_to_t":
        # if log_psi_parameterization_type == "log_q_s_1_to_t_minus_log_p_s_1_to_t":
            log_p_psi = torch.cumsum(log_p_psi, dim=1)
            base_action_log_probs = torch.cumsum(base_action_log_probs, dim=1)

        # log_p_psi = self.base_experience_maker.actor(experience.sequences, num_actions, experience.attention_mask, return_type=return_type)

        log_psi = log_p_psi - base_action_log_probs.detach()  # In the policy formulation, the actor directly outputs log (p psi) = log_p + log_psi, so get log_psi by subtracting log_p
        # For gradients this subtraction does nothing, however it should be needed to get the correct importance weights

        return log_psi


    def training_step_critic(self, experience: Experience, custom_prompt=None) -> Dict[str, float]:
        raise NotImplementedError # Not yet tested
        if self.model_eval:
            self.base_critic.eval()
        else:
            self.base_critic.train()

        # critic loss
        values, output = self.base_critic(
            experience.sequences,
            action_mask=experience.action_mask,
            attention_mask=experience.attention_mask,
            return_output=True,
        )
        # loss function
        critic_loss = self.get_critic_loss(experience, values, custom_prompt=custom_prompt)
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        loss = loss.float()
        self.strategy.backward(loss, self.base_critic, self.base_critic_optim)
        self.strategy.optimizer_step(self.base_critic_optim, self.base_critic, self.base_critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.base_critic_scheduler.get_last_lr()[0],
        }
        return status

    def get_critic_loss(self, experience, values, custom_prompt=None):
        raise NotImplementedError # not yet tested
        # return critic_loss


    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None, client_states=None):
        if logs_dict is None:
            logs_dict = {}
        if client_states is None:
            client_states = {}
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric

        if self.total_steps > 0 and self.total_steps % args.save_steps_harmless == 0:
        # if global_step % args.save_steps == 0:
            print(f"SAVING CHECKPOINT AT TOTAL POLICY HARMLESSNESS TRAINING STEPs {self.total_steps}", flush=True)
            tag = f"total_step{self.total_steps}"
            self._save_base_checkpoint(args, tag, client_states)
            if getattr(args, 'rejection_sample_each_save', False):
                self._attempt_rejection_sampling_at_checkpoint(args, tag)

        if self.total_steps > 0 and self.total_steps % args.save_steps == 0:
        # if global_step % args.save_steps == 0:
        #     print(f"SAVING CHECKPOINT AT GLOBAL STEP {global_step}", flush=True)
            print(f"SAVING CHECKPOINT AT TOTAL PROPOSAL/TWIST LEARNING STEPs {self.total_steps}", flush=True)
            tag = f"total_step{self.total_steps}"
            self._save_proposal_checkpoint(args, tag, client_states)


    def _save_base_checkpoint(self, args, tag, client_states):

        info_name_str = get_info_name_str(args)
        save_str = f"{info_name_str}"

        save_dir = os.path.join(args.ckpt_path, f"{save_str}_harml_actor")

        if getattr(args, 'no_save_optim', False):
            # Save HuggingFace-format weights only (no optimizer states) — much smaller.
            # Note: save_model does NOT prune old checkpoints (unlike save_ckpt).
            tag_dir = os.path.join(save_dir, tag)
            self.strategy.save_model(self.base_actor.model, self.tokenizer, tag_dir)
        else:
            self.strategy.save_ckpt(
                self.base_actor.model,
                save_dir,
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
        if self.base_critic is not None:
            self.strategy.save_ckpt(
                self.base_critic, os.path.join(args.ckpt_path, f"{save_str}_harml_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
            )

    def _attempt_rejection_sampling_at_checkpoint(self, args, step_tag):
        """Attempt rejection sampling at the current base actor state and save results."""
        from openrlhf.utils.utils import rejection_sample_for_prompt

        # Single-prompt only for now
        if not args.new_custom_single_prompt:
            raise NotImplementedError(
                "Rejection sampling at checkpoint save is only supported for single-prompt mode. "
                "TODO: multi-prompt support requires per-prompt rejection sampling."
            )

        prompt_text = get_custom_prompt_with_chat_template(
            self.tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), self.strategy
        )

        generate_kwargs = {
            "max_new_tokens": args.generate_max_len,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 1.0,
        }

        batch_size = args.batch_size_rejection_sample if args.batch_size_rejection_sample is not None else args.duplicate_rollout_batch_by
        max_gen = getattr(args, 'max_gen_per_prompt_rejection', None)

        self.base_actor.eval()
        target_sample_amount = getattr(args, 'true_target_sample_amount', None)
        accepted_seqs, accepted_rewards, total_generated = rejection_sample_for_prompt(
            actor=self.base_actor,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            prompt=prompt_text,
            target_dist_beta=args.target_dist_beta,
            reward_clamp=args.reward_clamp,
            reward_cap=getattr(args, 'reward_cap', None),
            prompt_max_len=args.prompt_max_len,
            generate_kwargs=generate_kwargs,
            batch_size=batch_size,
            max_gen=max_gen,
            target_sample_amount=target_sample_amount,
            tile_prompts_fn=tile_prompts,
            rm_type=args.rm_type,
            strategy=self.strategy,
        )

        info_name_str = get_info_name_str(args)
        rejection_dir = os.path.join(args.ckpt_path, f"{info_name_str}_rejection_samples")
        os.makedirs(rejection_dir, exist_ok=True)

        total_accepted = len(accepted_seqs)
        rate = total_accepted / total_generated if total_generated > 0 else 0.0
        print(f"Rejection sampling at {step_tag}: {total_accepted} accepted from "
              f"{total_generated} generated (rate: {rate:.4f})", flush=True)

        if total_accepted > 0:
            save_data = {
                "accepted_seqs": accepted_seqs,
                "accepted_rewards": accepted_rewards,
                "total_generated": total_generated,
                "total_accepted": total_accepted,
                "prompt": args.custom_prompt,
            }
            save_path = os.path.join(rejection_dir, f"{step_tag}.pt")
            torch.save(save_data, save_path)
            print(f"Saved rejection samples to {save_path}", flush=True)

            # Track steps with rejection samples
            if not hasattr(self, '_trajectory_steps_with_rejection_samples'):
                self._trajectory_steps_with_rejection_samples = []
            self._trajectory_steps_with_rejection_samples.append(step_tag)
        else:
            print(f"No samples accepted at {step_tag}; skipping save.", flush=True)

    def _load_trajectory_checkpoint(self, index):
        """Load a trajectory checkpoint by index into the base actor."""
        step_num, tag = self.trajectory_checkpoints[index]
        ckpt_dir = os.path.join(self._trajectory_dir, tag)
        print(f"Loading trajectory checkpoint {index + 1}/{len(self.trajectory_checkpoints)}: "
              f"tag={tag} at total_steps={self.total_steps}, path={ckpt_dir}", flush=True)

        if self.trajectory_is_hf_format:
            # Load HuggingFace format (saved with strategy.save_model / --no_save_optim)
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
            pytorch_bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")

            if os.path.exists(safetensors_path):
                weights_path = safetensors_path
            elif os.path.exists(pytorch_bin_path):
                weights_path = pytorch_bin_path
            else:
                raise FileNotFoundError(
                    f"No model weights found in {ckpt_dir}. "
                    f"Expected model.safetensors or pytorch_model.bin"
                )

            # strict=False (default in load_model) handles tied weights (e.g. SmolLM
            # ties lm_head.weight to model.embed_tokens.weight; save_model skips the
            # duplicate key). Load directly to the model's device to avoid mismatch.
            device = next(self.strategy._unwrap_model(self.base_actor.model).parameters()).device
            self.strategy.load_model(self.base_actor.model, weights_path, map_location=device)
            print(f"Loaded HuggingFace checkpoint from {ckpt_dir}", flush=True)
        else:
            # Load DeepSpeed format
            self.strategy.load_ckpt(
                self.base_actor.model,
                self._trajectory_dir,
                tag=tag,
                load_module_only=True,
            )
            print(f"Loaded DeepSpeed checkpoint from {ckpt_dir}", flush=True)

        # Update current rejection samples
        if hasattr(self, 'trajectory_rejection_samples'):
            self.current_trajectory_rejection_samples = self.trajectory_rejection_samples.get(tag, None)

    def _maybe_load_next_trajectory_checkpoint(self):
        """Check if it's time to load the next trajectory checkpoint."""
        if not hasattr(self, 'trajectory_checkpoints'):
            return

        next_load_step = (self.trajectory_ckpt_index + 1) * self.trajectory_steps_per_ckpt
        if self.total_steps >= next_load_step:
            self.trajectory_ckpt_index += 1
            if self.trajectory_ckpt_index < len(self.trajectory_checkpoints):
                self._load_trajectory_checkpoint(self.trajectory_ckpt_index)
            else:
                # Trajectory exhausted — raise error
                raise RuntimeError(
                    f"Trajectory exhausted at step {self.total_steps}. "
                    f"The trajectory has {len(self.trajectory_checkpoints)} checkpoints "
                    f"but training reached step {self.total_steps}. "
                    f"Reduce training steps or extend the trajectory."
                )

    def _save_proposal_checkpoint(self, args, tag, client_states):
        info_name_str = get_info_name_str(args)
        save_str = f"{info_name_str}"
        # save_str = f"PPOepochs{args.max_epochs}{eval_str}_lrschedule{args.lr_scheduler}_{lr_str}_criticloss{args.critic_loss_type}_{extra_str}_seed{args.seed}"

        if args.parameterization == "modulation_model":
            self.strategy.save_ckpt(
                self.sampling_actor,
                os.path.join(args.ckpt_path, f"{save_str}_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )

        elif args.parameterization in ["modulation_linear_head", "modulation_nn_head"]:
            save_path = os.path.join(args.ckpt_path, f"{save_str}_actor_step{tag}")
            torch.save(self.sampling_actor.modulation_head.state_dict(), save_path)


        else:
            self.strategy.save_ckpt(
                self.sampling_actor.model,
                os.path.join(args.ckpt_path, f"{save_str}_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
            if self.sampling_critic is not None:
                if not args.no_save_critic:
                    self.strategy.save_ckpt(
                        self.sampling_critic, os.path.join(args.ckpt_path, f"{save_str}_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
                    )
