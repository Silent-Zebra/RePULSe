import math
import os.path
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
from openrlhf.models.loss import REINFORCELoss, NegTrainingLoss, NegREINFORCELoss, CTLLoss, DPGLoss
from openrlhf.models.utils import masked_mean, compute_approx_kl, compute_reward
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.utils import get_info_name_str, tile_prompts, inspect_rewards_list, log_sequence_for_negatives

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveReplayBuffer
from openrlhf.trainer.ppo_utils.experience_maker import BaseExperienceMaker

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
        reward_cap: float = 4.5,
        target_dist_beta: float = 1,
        n_seeds_f_q: int = 4,
        rm_type: str = '',
        bc_coef: float = 0,
        bc_steps: int = -1,
        true_posterior_samples = None, # would otherwise be torch.Tensor
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

        self.true_posterior_samples = true_posterior_samples

        self.model_eval = model_eval

        self.n_seeds_f_q = n_seeds_f_q

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
            reward_cap,
            1, # target_dist_beta 1 here, because this is just going to need regular rewards for REINFORCE
            self.rew_trans_alpha,
            "rlhf", # Use this to ensure the standard reward formulation
            base_actor_loss_type, # Does not matter, when the target_dist_beta is 1
            self.generate_kwargs['max_new_tokens'],
            save_negdata=save_negdata,
            save_negdata_threshold=save_negdata_threshold,
            neg_data=self.neg_data,
            reward_transform = self.reward_transform,
            reward_transform_beta = self.rew_trans_beta
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
        true_posterior_samples=None,
    ) -> (List, List, List, List):

        if args.custom_single_prompt:
            raise NotImplementedError

        else:
            num_rollouts_per_episodes = (
                num_update_steps_per_episodes * args.train_batch_size // args.max_epochs // args.rollout_batch_size
            )
            update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)

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

        print("INSPECT_HARMLESS")
        print(num_update_steps_per_episodes)
        print(args.train_batch_size)
        print(args.max_epochs)
        print(args.rollout_batch_size)
        print(args.train_batch_size // args.max_epochs // args.rollout_batch_size)

        print(consumed_samples)
        print(args.rollout_batch_size)
        print(num_rollouts_per_episodes)


        steps = consumed_samples // args.rollout_batch_size * update_timesteps + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        print("INSPECT_HARMLESS2")
        print(steps)
        print(start_episode)
        print(consumed_samples)

        if consumed_samples > 0:
            raise NotImplementedError # Should check that this all works correctly after I modified it.

        iwae_lbs_list = []
        iwae_ubs_list = []
        f_q_estimates_list = []
        g_q_estimates_list = []
        rewards_list = []
        kl_vals_list = []
        entropy_list = []
        untrans_ret_list = []

        estimates_list = (f_q_estimates_list, rewards_list, kl_vals_list, entropy_list, untrans_ret_list)

        custom_prompt = None

        assert start_episode < args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop # Otherwise no updates done; this might be ok depending on setup, but for now this would be unexpected behaviour.

        total_update_steps = self.prompts_dataloader.__len__() * args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop * args.fit_steps

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

        for episode in range(start_episode, args.harmlessness_training_num_episodes * args.harmlessness_training_episodes_per_loop): # Actually with this current setup is kind of redundant to have these 2 hyperparameters, loops here or in the outer loop, just pick one, doesn't really matter with 1 update each...
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
                    new_beta = beta_schedule[self.total_steps]
                    self.sampling_experience_maker_neg.target_dist_beta = new_beta
                    print(f"Using new beta: {new_beta}")
                if args.start_alpha is not None:
                    new_alpha = alpha_schedule[self.total_steps]
                    self.alpha = new_alpha
                    self.base_actor_loss_fn = self.get_base_actor_loss_fn()
                    print(f"Using new alpha: {new_alpha}")

                if args.new_custom_single_prompt:
                    rand_prompts = ["This man is a"]

                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #              profile_memory=True, record_shapes=True) as prof:

                if args.num_episodes > 1:
                    for q_train_step in range(args.num_episodes - 1):
                        self.make_experience_and_do_update(args, custom_prompt, pbar, rand_prompts, rewards_list, steps,
                                                           untrans_ret_list, update_timesteps, neg_sample_only=True)


                self.make_experience_and_do_update(args, custom_prompt, pbar, rand_prompts, rewards_list, steps,
                                                   untrans_ret_list, update_timesteps, neg_sample_only=False)

        if args.custom_single_prompt:
            return iwae_lbs_list, iwae_ubs_list, f_q_estimates_list, g_q_estimates_list
        else:
            return estimates_list

    def make_experience_and_do_update(self, args, custom_prompt, pbar, rand_prompts, rewards_list, steps,
                                      untrans_ret_list, update_timesteps, neg_sample_only=False):
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

        if self.separate_neg_samples:
            print("Making experience: neg sampling")

            experience_neg_sampling = self.sampling_experience_maker_neg.make_experience(
                rand_prompts,
                samples_per_prompt=args.duplicate_rollout_batch_by,
                **self.generate_kwargs
            )
            self.sampling_replay_buffer_neg.append(experience_neg_sampling)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #              profile_memory=True, record_shapes=True) as prof:

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
            status = self.train(global_steps, custom_prompt=custom_prompt, neg_sample_only=neg_sample_only)

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
                    # assert len(dataloader) == len(dataloader_neg)
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

            if "base_policy_loss" in status:
                short_status["bpg"] = status["base_policy_loss"]

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
            raise NotImpelementedError
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
        if self.model_eval:
            self.sampling_actor.eval()
        else:
            self.sampling_actor.train()

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

        self.strategy.backward(loss, self.sampling_actor, self.sampling_actor_optim)

        self.strategy.optimizer_step(self.sampling_actor_optim, self.sampling_actor, self.sampling_actor_scheduler, name="actor")

        if self.ema_model:
            raise NotImplementedError
            # self.strategy.moving_average(self.sampling_actor, self.ema_model, self.ema_beta, "cpu")

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

            if self.separate_neg_samples:
                final_reward_neg = experience_neg_sampling.info["reward"].view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device)
            else:
                # experience_neg_sampling = experience here
                log_phi, _ = self.sampling_experience_maker_neg.compute_reward_no_kl(experience_neg_sampling.sequences, experience_neg_sampling.attention_mask, multiply_by_beta=True)
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

                print("Indicator weights inspection")
                print(final_reward_neg)
                print(normalized_w_t_approx_sigma_samples)

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


            reward_neg, _ = compute_reward(
                experience_neg_sampling.info["reward"],
                self.kl_ctl.value,
                action_log_probs_neg,
                base_action_log_probs_neg,
                action_mask=experience_neg_sampling.action_mask,
            )

            action_log_probs = action_log_probs.view(num_prompts, samples_per_prompt, -1)
            action_log_probs_neg = action_log_probs_neg.view(num_prompts, samples_per_prompt, -1)
            # base_action_log_probs_neg = base_action_log_probs_neg.view(num_prompts, samples_per_prompt, -1)

            final_reward_no_kl = experience.info["reward"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)
            final_reward_including_kl = experience.info["return"].view(num_prompts, samples_per_prompt).to(action_log_probs.device)

            final_reward_neg = experience_neg_sampling.info["reward"].view(num_prompts, samples_per_prompt).to(action_log_probs_neg.device)
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

            # Calculate loss for all groups at once
            sampling_actor_loss = self.sampling_actor_loss_fn(
                log_psi,  # shape: [num_prompts, samples_per_prompt, num_actions]
                log_phi,  # shape: [num_prompts, samples_per_prompt]
                exper_action_mask,
                exper_action_log_probs,
                base_action_log_probs,
                # reduce_mean_per_prompt=True
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
                log_psi_all_vocab, log_psi = self.experience_maker.actor(experience.sequences, experience.action_mask.size(1), experience.attention_mask,
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


    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
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

        if self.total_steps > 0 and self.total_steps % args.save_steps == 0:
        # if global_step % args.save_steps == 0:
        #     print(f"SAVING CHECKPOINT AT GLOBAL STEP {global_step}", flush=True)
            print(f"SAVING CHECKPOINT AT TOTAL PROPOSAL/TWIST LEARNING STEPs {self.total_steps}", flush=True)
            tag = f"total_step{self.total_steps}"
            self._save_proposal_checkpoint(args, tag, client_states)


    def _save_base_checkpoint(self, args, tag, client_states):

        info_name_str = get_info_name_str(args)
        save_str = f"{info_name_str}"
        # save_str = f"PPOepochs{args.max_epochs}{eval_str}_lrschedule{args.lr_scheduler}_{lr_str}_criticloss{args.base_criti_loss_type}_{extra_str}_seed{args.seed}"

        self.strategy.save_ckpt(
            self.base_actor.model,
            os.path.join(args.ckpt_path, f"{save_str}_harml_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.base_critic is not None:
            self.strategy.save_ckpt(
                self.base_critic, os.path.join(args.ckpt_path, f"{save_str}_harml_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
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
            save_path = os.path.join(args.ckpt_path, f"{save_str}_actor")

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
