import math
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

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
from openrlhf.models.loss import REINFORCELoss, NegTrainingLoss, NegREINFORCELoss
from openrlhf.models.utils import masked_mean, compute_approx_kl
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.utils import get_info_name_str, tile_prompts

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer


class HarmlessnessTrainer(ABC):
    """
        Trainer for Harmlessness training algorithm.

    Args:
        TODO THIS DOCUMENTATION NOT UPDATED
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (nn.Module): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
        remote_rm_url (str, optional): function for reward model api
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        sampling_actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
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
        actor_loss_type: str = 'ppo',
        critic_loss_type: str = 'mse',
        alpha: float = 0.5,
        parameterization: str = '',
        save_negdata=False,
        save_negdata_threshold=-10000,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
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

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        assert parameterization != ""
        self.parameterization = parameterization


        # Just do very simple negative training, REINFORCE (on base samples), and REINFORCE (on sigma samples)
        # Then have the ability to combine the above ones (we need REINFORCE on base samples, what is "actor" here, plus with either neg train or reinforce on bad (sigma) samples)

        self.actor_loss_type = actor_loss_type
        if self.actor_loss_type == "reinforce":
            self.actor_loss_fn = REINFORCELoss() # PolicyLoss(eps_clip)
        elif self.actor_loss_type == "neg_training":
            self.actor_loss_fn = NegTrainingLoss(alpha)
        elif self.actor_loss_type == "neg_reinforce":
            self.actor_loss_fn = NegREINFORCELoss(alpha)
        else:
            raise NotImplementedError


        self.shuffle_replay_buffer_sample = False


        self.critic_loss_type = critic_loss_type



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
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.shared_actorcritic = shared_actorcritic

        self.experience_maker = NaiveExperienceMaker(
            actor,
            None,
            reward_model,
            initial_model,
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
            rm_type,
            actor_loss_type,
            self.generate_kwargs['max_new_tokens'],
            save_negdata=save_negdata,
            save_negdata_threshold=save_negdata_threshold,
        )
        self.experience_maker_neg_sampling = NaiveExperienceMaker(
            sampling_actor,
            critic,
            reward_model,
            initial_model,
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
            rm_type,
            actor_loss_type,
            self.generate_kwargs['max_new_tokens'],
            save_negdata=save_negdata,
            save_negdata_threshold=save_negdata_threshold,
        )
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)
        self.replay_buffer_neg_sampling = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

        from collections import defaultdict
        self.gradient_history = defaultdict(list)

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
            update_timesteps = 1
            num_rollouts_per_episodes = 1

        else:
            num_rollouts_per_episodes = (
                num_update_steps_per_episodes * args.train_batch_size // args.max_epochs // args.rollout_batch_size
            )
            update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch

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


        iwae_lbs_list = []
        iwae_ubs_list = []
        f_q_estimates_list = []
        g_q_estimates_list = []
        rewards_list = []
        kl_vals_list = []
        entropy_list = []


        # if true_posterior_samples is not None:
        #     n_seeds_f_q = true_posterior_samples.shape[0] // args.train_batch_size
        #     print(f"n_seeds_f_q: {n_seeds_f_q}")
        # rewards_list = []
        # kl_to_prior_list = []

        estimates_list = (f_q_estimates_list, rewards_list, kl_vals_list, entropy_list)

        custom_prompt = None
        if args.custom_single_prompt:
            raise NotImplementedError # Not yet tested
            # if 'TinyStories' in args.pretrain:
            #     prompt_text = 'Once upon a time, there was a'
            # elif 'gpt2' in args.pretrain:
            #     if args.rm_type == 'toy_rlhf':
            #         prompt_text = "Who is the greatest basketball player of all time?"
            #     else:
            #         raise NotImplementedError
            # else:
            #     raise NotImplementedError
            #
            # custom_prompt = [prompt_text] * args.rollout_batch_size
            # print("USING CUSTOM PROMPT")
            # print(len(custom_prompt))
            # start_episode = 0 # TODO later make sure this hasn't messed things up
            # steps = 0 # TODO later make sure this hasn't messed things up
            #
            #
            #
            # if not args.no_test_info:
            #     self.f_q_g_q_evaluation(args, f_q_estimates_list,
            #                             g_q_estimates_list, iwae_lbs_list,
            #                             iwae_ubs_list, prompt_text,
            #                             true_posterior_samples)
            #
            #
            #
            # for episode in range(start_episode, args.num_episodes):
            #
            #     print(f"Episode: {episode}", flush=True)
            #
            #     if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
            #         self.prompts_dataloader.sampler.set_epoch(
            #             episode, consumed_samples=0 if episode > start_episode else consumed_samples
            #         )
            #     pbar = tqdm(
            #         range(self.prompts_dataloader.__len__()),
            #         desc=f"Episode [{episode + 1}/{args.num_episodes}]",
            #         disable=not self.strategy.is_rank_0(),
            #     )
            #
            #
            #     if steps % update_timesteps == 0:
            #
            #         print(f"Step: {steps}")
            #
            #         global_steps = steps // update_timesteps
            #
            #         if self.bc_steps > 0:
            #             if global_steps >= self.bc_steps:
            #                 self.bc_coef = 0
            #
            #         num_twist_updates_to_do = args.update_steps_per_episode
            #         if args.exp_num_twist_updates:
            #             if episode == 0:
            #                 num_twist_updates_to_do = 2
            #             else:
            #                 num_twist_updates_to_do = 2 ** episode
            #
            #         # if self.shared_actorcritic:
            #         #     vhead_weight = torch.load(
            #         #         f"/h/zhaostep/twisted-smc-lm/vhead_weight_0.pt",
            #         #         weights_only=True)
            #         #     vhead_bias = torch.load(
            #         #         f"/h/zhaostep/twisted-smc-lm/vhead_bias_0.pt",
            #         #         weights_only=True)
            #         #
            #         #     self.actor.critic_head.weight.data = vhead_weight
            #         #     self.actor.critic_head.bias.data = vhead_bias
            #         #     # TODO REMOVE LATER DEBUG ONLY
            #
            #         # print(self.generate_kwargs)
            #         # print(self.generate_kwargs['attention_mask'])
            #         # 1/0
            #
            #         for update in range(num_twist_updates_to_do):
            #             experience = self.experience_maker.make_experience(
            #                 custom_prompt,
            #                 samples_per_prompt=args.duplicate_rollout_batch_by,
            #                 **self.generate_kwargs)
            #
            #             if update == 0:
            #                 # print prompt/answer ONCE per number of updates
            #                 output = self.tokenizer.batch_decode(
            #                     experience.sequences,
            #                     skip_special_tokens=True)
            #                 self.strategy.print(output[0])
            #
            #             self.replay_buffer.append(experience)
            #
            #             torch.cuda.empty_cache()
            #             # print("REPLAY BUFFER BEFORE NORMALIZATION")
            #             # print(self.replay_buffer.items)
            #             self.replay_buffer.normalize("advantages", self.strategy)
            #             # print("REPLAY BUFFER AFTER NORMALIZATION")
            #             # print(self.replay_buffer.items)
            #
            #             status = self.train(global_steps, custom_prompt=custom_prompt)
            #             self.replay_buffer.clear()
            #             torch.cuda.empty_cache()
            #
            #             if "kl" in status:
            #                 self.kl_ctl.update(status["kl"],
            #                                    args.rollout_batch_size)
            #             pbar.set_postfix(status)
            #
            #         steps = steps + 1
            #         global_steps = steps // update_timesteps
            #
            #         # logs/checkpoints
            #         client_states = {
            #             "consumed_samples": global_steps * args.rollout_batch_size}
            #         self.save_logs_and_checkpoints(args, global_steps, pbar,
            #                                        status, client_states)
            #
            #     if not args.no_test_info:
            #         self.f_q_g_q_evaluation(args, f_q_estimates_list,
            #                                 g_q_estimates_list, iwae_lbs_list,
            #                                 iwae_ubs_list, prompt_text,
            #                                 true_posterior_samples)
            #
            #     pbar.update()

        else:
            for episode in range(start_episode, args.num_episodes):
                if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                    self.prompts_dataloader.sampler.set_epoch(
                        episode, consumed_samples=0 if episode > start_episode else consumed_samples
                    )
                pbar = tqdm(
                    range(self.prompts_dataloader.__len__()),
                    desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                    disable=not self.strategy.is_rank_0(),
                )

                print("DATALOADER_HARMLESS")
                print(self.prompts_dataloader.sampler, flush=True)
                print(self.prompts_dataloader.__len__(), flush=True)

                for rand_prompts in self.prompts_dataloader:

                    print("rand_prompts_HARMLESS")
                    print(rand_prompts, flush=True)

                    # if not args.no_test_info:
                    #     if steps == 1: # do some test at the very beginning
                    #         self.test_info_multiprompt(args, rand_prompts, estimates_list)

                    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    #              profile_memory=True, record_shapes=True) as prof:

                    experience = self.experience_maker.make_experience(
                        rand_prompts,
                        samples_per_prompt=args.duplicate_rollout_batch_by,
                        **self.generate_kwargs
                    )

                    experience_neg = self.experience_maker_neg_sampling.make_experience(
                        rand_prompts,
                        samples_per_prompt=args.duplicate_rollout_batch_by,
                        **self.generate_kwargs
                    )

                    # print("PROFILE1")
                    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


                    # print prompt/answer in each update step
                    if steps % update_timesteps == 0:
                        output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)
                        self.strategy.print(output[0])
                    self.replay_buffer.append(experience)
                    self.replay_buffer_neg_sampling.append(experience_neg)

                    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    #              profile_memory=True, record_shapes=True) as prof:

                    if steps % update_timesteps == 0:
                        global_steps = steps // update_timesteps

                        torch.cuda.empty_cache()
                        self.replay_buffer.normalize("advantages", self.strategy)
                        self.replay_buffer_neg_sampling.normalize("advantages", self.strategy)

                        assert custom_prompt is None
                        status = self.train(global_steps, custom_prompt=custom_prompt)
                        self.replay_buffer.clear()
                        self.replay_buffer_neg_sampling.clear()
                        torch.cuda.empty_cache()

                        if "kl" in status:
                            self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                        pbar.set_postfix(status)

                        # logs/checkpoints
                        client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
                        self.save_logs_and_checkpoints(args, global_steps, pbar, status, client_states)

                        # if not args.no_test_info:
                        #     if steps % args.test_info_every == 0:
                        #         self.test_info_multiprompt(args, rand_prompts, estimates_list)

                    # print("PROFILE2")
                    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


                    pbar.update()
                    steps = steps + 1
        if args.custom_single_prompt:
            return iwae_lbs_list, iwae_ubs_list, f_q_estimates_list, g_q_estimates_list
        else:
            return estimates_list



    def train(self, global_steps=0, custom_prompt=None):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=self.shuffle_replay_buffer_sample,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            assert len(dataloader) == len(dataloader_neg)
            pbar = tqdm(
                zip(dataloader, dataloader_neg),  # Zip both dataloaders
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
                total=min(len(dataloader), len(dataloader_neg))  # Ensure tqdm gets a proper length
            )

            # pbar = tqdm(
            #     dataloader,
            #     desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
            #     disable=not self.strategy.is_rank_0(),
            # )
            # for experience in pbar:
            for experience, experience_neg in pbar:
                experience.to_device(device)
                experience_neg.to_device(device)
                status = self.training_step(experience, experience_neg, global_steps, custom_prompt=custom_prompt)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    if "critic_lr" in status:
                        short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean


    def training_step(self, experience: Experience, experience_neg: Experience, global_steps, custom_prompt=None) -> Dict[str, float]:
        status = {}
        if self.shared_actorcritic:
            raise NotImplementedError
        else:
            if global_steps > self.freezing_actor_steps:
                status = self.training_step_actor(experience, experience_neg, custom_prompt=custom_prompt)

            if self.critic is not None:
                raise NotImplementedError
                status.update(self.training_step_critic(experience, custom_prompt=custom_prompt))

        return status

    def training_step_actor(self, experience: Experience, experience_neg: Experience, custom_prompt=None) -> Dict[str, float]:
        if self.model_eval:
            self.actor.eval()
        else:
            self.actor.train()

        actor_loss, num_actions = self.get_actor_loss(experience, experience_neg, custom_prompt)

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

        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            raise NotImplementedError # not yet checked/fixed
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)


        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status

    def get_actor_loss(self, experience: Experience, experience_neg: Experience, custom_prompt=None):

        batch_size = experience.sequences.size(0)
        samples_per_prompt = self.args.duplicate_rollout_batch_by
        num_prompts = batch_size // samples_per_prompt

        print("inspection 03-29")
        print(experience.action_mask)
        print(experience.action_mask.shape)
        # print(experience.action_mask.size(1))
        print(experience.sequences)
        print(experience.sequences.shape)

        if self.actor_loss_type == "reinforce":
            action_log_probs = self.actor(
                experience.sequences, experience.action_mask.size(1),
                attention_mask=experience.attention_mask, return_output=False
            )

            # print(action_log_probs)
            # print(experience.action_log_probs)
            # print(experience.advantages)
            # print(experience.action_mask)

            actor_loss = self.actor_loss_fn(
                action_log_probs,
                experience.returns,
                action_mask=experience.action_mask,
                baseline_type="expectation",
            )

        elif self.actor_loss_type == "neg_training":
            action_log_probs = self.actor(
                experience.sequences, experience.action_mask.size(1),
                attention_mask=experience.attention_mask, return_output=False
            )

            action_log_probs_neg = self.actor(
                experience_neg.sequences, experience_neg.action_mask.size(1),
                attention_mask=experience_neg.attention_mask, return_output=False
            )

            actor_loss = self.actor_loss_fn(
                action_log_probs,
                action_log_probs_neg,
                experience.returns,
                sigma_over_q_importance_wgts=1/0, # TODO fill in with maybe the log p phi / q calculation. p has to be using what, using the base_actor I guess, whereas q is the proposal or sampling actor now.
                action_mask=experience.action_mask,
                baseline_type="expectation",
            )
        elif self.actor_loss_type == "neg_reinforce":
            action_log_probs = self.actor(
                experience.sequences, experience.action_mask.size(1),
                attention_mask=experience.attention_mask, return_output=False
            )

            action_log_probs_neg = self.actor(
                experience_neg.sequences, experience_neg.action_mask.size(1),
                attention_mask=experience_neg.attention_mask, return_output=False
            )

            actor_loss = self.actor_loss_fn(
                action_log_probs,
                action_log_probs_neg,
                experience.returns,
                experience_neg.returns,
                sigma_over_q_importance_wgts=1/0, # TODO fill in with maybe the log p phi / q calculation. p has to be using what, using the base_actor I guess, whereas q is the proposal or sampling actor now.
                action_mask=experience.action_mask,
                action_mask_neg=experience_neg.action_mask,
                baseline_type="expectation",
                baseline_type_neg="expectation",
            )

            # def forward(
            #     self,
            #     log_probs: torch.Tensor,
            #     log_probs_neg: torch.Tensor,
            #     rewards: torch.Tensor,
            #     rewards_neg: torch.Tensor,
            #     action_mask: Optional[torch.Tensor] = None,
            #     action_mask_neg: Optional[torch.Tensor] = None,
            #     baseline_type: Optional[str] = None,
            #     baseline_type_neg: Optional[str] = None,
            #     hardcoded_baseline: Optional[float] = None,
            #     hardcoded_baseline_neg: Optional[float] = None,
            # )

        else:
            raise NotImplementedError

        return actor_loss, num_actions


    def training_step_critic(self, experience: Experience, custom_prompt=None) -> Dict[str, float]:
        if self.model_eval:
            self.critic.eval()
        else:
            self.critic.train()

        # critic loss
        values, output = self.critic(
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
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def get_critic_loss(self, experience, values, custom_prompt=None):
        if self.critic_loss_type == "mse":
            critic_loss = self.critic_loss_fn(
                values,
                experience.values,
                experience.returns,
                action_mask=experience.action_mask,
            )
        elif self.critic_loss_type == "ctl":
            raise NotImplementedError # no longer tested
            num_actions = experience.action_mask.size(1)
            with torch.no_grad():
                base_action_log_probs = self.experience_maker.initial_model(
                    experience.sequences, num_actions,
                    experience.attention_mask)
            final_reward = self.experience_maker.compute_reward_no_kl(experience.sequences, experience.attention_mask)

            print("FINAL RETURN COMPARISON")
            print(final_reward)
            print(experience.returns[:, -1])
            print(experience.returns[:, -1] - final_reward)


            critic_loss = self.critic_loss_fn(
                values,
                final_reward=final_reward,
                action_mask=experience.action_mask,
                curr_log_probs=experience.action_log_probs,
                base_action_log_probs=base_action_log_probs
            )
        elif self.critic_loss_type == "mixed_ctl_mse":
            raise NotImplementedError # no longer tested

            num_actions = experience.action_mask.size(1)
            final_reward = self.reward_model(experience.sequences, experience.attention_mask)
            with torch.no_grad():
                base_action_log_probs = self.experience_maker.initial_model(
                    experience.sequences, num_actions,
                    experience.attention_mask)
            critic_loss = self.critic_loss_fn(
                values,
                experience.values,
                experience.returns,
                action_mask=experience.action_mask,
                curr_log_probs=experience.action_log_probs,
                base_action_log_probs=base_action_log_probs,
                final_reward=final_reward
            )
        elif self.critic_loss_type in ["sixo", "sixo_approxneg"]:
            raise NotImplementedError # no longer tested

            num_actions = experience.action_mask.size(1)
            with torch.no_grad():
                base_action_log_probs = self.experience_maker.initial_model(
                    experience.sequences, num_actions,
                    experience.attention_mask) # NOTE: for clarity, these are p(seqs), where seqs are generated according to the (twisted) proposal. This is used in the p phi / q calculation for positive samples
            # This is different from p(seqs) where seqs are generated according to p
            final_reward = self.reward_model(experience.sequences, experience.attention_mask)

            values_on_base_samples = None
            if self.critic_loss_type == "sixo":
                raise NotImplementedError # Not yet tested after changes
                base_action_mask, base_attention_mask, base_sequences = self.generate_base_seqs(custom_prompt)


                values_on_base_samples = self.critic(
                    base_sequences,
                    action_mask=base_action_mask,
                    attention_mask=base_attention_mask,
                    return_output=False,
                )

            critic_loss = self.critic_loss_fn(
                values,
                final_reward=final_reward,
                action_mask=experience.action_mask,
                curr_log_probs=experience.action_log_probs,
                base_action_log_probs=base_action_log_probs,
                values_on_base_samples=values_on_base_samples
            )
        else:
            raise NotImplementedError
        return critic_loss

    def generate_base_seqs_from_str_prompt(self, custom_prompt):
        self.initial_model.eval()
        inputs = self.experience_maker.tokenize_fn(custom_prompt, self.prompt_max_len,
                                                   device="cuda")

        base_sequences, base_attention_mask, base_action_mask = self.initial_model.generate(
            **inputs,
            **self.generate_kwargs)
        return base_action_mask, base_attention_mask, base_sequences

    def generate_base_seqs_from_torch_prompt(self, input_ids, attention_mask):
        self.initial_model.eval()

        base_sequences, base_attention_mask, base_action_mask = self.initial_model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            **self.generate_kwargs)
        return base_action_mask, base_attention_mask, base_sequences

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



        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):

        info_name_str = get_info_name_str(args)
        save_str = f"{info_name_str}"
        # save_str = f"PPOepochs{args.max_epochs}{eval_str}_lrschedule{args.lr_scheduler}_{lr_str}_criticloss{args.critic_loss_type}_{extra_str}_seed{args.seed}"

        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, f"{save_str}_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        if self.critic is not None:
            self.strategy.save_ckpt(
                self.critic, os.path.join(args.ckpt_path, f"{save_str}_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
            )
