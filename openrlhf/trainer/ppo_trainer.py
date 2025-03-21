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

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.loss import CTLLoss, MixedCTLValueLoss, SIXOLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.utils import get_info_name_str, tile_prompts

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer


class PPOTrainer(ABC):
    """
        Trainer for PPO algorithm.

    Args:
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


        self.actor_loss_type = actor_loss_type
        if self.actor_loss_type == "ppo":
            self.actor_loss_fn = PolicyLoss(eps_clip)
        elif self.actor_loss_type == "ctl":
            self.actor_loss_fn = CTLLoss()
        elif self.actor_loss_type == "sixo":
            self.actor_loss_fn = SIXOLoss()
        elif self.actor_loss_type == "sixo_approxneg":
            self.actor_loss_fn = SIXOLoss(approx_neg=True)
        else:
            raise NotImplementedError

        if self.actor_loss_type == "ppo":
            self.shuffle_replay_buffer_sample = True
        else:
            self.shuffle_replay_buffer_sample = False


        self.critic_loss_type = critic_loss_type
        if critic_loss_type == "mse":
            self.critic_loss_fn = ValueLoss(value_clip)
        elif critic_loss_type == "ctl":
            self.critic_loss_fn = CTLLoss()
        elif critic_loss_type == "mixed_ctl_mse":
            self.critic_loss_fn = MixedCTLValueLoss(clip_eps=value_clip, alpha=alpha)
        elif critic_loss_type == "sixo":
            self.critic_loss_fn = SIXOLoss()
        elif critic_loss_type == "sixo_approxneg":
            self.critic_loss_fn = SIXOLoss(approx_neg=True)
        else:
            raise NotImplementedError
        self.ptx_loss_fn = GPTLMLoss()

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
            self.generate_kwargs['max_new_tokens']
        )
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

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

        print("INSPECT")
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

        print("INSPECT2")
        print(steps)
        print(start_episode)
        print(consumed_samples)


        iwae_lbs_list = []
        iwae_ubs_list = []
        f_q_estimates_list = []
        g_q_estimates_list = []

        # if true_posterior_samples is not None:
        #     n_seeds_f_q = true_posterior_samples.shape[0] // args.train_batch_size
        #     print(f"n_seeds_f_q: {n_seeds_f_q}")
        # rewards_list = []
        # kl_to_prior_list = []

        custom_prompt = None
        if args.custom_single_prompt:
            if 'TinyStories' in args.pretrain:
                prompt_text = 'Once upon a time, there was a'
            elif 'gpt2' in args.pretrain:
                if args.rm_type == 'toy_rlhf':
                    prompt_text = "Who is the greatest basketball player of all time?"
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            custom_prompt = [prompt_text] * args.rollout_batch_size
            print("USING CUSTOM PROMPT")
            print(len(custom_prompt))
            start_episode = 0 # TODO later make sure this hasn't messed things up
            steps = 0 # TODO later make sure this hasn't messed things up



            if not args.no_test_info:
                self.f_q_g_q_evaluation(args, f_q_estimates_list,
                                        g_q_estimates_list, iwae_lbs_list,
                                        iwae_ubs_list, prompt_text,
                                        true_posterior_samples)



            for episode in range(start_episode, args.num_episodes):

                print(f"Episode: {episode}", flush=True)

                if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                    self.prompts_dataloader.sampler.set_epoch(
                        episode, consumed_samples=0 if episode > start_episode else consumed_samples
                    )
                pbar = tqdm(
                    range(self.prompts_dataloader.__len__()),
                    desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                    disable=not self.strategy.is_rank_0(),
                )


                if steps % update_timesteps == 0:

                    print(f"Step: {steps}")

                    global_steps = steps // update_timesteps

                    if self.bc_steps > 0:
                        if global_steps >= self.bc_steps:
                            self.bc_coef = 0

                    num_twist_updates_to_do = args.update_steps_per_episode
                    if args.exp_num_twist_updates:
                        if episode == 0:
                            num_twist_updates_to_do = 2
                        else:
                            num_twist_updates_to_do = 2 ** episode

                    # if self.shared_actorcritic:
                    #     vhead_weight = torch.load(
                    #         f"/h/zhaostep/twisted-smc-lm/vhead_weight_0.pt",
                    #         weights_only=True)
                    #     vhead_bias = torch.load(
                    #         f"/h/zhaostep/twisted-smc-lm/vhead_bias_0.pt",
                    #         weights_only=True)
                    #
                    #     self.actor.critic_head.weight.data = vhead_weight
                    #     self.actor.critic_head.bias.data = vhead_bias
                    #     # TODO REMOVE LATER DEBUG ONLY

                    # print(self.generate_kwargs)
                    # print(self.generate_kwargs['attention_mask'])
                    # 1/0

                    for update in range(num_twist_updates_to_do):
                        experience = self.experience_maker.make_experience(
                            custom_prompt,
                            samples_per_prompt=args.duplicate_rollout_batch_by,
                            **self.generate_kwargs)

                        if update == 0:
                            # print prompt/answer ONCE per number of updates
                            output = self.tokenizer.batch_decode(
                                experience.sequences,
                                skip_special_tokens=True)
                            self.strategy.print(output[0])

                        self.replay_buffer.append(experience)

                        torch.cuda.empty_cache()
                        # print("REPLAY BUFFER BEFORE NORMALIZATION")
                        # print(self.replay_buffer.items)
                        self.replay_buffer.normalize("advantages", self.strategy)
                        # print("REPLAY BUFFER AFTER NORMALIZATION")
                        # print(self.replay_buffer.items)

                        status = self.ppo_train(global_steps, custom_prompt=custom_prompt)
                        self.replay_buffer.clear()
                        torch.cuda.empty_cache()

                        if "kl" in status:
                            self.kl_ctl.update(status["kl"],
                                               args.rollout_batch_size)
                        pbar.set_postfix(status)

                    steps = steps + 1
                    global_steps = steps // update_timesteps

                    # logs/checkpoints
                    client_states = {
                        "consumed_samples": global_steps * args.rollout_batch_size}
                    self.save_logs_and_checkpoints(args, global_steps, pbar,
                                                   status, client_states)

                if not args.no_test_info:
                    self.f_q_g_q_evaluation(args, f_q_estimates_list,
                                            g_q_estimates_list, iwae_lbs_list,
                                            iwae_ubs_list, prompt_text,
                                            true_posterior_samples)

                pbar.update()

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

                print("DATALOADER")
                print(self.prompts_dataloader.sampler, flush=True)
                print(self.prompts_dataloader.__len__(), flush=True)

                for rand_prompts in self.prompts_dataloader:

                    print("rand_prompts")
                    print(rand_prompts, flush=True)

                    if not args.no_test_info:
                        if steps == 1: # do some test at the very beginning
                            self.test_info_multiprompt(args, rand_prompts, samples_per_prompt=args.duplicate_rollout_batch_by)

                    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    #              profile_memory=True, record_shapes=True) as prof:

                    experience = self.experience_maker.make_experience(
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

                    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    #              profile_memory=True, record_shapes=True) as prof:

                    if steps % update_timesteps == 0:
                        global_steps = steps // update_timesteps

                        torch.cuda.empty_cache()
                        self.replay_buffer.normalize("advantages", self.strategy)
                        assert custom_prompt is None
                        status = self.ppo_train(global_steps, custom_prompt=custom_prompt)
                        self.replay_buffer.clear()
                        torch.cuda.empty_cache()

                        if "kl" in status:
                            self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                        pbar.set_postfix(status)

                        # logs/checkpoints
                        client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
                        self.save_logs_and_checkpoints(args, global_steps, pbar, status, client_states)

                        if not args.no_test_info:
                            if steps % args.test_info_every == 0:
                                self.test_info_multiprompt(args, rand_prompts, samples_per_prompt=args.duplicate_rollout_batch_by)

                    # print("PROFILE2")
                    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


                    pbar.update()
                    steps = steps + 1

        return iwae_lbs_list, iwae_ubs_list, f_q_estimates_list, g_q_estimates_list

    def test_info_multiprompt(self, args, rand_prompts, samples_per_prompt: int = 1):
        print("prompts")
        print(rand_prompts)
        # expanded_prompts = tile_prompts(rand_prompts, samples_per_prompt) # this is done within the f_q estimate...

        f_qs, attention_mask, num_actions, q_seqs = self.f_q_estimate(
            args, rand_prompts)
        # print("f_qs")
        # print(f_qs)
        print(f"Avg F_q: {f_qs.mean()}")
        output = self.tokenizer.batch_decode(
            q_seqs,
            skip_special_tokens=True)
        print("seqs")
        print(output)
        print("seqs2")
        self.strategy.print(output[0])
        # self.f_q_g_q_evaluation(args, f_q_estimates_list,
        #                         g_q_estimates_list, iwae_lbs_list,
        #                         iwae_ubs_list, prompt_text,
        #                         true_posterior_samples)

    def f_q_g_q_evaluation(self, args, f_q_estimates_list, g_q_estimates_list,
                           iwae_lbs_list, iwae_ubs_list,
                           prompt_text, true_posterior_samples):
        # This function appends to f_q_estimates_list and g_q_estimates_list
        iwae_lbs = torch.zeros((self.n_seeds_f_q,))
        iwae_ubs = torch.zeros((self.n_seeds_f_q,))
        total_f_qs = None
        total_g_qs = None
        for i in range(self.n_seeds_f_q):
            custom_prompt_for_f_q = [prompt_text] * args.n_samples_for_f_q

            f_qs, attention_mask, num_actions, q_seqs = self.f_q_estimate(
                args, custom_prompt_for_f_q)
            print("Avg F_q Estimate (Learned Model)")
            print(f_qs.mean())
            print("IWAE Lower Bound Estimate (Learned Model)")
            iwae_lower_bound_estimate = torch.logsumexp(f_qs,
                                                        dim=0) - torch.log(
                torch.tensor(f_qs.shape[0]))
            print(iwae_lower_bound_estimate)
            iwae_lbs[i] = iwae_lower_bound_estimate.item()
            # # TODO load the posterior samples, pass through to get g_q estimate
            # if true_posterior_samples is not None:
            #     true_posterior_samples = true_posterior_samples.to(
            #         q_seqs.device)
            #     # TODO later account for the above possiblity
            eos_token_id = self.generate_kwargs["eos_token_id"]
            pad_token_id = self.generate_kwargs["pad_token_id"]

            if i == 0:
                assert true_posterior_samples is not None
                range_val = (math.ceil(
                    true_posterior_samples.shape[0] / args.n_samples_for_f_q))
                print(range_val)
                for j in range(range_val):
                    samples = true_posterior_samples[
                              j * args.n_samples_for_f_q: (j + 1) * args.n_samples_for_f_q]
                    if samples.shape[0] != 0:
                        print("G_q Estimates Learned Model")
                        # print(samples.shape)
                        # print(condition_twist_on_tokens.shape)
                        # print(condition_twist_on_tokens[i * n_samples_f_q: (i+1) * n_samples_f_q].shape)
                        attention_mask_g_q = (
                                samples.ne(eos_token_id) & samples.ne(
                                pad_token_id)).to(
                            dtype=torch.long)

                        g_qs = self.g_q_estimate(args, samples,
                                                 num_actions, attention_mask_g_q) # using the f_q mask would be wrong here.
                        # No attention mask could cause issues with padding TODO should investigate, but at least for my current experiments is not an issue

                        print(g_qs)
                        print("Avg G_q Estimate (Learned Model)")
                        print(g_qs.mean())

                        if total_g_qs is None:
                            total_g_qs = g_qs
                        else:
                            total_g_qs = torch.cat((total_g_qs, g_qs),
                                                   axis=0)
                            print("Total G_qs shape")
                            print(total_g_qs.shape)

                # g_qs = self.g_q_estimate(args, true_posterior_samples[
                #                                :q_seqs.shape[0]],
                #                          num_actions, attention_mask)
                # print("Avg G_q Estimate (Learned Model)")
                # print(g_qs.mean())

            if true_posterior_samples is not None:
                iwae_mixture_with_one_post = q_seqs.detach().clone()
                iwae_mixture_with_one_post[i] = true_posterior_samples[
                    i]  # To keep the conditioning tokens constant
                attention_mask_g_q = (
                    iwae_mixture_with_one_post.ne(eos_token_id) & iwae_mixture_with_one_post.ne(
                    pad_token_id)).to(
                    dtype=torch.long)
                iwae_ub_weights = self.g_q_estimate(args,
                                                    iwae_mixture_with_one_post,
                                                    num_actions,
                                                    attention_mask_g_q
                                                    )
                # No attention mask - using the f_q mask would be wrong here.
                # No attention mask could cause issues with padding TODO should investigate, but at least for my current experiments is not an issue

                print("IWAE Upper Bound Estimate (Learned Model)")
                iwae_upper_bound_estimate = torch.logsumexp(
                    iwae_ub_weights, dim=0) - torch.log(
                    torch.tensor(iwae_ub_weights.shape[0]))
                print(iwae_upper_bound_estimate)

                iwae_ubs[i] = iwae_upper_bound_estimate.item()

            if total_f_qs is None:
                total_f_qs = f_qs
                # total_rewards = rewards
                # total_kl_vals = kl_vals
            else:
                total_f_qs = torch.cat((total_f_qs, f_qs), axis=0)
                print("F_Q Shape")
                print(total_f_qs.shape)
                # total_rewards = torch.cat((total_rewards, rewards),
                #                           axis=0)
                # print(total_rewards.shape)
                # total_kl_vals = torch.cat((total_kl_vals, kl_vals),
                #                           axis=0)
                # print(total_kl_vals.shape)

            # if total_g_qs is None:
            #     total_g_qs = g_qs
            # else:
            #     total_g_qs = torch.cat((total_g_qs, g_qs),
            #                            axis=0)
            #     print("Total G_qs shape")
            #     print(total_g_qs.shape)

        iwae_lbs_list.append(iwae_lbs)
        iwae_ubs_list.append(iwae_ubs)
        # 1/0
        print("IWAE LB AND UB")
        print(iwae_lbs)
        print(iwae_ubs)
        print("IWAE LB AND UB LISTS")
        print(iwae_lbs_list)
        print(iwae_ubs_list)
        print("Shapes")
        print(total_g_qs.shape)
        print(total_f_qs.shape)


        # print(total_rewards.shape)
        # print(total_kl_vals.shape)
        if total_g_qs is not None:
            g_q_estimates_list.append(
                total_g_qs.cpu())  # Only one G_q estimate (over all the posterior samples)
        f_q_estimates_list.append(total_f_qs.cpu())

        # return f_q_estimates_list, g_q_estimates_list

    def f_q_estimate(self, args, batch_prompt):
        self.experience_maker.set_all_eval()
        batch_prompt = tile_prompts(batch_prompt, args.duplicate_rollout_batch_by)
        with torch.no_grad():
            if self.shared_actorcritic:
                action_log_probs, action_mask, attention_mask, num_actions, sequences, value = self.experience_maker.generate_seqs_and_get_logprobs(
                    batch_prompt, **self.generate_kwargs)
            else:
                action_log_probs, action_mask, attention_mask, num_actions, sequences = self.experience_maker.generate_seqs_and_get_logprobs(
                    batch_prompt, **self.generate_kwargs)
            action_log_probs = action_log_probs.float() # more precision
            log_q = action_log_probs.sum(dim=-1)
            # print(log_q.shape)
            # print(action_mask.shape)

            # print("ACTION LOG PROBS")
            # print(action_log_probs)
            # print(action_log_probs.shape)
            # 1/0

            log_tilde_sigma = self.eval_log_p_plus_log_phi(args, action_log_probs,
                                                           attention_mask,
                                                           num_actions,
                                                           sequences)

            f_qs = log_tilde_sigma - log_q
            print("f_q estimate details")
            print("log q")
            print(log_q)
            print("log tilde sigma")
            print(log_tilde_sigma)
            print("f_qs")
            print(f_qs)
            print(f_qs.shape)

        return f_qs, attention_mask, num_actions, sequences

    def eval_log_p_plus_log_phi(self, args, action_log_probs, attention_mask,
                                num_actions, sequences):
        # rewards_no_kl = self.experience_maker.compute_reward_no_kl(sequences,
        #                                                            attention_mask, multiply_by_beta=True)
        # print("log p phi eval")
        # print(rewards_no_kl)
        # Recall that we have p(s_1:T)p(toxic class | s_1:T)^beta which is also
        # = p(s_1:T)e^{beta log p(toxic class | s_1:T))
        # Now consider r = log p(toxic class | s_1:T)), then we have the RL setting, but we must have KL penalties
        # Also, with phi = e^{beta log p(toxic class | s_1:T)), log_phi is simply just beta log p(toxic class | s_1:T)
        # rewards_no_kl = rewards_no_kl.float() # more precision
        # log_phi = args.target_dist_beta * rewards_no_kl
        log_phi = self.experience_maker.compute_reward_no_kl(sequences, attention_mask, multiply_by_beta=True)
        # print(args.target_dist_beta)
        # print("log_phi")
        # print(log_phi)
        base_action_log_probs = self.experience_maker.initial_model(sequences,
                                                                    num_actions,
                                                                    attention_mask)
        base_action_log_probs = base_action_log_probs.float() # more precision

        log_p = base_action_log_probs.sum(dim=-1)
        print('P PHI INSPECTION')
        print(log_p)
        print(log_phi)
        log_tilde_sigma = log_p + log_phi
        return log_tilde_sigma



    def g_q_estimate(self, args, true_sigma_samples, num_actions, attention_mask, condition_twist_on_tokens=None):
        self.experience_maker.set_all_eval()
        sequences = true_sigma_samples
        with torch.no_grad():
            if self.shared_actorcritic:
                action_log_probs, _ = self.experience_maker.actor(sequences,
                                                               num_actions,
                                                               attention_mask)
            else:
                action_log_probs = self.experience_maker.actor(sequences, num_actions,
                                              attention_mask)
            action_log_probs = action_log_probs.float() # more precision
            log_q = action_log_probs.sum(dim=-1)
            log_tilde_sigma = self.eval_log_p_plus_log_phi(args, action_log_probs,
                                    attention_mask,
                                    num_actions, sequences)
            log_tilde_sigma = log_tilde_sigma.float() # more precision

        return log_tilde_sigma - log_q

    def ppo_train(self, global_steps=0, custom_prompt=None):
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

            # if self.shared_actorcritic:
            #     vhead_weight = torch.load(f"/h/zhaostep/twisted-smc-lm/vhead_weight_{epoch}.pt", weights_only=True)
            #     vhead_bias = torch.load(f"/h/zhaostep/twisted-smc-lm/vhead_bias_{epoch}.pt", weights_only=True)
            #
            #     print("OPENRLHF CRITIC HEAD WEIGHT")
            #     print(self.actor.critic_head.weight)
            #     print(self.actor.critic_head.bias)
            #
            #     print("TRL CRITIC HEAD WEIGHT")
            #     print(vhead_weight)
            #     print(vhead_bias)
            #
            #     self.actor.critic_head.weight.data = vhead_weight
            #     self.actor.critic_head.bias.data = vhead_bias
            #     # TODO REMOVE LATER DEBUG ONLY

            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps, custom_prompt=custom_prompt)

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

    def training_step_shared_actorcritic(self, experience: Experience, custom_prompt=None) -> Dict[str, float]:
        # self.actor.train()

        if self.bc_coef > 0:
            raise NotImplementedError # see training_step_actor

        if self.model_eval:
            self.actor.eval() # Turn off dropout (no batch norm in GPT2)
            # In our setup, do we want dropout? Not sure what the answer is, but perhaps an argument for why
            # we may not want dropout is: we don't really need additional regularization here
            # our objective is the RL with KL penalties, which at its optimum, achieves the twists we want
            # We don't need further regularization which is what would be provided by dropout
            # That is, in most ML use cases, we're worried about overoptimizing, overfitting, etc.
            # But here we don't have this problem - we want to go as hard as we can on the optimization
            # Since we already have the in built KL penalty regularizer
            # Now I guess you could argue that this dropout may still help avoid finding high reward areas of reward model or classifier that
            # aren't ACTUALLY high reward (e.g. a human would not consider them good)
            # But on the other hand, I'm probably not training to convergence anyway
            # So I already have a kind of early stopping implicit regularization
            # I probably don't need further regularization from dropout
            # Actually: turns out not to make a big difference for the separate actor/critic, and dropout does help
            # Whereas for the shared actor/critic makes no difference at all, which seems like a bug to me.
        else:
            self.actor.train()

        actor_loss, num_actions = self.get_actor_loss(experience, custom_prompt)

        # action_log_probs, values = self.actor(
        #     experience.sequences, num_actions,
        #     attention_mask=experience.attention_mask, return_output=False
        # )  # TODO later revert this and fix the above (return_output=True)
        #
        # # print("LOG PROBS")
        # # print(action_log_probs)
        #
        # # print("VALUES")
        # # print(values)
        #
        # actor_loss = self.actor_loss_fn(
        #     action_log_probs,
        #     experience.action_log_probs,
        #     experience.advantages,
        #     action_mask=experience.action_mask,
        # )

        # print("ACTOR LOSS")
        # print(actor_loss)

        critic_loss = self.get_critic_loss(experience, values, custom_prompt=custom_prompt)

        # print("CRITIC LOSS")
        # print(critic_loss)

        loss = actor_loss + self.vf_coef * critic_loss

        # print(self.vf_coef)
        # print("TOTAL LOSS")
        # print(loss)

        self.strategy.backward(loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(self.actor_optim, self.actor,
                                     self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model,
                                         self.ema_beta, "cpu")

        # status
        status = {"policy_loss": actor_loss.item(),
                  "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        if self.pretrain_dataloader is not None:
            raise NotImplementedError
            # status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() /
                    experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        status_val = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            # "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        status.update(status_val)

        return status


    def training_step(self, experience: Experience, global_steps, custom_prompt=None) -> Dict[str, float]:
        status = {}
        if self.shared_actorcritic:
            status = self.training_step_shared_actorcritic(experience, custom_prompt=custom_prompt)
        else:
            if global_steps > self.freezing_actor_steps:
                status = self.training_step_actor(experience, custom_prompt=custom_prompt)

            if self.critic is not None:
                status.update(self.training_step_critic(experience, custom_prompt=custom_prompt))

        return status

    def training_step_actor(self, experience: Experience, custom_prompt=None) -> Dict[str, float]:
        if self.model_eval:
            self.actor.eval()
        else:
            self.actor.train()

        actor_loss, num_actions = self.get_actor_loss(experience, custom_prompt)

        # mixtral
        if self.aux_loss:
            raise NotImplementedError
            # aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef

        if self.bc_coef > 0:
            print("DOING BEHAVIOUR CLONING")
            # TODO implement also for shared actorcritic using
            # action_log_probs, _ = self.experience_maker.actor(sigma_samples,
            #                                                            num_actions,
            #                                                            attention_mask_sigma_samples)
            # THEN TODO set up the args properly for this
            # Including the coefficient, maybe also having an option to take away the coefficient or anneal the coefficient over time
            # Maybe just take away the coefficient halfway through is an easy thing to implement
            # First try just with it all the way through, later try taking away halfway through

            # Attend to all tokens in exact sample
            attention_mask_sigma_samples = torch.ones_like(self.true_posterior_samples).to(
                dtype=torch.long)

            # print("--DEVICES--")
            # print(experience.sequences.device)
            # print(experience.attention_mask.device)
            # print(self.true_posterior_samples.device)
            # print(attention_mask_sigma_samples.device)
            # print(action_log_probs.device)
            # print(next(self.experience_maker.actor.parameters()).device)

            action_log_probs = self.experience_maker.actor(self.true_posterior_samples,
                                                           num_actions,
                                                           attention_mask_sigma_samples)
            action_log_probs = action_log_probs.float()  # more precision
            log_q = action_log_probs.sum(dim=-1)

            loss = loss + self.bc_coef * (- log_q.sum()) # loss is - log prob, so decrease loss is increase log p

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

    def get_actor_loss(self, experience, custom_prompt=None):
        # actor loss
        # action_log_probs, output = self.actor(
        #     experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
        # )
        num_actions = experience.action_mask.size(1)
        batch_size = experience.sequences.size(0)
        samples_per_prompt = self.args.duplicate_rollout_batch_by
        num_prompts = batch_size // samples_per_prompt

        print("inspection 03-15")
        print(experience.action_mask)
        print(experience.action_mask.shape)
        # print(experience.action_mask.size(1))
        print(experience.sequences)
        print(experience.sequences.shape)


        if self.actor_loss_type == "ppo":
            action_log_probs = self.actor(
                experience.sequences, num_actions,
                attention_mask=experience.attention_mask, return_output=False
            )  # TODO later revert this and fix the above (return_output=True)

            # print(action_log_probs)
            # print(experience.action_log_probs)
            # print(experience.advantages)
            # print(experience.action_mask)

            actor_loss = self.actor_loss_fn(
                action_log_probs,
                experience.action_log_probs,
                experience.advantages,
                action_mask=experience.action_mask,
            )

        elif self.actor_loss_type == "ctl":
            # Right now by using experience_maker sequences, this is essentially just twisted proposal samples
            # And we do CTL by reweighting those according to the twist values and tilde sigma values.

            base_action_log_probs = self.experience_maker.initial_model(
                experience.sequences, num_actions,
                experience.attention_mask)
            log_phi = self.experience_maker.compute_reward_no_kl(
                experience.sequences, experience.attention_mask, multiply_by_beta=True # beta multiplied for non-PPO formulations
            )
            # print("REWARD COMPARISON")
            # print(experience.returns[:, -1] - log_phi) # same
            log_psi = self.experience_maker.actor(experience.sequences, num_actions, experience.attention_mask, return_only_modulation=True)

            # print("ACTOR LOSS STUFF")
            # print(experience.action_log_probs.shape)
            # print(base_action_log_probs.shape)
            # print(log_psi.shape)
            # log_psi = log_psi[:, -num_actions:]
            # print(log_psi.shape)

            # Reshape tensors to group samples by prompt
            log_psi = log_psi.view(num_prompts, samples_per_prompt, -1)
            log_phi = log_phi.view(num_prompts, samples_per_prompt)
            exper_action_mask = experience.action_mask.view(num_prompts, samples_per_prompt, -1)
            exper_action_log_probs = experience.action_log_probs.view(num_prompts, samples_per_prompt, -1)
            base_action_log_probs = base_action_log_probs.view(num_prompts, samples_per_prompt, -1)

            print("CTL INSPECTION")
            print(experience.sequences.shape)
            print(experience.sequences)
            print(experience.attention_mask.shape)
            print(experience.attention_mask)
            print(exper_action_mask.shape)
            print(exper_action_mask)
            print(exper_action_log_probs.shape)
            print(exper_action_log_probs)

            print(log_psi.shape)
            print(log_psi)
            print(log_phi.shape)
            print(log_phi)

            print(base_action_log_probs.shape)
            print(base_action_log_probs)

            # Calculate loss for all groups at once
            actor_loss = self.actor_loss_fn(
                log_psi,  # shape: [num_prompts, samples_per_prompt, num_actions]
                log_phi,  # shape: [num_prompts, samples_per_prompt]
                exper_action_mask,
                exper_action_log_probs,
                base_action_log_probs,
                # reduce_mean_per_prompt=True
            )

        elif self.actor_loss_type in ["sixo", "sixo_approxneg"]:

            log_psi_on_base_samples = None

            base_action_log_probs = self.experience_maker.initial_model(
                experience.sequences, num_actions,
                experience.attention_mask)
            log_phi = self.experience_maker.compute_reward_no_kl(
                experience.sequences, experience.attention_mask, multiply_by_beta=True
                # beta multiplied for non-PPO formulations
            )

            # print(experience.sequences)
            # print(experience.sequences.shape)
            # print(num_actions)
            # print(experience.sequences[:, :-num_actions])
            # print(experience.sequences[:, :-num_actions].shape)
            #
            # print(experience.attention_mask)
            # print(experience.attention_mask.shape)
            # print(experience.attention_mask[:, :-num_actions])
            # print(experience.attention_mask[:, :-num_actions].shape)

            if self.actor_loss_type == "sixo":

                base_action_mask, base_attention_mask, base_sequences = self.generate_base_seqs_from_torch_prompt(
                    experience.sequences[:, :-num_actions],
                    experience.attention_mask[:, :-num_actions],
                )
                # TODO not yet tested on multiple different prompts (though I expect it should work)
                num_actions = base_action_mask.size(1)

                log_psi_on_base_samples = self.experience_maker.actor(base_sequences, num_actions, base_attention_mask,
                                                                      return_only_modulation=True)
                # log_psi_on_base_samples = log_psi_on_base_samples[:, -num_actions:]


            log_psi = self.experience_maker.actor(experience.sequences, num_actions, experience.attention_mask, return_only_modulation=True)
            # log_psi = log_psi[:, -num_actions:]

            # print("ACTOR LOSS STUFF")
            # print(experience.action_log_probs.shape)
            # print(base_action_log_probs.shape)
            # print(log_psi.shape)
            # print(log_psi.shape)

            # Reshape tensors to group samples by prompt
            log_psi = log_psi.view(num_prompts, samples_per_prompt, -1)
            log_phi = log_phi.view(num_prompts, samples_per_prompt)
            exper_action_mask = experience.action_mask.view(num_prompts, samples_per_prompt, -1)
            exper_action_log_probs = experience.action_log_probs.view(num_prompts, samples_per_prompt, -1)
            base_action_log_probs = base_action_log_probs.view(num_prompts, samples_per_prompt, -1)
            if log_psi_on_base_samples is not None:
                log_psi_on_base_samples = log_psi_on_base_samples.view(num_prompts, samples_per_prompt, -1)

            # Calculate loss for all groups at once
            actor_loss = self.actor_loss_fn(
                log_psi,  # shape: [num_prompts, samples_per_prompt, num_actions]
                log_phi,  # shape: [num_prompts, samples_per_prompt]
                exper_action_mask,
                exper_action_log_probs,
                base_action_log_probs,
                log_psi_on_base_samples,
                # reduce_mean_per_prompt=True
            )

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
            num_actions = experience.action_mask.size(1)
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
            num_actions = experience.action_mask.size(1)
            final_reward = self.reward_model(experience.sequences, experience.attention_mask)
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
            num_actions = experience.action_mask.size(1)
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
        # eval_str = ""
        # extra_str = ""
        # lr_str = f"actorlr{args.actor_learning_rate}_criticlr{args.critic_learning_rate}"
        # if args.actor_modulates_base:
        #     extra_str = "actormodbase"
        # if args.shared_actorcritic:
        #     lr_str = f"sharedactorcritic_lr{args.actor_learning_rate}"
        # if args.model_eval:
        #     eval_str = "_eval"
        #
        # if args.bc_coef > 0:
        #     lr_str += f"_bc{args.bc_coef}"
        #
        # if args.critic_loss_type == "mixed_ctl_mse":
        #     lr_str += f"_alpha{args.alpha}"

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
