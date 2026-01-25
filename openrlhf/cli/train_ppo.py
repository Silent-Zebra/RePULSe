import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import pickle

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.models.actor_custom import ActorCustom, ActorCritic
from openrlhf.models.model import INDICATOR_REWARD_EPS
from openrlhf.trainer import BasePPOTrainer
# from openrlhf.trainer.harmlessness_trainer import HarmlessnessTrainer # Have not tested this in a while
from openrlhf.trainer.combined_harmlessness_trainer import CombinedHarmlessnessTrainer

from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer, tile_prompts
from openrlhf.models.model import _get_reward_model_custom
from openrlhf.utils.utils import get_info_name_str, inspect_rewards_list, get_posterior_samples_filename
from openrlhf.models.utils import (
    normalize_bad_word_indices,
    get_next_token_log_probs,
    get_good_word_indices,
    extract_log_probs_at_position_based_on_token_indices,
)

from typing import List, Union, Tuple, Optional
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


bad_word_tokens_ids = [
                5089, 9372, 20654, 25617, 30998, 31699, 34094, 46733,
                21551, 40267, 7510, 16211, 20041, 32574, 41356,
                31030, 47209, 18185, 29836
            ]

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    neg_data = None
    if args.save_negdata:
        neg_data = set()

    static_initial_model = None
    if not args.do_harmlessness_training:
        # load weights for reference actor
        base_actor = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        # Freeze initial model
        # This doesn't make a difference normally, but for my CustomActor
        # where I take this in as an argument, then the optimizer will optimize these
        # and this has two undesired effects 1) I'm basically adding parameters/capacity to the twist architecture
        # 2) More problematic is that I would then be modifying the initial model, so things like KL to prior
        # and F_q and G_q evaluations are all messed up.
        for param in base_actor.parameters():
            param.requires_grad = False

    else:
        base_actor = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
        )

        # This model is used for KL to base (if not used in training, then used in evaluation) for the harmlessness training
        static_initial_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        for param in static_initial_model.parameters():
            param.requires_grad = False
        get_tokenizer(args.pretrain, static_initial_model.model, "left", strategy)

    get_tokenizer(args.pretrain, base_actor.model, "left", strategy)

    if args.shared_actorcritic:

        assert not args.actor_modulates_base # not yet implemented
        actor = ActorCritic(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        critic = None

    else:
        if args.actor_modulates_base:
            actor = ActorCustom(
                args.pretrain,
                initial_model=base_actor,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
                parameterization=args.parameterization,
                additional_sd_divider=args.additional_sd_divider,
                init_head_from_base=args.init_head_from_base
            )
        else:
            # configure model
            # load huggingface model
            actor = Actor(
                args.pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=True),
            )

        if args.no_critic:
            critic = None

        else:
            critic = get_llm_for_sequence_regression(
                args.critic_pretrain,
                "critic",
                normalize_reward=args.normalize_reward,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                lora_dropout=args.lora_dropout,
                ds_config=strategy.get_ds_train_config(is_actor=False),
                value_head_prefix=args.value_head_prefix,
                init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
            )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    reward_model, strip_question_chat_template_fn = get_reward_model(args, strategy)

    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    if critic is not None:
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    if critic is not None:
        get_tokenizer(args.critic_pretrain, critic, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    info_name_str = get_info_name_str(args)



    strategy.print(actor)
    if critic is not None:
        strategy.print(critic)

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        if critic is not None:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

    if args.actor_modulates_base:
        if args.parameterization == "modulation_model":
            actor_optim = strategy.create_optimizer(
                actor.model, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
            )
        else:
            actor_optim = strategy.create_optimizer(
                actor.modulation_head, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
            )
    else:
        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

    critic_optim = None
    if critic is not None:
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

    if args.do_harmlessness_training:
        base_actor_optim = strategy.create_optimizer(
            base_actor, lr=args.base_actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
        )

        strategy.print("BASE ACTOR OPTIM")
        strategy.print(base_actor_optim)
        
        # If base_actor learning rate is 0, only sample from sampling_actor (q)
        if abs(args.base_actor_learning_rate) < 1e-10:
            strategy.print("Base actor learning rate is 0. Setting neg_sample_only=True everywhere (only sampling from q).")
            args.neg_sample_only = True
        else:
            args.neg_sample_only = False

    pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)

    if not args.only_evaluate_on_neg_data:

        # prepare dataloader
        prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
        if args.pretrain_data:
            pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            pretrain_dataloader = None

    # configure scheduler
    num_update_steps_per_episodes = len(prompts_dataset) // args.train_batch_size * args.max_epochs
    num_update_steps_per_episodes = max(num_update_steps_per_episodes, 1) # ensure at least 1


    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        args.lr_scheduler,
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )
    if args.do_harmlessness_training:
        base_actor_scheduler = get_scheduler(
            args.lr_scheduler,
            base_actor_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

    critic_scheduler = None
    if critic_optim is not None:
        critic_scheduler = get_scheduler(
            args.lr_scheduler,
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )

    # Initialize coin flip networks if needed for separate_nn architecture
    coin_flip_trainable_network = None
    coin_flip_frozen_prior_network = None
    if (args.do_harmlessness_training and 
        getattr(args, 'exploration_bonus_sampling_actor', None) == "coin_flip" and
        getattr(args, 'coin_flip_architecture', 'linear_head_on_static_initial_base') == "separate_nn"):
        # Create trainable network (copy of actor structure)
        coin_flip_trainable_network = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
        )
        # Create frozen prior network (copy of actor structure)
        coin_flip_frozen_prior_network = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
        )
        # Freeze the frozen prior network
        for param in coin_flip_frozen_prior_network.parameters():
            param.requires_grad = False

    if args.do_harmlessness_training:
        # Seems like the strategy.prepare handles None gracefully, so no need for the explicit critic check
        if critic is not None:
            # prepare models/optimizers...
            prepared = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
                coin_flip_trainable_network,
                coin_flip_frozen_prior_network,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            (
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
                coin_flip_trainable_network,
                coin_flip_frozen_prior_network,
            ) = prepared
        else:
            prepared = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
                coin_flip_trainable_network,
                coin_flip_frozen_prior_network,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            (
                (actor, actor_optim, actor_scheduler),
                (base_actor, base_actor_optim, base_actor_scheduler),
                reward_model,
                static_initial_model,
                coin_flip_trainable_network,
                coin_flip_frozen_prior_network,
            ) = prepared

    else:

        if critic is not None:
            # prepare models/optimizers...
            (
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                reward_model,
                base_actor,
            ) = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                reward_model,
                base_actor,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
        else:
            (
                (actor, actor_optim, actor_scheduler),
                reward_model,
                base_actor,
            ) = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                reward_model,
                base_actor,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True,
                                     gradient_accumulation_steps=args.gradient_accumulation_steps)

    if args.do_harmlessness_training:
        consumed_samples = do_load_checkpoints(args, base_actor, None, strategy)
    else:
        consumed_samples = do_load_checkpoints(args, actor, critic, strategy)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_info_path, exist_ok=True)


    if args.only_evaluate_on_neg_data:

        _ = do_load_checkpoints(args, actor, critic, strategy)

        do_evaluate_on_neg_data(actor, args, strip_question_chat_template_fn, tokenizer, info_name_str, strategy)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        raise SystemExit(0)  # Finished



    if args.actor_learning_rate == 0:
        assert not args.shared_actorcritic # Should not do this with shared actor critic
        vf_coef = 100000 # Dummy value
    else:
        vf_coef = args.critic_learning_rate / args.actor_learning_rate

    true_posterior_samples = None
    if args.load_posterior_samples:

        strategy.print("Loading true posterior samples")

        true_posterior_samples_by_prompt_and_by_token = torch.load(f"{args.load_posterior_samples_name}")
        true_posterior_samples = \
            true_posterior_samples_by_prompt_and_by_token[
                0]
        true_posterior_samples = torch.tensor(
            true_posterior_samples,
            dtype=torch.int64)

        true_posterior_samples = true_posterior_samples.to(next(actor.parameters()).device)

    # Early exit for rejection sampling mode
    if args.rejection_sample_true_target_only:
        strategy.print("Running rejection sampling mode - skipping normal training")
        
        # Validation
        if args.rm_type != "rlhf":
            raise NotImplementedError(f"Rejection sampling currently only supports rm_type='rlhf', got '{args.rm_type}'")
        if args.reward_clamp is None:
            raise ValueError("--reward_clamp must be set (not None) when using --rejection_sample_true_target_only")
        if args.target_dist_beta is None:
            raise ValueError("--target_dist_beta must be set when using --rejection_sample_true_target_only")
        
        # Ensure we have prompts_dataloader set up (if not using custom prompt)
        prompts_dataloader = None
        if not args.new_custom_single_prompt:
            if args.only_evaluate_on_neg_data:
                raise ValueError("Cannot use --rejection_sample_true_target_only with --only_evaluate_on_neg_data when not using --new_custom_single_prompt")
            # Get prompts dataset
            pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)
            prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
        else:
            # For custom prompt, we'll handle it in the function
            strategy.print(f"Using custom prompt: {args.custom_prompt}")
        
        do_rejection_sampling_for_posterior_samples(
            args, base_actor, reward_model, tokenizer, strategy, prompts_dataloader
        )
        strategy.print("Rejection sampling complete. Exiting.")
        return

    estimates_list = None
    untrans_ret_list = None

    if args.do_harmlessness_training:
        harmlessness_trainer = CombinedHarmlessnessTrainer(
            sampling_target_updated_base=args.sampling_target_updated_base,
            strategy=strategy,
            base_actor=base_actor,
            sampling_actor=actor,
            base_critic=None,
            sampling_critic=None,
            reward_model=reward_model,
            static_initial_model=static_initial_model,
            ema_model=None,
            base_actor_optim=base_actor_optim,
            base_critic_optim=None,
            base_actor_scheduler=base_actor_scheduler,
            base_critic_scheduler=None,
            sampling_actor_optim=actor_optim,
            sampling_critic_optim=None,
            sampling_actor_scheduler=actor_scheduler,
            sampling_critic_scheduler=None, # TODO later setup if using PPO for this
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            tokenizer=tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            target_dist_beta=args.target_dist_beta,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # remote reward model
            remote_rm_url=args.remote_rm_url,
            shared_actorcritic=args.shared_actorcritic,
            vf_coef=vf_coef,
            model_eval=args.model_eval,
            threshold=args.threshold,
            reward_clamp=args.reward_clamp,
            n_seeds_f_q=args.n_seeds_f_q,
            rm_type=args.rm_type,
            bc_coef=args.bc_coef,
            bc_steps=args.bc_steps,
            true_posterior_samples=true_posterior_samples,
            sampling_actor_loss_type=args.actor_loss_type,
            sampling_critic_loss_type=args.critic_loss_type,
            base_actor_loss_type=args.harmlessness_training_loss_type,
            alpha=args.alpha,
            parameterization=args.parameterization,
            save_negdata=args.save_negdata,
            save_negdata_threshold=args.save_negdata_threshold,
            baseline_type=args.reinforce_baseline_type,
            hardcoded_baseline=args.reinforce_hardcoded_baseline,
            baseline_type_neg=args.neg_baseline_type,
            hardcoded_baseline_neg=args.neg_hardcoded_baseline,
            neg_data=neg_data,
            reward_transform=args.reward_transform,
            rew_trans_alpha=args.rew_trans_alpha,
            rew_trans_beta=args.rew_trans_beta,
            use_base_as_proposal=args.use_base_as_proposal,
            separate_reweighting_beta=args.separate_reweighting_beta,
            uniform_reweight=args.uniform_reweight,
            bad_word_tokens_ids=bad_word_tokens_ids,
            train_coin_flip_before=args.train_coin_flip_before,
            coin_flip_first_online=args.coin_flip_first_online,
            coin_flip_use_prioritization=args.coin_flip_use_prioritization,
            coin_flip_trainable_network=coin_flip_trainable_network,
            coin_flip_frozen_prior_network=coin_flip_frozen_prior_network,
        )


    else:
        trainer = get_base_ppo_trainer(actor, actor_optim, actor_scheduler, args, base_actor, critic, critic_optim,
                                       critic_scheduler, ema_model, neg_data, reward_model, strategy, tokenizer,
                                       true_posterior_samples, vf_coef)

    # Lists for base_actor (p) results
    total_log_prob_bad_list_base = []
    individual_bad_word_log_probs_t0_list_base = []
    individual_bad_word_log_probs_t1_list_base = []
    individual_bad_word_log_probs_combined_list_base = []
    # Lists for sampling_actor (q) results
    total_log_prob_bad_list_sampling = []
    individual_bad_word_log_probs_t0_list_sampling = []
    individual_bad_word_log_probs_t1_list_sampling = []
    individual_bad_word_log_probs_combined_list_sampling = []
    # Lists for threshold-based bad word calculations (base actor)
    total_log_prob_bad_list_base_threshold = []
    individual_bad_word_log_probs_t0_list_base_threshold = []
    individual_bad_word_log_probs_t1_list_base_threshold = []
    individual_bad_word_log_probs_combined_list_base_threshold = []
    # Lists for threshold-based bad word calculations (sampling actor)
    total_log_prob_bad_list_sampling_threshold = []
    individual_bad_word_log_probs_t0_list_sampling_threshold = []
    individual_bad_word_log_probs_t1_list_sampling_threshold = []
    individual_bad_word_log_probs_combined_list_sampling_threshold = []
    # Keep old names for backward compatibility when not doing harmlessness training
    total_log_prob_bad_list = total_log_prob_bad_list_base
    individual_bad_word_log_probs_t0_list = individual_bad_word_log_probs_t0_list_base
    individual_bad_word_log_probs_t1_list = individual_bad_word_log_probs_t1_list_base
    individual_bad_word_log_probs_combined_list = individual_bad_word_log_probs_combined_list_base
    total_log_prob_bad_list_threshold = total_log_prob_bad_list_base_threshold
    individual_bad_word_log_probs_t0_list_threshold = individual_bad_word_log_probs_t0_list_base_threshold
    individual_bad_word_log_probs_t1_list_threshold = individual_bad_word_log_probs_t1_list_base_threshold
    individual_bad_word_log_probs_combined_list_threshold = individual_bad_word_log_probs_combined_list_base_threshold
    total_kl_sigma_q_list = []
    total_kl_q_sigma_epsq_p_list = []
    diff_by_bad_word_case1_list = []  # List of dicts: {bad_word_id: sum_diff_case1}
    diff_by_bad_word_case2_list = []  # List of dicts: {bad_word_id: sum_diff_case2}
    diff_by_bad_word_list = []  # List of dicts: {bad_word_id: total_sum_diff}
    max_q_exceeds_list = []  # List of tuples: (diff, t0_token, t1_token, q_val, sigma_val)
    max_sigma_exceeds_list = []  # List of tuples: (diff, t0_token, t1_token, q_val, sigma_val)
    rew_over_time_list_base = []
    untrans_ret_over_time_list_base = []
    rew_over_time_list_sampling = []
    untrans_ret_over_time_list_sampling = []
    bonus_vals_over_time_list_sampling = []  # TODO: Add support for base_actor bonus tracking
    
    # Lists for analytic_calc results
    total_kl_sigma_q_list_analytic = []
    total_kl_q_sigma_list_analytic = []
    metrics_list_analytic = []  # List of metrics dicts
    
    # Precompute toxicity scores once at the beginning if using analytic_calc or analytic_bad_word_calc
    precomputed_toxicity_scores = None
    bad_word_tokens_ids_threshold = None
    if args.analytic_calc or args.analytic_bad_word_calc:
        strategy.print("Precomputing toxicity scores for all tokens...")
        prompt = args.custom_prompt  # Define prompt for analytic calculations
        precomputed_toxicity_scores = precompute_toxicity_scores_for_all_tokens(
            reward_model=reward_model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            batch_size=args.analytic_batch_size,
        )
        strategy.print(f"Precomputed toxicity scores shape: {precomputed_toxicity_scores.shape}")
        strategy.print(f"Toxicity scores range: [{precomputed_toxicity_scores.min().item():.4f}, {precomputed_toxicity_scores.max().item():.4f}]")
        
        # Compute threshold-based bad word list if analytic_bad_word_calc is enabled
        if args.analytic_bad_word_calc:
            bad_word_tokens_ids_threshold = torch.where(precomputed_toxicity_scores < args.threshold)[0].cpu().tolist()
            strategy.print(f"Threshold-based bad word list (reward < {args.threshold}): {bad_word_tokens_ids_threshold}")
            strategy.print(f"Number of tokens with reward < {args.threshold}: {len(bad_word_tokens_ids_threshold)}")

    # Fit steps is kind of like a chunk for how many points we want to track progress; do x harmlessness training steps each fit step
    for fit_step in range(args.fit_steps):
        prompt = args.custom_prompt  # Define prompt for analytic calculations
        if fit_step == 0 and args.analytic_bad_word_calc:
            
            precomputed_p = None
            precomputed_q = None
            if args.do_harmlessness_training:
                # For harmlessness training, actor is the sampling_actor (q) and base_actor is p

                # Calculate for both sampling_actor (q) and base_actor (p); only for the first step, do this for both
                # regardless of neg_sample_only or not. Because I want the values for p once at the start, just for reference
                precomputed_q = do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                          total_log_prob_bad_list_sampling, individual_bad_word_log_probs_t0_list_sampling,
                                          individual_bad_word_log_probs_t1_list_sampling, individual_bad_word_log_probs_combined_list_sampling,
                                          actor_to_test=actor, generate_max_len=args.generate_max_len)
                precomputed_p = do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                          total_log_prob_bad_list_base, individual_bad_word_log_probs_t0_list_base,
                                          individual_bad_word_log_probs_t1_list_base, individual_bad_word_log_probs_combined_list_base,
                                          actor_to_test=base_actor, generate_max_len=args.generate_max_len)
                # Parallel calculation with threshold-based bad word list
                if bad_word_tokens_ids_threshold is not None and len(bad_word_tokens_ids_threshold) > 0:
                    do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids_threshold, base_actor, prompt, tokenizer,
                                              total_log_prob_bad_list_sampling_threshold, individual_bad_word_log_probs_t0_list_sampling_threshold,
                                              individual_bad_word_log_probs_t1_list_sampling_threshold, individual_bad_word_log_probs_combined_list_sampling_threshold,
                                              actor_to_test=actor, generate_max_len=args.generate_max_len)
                    do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids_threshold, base_actor, prompt, tokenizer,
                                              total_log_prob_bad_list_base_threshold, individual_bad_word_log_probs_t0_list_base_threshold,
                                              individual_bad_word_log_probs_t1_list_base_threshold, individual_bad_word_log_probs_combined_list_base_threshold,
                                              actor_to_test=base_actor, generate_max_len=args.generate_max_len)
            else:
                # For non-harmlessness training, just use the standard actor
                do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                          total_log_prob_bad_list, individual_bad_word_log_probs_t0_list,
                                          individual_bad_word_log_probs_t1_list, individual_bad_word_log_probs_combined_list,
                                          generate_max_len=args.generate_max_len)
                # Parallel calculation with threshold-based bad word list
                if bad_word_tokens_ids_threshold is not None and len(bad_word_tokens_ids_threshold) > 0:
                    do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids_threshold, base_actor, prompt, tokenizer,
                                              total_log_prob_bad_list_threshold, individual_bad_word_log_probs_t0_list_threshold,
                                              individual_bad_word_log_probs_t1_list_threshold, individual_bad_word_log_probs_combined_list_threshold,
                                              generate_max_len=args.generate_max_len)
            
            if args.do_harmlessness_training:
                if "indicator" in args.rm_type: 
                    _, _, diff_by_bad_word_case1, diff_by_bad_word_case2, diff_by_bad_word, max_q_exceeds, max_sigma_exceeds = \
                        calculate_analytic_kl_indicator_bad_words_both_directions(
                            model_p_for_target=base_actor.model,
                            model_q=actor.model,
                            tokenizer=tokenizer,
                            prompt_text=prompt,
                            bad_word_indices=bad_word_tokens_ids,
                            batch_size=args.analytic_batch_size,
                            total_kl_sigma_q_list=total_kl_sigma_q_list,
                            total_kl_q_sigma_epsq_p_list=total_kl_q_sigma_epsq_p_list,
                            precomputed_p=precomputed_p,
                            precomputed_q=precomputed_q,
                        )
                    diff_by_bad_word_case1_list.append(diff_by_bad_word_case1)
                    diff_by_bad_word_case2_list.append(diff_by_bad_word_case2)
                    diff_by_bad_word_list.append(diff_by_bad_word)
                    max_q_exceeds_list.append(max_q_exceeds)
                    max_sigma_exceeds_list.append(max_sigma_exceeds)
        
        # Analytic calculation for single token with toxicity model
        if args.analytic_calc:
            prompt = args.custom_prompt  # Define prompt for analytic calculations
            do_analytic_kl_calc(
                base_actor=base_actor,
                actor=actor,
                args=args,
                tokenizer=tokenizer,
                prompt=prompt,
                precomputed_toxicity_scores=precomputed_toxicity_scores,
                total_kl_sigma_q_list_analytic=total_kl_sigma_q_list_analytic,
                total_kl_q_sigma_list_analytic=total_kl_q_sigma_list_analytic,
                metrics_list_analytic=metrics_list_analytic,
            )

        if args.do_harmlessness_training:
            strategy.print("-----HARMLESSNESS TRAINING-----")
            # Do the harmlessness training: combined now (1 set of samples for both the base_actor and sampling_actor updates)
            if args.harmlessness_training_num_episodes > 0:
                # assert args.num_episodes == 1  # Right now only supports 1 twist/proposal update per base_actor update
                estimates_list = harmlessness_trainer.fit(
                    args, prompts_dataloader, pretrain_dataloader, consumed_samples,
                    num_update_steps_per_episodes, true_posterior_samples
                )
        else:
            if args.num_episodes > 0:
                estimates_list = trainer.fit(
                    args, prompts_dataloader, pretrain_dataloader, consumed_samples,
                    num_update_steps_per_episodes, true_posterior_samples
                )

        rewards_list = None
        rewards_list_sampling = None
        untrans_ret_list_sampling = None

        if estimates_list is not None:
            # Unpack the base estimates_list (always returned)
            if args.do_harmlessness_training:
                # CombinedHarmlessnessTrainer format: (rewards_list, kl_vals_list, entropy_list, untrans_ret_list, rewards_list_sampling, untrans_ret_list_sampling, bonus_vals_list_sampling) = 7 elements
                # With f_q_g_q_eval: + (f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list) = 11 elements total
                if len(estimates_list) == 11:
                    # New format with f_q_g_q_eval: 7 base + 4 f_q/g_q/iwae
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list, rewards_list_sampling, untrans_ret_list_sampling, bonus_vals_list_sampling, f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list = estimates_list
                elif len(estimates_list) == 7:
                    # Base format without f_q_g_q_eval: 7 elements
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list, rewards_list_sampling, untrans_ret_list_sampling, bonus_vals_list_sampling = estimates_list
                    f_q_estimates_list = None
                    g_q_estimates_list = None
                    iwae_lbs_list = None
                    iwae_ubs_list = None
                else:
                    # Old format (6 elements without bonus)
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list, rewards_list_sampling, untrans_ret_list_sampling = estimates_list
                    bonus_vals_list_sampling = None
                    f_q_estimates_list = None
                    g_q_estimates_list = None
                    iwae_lbs_list = None
                    iwae_ubs_list = None
            else:
                # BasePPOTrainer format: (rewards_list, kl_vals_list, entropy_list, untrans_ret_list) = 4 elements
                # With f_q_g_q_eval: + (f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list) = 8 elements total
                if len(estimates_list) == 8:
                    # Base format with f_q_g_q_eval: 4 base + 4 f_q/g_q/iwae
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list, f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list = estimates_list
                    rewards_list_sampling = None
                    untrans_ret_list_sampling = None
                    bonus_vals_list_sampling = None
                else:
                    # Base format without f_q_g_q_eval: 4 elements
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list = estimates_list
                    rewards_list_sampling = None
                    untrans_ret_list_sampling = None
                    bonus_vals_list_sampling = None
                    f_q_estimates_list = None
                    g_q_estimates_list = None
                    iwae_lbs_list = None
                    iwae_ubs_list = None

            # Save f_q/g_q/iwae stuff separately (only if f_q_g_q_eval was done and lists are not empty)
            if args.f_q_g_q_eval and f_q_estimates_list is not None and len(f_q_estimates_list) > 0:
                print("FINAL RESULTS IWAE LB LIST", flush=True)
                print(iwae_lbs_list)
                print("FINAL RESULTS IWAE UB LIST", flush=True)
                print(iwae_ubs_list)
                print("FINAL RESULTS F_Q", flush=True)
                print(f_q_estimates_list)
                print("FINAL RESULTS G_Q", flush=True)
                print(g_q_estimates_list)
                print("SAVING F_Q/G_Q/IWAE RESULTS", flush=True)

                target_to_save = (
                    f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list
                )
                save_str = f"{args.save_info_path}/f_q_g_q_iwae_bounds_OpenRLHF_{info_name_str}"
                torch.save(target_to_save, save_str)

            # Save the base metrics separately (always saved if not empty)
            if not args.neg_sample_only: # This stuff records it for p (base actor), so if skipping training p, this stuff will be empty
                print("FINAL RESULTS REWARD", flush=True)
                print(rewards_list)
                print("FINAL RESULTS KL TO PRIOR", flush=True)
                print(kl_vals_list)
                print("FINAL RESULTS ENTROPY", flush=True)
                print(entropy_list)
                print("FINAL RESULTS UNTRANSFORMED RETURN (Including KL)", flush=True)
                print(untrans_ret_list)
                print("SAVING BASE METRICS", flush=True)

                target_to_save = (
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list
                )
                save_str = f"{args.save_info_path}/rew_kltoprior_ent_untransret_{info_name_str}"
                torch.save(target_to_save, save_str)

                inspect_rewards_list(rewards_list)

            # Save sampling metrics for harmlessness training (if available)
            if args.do_harmlessness_training and rewards_list_sampling is not None and len(rewards_list_sampling) > 0:
                print("FINAL RESULTS SAMPLING REWARD", flush=True)
                print(rewards_list_sampling)
                print("FINAL RESULTS SAMPLING UNTRANSFORMED RETURN", flush=True)
                print(untrans_ret_list_sampling)
                if bonus_vals_list_sampling is not None and len(bonus_vals_list_sampling) > 0:
                    print("FINAL RESULTS SAMPLING BONUS", flush=True)
                    print(bonus_vals_list_sampling)
                print("SAVING SAMPLING METRICS", flush=True)

                if bonus_vals_list_sampling is not None:
                    target_to_save = (
                        rewards_list_sampling, untrans_ret_list_sampling, bonus_vals_list_sampling
                    )
                else:
                    target_to_save = (
                        rewards_list_sampling, untrans_ret_list_sampling
                    )
                save_str = f"{args.save_info_path}/rew_untransret_sampling_{info_name_str}"
                torch.save(target_to_save, save_str)

        if args.save_negdata:
            strategy.print("SAVING NEG DATA")
            strategy.print(len(neg_data))
            # Save to file
            with open(f"{args.save_path}/neg_data_{info_name_str}_thr{args.save_negdata_threshold}.pkl", "wb") as f:
                pickle.dump(neg_data, f)

        if args.analytic_bad_word_calc:
            precomputed_p = None
            precomputed_q = None
            if args.do_harmlessness_training:
                # For harmlessness training, actor is the sampling_actor (q) and base_actor is p
                if args.neg_sample_only:
                    # Only calculate for sampling_actor (q)
                    precomputed_q = do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                              total_log_prob_bad_list_sampling, individual_bad_word_log_probs_t0_list_sampling,
                                              individual_bad_word_log_probs_t1_list_sampling, individual_bad_word_log_probs_combined_list_sampling,
                                              actor_to_test=actor, generate_max_len=args.generate_max_len)
                    # Parallel calculation with threshold-based bad word list
                    if bad_word_tokens_ids_threshold is not None and len(bad_word_tokens_ids_threshold) > 0:
                        do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids_threshold, base_actor, prompt, tokenizer,
                                                  total_log_prob_bad_list_sampling_threshold, individual_bad_word_log_probs_t0_list_sampling_threshold,
                                                  individual_bad_word_log_probs_t1_list_sampling_threshold, individual_bad_word_log_probs_combined_list_sampling_threshold,
                                                  actor_to_test=actor, generate_max_len=args.generate_max_len)
                else:
                    # Calculate for both sampling_actor (q) and base_actor (p)
                    precomputed_q = do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                              total_log_prob_bad_list_sampling, individual_bad_word_log_probs_t0_list_sampling,
                                              individual_bad_word_log_probs_t1_list_sampling, individual_bad_word_log_probs_combined_list_sampling,
                                              actor_to_test=actor, generate_max_len=args.generate_max_len)
                    precomputed_p = do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                              total_log_prob_bad_list_base, individual_bad_word_log_probs_t0_list_base,
                                              individual_bad_word_log_probs_t1_list_base, individual_bad_word_log_probs_combined_list_base,
                                              actor_to_test=base_actor, generate_max_len=args.generate_max_len)
                    # Parallel calculation with threshold-based bad word list
                    if bad_word_tokens_ids_threshold is not None and len(bad_word_tokens_ids_threshold) > 0:
                        do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids_threshold, base_actor, prompt, tokenizer,
                                                  total_log_prob_bad_list_sampling_threshold, individual_bad_word_log_probs_t0_list_sampling_threshold,
                                                  individual_bad_word_log_probs_t1_list_sampling_threshold, individual_bad_word_log_probs_combined_list_sampling_threshold,
                                                  actor_to_test=actor, generate_max_len=args.generate_max_len)
                        do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids_threshold, base_actor, prompt, tokenizer,
                                                  total_log_prob_bad_list_base_threshold, individual_bad_word_log_probs_t0_list_base_threshold,
                                                  individual_bad_word_log_probs_t1_list_base_threshold, individual_bad_word_log_probs_combined_list_base_threshold,
                                                  actor_to_test=base_actor, generate_max_len=args.generate_max_len)
            else:
                # For non-harmlessness training, just use the standard actor
                do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer,
                                          total_log_prob_bad_list, individual_bad_word_log_probs_t0_list,
                                          individual_bad_word_log_probs_t1_list, individual_bad_word_log_probs_combined_list,
                                          generate_max_len=args.generate_max_len)
                # Parallel calculation with threshold-based bad word list
                if bad_word_tokens_ids_threshold is not None and len(bad_word_tokens_ids_threshold) > 0:
                    do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids_threshold, base_actor, prompt, tokenizer,
                                              total_log_prob_bad_list_threshold, individual_bad_word_log_probs_t0_list_threshold,
                                              individual_bad_word_log_probs_t1_list_threshold, individual_bad_word_log_probs_combined_list_threshold,
                                              generate_max_len=args.generate_max_len)
            
            if args.do_harmlessness_training:
                if "indicator" in args.rm_type:
                    _, _, diff_by_bad_word_case1, diff_by_bad_word_case2, diff_by_bad_word, max_q_exceeds, max_sigma_exceeds = \
                        calculate_analytic_kl_indicator_bad_words_both_directions(
                            model_p_for_target=base_actor.model,
                            model_q=actor.model,
                            tokenizer=tokenizer,
                            prompt_text=prompt,
                            bad_word_indices=bad_word_tokens_ids,
                            batch_size=args.analytic_batch_size,
                            total_kl_sigma_q_list=total_kl_sigma_q_list,
                            total_kl_q_sigma_epsq_p_list=total_kl_q_sigma_epsq_p_list,
                            precomputed_p=precomputed_p,
                            precomputed_q=precomputed_q,
                        )
                    diff_by_bad_word_case1_list.append(diff_by_bad_word_case1)
                    diff_by_bad_word_case2_list.append(diff_by_bad_word_case2)
                    diff_by_bad_word_list.append(diff_by_bad_word)
                    max_q_exceeds_list.append(max_q_exceeds)
                    max_sigma_exceeds_list.append(max_sigma_exceeds)

        if not args.neg_sample_only:
            if rewards_list is not None:
                rewards_tensor = torch.tensor(rewards_list)
                if fit_step == 0:
                    rew_over_time_list_base.append(rewards_tensor[0].item()) # Get value at start of training
                rew_over_time_list_base.append(rewards_tensor[-1].item())

            if untrans_ret_list is not None:
                untrans_ret_tensor = torch.tensor(untrans_ret_list)
                if fit_step == 0:
                    untrans_ret_over_time_list_base.append(untrans_ret_tensor[0].item()) # Get value at start of training
                untrans_ret_over_time_list_base.append(untrans_ret_tensor[-1].item())
        
        # Track sampling actor rewards
        if args.do_harmlessness_training and rewards_list_sampling is not None:
            if len(rewards_list_sampling) > 0:
                rewards_tensor_sampling = torch.tensor(rewards_list_sampling)
                if fit_step == 0:
                    rew_over_time_list_sampling.append(rewards_tensor_sampling[0].item()) # Get value at start of training
                rew_over_time_list_sampling.append(rewards_tensor_sampling[-1].item())
            
            if untrans_ret_list_sampling is not None and len(untrans_ret_list_sampling) > 0:
                untrans_ret_tensor_sampling = torch.tensor(untrans_ret_list_sampling)
                if fit_step == 0:
                    untrans_ret_over_time_list_sampling.append(untrans_ret_tensor_sampling[0].item()) # Get value at start of training
                untrans_ret_over_time_list_sampling.append(untrans_ret_tensor_sampling[-1].item())
        
        # Track sampling actor exploration bonus values
        # TODO: Add support for base_actor bonus tracking
        if args.do_harmlessness_training and bonus_vals_list_sampling is not None:
            if len(bonus_vals_list_sampling) > 0:
                bonus_vals_tensor_sampling = torch.tensor(bonus_vals_list_sampling)
                if fit_step == 0:
                    bonus_vals_over_time_list_sampling.append(bonus_vals_tensor_sampling[0].item()) # Get value at start of training
                bonus_vals_over_time_list_sampling.append(bonus_vals_tensor_sampling[-1].item())

    # Calculate KL divergence one more time after training loop to get 51st value
    # (matching the 51 reward/return values: initial + 50 from loop)
    if args.analytic_calc:
        prompt = args.custom_prompt  # Define prompt for analytic calculations
        do_analytic_kl_calc(
            base_actor=base_actor,
            actor=actor,
            args=args,
            tokenizer=tokenizer,
            prompt=prompt,
            precomputed_toxicity_scores=precomputed_toxicity_scores,
            total_kl_sigma_q_list_analytic=total_kl_sigma_q_list_analytic,
            total_kl_q_sigma_list_analytic=total_kl_q_sigma_list_analytic,
            metrics_list_analytic=metrics_list_analytic,
        )

    if args.analytic_bad_word_calc:
        if args.do_harmlessness_training:
            # Save both base_actor and sampling_actor results separately
            save_str = f"{args.save_info_path}/analyticlogprob_rewsample_base_{info_name_str}"
            torch.save((total_log_prob_bad_list_base, individual_bad_word_log_probs_t0_list_base,
                       individual_bad_word_log_probs_t1_list_base, individual_bad_word_log_probs_combined_list_base,
                       rew_over_time_list_base, untrans_ret_over_time_list_base,
                       total_log_prob_bad_list_base_threshold, individual_bad_word_log_probs_t0_list_base_threshold,
                       individual_bad_word_log_probs_t1_list_base_threshold, individual_bad_word_log_probs_combined_list_base_threshold), save_str)
            print("Base actor (p) results:")
            print(total_log_prob_bad_list_base)
            print(individual_bad_word_log_probs_t0_list_base)
            print(individual_bad_word_log_probs_t1_list_base)
            print(individual_bad_word_log_probs_combined_list_base)
            print("Base actor (p) threshold-based results:")
            print(total_log_prob_bad_list_base_threshold)
            print(individual_bad_word_log_probs_t0_list_base_threshold)
            print(individual_bad_word_log_probs_t1_list_base_threshold)
            print(individual_bad_word_log_probs_combined_list_base_threshold)
            
            save_str = f"{args.save_info_path}/analyticlogprob_rewsample_sampling_{info_name_str}"
            torch.save((total_log_prob_bad_list_sampling, individual_bad_word_log_probs_t0_list_sampling,
                       individual_bad_word_log_probs_t1_list_sampling, individual_bad_word_log_probs_combined_list_sampling,
                       rew_over_time_list_sampling, untrans_ret_over_time_list_sampling, bonus_vals_over_time_list_sampling,
                       total_log_prob_bad_list_sampling_threshold, individual_bad_word_log_probs_t0_list_sampling_threshold,
                       individual_bad_word_log_probs_t1_list_sampling_threshold, individual_bad_word_log_probs_combined_list_sampling_threshold), save_str)
            print("Sampling actor (q) results:")
            print(total_log_prob_bad_list_sampling)
            print(individual_bad_word_log_probs_t0_list_sampling)
            print(individual_bad_word_log_probs_t1_list_sampling)
            print(individual_bad_word_log_probs_combined_list_sampling)
            print(rew_over_time_list_sampling)
            print(untrans_ret_over_time_list_sampling)
            print("Sampling actor (q) threshold-based results:")
            print(total_log_prob_bad_list_sampling_threshold)
            print(individual_bad_word_log_probs_t0_list_sampling_threshold)
            print(individual_bad_word_log_probs_t1_list_sampling_threshold)
            print(individual_bad_word_log_probs_combined_list_sampling_threshold)
        else:
            # For non-harmlessness training, use the standard lists (which are now base lists)
            save_str = f"{args.save_info_path}/analyticlogprob_rewsample_{info_name_str}"
            torch.save((total_log_prob_bad_list, individual_bad_word_log_probs_t0_list,
                       individual_bad_word_log_probs_t1_list, individual_bad_word_log_probs_combined_list,
                       rew_over_time_list_base, untrans_ret_over_time_list_base,
                       total_log_prob_bad_list_threshold, individual_bad_word_log_probs_t0_list_threshold,
                       individual_bad_word_log_probs_t1_list_threshold, individual_bad_word_log_probs_combined_list_threshold), save_str)
            print(total_log_prob_bad_list)
            print(individual_bad_word_log_probs_t0_list)
            print(individual_bad_word_log_probs_t1_list)
            print(individual_bad_word_log_probs_combined_list)
            print(rew_over_time_list_base)
            print(untrans_ret_over_time_list_base)
            print("Threshold-based results:")
            print(total_log_prob_bad_list_threshold)
            print(individual_bad_word_log_probs_t0_list_threshold)
            print(individual_bad_word_log_probs_t1_list_threshold)
            print(individual_bad_word_log_probs_combined_list_threshold)
        
        if total_kl_sigma_q_list:
            save_str = f"{args.save_info_path}/analytic_kls_indicator_{info_name_str}"
            torch.save((total_kl_sigma_q_list, total_kl_q_sigma_epsq_p_list, 
                       diff_by_bad_word_case1_list, diff_by_bad_word_case2_list, diff_by_bad_word_list, 
                       max_q_exceeds_list, max_sigma_exceeds_list), save_str)
            print(f"KL sigma_q list: {total_kl_sigma_q_list}")
            print(f"KL q_sigma_epsq_p list: {total_kl_q_sigma_epsq_p_list}")
            print(f"Difference by bad word case1 list: {diff_by_bad_word_case1_list}")
            print(f"Difference by bad word case2 list: {diff_by_bad_word_case2_list}")
            print(f"Difference by bad word list (total): {diff_by_bad_word_list}")
            print(f"Max q exceeds list: {max_q_exceeds_list}")
            print(f"Max sigma exceeds list: {max_sigma_exceeds_list}")
    
    if args.analytic_calc:
        save_str = f"{args.save_info_path}/analytic_kls_toxicity_{info_name_str}"
        torch.save((total_kl_sigma_q_list_analytic, total_kl_q_sigma_list_analytic, metrics_list_analytic), save_str)
        print(f"KL sigma_q list (analytic): {total_kl_sigma_q_list_analytic}")
        print(f"KL q_sigma list (analytic): {total_kl_q_sigma_list_analytic}")
        print(f"Metrics list (analytic): {metrics_list_analytic}")


    if args.do_harmlessness_training:
        actor_to_test = base_actor
        initial_model = static_initial_model
    else:
        actor_to_test = actor
        initial_model = base_actor
    if args.evaluate_heldout_sampling or args.evaluate_on_neg_data:
        args.rm_type = "rlhf"
        args.target_dist_beta = 1
        args.reward_transform = None
        reward_model, strip_question_chat_template_fn = get_reward_model(args, strategy)
        reward_model = strategy.prepare(
            reward_model,
            is_rlhf=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        assert args.heldout_prompt_data is not None
        assert args.heldout_input_key is not None
        args.no_critic = True
        critic, critic_optim, critic_scheduler = None, None, None
        actor_optim, actor_scheduler = None, None
        args.parameterization = "policy"
        args.prompt_data = args.heldout_prompt_data
        args.prompt_split = args.heldout_prompt_split
        args.input_key = args.heldout_input_key
        args.input_template = args.heldout_input_template
        args.model_eval = True

        strategy = get_strategy(args)
        strategy.setup_distributed()

        if args.evaluate_heldout_sampling:
            strategy.print("DOING evaluate_heldout_sampling")
            do_evaluate_heldout_sampling(actor_optim, actor_scheduler, actor_to_test, args, critic, critic_optim,
                                         critic_scheduler, ema_model, info_name_str, initial_model, neg_data, reward_model,
                                         strategy, tokenizer, true_posterior_samples, vf_coef)

        if args.evaluate_on_neg_data:
            strategy.print("DOING evaluate_on_neg_data")
            do_evaluate_on_neg_data(actor_to_test, args, strip_question_chat_template_fn, tokenizer, info_name_str, strategy)


    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def do_analytic_kl_calc(
    base_actor, actor, args, tokenizer, prompt, 
    precomputed_toxicity_scores,
    total_kl_sigma_q_list_analytic, total_kl_q_sigma_list_analytic, metrics_list_analytic
) -> dict:
    """
    Calculate analytic KL divergence between target distribution and actor model.
    
    Args:
        base_actor: The base actor model (used as p in target distribution)
        actor: The current actor model (used as q)
        args: Training arguments
        tokenizer: Tokenizer
        prompt: Prompt text for calculations
        precomputed_toxicity_scores: Precomputed toxicity scores for all tokens
        total_kl_sigma_q_list_analytic: List to append KL(sigma_p || q) values to
        total_kl_q_sigma_list_analytic: List to append KL(q || sigma_p) values to
        metrics_list_analytic: List to append metrics dictionaries to
        
    Returns:
        metrics_dict: Dictionary containing metrics from the calculation
    """
    if args.do_harmlessness_training:
        # For harmlessness training, actor is the sampling_actor (q) and base_actor is p
        kl_sigma_q, kl_q_sigma, metrics_dict = calculate_analytic_kl_toxicity_single_token(
            model_p_for_target=base_actor.model,
            model_q=actor.model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            target_dist_beta=args.target_dist_beta,
            precomputed_toxicity_scores=precomputed_toxicity_scores,
            total_kl_sigma_q_list=total_kl_sigma_q_list_analytic,
            total_kl_q_sigma_list=total_kl_q_sigma_list_analytic,
        )
    else:
        # For non-harmlessness training, just use the standard actor
        kl_sigma_q, kl_q_sigma, metrics_dict = calculate_analytic_kl_toxicity_single_token(
            model_p_for_target=base_actor.model,
            model_q=actor.model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            target_dist_beta=args.target_dist_beta,
            precomputed_toxicity_scores=precomputed_toxicity_scores,
            total_kl_sigma_q_list=total_kl_sigma_q_list_analytic,
            total_kl_q_sigma_list=total_kl_q_sigma_list_analytic,
        )
    metrics_list_analytic.append(metrics_dict)
    return metrics_dict



def do_analytic_bad_word_calc(actor, args, bad_word_tokens_ids, base_actor, prompt, tokenizer, 
                              total_log_prob_bad_list, individual_bad_word_log_probs_t0_list,
                              individual_bad_word_log_probs_t1_list, individual_bad_word_log_probs_combined_list,
                              actor_to_test=None, generate_max_len=None):
    # Validate generate_max_len
    if generate_max_len is None:
        generate_max_len = args.generate_max_len
    if generate_max_len not in [1, 2]:
        raise NotImplementedError(f"generate_max_len={generate_max_len} is not supported. Only 1 and 2 are supported.")
    
    # If actor_to_test is explicitly provided, use it; otherwise use the old logic
    if actor_to_test is None:
        if args.do_harmlessness_training:
            actor_to_test = base_actor
        else:
            actor_to_test = actor
    
    # Compute shared intermediate results once
    precomputed = _compute_bad_word_sequence_log_probs(
        model=actor_to_test.model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        bad_word_indices=bad_word_tokens_ids,
        batch_size=args.analytic_batch_size,
        generate_max_len=generate_max_len,
    )
    
    # Use precomputed results to calculate the log probability metrics
    (total_log_prob, individual_bad_word_log_probs_t0, 
     individual_bad_word_log_probs_t1, individual_bad_word_log_probs_combined) = \
        _calculate_bad_word_log_prob_from_precomputed(precomputed, bad_word_tokens_ids, generate_max_len=generate_max_len)
    
    total_log_prob_bad_list.append(total_log_prob)
    individual_bad_word_log_probs_t0_list.append(individual_bad_word_log_probs_t0)
    individual_bad_word_log_probs_t1_list.append(individual_bad_word_log_probs_t1)
    individual_bad_word_log_probs_combined_list.append(individual_bad_word_log_probs_combined)
    
    # Return precomputed intermediate results for reuse
    return precomputed


@torch.no_grad() # Ensure no gradients are computed during evaluation
def _compute_bad_word_sequence_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    bad_word_indices: Union[List[int], torch.Tensor],
    batch_size: int,
    generate_max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Shared computation function that computes log probabilities for sequences containing bad words.
    
    Args:
        generate_max_len: Maximum generation length. If 1, only computes t=0 probabilities. If 2, computes both t=0 and t=1.
    
    Returns:
        A tuple containing:
        - prompt_ids: Tokenized prompt IDs
        - bad_word_indices_tensor: Normalized bad word indices
        - good_word_indices: Indices of good words
        - log_probs_t0: Log probabilities at t=0 for all vocab tokens
        - log_probs_case2: Log probabilities for Case 2 sequences (good at t=0, bad at t=1) [n_good_words, n_bad_words]
          or None if generate_max_len == 1
    """
    device = model.device if hasattr(model, 'device') else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # --- Preprocessing ---
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    # Normalize bad word indices to tensor
    bad_word_indices_tensor = normalize_bad_word_indices(bad_word_indices, device)

    n_vocab = 50257
    n_bad_words = len(bad_word_indices_tensor)

    # Identify indices of "good" words (all vocab except bad words)
    good_word_indices, n_good_words = get_good_word_indices(bad_word_indices_tensor, n_vocab, device)

    # Get log probabilities at t=0 (always computed for both generate_max_len == 1 and == 2)
    log_probs_t0 = get_next_token_log_probs(model, prompt_ids)  # Shape: (n_vocab,)

    # --- Case 2: Good word at t=0, Bad word at t=1 ---
    # Only compute if generate_max_len == 2
    if generate_max_len == 2:
        # Get log probabilities of good words at t=0
        log_probs_good_at_t0 = log_probs_t0[good_word_indices]  # Shape: (n_good_words,)
        
        # Initialize tensor to store Case 2 probabilities
        # Shape: (n_good_words, n_bad_words)
        log_probs_case2 = torch.full((n_good_words, n_bad_words), float('-inf'), device=device)
        
        # Process in batches to manage memory
        for i in range(0, n_good_words, batch_size):
            batch_good_indices = good_word_indices[i : i + batch_size]
            current_batch_size = len(batch_good_indices)

            # Log probabilities of these specific good words at t=0
            batch_log_probs_good_t0 = log_probs_good_at_t0[i : i + current_batch_size]  # Shape: (current_batch_size,)

            # Construct input sequences: prompt + good_word_j
            # Shape: (current_batch_size, prompt_len + 1)
            batch_inputs_t1 = torch.cat(
                (prompt_ids.repeat(current_batch_size, 1), batch_good_indices.unsqueeze(1)),
                dim=1
            )

            # Get log probabilities for all tokens at t=1, conditioned on (prompt + good_word_j)
            log_probs_t1 = get_next_token_log_probs(model, batch_inputs_t1)  # Shape: (current_batch_size, n_vocab)

            # Select log probabilities of bad words at t=1
            log_probs_bad_at_t1 = log_probs_t1[:, bad_word_indices_tensor]  # Shape: (current_batch_size, n_bad_words)

            # Broadcast t=0 probs [current_batch_size, 1] with t=1 bad word probs [current_batch_size, n_bad_words]
            # to get sequence log probs [current_batch_size, n_bad_words]
            batch_log_probs_case2 = batch_log_probs_good_t0.unsqueeze(1) + log_probs_bad_at_t1
            
            # Store in the result tensor
            log_probs_case2[i : i + current_batch_size] = batch_log_probs_case2
    else:
        # generate_max_len == 1: don't compute Case 2
        log_probs_case2 = None

    return prompt_ids, bad_word_indices_tensor, good_word_indices, log_probs_t0, log_probs_case2


@torch.no_grad() # Ensure no gradients are computed during evaluation
def _calculate_bad_word_log_prob_from_precomputed(
    precomputed: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    bad_word_indices: Union[List[int], torch.Tensor],
    generate_max_len: int,
) -> Tuple[float, dict, dict, dict]:
    """
    Calculate bad word log probability metrics from precomputed sequence log probabilities.
    
    Args:
        precomputed: Tuple from _compute_bad_word_sequence_log_probs
        bad_word_indices: Bad word indices (for creating dictionaries)
        generate_max_len: Maximum generation length (1 or 2)
    
    Returns:
        Same as calculate_bad_word_log_prob_pytorch
    """
    prompt_ids, bad_word_indices_tensor, good_word_indices, log_probs_t0, log_probs_case2 = precomputed
    
    n_bad_words = len(bad_word_indices_tensor)
    device = prompt_ids.device

    # --- Case 1: Bad word at t=0 ---
    # For this function, Case 1 is just P(bad_word at t=0 | prompt), not all sequences with bad at t=0
    # So we extract the t=0 probabilities directly
    log_probs_bad_at_t0 = log_probs_t0[bad_word_indices_tensor]  # Shape: (n_bad_words,)

    # Calculate total log probability for Case 1 using logsumexp
    # log P(any bad_word at t=0 | prompt)
    total_log_prob_case1 = torch.logsumexp(log_probs_bad_at_t0, dim=0)

    if generate_max_len == 2:
        # --- Case 2: Good word at t=0, Bad word at t=1 ---
        # log_probs_case2 is [n_good_words, n_bad_words]
        # For total Case 2: sum over all good words j and all bad words k
        total_log_prob_case2 = torch.logsumexp(log_probs_case2.flatten(), dim=0)
        
        # For individual bad word calculation: For each bad word k, sum over all good words j
        # Shape: (n_bad_words,)
        log_probs_case2_per_bad_word = torch.logsumexp(log_probs_case2, dim=0)

        # --- Combine Case 1 and Case 2 ---
        final_combined_log_probs = torch.stack([total_log_prob_case1, total_log_prob_case2])
        print("log prob by cases")
        print(total_log_prob_case1)
        print(total_log_prob_case2)
        total_log_prob = torch.logsumexp(final_combined_log_probs, dim=0).item()
        print(f"Total log prob of bad word: {total_log_prob}")

        # --- Combine Case 1 and Case 2 for each bad word ---
        # For each bad word k:
        # Total Log Probability = logsumexp([log P(bad_k at t=0), log P(good_word at t=0, bad_k at t=1)])
        # Shape: (n_bad_words,)
        total_log_probs_per_bad_word = torch.logsumexp(
            torch.stack([log_probs_bad_at_t0, log_probs_case2_per_bad_word]), dim=0
        )
    else:
        # generate_max_len == 1: only Case 1 (t=0), no Case 2
        total_log_prob_case2 = None
        log_probs_case2_per_bad_word = None
        total_log_prob = total_log_prob_case1.item()
        print("log prob by cases")
        print(total_log_prob_case1)
        print("Case 2: N/A (generate_max_len == 1)")
        print(f"Total log prob of bad word: {total_log_prob}")
        
        # For generate_max_len == 1, combined probabilities are just t=0 probabilities
        total_log_probs_per_bad_word = log_probs_bad_at_t0

    # Create dictionaries mapping token IDs to log probabilities
    individual_bad_word_log_probs_t0 = {}
    individual_bad_word_log_probs_t1 = {}
    individual_bad_word_log_probs_combined = {}

    for idx, bad_word_id in enumerate(bad_word_indices_tensor):
        token_id = bad_word_id.item()
        
        individual_bad_word_log_probs_t0[token_id] = log_probs_bad_at_t0[idx].item()
        if generate_max_len == 2:
            individual_bad_word_log_probs_t1[token_id] = log_probs_case2_per_bad_word[idx].item()
            individual_bad_word_log_probs_combined[token_id] = total_log_probs_per_bad_word[idx].item()
        else:
            # generate_max_len == 1: t=1 is None, combined is just t=0
            individual_bad_word_log_probs_t1[token_id] = None
            individual_bad_word_log_probs_combined[token_id] = total_log_probs_per_bad_word[idx].item()

    # Verification checks (only for generate_max_len == 2)
    if generate_max_len == 2:
        for idx, bad_word_id in enumerate(bad_word_indices_tensor):
            token_id = bad_word_id.item()
            t0_val = individual_bad_word_log_probs_t0[token_id]
            t1_val = individual_bad_word_log_probs_t1[token_id]
            combined_val = individual_bad_word_log_probs_combined[token_id]
            
            # Check that combined = logsumexp(t0, t1) for each bad token
            expected_combined = torch.logsumexp(
                torch.tensor([t0_val, t1_val]), dim=0
            ).item()
            assert math.isclose(combined_val, expected_combined, abs_tol=1e-5), \
                f"For token {token_id}: combined={combined_val}, expected={expected_combined}, t0={t0_val}, t1={t1_val}"
        
        # Check that total_log_prob = logsumexp of all individual_bad_word_log_probs_combined
        all_combined_values = torch.tensor(list(individual_bad_word_log_probs_combined.values()))
        expected_total = torch.logsumexp(all_combined_values, dim=0).item()
        assert math.isclose(total_log_prob, expected_total, abs_tol=1e-5), \
            f"total_log_prob={total_log_prob}, expected={expected_total}"

    print("Individual log prob breakdowns:")
    print(individual_bad_word_log_probs_t0)
    print(individual_bad_word_log_probs_t1)
    print(individual_bad_word_log_probs_combined)

    return (
        total_log_prob,
        individual_bad_word_log_probs_t0,
        individual_bad_word_log_probs_t1,
        individual_bad_word_log_probs_combined
    )


@torch.no_grad() # Ensure no gradients are computed during evaluation
def calculate_bad_word_log_prob_pytorch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    bad_word_indices: Union[List[int], torch.Tensor],
    batch_size: int,
) -> Tuple[float, dict, dict, dict]:
    """
    Calculates the total log probability of generating a sequence of length 2
    (after the prompt) that contains at least one "bad word", along with
    individual bad word log probabilities.

    This is done by summing the probabilities of two disjoint cases:
    1. P(bad_word at t=0 | prompt)
    2. P(good_word at t=0, bad_word at t=1 | prompt)

    Args:
        model: The Hugging Face causal language model (e.g., GPT2LMHeadModel).
        tokenizer: The corresponding tokenizer.
        prompt_text: The input prompt string.
        bad_word_indices: A list or tensor of token IDs considered "bad words".
        batch_size: Batch size for processing vocabulary in the second case
                    to manage memory usage.

    Returns:
        A tuple containing:
        1. total_log_prob (float): Total log probability of a bad word appearing
           in the first or second generated token position.
        2. individual_bad_word_log_probs_t0 (dict): Dictionary mapping bad word
           token IDs to their log probabilities at t=0 position.
        3. individual_bad_word_log_probs_t1 (dict): Dictionary mapping bad word
           token IDs to their log probabilities at t=1 position (summed over
           good tokens at t=0).
        4. individual_bad_word_log_probs_combined (dict): Dictionary mapping bad
           word token IDs to their combined log probabilities (t=0 and t=1 combined).
        Note: I purposefully didn't include the log probs of bad words at t=1 if there is already a
        bad word at t=0. Of course you could do that, but I'm doing this because I think this
        gives better insight into missing/found modes. E.g., if we really care about avoiding
        any bad output, then if we already found a bad token at t=0, that token will have its prob
        reduced (by RePULSe), and same for all the bad tokens following it at t=1. We don't need to worry too much
        about which bad tokens are found at t=1 in this case since they'll all have their probability reduced.
    """

    # Use shared computation function
    precomputed = _compute_bad_word_sequence_log_probs(model, tokenizer, prompt_text, bad_word_indices, batch_size)
    return _calculate_bad_word_log_prob_from_precomputed(precomputed, bad_word_indices)



@torch.no_grad() # Ensure no gradients are computed during evaluation
def calculate_analytic_kl_indicator_bad_words_both_directions(
    model_p_for_target: AutoModelForCausalLM,
    model_q: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    bad_word_indices: Union[List[int], torch.Tensor],
    batch_size: int,
    total_kl_sigma_q_list: List[float],
    total_kl_q_sigma_epsq_p_list: List[float],
    precomputed_p: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    precomputed_q: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[float, float]:
    """
    Calculates the analytic KL divergence in both directions between target distributions and q(x), the proposal distribution,
    given a prompt (assuming 2 output tokens).
    
    1. KL(sigma_p(x) || q) = E_(sigma_p) [ log(sigma_p(x) / q(x)) ]
       The unnormalized target distribution ~sigma_p(x) is the original model probability multiplied by the
       indicator function of the output containing a bad word.
       That is, ~sigma_p(x) := p(x) * I(x contains a bad word).

    2. KL(q || sigma_epsq_p(x)) = E_(q) [ log(q(x) / sigma_epsq_p(x)) ]
       The unnormalized target distribution ~sigma_epsq_p(x) is defined as:
       ~sigma_epsq_p(x) := p(x)                if x contains a bad word
       ~sigma_epsq_p(x) := q(x) * epsilon      if x does not contain a bad word
       This approximation allows us to calculate KL(q || sigma_epsq_p) without enumerating all sequences

    For KL(sigma_p || q):
    Since the expectation is under sigma_p, we can ignore sequences with 0 probability under sigma_p. 
    That is, any sequence not containing a bad word/token.
    
    The remaining sequences of interest can be split into the following cases:
    Case 1: Bad word at t=0. This includes sequences with either good or bad words at t=1. This covers n_bad_words * n_vocab sequences.
    Case 2: Good word at t=0, Bad word at t=1. This covers n_good_words * n_bad_words sequences (where n_good_words = n_vocab - n_bad_words).

    In total, we have n_bad_words * n_vocab + n_good_words * n_bad_words = n_bad_words * (n_vocab + n_good_words) 
    = n_bad_words * (2 n_vocab - n_bad_words) which is approximately 2 n_bad_words * n_vocab sequences of interest.

    For each of these sequences of interest, we can analytically calculate their log probabilities under models p and q.
    Then, we can analytically calculate the log normalizing constant for the target distribution, log Z = log sum_x ~sigma_p(x)
    by summing the exp of the log probabilities of all sequences of interest, then taking the log of the sum.
    Finally, we can calculate the analytic log probability of each sequence x under the target by subtracting the log normalizing constant from the log probability under the base model.
    This is because log sigma_p(x) = log(~sigma_p(x) / sum_x ~sigma_p(x)) = log(~sigma_p(x)) - log Z
    From these log normalized probabilities, we can easily get the normalized probabilities as exp(log(sigma_p(x)))

    Then, once have all the normalized probabilities under the target for all the sequences x of interest,
    we can analytically calculate the KL divergence by taking log(sigma_p(x)) - log(q(x)) * sigma_p(x), then summing over all sequences x of interest

    For KL(q || sigma_epsq_p):
    First note: since the target distribution is an indicator function, if q places any mass on any sequences
    x such that the target has 0 probability (doesn't satisfy the indicator), then q(x)/sigma_p(x) blows up, and we have infinite KL divergence
    Since q(x) is a stochastic policy, there will always be some non zero q(x)
    So the mathematical KL(q || sigma_p) is always infinite
    This is one reason why I was using the INDICATOR_REWARD_EPS
    Now if we use that, we can calculate some meaningful KL divs
    But still, proper calculation requires all 50257^2 sequences
    Since 50257^2 sequences is a lot, consuming a lot of memory and time, we'll instead use an approximation
    That is, suppose instead of using the target defined as:
    ~sigma_p(x) := p(x) * (I(x contains a bad word) + epsilon).
    Instead consider:
    ~sigma_p(x) := p(x)                if x contains a bad word
    ~sigma_p(x) := q(x) * epsilon      if x does not contain a bad word
    Now, with this definition of the target distribution, we can calculate the KL as follows:
    Recall KL(q || sigma_p) = sum_x q(x) * (log(q(x)) - log(sigma_p(x)))
    For x containing a bad word, we already calculated the q(x) and ~sigma_p(x) above
    To calculate sigma_p(x), observe that the normalizing constant is the same sum, except now with an additional
    q(x) * epsilon on all the x that do not contain a bad word. The sum of all this is epsilon * sum_(x not containing bad words) q(x)
    And this is equal to epsilon * (1 - sum_(x containing a bad word) q(x))
    So we can again reuse the probs for q(x) that we've already calculated, and now modify the normlizing constant to add this additional term above
    Next, we can recalculate sigma_p(x) values based on this new normalizing constant
    (We should probably use a separate name to avoid confusion - maybe something like sigma_epsq_p? To denote that this target has the indicator eps based on q)
    Now, once we have our new sigma_epsq_p(x) values, we can calculate for x containing a bad word:
    sum_x q(x) * (log(q(x)) - log(sigma_epsq_p(x)))
    Now for the x not containing a bad word, we have: sum_x q(x) * (log(q(x)) - log(sigma_epsq_p(x)))
    = sum_x q(x) * log(q(x) / sigma_epsq_p(x)))
    = sum_x q(x) * log(q(x) / (q(x) * epsilon / (normalizing constant for sigma_epsq_p)) )
    = sum_x q(x) * log( (normalizing constant for sigma_epsq_p) / epsilon )
    = log( (normalizing constant for sigma_epsq_p) / epsilon ) * sum_x q(x)
    As the normalizing constant and epsilon are independent of x, so we can pull out of the sum
    Then, we already have the normalizing constant and epsilon, and for the sum_x q(x) for x not containing bad words,
    we observed previously this is equal to (1 - sum_(x containing a bad word) q(x)), so let's reuse that here as well
    How good of an approximation is this target? As epsilon tends to 0, it of course tends to the original desired target distribution... doesn't really seem any worse to use q instead of p for the x not containing a bad word

    Args:
        model_p_for_target: The language model used as p in the target distribution.
        model_q: The language model used as the proposal distribution q
        tokenizer: The corresponding tokenizer (assumed to work for both p and q).
        prompt_text: The input prompt string.
        bad_word_indices: A list or tensor of token IDs considered "bad words".
        batch_size: Batch size for processing vocabulary in the second case
                    to manage memory usage.
        total_kl_sigma_q_list: List to append KL(sigma_p || q) values to.
        total_kl_q_sigma_epsq_p_list: List to append KL(q || sigma_epsq_p) values to.

    Returns:
        A tuple of (kl_sigma_q, kl_q_sigma_epsq_p, diff_by_bad_word_case1, diff_by_bad_word_case2, diff_by_bad_word, max_q_exceeds, max_sigma_exceeds) where:
        - kl_sigma_q: KL(sigma_p || q)
        - kl_q_sigma_epsq_p: KL(q || sigma_epsq_p)
        - diff_by_bad_word_case1: dict mapping bad_word_id to sum of log differences for Case 1 (bad word at t=0)
        - diff_by_bad_word_case2: dict mapping bad_word_id to sum of log differences for Case 2 (bad word at t=1)
        - diff_by_bad_word: dict mapping bad_word_id to total sum of log differences (case1 + case2)
        - max_q_exceeds: tuple (log_diff, t0_token, t1_token, log_q_val, log_sigma_val) for largest log(q(x)) - log(sigma_p(x))
        - max_sigma_exceeds: tuple (log_diff, t0_token, t1_token, log_q_val, log_sigma_val) for smallest log(q(x)) - log(sigma_p(x)) (i.e., largest log(sigma_p(x)) - log(q(x)))
    """

    # Use precomputed values if available, otherwise compute them
    if precomputed_p is not None:
        prompt_ids_p, bad_word_indices_tensor, good_word_indices, log_probs_p_t0, log_probs_p_case2 = precomputed_p
        device = prompt_ids_p.device
    else:
        prompt_ids_p, bad_word_indices_tensor, good_word_indices, log_probs_p_t0, log_probs_p_case2 = \
            _compute_bad_word_sequence_log_probs(model_p_for_target, tokenizer, prompt_text, bad_word_indices, batch_size)
        device = prompt_ids_p.device

    if precomputed_q is not None:
        prompt_ids_q, _, _, log_probs_q_t0, log_probs_q_case2 = precomputed_q
        assert prompt_ids_q.device == device
    else:
        prompt_ids_q, _, _, log_probs_q_t0, log_probs_q_case2 = \
            _compute_bad_word_sequence_log_probs(model_q, tokenizer, prompt_text, bad_word_indices, batch_size)
        assert prompt_ids_q.device == device

    # Compute log_probs_case1 locally for KL calculations (bad word at t=0, any word at t=1)
    # This is needed for KL divergence but not returned from the shared function
    n_vocab = 50257
    n_bad_words = len(bad_word_indices_tensor)
    
    # For model p: compute sequences with bad word at t=0, any word at t=1
    batch_prompts_case1_p = prompt_ids_p.repeat(n_bad_words, 1)
    batch_inputs_case1_p = torch.cat(
        (batch_prompts_case1_p, bad_word_indices_tensor.unsqueeze(1)), dim=1
    )
    outputs_case1_p = model_p_for_target(batch_inputs_case1_p)
    log_probs_p_bad_t0 = extract_log_probs_at_position_based_on_token_indices(
        outputs_case1_p, position=-2, token_indices=bad_word_indices_tensor
    )
    log_probs_p_t1_case1 = get_next_token_log_probs(model_p_for_target, batch_inputs_case1_p)
    log_probs_p_case1 = log_probs_p_bad_t0.unsqueeze(1) + log_probs_p_t1_case1  # Shape: [n_bad_words, n_vocab]
    
    # For model q: compute sequences with bad word at t=0, any word at t=1
    batch_prompts_case1_q = prompt_ids_q.repeat(n_bad_words, 1)
    batch_inputs_case1_q = torch.cat(
        (batch_prompts_case1_q, bad_word_indices_tensor.unsqueeze(1)), dim=1
    )
    outputs_case1_q = model_q(batch_inputs_case1_q)
    log_probs_q_bad_t0 = extract_log_probs_at_position_based_on_token_indices(
        outputs_case1_q, position=-2, token_indices=bad_word_indices_tensor
    )
    log_probs_q_t1_case1 = get_next_token_log_probs(model_q, batch_inputs_case1_q)
    log_probs_q_case1 = log_probs_q_bad_t0.unsqueeze(1) + log_probs_q_t1_case1  # Shape: [n_bad_words, n_vocab]

    # Calculate log normalizing constant: log Z = log sum_x ~sigma_p(x)
    # Need to flatten for logsumexp (it requires 1D tensor)
    log_probs_p_flat = torch.cat([log_probs_p_case1.flatten(), log_probs_p_case2.flatten()], dim=0)
    log_Z = torch.logsumexp(log_probs_p_flat, dim=0)

    # Calculate normalized log probabilities under target directly from unflattened values
    # log sigma_p(x) = log ~sigma_p(x) - log Z
    log_probs_sigma_p_case1 = log_probs_p_case1 - log_Z  # Shape: (n_bad_words, n_vocab)
    log_probs_sigma_p_case2 = log_probs_p_case2 - log_Z  # Shape: (n_good_words, n_bad_words)
    
    # For Case 1: We need log probabilities at t=0 only for each bad word
    # log_probs_p_t0 and log_probs_q_t0 are already the conditional probabilities at t=0
    # For sigma_p at t=0, we need to normalize: log sigma_p(bad_word at t=0) = log p(bad_word at t=0) - log_Z_t0
    # where log_Z_t0 = logsumexp over all bad words at t=0 of p(bad_word at t=0)
    # Actually wait, sigma_p is normalized over all sequences with bad words, not just t=0
    # So log sigma_p(bad_word at t=0) = logsumexp over t=1 of log sigma_p(bad_word at t=0, t1)
    # = logsumexp(log_probs_sigma_p_case1[bad_idx, :])
    log_probs_sigma_p_t0 = torch.logsumexp(log_probs_sigma_p_case1, dim=1)  # Shape: (n_bad_words,) - marginal over t=1
    log_probs_q_t0 = torch.logsumexp(log_probs_q_case1, dim=1)  # Shape: (n_bad_words,) - marginal over t=1

    # Calculate normalized probabilities: sigma_p(x) = exp(log sigma_p(x)) (needed for KL)
    probs_sigma_p_case1 = torch.exp(log_probs_sigma_p_case1)
    probs_sigma_p_case2 = torch.exp(log_probs_sigma_p_case2)

    # Calculate KL divergence: KL(sigma_p || q) = sum_x sigma_p(x) * (log(sigma_p(x)) - log(q(x)))
    # Compute KL terms directly from unflattened tensors
    kl_terms_case1 = probs_sigma_p_case1 * (log_probs_sigma_p_case1 - log_probs_q_case1)
    kl_terms_case2 = probs_sigma_p_case2 * (log_probs_sigma_p_case2 - log_probs_q_case2)
    kl_sigma_q = (kl_terms_case1.sum() + kl_terms_case2.sum()).item()

    print(f"KL(sigma_p || q) where sigma_p = p * I[.] : {kl_sigma_q}")
    
    # Calculate differences in log space directly from unflattened values
    log_differences_case1 = log_probs_q_case1 - log_probs_sigma_p_case1  # Shape: (n_bad_words, n_vocab)
    log_differences_case2 = log_probs_q_case2 - log_probs_sigma_p_case2  # Shape: (n_good_words, n_bad_words)
    
    # Track differences aggregated by bad word and find largest differences
    n_vocab = 50257
    n_bad_words = len(bad_word_indices_tensor)
    
    # Aggregate log differences by bad word
    # Case 1: Just the log prob of t=0 token only (each individual bad token only)
    # This is the marginal log difference at t=0 for each bad word
    diff_by_bad_word_case1 = {}  # Bad word at t=0 only
    diff_by_bad_word_case2 = {}  # Bad word at t=1, aggregated over good words at t=0
    
    for bad_idx, bad_word_id in enumerate(bad_word_indices_tensor):
        token_id = bad_word_id.item()
        # Case 1: log difference for this bad word at t=0 only (marginal probability over t=1)
        # This gives us log(q(bad_word appears at t=0)) - log(sigma_p(bad_word appears at t=0))
        diff_by_bad_word_case1[token_id] = (log_probs_q_t0[bad_idx] - log_probs_sigma_p_t0[bad_idx]).item()
        
        # Case 2: sum log differences for this bad word at t=1 (over all good words at t=0)
        # This gives us sum over good_words of [log(q(good_word at t=0, bad_word at t=1)) - log(sigma_p(good_word at t=0, bad_word at t=1))]
        case2_diffs = log_differences_case2[:, bad_idx]  # Shape: (n_good_words,)
        diff_by_bad_word_case2[token_id] = case2_diffs.sum().item()
    
    print("Log difference (log(q) - log(sigma_p)) aggregated by bad word:")
    print("  Case 1 (bad word at t=0 only, marginal over t=1):")
    for token_id, diff in sorted(diff_by_bad_word_case1.items()):
        print(f"    Bad word {token_id}: {diff:.6e}")
    print("  Case 2 (bad word at t=1, summed over all good words at t=0):")
    for token_id, diff_sum in sorted(diff_by_bad_word_case2.items()):
        print(f"    Bad word {token_id}: {diff_sum:.6e}")
    
    # Combined dictionary for return value (sum of both cases)
    diff_by_bad_word = {}
    for token_id in diff_by_bad_word_case1.keys():
        diff_by_bad_word[token_id] = diff_by_bad_word_case1[token_id] + diff_by_bad_word_case2[token_id]
    
    # Find the two largest differences in log space
    # Flatten just for finding max/min indices
    log_differences_flat = torch.cat([log_differences_case1.flatten(), log_differences_case2.flatten()], dim=0)
    
    # 1. Where q(x) exceeds sigma_p(x) by the largest amount: max(log(q(x)) - log(sigma_p(x)))
    max_q_exceeds_flat_idx = log_differences_flat.argmax().item()
    max_q_exceeds_log_diff = log_differences_flat[max_q_exceeds_flat_idx].item()
    
    # 2. Where sigma_p(x) exceeds q(x) by the largest amount: min(log(q(x)) - log(sigma_p(x)))
    max_sigma_exceeds_flat_idx = log_differences_flat.argmin().item()
    max_sigma_exceeds_log_diff = log_differences_flat[max_sigma_exceeds_flat_idx].item()  # This will be negative
    
    # Map flattened indices back to token sequences
    def get_sequence_tokens(flat_idx):
        """Map flattened sequence index back to (t0_token, t1_token)"""
        if flat_idx < n_bad_words * n_vocab:
            # Case 1: bad word at t=0, any word at t=1
            bad_idx = flat_idx // n_vocab
            t1_idx = flat_idx % n_vocab
            t0_token = bad_word_indices_tensor[bad_idx].item()
            t1_token = t1_idx
            return t0_token, t1_token, bad_idx, t1_idx, True
        else:
            # Case 2: good word at t=0, bad word at t=1
            case2_idx = flat_idx - n_bad_words * n_vocab
            good_idx = case2_idx // n_bad_words
            bad_idx = case2_idx % n_bad_words
            t0_token = good_word_indices[good_idx].item()
            t1_token = bad_word_indices_tensor[bad_idx].item()
            return t0_token, t1_token, good_idx, bad_idx, False
    
    max_q_exceeds_t0, max_q_exceeds_t1, max_q_exceeds_idx0, max_q_exceeds_idx1, max_q_exceeds_is_case1 = get_sequence_tokens(max_q_exceeds_flat_idx)
    max_sigma_exceeds_t0, max_sigma_exceeds_t1, max_sigma_exceeds_idx0, max_sigma_exceeds_idx1, max_sigma_exceeds_is_case1 = get_sequence_tokens(max_sigma_exceeds_flat_idx)
    
    # Get log probabilities for printing
    if max_q_exceeds_is_case1:
        max_q_exceeds_log_q = log_probs_q_case1[max_q_exceeds_idx0, max_q_exceeds_idx1].item()
        max_q_exceeds_log_sigma = log_probs_sigma_p_case1[max_q_exceeds_idx0, max_q_exceeds_idx1].item()
    else:
        max_q_exceeds_log_q = log_probs_q_case2[max_q_exceeds_idx0, max_q_exceeds_idx1].item()
        max_q_exceeds_log_sigma = log_probs_sigma_p_case2[max_q_exceeds_idx0, max_q_exceeds_idx1].item()
    
    if max_sigma_exceeds_is_case1:
        max_sigma_exceeds_log_q = log_probs_q_case1[max_sigma_exceeds_idx0, max_sigma_exceeds_idx1].item()
        max_sigma_exceeds_log_sigma = log_probs_sigma_p_case1[max_sigma_exceeds_idx0, max_sigma_exceeds_idx1].item()
    else:
        max_sigma_exceeds_log_q = log_probs_q_case2[max_sigma_exceeds_idx0, max_sigma_exceeds_idx1].item()
        max_sigma_exceeds_log_sigma = log_probs_sigma_p_case2[max_sigma_exceeds_idx0, max_sigma_exceeds_idx1].item()
    
    print(f"\nLargest log difference where q(x) > sigma_p(x):")
    print(f"  Log difference (log(q) - log(sigma_p)): {max_q_exceeds_log_diff:.6e}")
    print(f"  Sequence: t0={max_q_exceeds_t0}, t1={max_q_exceeds_t1}")
    print(f"  log(q(x)) = {max_q_exceeds_log_q:.6e}, log(sigma_p(x)) = {max_q_exceeds_log_sigma:.6e}")
    
    print(f"\nLargest log difference where sigma_p(x) > q(x):")
    print(f"  Log difference (log(q) - log(sigma_p)): {max_sigma_exceeds_log_diff:.6e}")
    print(f"  Sequence: t0={max_sigma_exceeds_t0}, t1={max_sigma_exceeds_t1}")
    print(f"  log(q(x)) = {max_sigma_exceeds_log_q:.6e}, log(sigma_p(x)) = {max_sigma_exceeds_log_sigma:.6e}")

    # Calculate KL divergence in the other direction: KL(q || sigma_epsq_p)
    # We use an approximation where the target distribution sigma_epsq_p is defined as:
    # ~sigma_epsq_p(x) := p(x)                if x contains a bad word
    # ~sigma_epsq_p(x) := q(x) * epsilon      if x does not contain a bad word
    # This allows us to calculate KL(q || sigma_epsq_p) without enumerating all sequences.
    
    epsilon = INDICATOR_REWARD_EPS
    
    # Convert log probabilities to probabilities for sequences with bad words
    # Work directly with unflattened tensors
    probs_p_case1 = torch.exp(log_probs_p_case1)
    probs_p_case2 = torch.exp(log_probs_p_case2)
    probs_q_case1 = torch.exp(log_probs_q_case1)
    probs_q_case2 = torch.exp(log_probs_q_case2)
    
    # Calculate sums needed for the normalizing constant
    sum_p_bad = (probs_p_case1.sum() + probs_p_case2.sum()).item()  # sum_(x with bad) p(x)
    sum_q_bad = (probs_q_case1.sum() + probs_q_case2.sum()).item()  # sum_(x with bad) q(x)
    sum_q_good = 1.0 - sum_q_bad  # sum_(x without bad) q(x) = 1 - sum_(x with bad) q(x)
    
    # Calculate the new normalizing constant for sigma_epsq_p
    # Z_epsq = sum_(x with bad) p(x) + epsilon * sum_(x without bad) q(x)
    Z_epsq = sum_p_bad + epsilon * sum_q_good
    log_Z_epsq = math.log(Z_epsq)
    
    # For sequences with bad words: sigma_epsq_p(x) = p(x) / Z_epsq
    log_probs_sigma_epsq_p_case1 = log_probs_p_case1 - log_Z_epsq
    log_probs_sigma_epsq_p_case2 = log_probs_p_case2 - log_Z_epsq
    
    # KL contribution from sequences with bad words:
    # sum_(x with bad) q(x) * (log(q(x)) - log(sigma_epsq_p(x)))
    kl_terms_bad_case1 = probs_q_case1 * (log_probs_q_case1 - log_probs_sigma_epsq_p_case1)
    kl_terms_bad_case2 = probs_q_case2 * (log_probs_q_case2 - log_probs_sigma_epsq_p_case2)
    kl_bad = (kl_terms_bad_case1.sum() + kl_terms_bad_case2.sum()).item()
    
    # KL contribution from sequences without bad words:
    # For x without bad words: sigma_epsq_p(x) = q(x) * epsilon / Z_epsq
    # log(sigma_epsq_p(x)) = log(q(x)) + log(epsilon) - log(Z_epsq)
    # q(x) * (log(q(x)) - log(sigma_epsq_p(x))) = q(x) * (log(Z_epsq) - log(epsilon))
    # Summing over all x without bad words:
    # sum_(x without bad) q(x) * (log(Z_epsq) - log(epsilon))
    # = (log(Z_epsq) - log(epsilon)) * sum_(x without bad) q(x)
    log_epsilon = math.log(epsilon)
    kl_good = (log_Z_epsq - log_epsilon) * sum_q_good
    
    # Total KL(q || sigma_epsq_p)
    kl_q_sigma_epsq_p = kl_bad + kl_good
    
    print(f"KL(q || sigma_epsq_p) (epsilon approximation): {kl_q_sigma_epsq_p}")

    total_kl_sigma_q_list.append(kl_sigma_q)
    total_kl_q_sigma_epsq_p_list.append(kl_q_sigma_epsq_p)

    max_q_exceeds_info = (max_q_exceeds_log_diff, max_q_exceeds_t0, max_q_exceeds_t1, 
                          max_q_exceeds_log_q, max_q_exceeds_log_sigma)
    max_sigma_exceeds_info = (max_sigma_exceeds_log_diff, max_sigma_exceeds_t0, max_sigma_exceeds_t1,
                             max_sigma_exceeds_log_q, max_sigma_exceeds_log_sigma)

    return kl_sigma_q, kl_q_sigma_epsq_p, diff_by_bad_word_case1, diff_by_bad_word_case2, diff_by_bad_word, max_q_exceeds_info, max_sigma_exceeds_info 


@torch.no_grad()
def precompute_toxicity_scores_for_all_tokens(
    reward_model,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Precompute toxicity scores for all n_vocab tokens by creating sequences with each token
    at position t=0 and passing them through the reward model.
    
    Args:
        reward_model: The reward model to use for scoring
        tokenizer: Tokenizer for the model
        prompt_text: The prompt text
        batch_size: Batch size for processing
        
    Returns:
        Tensor of shape (n_vocab,) containing toxicity scores for each token
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    
    reward_model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    
    n_vocab = 50257
    all_token_ids = torch.arange(n_vocab, device=device)
    
    # Initialize tensor to store scores
    toxicity_scores = torch.zeros(n_vocab, device=device)
    
    # Process in batches to manage memory
    for i in range(0, n_vocab, batch_size):
        batch_token_ids = all_token_ids[i : i + batch_size]
        current_batch_size = len(batch_token_ids)
        
        # Construct input sequences: prompt + token
        batch_inputs = torch.cat(
            (prompt_ids.repeat(current_batch_size, 1), batch_token_ids.unsqueeze(1)),
            dim=1
        )
        
        # Create attention mask
        attention_mask = torch.ones_like(batch_inputs)
        
        # Get reward scores
        with torch.no_grad():
            scores = reward_model(input_ids=batch_inputs, attention_mask=attention_mask)
            if isinstance(scores, tuple):
                scores = scores[0]
            # Scores should be shape (batch_size,), extract the score for the last token
            if scores.dim() > 1:
                scores = scores[:, -1] if scores.shape[1] > 1 else scores.squeeze(-1)
            else:
                scores = scores
            # Ensure scores are on the correct device
            scores = scores.to(device)
        
        toxicity_scores[i : i + current_batch_size] = scores
    
    return toxicity_scores


@torch.no_grad()
def calculate_analytic_kl_toxicity_single_token(
    model_p_for_target: AutoModelForCausalLM,
    model_q: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    target_dist_beta: float,
    precomputed_toxicity_scores: torch.Tensor,
    total_kl_sigma_q_list: List[float],
    total_kl_q_sigma_list: List[float],
) -> Tuple[float, float, dict]:
    """
    Calculates the analytic KL divergence in both directions between target distribution and q(x),
    given a prompt and single token output.
    
    Target distribution: sigma_p(x) = p(x) * e^(target_dist_beta * r(x)) / Z
    where r(x) is the toxicity score for token x.
    
    Args:
        model_p_for_target: The language model used as p in the target distribution
        model_q: The language model used as the proposal distribution q
        tokenizer: Tokenizer
        prompt_text: The input prompt string
        target_dist_beta: Beta parameter for target distribution
        precomputed_toxicity_scores: Precomputed toxicity scores for all tokens, shape (n_vocab,)
        total_kl_sigma_q_list: List to append KL(sigma_p || q) values to
        total_kl_q_sigma_list: List to append KL(q || sigma_p) values to
        
    Returns:
        Tuple of (kl_sigma_q, kl_q_sigma, metrics_dict) where metrics_dict contains top token info
    """
    device = model_p_for_target.device if hasattr(model_p_for_target, 'device') else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_p_for_target.eval()
    model_q.eval()
    model_p_for_target.to(device)
    model_q.to(device)
    
    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    
    n_vocab = 50257
    
    # Get log probabilities for all tokens from prompt (p and q)
    log_probs_p = get_next_token_log_probs(model_p_for_target, prompt_ids)  # Shape: (n_vocab,)
    log_probs_q = get_next_token_log_probs(model_q, prompt_ids)  # Shape: (n_vocab,)
    
    # Calculate unnormalized target distribution: log ~sigma_p(x) = log p(x) + target_dist_beta * r(x)
    log_unnormalized_target = log_probs_p + target_dist_beta * precomputed_toxicity_scores
    
    # Calculate log normalizing constant: log Z = logsumexp(log ~sigma_p(x))
    log_Z = torch.logsumexp(log_unnormalized_target, dim=0)
    
    # Calculate normalized target distribution: log sigma_p(x) = log ~sigma_p(x) - log Z
    log_probs_target = log_unnormalized_target - log_Z
    
    # Convert to probabilities for KL calculations
    probs_target = torch.exp(log_probs_target)
    probs_q = torch.exp(log_probs_q)
    
    # Calculate KL(sigma_p || q) = sum_x sigma_p(x) * (log(sigma_p(x)) - log(q(x)))
    kl_terms = probs_target * (log_probs_target - log_probs_q)
    kl_sigma_q = kl_terms.sum().item()
    
    # Calculate KL(q || sigma_p) = sum_x q(x) * (log(q(x)) - log(sigma_p(x)))
    kl_terms_reverse = probs_q * (log_probs_q - log_probs_target)
    kl_q_sigma = kl_terms_reverse.sum().item()
    
    print(f"KL(sigma_p || q): {kl_sigma_q}")
    print(f"KL(q || sigma_p): {kl_q_sigma}")
    
    # Find top 10 tokens under q and target distribution
    top_10_q_indices = torch.topk(probs_q, k=10, dim=0).indices
    top_10_target_indices = torch.topk(probs_target, k=10, dim=0).indices
    
    # Combine and get unique tokens
    all_top_indices = torch.unique(torch.cat([top_10_q_indices, top_10_target_indices]))
    
    # Create metrics dictionary
    metrics_dict = {
        'top_10_q_tokens': top_10_q_indices.cpu().tolist(),
        'top_10_target_tokens': top_10_target_indices.cpu().tolist(),
        'all_tracked_tokens': all_top_indices.cpu().tolist(),
        'log_probs_q': {},
        'log_probs_target': {},
        'log_diff': {},
    }
    
    # Calculate log probs and differences for tracked tokens
    for token_idx in all_top_indices:
        token_id = token_idx.item()
        log_q_val = log_probs_q[token_idx].item()
        log_target_val = log_probs_target[token_idx].item()
        log_diff = log_q_val - log_target_val
        
        metrics_dict['log_probs_q'][token_id] = log_q_val
        metrics_dict['log_probs_target'][token_id] = log_target_val
        metrics_dict['log_diff'][token_id] = log_diff
    
    print("\nTop 10 tokens under q(x):")
    for token_id in metrics_dict['top_10_q_tokens']:
        print(f"  Token {token_id}: log_q={metrics_dict['log_probs_q'][token_id]:.6f}, "
              f"log_target={metrics_dict['log_probs_target'][token_id]:.6f}, "
              f"diff={metrics_dict['log_diff'][token_id]:.6f}")
    
    print("\nTop 10 tokens under target distribution:")
    for token_id in metrics_dict['top_10_target_tokens']:
        print(f"  Token {token_id}: log_q={metrics_dict['log_probs_q'][token_id]:.6f}, "
              f"log_target={metrics_dict['log_probs_target'][token_id]:.6f}, "
              f"diff={metrics_dict['log_diff'][token_id]:.6f}")
    
    total_kl_sigma_q_list.append(kl_sigma_q)
    total_kl_q_sigma_list.append(kl_q_sigma)
    
    return kl_sigma_q, kl_q_sigma, metrics_dict


def do_rejection_sampling_for_posterior_samples(args, base_actor, reward_model, tokenizer, strategy, prompts_dataloader):
    """
    Perform rejection sampling to generate true posterior samples.
    
    Target distribution: target(x) ∝ p(x) * e^(β * r(x))
    Proposal distribution: p(x) (base actor)
    Acceptance probability: e^(β * clamped_r) / M, where M = e^(|clamp * beta|)
    """
    # Validation
    if args.rm_type != "rlhf":
        raise NotImplementedError(f"Rejection sampling currently only supports rm_type='rlhf', got '{args.rm_type}'")
    if args.reward_clamp is None:
        raise ValueError("--reward_clamp must be set (not None) when using --rejection_sample_true_target_only")
    if args.target_dist_beta is None:
        raise ValueError("--target_dist_beta must be set when using --rejection_sample_true_target_only")
    if args.true_target_sample_amount <= 0:
        raise ValueError(f"--true_target_sample_amount must be > 0, got {args.true_target_sample_amount}")
    
    strategy.print("Starting rejection sampling for posterior samples...")
    
    # Setup
    base_actor.eval()
    reward_model.eval()
    device = next(base_actor.parameters()).device
    
    # Generate filename
    filename = get_posterior_samples_filename(args)
    strategy.print(f"Will save posterior samples to: {filename}")
    
    # Calculate rejection bound in log space: log_M = |clamp * beta|
    clamp_beta_product = args.reward_clamp * args.target_dist_beta
    log_M = abs(clamp_beta_product)
    strategy.print(f"Computing rejection bound in log space: log_M = |{args.reward_clamp} * {args.target_dist_beta}| = |{clamp_beta_product}| = {log_M}")
    strategy.print(f"  This corresponds to M = e^({log_M}) = {torch.exp(torch.tensor(log_M, dtype=torch.float32)).item():.4e} (for reference, may be inf)")
    
    # Handle prompts
    if args.new_custom_single_prompt:
        # Use custom prompt
        prompts = [args.custom_prompt]
        strategy.print(f"Using custom prompt: {args.custom_prompt}")
    else:
        # Extract prompts from dataloader
        prompts = []
        for batch in prompts_dataloader:
            # Dataloader returns batches, which are lists/tuples of prompt strings
            if isinstance(batch, (list, tuple)):
                prompts.extend(batch)
            elif isinstance(batch, str):
                prompts.append(batch)
            else:
                # If it's a tensor or other type, try to convert
                prompts.extend([str(p) for p in batch])
        strategy.print(f"Found {len(prompts)} prompts from dataloader")
    
    # Storage for accepted samples per prompt
    posterior_samples_by_prompt = []
    
    # Generation kwargs
    generate_kwargs = {
        "max_new_tokens": args.generate_max_len,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "temperature": 1.0,
    }
    
    # Tokenize function (similar to experience_maker)
    def tokenize_fn(texts, max_length, device):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}
    
    # Process each prompt
    total_generated_all = 0
    total_accepted_all = 0
    
    for prompt_idx, prompt in enumerate(prompts):
        strategy.print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts)}")
        accepted_samples = []
        total_generated = 0
        total_accepted = 0
        iteration = 0
        
        # Continue sampling until we have enough accepted samples
        while len(accepted_samples) < args.true_target_sample_amount:
            iteration += 1
            
            # Generate batch of samples from base actor
            # Use tile_prompts (as in make_experience / generate_seqs_and_get_all_data) to repeat
            # the prompt rollout_batch_size times for batched generation
            prompt_batch = tile_prompts(prompt, args.duplicate_rollout_batch_by)
            
            # Tokenize prompts
            inputs = tokenize_fn(prompt_batch, args.prompt_max_len, device=device)
            
            # Generate sequences
            with torch.no_grad():
                sequences, attention_mask, action_mask = base_actor.generate(
                    **inputs,
                    **generate_kwargs
                )
            
            # Compute rewards
            with torch.no_grad():
                rewards = reward_model(sequences, attention_mask)
                rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards  # Ensure shape (batch_size,)
            
            # Clamp rewards
            clamped_rewards = rewards.clamp(min=-args.reward_clamp, max=args.reward_clamp)
            
            # Compute log_phi = beta * clamped_r (in log space)
            log_phi = args.target_dist_beta * clamped_rewards
            
            # Compute acceptance probabilities in log space to avoid overflow
            # accept_prob = e^(beta * clamped_r) / e^(log_M) = e^(log_phi - log_M)
            log_ratio = log_phi - log_M
            accept_prob = torch.exp(log_ratio)
            accept_prob = torch.clamp(accept_prob, min=0.0, max=1.0)
            
            # Perform rejection sampling
            u = torch.rand_like(accept_prob)
            accept_mask = u < accept_prob
            
            # Extract accepted sequences
            accepted_sequences = sequences[accept_mask]
            
            # Convert to CPU and store
            for seq in accepted_sequences:
                accepted_samples.append(seq.cpu().tolist())
            
            # Update counters
            batch_size_actual = sequences.shape[0]
            total_generated += batch_size_actual
            total_accepted += accept_mask.sum().item()
            
            # Print progress periodically
            if iteration % 10 == 0 or len(accepted_samples) >= args.true_target_sample_amount:
                acceptance_rate = total_accepted / total_generated if total_generated > 0 else 0.0
                log_phi_mean = log_phi.mean().item()
                log_phi_min = log_phi.min().item()
                log_phi_max = log_phi.max().item()
                clamped_r_mean = clamped_rewards.mean().item()
                clamped_r_min = clamped_rewards.min().item()
                clamped_r_max = clamped_rewards.max().item()
                log_ratio_mean = log_ratio.mean().item()
                accept_prob_mean = accept_prob.mean().item()
                strategy.print(f"  Iteration {iteration}: {len(accepted_samples)}/{args.true_target_sample_amount} accepted, "
                             f"{total_generated} generated, acceptance rate: {acceptance_rate:.4f}")
                strategy.print(f"    clamped_r: mean={clamped_r_mean:.4f}, min={clamped_r_min:.4f}, max={clamped_r_max:.4f}")
                strategy.print(f"    log_phi (beta * clamped_r): mean={log_phi_mean:.4f}, min={log_phi_min:.4f}, max={log_phi_max:.4f}")
                strategy.print(f"    log_M={log_M:.4f}, log_ratio (log_phi - log_M): mean={log_ratio_mean:.4f}")
                strategy.print(f"    accept_prob: mean={accept_prob_mean:.6f}, beta={args.target_dist_beta}, clamp={args.reward_clamp}")
        
        # Truncate to exact target amount
        accepted_samples = accepted_samples[:args.true_target_sample_amount]
        
        # Store for this prompt
        posterior_samples_by_prompt.append(accepted_samples)
        
        # Print statistics for this prompt
        final_acceptance_rate = total_accepted / total_generated if total_generated > 0 else 0.0
        strategy.print(f"Prompt {prompt_idx + 1} complete: {len(accepted_samples)} samples accepted "
                      f"from {total_generated} generated (acceptance rate: {final_acceptance_rate:.4f})")
        
        total_generated_all += total_generated
        total_accepted_all += total_accepted
    
    # Format output (matching loading format)
    true_posterior_samples_by_prompt_and_by_token = posterior_samples_by_prompt
    
    # Save on rank 0 only
    if strategy.is_rank_0():
        torch.save(true_posterior_samples_by_prompt_and_by_token, filename)
        strategy.print(f"\nSaved posterior samples to: {filename}")
    
    # Print final statistics
    overall_acceptance_rate = total_accepted_all / total_generated_all if total_generated_all > 0 else 0.0
    strategy.print(f"\nFinal statistics:")
    strategy.print(f"  Total samples generated: {total_generated_all}")
    strategy.print(f"  Total samples accepted: {total_accepted_all}")
    strategy.print(f"  Overall acceptance rate: {overall_acceptance_rate:.4f}")
    strategy.print(f"  Samples per prompt: {args.true_target_sample_amount}")


def do_evaluate_heldout_sampling(actor_optim, actor_scheduler, actor_to_test, args, critic, critic_optim,
                                 critic_scheduler, ema_model, info_name_str, initial_model, neg_data, reward_model,
                                 strategy, tokenizer, true_posterior_samples, vf_coef):
    trainer = get_base_ppo_trainer(actor_to_test, actor_optim, actor_scheduler, args, initial_model, critic,
                                   critic_optim,
                                   critic_scheduler, ema_model, neg_data, reward_model, strategy, tokenizer,
                                   true_posterior_samples, vf_coef)
    rewards = []
    returns = []
    entropy = []
    kls = []
    pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
    for i in range(args.sampling_iters):
        strategy.print(f"Sampling iter: {i}")
        for rand_prompts in prompts_dataloader:
            experience = trainer.experience_maker.make_experience(
                rand_prompts,
                samples_per_prompt=args.duplicate_rollout_batch_by,
                # force_no_transform=True,
                **trainer.generate_kwargs
            )

            rewards.append(experience.info["reward"])
            returns.append(experience.info["return"])
            entropy.append(experience.info["entropy"])
            kls.append(experience.info["kl"])
            # print(experience.info["reward"])

            reward_scores = experience.info["reward"]

            for threshold in [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]:
                below_threshold = reward_scores < threshold
                total_below = below_threshold.sum().item()
                if total_below > 0:
                    print(f"BAD TEXT FOUND THRESHOLD {threshold}")
                print(f"TOTAL BELOW REWARD {threshold}")
                print(total_below)
                print(total_below / below_threshold.shape[-1])

                bad_text = tokenizer.batch_decode(experience.sequences[below_threshold], skip_special_tokens=True)
                print(f"BAD TEXT: threshold {threshold}")
                print(bad_text)

    rewards = torch.cat(rewards)
    returns = torch.cat(returns)
    entropy = torch.cat(entropy)
    kls = torch.cat(kls)

    strategy.print(f"Average reward: {rewards.mean().item()}")
    total_samples = rewards.shape[0]
    strategy.print(f"Total number of samples drawn: {total_samples}")
    for threshold in range(-6, 1):
        outputs_below_threshold = (rewards < threshold).sum()
        strategy.print(f"Number of outputs below threshold {threshold}: {outputs_below_threshold.item()}")
        strategy.print(f"Estimate of probability of bad outputs: {(outputs_below_threshold / total_samples).item()}")
        strategy.print(
            f"Estimate of log probability of bad outputs: {(torch.log(outputs_below_threshold) - torch.log(torch.tensor(total_samples))).item()}")


    save_str = f"{args.save_info_path}/info_eval_{info_name_str}"
    torch.save((rewards, returns, kls, entropy), save_str)



def do_evaluate_on_neg_data(actor, args, strip_question_chat_template_fn, tokenizer, info_name_str, strategy):
    with open(args.neg_data_load_path, "rb") as f:
        neg_data = pickle.load(f)
    neg_data = list(neg_data)
    actor.eval()
    actor = actor.to(torch.cuda.current_device())
    results = []
    prompts = []

    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    strategy.print(len(neg_data))
    import re
    def strip_leading_im_end(s):
        return re.sub(r'^(<\|im_end\|>)+', '', s)

    from functools import partial
    for i in range(len(neg_data) // args.train_batch_size + 1):
        if (i + 1) % 10 == 0:
            strategy.print(f"BATCH {i + 1}")

        batch = neg_data[i * args.train_batch_size: (i + 1) * args.train_batch_size]

        cleaned_batch = list(map(strip_leading_im_end, batch))

        strip_question_chat_template_fn_for_neg_data = partial(strip_question_chat_template_fn, additional_split=True)

        qa_list = list(map(strip_question_chat_template_fn_for_neg_data, cleaned_batch))
        text_question, text_answer = map(list, zip(*qa_list))

        inputs = tokenize_fn(cleaned_batch)

        sequences = inputs["input_ids"]

        sequences, attention_mask, action_mask = actor.process_sequences(sequences,
                                                                         sequences.size(1) - args.generate_max_len,
                                                                         tokenizer.eos_token_id, tokenizer.pad_token_id)


        with torch.no_grad():
            log_probs = actor(
                sequences=sequences,
                num_actions=args.generate_max_len,
                attention_mask=attention_mask,
            )


        total_log_prob = (log_probs * action_mask).sum(-1)

        results.append(total_log_prob)

        prompts.extend(text_question)
    result_stack = torch.cat(results, dim=0)
    strategy.print(result_stack.shape)
    detailed_dict = {}
    for prompt, log_prob in zip(prompts, result_stack):
        if prompt not in detailed_dict.keys():
            detailed_dict[prompt] = []
        detailed_dict[prompt].append(log_prob)
    mean_log_prob_by_prompt = []
    total_log_prob_by_prompt = []
    for prompt in detailed_dict.keys():
        detailed_dict[prompt] = torch.tensor(detailed_dict[prompt])
        strategy.print(f"Prompt: {prompt}")
        strategy.print(f"Number of bad sequences for this prompt: {len(detailed_dict[prompt])}")
        avg_log_prob = detailed_dict[prompt].mean()
        strategy.print(f"Average log prob on bad sequences: {avg_log_prob}")
        mean_log_prob_by_prompt.append(avg_log_prob)
        total_log_prob = torch.logsumexp(detailed_dict[prompt], dim=0)
        strategy.print(f"Total log prob of bad sequences (logsumexp): {total_log_prob}")
        total_log_prob_by_prompt.append(total_log_prob)
    strategy.print("Mean log prob on dataset")
    strategy.print(result_stack.mean().item())
    strategy.print("Averaging the mean log prob for each prompt over prompts")
    strategy.print(torch.tensor(mean_log_prob_by_prompt).mean().item())
    strategy.print("Averaging the total log prob for each prompt over prompts")
    strategy.print(torch.tensor(total_log_prob_by_prompt).mean().item())

    save_str = f"{args.save_info_path}/neg_data_dict_{info_name_str}"
    torch.save(detailed_dict, save_str)


def get_reward_model(args, strategy):
    strip_question_chat_template_fn = None
    if args.reward_pretrain == "indicator_bad_token":
        # Skip reward model loading - using handcrafted indicator function
        return None, strip_question_chat_template_fn
    if not args.remote_rm_url:
        if args.reward_pretrain == "nicholasKluge/ToxicityModel":
            strategy.print(f"USING CUSTOM REWARD MODEL {args.reward_pretrain}")
            from transformers import AutoTokenizer, AutoConfig, AutoModel

            def get_tokenizer_custom(model_config):
                tokenizer = AutoTokenizer.from_pretrained(model_config)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer

            tokenizer_base = get_tokenizer_custom(args.pretrain)

            rm_name = args.reward_pretrain
            config = AutoConfig.from_pretrained(rm_name, trust_remote_code=True)
            config.normalize_reward = False
            assert not args.normalize_reward  # Not yet implemented
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__
            reward_model = _get_reward_model_custom(
                base_pretrained_class, rm_name,
                tokenizer_base=tokenizer_base, config=config,
                rm_max_len=args.rm_max_len
            )

        elif args.reward_pretrain in [
            "OpenAssistant/reward-model-deberta-v3-base", "OpenAssistant/reward-model-deberta-v3-large-v2",
            "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft", "Skywork/Skywork-Reward-V2-Llama-3.2-1B", "meta-llama/Llama-Guard-3-1B"
        ]:
            strategy.print(f"USING CUSTOM REWARD MODEL {args.reward_pretrain}")
            from transformers import AutoTokenizer, AutoConfig, AutoModel

            def get_tokenizer_custom(model_config):
                tokenizer = AutoTokenizer.from_pretrained(model_config)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer

            tokenizer_base = get_tokenizer_custom(args.pretrain)

            rm_name = args.reward_pretrain
            config = AutoConfig.from_pretrained(rm_name, trust_remote_code=True)
            config.normalize_reward = False
            assert not args.normalize_reward  # Not yet implemented
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__
            strip_question_chat_template_fn = None
            if args.apply_chat_template:
                strip_question_chat_template_fn = get_strip_question_chat_template_fn(args)
            reward_model = _get_reward_model_custom(
                base_pretrained_class, rm_name,
                tokenizer_base=tokenizer_base,
                config=config,
                separatequeryanswer=True,
                rm_max_len=args.rm_max_len,
                strip_question_chat_template_fn=strip_question_chat_template_fn,
            )

        else:

            reward_model = get_llm_for_sequence_regression(
                args.reward_pretrain,
                "reward",
                normalize_reward=args.normalize_reward,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                ds_config=strategy.get_ds_train_config(is_actor=False),
                value_head_prefix=args.value_head_prefix,
            )
            get_tokenizer(args.reward_pretrain, reward_model, "left", strategy,
                          use_fast=not args.disable_fast_tokenizer)
    else:
        reward_model = None
    return reward_model, strip_question_chat_template_fn


def get_base_ppo_trainer(actor, actor_optim, actor_scheduler, args, base_actor, critic, critic_optim, critic_scheduler,
                         ema_model, neg_data, reward_model, strategy, tokenizer, true_posterior_samples, vf_coef):
    # configure Trainer
    trainer = BasePPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        base_actor,
        ema_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        target_dist_beta=args.target_dist_beta,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # fro GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # remote reward model
        remote_rm_url=args.remote_rm_url,
        shared_actorcritic=args.shared_actorcritic,
        vf_coef=vf_coef,
        model_eval=args.model_eval,
        threshold=args.threshold,
        reward_cap=args.reward_cap,
        n_seeds_f_q=args.n_seeds_f_q,
        rm_type=args.rm_type,
        bc_coef=args.bc_coef,
        bc_steps=args.bc_steps,
        true_posterior_samples=true_posterior_samples,
        actor_loss_type=args.actor_loss_type,
        critic_loss_type=args.critic_loss_type,
        alpha=args.alpha,
        parameterization=args.parameterization,
        save_negdata=args.save_negdata,
        save_negdata_threshold=args.save_negdata_threshold,
        neg_data=neg_data,
        reward_transform=args.reward_transform,
        bad_word_tokens_ids=bad_word_tokens_ids
    )
    return trainer


def get_prompts_data(args, strategy, tokenizer):
    # prepare datasets
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    pretrain_dataset = None
    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data,
            args.pretrain_data_probs,
            strategy,
            args.seed,
            return_eval=False,
            train_split=args.pretrain_split,
        )
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )
    return pretrain_dataset, prompts_dataset


def get_strip_question_chat_template_fn(args):
    if args.pretrain in [
        "HuggingFaceTB/SmolLM-135M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",

    ]:
        def strip_question_chat_template_fn(text, additional_split=False):
            question, answer = text.split('assistant\n',
                                          maxsplit=1)  # in case 'assistant\n' shows up in the output, only split on the first occurrence
            question = question.split('user\n',
                                          maxsplit=1)[-1].strip('\n')

            if additional_split:  # Used for the neg_data right now, kind of hacky
                question = question.split('<|im_end|>')[0]

            return question, answer
    elif args.pretrain in [
        "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"
    ]:
        def strip_question_chat_template_fn(text, additional_split=False):
            question, answer = text.split('assistant\n\n',
                                          maxsplit=1)  # in case 'assistant\n' shows up in the output, only split on the first occurrence
            question = question.split('user\n\n',
                                          maxsplit=1)[-1].strip('\n')

            if additional_split:  # Used for the neg_data right now, kind of hacky
                raise NotImplementedError # not tested
                question = question.split('<|im_end|>')[0]

            return question, answer
    else:
        raise NotImplementedError
    return strip_question_chat_template_fn


def do_load_checkpoints(args, actor, critic, strategy):
    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint:
        if os.path.exists(f"{args.ckpt_path}"):
            _, states = strategy.load_ckpt(actor.model, f"{args.ckpt_path}")
            if critic is not None:
                base_path = args.ckpt_path.split("_actor")[0]
                strategy.load_ckpt(critic, f"{base_path}_critic")
            consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
        else:
            raise Exception("Checkpoint not found")
    else:
        print("Skipping checkpoint loading. Use --load_checkpoint to load checkpoint.")

    return consumed_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_actor", action="store_true", default=False)

    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_steps_harmless", type=int, default=-1, help="For the harmlessness training only; save after x total steps of training")

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo", help="For loading, include the full path name including all the info_name_str (AND NOW INCLUDING '_actor'). For saving, the info_name_str and '_actor' will be auto-genereated, just include only the folder path")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1, help="For PPO, is the number of total times to do updates on the prompt dataset. For harmlessness training, is the number of times to do twist updates per each policy model update")
    parser.add_argument("--fit_steps", type=int, default=1, help="Used only for the toy environment setting for now, otherwise leave at 1")

    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of PPO inner loop steps")
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--rm_max_len", type=int, default=2048, help="Cut off tokens beyond this limit passed into the RM")

    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--analytic_batch_size", type=int, default=None, help="Batch size for analytic calculations. Defaults to train_batch_size if not set.")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")

    parser.add_argument("--bc_coef", type=float, default=0.0, help="Do behaviour cloning on exact posterior samples (cheating for the sake of illustrating optimality)")
    parser.add_argument("--bc_steps", type=int, default=-1, help="Default -1 means always use bc_coef; otherwise, after bc_steps, set bc_coef to 0")

    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation. THIS DUPLICATION HAPPENS AT THE DATASET LEVEL"
    )
    parser.add_argument("--duplicate_rollout_batch_by", type=int, default=1, help="For each prompt in the rollout batch, copy this duplicate_rollout_batch_by many times, before rolling out/generating experience/sequences. Specifically used for twist learning, where we need multiple samples for approximate positive sampling. For twist learning methods, should be > 1")

    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--base_actor_learning_rate", type=float, default=1e-6, help="Only used with --do_harmlessness_training")

    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=1, help="KL penalty to prior/base model in PPO/REINFORCE")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")


    parser.add_argument("--alpha", type=float, default=0.5, help="Choose how much to prioritize the harmlessness objective vs. standard RL (or for use in reward transformations)")
    parser.add_argument("--rew_trans_alpha", type=float, default=None, help="Only for reward transforms. To maintain compatibility with old commands, pick up from alpha if set to None")
    parser.add_argument("--rew_trans_beta", type=float, default=None, help="Only for reward transforms. To maintain compatibility with old commands, pick up from target_dist_beta if set to None")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument(
        "--new_custom_single_prompt", action="store_true", default=False, help="Use only a single custom prompt"
    )
    parser.add_argument(
        "--custom_prompt", type=str, default="This man is a", help="Custom prompt string to use for training and analytic calculations"
    )

    parser.add_argument("--heldout_prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument("--heldout_prompt_split", type=str, default="train")
    parser.add_argument("--heldout_input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--heldout_input_template", type=str, default=None)
    # parser.add_argument(
    #     "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    # )

    parser.add_argument("--rm_type", type=str, default="exp_beta_toxicity_class_logprob",
                        choices=["exp_beta_rew_p_continuation", "exp_beta_rew_p_continuation_divided_by_p",
                                 "p_continuation", "exp_beta_toxicity", "exp_beta_toxicity_class_logprob",
                                 "exp_beta_sentiment_class_logprob",
                                 "indicator_below_threshold", "sentiment_threshold",
                                 "p_last_tokens", "toy_test", "rlhf"])
    parser.add_argument("--threshold", type=float, default=-5., help="The threshold for the toxicity score (or whatever score used for indicator_below_threshold)")
    parser.add_argument("--reward_clamp", type=float, default=None, help="Clamp reward values between [-clamp, +clamp]. If None, no clamping is performed. Only for use with rlhf rm_type")
    parser.add_argument(
        "--save_negdata", action="store_true", default=False, help="Save a dataset of negative examples"
    )
    parser.add_argument("--save_negdata_threshold", type=float, default=-10000., help="The threshold below which we save examples for a negative dataset")

    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo",
    )
    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    parser.add_argument("--target_dist_beta", type=float, default=None, help="Beta in our SMC formulation of the target distribution of p_0 e^{beta r}. Auto-calculated based on KL coef for PPO. For harmlessness training, this is for the sigma target distribution for negative training/whatever unlearning method. Also used in reward transformations")

    parser.add_argument("--anneal_target_dist_beta", action="store_true", help="if set, anneal target_dist_beta")
    parser.add_argument("--start_target_dist_beta", type=float, default=None, help="Only used for annealing. Start at this beta value and anneal to final target_dist_beta value")
    parser.add_argument("--start_alpha", type=float, default=None, help="Only used for annealing alpha. Start at this alpha value and anneal to final alpha value")

    parser.add_argument("--separate_reweighting_beta", type=float, default=None, help="if set, use this instead of the target_dist_beta for reweighting samples for sigma only. Still use the target_dist_beta for training the proposal q")
    parser.add_argument("--uniform_reweight", action="store_true", help="if set, use uniform weights for reweighting. Basically skips the reweighting operation.")


    parser.add_argument("--load_posterior_samples", action="store_true", help="load posterior samples from saved checkpoint instead of creating new ones")
    parser.add_argument("--load_posterior_samples_name", type=str, default='.', help="Full filename of what to load for posterior samples")
    parser.add_argument("--rejection_sample_true_target_only", action="store_true", help="If set, skip normal training and only perform rejection sampling to generate true posterior samples. Saves samples to file. Requires --rm_type rlhf and --reward_clamp to be set.")
    parser.add_argument("--true_target_sample_amount", type=int, default=1000, help="Number of accepted samples to collect via rejection sampling (continues sampling until this many are accepted)")
    parser.add_argument("--save_info_path", type=str, default="./info")
    parser.add_argument("--n_samples_for_f_q", type=int, default=500, help="Number of samples to use for f_q (only for f_q_g_q_eval)")
    parser.add_argument("--n_seeds_f_q", type=int, default=4, help="Number of seeds to use for f_q")


    parser.add_argument("--update_steps_per_episode", type=int, default=1, help="Number of gradient updates (PPO loss outer loop) per episode")
    parser.add_argument("--exp_num_twist_updates", action="store_true", help="Use an exponentially increasing power of twist updates (base 2) instead of a set number of twist updates per epoch")
    parser.add_argument("--no_test_info", action="store_true", help="don't do the f_q_g_q stuff")
    parser.add_argument("--f_q_g_q_eval", action="store_true", default=False, help="Enable f_q/g_q/IWAE evaluation (requires --new_custom_single_prompt)")
    parser.add_argument("--test_info_every", type=int, default=1, help="Test info (e.g., F_q) after this many number of gradient updates")

    parser.add_argument(
        "--parameterization", type=str, default="policy",
        choices=["policy", "policy_psi_unnorm", "policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t", "modulation_model", "modulation_linear_head", "modulation_nn_head"]
    )

    # parser.add_argument("--actor_modulates_base", action="store_true", help="Use parameterization where actor outputs an addition (modulation) to base log prob")
    parser.add_argument("--shared_actorcritic", action="store_true", help="Use parameterization where actor and critic are just different heads, not separate networks. Uses actor lr for shared learning rate")
    parser.add_argument("--model_eval", action="store_true", help="Use model.eval() instead of model.train(). Turns off dropout and norm statistics")

    parser.add_argument("--no_critic", action="store_true", help="Do not use a critic")

    parser.add_argument("--clamp_reward", action="store_true", help="Clamp reward between -10 and 10")

    parser.add_argument("--additional_sd_divider", type=float, default=1., help="Reduce the SD on initialization of final linear layer (for CustomActor / --actor_modulates_base) further; additional divisor on SD. If --init_head_from_base, then this divides both weight and bias of final layer")
    parser.add_argument("--init_head_from_base", action="store_true", help="Init head from base model instead of using new random head")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient_accumulation_steps deepspeed config hyperparameter")

    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine_with_min_lr",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                 "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau",
                 "cosine_with_min_lr", "warmup_stable_decay"]
    )

    parser.add_argument(
        "--actor_loss_type", type=str, default="ppo",
        choices=[
            "ppo", "ctl", "ctl_nosecondterm", "sixo", "sixo_approxneg", "dpg"
        ]
    )

    parser.add_argument(
        "--critic_loss_type", type=str, default="mse",
        choices=["mse", "ctl", "mixed_ctl_mse", "sixo", "sixo_approxneg"]
    )

    parser.add_argument("--reward_transform", type=str, default=None)
    parser.add_argument("--exploration_bonus_sampling_actor", type=str, default=None, choices=["exact_count", "coin_flip"], help="Exploration bonus type for sampling_actor: 'exact_count' (requires max_new_tokens=1) or 'coin_flip' (learned via coin flip network)")
    parser.add_argument("--exploration_bonus_base_actor", type=str, default=None, choices=["exact_count", "coin_flip"], help="Exploration bonus type for base_actor (not yet implemented)")
    parser.add_argument("--bonus_alpha", type=float, default=1.0, help="Scaling factor for exploration bonus: bonus = bonus_alpha * (1/sqrt(N(x)))")
    parser.add_argument("--coin_flip_dim", type=int, default=64, help="Dimension d for coin flip vectors")
    parser.add_argument("--coin_flip_lr", type=float, default=None, help="Learning rate for coin flip network (defaults to sampling_actor_lr if None)")
    parser.add_argument("--coin_flip_normalization_momentum", type=float, default=None, help="Momentum for exponential moving average of running statistics used to normalize the exploration bonus. If None, normalization is disabled (default: None)")
    parser.add_argument("--coin_flip_head_init_std", type=float, default=0.001, help="Standard deviation for initializing the coin flip head weights (default: 0.001)")
    parser.add_argument("--frozen_prior_init_std", type=float, default=0.1, help="Standard deviation for initializing the frozen prior network weights (default: 0.1)")
    parser.add_argument("--coin_flip_linear_bias", action="store_true", default=False, help="If set to True, adds bias to the linear head for the trainable coin flip network (not the frozen prior)")
    parser.add_argument("--coin_flip_update_steps", type=int, default=1, help="Number of update steps to perform when training the coin flip network head per experience batch")
    parser.add_argument("--coin_flip_replay_buffer_batch_size", type=int, default=None, help="Batch size for sampling from coin flip replay buffer. Defaults to train_batch_size if None.")
    parser.add_argument("--train_coin_flip_before", action="store_true", default=False, help="Train coin flip network before computing exploration bonus")
    parser.add_argument("--coin_flip_first_online", action="store_true", default=False, help="For the first update step only, use the sequences that were just generated instead of randomly sampling from the replay buffer. After this step, continue sampling uniformly at random from the replay buffer.")
    parser.add_argument("--coin_flip_use_prioritization", action="store_true", default=False, help="Enable prioritized sampling for coin flip network replay buffer. Uses combination of inverse count estimate and number of times sampled.")
    parser.add_argument("--coin_flip_architecture", type=str, default="linear_head_on_static_initial_base", 
                        choices=["linear_head_on_static_initial_base", "linear_head_on_learning_base", 
                                "linear_head_on_learning_proposal", "separate_nn"], 
                        help="Architecture for coin flip network: 'linear_head_on_static_initial_base' (linear head on frozen base model copy), 'linear_head_on_learning_base' (linear head on live base_actor), 'linear_head_on_learning_proposal' (linear head on live sampling_actor), or 'separate_nn' (separate trainable and frozen networks)")

    parser.add_argument("--do_harmlessness_training", action="store_true", help="Have an outer loop where we do harmlessness training on the base/initial model. Use --num_episodes for the inner loop/proposal/twist training steps, --harmlessness_training_num_episodes for the number of outer loop steps, and --harmlessness_training_episodes_per_loop for the number of harmlessness training steps in each loop iteration. So total harmlessness_training_num_episodes * num_episodes twist/proposal updates will be done, and harmlessness_training_num_episodes * harmlessness_training_episodes_per_loop base model updates will be done)")
    parser.add_argument("--harmlessness_training_num_episodes", type=int, default=1, help="Total number of outer loop steps (where each inner loop does --num_episodes twist/proposal updates")
    parser.add_argument("--harmlessness_training_episodes_per_loop", type=int, default=1, help="Number of harmlessness training steps to do for each outer loop step")

    parser.add_argument("--sampling_target_updated_base", action="store_true", help="Only for the combined_harmlessness_trainer; if set, then the sampling_actor/twisted proposal tries to target the new, updated base model after base_actor has taken a gradient step. Otherwise, twisted proposal updates before the base_actor learns")
    parser.add_argument("--use_base_as_proposal", action="store_true", help="Only for the combined_harmlessness_trainer; if set, then the sampling_actor/twisted proposal is just equivalent to the base actor, and no updates will be done to this.")


    parser.add_argument("--only_evaluate_on_neg_data", action="store_true", help="Only evaluate on neg_data")
    parser.add_argument("--neg_data_load_path", type=str, help="Where to load the neg_data")

    parser.add_argument("--evaluate_heldout_sampling", action="store_true", help="Evaluate by doing sampling on prompts on heldout data after the training is done")
    parser.add_argument("--evaluate_on_neg_data", action="store_true", help="Evaluate on neg data (must provide --neg_data_load_path)")
    parser.add_argument("--analytic_bad_word_calc", action="store_true", help="Do analytic evaluation of bad word probabilities")
    parser.add_argument("--analytic_calc", action="store_true", help="Do analytic calculation with single token output and toxicity model")


    parser.add_argument("--sampling_iters", type=int, default=1, help="Do this many iterations of sampling over the whole dataset (only for evaluate_heldout_sampling)")


    parser.add_argument(
        "--harmlessness_training_loss_type", type=str, default=None,
        choices=[
            "reinforce", "neg_training", "neg_reinforce"
        ]
    )
    parser.add_argument("--reinforce_baseline_type", type=str, default=None, help="Only for --do_harmlessness_training")
    parser.add_argument("--neg_baseline_type", type=str, default=None, help="Only for --do_harmlessness_training")
    parser.add_argument("--reinforce_hardcoded_baseline", type=float, default=None, help="Only for --do_harmlessness_training. Value of hardcoded baseline")
    parser.add_argument("--neg_hardcoded_baseline", type=float, default=None, help="Only for --do_harmlessness_training. Value of hardcoded baseline")

    args = parser.parse_args()

    # Set analytic_batch_size to train_batch_size if not specified
    if args.analytic_batch_size is None:
        args.analytic_batch_size = args.train_batch_size

    # if not args.only_evaluate_on_neg_data and not args.evaluate_heldout_sampling:
    #     assert args.no_test_info # Right now the rewards_list is broken if you do test info instead of no_test_info
    args.no_test_info = True
    args.no_save_critic = True # save some memory usage

    args.actor_modulates_base = False
    if "policy" not in args.parameterization:
        args.actor_modulates_base = True


    if args.actor_loss_type == "ppo":
        assert args.target_dist_beta is None # We'll automatically calculate based on init_kl_coef
        if args.init_kl_coef == 0:
            args.target_dist_beta = 10000
        else:
            args.target_dist_beta = round(abs(1 / args.init_kl_coef), 3)
        print(f"target_dist_beta set to: {args.target_dist_beta}, based on init_kl_coef {args.init_kl_coef}")
        # assert math.isclose(args.init_kl_coef, abs(1 / args.target_dist_beta), abs_tol=0.01) # Because otherwise you don't have the equivalence between the RL formulation and the probabilistic inference formulation with target dist
    else:
        assert args.target_dist_beta is not None

    assert args.kl_target is None # Just use fixed KL for now

    if args.critic_pretrain is None:
        print("[Warning]: --critic_pretrain not specified, defaulting to --pretrain")
        args.critic_pretrain = args.pretrain

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    assert not args.clamp_reward # TODO I have this as default no clamp everywhere; if you want clamp, modify the code for it
    if not args.new_custom_single_prompt:
        print("[Warning] stuff like clamp reward, and probably some other things changed, and not yet tested without new_custom_single_prompt. The rm_type stuff might also need to be modified too") # TODO

    if args.actor_loss_type == "ppo":
        assert args.parameterization not in ["policy_psi_unnorm", "policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"]
    else: # Not PPO
        # assert args.actor_modulates_base # Need the twist formulation with the CustomActor for this # Now ok; can use policy parameterization directly outputting log(p psi), just need to subtract log_p then to get log_psi
        args.no_critic = True # No (PPO) critic when using the twist formulation
        if not args.do_harmlessness_training:
            assert args.init_kl_coef == 0 # Do not modify the reward with KL penalty for the twist learning losses (harmlessness training will use the KL for the base actor learning
            assert args.kl_target is None
        assert args.duplicate_rollout_batch_by > 1 # NOTE: this is also the "batch" or "number of particles" used in twist learning; for a given prompt, how many particles we use.
        assert args.parameterization != "policy" # Instead use one of "policy_psi_unnorm", "policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"

    if args.actor_loss_type == "ctl_nosecondterm":
        assert args.parameterization in ["policy_psi_q_p_s_t", "policy_psi_q_p_s_1_to_t"]

    if "indicator" in args.rm_type:
        assert args.target_dist_beta == 1 # otherwise multiply by beta screws things up
    
    if args.reward_pretrain == "indicator_bad_token":
        assert args.generate_max_len == 2, "indicator_bad_token reward_pretrain currently only supports generate_max_len == 2"

    if args.exploration_bonus_sampling_actor == "exact_count":
        assert args.generate_max_len == 1, "exploration_bonus_sampling_actor='exact_count' requires generate_max_len == 1"
    if args.exploration_bonus_base_actor == "exact_count":
        assert args.generate_max_len == 1, "exploration_bonus_base_actor='exact_count' requires generate_max_len == 1"

    if args.advantage_estimator not in ["gae"]:
        raise NotImplementedError # Not tested
        args.no_critic = True


    if args.only_evaluate_on_neg_data:
        assert args.parameterization == "policy"
        args.no_critic = True

    if args.analytic_bad_word_calc:
        assert args.rm_type in ["rlhf"] or args.reward_pretrain == "indicator_bad_token"
        assert "gpt" in args.pretrain
        # others not yet implemented/tested
        assert args.generate_max_len in [1, 2]
        assert args.new_custom_single_prompt

    if args.analytic_calc:
        assert args.rm_type in ["rlhf"]
        assert "gpt" in args.pretrain
        # others not yet implemented/tested
        assert args.generate_max_len == 1
        assert args.new_custom_single_prompt
        assert args.target_dist_beta is not None

    if args.fit_steps != 1:
        assert args.new_custom_single_prompt
        assert args.analytic_bad_word_calc or args.analytic_calc # otherwise not yet tested

    assert args.n_samples_per_prompt == 1 # Others may have weird behaviour with prompt dataset

    if args.reward_transform is not None:
        # To maintain compatibility with old setup where I would only have alpha and beta and did not separate these
        if args.rew_trans_alpha is None:
            args.rew_trans_alpha = args.alpha
        if args.rew_trans_beta is None:
            args.rew_trans_beta = args.target_dist_beta

    if args.anneal_target_dist_beta:
        assert args.start_target_dist_beta is not None

    train(args)
