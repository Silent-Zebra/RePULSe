import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import pickle

import torch
import torch.distributed as dist
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
from openrlhf.utils.utils import get_info_name_str, inspect_rewards_list, get_target_samples_filename, get_custom_prompt_with_chat_template, f_q_estimate, f_q_g_q_evaluation, f_q_g_q_evaluation_mixture, f_q_g_q_evaluation_mixture_multi_prompt, f_q_g_q_evaluation_multi_prompt, load_target_samples, compute_actor_log_probs_for_sequences, rejection_sample_for_prompt, rejection_sample_multi_prompt, generate_and_score_batch, collect_trajectory_rejection_prompt_texts
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


def _distributed_all_reduce_scalar_list(strategy, scalar_list):
    """All-reduce a list of Python scalars across ranks (mean). Returns the reduced list."""
    if strategy.world_size <= 1 or len(scalar_list) == 0:
        return scalar_list
    t = torch.tensor(scalar_list, dtype=torch.float64)
    t = strategy.all_reduce(t, op="mean")
    return t.tolist()


def _distributed_all_gather_tensor_list(strategy, tensor_list):
    """All-gather each tensor in a list across ranks (concatenate along dim 0).
    Handles None entries by skipping them. Returns the gathered list."""
    if strategy.world_size <= 1 or len(tensor_list) == 0:
        return tensor_list
    gathered = []
    for t in tensor_list:
        if t is None:
            gathered.append(None)
        else:
            gathered.append(strategy.all_gather(t))
    return gathered


def _distributed_all_gather_tensor(strategy, tensor):
    """All-gather a single tensor across ranks (concatenate along dim 0)."""
    if strategy.world_size <= 1:
        return tensor
    return strategy.all_gather(tensor)


def _distributed_all_gather_object(strategy, obj):
    """All-gather an arbitrary Python object across ranks using torch.distributed.all_gather_object.

    Returns a list of length world_size, where element i is the object from rank i.
    If world_size <= 1, returns [obj] for consistent interface.
    """
    if strategy.world_size <= 1:
        return [obj]
    output = [None] * strategy.world_size
    dist.all_gather_object(output, obj)
    return output


def _recompute_iwae_lb(f_qs):
    """Recompute IWAE lower bound from f_q values: logsumexp(f_qs) - log(n)."""
    if f_qs is None:
        return None
    return (torch.logsumexp(f_qs, dim=0) - math.log(f_qs.shape[0])).item()


def _recompute_iwae_ub(f_qs, g_qs):
    """Recompute IWAE upper bound using all gathered q samples + first target sample.

    IWAE UB = logsumexp(cat([g_qs[0:1], f_qs])) - log(N + 1)
    where g_qs[0:1] is the weight of the first (fixed) target sample, and f_qs are
    weights for all q samples. The target sample is the same across all ranks (loaded
    from a fixed file), so g_qs[0:1] is identical everywhere — no gathering needed.
    """
    if f_qs is None or g_qs is None:
        return None
    target_weight = g_qs[0:1].to(f_qs.device)
    all_weights = torch.cat([target_weight, f_qs])
    # all_weights has shape (N + 1,): one target sample + all q samples
    return (torch.logsumexp(all_weights, dim=0) - math.log(all_weights.shape[0])).item()


def _distributed_gather_f_q_g_q_lists(strategy, f_q_list, g_q_list, iwae_lbs_list, iwae_ubs_list):
    """All-gather f_q/g_q tensor lists across ranks and recompute both IWAE bounds.

    Each entry in f_q_list/g_q_list is a 1D tensor of per-sample values (or None).
    Gathering concatenates samples from all ranks, giving a larger sample set.
    Both IWAE bounds are recomputed from the gathered data; the per-rank values
    in iwae_ubs_list are discarded since this function is the single source of UBs.
    """
    gathered_f_q = _distributed_all_gather_tensor_list(strategy, f_q_list)
    gathered_g_q = _distributed_all_gather_tensor_list(strategy, g_q_list)
    new_iwae_lbs = [_recompute_iwae_lb(f) for f in gathered_f_q]
    new_iwae_ubs = [_recompute_iwae_ub(f, g) for f, g in zip(gathered_f_q, gathered_g_q)]
    return gathered_f_q, gathered_g_q, new_iwae_lbs, new_iwae_ubs


def _distributed_gather_f_q_g_q_by_prompt_lists(strategy, f_q_by_prompt_list, g_q_by_prompt_list,
                                                  iwae_lbs_by_prompt_list, iwae_ubs_by_prompt_list):
    """All-gather per-prompt f_q/g_q lists and recompute per-prompt IWAE bounds.

    Each entry in f_q_by_prompt_list is a list of tensors (one per prompt) at a given eval step.
    For each eval step and each prompt, gather the per-sample tensor across ranks.
    """
    gathered_f_q_bp = []
    gathered_g_q_bp = []
    new_iwae_lbs_bp = []
    new_iwae_ubs_bp = []
    for step_idx in range(len(f_q_by_prompt_list)):
        f_q_per_prompt = f_q_by_prompt_list[step_idx]  # list of tensors, one per prompt
        g_q_per_prompt = g_q_by_prompt_list[step_idx] if g_q_by_prompt_list else None
        gathered_f = _distributed_all_gather_tensor_list(strategy, f_q_per_prompt)
        gathered_g = _distributed_all_gather_tensor_list(strategy, g_q_per_prompt) if g_q_per_prompt is not None else [None] * len(f_q_per_prompt)
        gathered_f_q_bp.append(gathered_f)
        gathered_g_q_bp.append(gathered_g)
        new_iwae_lbs_bp.append([_recompute_iwae_lb(f) for f in gathered_f])
        new_iwae_ubs_bp.append([_recompute_iwae_ub(f, g) for f, g in zip(gathered_f, gathered_g)])
    return gathered_f_q_bp, gathered_g_q_bp, new_iwae_lbs_bp, new_iwae_ubs_bp


def _distributed_gather_by_prompt_list(strategy, by_prompt_list):
    """All-gather a per-prompt tensor list across ranks (no recomputation).

    Args:
        by_prompt_list: List over eval steps, each a list of tensors (or Nones) per prompt.

    Returns:
        Same structure, with each per-prompt tensor concatenated across ranks.
    """
    gathered = []
    for step_data in by_prompt_list:
        gathered.append(_distributed_all_gather_tensor_list(strategy, step_data))
    return gathered


def _make_f_q_tracking_lists():
    """Create a fresh set of f_q/g_q tracking lists for one eval set (fixed or heldout)."""
    return {
        "f_q_by_prompt": [],
        "g_q_by_prompt": [],
        "iwae_lbs_by_prompt": [],
        "iwae_ubs_by_prompt": [],
        "log_q_by_prompt": [],
        "log_p_by_prompt": [],
        "reward_by_prompt": [],
        "target_by_prompt": [],
        "log_q_g_q_by_prompt": [],
        "log_p_g_q_by_prompt": [],
        "reward_g_q_by_prompt": [],
        "target_g_q_by_prompt": [],
    }


def _append_f_q_result_to_tracking_lists(result, tracking_lists):
    """Append one eval step's results to the tracking lists dict."""
    for key in tracking_lists:
        tracking_lists[key].append(result[key])


def _run_f_q_g_q_eval_set(
    harmlessness_trainer, args, strategy,
    eval_prompts, eval_target_samples, tracking_lists,
    eval_prompts_random_source, n_eval_prompts,
    f_q_by_prompt_list_random, prompt_texts_random_per_timepoint,
    label="train",
):
    """Run f_q/g_q eval on Set A (fixed prompts) and optionally Set B (random coverage).

    Returns the Set A result dict (for caller to do aggregated backward-compat appends if needed).
    """
    import random as random_module
    from openrlhf.utils.utils import print_timestamp

    print_timestamp(f"per-fit-step eval: start f_q_g_q evaluation ({label} set A)")
    result = f_q_g_q_evaluation_multi_prompt(
        harmlessness_trainer, harmlessness_trainer.sampling_experience_maker_neg, args,
        eval_prompts, eval_target_samples,
        is_rank_0=strategy.is_rank_0(),
    )
    _append_f_q_result_to_tracking_lists(result, tracking_lists)
    print_timestamp(f"per-fit-step eval: end f_q_g_q evaluation ({label} set A)")

    # Set B (random prompts) - f_q only, no g_q/IWAE
    if eval_prompts_random_source is not None and f_q_by_prompt_list_random is not None:
        n = n_eval_prompts if n_eval_prompts is not None else len(eval_prompts_random_source)
        random_prompts = random_module.sample(eval_prompts_random_source, min(n, len(eval_prompts_random_source)))
        if prompt_texts_random_per_timepoint is not None:
            prompt_texts_random_per_timepoint.append(random_prompts)
        result_random = f_q_g_q_evaluation_multi_prompt(
            harmlessness_trainer, harmlessness_trainer.sampling_experience_maker_neg, args,
            random_prompts, None,
        )
        f_q_by_prompt_list_random.append(result_random["f_q_by_prompt"])

    return result


def _gather_and_aggregate_tracking_lists(strategy, tracking_lists):
    """Gather per-prompt tracking lists across ranks, compute aggregates and IWAE bounds.

    Returns a dict with gathered per-prompt data, aggregated data, and per-prompt IWAE bounds.
    """
    # f_q: gather across ranks (stochastic, each rank has independent samples)
    g_f_q_bp = _distributed_gather_by_prompt_list(strategy, tracking_lists["f_q_by_prompt"])
    g_log_q_bp = _distributed_gather_by_prompt_list(strategy, tracking_lists["log_q_by_prompt"])
    g_log_p_bp = _distributed_gather_by_prompt_list(strategy, tracking_lists["log_p_by_prompt"])
    g_reward_bp = _distributed_gather_by_prompt_list(strategy, tracking_lists["reward_by_prompt"])
    g_target_bp = _distributed_gather_by_prompt_list(strategy, tracking_lists["target_by_prompt"])
    # g_q: deterministic on target samples, rank 0 only — no gathering needed
    g_g_q_bp = tracking_lists["g_q_by_prompt"]
    g_log_q_g_q_bp = tracking_lists["log_q_g_q_by_prompt"]
    g_log_p_g_q_bp = tracking_lists["log_p_g_q_by_prompt"]
    g_reward_g_q_bp = tracking_lists["reward_g_q_by_prompt"]
    g_target_g_q_bp = tracking_lists["target_g_q_by_prompt"]
    # Aggregated
    g_f_q = [torch.cat([f for f in step if f is not None]) if any(f is not None for f in step) else None
             for step in g_f_q_bp]
    g_g_q = [torch.cat([g for g in step if g is not None]) if any(g is not None for g in step) else None
             for step in g_g_q_bp]
    g_iwae_lbs = [_recompute_iwae_lb(f) for f in g_f_q]
    g_iwae_ubs = [_recompute_iwae_ub(f, g) for f, g in zip(g_f_q, g_g_q)]
    # Per-prompt IWAE bounds
    g_iwae_lbs_bp = [[_recompute_iwae_lb(f) for f in step] for step in g_f_q_bp]
    g_iwae_ubs_bp = [[_recompute_iwae_ub(f, g) for f, g in zip(f_step, g_step)]
                     for f_step, g_step in zip(g_f_q_bp, g_g_q_bp)]
    return {
        "f_q_by_prompt": g_f_q_bp, "g_q_by_prompt": g_g_q_bp,
        "iwae_lbs_by_prompt": g_iwae_lbs_bp, "iwae_ubs_by_prompt": g_iwae_ubs_bp,
        "log_q_by_prompt": g_log_q_bp, "log_p_by_prompt": g_log_p_bp,
        "reward_by_prompt": g_reward_bp, "target_by_prompt": g_target_bp,
        "log_q_g_q_by_prompt": g_log_q_g_q_bp, "log_p_g_q_by_prompt": g_log_p_g_q_bp,
        "reward_g_q_by_prompt": g_reward_g_q_bp, "target_g_q_by_prompt": g_target_g_q_bp,
        "f_q_agg": g_f_q, "g_q_agg": g_g_q,
        "iwae_lbs_agg": g_iwae_lbs, "iwae_ubs_agg": g_iwae_ubs,
    }


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
            ds_config=strategy.get_ds_train_config(is_actor=True),
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
        # If base_actor learning rate is 0, only sample from sampling_actor (q)
        if abs(args.base_actor_learning_rate) < 1e-10:
            strategy.print("Base actor learning rate is 0. Setting neg_sample_only=True everywhere (only sampling from q).")
            strategy.print("Skipping base actor optimizer/scheduler creation to save memory (no optimizer states).")
            args.neg_sample_only = True
            base_actor_optim = None
            base_actor_scheduler = None
        else:
            base_actor_optim = strategy.create_optimizer(
                base_actor, lr=args.base_actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
            )
            strategy.print("BASE ACTOR OPTIM")
            strategy.print(base_actor_optim)
            args.neg_sample_only = False
    else:
        # Non-harmlessness training: no sampling actor, always train base actor
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
    if args.do_harmlessness_training and base_actor_optim is not None:
        base_actor_scheduler = get_scheduler(
            args.lr_scheduler,
            base_actor_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.base_actor_learning_rate * 0.1},
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
    coin_flip_trainable_module = None   # CoinFlipTrainableModule (before strategy.prepare())
    coin_flip_trainable_network = None  # DeepSpeedEngine (after strategy.prepare())
    coin_flip_trainable_optim = None
    coin_flip_trainable_scheduler = None
    coin_flip_trainable_is_module = False
    # Create q_best model for mixture proposal (frozen copy of sampling actor)
    q_best_model = None
    if args.do_harmlessness_training and getattr(args, 'mixture_proposal', False):
        q_best_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        for param in q_best_model.parameters():
            param.requires_grad = False

    coin_flip_frozen_prior_network = None
    coin_flip_tokenizer = None
    cf_strip_fn = None
    if (args.do_harmlessness_training and
        getattr(args, 'exploration_bonus_sampling_actor', None) == "coin_flip" and
        getattr(args, 'coin_flip_architecture', 'linear_head_on_static_initial_base') == "separate_nn"):
        from openrlhf.models.coin_flip_network import CoinFlipTrainableModule
        coin_flip_pretrain_path = getattr(args, 'coin_flip_pretrain', None) or args.pretrain

        # If a separate backbone is specified, check whether its tokenizer matches the main tokenizer.
        # When the vocabularies differ, sequences must be decoded with the main tokenizer and
        # re-tokenized with the coin flip tokenizer before being passed to the coin flip network.
        # Same vocab_size is NOT sufficient — independently trained BPE models can have the same
        # vocab_size but completely different token→string mappings.
        if getattr(args, 'coin_flip_pretrain', None):
            cf_tok = AutoTokenizer.from_pretrained(args.coin_flip_pretrain)
            cf_tok.padding_side = "left"
            if cf_tok.pad_token is None:
                cf_tok.pad_token = cf_tok.eos_token
            if cf_tok.get_vocab() == tokenizer.get_vocab():
                coin_flip_tokenizer = None  # same tokenizer, no conversion needed
            else:
                coin_flip_tokenizer = cf_tok
                strategy.print(
                    f"--coin_flip_pretrain tokenizer differs from --pretrain tokenizer; "
                    "sequences will be decoded and re-tokenized before the coin flip network."
                )
                # Create a strip function so _get_cf_token_ids can split prompt/response text
                # and re-apply the CF model's own chat template (rather than passing raw decoded
                # text that still contains the main model's template markers like "user\n\n").
                if getattr(args, 'apply_chat_template', False):
                    cf_strip_fn = get_strip_question_chat_template_fn(args)
                elif getattr(args, 'new_custom_single_prompt', False):
                    cf_strip_fn = get_strip_question_raw_fn(args.custom_prompt, tokenizer)
                else:
                    cf_strip_fn = None
                    strategy.print(
                        "Warning: cross-tokenizer CF model in multi-prompt mode without "
                        "--apply_chat_template; decoded text passed directly without prompt/response split."
                    )

        # Load backbone via Actor loader, then extract the raw HF model for CoinFlipTrainableModule.
        # Using Actor's loading path ensures bf16, flash attention, LoRA, etc. are applied correctly.
        _loader_actor = Actor(
            coin_flip_pretrain_path,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
        )
        coin_flip_trainable_module = CoinFlipTrainableModule(
            backbone=_loader_actor.model,
            coin_flip_dim=args.coin_flip_dim,
            head_init_std=getattr(args, 'coin_flip_head_init_std', 0.001),
            coin_flip_linear_bias=getattr(args, 'coin_flip_linear_bias', False),
        )
        del _loader_actor

        coin_flip_lr = getattr(args, 'coin_flip_lr', None) or args.actor_learning_rate
        coin_flip_trainable_optim = strategy.create_optimizer(
            coin_flip_trainable_module,
            lr=coin_flip_lr,
            betas=args.adam_betas,
            weight_decay=args.l2,
        )
        coin_flip_trainable_scheduler = get_scheduler(
            args.lr_scheduler,
            coin_flip_trainable_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": coin_flip_lr * 0.1},
        )
        coin_flip_trainable_is_module = True

        # Frozen prior network (Actor; wrapped by DeepSpeed in eval mode via strategy.prepare())
        coin_flip_frozen_prior_network = Actor(
            coin_flip_pretrain_path,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        for param in coin_flip_frozen_prior_network.parameters():
            param.requires_grad = False

    if args.do_harmlessness_training:
        # When base_actor_learning_rate=0, pass base_actor as bare model (not tuple) so it goes
        # through _ds_init_eval_model instead of _ds_init_train_model, avoiding optimizer state allocation.
        base_actor_is_eval = (base_actor_optim is None)
        base_actor_arg = base_actor if base_actor_is_eval else (base_actor, base_actor_optim, base_actor_scheduler)

        # For separate_nn: pass CoinFlipTrainableModule as a (module, optim, scheduler) tuple so it
        # goes through _ds_init_train_model and gets proper DeepSpeed gradient synchronization.
        coin_flip_trainable_arg = (
            (coin_flip_trainable_module, coin_flip_trainable_optim, coin_flip_trainable_scheduler)
            if coin_flip_trainable_is_module else None
        )

        if critic is not None:
            # prepare models/optimizers...
            prepared = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                (critic, critic_optim, critic_scheduler),
                base_actor_arg,
                reward_model,
                static_initial_model,
                coin_flip_trainable_arg,
                coin_flip_frozen_prior_network,
                q_best_model,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            if base_actor_is_eval:
                (
                    (actor, actor_optim, actor_scheduler),
                    (critic, critic_optim, critic_scheduler),
                    base_actor,
                    reward_model,
                    static_initial_model,
                    coin_flip_trainable_network,
                    coin_flip_frozen_prior_network,
                    q_best_model,
                ) = prepared
            else:
                (
                    (actor, actor_optim, actor_scheduler),
                    (critic, critic_optim, critic_scheduler),
                    (base_actor, base_actor_optim, base_actor_scheduler),
                    reward_model,
                    static_initial_model,
                    coin_flip_trainable_network,
                    coin_flip_frozen_prior_network,
                    q_best_model,
                ) = prepared
        else:
            prepared = strategy.prepare(
                (actor, actor_optim, actor_scheduler),
                base_actor_arg,
                reward_model,
                static_initial_model,
                coin_flip_trainable_arg,
                coin_flip_frozen_prior_network,
                q_best_model,
                is_rlhf=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            if base_actor_is_eval:
                (
                    (actor, actor_optim, actor_scheduler),
                    base_actor,
                    reward_model,
                    static_initial_model,
                    coin_flip_trainable_network,
                    coin_flip_frozen_prior_network,
                    q_best_model,
                ) = prepared
            else:
                (
                    (actor, actor_optim, actor_scheduler),
                    (base_actor, base_actor_optim, base_actor_scheduler),
                    reward_model,
                    static_initial_model,
                    coin_flip_trainable_network,
                    coin_flip_frozen_prior_network,
                    q_best_model,
                ) = prepared

        # For separate_nn: strategy.prepare() returns a (engine, optim, scheduler) tuple for
        # coin_flip_trainable_arg; unpack it to get the DeepSpeedEngine and updated optim/scheduler.
        if coin_flip_trainable_is_module:
            assert isinstance(coin_flip_trainable_network, tuple) and len(coin_flip_trainable_network) == 3, (
                "Expected (engine, optim, scheduler) tuple from _ds_init_train_model for coin_flip_trainable"
            )
            coin_flip_trainable_network, coin_flip_trainable_optim, coin_flip_trainable_scheduler = (
                coin_flip_trainable_network
            )

        # After prepare, copy q_current's state_dict to q_best (they start identical)
        if q_best_model is not None:
            # Unwrap DeepSpeed engines if needed
            q_current_unwrapped = strategy._unwrap_model(actor)
            q_best_unwrapped = strategy._unwrap_model(q_best_model)
            q_best_unwrapped.load_state_dict(q_current_unwrapped.state_dict())
            strategy.print("Initialized q_best_model with q_current's state_dict")

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

    # Trajectory replay validation (before checkpoint loading)
    if getattr(args, 'load_base_actor_trajectory', None):
        assert os.path.isdir(args.load_base_actor_trajectory), \
            f"Trajectory directory not found: {args.load_base_actor_trajectory}"
        assert abs(getattr(args, 'base_actor_learning_rate', 0)) < 1e-10, (
            f"--load_base_actor_trajectory requires --base_actor_learning_rate 0, "
            f"but got {args.base_actor_learning_rate}. The base actor weights are loaded "
            f"from the trajectory; gradient updates would conflict."
        )
        assert args.do_harmlessness_training, \
            "--load_base_actor_trajectory requires --do_harmlessness_training"
        strategy.print(f"Trajectory replay mode: loading base actor from {args.load_base_actor_trajectory}")
        strategy.print("Skipping normal base_actor checkpoint loading (trajectory handles it).")

    if args.do_harmlessness_training:
        # Skip normal checkpoint loading for base_actor when trajectory replay is active
        if not getattr(args, 'load_base_actor_trajectory', None):
            consumed_samples = do_load_checkpoints(args, base_actor, None, strategy)
        else:
            consumed_samples = 0
    else:
        consumed_samples = do_load_checkpoints(args, actor, critic, strategy)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_info_path, exist_ok=True)


    # Check incompatible flags before the only_evaluate_on_neg_data early exit,
    # since that path would exit before reaching the rejection_sample_true_target_only block.
    if args.only_evaluate_on_neg_data and getattr(args, 'rejection_sample_true_target_only', False):
        if not args.new_custom_single_prompt:
            raise ValueError("Cannot use --rejection_sample_true_target_only with --only_evaluate_on_neg_data when not using --new_custom_single_prompt")

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

    true_target_samples = None
    true_target_samples_by_prompt = None
    prompt_texts_from_target_samples = None
    if args.load_target_samples_name is not None:
        # Validate that reward_clamp in filename matches the current --reward_clamp arg
        import re as _re
        _rc_match = _re.search(r'_rc([\d.]+)', args.load_target_samples_name)
        if _rc_match:
            _rc_in_filename = float(_rc_match.group(1))
            if args.reward_clamp is None:
                raise ValueError(
                    f"Target samples filename contains rc={_rc_in_filename} but "
                    f"--reward_clamp is not set. This likely indicates mismatched settings. "
                    f"File: {args.load_target_samples_name}"
                )
            elif _rc_in_filename != float(args.reward_clamp):
                raise ValueError(
                    f"Mismatch between reward_clamp in target samples filename "
                    f"(rc={_rc_in_filename}) and --reward_clamp={args.reward_clamp}. "
                    f"File: {args.load_target_samples_name}"
                )
        strategy.print("Loading true target samples")
        device = next(actor.parameters()).device
        true_target_samples_by_prompt, prompt_texts_from_target_samples = load_target_samples(
            args.load_target_samples_name, device, strategy
        )
        # For backward compat with single-prompt: true_target_samples = first prompt's samples
        if args.new_custom_single_prompt:
            true_target_samples = true_target_samples_by_prompt[0]
        else:
            # Multi-prompt: set true_target_samples to first prompt's samples for backward compat
            # (used by trainers, do_evaluate_heldout_sampling, etc.)
            true_target_samples = true_target_samples_by_prompt[0] if len(true_target_samples_by_prompt) > 0 else None

    # Early exit for rejection sampling mode
    if args.rejection_sample_true_target_only:
        strategy.print("Running rejection sampling mode - skipping normal training")
        
        # Validation
        if args.rm_type != "rlhf":
            raise NotImplementedError(f"Rejection sampling currently only supports rm_type='rlhf', got '{args.rm_type}'")
        if args.reward_clamp is None and args.reward_cap is None:
            raise ValueError("Either --reward_clamp or --reward_cap must be set when using --rejection_sample_true_target_only")
        if args.target_dist_beta is None:
            raise ValueError("--target_dist_beta must be set when using --rejection_sample_true_target_only")
        
        # Ensure we have prompts_dataloader set up (if not using custom prompt)
        prompts_dataloader = None
        if not args.new_custom_single_prompt:
            # Get prompts dataset
            # NOTE: This re-calls get_prompts_data (already called at line 243), but this is a
            # one-off rejection sampling path that exits immediately after, so the redundancy is minor.
            pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)
            prompts_dataloader = strategy.setup_dataloader(
                prompts_dataset, args.micro_rollout_batch_size, True, True, drop_last=False
            )
        else:
            # For custom prompt, we'll handle it in the function
            strategy.print(f"Using custom prompt: {args.custom_prompt}")
        
        do_rejection_sampling_for_target_samples(
            args, base_actor, reward_model, tokenizer, strategy, prompts_dataloader
        )
        strategy.print("Rejection sampling complete. Exiting.")
        return

    # Early exit for reward signal analysis mode
    if args.reward_signal_analysis_only:
        strategy.print("Running reward signal analysis mode - skipping normal training")

        # Validation
        if args.rm_type != "rlhf":
            raise NotImplementedError(f"Reward signal analysis currently only supports rm_type='rlhf', got '{args.rm_type}'")
        if args.reward_clamp is None and args.reward_cap is None:
            raise ValueError("Either --reward_clamp or --reward_cap must be set when using --reward_signal_analysis_only")
        if args.target_dist_beta is None:
            raise ValueError("--target_dist_beta must be set when using --reward_signal_analysis_only")

        # Ensure we have prompts_dataloader set up (if not using custom prompt)
        prompts_dataloader = None
        if not args.new_custom_single_prompt:
            pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)
            prompts_dataloader = strategy.setup_dataloader(
                prompts_dataset, args.micro_rollout_batch_size, True, True, drop_last=False
            )
        else:
            strategy.print(f"Using custom prompt: {args.custom_prompt}")

        do_reward_signal_analysis(
            args, base_actor, reward_model, tokenizer, strategy, prompts_dataloader
        )
        strategy.print("Reward signal analysis complete. Exiting.")
        return

    if args.generate_embedding_pca_only:
        strategy.print("Running embedding PCA generation mode - skipping normal training")

        assert args.new_custom_single_prompt, "--generate_embedding_pca_only requires --new_custom_single_prompt"
        assert args.generate_max_len == 1, "--generate_embedding_pca_only requires --generate_max_len 1"
        assert args.embedding_pca_save_path is not None, "--generate_embedding_pca_only requires --embedding_pca_save_path"

        prompt_text = get_custom_prompt_with_chat_template(
            tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), strategy
        )
        strategy.print(f"Using custom prompt: {args.custom_prompt}")
        strategy.print(f"Tokenized prompt: {prompt_text}")

        generate_embedding_pca(
            model=base_actor.model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            batch_size=args.analytic_batch_size,
            save_path=args.embedding_pca_save_path,
        )
        strategy.print(f"Embedding PCA saved to {args.embedding_pca_save_path}. Exiting.")
        return

    if args.generate_embedding_tsne_only:
        strategy.print("Running embedding t-SNE generation mode - skipping normal training")

        assert args.new_custom_single_prompt, "--generate_embedding_tsne_only requires --new_custom_single_prompt"
        assert args.generate_max_len == 1, "--generate_embedding_tsne_only requires --generate_max_len 1"
        assert args.embedding_tsne_save_path is not None, "--generate_embedding_tsne_only requires --embedding_tsne_save_path"

        prompt_text = get_custom_prompt_with_chat_template(
            tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), strategy
        )
        strategy.print(f"Using custom prompt: {args.custom_prompt}")
        strategy.print(f"Tokenized prompt: {prompt_text}")

        generate_embedding_tsne(
            model=base_actor.model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            batch_size=args.analytic_batch_size,
            save_path=args.embedding_tsne_save_path,
            perplexity=args.tsne_perplexity,
            max_iter=args.tsne_max_iter,
            random_state=args.tsne_random_state,
        )
        strategy.print(f"Embedding t-SNE saved to {args.embedding_tsne_save_path}. Exiting.")
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
            sampling_actor_init_kl_coef=args.sampling_actor_init_kl_coef,
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
            reward_cap=args.reward_cap,
            rm_type=args.rm_type,
            bc_coef=args.bc_coef,
            bc_steps=args.bc_steps,
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
            coin_flip_trainable_optim=coin_flip_trainable_optim,
            coin_flip_trainable_scheduler=coin_flip_trainable_scheduler,
            q_best_model=q_best_model,
            coin_flip_tokenizer=coin_flip_tokenizer,
            cf_strip_fn=cf_strip_fn,
        )


    else:
        trainer = get_base_ppo_trainer(actor, actor_optim, actor_scheduler, args, base_actor, critic, critic_optim,
                                       critic_scheduler, ema_model, neg_data, reward_model, strategy, tokenizer,
                                       true_target_samples, vf_coef)

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
    bonus_history = []  # One snapshot per fit_step (like sis_weights_history); TODO: Add support for base_actor
    
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
            actor_model=base_actor,
        )
        strategy.print(f"Precomputed toxicity scores shape: {precomputed_toxicity_scores.shape}")
        strategy.print(f"Toxicity scores range: [{precomputed_toxicity_scores.min().item():.4f}, {precomputed_toxicity_scores.max().item():.4f}]")
        
        # Compute threshold-based bad word list if analytic_bad_word_calc is enabled
        if args.analytic_bad_word_calc:
            bad_word_tokens_ids_threshold = torch.where(precomputed_toxicity_scores < args.threshold)[0].cpu().tolist()
            strategy.print(f"Number of tokens with reward < {args.threshold}: {len(bad_word_tokens_ids_threshold)}")

    # Initialize cumulative q sample counts for all harmlessness training runs.
    # Tracks unigram token frequency across all q-generated sequences (for vocab coverage analysis,
    # checking presence of indicator tokens like swear words, etc.).
    cumulative_q_sample_counts = None
    if args.do_harmlessness_training:
        n_vocab = len(tokenizer)
        if args.analytic_calc and precomputed_toxicity_scores is not None:
            # Sanity check: toxicity score tensor should cover exactly the model's vocab
            assert precomputed_toxicity_scores.shape[0] == n_vocab, (
                f"precomputed_toxicity_scores vocab size {precomputed_toxicity_scores.shape[0]} "
                f"!= tokenizer vocab size {n_vocab}"
            )
        harmlessness_trainer.set_cumulative_q_sample_counts(n_vocab)
        # Reference for passing to do_analytic_kl_calc
        cumulative_q_sample_counts = harmlessness_trainer.cumulative_q_sample_counts

    # Initialize lists to track metrics across all fit_steps
    # These will be passed into fit() so they accumulate data across all fit steps
    iwae_lbs_list = []
    iwae_ubs_list = []
    f_q_estimates_list = []
    g_q_estimates_list = []
    rewards_list = []
    kl_vals_list = []
    entropy_list = []
    untrans_ret_list = []
    rewards_list_sampling = []
    untrans_ret_list_sampling = []
    bonus_vals_list_sampling = []
    # Mixture proposal f_q/g_q eval lists (separate diagnostic tracking for q_mix)
    f_q_mix_estimates_list = []
    g_q_mix_estimates_list = []
    iwae_mix_lbs_list = []
    iwae_mix_ubs_list = []
    # Per-fit-step heldout and f_q tracking (train_ppo owns these; populated when evaluate_heldout_sampling == "each_fit_step")
    heldout_reward_over_time_list = []
    heldout_return_over_time_list = []
    f_q_over_time_list = []
    target_samples_logprob_over_time_list = []
    # Per-prompt tracking lists for multi-prompt f_q/g_q eval
    # Fixed set (stable tracking over time)
    tracking_lists_fixed = _make_f_q_tracking_lists()
    # Random set (coverage)
    f_q_by_prompt_list_random = []
    prompt_texts_random_per_timepoint = []
    # SIS weights history: one snapshot per fit_step (matching other metrics' saving interval)
    sis_weights_history = []
    token_counts_history = []  # One snapshot per fit_step of cumulative_q_sample_counts

    # Initial point (before fit loop): heldout eval + f_q when each_fit_step + harmlessness
    _per_fit_step_heldout = (
        getattr(args, "evaluate_heldout_sampling", None) == "each_fit_step"
        and args.do_harmlessness_training
    )
    _per_fit_step_f_q_eval = (
        getattr(args, "f_q_g_q_eval", False)
        and args.do_harmlessness_training
    )

    # Build eval prompt sets (Set A: fixed, Set B: random)
    eval_prompts_fixed = None
    eval_target_samples_fixed = None  # list of tensors or None (only for prompts with target samples)
    eval_prompts_random_source = None  # full prompt list to subsample from for Set B
    n_eval_prompts = getattr(args, "n_eval_prompts_for_f_q", None)

    if _per_fit_step_f_q_eval or _per_fit_step_heldout:
        if args.new_custom_single_prompt:
            # Single-prompt mode: just use the custom prompt (backward compat)
            prompt_text_heldout = get_custom_prompt_with_chat_template(
                tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), strategy
            )
            eval_prompts_fixed = [prompt_text_heldout]
            eval_target_samples_fixed = [true_target_samples] if true_target_samples is not None else None
            # No random set for single-prompt
            eval_prompts_random_source = None
        else:
            # Multi-prompt mode: build eval prompt sets
            # Set A: Fixed prompts for stable tracking
            if prompt_texts_from_target_samples is not None:
                # Use target sample prompts (they have g_q data)
                eval_prompts_fixed = list(prompt_texts_from_target_samples)
                eval_target_samples_fixed = list(true_target_samples_by_prompt) if true_target_samples_by_prompt is not None else None
                # After load_target_samples filtering, all entries should have samples
                if eval_target_samples_fixed is not None:
                    for i, t in enumerate(eval_target_samples_fixed):
                        assert t is not None and t.numel() > 0, (
                            f"eval_target_samples_fixed[{i}] is empty — "
                            f"load_target_samples should have filtered these out"
                        )
            else:
                # Use prompts from dataset
                _, eval_prompts_dataset = get_prompts_data(args, strategy, tokenizer)
                all_eval_prompts = [eval_prompts_dataset[i] for i in range(len(eval_prompts_dataset))]

                # In trajectory replay, prioritize prompts that have rejection samples
                # so g_q can be computed at those trajectory steps.
                if getattr(args, 'load_base_actor_trajectory', None):
                    traj_prompt_set = collect_trajectory_rejection_prompt_texts(args.load_base_actor_trajectory)
                    if traj_prompt_set:
                        priority = [p for p in all_eval_prompts if p in traj_prompt_set]
                        remaining = [p for p in all_eval_prompts if p not in traj_prompt_set]
                        all_eval_prompts = priority + remaining
                        strategy.print(
                            f"Trajectory prompt prioritization: {len(priority)} dataset prompts have "
                            f"rejection samples (out of {len(traj_prompt_set)} unique in trajectory), "
                            f"{len(remaining)} remaining"
                        )

                if n_eval_prompts is not None:
                    eval_prompts_fixed = all_eval_prompts[:n_eval_prompts]
                else:
                    eval_prompts_fixed = all_eval_prompts
                eval_target_samples_fixed = None  # No target samples

            # Set B: Random prompts for coverage
            _, random_prompts_dataset = get_prompts_data(args, strategy, tokenizer)
            all_random_prompts = [random_prompts_dataset[i] for i in range(len(random_prompts_dataset))]
            # Set B is only meaningful when it differs from Set A
            if n_eval_prompts is not None and len(all_random_prompts) > n_eval_prompts:
                eval_prompts_random_source = all_random_prompts
            else:
                eval_prompts_random_source = None  # Skip Set B (same as Set A)

            strategy.print(f"Eval prompt sets: Fixed={len(eval_prompts_fixed)} prompts"
                           + (f", Random source={len(eval_prompts_random_source)} prompts" if eval_prompts_random_source else ", No random set"))

    # Build heldout eval prompt sets (for f_q/g_q evaluation on held-out data)
    eval_prompts_heldout = None
    eval_target_samples_heldout = None
    tracking_lists_heldout = None
    eval_prompts_random_source_heldout = None
    f_q_by_prompt_list_random_heldout = []
    prompt_texts_random_per_timepoint_heldout = []

    if _per_fit_step_f_q_eval and not args.new_custom_single_prompt:
        if getattr(args, 'heldout_target_samples_name', None) is not None:
            # Load heldout target samples (Set A prompts come from the file)
            heldout_samples, heldout_prompt_texts = load_target_samples(
                args.heldout_target_samples_name, torch.cuda.current_device(), strategy)
            assert heldout_prompt_texts is not None, (
                "--heldout_target_samples_name must be a v2 target samples file with prompt texts")
            eval_prompts_heldout = list(heldout_prompt_texts)
            eval_target_samples_heldout = list(heldout_samples) if heldout_samples is not None else None
            tracking_lists_heldout = _make_f_q_tracking_lists()

            # Set B source: load from --heldout_prompt_data if provided
            if getattr(args, 'heldout_prompt_data', None) is not None:
                saved_args = (args.prompt_data, args.prompt_split, args.input_key, args.input_template)
                args.prompt_data = args.heldout_prompt_data
                args.prompt_split = args.heldout_prompt_split
                args.input_key = args.heldout_input_key
                args.input_template = args.heldout_input_template
                _, heldout_random_dataset = get_prompts_data(args, strategy, tokenizer)
                args.prompt_data, args.prompt_split, args.input_key, args.input_template = saved_args
                all_heldout_prompts = [heldout_random_dataset[i] for i in range(len(heldout_random_dataset))]
                # Set B only meaningful when heldout dataset is larger than Set A
                if len(all_heldout_prompts) > len(eval_prompts_heldout):
                    eval_prompts_random_source_heldout = all_heldout_prompts

        elif getattr(args, 'heldout_prompt_data', None) is not None:
            # No target samples — load prompts from dataset (f_q only for Set A)
            saved_args = (args.prompt_data, args.prompt_split, args.input_key, args.input_template)
            args.prompt_data = args.heldout_prompt_data
            args.prompt_split = args.heldout_prompt_split
            args.input_key = args.heldout_input_key
            args.input_template = args.heldout_input_template
            _, heldout_dataset = get_prompts_data(args, strategy, tokenizer)
            args.prompt_data, args.prompt_split, args.input_key, args.input_template = saved_args
            all_heldout_prompts = [heldout_dataset[i] for i in range(len(heldout_dataset))]
            if n_eval_prompts is not None:
                eval_prompts_heldout = all_heldout_prompts[:n_eval_prompts]
            else:
                eval_prompts_heldout = all_heldout_prompts
            eval_target_samples_heldout = None
            tracking_lists_heldout = _make_f_q_tracking_lists()
            # Set B: use full dataset as random source (same logic as training set)
            if n_eval_prompts is not None and len(all_heldout_prompts) > n_eval_prompts:
                eval_prompts_random_source_heldout = all_heldout_prompts

        if eval_prompts_heldout is not None:
            strategy.print(
                f"Heldout eval: {len(eval_prompts_heldout)} prompts (Set A)"
                + (f", {sum(1 for t in eval_target_samples_heldout if t is not None and t.numel() > 0)} with target samples"
                   if eval_target_samples_heldout else ", no target samples (f_q only)")
                + (f", random source={len(eval_prompts_random_source_heldout)} prompts (Set B)"
                   if eval_prompts_random_source_heldout else ", no random set"))

    # Wire rejection_sample_prompts onto the trainer so _attempt_rejection_sampling_at_checkpoint
    # can use them for multi-prompt rejection sampling at trajectory save time.
    # This is decoupled from f_q eval: a recording run may save rejection samples without
    # running f_q evaluation.
    if (args.do_harmlessness_training
            and getattr(args, 'rejection_sample_each_save', False)
            and not args.new_custom_single_prompt):
        if eval_prompts_fixed is not None:
            # Reuse the eval prompts already built above (from target samples or dataset)
            rej_prompts = list(eval_prompts_fixed)
        else:
            # f_q eval / heldout eval not enabled, so eval_prompts_fixed wasn't built.
            # Build prompts from the same sources: target sample prompts if available, else dataset.
            if prompt_texts_from_target_samples is not None:
                rej_prompts = list(prompt_texts_from_target_samples)
            else:
                _, rej_prompts_dataset = get_prompts_data(args, strategy, tokenizer)
                rej_prompts = [rej_prompts_dataset[i] for i in range(len(rej_prompts_dataset))]
            strategy.print(f"Built rejection_sample_prompts ({len(rej_prompts)} prompts) "
                           f"independently of f_q eval")

        # Cap the number of prompts for rejection sampling at checkpoint saves
        max_prompts_rej = getattr(args, 'max_prompts_rejection_sample', 2000)
        if max_prompts_rej > 0 and len(rej_prompts) > max_prompts_rej:
            strategy.print(f"Capping rejection_sample_prompts from {len(rej_prompts)} to {max_prompts_rej} "
                           f"(--max_prompts_rejection_sample)")
            rej_prompts = rej_prompts[:max_prompts_rej]
        harmlessness_trainer.rejection_sample_prompts = rej_prompts

    # Helper to get eval target samples, updated from trajectory rejection samples if in replay mode.
    # Handles both v1 (single-prompt) and v2 (multi-prompt) rejection sample formats.
    def _get_eval_target_for_trajectory():
        if getattr(args, 'load_base_actor_trajectory', None) and args.do_harmlessness_training:
            current_rej = getattr(harmlessness_trainer, 'current_trajectory_rejection_samples', None)
            if current_rej is None:
                return None  # No rejection samples for current checkpoint

            if current_rej.get("version", 1) >= 2:
                # v2 multi-prompt format: match rejection sample prompts to eval_prompts_fixed
                rej_prompt_texts = current_rej["prompt_texts"]
                rej_samples_by_prompt = current_rej["samples_by_prompt"]
                # Build lookup from prompt text to samples
                rej_lookup = {}
                for pt, samples_list in zip(rej_prompt_texts, rej_samples_by_prompt):
                    if len(samples_list) > 0:
                        rej_lookup[pt] = torch.tensor(
                            samples_list, dtype=torch.long, device=torch.cuda.current_device()
                        )
                if not rej_lookup:
                    return None
                # Align with eval_prompts_fixed
                assert eval_prompts_fixed is not None, (
                    "eval_prompts_fixed must be set for multi-prompt trajectory replay"
                )
                result = []
                for ep in eval_prompts_fixed:
                    result.append(rej_lookup.get(ep, None))
                # Return None if no prompts matched at all
                if all(r is None for r in result):
                    return None
                return result
            else:
                # v1 single-prompt format (backward compat)
                if current_rej.get("accepted_seqs"):
                    seqs_tensor = torch.tensor(
                        current_rej["accepted_seqs"], dtype=torch.long, device=torch.cuda.current_device()
                    )
                    return [seqs_tensor]
                return None
        return eval_target_samples_fixed

    if _per_fit_step_heldout or _per_fit_step_f_q_eval:
        _run_per_fit_step_heldout_and_f_q(
            eval_prompts_fixed,
            harmlessness_trainer,
            args,
            base_actor_optim,
            base_actor_scheduler,
            base_actor,
            critic,
            critic_optim,
            critic_scheduler,
            ema_model,
            info_name_str,
            static_initial_model,
            neg_data,
            reward_model,
            strategy,
            tokenizer,
            vf_coef,
            heldout_reward_over_time_list,
            heldout_return_over_time_list,
            f_q_estimates_list,
            g_q_estimates_list,
            iwae_lbs_list,
            iwae_ubs_list,
            f_q_over_time_list,
            target_samples_logprob_over_time_list,
            eval_target_samples_fixed=_get_eval_target_for_trajectory(),
            tracking_lists_fixed=tracking_lists_fixed,
            eval_prompts_random_source=eval_prompts_random_source,
            n_eval_prompts=n_eval_prompts,
            f_q_by_prompt_list_random=f_q_by_prompt_list_random,
            prompt_texts_random_per_timepoint=prompt_texts_random_per_timepoint,
            eval_prompts_heldout=eval_prompts_heldout,
            eval_target_samples_heldout=eval_target_samples_heldout,
            tracking_lists_heldout=tracking_lists_heldout,
            eval_prompts_random_source_heldout=eval_prompts_random_source_heldout,
            f_q_by_prompt_list_random_heldout=f_q_by_prompt_list_random_heldout,
            prompt_texts_random_per_timepoint_heldout=prompt_texts_random_per_timepoint_heldout,
            f_q_mix_estimates_list=f_q_mix_estimates_list,
            g_q_mix_estimates_list=g_q_mix_estimates_list,
            iwae_mix_lbs_list=iwae_mix_lbs_list,
            iwae_mix_ubs_list=iwae_mix_ubs_list,
        )

        # Update q_best if mixture proposal is enabled with "best" strategy and g_q improved (initial eval)
        if (getattr(args, 'mixture_proposal', False)
                and getattr(args, 'mixture_other_model', 'best') == 'best'
                and g_q_estimates_list and g_q_estimates_list[-1] is not None):
            current_g_q = g_q_estimates_list[-1].mean().item()
            harmlessness_trainer.maybe_update_q_best(current_g_q)

    # Mid-fit-step f_q/g_q evaluation callback (multi-prompt only)
    mid_fit_callback = None
    f_q_eval_interval = getattr(args, "f_q_g_q_eval_interval", None)
    if (f_q_eval_interval is not None
        and _per_fit_step_f_q_eval
        and not args.new_custom_single_prompt):
        _prompts_since_last_eval = [0]  # mutable for closure

        def mid_fit_callback(n_new_prompts):
            _prompts_since_last_eval[0] += n_new_prompts
            if _prompts_since_last_eval[0] >= f_q_eval_interval:
                # Snapshot SIS weights, bonus, and token counts at each eval point (gives a
                # time series aligned with f_q/g_q evaluations, rather than once per outer fit_step)
                if (hasattr(harmlessness_trainer, 'latest_sis_weights')
                        and harmlessness_trainer.latest_sis_weights is not None):
                    sis_weights_history.append(harmlessness_trainer.latest_sis_weights.clone())
                if harmlessness_trainer.latest_bonus_val is not None:
                    bonus_history.append(harmlessness_trainer.latest_bonus_val)
                if harmlessness_trainer.cumulative_q_sample_counts is not None:
                    token_counts_history.append(harmlessness_trainer.cumulative_q_sample_counts.clone())
                strategy.print(f"[mid-fit eval] {_prompts_since_last_eval[0]} prompts "
                               f"(>= {f_q_eval_interval}), running f_q/g_q evaluation...")
                _run_per_fit_step_heldout_and_f_q(
                    eval_prompts_fixed,
                    harmlessness_trainer,
                    args,
                    base_actor_optim,
                    base_actor_scheduler,
                    base_actor,
                    critic,
                    critic_optim,
                    critic_scheduler,
                    ema_model,
                    info_name_str,
                    static_initial_model,
                    neg_data,
                    reward_model,
                    strategy,
                    tokenizer,
                    vf_coef,
                    heldout_reward_over_time_list,
                    heldout_return_over_time_list,
                    f_q_estimates_list,
                    g_q_estimates_list,
                    iwae_lbs_list,
                    iwae_ubs_list,
                    f_q_over_time_list,
                    target_samples_logprob_over_time_list,
                    eval_target_samples_fixed=_get_eval_target_for_trajectory(),
                    tracking_lists_fixed=tracking_lists_fixed,
                    eval_prompts_random_source=eval_prompts_random_source,
                    n_eval_prompts=n_eval_prompts,
                    f_q_by_prompt_list_random=f_q_by_prompt_list_random,
                    prompt_texts_random_per_timepoint=prompt_texts_random_per_timepoint,
                    eval_prompts_heldout=eval_prompts_heldout,
                    eval_target_samples_heldout=eval_target_samples_heldout,
                    tracking_lists_heldout=tracking_lists_heldout,
                    eval_prompts_random_source_heldout=eval_prompts_random_source_heldout,
                    f_q_by_prompt_list_random_heldout=f_q_by_prompt_list_random_heldout,
                    prompt_texts_random_per_timepoint_heldout=prompt_texts_random_per_timepoint_heldout,
                    f_q_mix_estimates_list=f_q_mix_estimates_list,
                    g_q_mix_estimates_list=g_q_mix_estimates_list,
                    iwae_mix_lbs_list=iwae_mix_lbs_list,
                    iwae_mix_ubs_list=iwae_mix_ubs_list,
                )
                # Update q_best if mixture proposal is enabled with "best" strategy and g_q improved (mid-fit eval)
                if (getattr(args, 'mixture_proposal', False)
                        and getattr(args, 'mixture_other_model', 'best') == 'best'
                        and g_q_estimates_list and g_q_estimates_list[-1] is not None):
                    current_g_q = g_q_estimates_list[-1].mean().item()
                    harmlessness_trainer.maybe_update_q_best(current_g_q)
                _prompts_since_last_eval[0] = 0

    from openrlhf.utils.utils import print_timestamp
    # Fit steps is kind of like a chunk for how many points we want to track progress; do x harmlessness training steps each fit step
    for fit_step in range(args.fit_steps):
        print_timestamp(f"=== fit_step {fit_step}/{args.fit_steps}: start ===")
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
                            generate_max_len=args.generate_max_len,
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
                cumulative_q_sample_counts=cumulative_q_sample_counts,
            )

        if args.do_harmlessness_training:
            strategy.print("-----HARMLESSNESS TRAINING-----")
            print_timestamp(f"fit_step {fit_step}: start harmlessness_trainer.fit()")
            # Do the harmlessness training: combined now (1 set of samples for both the base_actor and sampling_actor updates)
            if args.harmlessness_training_num_episodes > 0:
                estimates_list = harmlessness_trainer.fit(
                    args, prompts_dataloader, pretrain_dataloader, consumed_samples,
                    num_update_steps_per_episodes,
                    is_first_fit_step=(fit_step == 0),
                    rewards_list=rewards_list,
                    kl_vals_list=kl_vals_list,
                    entropy_list=entropy_list,
                    untrans_ret_list=untrans_ret_list,
                    rewards_list_sampling=rewards_list_sampling,
                    untrans_ret_list_sampling=untrans_ret_list_sampling,
                    bonus_vals_list_sampling=bonus_vals_list_sampling,
                    mid_fit_callback=mid_fit_callback,
                )
        else:
            if args.num_episodes > 0:
                estimates_list = trainer.fit(
                    args, prompts_dataloader, pretrain_dataloader, consumed_samples,
                    num_update_steps_per_episodes, true_target_samples
                )

        print_timestamp(f"fit_step {fit_step}: end harmlessness_trainer.fit()")

        # Snapshot SIS weights once per fit_step (matches saving interval of other metrics)
        if (args.do_harmlessness_training
                and hasattr(harmlessness_trainer, 'latest_sis_weights')
                and harmlessness_trainer.latest_sis_weights is not None):
            sis_weights_history.append(harmlessness_trainer.latest_sis_weights.clone())

        # Snapshot token counts once per fit_step (cumulative running total at this point in training)
        if args.do_harmlessness_training and harmlessness_trainer.cumulative_q_sample_counts is not None:
            token_counts_history.append(harmlessness_trainer.cumulative_q_sample_counts.clone())

        # Snapshot exploration bonus once per fit_step (mirrors SIS weights pattern)
        if (args.do_harmlessness_training
                and harmlessness_trainer.latest_bonus_val is not None):
            bonus_history.append(harmlessness_trainer.latest_bonus_val)

        # Lists are now passed into fit() and modified in place, so we can use them directly
        # The return value from fit() contains the same list objects for backward compatibility
        if estimates_list is not None:
            # Unpack the base estimates_list (always returned)
            if args.do_harmlessness_training:
                # CombinedHarmlessnessTrainer always returns 7 elements (f_q/g_q/iwae are owned by train_ppo and populated via f_q_g_q_evaluation calls)
                if len(estimates_list) == 7:
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list, rewards_list_sampling, untrans_ret_list_sampling, bonus_vals_list_sampling = estimates_list
                else:
                    # Old format (6 elements without bonus)
                    rewards_list, kl_vals_list, entropy_list, untrans_ret_list, rewards_list_sampling, untrans_ret_list_sampling = estimates_list
                    bonus_vals_list_sampling = None
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

            # Per-fit-step heldout + f_q (after each fit step)
            if (_per_fit_step_heldout or _per_fit_step_f_q_eval) and args.do_harmlessness_training:
                print_timestamp(f"fit_step {fit_step}: start per-fit-step eval (_run_per_fit_step_heldout_and_f_q)")
                _run_per_fit_step_heldout_and_f_q(
                    eval_prompts_fixed,
                    harmlessness_trainer,
                    args,
                    base_actor_optim,
                    base_actor_scheduler,
                    base_actor,
                    critic,
                    critic_optim,
                    critic_scheduler,
                    ema_model,
                    info_name_str,
                    static_initial_model,
                    neg_data,
                    reward_model,
                    strategy,
                    tokenizer,
                    vf_coef,
                    heldout_reward_over_time_list,
                    heldout_return_over_time_list,
                    f_q_estimates_list,
                    g_q_estimates_list,
                    iwae_lbs_list,
                    iwae_ubs_list,
                    f_q_over_time_list,
                    target_samples_logprob_over_time_list,
                    eval_target_samples_fixed=_get_eval_target_for_trajectory(),
                    tracking_lists_fixed=tracking_lists_fixed,
                    eval_prompts_random_source=eval_prompts_random_source,
                    n_eval_prompts=n_eval_prompts,
                    f_q_by_prompt_list_random=f_q_by_prompt_list_random,
                    prompt_texts_random_per_timepoint=prompt_texts_random_per_timepoint,
                    eval_prompts_heldout=eval_prompts_heldout,
                    eval_target_samples_heldout=eval_target_samples_heldout,
                    tracking_lists_heldout=tracking_lists_heldout,
                    eval_prompts_random_source_heldout=eval_prompts_random_source_heldout,
                    f_q_by_prompt_list_random_heldout=f_q_by_prompt_list_random_heldout,
                    prompt_texts_random_per_timepoint_heldout=prompt_texts_random_per_timepoint_heldout,
                    f_q_mix_estimates_list=f_q_mix_estimates_list,
                    g_q_mix_estimates_list=g_q_mix_estimates_list,
                    iwae_mix_lbs_list=iwae_mix_lbs_list,
                    iwae_mix_ubs_list=iwae_mix_ubs_list,
                )

                print_timestamp(f"fit_step {fit_step}: end per-fit-step eval")
                # Update q_best if mixture proposal is enabled with "best" strategy and g_q improved.
                # g_q is only computed on rank 0 (deterministic on fixed target samples).
                # Broadcast the scalar so all ranks make the same update decision
                # (maybe_update_q_best does load_state_dict which must be consistent).
                if (getattr(args, 'mixture_proposal', False)
                        and getattr(args, 'mixture_other_model', 'best') == 'best'
                        and g_q_estimates_list):
                    # Rank 0 has the g_q value; other ranks have None
                    if strategy.is_rank_0() and g_q_estimates_list[-1] is not None:
                        current_g_q = g_q_estimates_list[-1].mean().item()
                    else:
                        current_g_q = None
                    # Broadcast from rank 0 to all ranks
                    g_q_broadcast = [current_g_q]
                    dist.broadcast_object_list(g_q_broadcast, src=0)
                    current_g_q = g_q_broadcast[0]
                    if current_g_q is not None:
                        harmlessness_trainer.maybe_update_q_best(current_g_q)

            # Save f_q/g_q/iwae stuff separately (only if f_q_g_q_eval was done and lists are not empty).
            # Indexing: f_q_estimates_list[0] = initial (before training), f_q_estimates_list[k+1] = after fit step k.
            if args.f_q_g_q_eval and f_q_estimates_list is not None and len(f_q_estimates_list) > 0:
                # print("FINAL RESULTS IWAE LB LIST", flush=True)
                # print(iwae_lbs_list)
                # print("FINAL RESULTS IWAE UB LIST", flush=True)
                # print(iwae_ubs_list)
                # print("FINAL RESULTS F_Q", flush=True)
                # print(f_q_estimates_list)
                # print("FINAL RESULTS G_Q", flush=True)
                # print(g_q_estimates_list)
                print("SAVING F_Q/G_Q/IWAE RESULTS", flush=True)

                # Gather tracking lists across ranks (collective ops — all ranks must participate).
                gathered_fixed = _gather_and_aggregate_tracking_lists(strategy, tracking_lists_fixed)
                # Random set: f_q only, gather per-prompt tensors
                g_f_q_bp_random = []
                if f_q_by_prompt_list_random:
                    for step_data in f_q_by_prompt_list_random:
                        g_f_q_bp_random.append(_distributed_all_gather_tensor_list(strategy, step_data))

                # Heldout gathering (collective — all ranks participate even if heldout is None)
                gathered_heldout = None
                if tracking_lists_heldout is not None and len(tracking_lists_heldout["f_q_by_prompt"]) > 0:
                    gathered_heldout = _gather_and_aggregate_tracking_lists(strategy, tracking_lists_heldout)
                g_f_q_bp_random_heldout = []
                if f_q_by_prompt_list_random_heldout:
                    for step_data in f_q_by_prompt_list_random_heldout:
                        g_f_q_bp_random_heldout.append(_distributed_all_gather_tensor_list(strategy, step_data))

                if strategy.is_rank_0():
                    save_str = f"{args.save_info_path}/f_q_g_q_iwae_bounds_OpenRLHF_{info_name_str}"
                    # Always save v2 dict format with per-prompt breakdowns
                    target_to_save = {
                        "version": 2,
                        "prompt_texts_fixed": eval_prompts_fixed,
                        "prompt_texts_random_per_timepoint": prompt_texts_random_per_timepoint,
                        # Fixed set (per-prompt):
                        "f_q_by_prompt_fixed": gathered_fixed["f_q_by_prompt"],
                        "g_q_by_prompt_fixed": gathered_fixed["g_q_by_prompt"],
                        "iwae_lbs_by_prompt_fixed": gathered_fixed["iwae_lbs_by_prompt"],
                        "iwae_ubs_by_prompt_fixed": gathered_fixed["iwae_ubs_by_prompt"],
                        # Per-sample components for f_q samples (fixed set):
                        "log_q_by_prompt_fixed": gathered_fixed["log_q_by_prompt"],
                        "log_p_by_prompt_fixed": gathered_fixed["log_p_by_prompt"],
                        "reward_by_prompt_fixed": gathered_fixed["reward_by_prompt"],
                        "target_by_prompt_fixed": gathered_fixed["target_by_prompt"],
                        # Per-sample components for g_q target samples (fixed set):
                        "log_q_g_q_by_prompt_fixed": gathered_fixed["log_q_g_q_by_prompt"],
                        "log_p_g_q_by_prompt_fixed": gathered_fixed["log_p_g_q_by_prompt"],
                        "reward_g_q_by_prompt_fixed": gathered_fixed["reward_g_q_by_prompt"],
                        "target_g_q_by_prompt_fixed": gathered_fixed["target_g_q_by_prompt"],
                        # Random set (per-prompt):
                        "f_q_by_prompt_random": g_f_q_bp_random if g_f_q_bp_random else f_q_by_prompt_list_random,
                        # Aggregated (backward compat):
                        "f_q_estimates_list": gathered_fixed["f_q_agg"],
                        "g_q_estimates_list": gathered_fixed["g_q_agg"],
                        "iwae_lbs_list": gathered_fixed["iwae_lbs_agg"],
                        "iwae_ubs_list": gathered_fixed["iwae_ubs_agg"],
                    }
                    # Add heldout data if present
                    if gathered_heldout is not None:
                        target_to_save["prompt_texts_heldout"] = eval_prompts_heldout
                        target_to_save["prompt_texts_random_per_timepoint_heldout"] = prompt_texts_random_per_timepoint_heldout
                        # Heldout set (per-prompt):
                        target_to_save["f_q_by_prompt_heldout"] = gathered_heldout["f_q_by_prompt"]
                        target_to_save["g_q_by_prompt_heldout"] = gathered_heldout["g_q_by_prompt"]
                        target_to_save["iwae_lbs_by_prompt_heldout"] = gathered_heldout["iwae_lbs_by_prompt"]
                        target_to_save["iwae_ubs_by_prompt_heldout"] = gathered_heldout["iwae_ubs_by_prompt"]
                        # Per-sample components for f_q samples (heldout):
                        target_to_save["log_q_by_prompt_heldout"] = gathered_heldout["log_q_by_prompt"]
                        target_to_save["log_p_by_prompt_heldout"] = gathered_heldout["log_p_by_prompt"]
                        target_to_save["reward_by_prompt_heldout"] = gathered_heldout["reward_by_prompt"]
                        target_to_save["target_by_prompt_heldout"] = gathered_heldout["target_by_prompt"]
                        # Per-sample components for g_q target samples (heldout):
                        target_to_save["log_q_g_q_by_prompt_heldout"] = gathered_heldout["log_q_g_q_by_prompt"]
                        target_to_save["log_p_g_q_by_prompt_heldout"] = gathered_heldout["log_p_g_q_by_prompt"]
                        target_to_save["reward_g_q_by_prompt_heldout"] = gathered_heldout["reward_g_q_by_prompt"]
                        target_to_save["target_g_q_by_prompt_heldout"] = gathered_heldout["target_g_q_by_prompt"]
                        # Random set (heldout):
                        target_to_save["f_q_by_prompt_random_heldout"] = (
                            g_f_q_bp_random_heldout if g_f_q_bp_random_heldout else f_q_by_prompt_list_random_heldout)
                        # Aggregated (heldout):
                        target_to_save["f_q_estimates_list_heldout"] = gathered_heldout["f_q_agg"]
                        target_to_save["g_q_estimates_list_heldout"] = gathered_heldout["g_q_agg"]
                        target_to_save["iwae_lbs_list_heldout"] = gathered_heldout["iwae_lbs_agg"]
                        target_to_save["iwae_ubs_list_heldout"] = gathered_heldout["iwae_ubs_agg"]
                    torch.save(target_to_save, save_str)

                # Save mixture eval results separately (if enabled and non-empty)
                if getattr(args, 'mixture_eval', False) and len(f_q_mix_estimates_list) > 0:
                    # Collective ops — all ranks participate
                    g_f_q_mix, g_g_q_mix, g_iwae_mix_lbs, g_iwae_mix_ubs = _distributed_gather_f_q_g_q_lists(
                        strategy, f_q_mix_estimates_list, g_q_mix_estimates_list,
                        iwae_mix_lbs_list, iwae_mix_ubs_list)
                    if strategy.is_rank_0():
                        mix_save_str = f"{args.save_info_path}/f_q_g_q_iwae_bounds_mixeval_OpenRLHF_{info_name_str}"
                        mix_target = (
                            g_f_q_mix, g_g_q_mix,
                            g_iwae_mix_lbs, g_iwae_mix_ubs
                        )
                        torch.save(mix_target, mix_save_str)
                        strategy.print(f"Saved mixture eval results to {mix_save_str}")

            # Save the base metrics separately (always saved if not empty)
            if not args.neg_sample_only: # This stuff records it for p (base actor), so if skipping training p, this stuff will be empty
                # print("FINAL RESULTS REWARD", flush=True)
                # print(rewards_list)
                # print("FINAL RESULTS KL TO PRIOR", flush=True)
                # print(kl_vals_list)
                # print("FINAL RESULTS ENTROPY", flush=True)
                # print(entropy_list)
                # print("FINAL RESULTS UNTRANSFORMED RETURN (Including KL)", flush=True)
                # print(untrans_ret_list)
                print("SAVING BASE METRICS", flush=True)

                # Each rank has per-step means over its own prompt shard; average across ranks
                g_rewards = _distributed_all_reduce_scalar_list(strategy, rewards_list)
                g_kl_vals = _distributed_all_reduce_scalar_list(strategy, kl_vals_list)
                g_entropy = _distributed_all_reduce_scalar_list(strategy, entropy_list)
                g_untrans_ret = _distributed_all_reduce_scalar_list(strategy, untrans_ret_list)
                if strategy.is_rank_0():
                    target_to_save = (
                        g_rewards, g_kl_vals, g_entropy, g_untrans_ret
                    )
                    save_str = f"{args.save_info_path}/rew_kltoprior_ent_untransret_{info_name_str}"
                    torch.save(target_to_save, save_str)

                inspect_rewards_list(rewards_list, label="untransformed reward")

            # Save sampling metrics for harmlessness training (if available)
            if args.do_harmlessness_training and rewards_list_sampling is not None and len(rewards_list_sampling) > 0:
                # print("FINAL RESULTS SAMPLING REWARD", flush=True)
                # print(rewards_list_sampling)
                # print("FINAL RESULTS SAMPLING UNTRANSFORMED RETURN", flush=True)
                # print(untrans_ret_list_sampling)
                # if bonus_vals_list_sampling is not None and len(bonus_vals_list_sampling) > 0:
                #     print("FINAL RESULTS SAMPLING BONUS", flush=True)
                #     print(bonus_vals_list_sampling)
                print("SAVING SAMPLING METRICS", flush=True)

                # Each rank has per-step means over its own prompt shard; average across ranks
                g_rew_sampling = _distributed_all_reduce_scalar_list(strategy, rewards_list_sampling)
                g_untrans_ret_sampling = _distributed_all_reduce_scalar_list(strategy, untrans_ret_list_sampling)
                if bonus_history:
                    g_bonus_history = _distributed_all_reduce_scalar_list(strategy, bonus_history)
                    target_to_save = (
                        g_rew_sampling, g_untrans_ret_sampling, g_bonus_history,
                        True,  # bonus_unscaled: bonus values are raw (not multiplied by bonus_alpha)
                    )
                else:
                    target_to_save = (
                        g_rew_sampling, g_untrans_ret_sampling
                    )
                if strategy.is_rank_0():
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
                            generate_max_len=args.generate_max_len,
                        )
                    diff_by_bad_word_case1_list.append(diff_by_bad_word_case1)
                    diff_by_bad_word_case2_list.append(diff_by_bad_word_case2)
                    diff_by_bad_word_list.append(diff_by_bad_word)
                    max_q_exceeds_list.append(max_q_exceeds)
                    max_sigma_exceeds_list.append(max_sigma_exceeds)

        if not args.neg_sample_only:
            if rewards_list is not None and len(rewards_list) > 0:
                rewards_tensor = torch.tensor(rewards_list)
                if fit_step == 0:
                    rew_over_time_list_base.append(rewards_tensor[0].item()) # Get value at start of training
                rew_over_time_list_base.append(rewards_tensor[-1].item())

            if untrans_ret_list is not None and len(untrans_ret_list) > 0:
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
        

    # Save per-fit-step heldout and f_q over time (when each_fit_step mode was used)
    # Heldout rewards/returns and f_q are stochastic (different samples per rank); gather them.
    # target_samples_logprob is deterministic (same model weights, same target samples).
    if len(heldout_reward_over_time_list) > 0:
        g_heldout_rew = _distributed_all_gather_tensor_list(strategy, heldout_reward_over_time_list)
        g_heldout_ret = _distributed_all_gather_tensor_list(strategy, heldout_return_over_time_list)
        g_f_q_over_time = _distributed_all_gather_tensor_list(strategy, f_q_over_time_list)
        g_f_q_mean_list = [t.mean().item() if t is not None else None for t in g_f_q_over_time]
        if strategy.is_rank_0():
            save_str = f"{args.save_info_path}/heldout_over_time_{info_name_str}"
            if len(target_samples_logprob_over_time_list) > 0:
                torch.save((g_heldout_rew, g_heldout_ret, g_f_q_mean_list, target_samples_logprob_over_time_list), save_str)
            else:
                torch.save((g_heldout_rew, g_heldout_ret, g_f_q_mean_list), save_str)
            strategy.print(f"Saved heldout/f_q over time to {save_str}")

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
            cumulative_q_sample_counts=cumulative_q_sample_counts,
        )

    if args.analytic_bad_word_calc:
        if args.do_harmlessness_training:
            # Analytic results are deterministic (same model weights, same tokens) — just rank-0 guard
            if strategy.is_rank_0():
                # Save both base_actor and sampling_actor results separately
                save_str = f"{args.save_info_path}/analyticlogprob_rewsample_base_{info_name_str}"
                torch.save((total_log_prob_bad_list_base, individual_bad_word_log_probs_t0_list_base,
                           individual_bad_word_log_probs_t1_list_base, individual_bad_word_log_probs_combined_list_base,
                           rew_over_time_list_base, untrans_ret_over_time_list_base,
                           total_log_prob_bad_list_base_threshold, individual_bad_word_log_probs_t0_list_base_threshold,
                           individual_bad_word_log_probs_t1_list_base_threshold, individual_bad_word_log_probs_combined_list_base_threshold), save_str)
            print("Base actor (p) results:")
            print(total_log_prob_bad_list_base)
            print("Base actor (p) threshold-based results:")
            print(total_log_prob_bad_list_base_threshold)
            
            if strategy.is_rank_0():
                save_str = f"{args.save_info_path}/analyticlogprob_rewsample_sampling_{info_name_str}"
                torch.save((total_log_prob_bad_list_sampling, individual_bad_word_log_probs_t0_list_sampling,
                           individual_bad_word_log_probs_t1_list_sampling, individual_bad_word_log_probs_combined_list_sampling,
                           rew_over_time_list_sampling, untrans_ret_over_time_list_sampling, bonus_history,
                           total_log_prob_bad_list_sampling_threshold, individual_bad_word_log_probs_t0_list_sampling_threshold,
                           individual_bad_word_log_probs_t1_list_sampling_threshold, individual_bad_word_log_probs_combined_list_sampling_threshold,
                           True,  # bonus_unscaled: bonus values are raw (not multiplied by bonus_alpha)
                           ), save_str)
            print("Sampling actor (q) results:")
            print(total_log_prob_bad_list_sampling)
            print(rew_over_time_list_sampling)
            print(untrans_ret_over_time_list_sampling)
            print("Sampling actor (q) threshold-based results:")
            print(total_log_prob_bad_list_sampling_threshold)
        else:
            # For non-harmlessness training, use the standard lists (which are now base lists)
            if strategy.is_rank_0():
                save_str = f"{args.save_info_path}/analyticlogprob_rewsample_{info_name_str}"
                torch.save((total_log_prob_bad_list, individual_bad_word_log_probs_t0_list,
                           individual_bad_word_log_probs_t1_list, individual_bad_word_log_probs_combined_list,
                           rew_over_time_list_base, untrans_ret_over_time_list_base,
                           total_log_prob_bad_list_threshold, individual_bad_word_log_probs_t0_list_threshold,
                           individual_bad_word_log_probs_t1_list_threshold, individual_bad_word_log_probs_combined_list_threshold), save_str)
            print(total_log_prob_bad_list)
            print(rew_over_time_list_base)
            print(untrans_ret_over_time_list_base)
            print("Threshold-based results:")
            print(total_log_prob_bad_list_threshold)
        
        if total_kl_sigma_q_list and strategy.is_rank_0():
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
    
    if args.analytic_calc and strategy.is_rank_0():
        save_str = f"{args.save_info_path}/analytic_kls_toxicity_{info_name_str}"
        torch.save((total_kl_sigma_q_list_analytic, total_kl_q_sigma_list_analytic, metrics_list_analytic), save_str)
        print(f"KL sigma_q list (analytic): {total_kl_sigma_q_list_analytic}")
        print(f"KL q_sigma list (analytic): {total_kl_q_sigma_list_analytic}")
        print(f"Metrics list (analytic): {metrics_list_analytic}")

    # Save SIS weights history (one snapshot per fit_step, matching other metrics' interval).
    # Each rank has SIS weights for its own shard of prompts, so we all_gather across ranks
    # before saving to get weights for all prompts.
    if args.do_harmlessness_training and sis_weights_history:
        if strategy.world_size > 1:
            # Each entry is (num_prompts_per_rank, samples_per_prompt); gather along prompt dim
            gathered_history = [strategy.all_gather(w) for w in sis_weights_history]
        else:
            gathered_history = sis_weights_history
        if strategy.is_rank_0():
            save_str = f"{args.save_info_path}/sis_weights_history_{info_name_str}"
            torch.save(gathered_history, save_str)
            print(f"Saved SIS weights history ({len(gathered_history)} fit_steps) to {save_str}")

    # Save cumulative q token frequency counts history (one snapshot per fit_step).
    # Each rank counted its own generated tokens independently, so sum across ranks.
    if args.do_harmlessness_training and token_counts_history:
        if strategy.world_size > 1:
            gathered_counts_history = [strategy.all_reduce(c.float(), op="sum").long()
                                       for c in token_counts_history]
        else:
            gathered_counts_history = token_counts_history
        if strategy.is_rank_0():
            save_str = f"{args.save_info_path}/token_counts_history_{info_name_str}"
            torch.save(gathered_counts_history, save_str)
            final_counts = gathered_counts_history[-1]
            total_tokens = final_counts.sum().item()
            n_unique = (final_counts > 0).sum().item()
            print(f"Saved q token counts history ({len(gathered_counts_history)} fit_steps): {save_str} "
                  f"(final: total tokens={total_tokens}, unique tokens={n_unique}/{len(final_counts)})")

    if args.do_harmlessness_training:
        actor_to_test = base_actor
        initial_model = static_initial_model
    else:
        actor_to_test = actor
        initial_model = base_actor
    # Only enter this block for "end" mode or evaluate_on_neg_data. "each_fit_step" evaluation
    # was already handled during the training loop and doesn't need post-training setup.
    if (args.evaluate_heldout_sampling is not None and args.evaluate_heldout_sampling == "end") or args.evaluate_on_neg_data:
        args.rm_type = "rlhf"
        args.target_dist_beta = 1
        args.reward_transform = None
        reward_model, strip_question_chat_template_fn = get_reward_model(args, strategy)
        reward_model = strategy.prepare(
            reward_model,
            is_rlhf=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        args.no_critic = True
        critic, critic_optim, critic_scheduler = None, None, None
        actor_optim, actor_scheduler = None, None
        args.parameterization = "policy"
        # NOTE: Mutating args.prompt_data etc. so that downstream functions (e.g. get_prompts_data,
        # do_evaluate_heldout_sampling) load heldout data instead of training data. This is fragile:
        # any code after this point that calls get_prompts_data(args, ...) will get heldout data.
        args.prompt_data = args.heldout_prompt_data
        args.prompt_split = args.heldout_prompt_split
        args.input_key = args.heldout_input_key
        args.input_template = args.heldout_input_template
        args.model_eval = True

        strategy = get_strategy(args)
        strategy.setup_distributed()

        if args.evaluate_heldout_sampling == "end":
            strategy.print("DOING evaluate_heldout_sampling (end)")
            # Use existing trainer's experience_maker if available
            experience_maker = None
            generate_kwargs = None
            if args.do_harmlessness_training:
                experience_maker = harmlessness_trainer.base_experience_maker
                generate_kwargs = harmlessness_trainer.generate_kwargs
            else:
                # trainer is created in the else branch above
                experience_maker = trainer.experience_maker
                generate_kwargs = trainer.generate_kwargs
            # Build eval prompts for logprob: use target sample prompts if available,
            # else custom_prompt for single-prompt, else None (skip logprob)
            end_eval_prompts_for_logprob = None
            if prompt_texts_from_target_samples is not None:
                end_eval_prompts_for_logprob = list(prompt_texts_from_target_samples)
            elif args.new_custom_single_prompt and true_target_samples_by_prompt is not None:
                end_eval_prompts_for_logprob = [get_custom_prompt_with_chat_template(
                    tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), strategy
                )]
            do_evaluate_heldout_sampling(actor_optim, actor_scheduler, actor_to_test, args, critic, critic_optim,
                                         critic_scheduler, ema_model, info_name_str, initial_model, neg_data, reward_model,
                                         strategy, tokenizer, vf_coef, mode="end",
                                         experience_maker=experience_maker, generate_kwargs=generate_kwargs,
                                         true_target_samples_by_prompt=true_target_samples_by_prompt,
                                         eval_prompts_for_logprob=end_eval_prompts_for_logprob)

        if args.evaluate_on_neg_data:
            strategy.print("DOING evaluate_on_neg_data")
            do_evaluate_on_neg_data(actor_to_test, args, strip_question_chat_template_fn, tokenizer, info_name_str, strategy)


    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def do_analytic_kl_calc(
    base_actor, actor, args, tokenizer, prompt,
    precomputed_toxicity_scores,
    total_kl_sigma_q_list_analytic, total_kl_q_sigma_list_analytic, metrics_list_analytic,
    cumulative_q_sample_counts=None,
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
        cumulative_q_sample_counts: Optional tensor of shape (n_vocab,) tracking cumulative
            token sample counts from q. Snapshot is included in metrics_dict if provided.

    Returns:
        metrics_dict: Dictionary containing metrics from the calculation
    """
    # For harmlessness training, actor is the sampling_actor (q) and base_actor is p.
    # For non-harmlessness training, actor is the standard actor and base_actor is p.
    # Both paths use the same arguments.
    kl_sigma_q, kl_q_sigma, metrics_dict = calculate_analytic_kl_toxicity_single_token(
        model_p_for_target=base_actor.model,
        model_q=actor.model,
        tokenizer=tokenizer,
        prompt_text=prompt,
        target_dist_beta=args.target_dist_beta,
        precomputed_toxicity_scores=precomputed_toxicity_scores,
        total_kl_sigma_q_list=total_kl_sigma_q_list_analytic,
        total_kl_q_sigma_list=total_kl_q_sigma_list_analytic,
        cumulative_q_sample_counts=cumulative_q_sample_counts,
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

    assert generate_max_len == 1 or generate_max_len == 2

    model.eval()
    model.to(device)

    # --- Preprocessing ---
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    # Normalize bad word indices to tensor
    bad_word_indices_tensor = normalize_bad_word_indices(bad_word_indices, device)

    n_vocab = _get_vocab_size(_get_model_config(model))
    assert n_vocab is not None, (
        f"Could not determine vocab size from model config. "
        f"model type: {type(model)}, config type: {type(getattr(model, 'config', None))}"
    )
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
            # Note: we inline the forward + log_softmax instead of using get_next_token_log_probs,
            # because that utility squeezes the batch dim when batch_size==1, which would break
            # the 2D indexing below when the last chunk has exactly 1 element.
            outputs_t1 = model(batch_inputs_t1)
            log_probs_t1 = F.log_softmax(outputs_t1.logits[:, -1, :], dim=-1)  # Shape: (current_batch_size, n_vocab)

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
        A tuple containing:
        1. total_log_prob (float): Total log probability of a bad word appearing.
        2. individual_bad_word_log_probs_t0 (dict): Bad word token ID -> log prob at t=0.
        3. individual_bad_word_log_probs_t1 (dict): Bad word token ID -> log prob at t=1.
        4. individual_bad_word_log_probs_combined (dict): Bad word token ID -> combined log prob.
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

    return (
        total_log_prob,
        individual_bad_word_log_probs_t0,
        individual_bad_word_log_probs_t1,
        individual_bad_word_log_probs_combined
    )





@torch.no_grad() # Ensure no gradients are computed during evaluation
def calculate_analytic_kl_indicator_bad_words_both_directions(
    model_p_for_target: AutoModelForCausalLM,
    model_q: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    bad_word_indices: Union[List[int], torch.Tensor],
    batch_size: int,
    generate_max_len: int,
    total_kl_sigma_q_list: List[float],
    total_kl_q_sigma_epsq_p_list: List[float],
    precomputed_p: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None,
    precomputed_q: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None,
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

    # This function's body hardcodes Case 1 + Case 2 structure assuming 2 output tokens.
    assert generate_max_len == 2, (
        f"calculate_analytic_kl_indicator_bad_words_both_directions only supports generate_max_len=2, "
        f"got {generate_max_len}"
    )

    # Use precomputed values if available, otherwise compute them
    if precomputed_p is not None:
        prompt_ids_p, bad_word_indices_tensor, good_word_indices, log_probs_p_t0, log_probs_p_case2 = precomputed_p
        device = prompt_ids_p.device
    else:
        prompt_ids_p, bad_word_indices_tensor, good_word_indices, log_probs_p_t0, log_probs_p_case2 = \
            _compute_bad_word_sequence_log_probs(model_p_for_target, tokenizer, prompt_text, bad_word_indices, batch_size, generate_max_len)
        device = prompt_ids_p.device

    if precomputed_q is not None:
        prompt_ids_q, _, _, log_probs_q_t0, log_probs_q_case2 = precomputed_q
        assert prompt_ids_q.device == device
    else:
        prompt_ids_q, _, _, log_probs_q_t0, log_probs_q_case2 = \
            _compute_bad_word_sequence_log_probs(model_q, tokenizer, prompt_text, bad_word_indices, batch_size, generate_max_len)
        assert prompt_ids_q.device == device

    # Compute log_probs_case1 locally for KL calculations (bad word at t=0, any word at t=1)
    # This is needed for KL divergence but not returned from the shared function
    n_vocab = _get_vocab_size(_get_model_config(model_p_for_target))
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
    # Reuse outputs from the forward pass above to get next-token (t=1) log probs
    log_probs_p_t1_case1 = F.log_softmax(outputs_case1_p.logits[:, -1, :], dim=-1)  # Shape: [n_bad_words, n_vocab]
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
    # Reuse outputs from the forward pass above to get next-token (t=1) log probs
    log_probs_q_t1_case1 = F.log_softmax(outputs_case1_q.logits[:, -1, :], dim=-1)  # Shape: [n_bad_words, n_vocab]
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
    # For sigma_p at t=0, we need to normalize: log sigma_p(bad_word at t=0) = log p(bad_word at t=0) - log_Z_t0
    # where log_Z_t0 = logsumexp over all bad words at t=0 of p(bad_word at t=0)
    # Actually wait, sigma_p is normalized over all sequences with bad words, not just t=0
    # So log sigma_p(bad_word at t=0) = logsumexp over t=1 of log sigma_p(bad_word at t=0, t1)
    # = logsumexp(log_probs_sigma_p_case1[bad_idx, :])
    log_probs_sigma_p_t0 = torch.logsumexp(log_probs_sigma_p_case1, dim=1)  # Shape: (n_bad_words,) - marginal over t=1
    # Note: this shadows the full-vocab log_probs_q_t0 from _compute_bad_word_sequence_log_probs above,
    # replacing it with the bad-word-marginal (shape (n_bad_words,)). The original is not used further.
    log_probs_q_t0_marginal = torch.logsumexp(log_probs_q_case1, dim=1)  # Shape: (n_bad_words,) - marginal over t=1

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
    n_vocab = _get_vocab_size(_get_model_config(model_p_for_target))
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
        diff_by_bad_word_case1[token_id] = (log_probs_q_t0_marginal[bad_idx] - log_probs_sigma_p_t0[bad_idx]).item()
        
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


def _get_model_config(model):
    """Get the HuggingFace model config, unwrapping DeepSpeed/DataParallel wrappers if needed.

    After strategy.prepare(), Actor.model is a DeepSpeedEngine wrapping the HF model.
    DeepSpeedEngine.config is the DS config dict (not HF config), so we need to
    unwrap through .module to reach the actual HF model config.
    """
    if hasattr(model, 'module'):
        return _get_model_config(model.module)
    return model.config


def _get_vocab_size(config):
    """Get vocab size from a model config that may be a dict or config object."""
    if isinstance(config, dict):
        return config.get("vocab_size") or config.get("n_vocab")
    return getattr(config, "vocab_size", None) or getattr(config, "n_vocab", None)


@torch.no_grad()
def _collect_hidden_states(
    model,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    batch_size: int,
):
    """
    For each token in the vocabulary, constructs [prompt + token], passes through the model,
    and extracts the final-layer hidden state at the last position.

    Returns:
        hidden_states_all: (n_vocab, hidden_dim) float32 CPU tensor
        token_strings: list of n_vocab decoded token strings
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")

    model.eval()
    model.to(device)

    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)

    n_vocab = _get_vocab_size(_get_model_config(model)) or tokenizer.vocab_size
    all_token_ids = torch.arange(n_vocab, device=device)

    print(f"Generating hidden-state embeddings for {n_vocab} tokens...")

    # Collect final-layer hidden states for all tokens
    hidden_states_list = []
    for i in range(0, n_vocab, batch_size):
        batch_token_ids = all_token_ids[i : i + batch_size]
        current_batch_size = len(batch_token_ids)

        # Construct input sequences: prompt + token
        batch_inputs = torch.cat(
            (prompt_ids.repeat(current_batch_size, 1), batch_token_ids.unsqueeze(1)),
            dim=1
        )
        attention_mask = torch.ones_like(batch_inputs)

        outputs = model(input_ids=batch_inputs, attention_mask=attention_mask, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq_len, hidden_dim)
        # Take the last layer's hidden state at the last token position
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_dim)
        hidden_states_list.append(last_hidden.cpu().float())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, n_vocab)}/{n_vocab} tokens")

    hidden_states_all = torch.cat(hidden_states_list, dim=0)  # (n_vocab, hidden_dim)
    assert hidden_states_all.shape[0] == n_vocab
    print(f"Hidden states shape: {hidden_states_all.shape}")

    # Decode token strings
    print("Decoding token strings...")
    token_strings = [tokenizer.decode([i]) for i in range(n_vocab)]

    return hidden_states_all, token_strings


@torch.no_grad()
def generate_embedding_pca(
    model,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    batch_size: int,
    save_path: str,
):
    """
    Generate 2D PCA embedding of the model's hidden states for all vocab tokens.

    For each token in the vocabulary, constructs [prompt + token], passes through the model,
    and extracts the final-layer hidden state at the last position. Then runs PCA on the
    resulting (n_vocab, hidden_dim) matrix to produce 2D coordinates.

    Args:
        model: The language model (Actor.model after unwrapping)
        tokenizer: Tokenizer for the model
        prompt_text: The prompt text to prepend to each token
        batch_size: Batch size for processing
        save_path: Path to save the resulting .pt file
    """
    from sklearn.decomposition import PCA

    hidden_states_all, token_strings = _collect_hidden_states(model, tokenizer, prompt_text, batch_size)

    # Run PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(hidden_states_all.numpy())
    print(f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")

    result = {
        "pca_coords": torch.tensor(pca_coords, dtype=torch.float32),
        "token_strings": token_strings,
        "model_name": tokenizer.name_or_path,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "prompt_text": prompt_text,
    }
    torch.save(result, save_path)
    print(f"Saved PCA embedding to {save_path}")
    print(f"  pca_coords shape: {result['pca_coords'].shape}")
    print(f"  {len(token_strings)} token strings")


@torch.no_grad()
def generate_embedding_tsne(
    model,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    batch_size: int,
    save_path: str,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 1,
):
    """
    Generate 2D t-SNE embedding of the model's hidden states for all vocab tokens.

    For each token in the vocabulary, constructs [prompt + token], passes through the model,
    and extracts the final-layer hidden state at the last position. Then runs t-SNE on the
    resulting (n_vocab, hidden_dim) matrix to produce 2D coordinates.

    Args:
        model: The language model (Actor.model after unwrapping)
        tokenizer: Tokenizer for the model
        prompt_text: The prompt text to prepend to each token
        batch_size: Batch size for processing
        save_path: Path to save the resulting .pt file
        perplexity: t-SNE perplexity parameter (default 30.0)
        max_iter: Number of t-SNE optimization iterations (default 1000)
        random_state: Random seed for reproducibility (default 1)
    """
    from sklearn.manifold import TSNE

    hidden_states_all, token_strings = _collect_hidden_states(model, tokenizer, prompt_text, batch_size)

    # Run t-SNE
    print(f"Running t-SNE (perplexity={perplexity}, max_iter={max_iter}, random_state={random_state})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    tsne_coords = tsne.fit_transform(hidden_states_all.numpy())

    result = {
        "tsne_coords": torch.tensor(tsne_coords, dtype=torch.float32),
        "token_strings": token_strings,
        "model_name": tokenizer.name_or_path,
        "perplexity": perplexity,
        "max_iter": max_iter,
        "random_state": random_state,
        "prompt_text": prompt_text,
    }
    torch.save(result, save_path)
    print(f"Saved t-SNE embedding to {save_path}")
    print(f"  tsne_coords shape: {result['tsne_coords'].shape}")
    print(f"  {len(token_strings)} token strings")


@torch.no_grad()
def precompute_toxicity_scores_for_all_tokens(
    reward_model,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    batch_size: int,
    actor_model=None,
) -> torch.Tensor:
    """
    Precompute toxicity scores for all n_vocab tokens by creating sequences with each token
    at position t=0 and passing them through the reward model.

    Args:
        reward_model: The reward model to use for scoring
        tokenizer: Tokenizer for the model
        prompt_text: The prompt text
        batch_size: Batch size for processing
        actor_model: Actor model (used to determine vocab size from model.config.vocab_size
            for consistency with calculate_analytic_kl_toxicity_single_token). If None,
            falls back to tokenizer.vocab_size.

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

    # Use model.config.vocab_size for consistency with calculate_analytic_kl_toxicity_single_token,
    # which uses model_p_for_target.config.vocab_size. tokenizer.vocab_size can differ
    # (e.g., padded embedding tables, added special tokens).
    if actor_model is not None:
        n_vocab = _get_vocab_size(_get_model_config(actor_model.model)) or tokenizer.vocab_size
    else:
        n_vocab = tokenizer.vocab_size
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
    cumulative_q_sample_counts: Optional[torch.Tensor] = None,
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
        cumulative_q_sample_counts: Optional tensor of shape (n_vocab,) tracking how many times
            each token has been sampled by q up to this point. If provided, included in metrics_dict.

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

    n_vocab = _get_vocab_size(_get_model_config(model_p_for_target))

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
    
    # Create metrics dictionary with full-vocab tensors, shape (n_vocab,)
    log_probs_q_cpu = log_probs_q.detach().cpu()
    log_probs_target_cpu = log_probs_target.detach().cpu()
    log_probs_p_cpu = log_probs_p.detach().cpu()

    metrics_dict = {
        'log_probs_q_full': log_probs_q_cpu,
        'log_probs_target_full': log_probs_target_cpu,
        'log_probs_base_full': log_probs_p_cpu,
    }

    # Include cumulative sample counts snapshot if available
    if cumulative_q_sample_counts is not None:
        metrics_dict['cumulative_q_sample_counts'] = cumulative_q_sample_counts.clone()
    
    total_kl_sigma_q_list.append(kl_sigma_q)
    total_kl_q_sigma_list.append(kl_q_sigma)
    
    return kl_sigma_q, kl_q_sigma, metrics_dict


def _extract_prompts_for_sampling(args, tokenizer, strategy, prompts_dataloader):
    """Extract prompt strings for sampling-based modes (rejection sampling, analysis, etc.).

    Handles both single-prompt (via args.new_custom_single_prompt + args.custom_prompt)
    and multi-prompt (via prompts_dataloader iteration) modes.

    Args:
        args: Parsed CLI args.
        tokenizer: HF tokenizer (used for chat template application).
        strategy: Strategy object with .print() for logging.
        prompts_dataloader: DataLoader of prompt strings (used only in multi-prompt mode;
                            may be None if args.new_custom_single_prompt is True).

    Returns:
        List of prompt strings.
    """
    if args.new_custom_single_prompt:
        prompt_str = get_custom_prompt_with_chat_template(
            tokenizer, args.custom_prompt, args.apply_chat_template, strategy
        )
        prompts = [prompt_str]
        strategy.print(
            f"Using custom prompt (with chat template): {args.custom_prompt}"
            if args.apply_chat_template
            else f"Using custom prompt: {args.custom_prompt}"
        )
    else:
        prompts = []
        for batch in prompts_dataloader:
            if isinstance(batch, (list, tuple)):
                prompts.extend(batch)
            elif isinstance(batch, str):
                prompts.append(batch)
            else:
                prompts.extend([str(p) for p in batch])
        strategy.print(f"Found {len(prompts)} prompts from dataloader")
    return prompts


def do_rejection_sampling_for_target_samples(args, base_actor, reward_model, tokenizer, strategy, prompts_dataloader):
    """
    Perform rejection sampling to generate true target samples.

    Target distribution: target(x) ∝ p(x) * e^(β * r(x))
    Proposal distribution: p(x) (base actor)
    Acceptance probability: e^(β * clamped_r) / M, where M = e^(|clamp * beta|)
    """
    # Validation
    if args.rm_type != "rlhf":
        raise NotImplementedError(f"Rejection sampling currently only supports rm_type='rlhf', got '{args.rm_type}'")
    if args.reward_clamp is None and args.reward_cap is None:
        raise ValueError("Either --reward_clamp or --reward_cap must be set when using --rejection_sample_true_target_only")
    if args.target_dist_beta is None:
        raise ValueError("--target_dist_beta must be set when using --rejection_sample_true_target_only")
    if args.reward_clamp is None and args.target_dist_beta < 0:
        raise ValueError(
            "Rejection sampling with --reward_cap (one-sided clamping) is not supported with negative "
            "target_dist_beta because acceptance probabilities can exceed 1 (rewards below -reward_cap "
            "are unbounded, so e^(beta * r) is unbounded). Use --reward_clamp (symmetric clamping) instead."
        )
    if args.true_target_sample_amount <= 0:
        raise ValueError(f"--true_target_sample_amount must be > 0, got {args.true_target_sample_amount}")

    strategy.print("Starting rejection sampling for target samples...")

    # Setup
    base_actor.eval()
    reward_model.eval()

    # Generate filename
    filename = get_target_samples_filename(args)
    strategy.print(f"Will save target samples to: {filename}")

    # Calculate rejection bound in log space: log_M = |clamp/cap * beta|
    if args.reward_clamp is not None:
        clamp_val = args.reward_clamp
        clamp_beta_product = args.reward_clamp * args.target_dist_beta
    else:
        clamp_val = args.reward_cap
        clamp_beta_product = args.reward_cap * args.target_dist_beta
    log_M = abs(clamp_beta_product)
    strategy.print(f"Computing rejection bound in log space: log_M = |{clamp_val} * {args.target_dist_beta}| = |{clamp_beta_product}| = {log_M}")
    strategy.print(f"  This corresponds to M = e^({log_M}) = {torch.exp(torch.tensor(log_M, dtype=torch.float32)).item():.4e} (for reference, may be inf)")

    # Handle prompts
    prompts = _extract_prompts_for_sampling(args, tokenizer, strategy, prompts_dataloader)
    
    # Storage for accepted samples per prompt
    target_samples_by_prompt = []
    
    # Generation kwargs
    generate_kwargs = {
        "max_new_tokens": args.generate_max_len,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "temperature": 1.0,
    }
    
    # Batch size for rejection sampling
    rejection_batch_size = args.batch_size_rejection_sample if args.batch_size_rejection_sample is not None else args.duplicate_rollout_batch_by
    strategy.print(f"Rejection sampling batch size: {rejection_batch_size}")

    # Process prompts
    total_generated_all = 0
    total_accepted_all = 0
    max_gen_per_prompt = getattr(args, "max_gen_per_prompt_rejection", None)

    if args.new_custom_single_prompt:
        # Single-prompt mode: collect true_target_sample_amount for each prompt.
        # Distribute work across ranks: each rank collects ceil(target / world_size)
        # samples, then we all-gather and truncate to the exact target.
        world_size = strategy.world_size
        per_rank_target = math.ceil(args.true_target_sample_amount / world_size)
        if world_size > 1:
            strategy.print(f"Distributing rejection sampling across {world_size} ranks: "
                           f"{per_rank_target} samples/rank (target: {args.true_target_sample_amount})")

        for prompt_idx, prompt in enumerate(prompts):
            strategy.print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts)}")

            accepted_samples, accepted_rewards, total_generated = rejection_sample_for_prompt(
                actor=base_actor,
                reward_model=reward_model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_dist_beta=args.target_dist_beta,
                reward_clamp=args.reward_clamp,
                reward_cap=args.reward_cap,
                prompt_max_len=args.prompt_max_len,
                generate_kwargs=generate_kwargs,
                batch_size=rejection_batch_size,
                max_gen=max_gen_per_prompt,
                tile_prompts_fn=tile_prompts,
                target_sample_amount=per_rank_target,
                rm_type=args.rm_type,
                strategy=strategy,
            )

            # All-gather accepted samples and stats across ranks
            if world_size > 1:
                all_rank_results = _distributed_all_gather_object(
                    strategy, (accepted_samples, accepted_rewards, total_generated)
                )
                accepted_samples = []
                accepted_rewards = []
                total_generated = 0
                for rank_samples, rank_rewards, rank_generated in all_rank_results:
                    accepted_samples.extend(rank_samples)
                    accepted_rewards.extend(rank_rewards)
                    total_generated += rank_generated
                # Truncate to the exact target (may overshoot since each rank collects ceil)
                accepted_samples = accepted_samples[:args.true_target_sample_amount]
                accepted_rewards = accepted_rewards[:args.true_target_sample_amount]

            strategy.print(f"\n--- Accepted samples for prompt {prompt_idx + 1} (decoded text and clamped reward) ---")
            for i, (tokens, rew) in enumerate(zip(accepted_samples, accepted_rewards)):
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                strategy.print(f"[{i + 1}] reward (clamped) = {rew:.4f}")
                strategy.print(f"    text: {text}")
            strategy.print("---")
            target_samples_by_prompt.append(accepted_samples)
            total_accepted = len(accepted_samples)
            final_rate = total_accepted / total_generated if total_generated > 0 else 0.0
            strategy.print(f"Prompt {prompt_idx + 1} complete: {total_accepted} samples accepted "
                           f"from {total_generated} generated (acceptance rate: {final_rate:.4f})")
            total_generated_all += total_generated
            total_accepted_all += total_accepted
    else:
        # Multi-prompt mode: at most 1 accepted sample per prompt, global budget.
        # Each rank has a different shard of prompts (from DistributedSampler).
        # Each rank works on its shard, then we all-gather results across ranks.
        world_size = strategy.world_size
        per_rank_target = math.ceil(args.true_target_sample_amount / world_size)
        if world_size > 1:
            strategy.print(f"Distributing multi-prompt rejection sampling across {world_size} ranks: "
                           f"{per_rank_target} samples/rank (target: {args.true_target_sample_amount}), "
                           f"{len(prompts)} prompts on this rank")

        first_pass_limit = getattr(args, "max_gen_per_prompt_rejection_first_pass", None)
        accepted_by_prompt, rewards_by_prompt, total_gen, total_acc = rejection_sample_multi_prompt(
            actor=base_actor,
            reward_model=reward_model,
            tokenizer=tokenizer,
            prompts=prompts,
            target_dist_beta=args.target_dist_beta,
            reward_clamp=args.reward_clamp,
            reward_cap=args.reward_cap,
            prompt_max_len=args.prompt_max_len,
            generate_kwargs=generate_kwargs,
            batch_size=rejection_batch_size,
            total_target=per_rank_target,
            max_gen_per_prompt=max_gen_per_prompt,
            first_pass_limit=first_pass_limit,
            rm_type=args.rm_type,
            strategy=strategy,
        )

        # All-gather results across ranks: each rank contributes its shard of
        # (prompts, accepted_by_prompt, rewards_by_prompt, total_gen, total_acc).
        if world_size > 1:
            all_rank_results = _distributed_all_gather_object(
                strategy, (prompts, accepted_by_prompt, rewards_by_prompt, total_gen, total_acc)
            )
            # Combine all ranks' shards into global lists
            prompts = []
            accepted_by_prompt = []
            rewards_by_prompt = []
            total_gen = 0
            total_acc = 0
            for rank_prompts, rank_accepted, rank_rewards, rank_gen, rank_acc in all_rank_results:
                prompts.extend(rank_prompts)
                accepted_by_prompt.extend(rank_accepted)
                rewards_by_prompt.extend(rank_rewards)
                total_gen += rank_gen
                total_acc += rank_acc

        total_generated_all += total_gen
        total_accepted_all += total_acc

        # Store results
        for prompt_idx in range(len(prompts)):
            target_samples_by_prompt.append(accepted_by_prompt[prompt_idx])
            n_accepted = len(accepted_by_prompt[prompt_idx])
            strategy.print(f"Prompt {prompt_idx + 1}: {n_accepted} samples accepted")
            # Print decoded text for each accepted sample
            if n_accepted > 0:
                strategy.print(f"--- Accepted samples for prompt {prompt_idx + 1} ---")
                for i, (tokens, rew) in enumerate(zip(accepted_by_prompt[prompt_idx], rewards_by_prompt[prompt_idx])):
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    strategy.print(f"[{i + 1}] reward (clamped) = {rew:.4f}")
                    strategy.print(f"    text: {text}")
                strategy.print("---")

    # Save on rank 0 only
    if strategy.is_rank_0():
        if args.new_custom_single_prompt:
            # v1 format for single-prompt (backward compat)
            torch.save(target_samples_by_prompt, filename)
        else:
            # v2 format for multi-prompt: filter out prompts with 0 accepted samples
            filtered_prompts = []
            filtered_samples = []
            for prompt_text, samples in zip(prompts, target_samples_by_prompt):
                if len(samples) > 0:
                    filtered_prompts.append(prompt_text)
                    filtered_samples.append(samples)
            strategy.print(f"Saving {len(filtered_prompts)}/{len(prompts)} prompts with >=1 target sample")
            save_data = {
                "version": 2,
                "prompt_texts": filtered_prompts,
                "samples_by_prompt": filtered_samples,
            }
            torch.save(save_data, filename)
        strategy.print(f"\nSaved target samples to: {filename}")

    # Print final statistics
    overall_acceptance_rate = total_accepted_all / total_generated_all if total_generated_all > 0 else 0.0
    strategy.print(f"\nFinal statistics:")
    strategy.print(f"  Total samples generated: {total_generated_all}")
    strategy.print(f"  Total samples accepted: {total_accepted_all}")
    strategy.print(f"  Overall acceptance rate: {overall_acceptance_rate:.7f}")
    if args.new_custom_single_prompt:
        strategy.print(f"  Samples per prompt: {args.true_target_sample_amount}")
    else:
        strategy.print(f"  Total target: {args.true_target_sample_amount}")
        strategy.print(f"  Samples by prompt: {[len(s) for s in target_samples_by_prompt]}")


def do_reward_signal_analysis(args, base_actor, reward_model, tokenizer, strategy, prompts_dataloader):
    """Generate samples from the base model, score them, and compute reward signal metrics.

    This mode characterizes the reward landscape to help diagnose why exploration
    bonuses help with some reward models but not others.

    Metrics per prompt:
      - Reward statistics (mean, variance, std, histogram entropy, min, max) for both
        clamped and unclamped rewards.
      - ESS (effective sample size) of importance weights w_i = exp(beta * r_clamped_i).
        Low ESS means a few samples dominate the weights (reward signal is concentrated).
      - Diversity metrics (distinct-1/2, self-BLEU-2/4) on the top-K highest-sigma samples,
        measuring how varied the high-density region of the target distribution is.

    Results are saved to ``{save_info_path}/reward_signal_analysis_{info_name_str}.pt``
    and a summary is printed.
    """
    import numpy as np
    from openrlhf.utils.diversity_metrics import compute_distinct_n, compute_self_bleu

    # --- Setup ---
    base_actor.eval()
    reward_model.eval()

    prompts = _extract_prompts_for_sampling(args, tokenizer, strategy, prompts_dataloader)
    assert len(prompts) > 0, "No prompts found for reward signal analysis"

    generate_kwargs = {
        "max_new_tokens": args.generate_max_len,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "temperature": 1.0,
    }

    batch_size = args.batch_size_rejection_sample if args.batch_size_rejection_sample is not None else args.duplicate_rollout_batch_by
    num_samples = args.analysis_num_samples
    top_k_count = args.analysis_top_k_sigma
    reward_histogram_bins = args.analysis_reward_bins
    beta = args.target_dist_beta

    strategy.print(f"Reward signal analysis: {num_samples} samples/prompt, "
                    f"top_k_count={top_k_count}, reward_histogram_bins={reward_histogram_bins}, beta={beta}")
    strategy.print(f"  Batch size: {batch_size}, prompts: {len(prompts)}")

    per_prompt_results = []

    for prompt_idx, prompt in enumerate(prompts):
        strategy.print(f"\n=== Prompt {prompt_idx + 1}/{len(prompts)} ===")

        all_sequences = []          # list of 1-D tensors (full sequences including prompt, on CPU)
        all_unclamped_rewards = []   # list of floats (raw reward model outputs)
        all_clamped_rewards = []     # list of floats (rewards after symmetric/one-sided clamping)
        prompt_len = None            # number of prompt tokens (detected from first batch)

        total_generated = 0
        while total_generated < num_samples:
            cur_batch = min(batch_size, num_samples - total_generated)
            with torch.no_grad():
                sequences, attention_mask, action_mask, unclamped_rewards, clamped_rewards = \
                    generate_and_score_batch(
                        base_actor, reward_model, tokenizer, prompt, cur_batch,
                        args.prompt_max_len, args.reward_clamp, args.reward_cap, generate_kwargs,
                    )

            # Detect prompt length from the first batch:
            # sequences has shape [batch, prompt_len + gen_len], action_mask has shape [batch, gen_len]
            if prompt_len is None:
                prompt_len = sequences.shape[1] - action_mask.shape[1]

            for i in range(sequences.shape[0]):
                all_sequences.append(sequences[i].cpu())
                all_unclamped_rewards.append(unclamped_rewards[i].cpu().item())
                all_clamped_rewards.append(clamped_rewards[i].cpu().item())
            total_generated += sequences.shape[0]

            if total_generated % (batch_size * 10) == 0 or total_generated >= num_samples:
                strategy.print(f"  Generated {total_generated}/{num_samples} samples")

        # Trim to exactly num_samples (last batch may overshoot)
        all_sequences = all_sequences[:num_samples]
        all_unclamped_rewards = all_unclamped_rewards[:num_samples]
        all_clamped_rewards = all_clamped_rewards[:num_samples]

        unclamped_reward_arr = np.array(all_unclamped_rewards, dtype=np.float64)
        clamped_reward_arr = np.array(all_clamped_rewards, dtype=np.float64)

        # --- Reward statistics ---
        def _compute_reward_stats(reward_arr, label):
            """Compute summary statistics for a reward array.

            Returns a dict with keys prefixed by 'reward_{stat}_{label}'.
            """
            reward_mean = float(np.mean(reward_arr))
            reward_variance = float(np.var(reward_arr))
            reward_std = float(np.std(reward_arr))
            reward_min = float(np.min(reward_arr))
            reward_max = float(np.max(reward_arr))

            # Reward distribution entropy via histogram.
            # Measures how "spread out" the reward distribution is in an information-theoretic sense.
            # Higher entropy → more uniform/spread-out distribution; lower → more concentrated.
            bin_counts, _ = np.histogram(reward_arr, bins=reward_histogram_bins)
            bin_probs = bin_counts / bin_counts.sum()
            nonzero_probs = bin_probs[bin_probs > 0]
            reward_distribution_entropy = float(-np.sum(nonzero_probs * np.log(nonzero_probs)))

            return {
                f"reward_mean_{label}": reward_mean,
                f"reward_var_{label}": reward_variance,
                f"reward_std_{label}": reward_std,
                f"reward_distribution_entropy_{label}": reward_distribution_entropy,
                f"reward_min_{label}": reward_min,
                f"reward_max_{label}": reward_max,
            }

        metrics = {}
        metrics.update(_compute_reward_stats(unclamped_reward_arr, "unclamped"))
        metrics.update(_compute_reward_stats(clamped_reward_arr, "clamped"))

        # --- Effective Sample Size (ESS) based on importance weights from clamped rewards ---
        # In our target distribution sigma(x) ∝ p(x) * exp(beta * r(x)), when we sample
        # from p(x), the (unnormalized) importance weights are w_i = exp(beta * r_clamped_i).
        # ESS = 1 / sum(w_normalized_i^2), where w_normalized_i = w_i / sum(w_j).
        # ESS close to N means weights are roughly uniform (reward signal doesn't concentrate).
        # ESS close to 1 means one sample dominates (reward signal is very peaked).
        log_importance_weights = beta * clamped_reward_arr
        # Shift by max for numerical stability before exponentiating (doesn't affect normalized weights)
        log_importance_weights_shifted = log_importance_weights - np.max(log_importance_weights)
        importance_weights = np.exp(log_importance_weights_shifted)
        normalized_importance_weights = importance_weights / importance_weights.sum()
        effective_sample_size = float(1.0 / np.sum(normalized_importance_weights ** 2))
        effective_sample_size_ratio = effective_sample_size / num_samples

        metrics["effective_sample_size"] = effective_sample_size
        metrics["effective_sample_size_ratio"] = effective_sample_size_ratio
        metrics["effective_sample_size_N"] = num_samples

        # --- Top-K sigma samples for diversity analysis ---
        # log_sigma_i = beta * r_clamped_i is proportional to log(sigma(x_i) / p(x_i)).
        # We pick the samples with the *highest* log_sigma values — these are the samples
        # that sigma up-weights the most relative to p.
        # Note: with beta < 0 (typical for harmlessness), highest log_sigma corresponds
        # to the lowest (most harmful) rewards.
        log_sigma_scores = beta * clamped_reward_arr
        # argsort gives ascending order; take the last top_k_count entries for the highest values
        sorted_indices_ascending = np.argsort(log_sigma_scores)
        top_k_actual = min(top_k_count, num_samples)
        top_k_indices = sorted_indices_ascending[-top_k_actual:][::-1]  # descending order

        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id

        def _strip_trailing_special_tokens(token_ids):
            """Remove trailing pad and eos tokens from a token ID list."""
            result = list(token_ids)
            while result and result[-1] in (pad_token_id, eos_token_id):
                result.pop()
            return result

        top_k_sigma_samples = []
        top_k_response_token_lists = []  # just the response portion, for diversity metrics
        for idx in top_k_indices:
            full_token_ids = all_sequences[idx].tolist()
            response_token_ids = _strip_trailing_special_tokens(full_token_ids[prompt_len:])
            top_k_sigma_samples.append({
                "token_ids": full_token_ids,
                "response_token_ids": response_token_ids,
                "unclamped_reward": all_unclamped_rewards[idx],
                "clamped_reward": all_clamped_rewards[idx],
                "log_sigma": float(log_sigma_scores[idx]),
            })
            top_k_response_token_lists.append(response_token_ids)

        # Diversity metrics on top-K response token sequences.
        # distinct-n: fraction of unique n-grams (higher = more lexically diverse).
        # self-BLEU-n: average BLEU of each sequence against all others
        #   (higher = more similar/less diverse, lower = more diverse).
        if len(top_k_response_token_lists) >= 2:
            metrics["distinct_1"] = compute_distinct_n(top_k_response_token_lists, 1)
            metrics["distinct_2"] = compute_distinct_n(top_k_response_token_lists, 2)
            metrics["self_bleu_2"] = compute_self_bleu(top_k_response_token_lists, max_n=2)
            metrics["self_bleu_4"] = compute_self_bleu(top_k_response_token_lists, max_n=4)
        else:
            metrics["distinct_1"] = float("nan")
            metrics["distinct_2"] = float("nan")
            metrics["self_bleu_2"] = float("nan")
            metrics["self_bleu_4"] = float("nan")
        metrics["top_k_sigma_count"] = top_k_actual

        per_prompt_results.append({
            "prompt": prompt,
            "prompt_idx": prompt_idx,
            "metrics": metrics,
            "top_k_sigma_samples": top_k_sigma_samples,
        })

        # Print per-prompt summary
        strategy.print(f"  Reward (unclamped): mean={metrics['reward_mean_unclamped']:.4f}, "
                        f"std={metrics['reward_std_unclamped']:.4f}, "
                        f"entropy={metrics['reward_distribution_entropy_unclamped']:.4f}, "
                        f"range=[{metrics['reward_min_unclamped']:.4f}, {metrics['reward_max_unclamped']:.4f}]")
        strategy.print(f"  Reward (clamped):   mean={metrics['reward_mean_clamped']:.4f}, "
                        f"std={metrics['reward_std_clamped']:.4f}, "
                        f"entropy={metrics['reward_distribution_entropy_clamped']:.4f}, "
                        f"range=[{metrics['reward_min_clamped']:.4f}, {metrics['reward_max_clamped']:.4f}]")
        strategy.print(f"  ESS (effective sample size): {effective_sample_size:.2f} / {num_samples} "
                        f"= {effective_sample_size_ratio:.4f}")
        strategy.print(f"  Diversity (top-{top_k_actual} sigma): "
                        f"distinct-1={metrics['distinct_1']:.4f}, distinct-2={metrics['distinct_2']:.4f}, "
                        f"self-BLEU-2={metrics['self_bleu_2']:.4f}, self-BLEU-4={metrics['self_bleu_4']:.4f}")

        # Print top-10 sigma samples (decoded text + rewards) for qualitative inspection
        num_samples_to_show = min(10, top_k_actual)
        strategy.print(f"\n  Top-{num_samples_to_show} sigma samples (highest log_sigma = beta * r_clamped):")
        for rank, sample in enumerate(top_k_sigma_samples[:num_samples_to_show]):
            decoded_text = tokenizer.decode(sample["token_ids"], skip_special_tokens=True)
            strategy.print(f"    [{rank + 1}] log_sigma={sample['log_sigma']:.4f}, "
                            f"r_clamped={sample['clamped_reward']:.4f}, "
                            f"r_unclamped={sample['unclamped_reward']:.4f}")
            strategy.print(f"         {decoded_text}")

    # --- Aggregated metrics (across prompts) ---
    aggregated_metrics = {}
    if len(per_prompt_results) == 1:
        # Single prompt: aggregated = same as per-prompt
        aggregated_metrics = dict(per_prompt_results[0]["metrics"])
    else:
        # Multiple prompts: compute mean/std/min/max of each metric across prompts
        all_metric_keys = list(per_prompt_results[0]["metrics"].keys())
        for key in all_metric_keys:
            values = [r["metrics"][key] for r in per_prompt_results
                      if not (isinstance(r["metrics"][key], float) and math.isnan(r["metrics"][key]))]
            if len(values) == 0:
                continue
            values_arr = np.array(values, dtype=np.float64)
            aggregated_metrics[f"{key}_mean"] = float(np.mean(values_arr))
            aggregated_metrics[f"{key}_std"] = float(np.std(values_arr))
            aggregated_metrics[f"{key}_min"] = float(np.min(values_arr))
            aggregated_metrics[f"{key}_max"] = float(np.max(values_arr))

        strategy.print(f"\n=== Aggregated metrics across {len(per_prompt_results)} prompts ===")
        for key, val in aggregated_metrics.items():
            strategy.print(f"  {key}: {val:.6f}")

    # --- Save ---
    if strategy.is_rank_0():
        info_name_str = get_info_name_str(args)
        os.makedirs(args.save_info_path, exist_ok=True)
        save_path = os.path.join(args.save_info_path, f"reward_signal_analysis_{info_name_str}.pt")
        save_data = {
            "version": 1,
            "args": {
                "target_dist_beta": args.target_dist_beta,
                "reward_clamp": args.reward_clamp,
                "reward_cap": args.reward_cap,
                "analysis_num_samples": args.analysis_num_samples,
                "analysis_top_k_sigma": args.analysis_top_k_sigma,
                "analysis_reward_bins": args.analysis_reward_bins,
                "generate_max_len": args.generate_max_len,
                "pretrain": args.pretrain,
                "reward_pretrain": args.reward_pretrain,
            },
            "per_prompt_results": per_prompt_results,
            "aggregated_metrics": aggregated_metrics,
        }
        torch.save(save_data, save_path)
        strategy.print(f"\nSaved reward signal analysis to: {save_path}")


def _heldout_one_batch_make_experience(experience_maker, generate_kwargs, prompts_batch, samples_per_prompt, return_entropy_kl=False):
    """Run make_experience on one batch of prompts; return reward and return tensors (and optionally entropy, kl).

    When return_entropy_kl=True, also returns untransformed_ret and untransformed_reward (pre-reward-transform
    values stored in experience.info), useful for plotting when a reward transform is active.
    """
    experience = experience_maker.make_experience(
        prompts_batch,
        samples_per_prompt=samples_per_prompt,
        force_no_exploration_bonus=True,
        **generate_kwargs
    )
    if return_entropy_kl:
        return (
            experience.info["reward"],
            experience.info["return"],
            experience.info["entropy"],
            experience.info["kl"],
            experience.info["untransformed_ret"],
            experience.info["untransformed_reward"],
        )
    return experience.info["reward"], experience.info["return"]


def compute_target_samples_logprob(base_actor, tokenizer, prompt_text, true_target_samples, strategy, batch_size=32):
    """
    Compute total log probability of target samples under the base actor.
    
    This function constructs full sequences (prompt + target) and uses the common
    compute_actor_log_probs_for_sequences utility to compute log probabilities.
    
    Args:
        base_actor: The base actor model
        tokenizer: Tokenizer
        prompt_text: Prompt text string
        true_target_samples: Tensor of shape (num_samples, seq_len) containing target token sequences
        strategy: Strategy object for printing
        batch_size: Batch size for processing samples
    
    Returns:
        Total log probability (logsumexp of all sample log probabilities)
    """
    if true_target_samples is None:
        return None
    
    base_actor.eval()
    device = next(base_actor.parameters()).device
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prompt_tokens = [tokenizer.bos_token_id] + prompt_tokens
    
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    
    # Get token IDs for attention mask creation (if available)
    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    
    # Process samples in batches
    num_samples = true_target_samples.shape[0]
    all_log_probs = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_samples = true_target_samples[i:i+batch_size].to(device)
            batch_size_actual = batch_samples.shape[0]
            
            # Concatenate prompt with each target sample
            # Repeat prompt for each sample in batch
            prompt_batch = prompt_tensor.repeat(batch_size_actual, 1)
            
            # Concatenate prompt + target samples
            full_sequences = torch.cat([prompt_batch, batch_samples], dim=1)
            
            # Get num_actions (length of target sequence)
            num_actions = batch_samples.shape[1]
            
            # Compute log probabilities using the common utility function
            seq_log_probs, _ = compute_actor_log_probs_for_sequences(
                base_actor,
                full_sequences,
                num_actions,
                attention_mask=None,  # Let the function create it
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                shared_actorcritic=False  # base_actor is not ActorCritic
            )
            all_log_probs.append(seq_log_probs)
    
    # Concatenate all log probabilities
    all_log_probs = torch.cat(all_log_probs)
    
    # Compute logsumexp to get total log probability
    total_log_prob = torch.logsumexp(all_log_probs, dim=0)
    
    return total_log_prob.item()


def _compute_multi_prompt_target_logprob(actor, tokenizer, strategy, args,
                                          true_target_samples_by_prompt, eval_prompts_for_logprob):
    """Compute mean target samples logprob across prompts. Returns scalar or None."""
    if getattr(args, "load_target_samples_name", None) is None:
        return None
    if true_target_samples_by_prompt is None or eval_prompts_for_logprob is None:
        return None
    if len(eval_prompts_for_logprob) != len(true_target_samples_by_prompt):
        strategy.print(
            f"Warning: prompt count ({len(eval_prompts_for_logprob)}) != target samples count "
            f"({len(true_target_samples_by_prompt)}), skipping target samples logprob"
        )
        return None

    per_prompt_logprobs = []
    for prompt_text_lp, samples_lp in zip(eval_prompts_for_logprob, true_target_samples_by_prompt):
        if samples_lp is not None and samples_lp.numel() > 0:
            lp = compute_target_samples_logprob(actor, tokenizer, prompt_text_lp, samples_lp, strategy)
            if lp is not None:
                per_prompt_logprobs.append(lp)
    if not per_prompt_logprobs:
        return None
    mean_logprob = sum(per_prompt_logprobs) / len(per_prompt_logprobs)
    strategy.print(f"Target samples mean logprob across {len(per_prompt_logprobs)} prompts: {mean_logprob}")
    return mean_logprob


def do_evaluate_heldout_sampling(actor_optim, actor_scheduler, actor_to_test, args, critic, critic_optim,
                                 critic_scheduler, ema_model, info_name_str, initial_model, neg_data, reward_model,
                                 strategy, tokenizer, vf_coef,
                                 mode="end",
                                 heldout_reward_over_time_list=None,
                                 heldout_return_over_time_list=None,
                                 prompt_text=None,
                                 n_heldout_samples=None,
                                 experience_maker=None,
                                 generate_kwargs=None,
                                 target_samples_logprob_over_time_list=None,
                                 true_target_samples_by_prompt=None,
                                 eval_prompts_for_logprob=None):
    """
    Heldout evaluation: sample from actor_to_test on prompts and record reward/return.
    mode: "end" = full eval and save to file (current behaviour); "each_fit_step" = one batch, append full tensors to over-time lists.
    For mode "each_fit_step", single-prompt only: pass prompt_text and n_heldout_samples (or use args.n_heldout_samples_per_fit_step).
    If experience_maker and generate_kwargs are provided, use them instead of creating a new trainer.
    If args.load_target_samples_name is set and true_target_samples_by_prompt + eval_prompts_for_logprob are provided,
    computes and tracks log probability of target samples under the actor (per-prompt, then averaged).
    """
    n_heldout = n_heldout_samples if n_heldout_samples is not None else getattr(args, "n_heldout_samples_per_fit_step", 100)
    if experience_maker is None or generate_kwargs is None:
        raise NotImplementedError("experience_maker and generate_kwargs must be provided")

    if mode == "each_fit_step":
        # one batch, append full tensors to over-time lists
        if heldout_reward_over_time_list is None or heldout_return_over_time_list is None:
            raise ValueError("heldout_reward_over_time_list and heldout_return_over_time_list required when mode='each_fit_step'")
        if prompt_text is None:
            prompt_text = get_custom_prompt_with_chat_template(
                tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), strategy
            )
        expanded_prompts = tile_prompts([prompt_text], n_heldout)
        reward, return_ = _heldout_one_batch_make_experience(experience_maker, generate_kwargs, expanded_prompts, samples_per_prompt=1)
        heldout_reward_over_time_list.append(reward.cpu())
        heldout_return_over_time_list.append(return_.cpu())
        
        # Compute log probability of target samples if available (per-prompt, then averaged)
        target_logprob = _compute_multi_prompt_target_logprob(
            actor_to_test, tokenizer, strategy, args,
            true_target_samples_by_prompt, eval_prompts_for_logprob,
        )
        if target_logprob is not None:
            if target_samples_logprob_over_time_list is None:
                raise ValueError("target_samples_logprob_over_time_list required when load_target_samples_name is set and mode='each_fit_step'")
            target_samples_logprob_over_time_list.append(target_logprob)

        return

    # mode == "end"
    if getattr(args, "new_custom_single_prompt", False):
        # Single-prompt: one batch with n_heldout_samples_per_fit_step
        if prompt_text is None:
            prompt_text = get_custom_prompt_with_chat_template(
                tokenizer, args.custom_prompt, getattr(args, "apply_chat_template", False), strategy
            )
        expanded_prompts = tile_prompts([prompt_text], n_heldout)
        reward, return_, entropy, kls, untransformed_ret, untransformed_rew = _heldout_one_batch_make_experience(
            experience_maker, generate_kwargs, expanded_prompts, samples_per_prompt=1, return_entropy_kl=True
        )
        rewards = reward
        returns = return_
        untransformed_rets = untransformed_ret
        untransformed_rewards = untransformed_rew
    else:
        # Prompt-data: use heldout dataloader, loop over batches (one make_experience per batch)
        rewards = []
        returns = []
        entropy = []
        kls = []
        untransformed_rets_list = []
        untransformed_rewards_list = []
        pretrain_dataset, prompts_dataset = get_prompts_data(args, strategy, tokenizer)
        prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
        for i in range(args.sampling_iters):
            strategy.print(f"Sampling iter: {i}")
            for rand_prompts in prompts_dataloader:
                experience = experience_maker.make_experience(
                    rand_prompts,
                    samples_per_prompt=args.duplicate_rollout_batch_by,
                    force_no_exploration_bonus=True,
                    **generate_kwargs
                )
                rewards.append(experience.info["reward"])
                returns.append(experience.info["return"])
                entropy.append(experience.info["entropy"])
                kls.append(experience.info["kl"])
                untransformed_rets_list.append(experience.info["untransformed_ret"])
                untransformed_rewards_list.append(experience.info["untransformed_reward"])
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
        untransformed_rets = torch.cat(untransformed_rets_list)
        untransformed_rewards = torch.cat(untransformed_rewards_list)

    # Gather heldout samples across ranks (different prompts via DistributedSampler,
    # or different stochastic samples for same prompt in single-prompt mode)
    rewards = _distributed_all_gather_tensor(strategy, rewards)
    returns = _distributed_all_gather_tensor(strategy, returns)
    entropy = _distributed_all_gather_tensor(strategy, entropy)
    kls = _distributed_all_gather_tensor(strategy, kls)
    untransformed_rets = _distributed_all_gather_tensor(strategy, untransformed_rets)
    untransformed_rewards = _distributed_all_gather_tensor(strategy, untransformed_rewards)

    strategy.print(f"Average reward: {rewards.mean().item()}")
    total_samples = rewards.shape[0]
    strategy.print(f"Total number of samples drawn: {total_samples}")
    for threshold in range(-6, 1):
        outputs_below_threshold = (rewards < threshold).sum()
        strategy.print(f"Number of outputs below threshold {threshold}: {outputs_below_threshold.item()}")
        strategy.print(f"Estimate of probability of bad outputs: {(outputs_below_threshold / total_samples).item()}")
        strategy.print(
            f"Estimate of log probability of bad outputs: {(torch.log(outputs_below_threshold) - torch.log(torch.tensor(total_samples))).item()}")

    # Compute log probability of target samples if available (per-prompt, then averaged)
    # This is deterministic (same model, same target samples) — identical across ranks.
    target_samples_logprob = _compute_multi_prompt_target_logprob(
        actor_to_test, tokenizer, strategy, args,
        true_target_samples_by_prompt, eval_prompts_for_logprob,
    )

    if strategy.is_rank_0():
        save_str = f"{args.save_info_path}/info_eval_{info_name_str}"
        # Always save 7-element tuple for consistent indexing.
        # Indices: 0=rewards, 1=returns, 2=kls, 3=entropy, 4=target_samples_logprob (None if absent),
        #          5=untransformed_rets, 6=untransformed_rewards.
        # When no reward transform is active, indices 5/6 equal indices 1/0 respectively.
        torch.save((rewards, returns, kls, entropy, target_samples_logprob, untransformed_rets, untransformed_rewards), save_str)


def _run_per_fit_step_heldout_and_f_q(
    eval_prompts_fixed,
    harmlessness_trainer,
    args,
    base_actor_optim,
    base_actor_scheduler,
    base_actor,
    critic,
    critic_optim,
    critic_scheduler,
    ema_model,
    info_name_str,
    static_initial_model,
    neg_data,
    reward_model,
    strategy,
    tokenizer,
    vf_coef,
    heldout_reward_over_time_list,
    heldout_return_over_time_list,
    f_q_estimates_list,
    g_q_estimates_list,
    iwae_lbs_list,
    iwae_ubs_list,
    f_q_over_time_list,
    target_samples_logprob_over_time_list,
    # Multi-prompt kwargs
    eval_target_samples_fixed=None,
    tracking_lists_fixed=None,
    eval_prompts_random_source=None,
    n_eval_prompts=None,
    f_q_by_prompt_list_random=None,
    prompt_texts_random_per_timepoint=None,
    # Heldout eval kwargs
    eval_prompts_heldout=None,
    eval_target_samples_heldout=None,
    tracking_lists_heldout=None,
    eval_prompts_random_source_heldout=None,
    f_q_by_prompt_list_random_heldout=None,
    prompt_texts_random_per_timepoint_heldout=None,
    # Mixture proposal eval kwargs
    f_q_mix_estimates_list=None,
    g_q_mix_estimates_list=None,
    iwae_mix_lbs_list=None,
    iwae_mix_ubs_list=None,
):
    """Run heldout evaluation (each_fit_step mode) and f_q tracking; append to over-time lists.

    eval_prompts_fixed: list of prompt strings (even for single-prompt mode, wrapped in a list).
    """
    from openrlhf.utils.utils import print_timestamp

    if getattr(args, "evaluate_heldout_sampling", None) == "each_fit_step":
        print_timestamp("per-fit-step eval: start heldout evaluation")
        # For heldout sampling, use the first prompt for now (multi-prompt heldout looping is future work)
        prompt_text_for_heldout = eval_prompts_fixed[0] if eval_prompts_fixed else None
        do_evaluate_heldout_sampling(
            base_actor_optim, base_actor_scheduler, base_actor, args, critic, critic_optim,
            critic_scheduler, ema_model, info_name_str, static_initial_model, neg_data, reward_model,
            strategy, tokenizer, vf_coef,
            mode="each_fit_step",
            heldout_reward_over_time_list=heldout_reward_over_time_list,
            heldout_return_over_time_list=heldout_return_over_time_list,
            prompt_text=prompt_text_for_heldout,
            n_heldout_samples=getattr(args, "n_heldout_samples_per_fit_step", 100),
            experience_maker=harmlessness_trainer.base_experience_maker,
            generate_kwargs=harmlessness_trainer.generate_kwargs,
            target_samples_logprob_over_time_list=target_samples_logprob_over_time_list,
            true_target_samples_by_prompt=eval_target_samples_fixed,
            eval_prompts_for_logprob=eval_prompts_fixed,
        )
        print_timestamp("per-fit-step eval: end heldout evaluation")

    # When annealing target_dist_beta, temporarily restore the final beta for f_q/g_q
    # evaluation so that we always measure coverage of the final target distribution.
    # (The target samples were generated with the final beta, so log_phi must match.)
    _saved_beta = None
    if getattr(args, 'anneal_target_dist_beta', False):
        _saved_beta = harmlessness_trainer.sampling_experience_maker_neg.target_dist_beta
        harmlessness_trainer.sampling_experience_maker_neg.target_dist_beta = args.target_dist_beta

    if getattr(args, "f_q_g_q_eval", False):
        # Training set: Set A + Set B
        result_fixed = _run_f_q_g_q_eval_set(
            harmlessness_trainer, args, strategy,
            eval_prompts_fixed, eval_target_samples_fixed, tracking_lists_fixed,
            eval_prompts_random_source, n_eval_prompts,
            f_q_by_prompt_list_random, prompt_texts_random_per_timepoint,
            label="train",
        )
        # Append aggregated results (backward compat lists).
        # Always append (even None) to keep lists aligned with f_q_estimates_list,
        # so that index t in each list corresponds to the same eval step.
        f_q_agg = result_fixed["f_q_agg"]
        if f_q_agg is not None:
            f_q_estimates_list.append(f_q_agg.cpu())
            f_q_over_time_list.append(f_q_agg.cpu())
        else:
            f_q_estimates_list.append(None)
            f_q_over_time_list.append(None)
        g_q_agg = result_fixed["g_q_agg"]
        g_q_estimates_list.append(g_q_agg.cpu() if g_q_agg is not None else None)
        iwae_lbs_agg = result_fixed["iwae_lbs_agg"]
        iwae_lbs_list.append(iwae_lbs_agg)
        iwae_ubs_agg = result_fixed["iwae_ubs_agg"]
        iwae_ubs_list.append(iwae_ubs_agg)

        # Print target sample text (up to 5 samples per prompt)
        if eval_target_samples_fixed is not None:
            for prompt_i, target_samples in enumerate(eval_target_samples_fixed):
                if target_samples is not None and target_samples.numel() > 0:
                    n_print = min(5, target_samples.shape[0])
                    target_texts = tokenizer.batch_decode(target_samples[:n_print], skip_special_tokens=True)
                    print(f"Target samples text for prompt {prompt_i} ({n_print}/{target_samples.shape[0]} shown):")
                    for i, txt in enumerate(target_texts):
                        print(f"  [{i}] {txt}")

        # Heldout set: Set A + Set B
        if eval_prompts_heldout is not None and tracking_lists_heldout is not None:
            _run_f_q_g_q_eval_set(
                harmlessness_trainer, args, strategy,
                eval_prompts_heldout, eval_target_samples_heldout, tracking_lists_heldout,
                eval_prompts_random_source_heldout, n_eval_prompts,
                f_q_by_prompt_list_random_heldout, prompt_texts_random_per_timepoint_heldout,
                label="heldout",
            )

        # Mixture proposal eval (only if --mixture_eval is explicitly enabled)
        if (getattr(args, 'mixture_eval', False)
                and f_q_mix_estimates_list is not None
                and hasattr(harmlessness_trainer, 'q_best_model')
                and harmlessness_trainer.q_best_model is not None):
            print_timestamp("per-fit-step eval: start mixture proposal eval")
            f_q_g_q_evaluation_mixture_multi_prompt(
                harmlessness_trainer, harmlessness_trainer.sampling_experience_maker_neg, args,
                f_q_mix_estimates_list, g_q_mix_estimates_list,
                iwae_mix_lbs_list, iwae_mix_ubs_list,
                eval_prompts_fixed, eval_target_samples_fixed,
                harmlessness_trainer.q_best_model,
                harmlessness_trainer.log_w_current, harmlessness_trainer.log_w_best,
            )
    else:
        # No f_q_g_q_eval, just do f_q_estimate on the first prompt
        prompt_for_f_q = eval_prompts_fixed[0] if eval_prompts_fixed else None
        if prompt_for_f_q is not None:
            f_qs, *_ = f_q_estimate(
                harmlessness_trainer, harmlessness_trainer.sampling_experience_maker_neg, args, prompt_for_f_q
            )
            f_q_over_time_list.append(f_qs.cpu())

    # Restore annealed beta after f_q/g_q evaluation
    if _saved_beta is not None:
        harmlessness_trainer.sampling_experience_maker_neg.target_dist_beta = _saved_beta


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
            max_length=args.prompt_max_len + args.generate_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    strategy.print(len(neg_data))
    import re
    def strip_leading_im_end(s):
        return re.sub(r'^(<\|im_end\|>)+', '', s)

    from functools import partial
    # strip_question_chat_template_fn is None when --apply_chat_template is not set;
    # partial(None, ...) would give a confusing TypeError, so fail early with a clear message.
    if strip_question_chat_template_fn is None:
        raise ValueError(
            "strip_question_chat_template_fn is None (--apply_chat_template not set?), "
            "but it is required for do_evaluate_on_neg_data."
        )
    strip_question_chat_template_fn_for_neg_data = partial(strip_question_chat_template_fn, additional_split=True)
    for i in range(len(neg_data) // args.train_batch_size + 1):
        if (i + 1) % 10 == 0:
            strategy.print(f"BATCH {i + 1}")

        batch = neg_data[i * args.train_batch_size: (i + 1) * args.train_batch_size]
        if len(batch) == 0:
            continue

        cleaned_batch = list(map(strip_leading_im_end, batch))

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

    if strategy.is_rank_0():
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
            if args.apply_chat_template:
                strip_question_chat_template_fn = get_strip_question_chat_template_fn(args)
            reward_model = _get_reward_model_custom(
                base_pretrained_class, rm_name,
                tokenizer_base=tokenizer_base, config=config,
                rm_max_len=args.rm_max_len,
                strip_question_chat_template_fn=strip_question_chat_template_fn,
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
            if args.apply_chat_template:
                strip_question_chat_template_fn = get_strip_question_chat_template_fn(args)
            else:
                assert args.new_custom_single_prompt, (
                    "Multi-prompt mode with separatequeryanswer=True reward models (e.g., deberta) "
                    "requires --apply_chat_template. Without chat template, the code cannot determine "
                    "how to split decoded text into question and answer."
                )
                strip_question_chat_template_fn = get_strip_question_raw_fn(args.custom_prompt, tokenizer_base)
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
                         ema_model, neg_data, reward_model, strategy, tokenizer, true_target_samples, vf_coef):
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
        reward_clamp=args.reward_clamp,
        reward_cap=args.reward_cap,
        rm_type=args.rm_type,
        bc_coef=args.bc_coef,
        bc_steps=args.bc_steps,
            true_target_samples=true_target_samples,
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
    # With new_custom_single_prompt, the dataset prompts are unused (overwritten by custom_prompt
    # in the training loop), so truncate to 1 row to avoid tying iteration count to dataset size.
    if getattr(args, 'new_custom_single_prompt', False):
        prompts_data = prompts_data.select(range(1))
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


def get_strip_question_raw_fn(custom_prompt, tokenizer_base):
    """Create a strip function for non-chat-templated text using the known prompt.

    Canonicalizes the prompt via encode->decode so it matches what batch_decode produces.
    """
    prompt_tokens = tokenizer_base.encode(custom_prompt, add_special_tokens=False)
    canonical_prompt = tokenizer_base.decode(prompt_tokens, skip_special_tokens=True)

    def strip_question_fn(text, additional_split=False):
        text_stripped = text.strip()
        if text_stripped.startswith(canonical_prompt):
            question = canonical_prompt.strip()
            answer = text_stripped[len(canonical_prompt):].strip()
            return question, answer
        raise ValueError(
            f"Could not find prompt in decoded text.\n"
            f"Expected prompt: {canonical_prompt[:100]}\n"
            f"Text starts with: {text_stripped[:200]}"
        )

    return strip_question_fn


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
    parser.add_argument("--max_ckpt_mem", type=int, default=100000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # Trajectory recording & replay (for fixed base actor experiments)
    parser.add_argument("--save_trajectory_metadata", action="store_true", default=False,
                        help="Save trajectory metadata alongside base actor checkpoints for later replay")
    parser.add_argument("--rejection_sample_each_save", action="store_true", default=False,
                        help="Attempt rejection sampling after each base actor checkpoint save during recording")
    parser.add_argument("--no_save_optim", action="store_true", default=False,
                        help="Save only model weights (HuggingFace format, no optimizer states) — much smaller checkpoints")
    parser.add_argument("--load_base_actor_trajectory", type=str, default=None,
                        help="Path to saved trajectory directory (the _harml_actor dir) for replay. "
                             "Requires --base_actor_learning_rate 0.")
    parser.add_argument("--trajectory_steps_per_ckpt", type=int, default=None,
                        help="q training steps between loading successive trajectory checkpoints during replay. "
                             "If None, inferred from checkpoint tag intervals.")

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

    parser.add_argument("--bc_coef", type=float, default=0.0, help="Do behaviour cloning on exact target samples (cheating for the sake of illustrating optimality)")
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
    parser.add_argument("--sampling_actor_init_kl_coef", type=float, default=0,
        help="KL penalty coefficient for sampling actor experience maker. "
             "Must be 1 when actor_loss_type='reinforce'. Default 0 for backward compat.")
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
    parser.add_argument("--heldout_target_samples_name", type=str, default=None,
                        help="Path to target samples .pt file for held-out prompts (for f_q/g_q eval)")
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
    parser.add_argument("--reward_clamp", type=float, default=None, help="Clamp reward values between [-clamp, +clamp]. If None, no clamping is performed. Only for use with rlhf rm_type. Mutually exclusive with --reward_cap.")
    parser.add_argument("--reward_cap", type=float, default=None, help="Cap reward values at the high end only (clamp max=cap). If None, no capping. Only for use with rlhf rm_type. Mutually exclusive with --reward_clamp.")
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


    parser.add_argument("--load_target_samples_name", type=str, default=None, help="Path to load target samples from. If None, target samples are not loaded.")
    parser.add_argument("--rejection_sample_true_target_only", action="store_true", help="If set, skip normal training and only perform rejection sampling to generate true target samples. Saves samples to file. Requires --rm_type rlhf and either --reward_clamp or --reward_cap to be set.")
    parser.add_argument("--true_target_sample_amount", type=int, default=1000, help="Number of accepted samples to collect via rejection sampling. For single-prompt: per prompt. For multi-prompt: total across all prompts.")
    parser.add_argument("--max_gen_per_prompt_rejection", type=int, default=None, help="Max samples to generate per prompt during rejection sampling before giving up (default: no limit)")
    parser.add_argument("--max_gen_per_prompt_rejection_first_pass", type=int, default=None, help="Max samples to generate per prompt in the first pass through the dataset during multi-prompt rejection sampling. If not set, defaults to max_gen_per_prompt_rejection. Use a smaller value to quickly scan all prompts before spending more budget on harder ones.")
    parser.add_argument("--batch_size_rejection_sample", type=int, default=None, help="Batch size (number of sequences generated per iteration) during rejection sampling. Defaults to duplicate_rollout_batch_by if not set.")
    parser.add_argument("--max_prompts_rejection_sample", type=int, default=2000, help="Maximum number of prompts to consider during multi-prompt rejection sampling at checkpoint saves. The first N prompts from the prompt list are used. Set to -1 to use all prompts.")
    parser.add_argument("--generate_embedding_pca_only", action="store_true", default=False,
                        help="Early-exit mode: for each vocab token, pass [prompt + token] through the model, "
                             "extract final hidden state, run PCA to 2D, and save. Requires --new_custom_single_prompt "
                             "and --generate_max_len 1.")
    parser.add_argument("--embedding_pca_save_path", type=str, default=None,
                        help="Path to save the embedding PCA .pt file (used with --generate_embedding_pca_only)")
    parser.add_argument("--generate_embedding_tsne_only", action="store_true", default=False,
                        help="Early-exit mode: for each vocab token, pass [prompt + token] through the model, "
                             "extract final hidden state, run t-SNE to 2D, and save. Requires --new_custom_single_prompt "
                             "and --generate_max_len 1.")
    parser.add_argument("--embedding_tsne_save_path", type=str, default=None,
                        help="Path to save the embedding t-SNE .pt file (used with --generate_embedding_tsne_only)")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0,
                        help="t-SNE perplexity parameter (used with --generate_embedding_tsne_only)")
    parser.add_argument("--tsne_max_iter", type=int, default=1000,
                        help="Number of t-SNE optimization iterations (used with --generate_embedding_tsne_only)")
    parser.add_argument("--tsne_random_state", type=int, default=1,
                        help="Random seed for t-SNE (used with --generate_embedding_tsne_only)")
    parser.add_argument("--reward_signal_analysis_only", action="store_true", default=False,
                        help="Early-exit mode: generate samples from the base model, score with reward model, "
                             "and compute reward signal metrics (stats, ESS, diversity). Exits before training.")
    parser.add_argument("--analysis_num_samples", type=int, default=1000,
                        help="Number of samples to generate per prompt for reward signal analysis.")
    parser.add_argument("--analysis_top_k_sigma", type=int, default=100,
                        help="Number of top-sigma samples to keep for diversity metrics in reward signal analysis.")
    parser.add_argument("--analysis_reward_bins", type=int, default=50,
                        help="Number of histogram bins for reward entropy computation in reward signal analysis.")
    parser.add_argument("--save_info_path", type=str, default="./info")
    parser.add_argument("--n_samples_for_f_q_g_q", type=int, default=500, help="Number of samples to use for f_q/g_q evaluation (only for f_q_g_q_eval)")
    parser.add_argument("--n_eval_prompts_for_f_q", type=int, default=None, help="Number of prompts to subsample for f_q/g_q eval (default: all prompts)")
    parser.add_argument("--n_prompts_f_q_g_q", type=int, default=None, help="Number of prompts to batch together for f_q/g_q eval in multi-prompt mode. If not set, uses per-prompt for-loop (current behavior).")


    parser.add_argument("--update_steps_per_episode", type=int, default=1, help="Number of gradient updates (PPO loss outer loop) per episode")
    parser.add_argument("--exp_num_twist_updates", action="store_true", help="Use an exponentially increasing power of twist updates (base 2) instead of a set number of twist updates per epoch")
    parser.add_argument("--no_test_info", action="store_true", help="don't do the f_q_g_q stuff")
    parser.add_argument("--f_q_g_q_eval", action="store_true", default=False, help="Enable f_q/g_q/IWAE evaluation (supports both single-prompt and multi-prompt modes)")
    parser.add_argument("--f_q_g_q_eval_interval", type=int, default=None,
        help="In multi-prompt mode, evaluate f_q/g_q every N unique prompts processed "
             "during training (mid-fit-step). Requires --f_q_g_q_eval. If None, only "
             "evaluates between fit steps.")
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
            "ppo", "ctl", "ctl_nosecondterm", "sixo", "sixo_approxneg", "dpg", "reinforce"
        ]
    )

    parser.add_argument("--divide_actor_loss_by_abs_beta", action="store_true",
        help="If set, divide the sampling actor loss by abs(target_dist_beta). Mathematically a wash on "
             "the gradient direction, but rescales the loss magnitude — equivalent to changing the "
             "formulation from beta*reward - 1*KL to reward - (1/|beta|)*KL, which can change Adam's "
             "optimizer dynamics (effective step size, second-moment estimates).")

    parser.add_argument("--actor_loss_entropy_bonus", type=float, default=None,
        help="If set, subtract coef * mean_per_token_entropy from the actor loss to encourage higher entropy.")
    parser.add_argument("--start_actor_loss_entropy_bonus", type=float, default=None,
        help="If set, anneal actor_loss_entropy_bonus from this value to --actor_loss_entropy_bonus over training")
    parser.add_argument("--actor_loss_entropy_bonus_schedule", type=str, default="log",
        choices=["linear", "log"],
        help="Schedule type for entropy bonus annealing: 'log' or 'linear'. Must be 'linear' if start or end is 0.")

    parser.add_argument(
        "--critic_loss_type", type=str, default="mse",
        choices=["mse", "ctl", "mixed_ctl_mse", "sixo", "sixo_approxneg"]
    )

    parser.add_argument("--reward_transform", type=str, default=None)
    parser.add_argument("--exploration_bonus_sampling_actor", type=str, default=None, choices=["exact_count", "coin_flip"], help="Exploration bonus type for sampling_actor: 'exact_count' (requires max_new_tokens=1) or 'coin_flip' (learned via coin flip network)")
    parser.add_argument("--exploration_bonus_base_actor", type=str, default=None, choices=["exact_count", "coin_flip"], help="Exploration bonus type for base_actor (not yet implemented)")
    parser.add_argument("--bonus_alpha", type=float, default=1.0, help="Scaling factor for exploration bonus: bonus = bonus_alpha * (1/sqrt(N(x)))")
    parser.add_argument("--start_bonus_alpha", type=float, default=None, help="If set, anneal bonus_alpha from this value to --bonus_alpha over training")
    parser.add_argument("--bonus_alpha_schedule", type=str, default="log", choices=["linear", "log"], help="Schedule type for bonus_alpha annealing: 'log' (logarithmic) or 'linear'. Must be 'linear' if start or end is 0.")
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
    parser.add_argument("--coin_flip_warmup_steps", type=int, default=0, help="Number of calls to compute_intrinsic_reward (i.e., batches of generated sequences) before returning non-zero bonuses. During warmup, Welford normalization stats accumulate but bonus is 0. (default: 0, no warmup)")
    parser.add_argument("--coin_flip_pretrain", type=str, default=None, help="Path to pretrained model for the coin flip network backbone (separate_nn architecture only). Defaults to --pretrain if not set. Allows using a smaller model for the coin flip network.")

    parser.add_argument("--do_harmlessness_training", action="store_true", help="Have an outer loop where we do harmlessness training on the base/initial model. Use --num_episodes for the inner loop/proposal/twist training steps, --harmlessness_training_num_episodes for the number of outer loop steps, and --harmlessness_training_episodes_per_loop for the number of harmlessness training steps in each loop iteration. So total harmlessness_training_num_episodes * num_episodes twist/proposal updates will be done, and harmlessness_training_num_episodes * harmlessness_training_episodes_per_loop base model updates will be done)")
    parser.add_argument("--harmlessness_training_num_episodes", type=int, default=1, help="Total number of outer loop steps (where each inner loop does --num_episodes twist/proposal updates")
    parser.add_argument("--harmlessness_training_episodes_per_loop", type=int, default=1, help="Number of harmlessness training steps to do for each outer loop step")

    parser.add_argument("--sampling_target_updated_base", action="store_true", help="Only for the combined_harmlessness_trainer; if set, then the sampling_actor/twisted proposal tries to target the new, updated base model after base_actor has taken a gradient step. Otherwise, twisted proposal updates before the base_actor learns")
    parser.add_argument("--use_base_as_proposal", action="store_true", help="Only for the combined_harmlessness_trainer; if set, then the sampling_actor/twisted proposal is just equivalent to the base actor, and no updates will be done to this.")


    parser.add_argument("--only_evaluate_on_neg_data", action="store_true", help="Only evaluate on neg_data")
    parser.add_argument("--neg_data_load_path", type=str, help="Where to load the neg_data")

    parser.add_argument("--evaluate_heldout_sampling", type=str, default=None, choices=["end", "each_fit_step"],
                        help="Evaluate by sampling on heldout prompts: 'end' = once after training; 'each_fit_step' = before fit loop and after each fit step (single-prompt only)")
    parser.add_argument("--n_heldout_samples_per_fit_step", type=int, default=100,
                        help="Number of samples in single-prompt heldout batch (per fit step or for 'end')")
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

    # Mixture proposal distribution
    parser.add_argument("--mixture_proposal", action="store_true", default=False,
                        help="Use a mixture proposal q_mix = w*q_current + (1-w)*q_other for CTL training")
    parser.add_argument("--mixture_other_model", type=str, default="best",
                        choices=["best", "first", "lag"],
                        help="Strategy for the 'other' model in the mixture proposal: "
                             "'best' = track the best q by g_q/KL(sigma|q) (requires f_q_g_q_eval, "
                             "target samples, and base_actor_lr=0; original behavior), "
                             "'first' = static copy of q from before the first iteration (never updated), "
                             "'lag' = copy of q from mixture_lag_steps steps in the past")
    parser.add_argument("--mixture_lag_steps", type=int, default=None,
                        help="For --mixture_other_model lag: rotate the other model every this many steps. "
                             "The other model is between mixture_lag_steps and 2*mixture_lag_steps-1 steps behind.")
    parser.add_argument("--mixture_optimization", type=str, default="mixture",
                        choices=["mixture", "q_independent", "q_half"],
                        help="Optimization mode for mixture proposal: 'mixture' uses q_mix everywhere, "
                             "'q_independent' uses separate q_current samples for the negative term, "
                             "'q_half' reuses q_current samples from mixture for the negative term")
    parser.add_argument("--mixture_eval", action="store_true", default=False,
                        help="Enable f_q/g_q evaluation on the mixture proposal q_mix (expensive; off by default). "
                             "Requires --mixture_proposal.")
    parser.add_argument("--mixture_psi_use_mix", action="store_true", default=False,
                        help="When using mixture proposal, also use q_mix (instead of q_current) for log_psi. "
                             "log_psi_mix = log q_mix(s_t|...) - log p(s_t|...). Gradient through logaddexp "
                             "includes responsibility factor r(x) = w*q_current/q_mix.")

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
    elif args.actor_loss_type == "reinforce":
        # REINFORCE for sampling actor: treat q as a standard RL policy optimizing
        # reward = log_phi with KL penalty against p. Optimum is q* ∝ p·φ = σ.
        assert args.do_harmlessness_training, (
            "actor_loss_type='reinforce' is for the sampling actor in probabilistic inference; "
            "requires --do_harmlessness_training")
        assert args.parameterization == "policy", (
            f"actor_loss_type='reinforce' requires parameterization='policy' (raw log q output, "
            f"not log_psi), got '{args.parameterization}'")
        assert args.sampling_actor_init_kl_coef == 1, (
            f"actor_loss_type='reinforce' requires sampling_actor_init_kl_coef=1 for probabilistic "
            f"inference equivalence (q* = p·phi/Z), got {args.sampling_actor_init_kl_coef}")
        args.no_critic = True
        assert args.duplicate_rollout_batch_by > 1, (
            "REINFORCE with RLOO baseline requires duplicate_rollout_batch_by > 1")
    else: # Twist learning losses (CTL, SIXO, DPG)
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

    if args.mixture_proposal:
        assert args.duplicate_rollout_batch_by >= 2, (
            f"--mixture_proposal requires --duplicate_rollout_batch_by >= 2 (need at least 1 sample from each component), "
            f"but got {args.duplicate_rollout_batch_by}"
        )

        mixture_strategy = getattr(args, 'mixture_other_model', 'best')
        if mixture_strategy == "best":
            # "best" tracks the best q by g_q, so it needs g_q evaluation infrastructure
            # and a fixed target distribution (base_actor_lr=0 / SMC setting).
            assert args.f_q_g_q_eval, (
                "--mixture_other_model best requires --f_q_g_q_eval (for g_q tracking to select q_best)"
            )
            assert args.load_target_samples_name is not None or getattr(args, 'analytic_bad_word_calc', False), (
                "--mixture_other_model best requires either --load_target_samples_name or analytic KL "
                "(for g_q evaluation to determine which model is best)"
            )
            assert abs(getattr(args, 'base_actor_learning_rate', 0)) < 1e-10, (
                f"--mixture_other_model best requires --base_actor_learning_rate 0 (SMC setting, "
                f"fixed target distribution), but got {args.base_actor_learning_rate}"
            )
        elif mixture_strategy == "first":
            pass  # no additional requirements; q_other is just the initial model copy
        elif mixture_strategy == "lag":
            assert args.mixture_lag_steps is not None and args.mixture_lag_steps > 0, (
                f"--mixture_other_model lag requires --mixture_lag_steps > 0, got {args.mixture_lag_steps}"
            )
        else:
            raise ValueError(f"Unknown mixture_other_model strategy: {mixture_strategy}")

    if getattr(args, 'mixture_eval', False):
        assert args.mixture_proposal, "--mixture_eval requires --mixture_proposal"

    if args.advantage_estimator not in ["gae"]:
        raise NotImplementedError # Not tested
        args.no_critic = True


    if args.only_evaluate_on_neg_data:
        assert args.parameterization == "policy"
        args.no_critic = True

    if args.analytic_bad_word_calc:
        assert args.rm_type in ["rlhf"] or args.reward_pretrain == "indicator_bad_token"
        assert args.generate_max_len in [1, 2]
        assert args.new_custom_single_prompt

    if args.analytic_calc:
        assert args.rm_type in ["rlhf"]
        assert args.generate_max_len == 1
        assert args.new_custom_single_prompt
        assert args.target_dist_beta is not None

    if args.fit_steps != 1:
        if not args.new_custom_single_prompt:
            print("[Warning] fit_steps != 1 without --new_custom_single_prompt: multi-prompt fit steps support is new; verify results carefully")

    assert args.n_samples_per_prompt == 1 # Others may have weird behaviour with prompt dataset

    if args.reward_transform is not None:
        # To maintain compatibility with old setup where I would only have alpha and beta and did not separate these
        if args.rew_trans_alpha is None:
            args.rew_trans_alpha = args.alpha
        if args.rew_trans_beta is None:
            args.rew_trans_beta = args.target_dist_beta

    if args.anneal_target_dist_beta:
        assert args.start_target_dist_beta is not None

    if args.reward_clamp is not None and args.reward_cap is not None:
        raise ValueError("Only one of --reward_clamp and --reward_cap may be set, not both.")

    if args.reward_signal_analysis_only and args.rejection_sample_true_target_only:
        raise ValueError("Cannot use both --reward_signal_analysis_only and --rejection_sample_true_target_only")

    train(args)
