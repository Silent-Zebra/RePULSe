import os
import math
import time as _time_module
from contextlib import contextmanager
from datetime import datetime as _datetime
from pathlib import Path

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
import torch
import numpy as np

import re


_last_timestamp = [0.0]


def print_timestamp(label):
    """Print a timestamp with a label for performance profiling.
    Shows wall-clock time and seconds elapsed since last timestamp call."""
    # t = _time_module.time()
    # dt = _datetime.now().strftime("%H:%M:%S.%f")[:-3]
    # elapsed = t - _last_timestamp[0] if _last_timestamp[0] > 0 else 0.0
    # print(f"[TIMESTAMP {dt} | elapsed {elapsed:.1f}s] {label}", flush=True)
    # _last_timestamp[0] = t
    pass

from openrlhf.models import Actor
from openrlhf.models.actor_custom import ActorCustom

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def tile_prompts(prompts, samples_per_prompt):
    # Convert single prompt to list
    if isinstance(prompts, str):
        prompts = [prompts]
    # Repeat each prompt samples_per_prompt times
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * samples_per_prompt)
    # print("expanded prompts")
    # print(expanded_prompts)
    return expanded_prompts


def get_custom_prompt_with_chat_template(tokenizer, custom_prompt, apply_chat_template, strategy=None):
    """
    Return the prompt string for a custom prompt, applying the tokenizer's chat template if requested.

    When apply_chat_template is True, wraps custom_prompt in a user message and applies the
    tokenizer's chat template with add_generation_prompt=True. If the tokenizer has no
    chat_template, falls back to OpenRLHF/Llama-3-8b-sft-mixture and logs a warning.

    Args:
        tokenizer: HF tokenizer (may be mutated with chat_template if None and apply_chat_template).
        custom_prompt: Raw custom prompt string.
        apply_chat_template: If True, apply chat template; otherwise return custom_prompt as-is.
        strategy: Optional object with .print() for warning output; if None, uses print().

    Returns:
        Prompt string (either chat-templated or raw custom_prompt).
    """
    if not apply_chat_template:
        return custom_prompt
    chat = [{"role": "user", "content": custom_prompt}]
    if tokenizer.chat_template is None:
        msg = "[Warning]: no chat template specified, defaulting to the one from OpenRLHF/Llama-3-8b-sft-mixture"
        if strategy is not None and hasattr(strategy, "print"):
            strategy.print(msg)
        else:
            print(msg)
        tokenizerchat = AutoTokenizer.from_pretrained("OpenRLHF/Llama-3-8b-sft-mixture")
        tokenizer.chat_template = tokenizerchat.chat_template
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if hasattr(model, 'config') and hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        elif strategy.args.use_ms:
            from modelscope.msdatasets import MsDataset

            namespace, dataset = dataset.split("/")
            data = MsDataset.load(dataset, namespace=namespace)
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset

def convert_token_to_id(token, tokenizer):
    if isinstance(token, int):
        return token
    elif isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")



def get_info_name_str(args):
    eval_str = ""
    init_head_base_str = ""
    # extra_str = ""
    if args.no_critic:
        critic_loss_str = ""
        lr_str = f"al{args.actor_learning_rate}"
    else:
        # Shorten critic_loss_type
        critic_loss_short = args.critic_loss_type
        critic_loss_map = {
            "mse": "mse",
            "ctl": "ctl",
            "mixed_ctl_mse": "mcm",
            "sixo": "sixo",
            "sixo_approxneg": "sixoa"
        }
        if critic_loss_short in critic_loss_map:
            critic_loss_short = critic_loss_map[critic_loss_short]
        critic_loss_str = f"_ct{critic_loss_short}"
        lr_str = f"al{args.actor_learning_rate}_cl{args.critic_learning_rate}"
    # if args.actor_modulates_base:
    #     extra_str = "actormodbase"
    if args.shared_actorcritic:
        lr_str = f"sac_l{args.actor_learning_rate}"
    if args.model_eval:
        eval_str = "_ev"
    if args.bc_coef > 0:
        lr_str += f"_bc{args.bc_coef}"

    if args.init_head_from_base:
        init_head_base_str = "_ihb"

    if args.critic_loss_type == "mixed_ctl_mse":
        lr_str += f"_a{args.alpha}"

    harmlessness_train_str = ""
    if args.do_harmlessness_training:
        lr_str += f"_bl{args.base_actor_learning_rate}"

        start_alpha_str = ""
        if args.start_alpha is not None:
            start_alpha_str = f"_s{args.start_alpha}"

        # Shorten harmlessness training loss type
        loss_type_short = args.harmlessness_training_loss_type
        if loss_type_short == "reinforce":
            loss_type_short = "r"
        elif loss_type_short == "neg_training":
            loss_type_short = "nt"
        elif loss_type_short == "neg_reinforce":
            loss_type_short = "nr"
        
        harmlessness_train_str = f"_hl{loss_type_short}{start_alpha_str}_a{args.alpha}"

    # pretrain_str = args.pretrain.split("/")[-1]
    pretrain_str = "".join([x[:2] for x in re.split(r"[-_]", args.pretrain.split("/")[-1])])

    # reward_pretrain_str = args.reward_pretrain.split("/")[-1]
    reward_pretrain_str = "".join([x[:2] for x in re.split(r"[-_]", args.reward_pretrain.split("/")[-1])])

    # Check if custom_prompt is set (different from default)
    if hasattr(args, 'custom_prompt') and args.custom_prompt and args.custom_prompt != "This man is a":
        # Remove quotes from beginning and end
        custom_prompt_val = args.custom_prompt.strip('"\'')
        # Extract first character of first word
        words = custom_prompt_val.split()
        first_word = words[0] if words else ""
        prompt_data_str = first_word[0] if first_word else ""
    else:
        # Use prompt_data if available
        if args.prompt_data:
            prompt_data_str = "".join([x[:2] for x in re.split(r"[-_]", args.prompt_data.split("/")[-1])])
        else:
            prompt_data_str = ""

    # adam_betas_str = f"_adambetas{(args.adam_betas[0])}_{(args.adam_betas[1])}"
    adam_betas_str = ""

    # sddiv_str = f"_sddiv{args.additional_sd_divider}"
    sddiv_str = ""

    # Include both episode values separately
    epi_str = f"_e{args.num_episodes}"
    if args.do_harmlessness_training:
        epi_str += f"_he{args.harmlessness_training_num_episodes}"

    # Fit steps (multiplier on samples per timestep); only include when != 1
    fit_steps_str = f"_fs{getattr(args, 'fit_steps', 1)}" if getattr(args, 'fit_steps', 1) != 1 else ""

    rm_type_str = args.rm_type
    if args.rm_type == "indicator_below_threshold":
        rm_type_str = f"it{args.threshold}"
    if args.use_base_as_proposal:
        rm_type_str += "_bp"

    reward_clamp_str = ""
    if hasattr(args, "reward_clamp") and args.reward_clamp is not None:
        reward_clamp_str = f"_rc{args.reward_clamp}"
    elif hasattr(args, "reward_cap") and args.reward_cap is not None:
        reward_clamp_str = f"_rcap{args.reward_cap}"

    rew_trans_str = ""
    if args.reward_transform:
        if args.reward_transform == "minus_alpha_exp_beta_r":
            rew_trans_str = f"rt{args.rew_trans_alpha}_b{args.rew_trans_beta}"
        elif args.reward_transform == "minus_alpha_ind":
            rew_trans_str = f"rt{args.rew_trans_alpha}_t{args.threshold}"

    start_beta_str = ""
    if args.anneal_target_dist_beta:
        start_beta_str = f"_s{args.start_target_dist_beta}"

    sep_beta_str = ""
    if args.uniform_reweight:
        sep_beta_str = f"_uw"
    elif args.separate_reweighting_beta is not None:
        sep_beta_str = f"_sb{args.separate_reweighting_beta}"

    exploration_bonus_str = ""
    if hasattr(args, 'exploration_bonus_sampling_actor') and args.exploration_bonus_sampling_actor is not None:
        if args.exploration_bonus_sampling_actor == "exact_count":
            exploration_bonus_str = "_c" + str(args.bonus_alpha)
        elif args.exploration_bonus_sampling_actor == "coin_flip":
            exploration_bonus_str = "_cf" + str(args.bonus_alpha)
            # Add coin_flip parameters
            coin_flip_dim = getattr(args, 'coin_flip_dim', 64)
            coin_flip_lr = getattr(args, 'coin_flip_lr', None)
            coin_flip_update_steps = getattr(args, 'coin_flip_update_steps', 1)
            train_coin_flip_before = getattr(args, 'train_coin_flip_before', False)
            coin_flip_first_online = getattr(args, 'coin_flip_first_online', False)
            coin_flip_use_prioritization = getattr(args, 'coin_flip_use_prioritization', False)
            coin_flip_head_init_std = getattr(args, 'coin_flip_head_init_std', 0.001)
            frozen_prior_init_std = getattr(args, 'frozen_prior_init_std', 0.1)
            coin_flip_linear_bias = getattr(args, 'coin_flip_linear_bias', False)
            coin_flip_architecture = getattr(args, 'coin_flip_architecture', 'linear_head_on_static_initial_base')
            # Backward compatibility: map old name to new name
            if coin_flip_architecture == "linear_head_on_base":
                coin_flip_architecture = "linear_head_on_static_initial_base"
            exploration_bonus_str += f"_cd{coin_flip_dim}"
            if coin_flip_lr is not None:
                exploration_bonus_str += f"_cfr{coin_flip_lr}"
            if coin_flip_update_steps != 1:
                exploration_bonus_str += f"_cfu{coin_flip_update_steps}"
            if coin_flip_head_init_std != 0.001:
                exploration_bonus_str += f"_cfh{coin_flip_head_init_std}"
            if frozen_prior_init_std != 0.1:
                exploration_bonus_str += f"_fp{frozen_prior_init_std}"
            if coin_flip_linear_bias:
                exploration_bonus_str += "_cfb"
            if coin_flip_architecture == "linear_head_on_static_initial_base":
                exploration_bonus_str += f"_cfs"
            elif coin_flip_architecture == "linear_head_on_learning_base":
                exploration_bonus_str += f"_cfl"
            elif coin_flip_architecture == "linear_head_on_learning_proposal":
                exploration_bonus_str += f"_cfq"
            elif coin_flip_architecture == "separate_nn":
                exploration_bonus_str += f"_cfsn"
                coin_flip_pretrain = getattr(args, 'coin_flip_pretrain', None)
                if coin_flip_pretrain and coin_flip_pretrain != args.pretrain:
                    cfp_str = "".join([x[:2] for x in re.split(r"[-_]", coin_flip_pretrain.split("/")[-1])])
                    exploration_bonus_str += f"_cfp{cfp_str}"
            else:
                raise ValueError(f"Unknown coin flip architecture: {coin_flip_architecture}")
            if train_coin_flip_before:
                exploration_bonus_str += "_bf"
            else:
                exploration_bonus_str += "_af"
            if coin_flip_first_online:
                exploration_bonus_str += "_fo"
            if coin_flip_use_prioritization:
                exploration_bonus_str += "_pr"
            coin_flip_warmup_steps = getattr(args, 'coin_flip_warmup_steps', 0)
            if coin_flip_warmup_steps > 0:
                exploration_bonus_str += f"_wu{coin_flip_warmup_steps}"

    mixture_str = ""
    if getattr(args, 'mixture_proposal', False):
        mix_opt = getattr(args, 'mixture_optimization', 'mixture')
        mix_short_map = {"mixture": "mx", "q_independent": "qi", "q_half": "qh"}
        mixture_str = f"_mix{mix_short_map.get(mix_opt, mix_opt)}"
        # Append the other-model strategy (omit for "best" to preserve backward compat)
        mix_other = getattr(args, 'mixture_other_model', 'best')
        if mix_other == "first":
            mixture_str += "_first"
        elif mix_other == "lag":
            mixture_str += f"_lag{getattr(args, 'mixture_lag_steps', '?')}"
        if getattr(args, 'mixture_psi_use_mix', False):
            mixture_str += "_mpsi"

    # Shorten parameterization
    param_short = args.parameterization
    param_map = {
        "policy": "p",
        "policy_psi_unnorm": "ppu",
        "policy_psi_q_p_s_t": "ppq",
        "policy_psi_q_p_s_1_to_t": "ppq1",
        "modulation_model": "mm",
        "modulation_linear_head": "mlh",
        "modulation_nn_head": "mnh"
    }
    if param_short in param_map:
        param_short = param_map[param_short]
    
    # Shorten actor_loss_type
    loss_type_short = args.actor_loss_type
    loss_map = {
        "ppo": "ppo",
        "ctl": "ctl",
        "ctl_nosecondterm": "ctln",
        "sixo": "sixo",
        "sixo_approxneg": "sixoa",
        "dpg": "dpg"
    }
    if loss_type_short in loss_map:
        loss_type_short = loss_map[loss_type_short]
    
    # Shorten lr_scheduler
    scheduler_short = args.lr_scheduler
    scheduler_map = {
        "linear": "lin",
        "cosine": "cos",
        "cosine_with_restarts": "cosr",
        "polynomial": "poly",
        "constant": "c",
        "constant_with_warmup": "cw",
        "inverse_sqrt": "isq",
        "reduce_lr_on_plateau": "rlop"
    }
    if scheduler_short in scheduler_map:
        scheduler_short = scheduler_map[scheduler_short]
    
    info_name_str = f"{rm_type_str}{reward_clamp_str}_{pretrain_str}_{reward_pretrain_str}_{prompt_data_str}_l{args.generate_max_len}_kl{args.init_kl_coef}{start_beta_str}_b{args.target_dist_beta}{sep_beta_str}{harmlessness_train_str}{rew_trans_str}_{param_short}_{loss_type_short}_ep{args.max_epochs}{epi_str}{fit_steps_str}{eval_str}_sc{scheduler_short}_{lr_str}{critic_loss_str}{adam_betas_str}_{param_short}{init_head_base_str}{sddiv_str}{exploration_bonus_str}{mixture_str}_tb{args.train_batch_size}_s{args.seed}"

    return info_name_str


def get_target_samples_filename(args):
    """
    Generate filename for target samples based on args, similar to get_info_name_str pattern.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Full path to save target samples file
    """
    import re
    
    # Abbreviate pretrain (first 2 chars of each segment)
    pretrain_str = "".join([x[:2] for x in re.split(r"[-_]", args.pretrain.split("/")[-1])])
    
    # Abbreviate reward_pretrain (first 2 chars of each segment)
    reward_pretrain_str = "".join([x[:2] for x in re.split(r"[-_]", args.reward_pretrain.split("/")[-1])])
    
    # Get rm_type
    rm_type_str = args.rm_type
    
    # Format target_dist_beta
    beta_str = f"b{args.target_dist_beta}"
    
    # Format reward_clamp or reward_cap if rm_type is rlhf (only one may be set)
    reward_clamp_str = ""
    if args.rm_type == "rlhf" and hasattr(args, 'reward_clamp') and args.reward_clamp is not None:
        reward_clamp_str = f"_rc{args.reward_clamp}"
    elif args.rm_type == "rlhf" and hasattr(args, 'reward_cap') and args.reward_cap is not None:
        reward_clamp_str = f"_rcap{args.reward_cap}"
    
    # Get prompt abbreviation
    if hasattr(args, 'custom_prompt') and args.custom_prompt and args.custom_prompt != "This man is a":
        # Use first character of first word
        custom_prompt_val = args.custom_prompt.strip('"\'')
        words = custom_prompt_val.split()
        first_word = words[0] if words else ""
        prompt_str = first_word[0] if first_word else ""
    else:
        # Use abbreviation of prompt_data
        if args.prompt_data:
            prompt_str = "".join([x[:2] for x in re.split(r"[-_]", args.prompt_data.split("/")[-1])])
        else:
            prompt_str = ""
    
    # Format true_target_sample_amount (at the end)
    tsa_str = f"tsa{args.true_target_sample_amount}"
    
    # Construct filename
    filename = f"target_samples_{pretrain_str}_{reward_pretrain_str}_{rm_type_str}_l{args.generate_max_len}_{beta_str}{reward_clamp_str}_{prompt_str}_{tsa_str}.pt"
    
    # Return full path
    return f"{args.save_path}/{filename}"


def inspect_rewards_list(rewards_list, label="reward"):
    # print(rewards_list)
    rewards_tensor = torch.tensor(rewards_list)
    print(f"{label} record shape")
    print(rewards_tensor.shape)
    firsts = [5, 10, 50]
    for first in firsts:
        print(f"First {first} {label} average")
        print(rewards_tensor[:first].mean())
    lasts = [500, 200, 100, 50, 10, 5]
    for last in lasts:
        print(f"Last {last} {label} average")
        print(rewards_tensor[-last:].mean())

# New function to load model and tokenizer
def load_model_and_tokenizer(args, strategy):
    """
    Loads the actor model and tokenizer based on provided arguments and strategy.

    Args:
        args: Command line arguments containing model configuration.
        strategy: The distributed training/evaluation strategy object.

    Returns:
        tuple: A tuple containing the loaded and prepared actor model and the tokenizer.
    """
    # Create base actor first
    base_actor = Actor(
        args.pretrain,
        use_flash_attention_2=getattr(args, 'flash_attn', False),
        bf16=getattr(args, 'bf16', False),
        load_in_4bit=getattr(args, 'load_in_4bit', False),
        ds_config=strategy.get_ds_eval_config(offload=False), # Use eval config for base
    )

    # Create the actual actor based on parameterization
    if "policy" not in getattr(args, 'parameterization', 'policy'): # Default to 'policy' if not present
        actor = ActorCustom(
            args.pretrain,
            initial_model=base_actor,
            use_flash_attention_2=getattr(args, 'flash_attn', False),
            bf16=getattr(args, 'bf16', False),
            load_in_4bit=getattr(args, 'load_in_4bit', False),
            lora_rank=getattr(args, 'lora_rank', 0),
            lora_alpha=getattr(args, 'lora_alpha', 0),
            target_modules=getattr(args, 'target_modules', None),
            lora_dropout=getattr(args, 'lora_dropout', 0),
            ds_config=strategy.get_ds_train_config(is_actor=True), # Use train config for custom actor
            parameterization=args.parameterization,
            additional_sd_divider=getattr(args, 'additional_sd_divider', 1.0),
            init_head_from_base=getattr(args, 'init_head_from_base', False)
        )
    else:
        actor = base_actor

    # Initialize with DeepSpeed
    # Pass is_rlhf=True as it was in the original script context
    actor = strategy.prepare(actor.eval(), is_rlhf=True)

    # Get tokenizer for the actor model
    # Use getattr for optional 'disable_fast_tokenizer' arg
    use_fast = not getattr(args, 'disable_fast_tokenizer', False)
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=use_fast)

    # Ensure pad token is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if hasattr(actor.model, 'config') and hasattr(actor.model.config, 'pad_token_id'):
            actor.model.config.pad_token_id = tokenizer.pad_token_id

    return actor, tokenizer


def log_sequence_for_negatives(start, end, steps):
    if start < 0 and end < 0:
        sign = -1
    else:
        assert start > 0 and end > 0
        sign = 1
    start_abs, end_abs = abs(start), abs(end)
    # Using natural logs (ln) and exp
    logs = np.linspace(np.log(start_abs), np.log(end_abs), steps)
    seq = np.exp(logs)
    return (sign * seq).tolist()


def left_pad_sequences(tensor, target_seq_len, pad_value):
    """Left-pad a 2D tensor (batch, seq_len) to target_seq_len along dim=1."""
    assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D"
    if tensor.shape[1] >= target_seq_len:
        return tensor
    pad_size = target_seq_len - tensor.shape[1]
    padding = torch.full((tensor.shape[0], pad_size), pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([padding, tensor], dim=1)


def compute_action_mask_from_sequences(sequences, num_actions, eos_token_id, pad_token_id):
    """
    Compute action_mask from sequences following the pattern in actor.py's process_sequences.
    
    Args:
        sequences: Full sequences tensor of shape (batch_size, seq_len)
        num_actions: Number of action tokens (response tokens)
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
        
    Returns:
        action_mask: Boolean tensor of shape (batch_size, num_actions) indicating valid action tokens
    """
    # Compute input_len from sequence length and num_actions
    # sequences shape is (batch_size, seq_len), where seq_len = input_len + num_actions
    input_len = sequences.shape[1] - num_actions
    
    # Extract state sequence (response tokens): state_i (current token) + action_i (next token) -> state_i+1
    # Following the pattern from actor.py process_sequences
    state_seq = sequences[:, input_len - 1 : -1]
    action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
    action_mask[:, 0] = 1  # First token is always valid
    
    return action_mask


@torch.no_grad()
def compute_actor_log_probs_for_sequences(actor, sequences, num_actions, attention_mask=None, 
                                         eos_token_id=None, pad_token_id=None, shared_actorcritic=False):
    """
    Compute log probabilities for sequences using an actor model.
    
    This is a utility function that extracts the common pattern of calling actor.forward
    and summing log probabilities, used by both g_q_estimate and compute_target_samples_logprob.
    
    Args:
        actor: Actor model (can be Actor or ActorCritic)
        sequences: Full sequences tensor of shape (batch_size, seq_len) where seq_len = prompt_len + num_actions
        num_actions: Number of action tokens (response tokens)
        attention_mask: Optional attention mask tensor. If None, will be created from sequences.
        eos_token_id: Optional EOS token ID for creating attention mask
        pad_token_id: Optional pad token ID for creating attention mask
        shared_actorcritic: If True, actor returns (log_probs, values) tuple
        
    Returns:
        log_probs_per_seq: Tensor of shape (batch_size,) containing log probability per sequence
        action_log_probs: Tensor of shape (batch_size, num_actions) containing log probabilities for each action token
    """
    if attention_mask is None:
        if eos_token_id is not None and pad_token_id is not None:
            # Create attention mask respecting EOS and pad tokens
            attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        else:
            # Fallback: all ones (assumes no padding/EOS in sequences)
            attention_mask = torch.ones_like(sequences, dtype=torch.long)
    
    if shared_actorcritic:
        action_log_probs, _ = actor(sequences, num_actions, attention_mask)
    else:
        action_log_probs = actor(sequences, num_actions, attention_mask)
    
    action_log_probs = action_log_probs.float()  # More precision

    # Compute action_mask to zero out log probs for padding tokens after EOS
    action_mask = compute_action_mask_from_sequences(sequences, num_actions, eos_token_id, pad_token_id)

    # Sum log probabilities for each sequence (masked)
    log_probs_per_seq = (action_log_probs * action_mask).sum(dim=-1)

    return log_probs_per_seq, action_log_probs


@contextmanager
def swap_actor(experience_maker, replacement_actor):
    """Context manager to temporarily replace the actor in an experience maker.

    Usage:
        with swap_actor(experience_maker, q_best_model):
            # experience_maker.actor is now q_best_model
            ...generate samples...
        # experience_maker.actor is restored to the original
    """
    original = experience_maker.actor
    experience_maker.actor = replacement_actor
    try:
        yield
    finally:
        experience_maker.actor = original


@torch.no_grad()
def compute_mixture_seq_log_prob(q_current_model, q_best_model, sequences, num_actions,
                                  attention_mask, eos_token_id, pad_token_id,
                                  log_w_current, log_w_best, shared_actorcritic=False):
    """Compute mixture sequence-level log prob for given sequences.

    log q_mix(s) = logsumexp(log_w_current + log_q_current(s), log_w_best + log_q_best(s))

    Args:
        q_current_model: Current proposal model
        q_best_model: Best historical proposal model
        sequences: (batch, seq_len) token sequences
        num_actions: Number of action tokens
        attention_mask: (batch, seq_len) attention mask
        eos_token_id: EOS token ID
        pad_token_id: Pad token ID
        log_w_current: Log weight for q_current component (scalar)
        log_w_best: Log weight for q_best component (scalar)
        shared_actorcritic: Whether q_current is a shared actor-critic model

    Returns:
        log_q_mix: (batch,) sequence-level mixture log probs
        action_mask: (batch, num_actions) action mask
    """
    _, alp_current = compute_actor_log_probs_for_sequences(
        q_current_model, sequences, num_actions,
        attention_mask=attention_mask,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
        shared_actorcritic=shared_actorcritic
    )
    _, alp_best = compute_actor_log_probs_for_sequences(
        q_best_model, sequences, num_actions,
        attention_mask=attention_mask,
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
        shared_actorcritic=False  # q_best is always a plain eval model
    )

    action_mask = compute_action_mask_from_sequences(sequences, num_actions, eos_token_id, pad_token_id)

    alp_current = alp_current.float() * action_mask
    alp_best = alp_best.float() * action_mask

    log_q_current = alp_current.sum(dim=-1)
    log_q_best = alp_best.sum(dim=-1)

    log_q_mix = torch.logaddexp(
        log_w_current + log_q_current,
        log_w_best + log_q_best
    )

    return log_q_mix, action_mask


@torch.no_grad()
def eval_log_p_plus_log_phi(trainer, experience_maker, args, attention_mask, action_mask,
                            num_actions, sequences, return_extra_info=False, force_no_exploration_bonus=False):
    """
    Evaluate log(p) + log(phi) for target distribution computation.

    Args:
        trainer: Trainer instance (needed for access to methods, but not used directly here)
        experience_maker: Experience maker instance
        args: Training arguments
        attention_mask: Attention mask
        action_mask: Action mask
        num_actions: Number of actions
        sequences: Generated sequences
        return_extra_info: Whether to return extra info (log_p, log_phi)
        force_no_exploration_bonus: If True, compute_reward_no_kl skips exploration bonus (e.g. for f_q/g_q evaluation)
        
    Returns:
        log_tilde_sigma or (log_tilde_sigma, log_p, log_phi) if return_extra_info
    """
    log_phi, _, _, _ = experience_maker.compute_reward_no_kl(sequences, attention_mask, multiply_by_beta=True, force_no_exploration_bonus=force_no_exploration_bonus)

    base_action_log_probs = experience_maker.initial_model(sequences,
                                                            num_actions,
                                                            attention_mask)
    base_action_log_probs = base_action_log_probs.float() * action_mask # more precision

    log_p = base_action_log_probs.sum(dim=-1)

    log_tilde_sigma = log_p + log_phi
    if return_extra_info:
        return log_tilde_sigma, log_p, log_phi
    else:
        return log_tilde_sigma


def f_q_estimate(trainer, experience_maker, args, prompt):
    """
    Calculate E_q [log sigma(s) - log q(s)]
    
    Args:
        trainer: Trainer instance (needed for shared_actorcritic and generate_kwargs)
        experience_maker: Experience maker instance
        args: Training arguments
        prompt: prompt
        
    Returns:
        f_qs, attention_mask, num_actions, sequences, log_p, log_phi, log_q, action_mask
    """
    experience_maker.set_all_eval()
    batch_prompt = tile_prompts(prompt, args.n_samples_for_f_q_g_q)

    with torch.no_grad():
        print_timestamp("eval - f_q_estimate: start generation")
        if trainer.shared_actorcritic:
            action_log_probs, action_mask, attention_mask, num_actions, sequences, value = experience_maker.generate_seqs_and_get_logprobs(
                batch_prompt, **trainer.generate_kwargs)
        else:
            action_log_probs, action_mask, attention_mask, num_actions, sequences = experience_maker.generate_seqs_and_get_logprobs(
                batch_prompt, **trainer.generate_kwargs)
        print_timestamp("eval - f_q_estimate: end generation, start log_p + log_phi")
        action_log_probs = action_log_probs.float() * action_mask # more precision
        log_q = action_log_probs.sum(dim=-1)

        # generate_seqs_and_get_logprobs stays in eval mode throughout;
        # this call is redundant but kept for safety
        experience_maker.set_all_eval()

        log_tilde_sigma, log_p, log_phi = eval_log_p_plus_log_phi(
            trainer, experience_maker, args, attention_mask, action_mask, num_actions, sequences, return_extra_info=True, force_no_exploration_bonus=True
        )
        print_timestamp("eval - f_q_estimate: done")

        f_qs = log_tilde_sigma - log_q

    experience_maker.set_all_policies_train()

    return f_qs, attention_mask, num_actions, sequences, log_p, log_phi, log_q, action_mask


def f_q_estimate_batched(trainer, experience_maker, args, prompts):
    """
    Batched f_q estimation for multiple prompts. Tiles all P prompts to get P*N
    prompt strings, generates all at once, computes f_q for the batch.

    Args:
        trainer: Trainer instance
        experience_maker: Experience maker instance
        args: Training arguments
        prompts: List of P prompt strings

    Returns:
        dict with keys:
            "f_qs_per_prompt": list of P tensors, each shape (N,)
            "num_actions": int
            "q_seqs_per_prompt": list of P tensors, each shape (N, seq_len)
            "log_p_per_prompt": list of P tensors, each shape (N,)
            "log_phi_per_prompt": list of P tensors, each shape (N,)
            "log_q_per_prompt": list of P tensors, each shape (N,)
            "common_seq_len": int (seq_len of generated sequences)
    """
    P = len(prompts)
    N = args.n_samples_for_f_q_g_q

    experience_maker.set_all_eval()
    batch_prompts = tile_prompts(prompts, N)  # P*N prompts

    with torch.no_grad():
        if trainer.shared_actorcritic:
            action_log_probs, action_mask, attention_mask, num_actions, sequences, value = \
                experience_maker.generate_seqs_and_get_logprobs(batch_prompts, **trainer.generate_kwargs)
        else:
            action_log_probs, action_mask, attention_mask, num_actions, sequences = \
                experience_maker.generate_seqs_and_get_logprobs(batch_prompts, **trainer.generate_kwargs)

        action_log_probs = action_log_probs.float() * action_mask
        log_q = action_log_probs.sum(dim=-1)  # (P*N,)

        # generate_seqs_and_get_logprobs stays in eval mode throughout;
        # this call is redundant but kept for safety
        experience_maker.set_all_eval()

        log_tilde_sigma, log_p, log_phi = eval_log_p_plus_log_phi(
            trainer, experience_maker, args, attention_mask, action_mask,
            num_actions, sequences, return_extra_info=True, force_no_exploration_bonus=True
        )

        f_qs = log_tilde_sigma - log_q  # (P*N,)

    experience_maker.set_all_policies_train()

    # Reshape to per-prompt: (P*N,) -> list of P tensors each (N,)
    f_qs_per_prompt = list(f_qs.reshape(P, N))
    log_p_per_prompt = list(log_p.reshape(P, N))
    log_phi_per_prompt = list(log_phi.reshape(P, N))
    log_q_per_prompt = list(log_q.reshape(P, N))
    q_seqs_per_prompt = list(sequences.reshape(P, N, -1))

    return {
        "f_qs_per_prompt": f_qs_per_prompt,
        "num_actions": num_actions,
        "q_seqs_per_prompt": q_seqs_per_prompt,
        "log_p_per_prompt": log_p_per_prompt,
        "log_phi_per_prompt": log_phi_per_prompt,
        "log_q_per_prompt": log_q_per_prompt,
        "common_seq_len": sequences.shape[1],
    }


def g_q_estimate(trainer, experience_maker, args, true_sigma_samples, num_actions, attention_mask):
    """
    Calculate g_q estimate: log(sigma) - log(q) for true sigma samples.
    
    Args:
        trainer: Trainer instance (needed for shared_actorcritic and generate_kwargs)
        experience_maker: Experience maker instance
        args: Training arguments
        true_sigma_samples: True samples from sigma distribution
        num_actions: Number of actions
        attention_mask: Attention mask
        condition_twist_on_tokens: Optional condition tokens
        
    Returns:
        log_tilde_sigma - log_q
    """
    experience_maker.set_all_eval()
    sequences = true_sigma_samples
    with torch.no_grad():
        # Use common utility function to compute log_q and action_log_probs (avoid double computation)
        eos_token_id = trainer.generate_kwargs["eos_token_id"]
        pad_token_id = trainer.generate_kwargs["pad_token_id"]
        _, action_log_probs = compute_actor_log_probs_for_sequences(
            experience_maker.actor,
            sequences,
            num_actions,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            shared_actorcritic=trainer.shared_actorcritic
        )

        # Compute action_mask from sequences respecting EOS and pad tokens
        action_mask = compute_action_mask_from_sequences(sequences, num_actions, eos_token_id, pad_token_id)

        # Apply action_mask to log_q (matching f_q_estimate behavior)
        action_log_probs = action_log_probs.float() * action_mask
        log_q = action_log_probs.sum(dim=-1)

        log_tilde_sigma = eval_log_p_plus_log_phi(trainer, experience_maker, args,
                                attention_mask, action_mask,
                                num_actions, sequences, force_no_exploration_bonus=True)
        log_tilde_sigma = log_tilde_sigma.float() # more precision

    experience_maker.set_all_policies_train()

    return log_tilde_sigma - log_q


def _compute_num_actions_for_target_samples(target_samples, prompt_text, experience_maker):
    """Compute num_actions for target samples by tokenizing the prompt.

    Target samples have shape (K, seq_len) where seq_len = prompt_tokens + gen_tokens.
    We tokenize the prompt to find prompt_tokens, then num_actions = seq_len - prompt_tokens.

    Uses experience_maker.tokenize_fn to ensure tokenization is consistent with generation.
    """
    # tokenize_fn with padding=True returns tensors (consistent with generation usage)
    # For a single prompt, no padding is needed, so result matches non-padded tokenization
    inputs = experience_maker.tokenize_fn([prompt_text], experience_maker.prompt_max_len, device="cpu")
    assert inputs["input_ids"].shape[0] == 1
    prompt_len = inputs["input_ids"].shape[1]
    num_actions = target_samples.shape[1] - prompt_len
    assert num_actions > 0, (
        f"Target sample seq_len ({target_samples.shape[1]}) <= prompt_len ({prompt_len}). "
        f"This suggests a tokenization mismatch between target sample generation and current run."
    )
    return num_actions


def g_q_estimate_batched(trainer, experience_maker, args, target_samples_by_prompt,
                         prompt_texts):
    """
    Batched g_q estimation across multiple prompts' target samples.
    Processes each prompt's target samples with the correct num_actions
    (derived from tokenizing the prompt text), since different prompts
    may have different prompt token lengths.

    Args:
        trainer: Trainer instance
        experience_maker: Experience maker instance
        args: Training arguments
        target_samples_by_prompt: list of tensors, each shape (K_p, seq_len_p)
        prompt_texts: list of prompt strings (one per prompt, same length as
            target_samples_by_prompt)

    Returns:
        g_qs_per_prompt: list of tensors, each shape (K_p,)
    """
    assert len(target_samples_by_prompt) == len(prompt_texts)
    eos_token_id = trainer.generate_kwargs["eos_token_id"]
    pad_token_id = trainer.generate_kwargs["pad_token_id"]
    chunk_size = args.n_samples_for_f_q_g_q

    g_qs_per_prompt = []
    for t, prompt_text in zip(target_samples_by_prompt, prompt_texts):
        num_actions = _compute_num_actions_for_target_samples(
            t, prompt_text, experience_maker)

        # Process this prompt's target samples in chunks
        prompt_g_qs = []
        for start in range(0, t.shape[0], chunk_size):
            chunk = t[start:start + chunk_size]
            if chunk.shape[0] == 0:
                continue
            attention_mask_chunk = (
                chunk.ne(eos_token_id) & chunk.ne(pad_token_id)
            ).to(dtype=torch.long)
            g_qs_chunk = g_q_estimate(trainer, experience_maker, args, chunk,
                                      num_actions, attention_mask_chunk)
            prompt_g_qs.append(g_qs_chunk)

        g_qs_per_prompt.append(torch.cat(prompt_g_qs, dim=0))

    return g_qs_per_prompt


def f_q_g_q_evaluation(trainer, experience_maker, args, f_q_estimates_list, g_q_estimates_list,
                       iwae_lbs_list, iwae_ubs_list,
                       prompt_text, true_target_samples):
    """
    Evaluate f_q, g_q, and IWAE bounds (single seed).

    Args:
        trainer: Trainer instance (needed for generate_kwargs and method calls)
        experience_maker: Experience maker instance
        args: Training arguments
        f_q_estimates_list: List to append f_q estimates to
        g_q_estimates_list: List to append g_q estimates to
        iwae_lbs_list: List to append IWAE lower bounds to (scalar floats)
        iwae_ubs_list: List to append IWAE upper bounds to (scalar floats)
        prompt_text: Prompt text for evaluation
        true_target_samples: True target samples
    """
    print_timestamp("eval - f_q_g_q_evaluation: start f_q_estimate")
    f_qs, attention_mask, num_actions, q_seqs, log_p, log_phi, log_q, action_mask = f_q_estimate(
        trainer, experience_maker, args, prompt_text)
    print_timestamp("eval - f_q_g_q_evaluation: end f_q_estimate")
    print("Avg F_q Estimate (Learned Model)")
    print(f_qs.mean())

    # IWAE Lower Bound
    print("IWAE Lower Bound Estimate (Learned Model)")
    iwae_lb = (torch.logsumexp(f_qs, dim=0) - math.log(f_qs.shape[0])).item()
    print(iwae_lb)

    eos_token_id = trainer.generate_kwargs["eos_token_id"]
    pad_token_id = trainer.generate_kwargs["pad_token_id"]

    # g_q estimates over all target samples
    assert true_target_samples is not None
    # Compute num_actions from prompt tokenization, not from q generation,
    # since q generation may truncate early (all samples hitting EOS).
    # All chunks of the same prompt's target samples have the same seq_len, so compute once.
    target_num_actions = _compute_num_actions_for_target_samples(
        true_target_samples, prompt_text, experience_maker)
    print_timestamp("eval - f_q_g_q_evaluation: start g_q loop")
    total_g_qs = None
    range_val = math.ceil(true_target_samples.shape[0] / args.n_samples_for_f_q_g_q)
    print(range_val)
    for j in range(range_val):
        samples = true_target_samples[
                  j * args.n_samples_for_f_q_g_q: (j + 1) * args.n_samples_for_f_q_g_q]
        if samples.shape[0] != 0:
            print("G_q Estimates Learned Model")

            attention_mask_g_q = (
                    samples.ne(eos_token_id) & samples.ne(pad_token_id)).to(
                dtype=torch.long)

            g_qs = g_q_estimate(trainer, experience_maker, args, samples,
                                 target_num_actions, attention_mask_g_q)

            print(g_qs)
            print("Avg G_q Estimate (Learned Model)")
            print(g_qs.mean())

            if total_g_qs is None:
                total_g_qs = g_qs
            else:
                total_g_qs = torch.cat((total_g_qs, g_qs), axis=0)
                print("Total G_qs shape")
                print(total_g_qs.shape)

    # IWAE Upper Bound: combine g_q(target_sample) with f_q(q_samples)
    # Reuse pre-computed values instead of splicing target into q sequences
    # (which fails when sequence lengths differ due to early EOS truncation).
    iwae_ub = None
    if true_target_samples is not None and total_g_qs is not None:
        target_weight = total_g_qs[0:1]  # g_q for first target sample
        q_weights = f_qs[1:]  # f_q for N-1 q samples (drop one to keep total = N)
        all_weights = torch.cat([target_weight.to(q_weights.device), q_weights])
        print("IWAE Upper Bound Estimate (Learned Model)")
        iwae_ub = (torch.logsumexp(all_weights, dim=0) - math.log(all_weights.shape[0])).item()
        print(iwae_ub)

    iwae_lbs_list.append(iwae_lb)
    iwae_ubs_list.append(iwae_ub)
    print("IWAE LB AND UB")
    print(iwae_lb)
    print(iwae_ub)
    print("IWAE LB AND UB LISTS")
    print(iwae_lbs_list)
    print(iwae_ubs_list)
    print("Shapes")
    print(total_g_qs.shape if total_g_qs is not None else None)
    print(f_qs.shape)

    print_timestamp("eval - f_q_g_q_evaluation: done")
    if total_g_qs is not None:
        g_q_estimates_list.append(
            total_g_qs.cpu())  # Only one G_q estimate (over all the target samples)
    f_q_estimates_list.append(f_qs.cpu())


def g_q_estimate_mixture(trainer, experience_maker, args, true_sigma_samples, num_actions,
                          attention_mask, q_best_model, log_w_current, log_w_best):
    """
    Compute g_q for the mixture q_mix = w*q_current + (1-w)*q_best on target samples.
    g_q_mix = log_sigma - log_q_mix where log_q_mix = logsumexp(log_w_current + log_q_current, log_w_best + log_q_best).
    """
    experience_maker.set_all_eval()
    with torch.no_grad():
        eos_token_id = trainer.generate_kwargs["eos_token_id"]
        pad_token_id = trainer.generate_kwargs["pad_token_id"]

        log_q_mix, action_mask = compute_mixture_seq_log_prob(
            experience_maker.actor, q_best_model, true_sigma_samples, num_actions,
            attention_mask, eos_token_id, pad_token_id,
            log_w_current, log_w_best, shared_actorcritic=trainer.shared_actorcritic
        )

        log_tilde_sigma = eval_log_p_plus_log_phi(
            trainer, experience_maker, args,
            attention_mask, action_mask, num_actions, true_sigma_samples,
            force_no_exploration_bonus=True
        )
        log_tilde_sigma = log_tilde_sigma.float()

    experience_maker.set_all_policies_train()
    return log_tilde_sigma - log_q_mix


def f_q_g_q_evaluation_mixture(trainer, experience_maker, args,
                                f_q_mix_list, g_q_mix_list, iwae_mix_lbs_list, iwae_mix_ubs_list,
                                prompt_text, true_target_samples,
                                q_best_model, log_w_current, log_w_best):
    """
    Evaluate f_q, g_q, IWAE bounds for the mixture q_mix (single prompt, single seed).
    Generates samples from q_mix (split between q_current and q_best), evaluates metrics.
    """
    import math as _math

    print_timestamp("eval - f_q_g_q_evaluation_mixture: start")
    experience_maker.set_all_eval()
    N = args.n_samples_for_f_q_g_q
    n_eval_current = N // 2
    n_eval_best = N - n_eval_current

    with torch.no_grad():
        # Generate from q_current
        batch_current = tile_prompts(prompt_text, n_eval_current)
        if trainer.shared_actorcritic:
            alp_c, amask_c, atmask_c, nact_c, seq_c, _ = experience_maker.generate_seqs_and_get_logprobs(
                batch_current, **trainer.generate_kwargs)
        else:
            alp_c, amask_c, atmask_c, nact_c, seq_c = experience_maker.generate_seqs_and_get_logprobs(
                batch_current, **trainer.generate_kwargs)

        # Generate from q_best
        with swap_actor(experience_maker, q_best_model):
            batch_best = tile_prompts(prompt_text, n_eval_best)
            if trainer.shared_actorcritic:
                alp_b, amask_b, atmask_b, nact_b, seq_b, _ = experience_maker.generate_seqs_and_get_logprobs(
                    batch_best, **trainer.generate_kwargs)
            else:
                alp_b, amask_b, atmask_b, nact_b, seq_b = experience_maker.generate_seqs_and_get_logprobs(
                    batch_best, **trainer.generate_kwargs)

        # Pad shorter batch to match longer one (different lengths due to early EOS stopping)
        eos_token_id = trainer.generate_kwargs["eos_token_id"]
        pad_token_id = trainer.generate_kwargs["pad_token_id"]
        max_nact = max(nact_c, nact_b)
        if nact_c != nact_b:
            print(f"[Mixture eval] Padding: nact_c={nact_c}, nact_b={nact_b} -> {max_nact}")
        if nact_c < max_nact:
            pad_len = max_nact - nact_c
            seq_c = torch.nn.functional.pad(seq_c, (0, pad_len), value=pad_token_id)
            amask_c = torch.nn.functional.pad(amask_c, (0, pad_len), value=0)
            atmask_c = torch.nn.functional.pad(atmask_c, (0, pad_len), value=0)
        if nact_b < max_nact:
            pad_len = max_nact - nact_b
            seq_b = torch.nn.functional.pad(seq_b, (0, pad_len), value=pad_token_id)
            amask_b = torch.nn.functional.pad(amask_b, (0, pad_len), value=0)
            atmask_b = torch.nn.functional.pad(atmask_b, (0, pad_len), value=0)
        num_actions = max_nact

        # Concatenate sequences
        all_sequences = torch.cat([seq_c, seq_b], dim=0)  # (N, seq_len)
        all_action_mask = torch.cat([amask_c, amask_b], dim=0)
        all_attention_mask = torch.cat([atmask_c, atmask_b], dim=0)

        # Compute mixture log prob from both models for ALL samples

        log_q_mix, _ = compute_mixture_seq_log_prob(
            experience_maker.actor, q_best_model, all_sequences, num_actions,
            all_attention_mask, eos_token_id, pad_token_id,
            log_w_current, log_w_best, shared_actorcritic=trainer.shared_actorcritic
        )

        log_tilde_sigma = eval_log_p_plus_log_phi(
            trainer, experience_maker, args,
            all_attention_mask, all_action_mask, num_actions, all_sequences,
            force_no_exploration_bonus=True
        )
        log_tilde_sigma = log_tilde_sigma.float()

        f_qs_mix = log_tilde_sigma - log_q_mix
        print(f"[Mixture eval] Avg F_q_mix = {f_qs_mix.mean().item():.4f}")

        # IWAE lower bound for mixture
        iwae_lb = (torch.logsumexp(f_qs_mix, dim=0) - _math.log(f_qs_mix.shape[0])).item()
        print(f"[Mixture eval] IWAE LB (mix) = {iwae_lb:.4f}")

    # g_q_mix on target samples
    total_g_qs_mix = None
    if true_target_samples is not None:
        eos_token_id = trainer.generate_kwargs["eos_token_id"]
        pad_token_id = trainer.generate_kwargs["pad_token_id"]
        # All chunks of the same prompt's target samples have the same seq_len, so compute once.
        target_num_actions = _compute_num_actions_for_target_samples(
            true_target_samples, prompt_text, experience_maker)
        range_val = _math.ceil(true_target_samples.shape[0] / args.n_samples_for_f_q_g_q)
        for j in range(range_val):
            samples = true_target_samples[j * N: (j + 1) * N]
            if samples.shape[0] != 0:
                attention_mask_g_q = (
                    samples.ne(eos_token_id) & samples.ne(pad_token_id)
                ).to(dtype=torch.long)
                g_qs_mix = g_q_estimate_mixture(
                    trainer, experience_maker, args, samples, target_num_actions,
                    attention_mask_g_q, q_best_model, log_w_current, log_w_best
                )
                if total_g_qs_mix is None:
                    total_g_qs_mix = g_qs_mix
                else:
                    total_g_qs_mix = torch.cat((total_g_qs_mix, g_qs_mix), dim=0)
        if total_g_qs_mix is not None:
            print(f"[Mixture eval] Avg G_q_mix = {total_g_qs_mix.mean().item():.4f}")

    # IWAE upper bound: reuse pre-computed g_q_mix (target) + f_q_mix (q samples)
    iwae_ub = None
    if true_target_samples is not None and total_g_qs_mix is not None:
        target_weight = total_g_qs_mix[0:1]
        q_weights = f_qs_mix[1:]
        all_weights = torch.cat([target_weight.to(q_weights.device), q_weights])
        iwae_ub = (torch.logsumexp(all_weights, dim=0) - _math.log(all_weights.shape[0])).item()
        print(f"[Mixture eval] IWAE UB (mix) = {iwae_ub:.4f}")

    # Append to lists
    f_q_mix_list.append(f_qs_mix.cpu())
    if total_g_qs_mix is not None:
        g_q_mix_list.append(total_g_qs_mix.cpu())
    iwae_mix_lbs_list.append(iwae_lb)
    iwae_mix_ubs_list.append(iwae_ub)

    print_timestamp("eval - f_q_g_q_evaluation_mixture: done")
    experience_maker.set_all_policies_train()


def f_q_g_q_evaluation_mixture_multi_prompt(
    trainer, experience_maker, args,
    f_q_mix_list, g_q_mix_list, iwae_mix_lbs_list, iwae_mix_ubs_list,
    prompt_texts, true_target_samples_by_prompt,
    q_best_model, log_w_current, log_w_best):
    """
    Multi-prompt wrapper for f_q_g_q_evaluation_mixture. Loops over prompts,
    calls the single-prompt mixture eval per prompt, then aggregates and appends
    one entry per timepoint to the accumulator lists.
    """
    print_timestamp(f"eval - f_q_g_q_evaluation_mixture_multi_prompt: start ({len(prompt_texts)} prompts)")
    all_f_qs = []
    all_g_qs = []
    all_iwae_lbs = []
    all_iwae_ubs = []

    for i, prompt_text in enumerate(prompt_texts):
        print_timestamp(f"eval - mixture_multi_prompt: prompt {i+1}/{len(prompt_texts)}")
        has_targets = (
            true_target_samples_by_prompt is not None
            and i < len(true_target_samples_by_prompt)
            and true_target_samples_by_prompt[i] is not None
            and true_target_samples_by_prompt[i].numel() > 0
        )
        target_samples = true_target_samples_by_prompt[i] if has_targets else None

        # Per-prompt temp lists (f_q_g_q_evaluation_mixture appends exactly one entry)
        f_q_tmp, g_q_tmp, iwae_lb_tmp, iwae_ub_tmp = [], [], [], []

        f_q_g_q_evaluation_mixture(
            trainer, experience_maker, args,
            f_q_tmp, g_q_tmp, iwae_lb_tmp, iwae_ub_tmp,
            prompt_text, target_samples,
            q_best_model, log_w_current, log_w_best,
        )

        assert len(f_q_tmp) == 1, f"Expected 1 f_q entry per prompt, got {len(f_q_tmp)}"
        all_f_qs.append(f_q_tmp[0])
        all_g_qs.append(g_q_tmp[0] if g_q_tmp else None)
        all_iwae_lbs.append(iwae_lb_tmp[0] if iwae_lb_tmp else None)
        all_iwae_ubs.append(iwae_ub_tmp[0] if iwae_ub_tmp else None)

    # Aggregate across prompts: one entry per timepoint
    f_q_valid = [x for x in all_f_qs if x is not None]
    f_q_agg = torch.cat(f_q_valid) if f_q_valid else None

    g_q_valid = [x for x in all_g_qs if x is not None]
    g_q_agg = torch.cat(g_q_valid) if g_q_valid else None

    iwae_lbs_valid = [x for x in all_iwae_lbs if x is not None]
    iwae_lbs_agg = torch.tensor(iwae_lbs_valid).mean().item() if iwae_lbs_valid else None

    iwae_ubs_valid = [x for x in all_iwae_ubs if x is not None]
    iwae_ubs_agg = torch.tensor(iwae_ubs_valid).mean().item() if iwae_ubs_valid else None

    if f_q_agg is not None:
        f_q_mix_list.append(f_q_agg)
        print(f"[Mixture eval multi-prompt] Aggregated F_q_mix = {f_q_agg.mean().item():.4f} ({len(f_q_valid)} prompts)")
    if g_q_agg is not None:
        g_q_mix_list.append(g_q_agg)
        print(f"[Mixture eval multi-prompt] Aggregated G_q_mix = {g_q_agg.mean().item():.4f} ({len(g_q_valid)} prompts)")
    iwae_mix_lbs_list.append(iwae_lbs_agg)
    iwae_mix_ubs_list.append(iwae_ubs_agg)
    print_timestamp("eval - f_q_g_q_evaluation_mixture_multi_prompt: done")


def f_q_g_q_evaluation_batched(trainer, experience_maker, args, prompt_texts,
                                true_target_samples_by_prompt=None):
    """
    Batched evaluation of f_q, g_q, IWAE bounds across multiple prompts (single seed).

    Processes all prompts in prompt_texts in one batch (caller handles chunking
    via n_prompts_f_q_g_q).

    Args:
        trainer: Trainer instance
        experience_maker: Experience maker instance
        args: Training arguments
        prompt_texts: List of P prompt strings
        true_target_samples_by_prompt: List of P tensors or Nones (one per prompt)

    Returns:
        dict with same format as f_q_g_q_evaluation_multi_prompt
    """
    P = len(prompt_texts)
    N = args.n_samples_for_f_q_g_q
    # pad_token_id = trainer.generate_kwargs["pad_token_id"]
    # eos_token_id = trainer.generate_kwargs["eos_token_id"]

    # --- f_q: batched generation ---
    f_q_result = f_q_estimate_batched(trainer, experience_maker, args, prompt_texts)
    f_qs_pp = f_q_result["f_qs_per_prompt"]       # P tensors, each (N,)
    # q_seqs_pp = f_q_result["q_seqs_per_prompt"]    # P tensors, each (N, seq_len)
    # num_actions = f_q_result["num_actions"]
    # common_seq_len = f_q_result["common_seq_len"]

    # Per-prompt results
    f_q_by_prompt = [f_qs_pp[p].cpu() for p in range(P)]
    g_qs_per_prompt = [None] * P
    iwae_lbs_by_prompt = []
    iwae_ubs_by_prompt = [None] * P

    # IWAE LB per prompt (scalar)
    for p in range(P):
        iwae_lb = (torch.logsumexp(f_qs_pp[p], dim=0) - math.log(N)).item()
        iwae_lbs_by_prompt.append(iwae_lb)

    print(f"[batched] mean f_q per prompt = {[f.mean().item() for f in f_qs_pp]}")

    # --- g_q: only for prompts with target samples ---
    # Identify prompts with target samples
    prompts_with_targets = []
    for p in range(P):
        has_target = (
            true_target_samples_by_prompt is not None
            and p < len(true_target_samples_by_prompt)
            and true_target_samples_by_prompt[p] is not None
            and true_target_samples_by_prompt[p].numel() > 0
        )
        if has_target:
            prompts_with_targets.append(p)

    if prompts_with_targets and true_target_samples_by_prompt is not None:
        target_tensors = [true_target_samples_by_prompt[p] for p in prompts_with_targets]
        target_prompt_texts = [prompt_texts[p] for p in prompts_with_targets]
        g_qs_list = g_q_estimate_batched(
            trainer, experience_maker, args, target_tensors, target_prompt_texts)
        for i, p in enumerate(prompts_with_targets):
            g_qs_per_prompt[p] = g_qs_list[i].cpu()
            print(f"[batched] g_q prompt {p}: mean = {g_qs_list[i].mean().item()}")

    # --- IWAE UB: reuse pre-computed f_q (q samples) and g_q (target samples) ---
    # IWAE UB = logsumexp(w_1, ..., w_N) - log(N) where:
    #   w_1 = g_q(target_sample) = log sigma(x_target) - log q(x_target)
    #   w_2..w_N = f_q(q_samples) = log sigma(x_q) - log q(x_q)
    # Both are already computed above, so we just combine them.
    # This avoids the previous approach of splicing target samples into q-generated
    # sequences, which failed when sequence lengths differed (e.g. early EOS truncation
    # in f_q generation vs full-length target samples).
    if prompts_with_targets:
        for i, p in enumerate(prompts_with_targets):
            assert g_qs_per_prompt[p] is not None, f"g_q not computed for prompt {p}"
            assert g_qs_per_prompt[p].shape[0] > 0, f"Prompt {p} has no target samples for IWAE UB"
            # Take first target sample's weight, drop one q sample to keep total = N
            target_weight = g_qs_per_prompt[p][0:1]  # (1,)
            q_weights = f_qs_pp[p][1:]  # (N-1,)
            all_weights = torch.cat([target_weight.to(q_weights.device), q_weights])  # (N,)
            iwae_ub = (torch.logsumexp(all_weights, dim=0) - math.log(N)).item()
            iwae_ubs_by_prompt[p] = iwae_ub
            print(f"[batched] IWAE UB prompt {p}: {iwae_ub}")

    # Aggregate across prompts
    f_q_valid = [x for x in f_q_by_prompt if x is not None]
    f_q_agg = torch.cat(f_q_valid) if f_q_valid else None

    g_q_valid = [x for x in g_qs_per_prompt if x is not None]
    g_q_agg = torch.cat(g_q_valid) if g_q_valid else None

    iwae_lbs_valid = [x for x in iwae_lbs_by_prompt if x is not None]
    iwae_lbs_agg = torch.tensor(iwae_lbs_valid) if iwae_lbs_valid else None

    iwae_ubs_valid = [x for x in iwae_ubs_by_prompt if x is not None]
    iwae_ubs_agg = torch.tensor(iwae_ubs_valid) if iwae_ubs_valid else None

    print(f"[batched] f_q_agg shape: {f_q_agg.shape if f_q_agg is not None else None}")
    if g_q_agg is not None:
        print(f"[batched] g_q_agg shape: {g_q_agg.shape}")
    print(f"[batched] IWAE LBs: {iwae_lbs_agg}")
    print(f"[batched] IWAE UBs: {iwae_ubs_agg}")

    return {
        "f_q_by_prompt": f_q_by_prompt,
        "g_q_by_prompt": g_qs_per_prompt,
        "iwae_lbs_by_prompt": iwae_lbs_by_prompt,
        "iwae_ubs_by_prompt": iwae_ubs_by_prompt,
        "f_q_agg": f_q_agg,
        "g_q_agg": g_q_agg,
        "iwae_lbs_agg": iwae_lbs_agg,
        "iwae_ubs_agg": iwae_ubs_agg,
    }


def load_target_samples(path, device, strategy):
    """
    Load target samples from a file, handling both v1 (list) and v2 (dict) formats.

    Returns:
        (samples_by_prompt, prompt_texts_or_none):
            samples_by_prompt: list of tensors, one per prompt (each shape [n_samples, seq_len])
            prompt_texts_or_none: list of prompt strings (v2) or None (v1)
    """
    raw = torch.load(path, map_location="cpu")

    if isinstance(raw, dict) and raw.get("version", 1) >= 2:
        # v2 format
        prompt_texts = raw["prompt_texts"]
        samples_by_prompt_raw = raw["samples_by_prompt"]
        samples_by_prompt = []
        filtered_prompt_texts = []
        for prompt_text, samples_list in zip(prompt_texts, samples_by_prompt_raw):
            if len(samples_list) > 0:
                t = torch.tensor(samples_list, dtype=torch.int64).to(device)
                samples_by_prompt.append(t)
                filtered_prompt_texts.append(prompt_text)
            # else: skip entirely (don't add empty entries)
        n_skipped = len(prompt_texts) - len(filtered_prompt_texts)
        if n_skipped > 0:
            strategy.print(f"Filtered out {n_skipped} prompts with 0 target samples")
        strategy.print(f"Loaded v2 target samples: {len(samples_by_prompt)} prompts, "
                       f"samples per prompt: {[s.shape[0] for s in samples_by_prompt]}")
        return samples_by_prompt, filtered_prompt_texts
    else:
        # v1 format: list of lists (one element = one prompt's samples)
        # The old format stores as a list where index 0 is the first (and usually only) prompt's samples
        if isinstance(raw, list) and len(raw) > 0:
            samples_by_prompt = []
            for prompt_samples in raw:
                t = torch.tensor(prompt_samples, dtype=torch.int64).to(device)
                samples_by_prompt.append(t)
            strategy.print(f"Loaded v1 target samples: {len(samples_by_prompt)} prompt(s), "
                           f"samples per prompt: {[s.shape[0] for s in samples_by_prompt]}")
            return samples_by_prompt, None
        else:
            raise ValueError(f"Unexpected target samples format: {type(raw)}")


def f_q_g_q_evaluation_multi_prompt(trainer, experience_maker, args,
                                     prompt_texts, true_target_samples_by_prompt=None):
    """
    Multi-prompt wrapper around f_q_g_q_evaluation. Loops over prompts, calls
    existing f_q_g_q_evaluation per-prompt, and collects per-prompt + aggregated results.

    Args:
        trainer: Trainer instance
        experience_maker: Experience maker instance
        args: Training arguments
        prompt_texts: List of prompt strings to evaluate
        true_target_samples_by_prompt: List of tensors (one per prompt) or None.
            If provided, must be same length as prompt_texts. Entries can be None for prompts
            without target samples (g_q/IWAE will be skipped for those).

    Returns:
        dict with keys:
            "f_q_by_prompt": list of tensors (one per prompt)
            "g_q_by_prompt": list of tensors or Nones (one per prompt)
            "iwae_lbs_by_prompt": list of scalar floats or Nones (one per prompt)
            "iwae_ubs_by_prompt": list of scalar floats or Nones (one per prompt)
            "f_q_agg": concatenated f_q across all prompts
            "g_q_agg": concatenated g_q across prompts with target samples, or None
            "iwae_lbs_agg": 1D tensor of shape (n_prompts_with_targets,) or None
            "iwae_ubs_agg": 1D tensor of shape (n_prompts_with_targets,) or None
    """
    print_timestamp(f"eval - f_q_g_q_evaluation_multi_prompt: start ({len(prompt_texts)} prompts)")
    n_prompts_f_q_g_q = getattr(args, 'n_prompts_f_q_g_q', None)

    if n_prompts_f_q_g_q is not None:
        # Batched path: chunk prompts into groups of n_prompts_f_q_g_q
        print(f"[multi_prompt] Using batched path with n_prompts_f_q_g_q={n_prompts_f_q_g_q}")
        all_chunk_results = []
        for start in range(0, len(prompt_texts), n_prompts_f_q_g_q):
            end = min(start + n_prompts_f_q_g_q, len(prompt_texts))
            chunk_prompts = prompt_texts[start:end]
            chunk_targets = (true_target_samples_by_prompt[start:end]
                            if true_target_samples_by_prompt is not None else None)
            result = f_q_g_q_evaluation_batched(
                trainer, experience_maker, args, chunk_prompts, chunk_targets)
            all_chunk_results.append(result)
        return _merge_batched_results(all_chunk_results)
    else:
        # Existing per-prompt for-loop (unchanged)
        result = _f_q_g_q_evaluation_multi_prompt_unbatched(
            trainer, experience_maker, args, prompt_texts, true_target_samples_by_prompt)
        print_timestamp("eval - f_q_g_q_evaluation_multi_prompt: done")
        return result


def _merge_batched_results(chunk_results):
    """Merge results from multiple batched evaluation chunks into a single result dict."""
    f_q_by_prompt = []
    g_q_by_prompt = []
    iwae_lbs_by_prompt = []
    iwae_ubs_by_prompt = []

    for result in chunk_results:
        f_q_by_prompt.extend(result["f_q_by_prompt"])
        g_q_by_prompt.extend(result["g_q_by_prompt"])
        iwae_lbs_by_prompt.extend(result["iwae_lbs_by_prompt"])
        iwae_ubs_by_prompt.extend(result["iwae_ubs_by_prompt"])

    # Re-aggregate across all prompts
    f_q_valid = [x for x in f_q_by_prompt if x is not None]
    f_q_agg = torch.cat(f_q_valid) if f_q_valid else None

    g_q_valid = [x for x in g_q_by_prompt if x is not None]
    g_q_agg = torch.cat(g_q_valid) if g_q_valid else None

    iwae_lbs_valid = [x for x in iwae_lbs_by_prompt if x is not None]
    iwae_lbs_agg = torch.tensor(iwae_lbs_valid) if iwae_lbs_valid else None

    iwae_ubs_valid = [x for x in iwae_ubs_by_prompt if x is not None]
    iwae_ubs_agg = torch.tensor(iwae_ubs_valid) if iwae_ubs_valid else None

    return {
        "f_q_by_prompt": f_q_by_prompt,
        "g_q_by_prompt": g_q_by_prompt,
        "iwae_lbs_by_prompt": iwae_lbs_by_prompt,
        "iwae_ubs_by_prompt": iwae_ubs_by_prompt,
        "f_q_agg": f_q_agg,
        "g_q_agg": g_q_agg,
        "iwae_lbs_agg": iwae_lbs_agg,
        "iwae_ubs_agg": iwae_ubs_agg,
    }


def _f_q_g_q_evaluation_multi_prompt_unbatched(trainer, experience_maker, args,
                                                prompt_texts, true_target_samples_by_prompt=None):
    """Original per-prompt for-loop implementation of f_q_g_q_evaluation_multi_prompt."""
    f_q_by_prompt = []
    g_q_by_prompt = []
    iwae_lbs_by_prompt = []
    iwae_ubs_by_prompt = []

    for i, prompt_text in enumerate(prompt_texts):
        print_timestamp(f"eval - multi_prompt unbatched: prompt {i+1}/{len(prompt_texts)}")
        has_target_samples = (
            true_target_samples_by_prompt is not None
            and i < len(true_target_samples_by_prompt)
            and true_target_samples_by_prompt[i] is not None
            and true_target_samples_by_prompt[i].numel() > 0
        )
        target_samples_for_prompt = true_target_samples_by_prompt[i] if has_target_samples else None

        # Use per-prompt lists to collect results from f_q_g_q_evaluation
        f_q_list_prompt = []
        g_q_list_prompt = []
        iwae_lbs_list_prompt = []
        iwae_ubs_list_prompt = []

        if has_target_samples:
            f_q_g_q_evaluation(
                trainer, experience_maker, args,
                f_q_list_prompt, g_q_list_prompt,
                iwae_lbs_list_prompt, iwae_ubs_list_prompt,
                prompt_text, target_samples_for_prompt,
            )
        else:
            # f_q only (no g_q/IWAE without target samples)
            f_qs, *_ = f_q_estimate(trainer, experience_maker, args, prompt_text)
            f_q_list_prompt.append(f_qs.cpu())

        # f_q_g_q_evaluation appends exactly one entry per call. Same for the f_q_estimate path above.
        assert len(f_q_list_prompt) == 1, f"Expected exactly 1 f_q entry per prompt, got {len(f_q_list_prompt)}"
        f_q_by_prompt.append(f_q_list_prompt[0])
        g_q_by_prompt.append(g_q_list_prompt[0] if g_q_list_prompt else None)
        iwae_lbs_by_prompt.append(iwae_lbs_list_prompt[0] if iwae_lbs_list_prompt else None)
        iwae_ubs_by_prompt.append(iwae_ubs_list_prompt[0] if iwae_ubs_list_prompt else None)

    # Aggregate across prompts
    f_q_valid = [x for x in f_q_by_prompt if x is not None]
    f_q_agg = torch.cat(f_q_valid) if f_q_valid else None

    g_q_valid = [x for x in g_q_by_prompt if x is not None]
    g_q_agg = torch.cat(g_q_valid) if g_q_valid else None

    iwae_lbs_valid = [x for x in iwae_lbs_by_prompt if x is not None]
    iwae_lbs_agg = torch.tensor(iwae_lbs_valid) if iwae_lbs_valid else None

    iwae_ubs_valid = [x for x in iwae_ubs_by_prompt if x is not None]
    iwae_ubs_agg = torch.tensor(iwae_ubs_valid) if iwae_ubs_valid else None

    return {
        "f_q_by_prompt": f_q_by_prompt,
        "g_q_by_prompt": g_q_by_prompt,
        "iwae_lbs_by_prompt": iwae_lbs_by_prompt,
        "iwae_ubs_by_prompt": iwae_ubs_by_prompt,
        "f_q_agg": f_q_agg,
        "g_q_agg": g_q_agg,
        "iwae_lbs_agg": iwae_lbs_agg,
        "iwae_ubs_agg": iwae_ubs_agg,
    }


@torch.no_grad()
def generate_and_score_batch(actor, reward_model, tokenizer, prompt, batch_size,
                             prompt_max_len, reward_clamp, reward_cap, generate_kwargs):
    """Generate a batch of sequences from `actor` and score them with `reward_model`.

    This is the shared generate-then-score primitive used by rejection sampling and
    reward signal analysis.  It tiles the prompt, tokenizes, generates, scores,
    and clamps rewards.

    Args:
        actor: Model to sample from.
        reward_model: Reward model for scoring sequences.
        tokenizer: Tokenizer for encoding prompts.
        prompt: A single prompt string.
        batch_size: Number of sequences to generate.
        prompt_max_len: Maximum prompt length for tokenization.
        reward_clamp: Symmetric reward clamp value (or None).
        reward_cap: Upper reward cap value (or None). At least one of reward_clamp /
                    reward_cap should be set if clamping is desired (both None → no clamping).
        generate_kwargs: Dict of generation kwargs (max_new_tokens, eos_token_id, etc.).

    Returns:
        sequences: Tensor [batch_size, seq_len] of token IDs (prompt + generation).
        attention_mask: Tensor [batch_size, seq_len].
        action_mask: Tensor [batch_size, gen_len].
        unclamped_rewards: Tensor [batch_size] of raw rewards.
        clamped_rewards: Tensor [batch_size] of clamped rewards.
    """
    device = next(actor.parameters()).device

    prompt_batch = tile_prompts(prompt, batch_size)
    inputs = tokenizer(
        prompt_batch,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=prompt_max_len,
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    sequences, attention_mask, action_mask = actor.generate(**inputs, **generate_kwargs)
    rewards = reward_model(sequences, attention_mask)
    unclamped_rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards

    if reward_clamp is not None:
        clamped_rewards = unclamped_rewards.clamp(min=-reward_clamp, max=reward_clamp)
    elif reward_cap is not None:
        clamped_rewards = unclamped_rewards.clamp(max=reward_cap)
    else:
        clamped_rewards = unclamped_rewards.clone()

    return sequences, attention_mask, action_mask, unclamped_rewards, clamped_rewards


@torch.no_grad()
def rejection_sample_for_prompt(
    actor, reward_model, tokenizer, prompt,
    target_dist_beta, reward_clamp, reward_cap, prompt_max_len,
    generate_kwargs, batch_size, max_gen,
    tile_prompts_fn,
    target_sample_amount=None,
    rm_type="rlhf",
    strategy=None,
):
    """Rejection sampling from sigma = p * exp(beta * r) for a single prompt.

    The target distribution formulation sigma(x) propto p(x) * exp(beta * r(x)) assumes
    rm_type="rlhf" (scalar reward model). Other reward types use different formulations.

    Generates sequences from `actor`, computes rewards, and accepts/rejects based on
    the ratio exp(beta * clamped_reward) / M, where M = exp(|clamp * beta|).

    Stops when either:
      - `target_sample_amount` accepted samples are collected (if set), OR
      - `max_gen` total sequences have been generated (if set), OR
      - both are None → raises AssertionError (at least one must be set).

    Args:
        actor: The model to sample from (e.g., base_actor).
        reward_model: Reward model for scoring sequences.
        tokenizer: Tokenizer for encoding prompts.
        prompt: A single prompt string.
        target_dist_beta: Beta parameter for the target distribution.
        reward_clamp: Symmetric reward clamp value (or None).
        reward_cap: Upper reward cap value (or None; one of reward_clamp/reward_cap must be set).
        prompt_max_len: Maximum prompt length for tokenization.
        generate_kwargs: Dict of generation kwargs (max_new_tokens, eos_token_id, etc.).
        batch_size: Number of sequences to generate per iteration.
        max_gen: Maximum total sequences to generate before stopping (None = no limit).
        tile_prompts_fn: Function to tile a prompt into a batch (e.g., tile_prompts).
        target_sample_amount: Stop after collecting this many accepted samples (None = no limit).
        rm_type: Reward model type. Must be "rlhf" — the target distribution formulation
                 sigma(x) propto p(x) * exp(beta * r(x)) assumes a scalar reward model.
        strategy: Optional strategy object with .print() method.

    Returns:
        accepted_seqs: list of token ID lists (may be shorter than target_sample_amount
                       if max_gen was reached first; may be empty if none accepted)
        accepted_rewards: list of float rewards (clamped)
        total_generated: int total sequences generated
    """
    assert rm_type == "rlhf", (
        f"Rejection sampling currently only supports rm_type='rlhf', got '{rm_type}'. "
        f"The target distribution formulation sigma(x) propto p(x) * exp(beta * r(x)) "
        f"assumes a scalar reward model."
    )
    assert reward_clamp is not None or reward_cap is not None, \
        "Either reward_clamp or reward_cap must be set for rejection sampling"

    def _print(msg):
        if strategy is not None:
            strategy.print(msg)
        else:
            print(msg)

    # Compute log_M = |clamp_val * beta|
    clamp_val = reward_clamp if reward_clamp is not None else reward_cap
    log_M = abs(clamp_val * target_dist_beta)

    assert target_sample_amount is not None or max_gen is not None, (
        "At least one of target_sample_amount or max_gen must be set to prevent an infinite loop "
        "in rejection sampling. Set --true_target_sample_amount and/or --max_gen_per_prompt_rejection."
    )

    accepted_seqs = []
    accepted_rewards = []
    total_generated = 0
    iteration = 0

    while True:
        # Check stopping criteria
        if target_sample_amount is not None and len(accepted_seqs) >= target_sample_amount:
            break
        if max_gen is not None and total_generated >= max_gen:
            if target_sample_amount is not None:
                _print(f"  Warning: Reached max_gen={max_gen} with only "
                       f"{len(accepted_seqs)}/{target_sample_amount} accepted. Stopping.")
            break

        iteration += 1
        sequences, attention_mask, action_mask, unclamped_rewards, clamped_rewards = \
            generate_and_score_batch(
                actor, reward_model, tokenizer, prompt, batch_size,
                prompt_max_len, reward_clamp, reward_cap, generate_kwargs,
            )

        log_phi = target_dist_beta * clamped_rewards
        log_ratio = log_phi - log_M
        raw_accept_prob = torch.exp(log_ratio)
        # Sanity check: acceptance probabilities must be in [0, 1] for valid rejection sampling.
        assert (raw_accept_prob >= -1e-6).all(), (
            f"Rejection sampling acceptance probability is negative (min={raw_accept_prob.min().item():.6f})."
        )
        assert (raw_accept_prob <= 1.0 + 1e-6).all(), (
            f"Rejection sampling acceptance probability exceeds 1 (max={raw_accept_prob.max().item():.6f}). "
            f"Check reward clamping and target_dist_beta settings."
        )
        accept_prob = raw_accept_prob.clamp(min=0.0, max=1.0)
        u = torch.rand_like(accept_prob)
        accept_mask = u < accept_prob

        batch_accepted_seqs = [seq.cpu().tolist() for seq in sequences[accept_mask]]
        batch_accepted_rews = [rew.cpu().item() for rew in clamped_rewards[accept_mask]]
        accepted_seqs.extend(batch_accepted_seqs)
        accepted_rewards.extend(batch_accepted_rews)
        total_generated += sequences.shape[0]

        if iteration % 10 == 0 or (target_sample_amount is not None and len(accepted_seqs) >= target_sample_amount):
            rate = len(accepted_seqs) / total_generated if total_generated > 0 else 0.0
            target_str = f"/{target_sample_amount}" if target_sample_amount is not None else ""
            _print(f"  Rejection sampling iteration {iteration}: {len(accepted_seqs)}{target_str} accepted, "
                   f"{total_generated} generated, acceptance rate: {rate:.4f}")

    # Truncate to target_sample_amount if we overshot
    if target_sample_amount is not None:
        accepted_seqs = accepted_seqs[:target_sample_amount]
        accepted_rewards = accepted_rewards[:target_sample_amount]

    return accepted_seqs, accepted_rewards, total_generated


@torch.no_grad()
def rejection_sample_multi_prompt(
    actor, reward_model, tokenizer, prompts,
    target_dist_beta, reward_clamp, reward_cap, prompt_max_len,
    generate_kwargs, batch_size, total_target,
    max_gen_per_prompt=None, first_pass_limit=None,
    rm_type="rlhf", strategy=None,
):
    """Multi-prompt rejection sampling: collect total_target samples across prompts.

    Collects at most 1 accepted sample per prompt, cycling through prompts in multi-pass
    fashion until the global target is reached.

    Pass 1: iterate through all prompts, spending at most first_pass_limit generations
    per prompt. Subsequent passes: revisit prompts without a sample, generating up to
    the overall max_gen_per_prompt limit.
    Stops when: enough total samples collected, OR all prompts have a sample,
    OR all remaining prompts have exhausted max_gen_per_prompt, OR no progress in a pass.

    Args:
        actor: The model to sample from.
        reward_model: Reward model for scoring sequences.
        tokenizer: Tokenizer for encoding prompts.
        prompts: List of prompt strings.
        target_dist_beta: Beta parameter for the target distribution.
        reward_clamp: Symmetric reward clamp value (or None).
        reward_cap: Upper reward cap value (or None).
        prompt_max_len: Maximum prompt length for tokenization.
        generate_kwargs: Dict of generation kwargs.
        batch_size: Number of sequences to generate per batch.
        total_target: Total number of accepted samples to collect across all prompts.
        max_gen_per_prompt: Maximum generations per prompt before giving up (None = no limit).
        first_pass_limit: Per-prompt generation limit for the first pass (None = use max_gen_per_prompt).
        rm_type: Reward model type (must be "rlhf").
        strategy: Optional strategy object with .print() method.

    Returns:
        accepted_by_prompt: list of lists of token id lists (one list per prompt, may be empty)
        rewards_by_prompt: list of lists of float rewards (parallel to accepted_by_prompt)
        total_generated: int total sequences generated across all prompts
        total_accepted: int total accepted samples
    """
    assert rm_type == "rlhf", (
        f"Multi-prompt rejection sampling only supports rm_type='rlhf', got '{rm_type}'"
    )

    def _print(msg):
        if strategy is not None:
            strategy.print(msg)
        else:
            print(msg, flush=True)

    if first_pass_limit is None:
        first_pass_limit = max_gen_per_prompt  # Could still be None (no limit)

    n_prompts = len(prompts)
    _print(f"Multi-prompt rejection sampling: collecting {total_target} total samples "
           f"across {n_prompts} prompts")
    _print(f"  first_pass_limit={first_pass_limit}, max_gen_per_prompt={max_gen_per_prompt}")

    accepted_by_prompt = [[] for _ in range(n_prompts)]
    rewards_by_prompt = [[] for _ in range(n_prompts)]
    generated_per_prompt = [0] * n_prompts
    total_generated = 0
    total_collected = 0

    max_accepted_per_prompt = 1
    pass_num = 0
    while total_collected < total_target:
        pass_num += 1
        made_progress_this_pass = False
        prompts_skipped_done = 0
        prompts_skipped_limit = 0

        for prompt_idx, prompt in enumerate(prompts):
            if total_collected >= total_target:
                break
            # Skip prompts that already have their one accepted sample
            if len(accepted_by_prompt[prompt_idx]) >= max_accepted_per_prompt:
                prompts_skipped_done += 1
                continue
            # Skip prompts that have hit the overall generation limit
            if max_gen_per_prompt is not None and generated_per_prompt[prompt_idx] >= max_gen_per_prompt:
                prompts_skipped_limit += 1
                continue

            # Determine per-prompt budget for this pass
            gen_at_pass_start = generated_per_prompt[prompt_idx]
            if pass_num == 1 and first_pass_limit is not None:
                pass_budget = first_pass_limit
            else:
                # Subsequent passes: no per-pass limit, just the overall limit
                pass_budget = None

            got_acceptance = False
            iteration = 0
            while not got_acceptance:
                if total_collected >= total_target:
                    break
                # Check overall per-prompt limit
                if max_gen_per_prompt is not None and generated_per_prompt[prompt_idx] >= max_gen_per_prompt:
                    print(f"  Prompt {prompt_idx + 1}/{n_prompts}: reached overall limit "
                          f"({max_gen_per_prompt}) with {len(accepted_by_prompt[prompt_idx])} accepted", flush=True)
                    break
                # Check per-pass budget (first pass uses first_pass_limit)
                if pass_budget is not None and (generated_per_prompt[prompt_idx] - gen_at_pass_start) >= pass_budget:
                    break

                iteration += 1
                sequences, attention_mask, action_mask, unclamped_rewards, clamped_rewards = \
                    generate_and_score_batch(
                        actor, reward_model, tokenizer, prompt, batch_size,
                        prompt_max_len, reward_clamp, reward_cap, generate_kwargs,
                    )

                # Compute acceptance probability
                clamp_val = reward_clamp if reward_clamp is not None else reward_cap
                log_M = abs(clamp_val * target_dist_beta)
                log_phi = target_dist_beta * clamped_rewards
                log_ratio = log_phi - log_M
                raw_accept_prob = torch.exp(log_ratio)
                assert (raw_accept_prob >= -1e-6).all(), (
                    f"Rejection sampling acceptance probability is negative (min={raw_accept_prob.min().item():.6f})."
                )
                assert (raw_accept_prob <= 1.0 + 1e-6).all(), (
                    f"Rejection sampling acceptance probability exceeds 1 (max={raw_accept_prob.max().item():.6f})."
                )
                accept_prob = raw_accept_prob.clamp(min=0.0, max=1.0)
                u = torch.rand_like(accept_prob)
                accept_mask = u < accept_prob
                seqs = [seq.cpu().tolist() for seq in sequences[accept_mask]]
                rews = [rew.cpu().item() for rew in clamped_rewards[accept_mask]]
                n_gen = sequences.shape[0]

                generated_per_prompt[prompt_idx] += n_gen
                total_generated += n_gen

                # Take at most 1 sample per prompt (and at most what's left to reach total_target)
                n_can_take_this_prompt = max_accepted_per_prompt - len(accepted_by_prompt[prompt_idx])
                n_to_take = min(len(seqs), n_can_take_this_prompt, total_target - total_collected)
                if n_to_take > 0:
                    accepted_by_prompt[prompt_idx].extend(seqs[:n_to_take])
                    rewards_by_prompt[prompt_idx].extend(rews[:n_to_take])
                    total_collected += n_to_take
                    made_progress_this_pass = True
                    got_acceptance = True

                print(f"  Pass {pass_num}, prompt {prompt_idx + 1}/{n_prompts}, iter {iteration}: "
                      f"{len(seqs)} accepted in batch ({n_gen} generated), "
                      f"took {n_to_take}, "
                      f"{len(accepted_by_prompt[prompt_idx])} this prompt, "
                      f"{total_collected}/{total_target} total, "
                      f"{generated_per_prompt[prompt_idx]} generated this prompt", flush=True)

        _print(f"  Pass {pass_num} complete: {total_collected}/{total_target} collected, "
               f"{prompts_skipped_done} prompts done, {prompts_skipped_limit} prompts at gen limit")

        if not made_progress_this_pass:
            _print(f"  No progress in pass {pass_num}. "
                   f"Collected {total_collected}/{total_target} total. Stopping.")
            break

    return accepted_by_prompt, rewards_by_prompt, total_generated, total_collected


def discover_trajectory_checkpoints(trajectory_dir):
    """Discover and sort checkpoint tags in a trajectory directory.

    Looks for subdirectories matching "total_step*" pattern, parses the step
    numbers, and returns them sorted numerically.

    Args:
        trajectory_dir: Path to the trajectory directory (e.g., the _harml_actor dir).

    Returns:
        List of (step_number, tag_name) tuples, sorted by step_number.
    """
    assert os.path.isdir(trajectory_dir), f"Trajectory directory not found: {trajectory_dir}"

    checkpoints = []
    for entry in os.listdir(trajectory_dir):
        full_path = os.path.join(trajectory_dir, entry)
        if not os.path.isdir(full_path):
            continue
        # Match "total_step<N>" pattern
        match = re.match(r'^total_step(\d+)$', entry)
        if match:
            step_num = int(match.group(1))
            checkpoints.append((step_num, entry))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def collect_trajectory_rejection_prompt_texts(trajectory_dir):
    """Scan all trajectory rejection sample files and return the set of prompt texts
    that have >=1 sample in any trajectory step.

    Args:
        trajectory_dir: Path to the trajectory directory (the _harml_actor dir).
            The rejection samples directory is inferred by replacing "_harml_actor"
            with "_rejection_samples" (same convention as combined_harmlessness_trainer).

    Returns:
        Set of prompt text strings found across all trajectory rejection sample files.
    """
    rejection_dir = trajectory_dir.replace("_harml_actor", "_rejection_samples")
    if not os.path.isdir(rejection_dir):
        return set()
    prompt_texts = set()
    for f in sorted(os.listdir(rejection_dir)):
        if f.endswith('.pt'):
            data = torch.load(os.path.join(rejection_dir, f), map_location='cpu')
            if isinstance(data, dict) and data.get("version", 1) >= 2:
                for pt in data.get("prompt_texts", []):
                    prompt_texts.add(pt)
            # v1 format has no prompt texts — skip (single-prompt mode)
    return prompt_texts