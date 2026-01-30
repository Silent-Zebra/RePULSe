import os
import math
from pathlib import Path

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
import torch
import numpy as np

import re

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
    if isinstance(token, str):
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


    rm_type_str = args.rm_type
    if args.rm_type == "indicator_below_threshold":
        rm_type_str = f"it{args.threshold}"
    if args.use_base_as_proposal:
        rm_type_str += "_bp"

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
    
    info_name_str = f"{rm_type_str}_{pretrain_str}_{reward_pretrain_str}_{prompt_data_str}_l{args.generate_max_len}_kl{args.init_kl_coef}{start_beta_str}_b{args.target_dist_beta}{sep_beta_str}{harmlessness_train_str}{rew_trans_str}_{param_short}_{loss_type_short}_ep{args.max_epochs}{epi_str}{eval_str}_sc{scheduler_short}_{lr_str}{critic_loss_str}{adam_betas_str}_{param_short}{init_head_base_str}{sddiv_str}{exploration_bonus_str}_tb{args.train_batch_size}_s{args.seed}"

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


def inspect_rewards_list(rewards_list):
    # print(rewards_list)
    rewards_tensor = torch.tensor(rewards_list)
    print("Rewards record shape")
    print(rewards_tensor.shape)
    firsts = [5, 10, 50]
    for first in firsts:
        print(f"First {first} reward average")
        print(rewards_tensor[:first].mean())
    lasts = [500, 200, 100, 50, 10, 5]
    for last in lasts:
        print(f"Last {last} reward average")
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
def eval_log_p_plus_log_phi(trainer, experience_maker, args, action_log_probs, attention_mask, action_mask,
                            num_actions, sequences, return_extra_info=False, force_no_exploration_bonus=False):
    """
    Evaluate log(p) + log(phi) for target distribution computation.
    
    Args:
        trainer: Trainer instance (needed for access to methods, but not used directly here)
        experience_maker: Experience maker instance
        args: Training arguments
        action_log_probs: Action log probabilities
        attention_mask: Attention mask
        action_mask: Action mask
        num_actions: Number of actions
        sequences: Generated sequences
        return_extra_info: Whether to return extra info (log_p, log_phi)
        force_no_exploration_bonus: If True, compute_reward_no_kl skips exploration bonus (e.g. for f_q/g_q evaluation)
        
    Returns:
        log_tilde_sigma or (log_tilde_sigma, log_p, log_phi) if return_extra_info
    """
    log_phi, _, _ = experience_maker.compute_reward_no_kl(sequences, attention_mask, multiply_by_beta=True, force_no_exploration_bonus=force_no_exploration_bonus)

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


def f_q_estimate(trainer, experience_maker, args, batch_prompt):
    """
    Calculate E_q [log sigma(s) - log q(s)]
    
    Args:
        trainer: Trainer instance (needed for shared_actorcritic and generate_kwargs)
        experience_maker: Experience maker instance
        args: Training arguments
        batch_prompt: Batch of prompts
        
    Returns:
        f_qs, attention_mask, num_actions, sequences, log_p, log_phi, log_q, action_mask
    """
    experience_maker.set_all_eval()
    batch_prompt = tile_prompts(batch_prompt, args.duplicate_rollout_batch_by)

    with torch.no_grad():
        if trainer.shared_actorcritic:
            action_log_probs, action_mask, attention_mask, num_actions, sequences, value = experience_maker.generate_seqs_and_get_logprobs(
                batch_prompt, **trainer.generate_kwargs)
        else:
            action_log_probs, action_mask, attention_mask, num_actions, sequences = experience_maker.generate_seqs_and_get_logprobs(
                batch_prompt, **trainer.generate_kwargs)
        action_log_probs = action_log_probs.float() * action_mask # more precision
        log_q = action_log_probs.sum(dim=-1)

        log_tilde_sigma, log_p, log_phi = eval_log_p_plus_log_phi(
            trainer, experience_maker, args, action_log_probs, attention_mask, action_mask, num_actions, sequences, return_extra_info=True, force_no_exploration_bonus=True
        )

        f_qs = log_tilde_sigma - log_q

    experience_maker.set_all_policies_train()

    return f_qs, attention_mask, num_actions, sequences, log_p, log_phi, log_q, action_mask


def g_q_estimate(trainer, experience_maker, args, true_sigma_samples, num_actions, attention_mask, condition_twist_on_tokens=None):
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
        if trainer.shared_actorcritic:
            action_log_probs, _ = experience_maker.actor(sequences,
                                                           num_actions,
                                                           attention_mask)
        else:
            action_log_probs = experience_maker.actor(sequences, num_actions,
                                          attention_mask)
        action_log_probs = action_log_probs.float() # more precision
        log_q = action_log_probs.sum(dim=-1)
        # Compute action_mask from sequences respecting EOS and pad tokens
        eos_token_id = trainer.generate_kwargs["eos_token_id"]
        pad_token_id = trainer.generate_kwargs["pad_token_id"]
        action_mask = compute_action_mask_from_sequences(sequences, num_actions, eos_token_id, pad_token_id)
        log_tilde_sigma = eval_log_p_plus_log_phi(trainer, experience_maker, args, action_log_probs,
                                attention_mask, action_mask,
                                num_actions, sequences, force_no_exploration_bonus=True)
        log_tilde_sigma = log_tilde_sigma.float() # more precision

    experience_maker.set_all_policies_train()

    return log_tilde_sigma - log_q


def f_q_g_q_evaluation(trainer, experience_maker, args, f_q_estimates_list, g_q_estimates_list,
                       iwae_lbs_list, iwae_ubs_list,
                       prompt_text, true_target_samples):
    """
    Evaluate f_q, g_q, and IWAE bounds.
    
    Args:
        trainer: Trainer instance (needed for n_seeds_f_q, generate_kwargs, and method calls)
        experience_maker: Experience maker instance
        args: Training arguments
        f_q_estimates_list: List to append f_q estimates to
        g_q_estimates_list: List to append g_q estimates to
        iwae_lbs_list: List to append IWAE lower bounds to
        iwae_ubs_list: List to append IWAE upper bounds to
        prompt_text: Prompt text for evaluation
        true_target_samples: True target samples
    """
    # This function appends to f_q_estimates_list and g_q_estimates_list
    iwae_lbs = torch.zeros((trainer.n_seeds_f_q,))
    iwae_ubs = torch.zeros((trainer.n_seeds_f_q,))
    total_f_qs = None
    total_g_qs = None
    for i in range(trainer.n_seeds_f_q):
        custom_prompt_for_f_q = [prompt_text] * args.n_samples_for_f_q

        f_qs, attention_mask, num_actions, q_seqs, log_p, log_phi, log_q, action_mask = f_q_estimate(
            trainer, experience_maker, args, custom_prompt_for_f_q)
        print("Avg F_q Estimate (Learned Model)")
        print(f_qs.mean())
        print("IWAE Lower Bound Estimate (Learned Model)")
        iwae_lower_bound_estimate = torch.logsumexp(f_qs,
                                                    dim=0) - torch.log(
            torch.tensor(f_qs.shape[0]))
        print(iwae_lower_bound_estimate)
        iwae_lbs[i] = iwae_lower_bound_estimate.item()
        # # TODO load the target samples, pass through to get g_q estimate
        # if true_target_samples is not None:
        #     true_target_samples = true_target_samples.to(
        #         q_seqs.device)
        #     # TODO later account for the above possiblity
        eos_token_id = trainer.generate_kwargs["eos_token_id"]
        pad_token_id = trainer.generate_kwargs["pad_token_id"]

        if i == 0:
            assert true_target_samples is not None
            range_val = (math.ceil(
                true_target_samples.shape[0] / args.n_samples_for_f_q))
            print(range_val)
            for j in range(range_val):
                samples = true_target_samples[
                          j * args.n_samples_for_f_q: (j + 1) * args.n_samples_for_f_q]
                if samples.shape[0] != 0:
                    print("G_q Estimates Learned Model")

                    attention_mask_g_q = (
                            samples.ne(eos_token_id) & samples.ne(
                            pad_token_id)).to(
                        dtype=torch.long)

                    g_qs = g_q_estimate(trainer, experience_maker, args, samples,
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

        if true_target_samples is not None:
            iwae_mixture_with_one_post = q_seqs.detach().clone()
            iwae_mixture_with_one_post[i] = true_target_samples[
                i]  # To keep the conditioning tokens constant
            attention_mask_g_q = (
                iwae_mixture_with_one_post.ne(eos_token_id) & iwae_mixture_with_one_post.ne(
                pad_token_id)).to(
                dtype=torch.long)
            iwae_ub_weights = g_q_estimate(trainer, experience_maker, args,
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

        else:
            total_f_qs = torch.cat((total_f_qs, f_qs), axis=0)
            print("F_Q Shape")
            print(total_f_qs.shape)

    iwae_lbs_list.append(iwae_lbs)
    iwae_ubs_list.append(iwae_ubs)
    print("IWAE LB AND UB")
    print(iwae_lbs)
    print(iwae_ubs)
    print("IWAE LB AND UB LISTS")
    print(iwae_lbs_list)
    print(iwae_ubs_list)
    print("Shapes")
    print(total_g_qs.shape)
    print(total_f_qs.shape)

    if total_g_qs is not None:
        g_q_estimates_list.append(
            total_g_qs.cpu())  # Only one G_q estimate (over all the target samples)
    f_q_estimates_list.append(total_f_qs.cpu())