import os
from pathlib import Path

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
import torch

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
        lr_str = f"alr{args.actor_learning_rate}"
    else:
        critic_loss_str = f"_closs{args.critic_loss_type}"
        lr_str = f"alr{args.actor_learning_rate}_clr{args.critic_learning_rate}"
    # if args.actor_modulates_base:
    #     extra_str = "actormodbase"
    if args.shared_actorcritic:
        lr_str = f"sharedac_lr{args.actor_learning_rate}"
    if args.model_eval:
        eval_str = "_eval"
    if args.bc_coef > 0:
        lr_str += f"_bc{args.bc_coef}"

    if args.init_head_from_base:
        init_head_base_str = "_initheadbase"

    if args.critic_loss_type == "mixed_ctl_mse":
        lr_str += f"_alpha{args.alpha}"

    harmlessness_train_str = ""
    if args.do_harmlessness_training:
        lr_str += f"_blr{args.base_actor_learning_rate}"
        harmlessness_train_str = f"_harml_{args.harmlessness_training_loss_type}_a{args.alpha}"

    # pretrain_str = args.pretrain.split("/")[-1]
    pretrain_str = "".join([x[:2] for x in re.split(r"[-_]", args.pretrain.split("/")[-1])])

    # reward_pretrain_str = args.reward_pretrain.split("/")[-1]
    reward_pretrain_str = "".join([x[:2] for x in re.split(r"[-_]", args.reward_pretrain.split("/")[-1])])

    prompt_data_str = "".join([x[:2] for x in re.split(r"[-_]", args.prompt_data.split("/")[-1])])

    # adam_betas_str = f"_adambetas{(args.adam_betas[0])}_{(args.adam_betas[1])}"
    adam_betas_str = ""

    # sddiv_str = f"_sddiv{args.additional_sd_divider}"
    sddiv_str = ""

    n_episodes = args.num_episodes
    if args.do_harmlessness_training:
        n_episodes = args.harmlessness_training_num_episodes


    rm_type_str = args.rm_type
    if args.rm_type == "indicator_below_threshold":
        rm_type_str = f"ind_thresh{args.threshold}"
    if args.use_base_as_proposal:
        rm_type_str += "_baseprop"

    rew_trans_str = ""
    if args.reward_transform:
        rew_trans_str = f"rta{args.rew_trans_alpha}_b{args.rew_trans_beta}"

    info_name_str = f"{rm_type_str}_{pretrain_str}_{reward_pretrain_str}_{prompt_data_str}_len{args.generate_max_len}_kl{args.init_kl_coef}_beta{args.target_dist_beta}{harmlessness_train_str}{rew_trans_str}_{args.parameterization}_{args.actor_loss_type}_epo{args.max_epochs}_epi{n_episodes}{eval_str}_sch{args.lr_scheduler}_{lr_str}{critic_loss_str}{adam_betas_str}_{args.parameterization}{init_head_base_str}{sddiv_str}_s{args.seed}"

    return info_name_str


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
    assert start < 0 and end < 0
    sign = -1
    start_abs, end_abs = abs(start), abs(end)
    # Using natural logs (ln) and exp
    logs = np.linspace(np.log(start_abs), np.log(end_abs), steps)
    seq = np.exp(logs)
    return (sign * seq).tolist()