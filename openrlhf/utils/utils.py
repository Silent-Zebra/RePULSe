import os
from pathlib import Path

from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

from openrlhf.utils import DeepspeedStrategy

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
        lr_str = f"actorlr{args.actor_learning_rate}"
    else:
        critic_loss_str = f"_criticloss{args.critic_loss_type}"
        lr_str = f"actorlr{args.actor_learning_rate}_criticlr{args.critic_learning_rate}"
    # if args.actor_modulates_base:
    #     extra_str = "actormodbase"
    if args.shared_actorcritic:
        lr_str = f"sharedactorcritic_lr{args.actor_learning_rate}"
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
        harmlessness_train_str = f"_harmloss_{args.harmlessness_training_loss_type}"

    pretrain_str = args.pretrain.split("/")[-1]
    reward_pretrain_str = args.reward_pretrain.split("/")[-1]

    info_name_str = f"{args.rm_type}_{pretrain_str}_{reward_pretrain_str}_beta{args.target_dist_beta}{harmlessness_train_str}_{args.parameterization}_{args.actor_loss_type}_epochs{args.max_epochs}{eval_str}_lrschedule{args.lr_scheduler}_{lr_str}{critic_loss_str}_adambetas{(args.adam_betas[0])}_{(args.adam_betas[1])}_{args.parameterization}{init_head_base_str}_sddivider{args.additional_sd_divider}_seed{args.seed}"

    return info_name_str