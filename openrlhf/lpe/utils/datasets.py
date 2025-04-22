import hashlib
import inspect
import itertools
import json
import os
import pickle
from functools import wraps
from string import Formatter

from datasets import config as datasets_config
from datasets import load_dataset_builder
from tqdm import tqdm

datasets_config.HF_DATASETS_TRUST_REMOTE_CODE = False
USE_CACHE = False

def get_dataset(dataset_name, split):
    rev_files = False
    if split.endswith("_rev"):
        split = split[: -len("_rev")]
        assert split == "train", "should not use valid_rev, valid not large enough"
        rev_files = True

    if dataset_name == "c4":
        # 364,868,892 train items, 364,608 valid items, ~485 tokens per item
        builder = load_dataset_builder("allenai/c4", "en")
        split = "validation" if split == "valid" else split
        text_column = "text"
    elif dataset_name == "c4_es":
        builder = load_dataset_builder("allenai/c4", "es")
        split = "validation" if split == "valid" else split
        text_column = "text"
    elif dataset_name == "code":
        # 5,300,000 train items, 61,373 valid items, ~2,800 tokens per item
        builder = load_dataset_builder(f"codeparrot/codeparrot-clean-{split}")
        split = "train"
        text_column = "content"
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if rev_files:
        builder.config.data_files[split] = builder.config.data_files[split][::-1]
    dataset = builder.as_streaming_dataset(split=split)
    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")
    if dataset_name == "c4_es":
        dataset = dataset.remove_columns(
            "timestamp"
        )  # for some reason, it's a non-serializable object
    return dataset


def cache_json(filename_template):
    """
    Example usage:

    @cache_json("foo.json")
    def foo():
        return [1, 2, 3]

    @cache_json("bar_{string}.json")
    def bar(string):
        return {"string": string}

    @cache_json("baz_{#}.json")
    def baz(obj, val):
        return (val, obj.method(val))
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not USE_CACHE:
                return func(*args, **kwargs)

            sig = inspect.signature(func)
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()
            all_args = ba.arguments
            if "{#}" in filename_template:
                template_arg_names = [
                    k for _, k, _, _ in Formatter().parse(filename_template)
                ]
                other_args = {
                    k: v for k, v in all_args.items() if k not in template_arg_names
                }
                hashed_other_args = hashlib.sha256(pickle.dumps(other_args)).hexdigest()
                all_args["#"] = hashed_other_args[:12]
            filename = filename_template.format(**all_args)

            file_path = os.path.join(
                datasets_config.HF_DATASETS_CACHE, "hex_nn_local", filename
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if os.path.exists(file_path):
                with open(file_path, "r") as fh:
                    return json.load(fh)
            else:
                result = func(*args, **kwargs)
                with open(file_path, "w") as fh:
                    json.dump(result, fh)
                return result

        return wrapper

    return decorator


def download_cache():
    from boostedblob.cli import sync

    remote_dir = "gs://arc-ml/datasets/hex_nn/"
    local_dir = os.path.join(datasets_config.HF_DATASETS_CACHE, "hex_nn_local", "")
    sync(remote_dir, local_dir, delete=False)


@cache_json("tokens/{dataset_name}_{split}_{n_items}.json")
def get_token_dataset(dataset_name, tokenizer, split="train_rev", n_items=2**16):
    def tokenize(item):
        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        return {"tokens": [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]}

    dataset = get_dataset(dataset_name, split)
    dataset = dataset.map(tokenize)
    return list(tqdm(itertools.islice(dataset, n_items), total=n_items))
