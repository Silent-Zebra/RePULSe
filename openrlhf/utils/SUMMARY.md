# OpenRLHF Utils Summary

This document provides a summary of the utility modules available in the `openrlhf/utils` directory.

## `__init__.py`

### Purpose

This file serves as the entry point for the `openrlhf.utils` package. It controls which functions and classes are directly importable from the `utils` namespace.

### Key Functionality

*   **Re-exports:** It explicitly imports and re-exports key utility functions from other modules within the `utils` directory, making them easily accessible via `from openrlhf.utils import ...`.
*   **Exports:** The `__all__` variable defines the public API of the `utils` package, listing the functions intended for external use:
    *   `get_processor`: From `processor.py`.
    *   `reward_normalization`: From `processor.py`.
    *   `blending_datasets`: From `utils.py`.
    *   `get_strategy`: From `utils.py`.
    *   `get_tokenizer`: From `utils.py`.

### Use Case

Provides a convenient and clean way to access the most commonly used utilities from the `openrlhf.utils` package without needing to import from the specific submodules directly.

## `distributed_sampler.py`

### Purpose

This module provides a custom `DistributedSampler` class, adapted from PyTorch's standard `DistributedSampler`. Its primary purpose is to partition a dataset across multiple processes (replicas) for distributed training, ensuring each process receives a unique, non-overlapping subset of the data.

### Key Functionality

*   **Distributed Data Partitioning:** Given a dataset, the number of replicas (`num_replicas`), and the rank of the current process (`rank`), it calculates the indices of the data samples assigned to the current process.
*   **Shuffling:** Optionally shuffles the entire dataset deterministically based on the epoch number and a seed before partitioning. This ensures that data order varies across epochs but remains consistent across processes within the same epoch.
*   **Handling Uneven Data:** Provides options (`drop_last`) to either drop the last few samples or pad the dataset with duplicates to ensure that the total number of samples is evenly divisible by the number of replicas. This guarantees that each process receives exactly the same number of samples.
*   **Resuming Training (`consumed_samples`):** Includes logic to handle resuming training from a checkpoint. By providing the number of `consumed_samples`, the sampler correctly skips the already processed data points for the current epoch, ensuring no data is repeated or skipped upon resuming.
*   **Epoch Setting:** The `set_epoch` method is crucial for shuffling. It must be called at the beginning of each epoch to ensure the shuffling pattern changes.

### Class: `DistributedSampler`

*   **Inheritance:** Inherits from `torch.utils.data.sampler.Sampler`.
*   **Initialization (`__init__`)**: Takes the dataset, optional `num_replicas`, `rank`, `shuffle` flag, `seed`, `drop_last` flag, and importantly, `consumed_samples` for resuming.
*   **Iteration (`__iter__`)**: Generates the list of indices assigned to the current rank for the current epoch, applying shuffling and handling `consumed_samples` and `drop_last` logic.
*   **Length (`__len__`)**: Returns the number of samples assigned to the current rank, accounting for `consumed_samples`.
*   **`set_epoch(epoch, consumed_samples=0)`**: Updates the current epoch number and recalculates the number of consumed indices for the current rank, essential for correct shuffling and resumption.

### Use Case

Essential component for setting up `torch.utils.data.DataLoader` in a distributed training environment (e.g., using PyTorch DDP or DeepSpeed). It ensures that each parallel training process works on its designated portion of the data, preventing data duplication and enabling correct gradient averaging and model updates across epochs, even when resuming from checkpoints.

## `distributed_util.py`

### Purpose

This module provides a utility function `init_process_group` which is a modified version of PyTorch's native `torch.distributed.init_process_group`. The key modification allows for the creation of multiple named process groups, which is generally not supported or easily managed by the standard PyTorch function that primarily focuses on initializing the default world process group.

### Function: `init_process_group`

*   **Signature:** Accepts parameters similar to `torch.distributed.init_process_group`:
    *   `backend`: Distributed backend to use (e.g., 'nccl', 'gloo').
    *   `init_method`: URL specifying how processes should coordinate (e.g., 'env://' or 'tcp://...').
    *   `timeout`: Timeout for operations.
    *   `world_size`: Total number of processes.
    *   `rank`: Rank of the current process.
    *   `store`: Key-value store for coordination (alternative to `init_method`).
    *   `group_name`: **Crucially**, a unique name for the process group being created.
    *   `pg_options`: Backend-specific options.
*   **Functionality:**
    1.  **Handles Initialization:** Manages the rendezvous process (either via `init_method` or a provided `store`) to establish communication between processes.
    2.  **Prefix Store:** If using `init_method`, it wraps the underlying store with a `PrefixStore` using the `group_name`. This prevents key collisions if the same store is used for multiple groups or other purposes.
    3.  **Creates Process Group:** Calls the internal PyTorch helper `_new_process_group_helper` to create a new process group instance associated with the specified `group_name` and configuration.
    4.  **Registers Group:** Stores the created process group and its associated ranks in the global `_world.pg_group_ranks` dictionary (this part seems less standard and might be specific to the library's internal management, potentially allowing lookup of groups by name later).
    5.  **Handles PyTorch Version Compatibility:** Detects the PyTorch version and uses the correct parameter name (`pg_options` or `backend_options`) for backend-specific configurations when calling `_new_process_group_helper`.
*   **Return Value:** Returns the newly created process group handle.

### Use Case

This utility is likely used in scenarios requiring more complex distributed communication patterns than a single default group provides. For example, it could be used in pipeline parallelism or other distributed training setups where different subsets of processes need to communicate independently within their own named groups while still being part of the overall distributed job.

## `logging_utils.py`

### Purpose

This module sets up a standardized logging configuration for the OpenRLHF project. It ensures consistent formatting and handling of log messages across different parts of the library.

### Key Functionality

*   **Root Logger Setup:** Defines and configures the root logger named `openrlhf`.
    *   Sets the default logging level to `DEBUG` internally.
    *   Creates a default stream handler (`_default_handler`) that writes to `sys.stdout`.
    *   Sets the handler's level to `INFO`, meaning only messages with level INFO or higher (WARNING, ERROR, CRITICAL) will be output by default via this handler.
    *   Attaches the handler to the root logger.
    *   Disables propagation (`_root_logger.propagate = False`) to prevent messages from being passed to any potential parent loggers (like the base Python root logger).
*   **Custom Formatter (`NewLineFormatter`):**
    *   Inherits from `logging.Formatter`.
    *   Overrides the `format` method to handle multi-line log messages. It inserts the standard log prefix (timestamp, level, file, line number) before each newline within a multi-line message, making them easier to read and parse.
    *   Uses a specific format string (`_FORMAT`) and date format (`_DATE_FORMAT`).
*   **Initialization Function (`init_logger`):**
    *   Provides a function to get a logger instance for a specific module or component (`name`).
    *   Configures the retrieved logger with the same settings as the root logger (DEBUG level, attaches the `_default_handler`, disables propagation).
*   **Automatic Setup:** The `_setup_logger()` function is called automatically when the module is imported, ensuring the root logger is configured early.

### Use Case

Provides a consistent way to log information throughout the OpenRLHF codebase. Developers should use `init_logger(__name__)` at the beginning of their modules to get a configured logger instance and then use standard logging methods (`logger.info`, `logger.warning`, `logger.debug`, etc.). The setup ensures logs are formatted uniformly and multi-line messages are aligned correctly.

## `processor.py`

### Purpose

This module provides data processing functions designed to transform datasets for specific training paradigms like Rejection Sampling, Conditional SFT, and Iterative DPO. It also includes a utility for reward normalization.

### Key Functions

*   **`reward_normalization(objs)`:**
    *   Takes a list of objects (dictionaries), each expected to have a `reward` key.
    *   Extracts all rewards, converts them to a tensor.
    *   Calculates the mean and standard deviation of the rewards.
    *   Normalizes the rewards (subtracts mean, divides by std deviation).
    *   Updates the `reward` key in the original objects with the normalized values.
    *   Used optionally by `conditional_sft_processor`.

*   **`conditional_sft_processor(args, objs)`:**
    *   **Purpose:** Prepares data for Conditional SFT (Supervised Fine-Tuning), where reward information is embedded directly into the input prompt. See [arXiv:2308.12050](https://arxiv.org/abs/2308.12050).
    *   **Workflow:**
        1.  Optionally normalizes rewards using `reward_normalization` if `args.normalize_reward` is True.
        2.  Determines the prompt template (defaulting to `"{input} <rm_score>: {reward} "` or using `args.reward_template`).
        3.  Iterates through objects, formatting the reward (to 2 decimal places).
        4.  Replaces `{input}` and `{reward}` placeholders in the template with the actual input and formatted reward.
        5.  Updates the `input` key in each object with the new formatted prompt.
    *   **Output:** Returns the list of modified objects.

*   **`rejection_sampling_processor(args, objs)`:**
    *   **Purpose:** Implements Rejection Sampling based on rewards. For each unique input prompt, it keeps only the output with the highest associated reward. See [arXiv:2307.09288](https://arxiv.org/abs/2307.09288).
    *   **Workflow:**
        1.  Uses a dictionary `out` to store the best output found so far for each unique `input`.
        2.  Iterates through objects. If the input is new, stores its output and reward.
        3.  If the input exists, compares the current object's reward with the stored reward. If the current reward is higher, updates the stored output and reward for that input.
    *   **Output:** Returns a new list of objects, containing only the highest-reward output for each unique input.

*   **`iterative_dpo_processor(args, objs)`:**
    *   **Purpose:** Prepares data potentially for an Iterative DPO (Direct Preference Optimization) setup. For each unique input prompt, it identifies the outputs with the highest (chosen) and lowest (rejected) rewards encountered so far.
    *   **Workflow:**
        1.  Uses a dictionary `out` to track the best chosen and worst rejected examples per input.
        2.  Iterates through objects. If the input is new, initializes chosen/rejected data with the current object's output/reward.
        3.  If the input exists, compares the current reward:
            *   If higher than `chosen_reward`, updates `chosen` and `chosen_reward`.
            *   If lower than `rejected_reward`, updates `rejected` and `rejected_reward`.
    *   **Output:** Returns a new list of objects, where each object represents a unique prompt and contains the identified `chosen` and `rejected` outputs along with their rewards.

*   **`get_processor(name)`:**
    *   A factory function that returns the appropriate processor function based on the provided `name` string ('rs', 'csft', 'iter_dpo').
    *   Raises a `ValueError` if the name is not recognized.

### Use Case

These processors are used to adapt datasets (typically containing prompts, outputs, and rewards) into formats suitable for specific advanced training techniques beyond standard SFT or PPO, enabling methods like rejection sampling, conditional fine-tuning, or iterative preference optimization.

## `remote_rm_utils.py`

### Purpose

This module provides utilities for interacting with a remote reward model (RM) service via an HTTP API. It allows obtaining reward scores for given text inputs without loading the RM locally.

### Key Functions

*   **`request_api_wrapper(url, data, score_key="rewards", try_max_times=5)`:**
    *   **Purpose:** A robust wrapper for making POST requests to a specified `url` with JSON `data`.
    *   **Features:**
        *   Sets appropriate `Content-Type` header.
        *   Includes a timeout (180 seconds).
        *   Implements a retry mechanism (`try_max_times`) with a 1-second delay between attempts.
        *   Checks for HTTP errors using `response.raise_for_status()`.
        *   Parses the JSON response and validates that the expected `score_key` is present.
        *   Logs errors encountered during requests.
        *   Raises an exception if all retry attempts fail.
    *   **Return:** Returns the list of scores extracted from the JSON response under the `score_key`.

*   **`remote_rm_fn(api_url, queries, prompts, labels, score_key="rewards")`:**
    *   **Purpose:** The main function to get rewards from a remote RM service.
    *   **Arguments:**
        *   `api_url`: The URL endpoint of the remote reward model API.
        *   `queries`: A list of input queries or prompts.
        *   `prompts`: A list of corresponding generated responses/prompts (the naming might be confusing, context suggests these are the texts to be scored alongside `queries`).
        *   `labels`: A list of corresponding labels (potentially for supervised signals, though usage might vary based on the specific RM API).
        *   `score_key`: The key in the API response JSON that contains the reward scores.
    *   **Workflow:**
        1.  Packages `queries`, `prompts`, and `labels` into a dictionary.
        2.  Calls `request_api_wrapper` to send the data to the `api_url` and retrieve the scores.
        3.  Converts the returned list of scores into a PyTorch tensor.
    *   **Return:** A PyTorch tensor containing the reward scores.

*   **`remote_rm_fn_ray(api_url, queries, prompts, labels, score_key="rewards")`:**
    *   **Purpose:** A Ray remote actor version of `remote_rm_fn`.
    *   **Decorator:** Decorated with `@ray.remote`, making it executable as a distributed Ray task/actor.
    *   **Functionality:** Simply calls the standard `remote_rm_fn`.

### Use Case

Used in distributed training setups (like PPO with Ray) where the reward model might be served as a separate microservice. The training coordinator or rollout workers can use `remote_rm_fn` (or `remote_rm_fn_ray` via Ray) to query this service for reward scores during the experience generation phase, avoiding the need to load the potentially large RM into every worker process.

## `utils.py`

### Purpose

This module serves as a collection of general-purpose utility functions used across the OpenRLHF project, particularly for data loading, tokenizer handling, strategy initialization, and miscellaneous tasks like prompt tiling and reward inspection.

### Key Functions

*   **Constants:** Defines default special tokens (`DEFAULT_PAD_TOKEN`, `DEFAULT_EOS_TOKEN`, `DEFAULT_BOS_TOKEN`, `DEFAULT_UNK_TOKEN`).

*   **`tile_prompts(prompts, samples_per_prompt)`:**
    *   Repeats each prompt in the input list `samples_per_prompt` times.
    *   Useful for generating multiple responses for the same prompt (e.g., in best-of-N sampling or PPO rollouts).

*   **`get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True)`:**
    *   Loads a Hugging Face tokenizer using `AutoTokenizer.from_pretrained` based on the `pretrain` path or name.
    *   Sets the `padding_side` (important for generation).
    *   Handles setting a default `pad_token` (using `eos_token`) if one is not already defined in the tokenizer config. Ensures the model's config is also updated.
    *   Crucially, it avoids resizing token embeddings if the tokenizer needs a pad token added, to maintain compatibility with external tools like VLLM which might expect the original vocabulary size.

*   **`get_strategy(args)`:**
    *   A factory function to instantiate and configure a `DeepspeedStrategy` object.
    *   Imports `DeepspeedStrategy` locally to avoid potential circular dependencies or unnecessary imports if DeepSpeed isn't used.
    *   Passes relevant arguments (`seed`, `max_norm`, batch sizes, `zero_stage`, `bf16`, etc.) from the `args` object to the strategy constructor.

*   **`blending_datasets(datasets, probabilities, strategy=None, seed=42, max_count=5000000, return_eval=True, stopping_strategy="first_exhausted", train_split="train", eval_split="test")`:**
    *   **Purpose:** Loads multiple datasets, potentially from different sources and formats, and interleaves them into single training and evaluation datasets based on specified probabilities.
    *   **Workflow:**
        1.  Parses comma-separated `datasets` and `probabilities` strings.
        2.  Iterates through each dataset identifier:
            *   Determines the dataset source/format (local Python script, local file [.json, .jsonl, .csv, .parquet], saved-to-disk directory, Hugging Face Hub path, ModelScope path if `args.use_ms`).
            *   Uses the appropriate `load_dataset` or `load_from_disk` function.
            *   Selects the specified `train_split` (or the default 'train') and `eval_split` (or a 3% slice of train if 'test' not available and `return_eval` is True), limiting the size with `max_count`.
        3.  Uses `datasets.interleave_datasets` to combine the loaded train and eval datasets according to `probabilities` and `stopping_strategy`.
    *   **Return:** Returns the interleaved training dataset, or a tuple of (train, eval) datasets if `return_eval` is True.

*   **`convert_token_to_id(token, tokenizer)`:**
    *   Converts a string token into its corresponding integer ID using the provided tokenizer.
    *   Asserts that the string token maps to exactly one ID.

*   **`get_info_name_str(args)`:**
    *   Constructs a descriptive string based on various training arguments (`args`).
    *   This string likely serves as a unique identifier for experiment runs, incorporating model names, learning rates, loss types, dataset names, hyperparameters, etc., into a concise filename-friendly format.
    *   Uses regular expressions and string formatting to abbreviate parts of the configuration.

*   **`inspect_rewards_list(rewards_list)`:**
    *   Takes a list of reward values.
    *   Converts it to a tensor.
    *   Prints the shape of the rewards tensor.
    *   Calculates and prints the average reward for the first N (5, 10, 50) and last N (500, 200, 100, 50, 10, 5) rewards recorded. Useful for quickly checking reward trends during or after training.

### Use Case

This module provides essential boilerplate functions frequently needed in training scripts: setting up tokenizers, initializing distributed strategies, loading and preparing potentially complex blended datasets, and generating informative run names or inspecting results.
