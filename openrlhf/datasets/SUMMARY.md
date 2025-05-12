# OpenRLHF Datasets Summary

This document provides a summary of the scripts available in the `openrlhf/datasets` directory.

## `__init__.py`

### Purpose

This is the standard Python package initializer for the `openrlhf.datasets` module. It serves two main purposes:

1.  **Convenience Imports:** It imports the primary dataset classes (`ProcessRewardDataset`, `PromptDataset`, `RewardDataset`, `SFTDataset`, `UnpairedPreferenceDataset`) from their respective modules within the `datasets` directory. This allows users to import these classes directly from `openrlhf.datasets` (e.g., `from openrlhf.datasets import RewardDataset`) instead of needing to know the specific file names.
2.  **Wildcard Import Control:** It defines the `__all__` list, explicitly specifying which symbols (the dataset class names) are exported when a wildcard import (`from openrlhf.datasets import *`) is used. This helps maintain a clean namespace.

In essence, it makes the dataset classes defined in this package readily available for use elsewhere in the `openrlhf` library or in user scripts.

## `utils.py`

### Purpose

This script provides general utility functions used across different dataset classes within the `openrlhf.datasets` package.

### Functions

*   **`zero_pad_sequences(sequences, side: str = "left", value=0)`:**
    *   **Description:** Takes a list of 1D PyTorch tensors (sequences) and pads them to the same length, equal to the maximum length found in the input list.
    *   **Arguments:**
        *   `sequences`: A list of PyTorch tensors.
        *   `side`: Specifies whether to pad on the `"left"` (default) or `"right"` side of each sequence.
        *   `value`: The value used for padding (default is `0`).
    *   **Returns:** A single PyTorch tensor containing all the padded sequences stacked along the first dimension.
    *   **Use Case:** Essential for creating uniform batches from sequences of variable lengths, often used in collate functions for dataloaders.

*   **`exist_and_not_none(d, key)`:**
    *   **Description:** A simple helper to check if a key exists in a dictionary and its associated value is not `None`.
    *   **Arguments:**
        *   `d`: The dictionary to check.
        *   `key`: The key to look for.
    *   **Returns:** `True` if the key exists and its value is not `None`, `False` otherwise.
    *   **Use Case:** Simplifies checks for optional fields or configurations within dataset processing logic.

## `unpaired_preference_dataset.py`

### Purpose

This script defines the `UnpairedPreferenceDataset` class, a PyTorch `Dataset` tailored for preference learning algorithms like KTO (Kahneman-Tversky Optimization) that operate on *unpaired* preferences. Unlike standard preference datasets which contain pairs of (chosen, rejected) responses for each prompt, this dataset handles individual prompt-response pairs, each associated with a preference label (e.g., 1 for desirable, 0 for undesirable).

### Class: `UnpairedPreferenceDataset`

*   **Initialization (`__init__`):**
    *   Takes a raw dataset, tokenizer, maximum sequence length (`max_length`), training strategy object, and optional input template.
    *   Stores tokenizer, strategy, and max length.
    *   Retrieves configuration for input/output/label keys and chat template application from the strategy args.
    *   Applies chat templates if configured.
    *   Uses `datasets.map` for efficient, parallel preprocessing via the `process_data` method.
    *   Filters out samples where the prompt length exceeds `max_length`.
    *   Stores the processed prompts, responses, labels, and prompt lengths.

*   **Data Processing (`process_data`):**
    *   Called by `datasets.map` during initialization.
    *   Uses the `preprocess_data` helper function to extract prompt, response, and label from a single data point, handling input templates or chat templates.
    *   Tokenizes the prompt to calculate its length (`prompt_ids_len`).
    *   Returns a dictionary containing the processed prompt, response, label, and prompt length, or sets prompt to `None` if it's too long (for filtering).

*   **Collation Function (`collate_fn`):**
    *   Designed to be used with a DataLoader.
    *   Takes a batch of items (prompt, response, label, prompt_length) from the dataset.
    *   For each item `(x_i, y_i, label_i)`:
        *   Tokenizes the combined sequence `x_i + y_i`, ensuring proper EOS handling and truncation to `max_length`.
    *   **KTO-Specific Step:** Creates additional "unmatched" pairs for KL divergence estimation. For each original pair `(x_i, y_i)`, it also tokenizes `x_i + y_j`, where `y_j` is the response from the *next* item in the batch (`j = (i+1) % batch_size`). These unmatched pairs are assigned a distinct label (e.g., -1).
    *   Uses `zero_pad_sequences` (from `.utils`) to pad all tokenized `input_ids` and `attention_mask` tensors to the maximum length within the batch.
    *   Returns a batch containing padded `input_ids`, `attention_mask`, a tensor of `labels` (including those for unmatched pairs), and a list of `prompt_ids_lens`.

### Helper Function: `preprocess_data`

*   Takes a raw data dictionary and configuration arguments (templates, keys, chat template function).
*   Extracts the prompt, response, and label according to the provided configuration.
*   Returns the extracted `prompt`, `response`, and `label` strings/values.

### Use Case

This dataset is specifically designed for training models using KTO or similar unpaired preference optimization algorithms, where the loss function requires evaluating the model's likelihood on both the actual response `y_i` given prompt `x_i` and on mismatched responses `y_j` given `x_i` to estimate a KL divergence term.

## `sft_dataset.py`

### Purpose

This script defines the `SFTDataset` class, a PyTorch `Dataset` specifically designed for Supervised Fine-Tuning (SFT) of language models. It prepares prompt-response pairs for training models to follow instructions or continue text generation.

### Class: `SFTDataset`

*   **Initialization (`__init__`):**
    *   Takes a raw dataset, tokenizer, maximum sequence length (`max_length`), training strategy object, and optional configurations like `input_template`, `pretrain_mode`, `multiturn`, and `multiple_of` (for packing).
    *   Stores tokenizer, strategy, max length, and other configurations.
    *   Retrieves settings for input/output keys and chat template application from strategy args.
    *   Applies chat templates if configured.
    *   Uses `datasets.map` for efficient parallel preprocessing via the `process_data` method.
    *   Filters out invalid samples (e.g., excessively long prompts).
    *   Stores the processed prompts, responses, prompt lengths, and optionally `response_ranges` for multi-turn data.

*   **Data Processing (`process_data`):**
    *   Called by `datasets.map` during initialization.
    *   Handles multi-turn data by iterating through conversation history to identify assistant responses and calculate their token ranges (`response_ranges`) if `multiturn` is enabled.
    *   Uses the `preprocess_data` helper to extract prompt and response, applying templates or chat formatting (unless in `pretrain_mode`).
    *   Calculates the tokenized prompt length (`prompt_ids_len`).
    *   Returns a dictionary with processed data, setting prompt to `None` for filtering if invalid.

*   **Get Item (`__getitem__`):**
    *   Retrieves a processed prompt and response by index.
    *   Constructs the full text (`prompt + response` or just `prompt` in `pretrain_mode`).
    *   Tokenizes the full text, ensuring proper EOS handling and truncation.
    *   Returns `prompt_ids_len`, `input_ids`, `attention_mask`, and an `info` dictionary containing original text and optionally `response_ranges`.

*   **Standard Collation (`collate_fn`):**
    *   Takes a list of items from `__getitem__`.
    *   Pads `input_ids` and `attention_masks` to the maximum length in the batch using `zero_pad_sequences`.
    *   Returns batches of `prompt_ids_lens`, padded `input_ids`, padded `attention_masks`, and the `infos` dictionary.

*   **Packing Collation (`packing_collate_fn`):**
    *   An alternative collation function for improved efficiency.
    *   Concatenates multiple sequences from the batch into a single long sequence (`packed_input_ids`).
    *   Creates a corresponding `packed_attention_masks` tensor where each original sequence is marked with a unique index.
    *   Adjusts `response_ranges` for multi-turn data to reflect their positions in the packed sequence.
    *   Optionally pads the packed sequence length to be a multiple of `multiple_of`.
    *   Returns `prompt_ids_lens`, `packed_input_ids`, `packed_attention_masks`, and the `infos` dictionary (with adjusted lengths and ranges).

### Helper Function: `preprocess_data`

*   Extracts `prompt` and `response` strings from a raw data dictionary based on configuration (keys, templates, chat template function, multi-turn flag).

### Use Case

This dataset is fundamental for training language models in a supervised manner, teaching them to generate desired outputs (responses) based on given inputs (prompts). The `pretrain_mode` allows using the same class structure for continued pretraining on unstructured text. The `packing_collate_fn` provides an efficient way to train on packed sequences, minimizing wasted computation due to padding.

## `reward_dataset.py`

### Purpose

This script defines the `RewardDataset` class, a PyTorch `Dataset` designed for training Reward Models (RMs) or for preference optimization algorithms like DPO (Direct Preference Optimization). It handles datasets containing a prompt and a pair of responses: one preferred (`chosen`) and one dispreferred (`rejected`).

### Class: `RewardDataset`

*   **Initialization (`__init__`):**
    *   Takes a raw dataset, tokenizer, maximum sequence length (`max_length`), training strategy object, and optional flags like `input_template`, `is_dpo`, and `multiple_of` (for packing).
    *   Stores tokenizer, strategy, max length, and other configurations.
    *   Retrieves settings for prompt/chosen/rejected keys and chat template application from strategy args.
    *   Applies chat templates if configured.
    *   Uses `datasets.map` for efficient parallel preprocessing via the `process_data` method.
    *   Filters out invalid samples if `is_dpo` is True (e.g., excessively long prompts).
    *   Stores the processed prompts, chosen responses, rejected responses, and `extras` (either margins or prompt lengths depending on `is_dpo`).

*   **Data Processing (`process_data`):**
    *   Called by `datasets.map` during initialization.
    *   Uses the `preprocess_data` helper to extract prompt, chosen response, rejected response, and margin from a single data point.
    *   If `is_dpo` is True, calculates and stores `prompt_ids_len` in the `extra` field; otherwise, stores the `margin`.
    *   Returns a dictionary with processed data, setting prompt to `None` for filtering if `is_dpo` and the prompt is invalid.

*   **Get Item (`__getitem__`):**
    *   Retrieves a processed prompt, chosen response, rejected response, and extra value by index.
    *   Constructs the full texts: `prompt + chosen` and `prompt + rejected`.
    *   Tokenizes the `chosen` sequence and `rejected` sequence separately, ensuring proper EOS handling and truncation.
    *   Returns the tokenized `chosen_ids`, `chosen_mask`, `reject_ids`, `reject_mask`, and the `extra` value.

*   **Standard Collation (`collate_fn`):**
    *   Takes a list of items from `__getitem__`.
    *   Pads the `chosen_ids`/`chosen_masks` together and `reject_ids`/`rejects_masks` together using `zero_pad_sequences`.
    *   Padding side is "left" by default but changes to "right" if `is_dpo` is True.
    *   Returns batches of padded chosen sequences, padded rejected sequences, and the `extras` list.

*   **Packing Collation (`packing_collate_fn`):**
    *   An alternative collation function for efficiency.
    *   Concatenates all `chosen` sequences into one packed tensor and all `rejected` sequences into another.
    *   Creates corresponding `packed_attention_masks` where each original sequence is marked with a unique index (chosen sequences get indices 1 to N, rejected get N+1 to 2N).
    *   Optionally pads the combined packed sequence length to be a multiple of `multiple_of`.
    *   Returns `packed_input_ids` (chosen + rejected), `packed_attention_masks`, a list of original sequence lengths (`packed_seq_lens`), and the `extras` list.

### Helper Function: `preprocess_data`

*   Extracts `prompt`, `chosen`, `rejected`, and `margin` from a raw data dictionary based on configuration (keys, templates, chat template function, `is_dpo` flag).

### Use Case

This dataset is central to training Reward Models, which learn to predict human preferences by assigning higher scores to `chosen` responses than `rejected` ones given the same `prompt`. It's also directly usable for DPO training, which optimizes the policy model directly on preference pairs without an explicit RM. The `packing_collate_fn` offers an efficient training alternative.

## `prompts_dataset.py`

### Purpose

This script defines the `PromptDataset` class, a PyTorch `Dataset` designed to provide prompts for Reinforcement Learning (RL) training loops, particularly for algorithms like PPO.

### Class: `PromptDataset`

*   **Initialization (`__init__`):**
    *   Takes a raw dataset, tokenizer, training strategy object, and an optional `input_template`.
    *   Stores the strategy and tokenizer.
    *   Retrieves configuration for input key, label key, and chat template application from strategy args.
    *   Applies chat templates if configured, defaulting to a specific Llama-3 template if none is found on the tokenizer.
    *   Iterates through the raw dataset, using the `preprocess_data` helper for each entry.
    *   Stores the extracted prompts (and optional labels, though currently unused in `__getitem__`) in internal lists.
    *   Uses `tqdm` for progress display during preprocessing.

*   **Get Item (`__getitem__`):**
    *   Returns the preprocessed prompt string corresponding to the given index.
    *   Note: Contains commented-out code related to handling multiple samples per prompt, but the active code returns one prompt per index.

*   **Collation:** This dataset does *not* typically use a custom `collate_fn`. The prompts are usually tokenized and handled dynamically during the experience generation phase of the RL trainer, as they form the input for model generation.

### Helper Function: `preprocess_data`

*   Extracts a `prompt` string and an optional `label` string from a raw data dictionary based on configuration (keys, templates, chat template function).

### Use Case

This dataset serves as the source of prompts for the RL agent (policy model) during training. In each step of algorithms like PPO, batches of prompts are drawn from this dataset, the agent generates responses, the responses are evaluated (e.g., by a reward model), and this experience is used to update the agent.

## `process_reward_dataset.py`

### Purpose

This script defines the `ProcessRewardDataset` class, a PyTorch `Dataset` intended for training Process Reward Models (PRMs) or similar models. These models typically learn to assign rewards or labels to specific *steps* or *tokens* within a generated sequence, often marked by special placeholder tokens, rather than assigning a single score to the entire output.

### Class: `ProcessRewardDataset`

*   **Initialization (`__init__`):**
    *   Takes a raw dataset, tokenizer, maximum sequence length (`max_length`), training strategy object, and `multiple_of` (for packing).
    *   Stores tokenizer, strategy, max length, etc.
    *   Retrieves configuration for input key, label key, `placeholder_token`, and optional allowed `reward_tokens` from strategy args.
    *   Converts the `placeholder_token` string to its token ID.
    *   Stores the input sequences and corresponding labels directly from the dataset.

*   **Get Item (`__getitem__`):**
    *   Retrieves an input sequence and its corresponding list of labels by index.
    *   Tokenizes the input sequence, applying truncation.
    *   Processes the labels:
        *   If labels are strings (e.g., '+', '-'), converts them to their corresponding token IDs (using `convert_token_to_id`). Optionally validates against allowed `reward_tokens`.
        *   If labels are numbers (e.g., float rewards), converts them to a tensor.
    *   Creates a `labels` tensor of the same shape as `input_ids`, initialized with -100 (ignore index).
    *   Identifies the positions of the `placeholder_token_id` within the (potentially truncated) `input_ids`.
    *   Places the corresponding label values (token IDs or floats) into the `labels` tensor at the placeholder token positions.
    *   Handles truncation: ensures the number of labels assigned matches the number of placeholder tokens remaining after input truncation.
    *   Returns the `input_ids`, `attention_mask`, and the constructed `labels` tensor.

*   **Standard Collation (`collate_fn`):**
    *   Takes a list of items from `__getitem__`.
    *   Pads `input_ids`, `input_masks`, and `label_ids` to the maximum length using `zero_pad_sequences` with "right" padding.
    *   Returns batches of padded sequences.

*   **Packing Collation (`packing_collate_fn`):**
    *   An alternative collation function for efficiency.
    *   Concatenates `input_ids`, `attention_masks`, and `label_ids` from multiple sequences into respective packed tensors.
    *   Creates packed attention masks with unique indices per sequence.
    *   Optionally pads the packed sequence length to be a multiple of `multiple_of`.
    *   Returns packed tensors for inputs, masks, labels, and a list of original sequence lengths.

### Use Case

This dataset is used for training models that predict values or classifications at specific points within a sequence, typically marked by placeholders. This is useful for process supervision or step-wise reward modeling, where feedback is provided incrementally throughout a generation process rather than just at the end.
