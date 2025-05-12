# OpenRLHF CLI Scripts Summary

This document provides a summary of the command-line interface (CLI) scripts available in the `openrlhf/cli` directory.


## `train_ppo.py`

### Purpose

This script implements RLHF (Reinforcement Learning from Human Feedback) training for large language models using Proximal Policy Optimization (PPO) and related methods. It is designed to support a wide range of RLHF research workflows, including standard PPO, twist/proposal learning, and harmlessness training (outer-loop adversarial alignment). The script is highly configurable and supports various model architectures, reward models, and data handling strategies.

### Workflow

1. **Argument Parsing:**
   * Parses a comprehensive set of command-line arguments to configure model architectures, data sources, optimization hyperparameters, RL settings, evaluation modes, and more.

2. **Distributed Strategy Setup:**
   * Initializes distributed training using DeepSpeed via the `get_strategy` utility, supporting ZeRO and other memory/performance optimizations.

3. **Model Construction:**
   * Builds the actor (policy) model, optionally with LoRA adapters, FlashAttention, or custom parameterizations (e.g., modulation, shared actor-critic, etc.).
   * Optionally constructs a critic (value) model for PPO, or disables it for twist/proposal learning.
   * Loads or constructs a reward model, supporting local, remote, and custom reward models (e.g., OpenAssistant, GRM, etc.).
   * Optionally builds an EMA (exponential moving average) model for stability.
   * Handles freezing/unfreezing of model parameters as needed for different training regimes.

4. **Data Loading and Preparation:**
   * Loads and blends prompt datasets for RLHF training, supporting multiple sources and sampling probabilities.
   * Optionally loads SFT (supervised fine-tuning) datasets for pretraining or auxiliary objectives.
   * Prepares PyTorch datasets and dataloaders for efficient batching and shuffling.
   * Supports negative data evaluation (for adversarial or failure case analysis).

5. **Optimizer and Scheduler Setup:**
   * Configures optimizers (Adam) and learning rate schedulers for actor, critic, and base actor (for harmlessness training), with support for gradient clipping, weight decay, and custom learning rate schedules.

6. **Trainer Construction:**
   * Instantiates a `BasePPOTrainer` for standard PPO or a `CombinedHarmlessnessTrainer` for outer-loop harmlessness training (adversarial alignment).
   * Passes all relevant models, optimizers, schedulers, and configuration to the trainer.

7. **Training and Evaluation Loop:**
   * Runs the main training loop, which may include:
     * Standard PPO updates (actor/critic optimization on RLHF objectives).
     * Twist/proposal learning (for advanced RLHF/inference formulations).
     * Harmlessness training (outer loop): alternates between proposal/twist updates and base model harmlessness updates.
     * Evaluation-only modes: negative data evaluation, sampling-based evaluation, etc.
   * Tracks and logs key statistics (rewards, KL, entropy, etc.), and saves results and checkpoints as configured.

8. **Checkpointing and Saving:**
   * Loads and saves model checkpoints, negative data, and evaluation statistics.
   * Supports resuming training and saving intermediate/final models and results.

### Supported Features and Customizations

* **Model Architectures:**
  * Standard actor/critic, shared actor-critic, custom parameterizations (modulation, twist, etc.), LoRA adapters, FlashAttention, EMA.
* **Reward Models:**
  * Local, remote (API), custom (OpenAssistant, GRM, etc.), with flexible normalization and transformation options.
* **Data Handling:**
  * Blending of multiple prompt datasets, SFT/pretrain data, negative data evaluation, prompt templates, and custom input keys.
* **Optimization:**
  * PPO, twist/proposal learning, harmlessness training (outer loop), behavior cloning, gradient clipping, learning rate scheduling, and more.
* **Evaluation Modes:**
  * Standard RLHF training, negative data evaluation, sampling-based evaluation, outer-loop harmlessness training, and custom test info modes.
* **Distributed Training:**
  * DeepSpeed ZeRO, gradient checkpointing, offloading, and other memory/performance optimizations.
* **Logging and Saving:**
  * Extensive logging, checkpointing, and result saving for reproducibility and analysis.

### Key Arguments

* `--pretrain`, `--critic_pretrain`, `--reward_pretrain`: Model and reward model paths or HF names.
* `--prompt_data`, `--pretrain_data`: Data sources for RLHF and SFT/pretrain.
* `--num_episodes`, `--max_epochs`, `--rollout_batch_size`, `--micro_train_batch_size`, etc.: Training loop and batch size configuration.
* `--actor_learning_rate`, `--critic_learning_rate`, `--base_actor_learning_rate`: Learning rates for different model components.
* `--parameterization`, `--shared_actorcritic`, `--actor_modulates_base`, etc.: Model architecture and parameterization options.
* `--do_harmlessness_training`, `--harmlessness_training_num_episodes`, etc.: Outer-loop harmlessness training configuration.
* `--reward_transform`, `--normalize_reward`, `--rm_type`, `--threshold`, etc.: Reward model and RL objective configuration.
* `--save_path`, `--ckpt_path`, `--load_checkpoint`, etc.: Checkpointing and saving.
* `--only_evaluate_on_neg_data`, `--only_evaluate_do_sampling`: Evaluation-only modes.
* `--flash_attn`, `--bf16`, `--load_in_4bit`, `--gradient_checkpointing`, etc.: Performance and memory optimization flags.

### Utilities and Special Modes

* `do_load_checkpoints`: Loads model checkpoints and tracks consumed samples.
* `inspect_rewards_list`: Utility for analyzing reward statistics.
* Negative data evaluation: Loads and evaluates on negative/adversarial data.
* Sampling-based evaluation: Runs sampling-only evaluation for reward estimation.
* Extensive support for custom reward models, prompt templates, and advanced RLHF research workflows.

## `evaluate_gcg_sz.py`

### Purpose

This script evaluates the robustness of a target causal language model (referred to as the "actor") against Gradient-based Coordinate Gradient (GCG) attacks. GCG is an optimization technique used here to find adversarial prompt suffixes that aim to bypass the model's safety alignment and elicit undesirable or harmful responses (jailbreaking).

The script measures the effectiveness of these attacks by calculating the Attack Success Rate (ASR).

### Workflow

1.  **Initialization:** Sets up the distributed environment using `DeepSpeed ZeRO` via the `get_strategy` utility.
2.  **Model Loading:**
    *   Loads the base actor model specified by `--pretrain`.
    *   Optionally adapts the actor model based on `--parameterization` arguments (e.g., applies LoRA adapters if `--lora_rank > 0`).
    *   Loads a checkpoint if `--load_checkpoint` is specified.
    *   Prepares the actor model using the distributed strategy.
    *   Loads the corresponding tokenizer, ensuring a padding token is set.
3.  **Reward Model Loading (Optional):**
    *   If `--atk_success_criteria` is set to `reward`, it loads a reward model specified by `--reward_pretrain`.
    *   Supports loading standard sequence regression reward models and custom models (e.g., `OpenAssistant/reward-model-deberta-v3-large-v2`). Handles specific loading logic and tokenizer requirements for different reward model types.
    *   Prepares the reward model using the distributed strategy.
4.  **Data Loading:**
    *   Loads evaluation data from the CSV file specified by `--file_path`.
    *   Handles two scenarios defined by `--scenario`:
        *   `behaviors` (default): Expects 'goal' and 'target' columns in the CSV. The 'goal' is the user's harmful instruction, and the 'target' is the desired harmful output.
        *   `strings`: Expects only a 'target' column. The 'goal' is implicitly empty.
    *   Optionally limits the number of targets processed using `--max_targets`.
5.  **GCG Configuration:** Sets up the `nanogcg.GCGConfig` object with parameters controlling the attack (e.g., `--gcg_steps`, `--gcg_search_width`, `--gcg_topk`).
6.  **Generation Configuration:** Sets up the `transformers.GenerationConfig` for generating responses from the actor model (e.g., `--max_new_tokens`).
7.  **Evaluation Loop:** Iterates through each goal-target pair from the loaded data:
    *   **(Optional) Baseline Generation:** Generates responses from the *unattacked* actor model (using only the 'goal') for comparison purposes and reward calculation. Also calculates baseline log probabilities of the target string.
    *   **GCG Attack:** Runs `nanogcg.run` to find the optimal adversarial suffix (`result.best_string`) that maximizes the likelihood of the 'target' string when appended to the 'goal'.
    *   **Adversarial Prompt Construction:** Creates the final prompt by combining the original 'goal' (if applicable) and the found `best_string`. Applies the model's chat template.
    *   **Adversarial Generation:** Generates multiple (`--samples_per_prompt`) responses from the actor model using the adversarial prompt.
    *   **Log Probability Calculation:** Calculates the log probabilities of the target string given the *attacked* prompt.
    *   **Success Check:** Determines if the attack was successful for the current target based on `--atk_success_criteria`:
        *   `prefix`: Checks if *any* of the generated responses contain predefined refusal prefixes (e.g., "I'm sorry", "I cannot"). If *no* response contains a refusal, the attack is successful.
        *   `reward`: Calculates reward scores for *each* generated response using the loaded reward model. If *any* sample's reward score is less than `--reward_threshold`, the attack is successful for that target. It also calculates success based on `reward_threshold - 1`. If `--rm_exclude_adv_suffix` is true, it attempts to calculate the reward *only* on the original goal and the generated response, excluding the adversarial suffix.
    *   **Result Tracking:** Increments the count of successful attacks.
8.  **Summary Calculation:** After processing all targets, calculates the overall ASR (`(successful_attacks / total_targets_processed) * 100`).
9.  **Reporting:** Prints a summary including the ASR, configuration details, reward statistics (mean, min, std dev, median if using reward criterion), and average log probabilities (before and after attack).

### Key Arguments

*   `--file_path`: Path to the evaluation CSV data.
*   `--scenario`: Data format ('behaviors' or 'strings').
*   `--pretrain`: Path or HF name of the target actor model.
*   `--load_checkpoint`, `--ckpt_path`: To load actor weights from a checkpoint.
*   `--parameterization`, `--lora_rank`, etc.: Options for how the actor model is parameterized (e.g., full model fine-tuning vs. LoRA).
*   `--gcg_steps`, `--gcg_search_width`, etc.: Control the GCG optimization process.
*   `--max_new_tokens`: Max length of the generated response.
*   `--samples_per_prompt`: Number of responses to generate per adversarial prompt for robustness checking.
*   `--atk_success_criteria`: Method for judging success ('prefix' or 'reward').
*   `--reward_pretrain`: Path or HF name of the reward model (if `criteria='reward'`).
*   `--reward_threshold`: Threshold for reward-based success (success if `score < threshold`).
*   `--rm_exclude_adv_suffix`: Whether to exclude the adversarial suffix when calculating reward.
*   `--zero_stage`, `--local_rank`: DeepSpeed configuration.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`: Model performance/memory optimization flags.

### Utilities Used

*   `openrlhf.utils.get_strategy`: Initializes DeepSpeed.
*   `openrlhf.utils.get_tokenizer`: Loads the tokenizer.
*   `openrlhf.models.Actor`, `ActorCustom`: Define the actor model structure.
*   `openrlhf.models.get_llm_for_sequence_regression`, `_get_reward_model_custom`: Load reward models.
*   `nanogcg`: External library performing the GCG attack.
*   `evaluate_rewards`: Helper function within the script for reward calculation.
*   `check_log_prob`: Helper function for log probability calculation.
*   `strip_user_assistant_template`: Helper for parsing specific prompt formats.


## `estimate_rare_harmful_outputs.py`

### Purpose

This script is a command-line tool designed to estimate the probability that a given language model will generate a specific, predefined "target" token, particularly when this event is rare. It utilizes specialized algorithms from Low Probability Estimation (LPE), namely Iterated Tempered Geometric Importance Sampling (ITGIS) and Metropolis-Hastings Importance Sampling (MHIS), as well as a baseline brute-force Monte Carlo method, to estimate this probability under a defined input token distribution.

### Key Components & Workflow

1.  **Argument Parsing:** Uses `argparse` to accept numerous command-line arguments controlling model loading, estimation method, hyperparameters, input distribution, target specification, and distributed setup.
2.  **Strategy Initialization:** Leverages `openrlhf.utils.get_strategy` to configure the execution environment (e.g., DeepSpeed, device placement).
3.  **Model & Tokenizer Loading:** Loads the specified language model (`Actor` or `ActorCustom` from `openrlhf.models`) potentially with LoRA or custom parameterization, and its corresponding tokenizer (`openrlhf.utils.get_tokenizer`). Supports loading from pretrained paths and checkpoints.
4.  **Input Distribution:** Creates a token-factorized input distribution using `openrlhf.evaluation.load_lpe_distributions.get_token_factorized_distribution` based on provided arguments (`--distribution_name`, `--sequence_length`).
5.  **Target Definition:** Takes a `--target_word` argument, tokenizes it, and validates that it corresponds to exactly one token ID. This ID becomes the rare event target.
6.  **Estimator Execution:** Based on the `--method` argument:
    *   **ITGIS/MHIS:** Initializes and runs the corresponding estimator from `openrlhf.lpe.methods` with the model, input distribution, target token ID, and specific hyperparameters (`--temperature`, `--n_samples`, `--batch_size`, `--itgis_decay_rate`, `--mhis_steps`).
    *   **Brute-Force:** Performs direct Monte Carlo sampling. Generates sequences from the model using the input distribution and checks if the target token appears using `openrlhf.evaluation.harmfulness_metrics.check_bad_token`.
7.  **Results:** Prints the estimated probability, standard error (if applicable), execution time, and configuration details.

### Key Arguments / Configuration (via CLI)

*   `--pretrain`: Path to the base pretrained model.
*   `--load_checkpoint`: Flag to load a model checkpoint.
*   `--ckpt_path`: Path to the model checkpoint file (required if `--load_checkpoint` is set).
*   `--method`: Estimation method ('itgis', 'mhis', 'bruteforce').
*   `--target_word`: The specific word whose single-token generation probability is to be estimated.
*   `--distribution_name`: Type of input distribution ('uniform', 'hex', etc.).
*   `--sequence_length`: Length of the sequence for the input distribution.
*   `--n_samples`: Number of samples for the estimation method.
*   `--batch_size`: Batch size for model inference.
*   `--temperature`: Temperature parameter for LPE methods.
*   `--itgis_decay_rate`, `--mhis_steps`: Method-specific hyperparameters.
*   LoRA/Parameterization args (`--lora_rank`, `--parameterization`, etc.)
*   Strategy args (`--local_rank`, `--bf16`, etc.)

### Dependencies

*   Core: `torch`, `transformers`, `argparse`, `os`, `sys`, `time`, `warnings`
*   OpenRLHF Internal:
    *   `openrlhf.utils` (`get_strategy`, `get_tokenizer`)
    *   `openrlhf.models` (`Actor`, `ActorCustom`)
    *   `openrlhf.evaluation.harmfulness_metrics` (`check_bad_token`)
    *   `openrlhf.evaluation.load_lpe_distributions` (`get_token_factorized_distribution`)
*   LPE Library:
    *   `openrlhf.lpe.methods` (`ITGIS`, `MHIS`)
*   Note: Contains robust import handling with warnings/dummies if dependencies are missing.

### Use Case

This tool is used to perform quantitative analysis of language model behavior, specifically focusing on the likelihood of producing rare, potentially undesirable single-token outputs. Researchers can use it to compare models, evaluate alignment techniques, or understand the impact of different parameters on the probability of generating specific tokens under controlled input conditions.


## `batch_inference.py`

### Purpose

This script performs batch inference using a pre-trained language model. It takes a dataset of prompts as input and generates corresponding outputs (responses) from the model. It supports both standard Hugging Face transformers generation and optimized generation using VLLM. Additionally, it can perform inference using a pre-trained reward model to score prompt-response pairs.

### Functions

*   **`batch_generate_vllm(args)`:**
    *   **Description:** Performs batch generation using the VLLM library for optimized inference.
    *   **Workflow:** Loads a tokenizer and model using VLLM's API, configures sampling parameters, loads prompts from the specified dataset, generates outputs for each prompt (optionally using best-of-N sampling), and saves the input-output pairs to a JSON Lines file.
    *   **Key Arguments:** `--pretrain`, `--dataset`, `--output_path`, `--max_new_tokens`, `--tp_size`, `--best_of_n`, `--vllm`.

*   **`batch_generate(args)`:**
    *   **Description:** Performs batch generation using standard Hugging Face `transformers` and DeepSpeed for distributed inference.
    *   **Workflow:** Sets up DeepSpeed strategy, loads the actor model and tokenizer, prepares the model for distributed execution, loads and tokenizes prompts, iterates through prompts in batches, generates outputs using `model.generate()`, handles best-of-N sampling, gathers results from all processes, and saves the combined input-output pairs to a JSON Lines file.
    *   **Key Arguments:** `--pretrain`, `--dataset`, `--output_path`, `--max_new_tokens`, `--flash_attn`, `--bf16`, `--zero_stage`, `--best_of_n`.

*   **`batch_rm_inference(args)`:**
    *   **Description:** Performs batch inference using a reward model (RM) to score input-output pairs.
    *   **Workflow:** Sets up DeepSpeed strategy, loads the reward model and tokenizer, prepares the model for distributed execution, loads an SFT dataset (containing prompts and responses), iterates through the data, tokenizes prompt-response pairs, calculates reward scores using the RM, gathers results, and saves inputs, outputs, and their scores to a JSON Lines file.
    *   **Key Arguments:** `--reward_pretrain`, `--dataset`, `--output_path`, `--zero_stage`.

### Key Arguments / Configuration

*   `--pretrain`: Path or HF name of the generative model.
*   `--reward_pretrain`: Path or HF name of the reward model (for `batch_rm_inference`).
*   `--dataset`: Path or HF name(s) of the input dataset(s) (prompts or prompt-response pairs).
*   `--dataset_probs`: Probabilities for blending multiple datasets.
*   `--output_path`: File path to save the generated outputs or reward scores (in JSON Lines format).
*   `--vllm`: Flag to enable VLLM for generation (`batch_generate_vllm`).
*   `--max_new_tokens`: Maximum number of tokens to generate.
*   `--max_samples`: Maximum number of prompts/samples to process.
*   `--best_of_n`: Number of samples to generate per prompt for best-of-N selection.
*   `--temperature`, `--top_p`, `--repetition_penalty`: Generation sampling parameters.
*   `--zero_stage`, `--tp_size`: DeepSpeed and tensor parallelism configuration.
*   `--flash_attn`, `--bf16`: Performance/memory optimization flags.

### Dependencies

*   `torch`, `transformers`, `datasets`, `jsonlines`, `tqdm`
*   `vllm` (optional, if `--vllm` is used)
*   `openrlhf.datasets`, `openrlhf.models`, `openrlhf.utils`

### Use Case

Used to generate large batches of text data from a trained model for evaluation, data augmentation, or downstream tasks. It can also be used to score existing datasets using a reward model.

## `evaluate_beast.py`

### Purpose

This script evaluates the robustness of a target causal language model against the BEAST (Beam Search Adversarial Attack) method. BEAST is an optimization technique used to find adversarial prompt suffixes aiming to jailbreak the model (i.e., elicit harmful or unintended responses despite safety alignment).

The script runs the BEAST attack for each goal-target pair provided in an input file and measures the Attack Success Rate (ASR) based on whether the model's output contains refusal phrases.

### Workflow

1.  **Initialization:** Sets up the distributed environment using DeepSpeed via `get_strategy`.
2.  **Model Loading:**
    *   Loads the target actor model specified by `--pretrain`.
    *   Handles different model parameterizations (full model, LoRA via `--parameterization` arguments).
    *   Loads checkpoints if specified (`--load_checkpoint`).
    *   Prepares the actor model using the distributed strategy.
    *   Loads the corresponding tokenizer.
3.  **Data Loading:**
    *   Loads evaluation data (goal-target pairs) from the CSV file specified by `--file_path` (expects 'goal' and 'target' columns).
    *   Optionally limits the number of targets processed using `--max_targets`.
4.  **BEAST Configuration:** Extracts parameters for the BEAST attack (`--beast_k1`, `--beast_k2`, `--beast_length`, `--beast_ngram`, `--beast_temperature`, `--beast_top_p`).
5.  **Generation Configuration:** Sets up `transformers.GenerationConfig` for generating responses from the actor model (`--max_new_tokens`).
6.  **Evaluation Loop (`evaluate_with_beast` function):** Iterates through each goal-target pair:
    *   **BEAST Attack (`run_beast_attack` function):** Runs the integrated BEAST beam search logic to find an optimal adversarial suffix (`best_suffix`) that maximizes the likelihood of the 'target' string when appended to the 'goal'. This involves:
        *   Initializing candidates based on the `goal`.
        *   Iteratively extending candidates using beam search (`k1` width, `k2` new candidates per step).
        *   Scoring candidates using `attack_objective_targeted` (calculating negative log-likelihood of the `target`).
        *   Selecting the best suffix found after `beast_length` steps.
    *   **Adversarial Prompt Construction:** Creates the final prompt using `format_beast_input` (combines `goal` and `best_suffix` with appropriate chat templating).
    *   **Adversarial Generation:** Generates multiple (`--samples_per_prompt`) responses from the actor model using the adversarial prompt.
    *   **Success Check (`check_jailbreak` function):** Determines if the attack was successful for the current target by checking if *any* of the generated responses start with predefined refusal prefixes (e.g., "I'm sorry", "I cannot"). If *no* response contains a refusal prefix, the attack is counted as successful.
    *   **Result Tracking:** Increments the count of successful attacks.
7.  **Summary Calculation:** Calculates the overall ASR (`(successful_attacks / total_targets_processed) * 100`).
8.  **Reporting:** Prints a summary including the ASR and configuration details.

### Key Internal Functions (BEAST Logic)

*   `sample_top_p`: Samples tokens based on top-p probability.
*   `format_beast_input`, `format_beast_attack_initial`, `get_beast_assistant_prompt`: Handle chat templating for different stages of the attack.
*   `attack_objective_targeted`: Calculates the negative log-likelihood (loss) of the target string given a prefix (used for scoring candidates).
*   `run_beast_attack`: Implements the core BEAST beam search algorithm.
*   `check_jailbreak`: Checks if a generated response contains refusal prefixes.

### Key Arguments

*   `--file_path`: Path to the evaluation CSV data (containing 'goal', 'target').
*   `--pretrain`: Path or HF name of the target actor model.
*   `--load_checkpoint`, `--ckpt_path`: To load actor weights.
*   `--parameterization`, `--lora_rank`: Model parameterization options.
*   `--beast_k1`, `--beast_k2`: Beam search width parameters for BEAST.
*   `--beast_length`: Length of the adversarial suffix to generate.
*   `--beast_ngram`: Tokens to add per BEAST step.
*   `--beast_temperature`, `--beast_top_p`: Sampling parameters for BEAST internal generation.
*   `--max_new_tokens`: Max length of the final generated response.
*   `--samples_per_prompt`: Number of responses to generate per attack for ASR check.
*   `--zero_stage`, `--local_rank`: DeepSpeed configuration.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`: Model performance/memory optimization flags.

### Dependencies

*   `torch`, `transformers`, `pandas`, `numpy`
*   `openrlhf.utils`, `openrlhf.models`

### Use Case

Evaluates model safety/robustness specifically against the BEAST adversarial attack method by attempting to jailbreak the model using optimized adversarial suffixes and measuring the success rate.

## `evaluate_gcg.py`

Note: We do not use this script anymore, it has been replaced by evaluate_gcg_sz.py.

## `interactive_chat.py`

### Purpose

This script provides a command-line interface (CLI) for interactively chatting with a specified pre-trained language model. It allows a user to have a conversation with the model by entering prompts and receiving generated responses in real-time.

### Workflow (`generate` function)

1.  **Initialization:** Sets up a minimal strategy object (no DeepSpeed needed for single-process interactive chat).
2.  **Model Loading:** Loads an `Actor` model using the specified `--pretrain` path or name. Supports FlashAttention, bf16, and 4-bit loading. Uses `device_map="auto"` for automatic device placement.
3.  **Tokenizer Loading:** Loads the corresponding tokenizer, configured for left-padding.
4.  **Initial Prompt:** Optionally loads an initial system prompt or context from a file (`--ta_prompt`).
5.  **Chat Loop:**
    *   Prompts the user for input.
    *   Handles special commands: `exit` to quit, `clear` to reset the chat history.
    *   Formats the input:
        *   If `--apply_chat_template` is used, it maintains a list of conversation turns (`conversations`) and uses `tokenizer.apply_chat_template` to format the full history.
        *   Otherwise, it appends the new input to the existing `user_prompt` string using `--input_template`.
    *   Optionally appends a conditional SFT prompt (`--enable_csft`, `--csft_prompt`).
    *   Encodes the formatted prompt.
    *   Generates a response using `model.generate()`, applying sampling parameters (`--max_len`, `--greedy_sampling`, `--top_p`, `--temperature`, `--repetition_penalty`).
    *   Decodes the generated response, isolating the newly generated text.
    *   If using chat templates, adds the assistant's response to the `conversations` history.
    *   Prints the model's response to the console.
    *   Repeats until the user types `exit`.

### Key Arguments

*   `--pretrain`: Path or HF name of the model to chat with.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`: Model loading optimizations.
*   `--max_len`: Maximum sequence length for generation.
*   `--greedy_sampling`: Use greedy decoding instead of sampling.
*   `--top_p`, `--temperature`, `--repetition_penalty`: Sampling parameters.
*   `--input_template`: String template for formatting user input (used if `--apply_chat_template` is False).
*   `--apply_chat_template`: Use the tokenizer's built-in chat template for conversation history management.
*   `--ta_prompt`: Path to a file containing an initial system prompt or context.
*   `--enable_csft`, `--csft_prompt`: Options for conditional SFT prompting.
*   `--use_ms`: Flag to patch Hugging Face Hub for ModelScope downloads.

### Dependencies

*   `torch`, `transformers`
*   `openrlhf.models`, `openrlhf.utils`
*   `modelscope` (optional, if `--use_ms` is used)

### Use Case

Provides a simple way to manually interact with and test a trained language model directly from the command line for qualitative assessment or demonstration.

## `lora_combiner.py`

### Purpose

This script merges the weights of a LoRA (Low-Rank Adaptation) adapter into a base pre-trained Hugging Face transformer model and saves the resulting merged model to a new directory. This is useful for creating a standalone model with the LoRA adaptations baked in, eliminating the need to load the adapter separately during inference.

### Workflow (`apply_lora` function)

1.  **Load Base Model:** Loads the base transformer model (`AutoModelForCausalLM` or `AutoModelForSequenceClassification` if `--is_rm` is specified) and its tokenizer from `--model_path`. Supports `bfloat16` (`--bf16`).
2.  **Load LoRA Adapter:** Loads the LoRA adapter weights from `--lora_path` using `peft.PeftModel.from_pretrained`, attaching them to the base model.
3.  **Merge Weights:** Merges the LoRA weights into the base model's weights using `lora_model.merge_and_unload()`. This modifies the base model in place.
4.  **Save Merged Model:** Saves the modified base model (which now includes the merged LoRA weights) and its tokenizer to the specified `--output_path`.

### Key Arguments

*   `--model_path`: Path or HF name of the base pre-trained model.
*   `--lora_path`: Path to the directory containing the trained LoRA adapter weights (e.g., `adapter_config.json`, `adapter_model.bin`).
*   `--output_path`: Path to the directory where the merged model should be saved.
*   `--is_rm`: Flag indicating whether the base model is a reward model (uses `AutoModelForSequenceClassification` instead of `AutoModelForCausalLM`). Defaults to `False`.
*   `--bf16`: Flag to load the model in `bfloat16` precision.

### Dependencies

*   `torch`, `transformers`, `peft`

### Use Case

Used after training a LoRA adapter to create a deployable model that incorporates the adapter's modifications without requiring the `peft` library at inference time.

## `serve_rm.py`

### Purpose

This script launches a web server (using FastAPI and Uvicorn) to serve a pre-trained reward model (RM). It provides a simple API endpoint for calculating reward scores for given text inputs (queries/prompts).

### Workflow

1.  **Initialization (`RewardModelProxy` class):**
    *   Loads a reward model using `get_llm_for_sequence_regression` based on `--reward_pretrain`. Supports normalization, FlashAttention, bf16, 4-bit loading, and custom value head prefix.
    *   Loads the corresponding tokenizer.
    *   Stores configuration like `max_length` and `batch_size`.
2.  **Preprocessing (`strip_sequence` function):**
    *   A utility function to remove leading/trailing pad and EOS tokens from input strings.
3.  **Reward Calculation (`RewardModelProxy.get_reward` method):**
    *   Takes lists of queries and prompts as input.
    *   Cleans the queries using `strip_sequence` and appends an EOS token.
    *   Iterates through the queries in batches.
    *   Tokenizes each batch using `tokenize_fn`.
    *   **NOTE:** The provided code includes a `raise NotImplementedError` within this method's loop, indicating that the core reward calculation call (`reward_model(...)`) might be incomplete or requires modification in this version of the script.
    *   (Intended) Calculates reward scores using the loaded `reward_model`.
    *   Returns a list of scores.
4.  **API Server Setup:**
    *   Initializes a `FastAPI` application.
    *   Creates an instance of `RewardModelProxy`.
    *   Defines a POST endpoint `/get_reward`:
        *   Accepts a JSON request containing `query` (list of strings) and `prompts` (list of strings - Note: `prompts` is accepted but not currently used in the `get_reward` logic).
        *   Calls `reward_model.get_reward()` with the queries.
        *   Returns a JSON response containing a list of `rewards`.
5.  **Server Launch:** Starts the Uvicorn server to listen on the specified `--host` and `--port`.

### Key Arguments

*   `--reward_pretrain`: Path or HF name of the reward model.
*   `--normalize_reward`: Enable reward normalization during model loading.
*   `--value_head_prefix`: Prefix for the value head layer in the RM.
*   `--max_len`: Maximum sequence length for tokenization.
*   `--batch_size`: Batch size for processing reward calculations.
*   `--port`, `--host`: Network configuration for the server.
*   `--load_in_4bit`, `--bf16`, `--flash_attn`: Performance/memory optimization flags.
*   `--use_ms`: Flag to patch Hugging Face Hub for ModelScope downloads.

### Dependencies

*   `torch`, `transformers`, `fastapi`, `uvicorn`
*   `openrlhf.models`, `openrlhf.utils`
*   `modelscope` (optional, if `--use_ms` is used)

### Use Case

Intended to provide reward model scoring functionality as a network service. Other processes (like RL training or evaluation scripts) can then query this server via HTTP requests to get reward scores without needing to load the RM themselves, potentially saving memory on the client processes.

## `train_dpo.py`

### Purpose

This script trains a language model using Direct Preference Optimization (DPO). DPO is an algorithm for aligning language models with human (or AI) preferences expressed as pairwise comparisons (chosen vs. rejected responses), bypassing the need for an explicit reward model training step often found in RLHF pipelines like PPO.

### Workflow (`train` function)

1.  **Strategy Setup:** Initializes the distributed training strategy using DeepSpeed.
2.  **Model Loading:**
    *   Loads the policy model (`Actor`) to be trained, specified by `--pretrain`. Supports LoRA (`--lora_rank > 0`), FlashAttention, bf16, 4-bit loading, packing, and Liger kernel.
    *   Loads the reference model (`Actor`), specified by `--ref_pretrain`. This model is kept frozen during training and provides baseline probabilities for the DPO loss calculation. It can be offloaded to CPU (`--ref_offload`).
3.  **Tokenizer Loading:** Loads the tokenizer associated with the policy model.
4.  **Gradient Checkpointing:** Enables gradient checkpointing for the policy model if `--gradient_checkpointing` is set.
5.  **Optimizer and Scheduler Setup:**
    *   Creates an Adam optimizer for the policy model's trainable parameters.
    *   Sets up a cosine learning rate scheduler with warmup.
6.  **Data Loading and Preparation:**
    *   Loads and blends preference datasets specified by `--dataset` and `--dataset_probs`. Datasets are expected to contain pairs of chosen and rejected responses for given prompts.
    *   Creates `RewardDataset` instances for training and evaluation, configured for DPO (`is_dpo=True`). This dataset format likely structures the prompt, chosen response, and rejected response.
    *   Sets up dataloaders for training and evaluation using the configured strategy and batch sizes.
7.  **Strategy Preparation:** Prepares the policy model, reference model, optimizer, and scheduler for distributed training using the strategy.
8.  **Checkpoint Loading:** Optionally loads a previous checkpoint for the policy model, optimizer, and scheduler state.
9.  **Trainer Initialization:** Creates a `DPOTrainer` instance, passing the models, tokenizer, strategy, optimizer, dataloaders, scheduler, and DPO-specific hyperparameters (`--beta`).
10. **Training Loop:** Calls `trainer.fit()` to execute the DPO training loop. This involves:
    *   Iterating through the training data.
    *   Calculating forward passes for both the policy and reference models on the chosen and rejected responses.
    *   Computing the DPO loss, which involves comparing the log-probabilities assigned by the policy and reference models to the chosen/rejected responses.
    *   Performing backpropagation and optimizer steps.
    *   Running evaluation periodically.
    *   Saving checkpoints.
11. **Final Model Saving:** Saves the final trained policy model and tokenizer.

### Key Arguments

*   `--pretrain`: Path or HF name of the base model to be trained.
*   `--ref_pretrain`: Path or HF name of the reference model (often the same as `--pretrain` initially, but kept frozen).
*   `--dataset`, `--dataset_probs`: Preference dataset(s) containing chosen/rejected pairs.
*   `--beta`: The temperature parameter for the DPO loss (controls how much to penalize divergence from the reference model).
*   `--learning_rate`, `--lr_warmup_ratio`, `--adam_betas`, `--l2`: Optimizer and scheduler hyperparameters.
*   `--lora_rank`, `--lora_alpha`, `--lora_dropout`: LoRA configuration if used.
*   `--max_len`: Maximum sequence length.
*   `--micro_train_batch_size`, `--train_batch_size`: Batch size configuration.
*   `--max_epochs`, `--max_samples`: Training duration control.
*   `--gradient_checkpointing`: Enable gradient checkpointing.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`: Performance/memory optimization flags.
*   `--ref_offload`: Offload the reference model to CPU.
*   `--save_path`, `--ckpt_path`, `--load_checkpoint`: Checkpointing configuration.

### Dependencies

*   `torch`, `transformers`
*   `openrlhf.datasets`, `openrlhf.models`, `openrlhf.trainer`, `openrlhf.utils`

### Use Case

Used to fine-tune a language model based on preference data (chosen/rejected examples) using the DPO algorithm. It offers a potentially simpler alternative to PPO-based RLHF when a dataset of pairwise preferences is available.

## `train_kd.py`

### Purpose

This script trains a "student" language model using Knowledge Distillation (KD) from a larger, pre-trained "teacher" model. The goal is typically to transfer the capabilities of the teacher model to the smaller, more efficient student model. It uses a standard SFT dataset but incorporates a distillation loss comparing the student's output distribution (logits) to the teacher's.

### Workflow (`train` function)

1.  **Strategy Setup:** Initializes the distributed training strategy using DeepSpeed.
2.  **Model Loading:**
    *   Loads the student model (`Actor`) to be trained, specified by `--pretrain`. Supports LoRA, FlashAttention, bf16, 4-bit loading, and Liger kernel.
    *   Loads the teacher model (`Actor`), specified by `--teacher_model`. This model is kept frozen (in eval mode) during training and provides target logits for the distillation loss. It can be offloaded to CPU (`--teacher_offload`).
3.  **Tokenizer Loading:** Loads the tokenizer associated with the student model (assuming compatibility with the teacher).
4.  **Optimizer and Scheduler Setup:**
    *   Creates an Adam optimizer for the student model's trainable parameters.
    *   Sets up a learning rate scheduler (e.g., cosine) with warmup.
5.  **Data Loading and Preparation:**
    *   Loads and blends SFT datasets specified by `--dataset` and `--dataset_probs`. These datasets contain prompt-completion pairs.
    *   Creates `SFTDataset` instances for training and evaluation.
    *   Sets up dataloaders.
6.  **Gradient Checkpointing:** Enables gradient checkpointing for the student model if `--gradient_checkpointing` is set.
7.  **Strategy Preparation:** Prepares the student model, teacher model, optimizer, and scheduler for distributed training.
8.  **Checkpoint Loading:** Optionally loads a previous checkpoint for the student model.
9.  **Trainer Initialization:** Creates a `KDTrainer` instance, passing the student and teacher models, strategy, optimizer, dataloaders, scheduler, and KD-specific hyperparameters (`--kd_coef`).
10. **Training Loop:** Calls `trainer.fit()` to execute the KD training loop. This involves:
    *   Iterating through the training data.
    *   Calculating forward passes for both the student and teacher models on the input data.
    *   Computing a combined loss: typically a weighted sum of the standard SFT loss (cross-entropy between student logits and target labels) and the KD loss (e.g., KL divergence or MSE between student and teacher logits).
    *   Performing backpropagation and optimizer steps.
    *   Running evaluation periodically.
    *   Saving checkpoints.
11. **Final Model Saving:** Saves the final trained student model and tokenizer.

### Key Arguments

*   `--pretrain`: Path or HF name of the base model to be trained (student).
*   `--teacher_model`: Path or HF name of the frozen teacher model.
*   `--dataset`, `--dataset_probs`: SFT dataset(s) containing prompt-completion pairs.
*   `--kd_coef`: Weighting coefficient for the knowledge distillation loss term (vs. the standard SFT loss).
*   `--learning_rate`, `--lr_scheduler`, `--lr_warmup_ratio`, `--adam_betas`, `--l2`: Optimizer and scheduler hyperparameters.
*   `--lora_rank`, `--lora_alpha`, `--lora_dropout`: LoRA configuration for the student model if used.
*   `--max_len`: Maximum sequence length.
*   `--micro_train_batch_size`, `--train_batch_size`: Batch size configuration.
*   `--max_epochs`, `--max_samples`: Training duration control.
*   `--gradient_checkpointing`: Enable gradient checkpointing for the student.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`: Performance/memory optimization flags.
*   `--teacher_offload`: Offload the teacher model to CPU.
*   `--pretrain_mode`: Flag potentially modifying the loss calculation (e.g., only calculating loss on completions).
*   `--save_path`, `--ckpt_path`, `--load_checkpoint`: Checkpointing configuration.

### Dependencies

*   `torch`, `transformers`
*   `openrlhf.datasets`, `openrlhf.models`, `openrlhf.trainer`, `openrlhf.utils`

### Use Case

Used to train smaller language models by leveraging the knowledge encoded in larger teacher models. This can be useful for creating models that are faster and require fewer resources while retaining much of the performance of the teacher on the target task/data.

## `train_kto.py`

### Purpose

This script trains a language model using Kahneman-Tversky Optimization (KTO). KTO is a preference alignment algorithm similar in spirit to DPO, but it works with *unpaired* preference data. Instead of requiring matched chosen/rejected pairs for each prompt, KTO learns from datasets where individual examples are simply labeled as desirable or undesirable.

### Workflow (`train` function)

1.  **Strategy Setup:** Initializes the distributed training strategy using DeepSpeed.
2.  **Model Loading:**
    *   Loads the policy model (`Actor`) to be trained (`--pretrain`). Supports LoRA, FlashAttention, bf16, 4-bit loading, etc.
    *   Loads the reference model (`Actor`), typically the same base model (`--pretrain`), kept frozen (`--ref_pretrain` usually equals `--pretrain`). Provides baseline probabilities for KTO loss. Can be offloaded (`--ref_offload`).
3.  **Tokenizer Loading:** Loads the tokenizer associated with the policy model.
4.  **Gradient Checkpointing:** Enables gradient checkpointing if specified.
5.  **Optimizer and Scheduler Setup:** Configures Adam optimizer and a learning rate scheduler.
6.  **Data Loading and Preparation:**
    *   Loads and blends datasets (`--dataset`, `--dataset_probs`). These datasets are expected to contain individual examples (prompt + completion) along with a label indicating whether the completion is desirable or undesirable (e.g., a `label` column where 1=desirable, 0=undesirable).
    *   Creates `UnpairedPreferenceDataset` instances for training and evaluation. This dataset handles the specific format required by KTO.
    *   Sets up dataloaders.
7.  **Strategy Preparation:** Prepares models, optimizer, and scheduler for distributed training.
8.  **Checkpoint Loading:** Optionally loads a previous checkpoint.
9.  **Trainer Initialization:** Creates a `KTOTrainer` instance, passing models, tokenizer, strategy, optimizer, dataloaders, scheduler, and KTO-specific hyperparameters (`--beta`, `--desirable_loss_weight`, `--undesirable_loss_weight`).
10. **Training Loop:** Calls `trainer.fit()` to execute the KTO training loop. This involves:
    *   Iterating through the training data (desirable and undesirable examples).
    *   Calculating forward passes for both the policy and reference models.
    *   Computing the KTO loss, which separately encourages high probabilities for desirable examples and low probabilities for undesirable examples, relative to the reference model and scaled by the `beta` parameter. Weights for desirable/undesirable losses can be adjusted.
    *   Performing backpropagation and optimizer steps.
    *   Running evaluation periodically.
    *   Saving checkpoints.
11. **Final Model Saving:** Saves the final trained policy model and tokenizer.

### Key Arguments

*   `--pretrain`: Path or HF name of the base model to be trained and used as reference.
*   `--dataset`, `--dataset_probs`: Unpaired preference dataset(s) with desirable/undesirable labels.
*   `--beta`: Temperature parameter for the KTO loss.
*   `--desirable_loss_weight`, `--undesirable_loss_weight`: Weights to apply to the loss terms for desirable and undesirable examples, respectively.
*   `--learning_rate`, `--lr_warmup_ratio`, `--adam_betas`, `--l2`: Optimizer/scheduler hyperparameters.
*   `--lora_rank`, `--lora_alpha`, `--lora_dropout`: LoRA configuration.
*   `--max_len`: Maximum sequence length.
*   `--micro_train_batch_size`, `--train_batch_size`: Batch size configuration.
*   `--max_epochs`, `--max_samples`: Training duration control.
*   `--gradient_checkpointing`: Enable gradient checkpointing.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`: Performance/memory optimization flags.
*   `--ref_offload`: Offload the reference model to CPU.
*   `--save_path`, `--ckpt_path`, `--load_checkpoint`: Checkpointing configuration.

### Dependencies

*   `torch`, `transformers`
*   `openrlhf.datasets`, `openrlhf.models`, `openrlhf.trainer`, `openrlhf.utils`

### Use Case

Used to fine-tune a language model based on preference data where examples are labeled individually as good or bad, rather than paired. This can be useful when pairwise comparison data is harder to collect than simple quality labels.

## `train_prm.py`

### Purpose

This script trains a Process Reward Model (PRM). Unlike standard reward models that output a single score for a completed sequence, a PRM is designed to evaluate intermediate steps or turns within a generation process. It is often trained on data where reasoning steps or intermediate turns of a dialogue are annotated with feedback or scores.

### Workflow (`train` function)

1.  **Strategy Setup:** Initializes the distributed training strategy using DeepSpeed.
2.  **Model Loading:** Loads the base model (`Actor`) that will be trained as the PRM (`--pretrain`). Supports LoRA, FlashAttention, bf16, 4-bit loading, packing, etc. The PRM typically uses the base language model architecture but is trained to predict rewards/scores at specific token positions (often indicated by special tokens).
3.  **Tokenizer Loading:** Loads the tokenizer.
4.  **Gradient Checkpointing:** Enables gradient checkpointing if specified.
5.  **Optimizer and Scheduler Setup:** Configures Adam optimizer and a learning rate scheduler.
6.  **Data Loading and Preparation:**
    *   Loads and blends datasets (`--dataset`, `--dataset_probs`). These datasets are expected to be formatted for process supervision, likely containing sequences with annotated steps or turns.
    *   Creates `ProcessRewardDataset` instances for training and evaluation. This dataset class handles the specific format and logic for PRM training (e.g., identifying labeled steps).
    *   Sets up dataloaders, potentially using packing collation if `--packing_samples` is enabled.
7.  **Strategy Preparation:** Prepares the model, optimizer, and scheduler.
8.  **Checkpoint Loading:** Optionally loads a previous checkpoint.
9.  **Trainer Initialization:** Creates a `ProcessRewardModelTrainer` instance, passing the model, strategy, optimizer, dataloaders, scheduler, etc.
10. **Training Loop:** Calls `trainer.fit()` to execute the PRM training loop. This involves:
    *   Iterating through the training data.
    *   Calculating forward passes for the model.
    *   Computing a loss based on the model's predictions at the specifically annotated steps/tokens compared to the ground truth labels in the dataset.
    *   Performing backpropagation and optimizer steps.
    *   Running evaluation periodically.
    *   Saving checkpoints.
11. **Final Model Saving:** Saves the final trained PRM model and tokenizer.

### Key Arguments

*   `--pretrain`: Path or HF name of the base model architecture to train as a PRM.
*   `--dataset`, `--dataset_probs`: Dataset(s) formatted for process supervision.
*   `--learning_rate`, `--lr_scheduler`, `--lr_warmup_ratio`, `--adam_betas`, `--l2`: Optimizer/scheduler hyperparameters.
*   `--lora_rank`, `--lora_alpha`, `--lora_dropout`: LoRA configuration.
*   `--max_len`: Maximum sequence length.
*   `--micro_train_batch_size`, `--train_batch_size`: Batch size configuration.
*   `--max_epochs`, `--max_samples`: Training duration control.
*   `--gradient_checkpointing`: Enable gradient checkpointing.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`, `--packing_samples`: Performance/memory optimization flags.
*   `--save_path`, `--ckpt_path`, `--load_checkpoint`: Checkpointing configuration.
*   `--placeholder_token`, `--reward_tokens`: Potentially used by the `ProcessRewardDataset` to identify labeled steps (details depend on dataset implementation).

### Dependencies

*   `torch`, `transformers`
*   `openrlhf.datasets`, `openrlhf.models`, `openrlhf.trainer`, `openrlhf.utils`

### Use Case

Used to train reward models that provide feedback on the intermediate steps of a task (like mathematical reasoning or complex instruction following) rather than just the final output. This fine-grained feedback can be valuable for training more capable and reliable models.

## `train_ppo_ray.py`

### Purpose

This script implements PPO training for RLHF, specifically adapted for distributed execution using the Ray framework. It orchestrates the training process by distributing different model components (actor, critic, reference, reward models) across multiple Ray actors, potentially spanning multiple nodes and GPUs. It also supports using VLLM via Ray for accelerated generation during rollouts.

### Workflow (`train` function)

1.  **Argument Validation:** Checks compatibility of arguments related to distributed setup (e.g., batch sizes vs. world sizes, ZeRO stage with VLLM).
2.  **Strategy Setup:** Initializes the base strategy object (mostly for configuration, as Ray handles distribution).
3.  **Placement Group Setup (Optional):** If colocation flags (`--colocate_...`) are set, creates Ray Placement Groups (`placement_group`) to enforce specific model components reside on the same node/GPU for potential communication efficiency.
4.  **VLLM Engine Initialization (Optional):** If `--vllm_num_engines > 0`, creates a pool of VLLM Ray actors (`create_vllm_engines`) for handling text generation during the PPO rollouts.
5.  **Ray Actor Group Initialization:**
    *   Creates `PPORayActorGroup` instances for the Actor model (`ActorModelRayActor`).
    *   Creates groups for Reference (`ReferenceModelRayActor`), Critic (`CriticModelRayActor`), and Reward models (`RewardModelRayActor`) based on provided arguments (`--init_kl_coef > 0`, `--critic_pretrain`, `--reward_pretrain`). Handles potential colocation via placement groups (`pg`).
    *   Supports multiple reward models by creating a list of `PPORayActorGroup` for them (although current code asserts only one is supported).
6.  **Model Initialization (Distributed):**
    *   Calls asynchronous initialization methods (`async_init_model_from_pretrained`) on each actor group. Ray distributes these calls to the underlying actors.
    *   Waits for all initializations to complete using `ray.get()`.
    *   Initializes the critic model specifically after others, as its scheduler setup might depend on `max_steps` derived from the actor.
7.  **Distributed Training Execution:**
    *   Calls the main training method on the actor model group: `actor_model.async_fit_actor_model(...)`.
    *   This method orchestrates the PPO loop across all involved actor groups (actor, critic, ref, reward, potentially VLLM engines), handling rollouts, experience gathering, and optimization steps in a distributed manner.
    *   Waits for the training to complete using `ray.get()`.
8.  **Model Saving (Distributed):**
    *   Calls asynchronous save methods (`async_save_model`) on the actor group (and critic group if requested).
    *   Waits for saving to complete using `ray.get()`.

### Key Arguments (Ray/Distributed Specific)

*   `--actor_num_nodes`, `--actor_num_gpus_per_node`: Resource allocation for the actor model actors.
*   `--ref_num_nodes`, `--ref_num_gpus_per_node`: Resource allocation for the reference model actors.
*   `--critic_num_nodes`, `--critic_num_gpus_per_node`: Resource allocation for the critic model actors.
*   `--reward_num_nodes`, `--reward_num_gpus_per_node`: Resource allocation for the reward model actors.
*   `--vllm_num_engines`: Number of VLLM engines to launch for generation.
*   `--vllm_tensor_parallel_size`: Tensor parallelism for each VLLM engine.
*   `--vllm_gpu_memory_utilization`, `--enable_prefix_caching`, `--enforce_eager`, `--vllm_enable_sleep`: VLLM configuration.
*   `--colocate_actor_ref`, `--colocate_critic_reward`, `--colocate_all_models`: Flags to control model placement using Ray Placement Groups.
*   **(Inherited):** Standard PPO arguments (`--pretrain`, `--critic_pretrain`, `--reward_pretrain`, learning rates, batch sizes, KL coefficient, etc.) are passed to the underlying Ray actors for their internal PPO logic.

### Dependencies

*   `torch`, `transformers`, `ray`
*   `vllm` (optional, if `--vllm_num_engines > 0`)
*   `openrlhf.trainer.ray`, `openrlhf.utils`

### Use Case

Used for scaling up PPO training for RLHF to leverage multiple GPUs and potentially multiple machines, managed by the Ray framework. Offers potentially faster training through parallelization and optional VLLM integration for rollouts.

## `train_sft.py`

### Purpose

This script performs Supervised Fine-Tuning (SFT) on a pre-trained language model. SFT is used to adapt a model to a specific style, task, or domain by training it on a dataset of prompt-completion pairs using a standard language modeling objective (predicting the next token).

### Workflow (`train` function)

1.  **Strategy Setup:** Initializes the distributed training strategy using DeepSpeed.
2.  **Model Loading:** Loads the base model (`Actor`) to be fine-tuned (`--pretrain`). Supports LoRA, FlashAttention, bf16, 4-bit loading, packing, Liger kernel, etc.
3.  **Tokenizer Loading:** Loads the tokenizer associated with the model.
4.  **Gradient Checkpointing:** Enables gradient checkpointing if specified.
5.  **Optimizer and Scheduler Setup:** Configures an Adam optimizer and a learning rate scheduler.
6.  **Data Loading and Preparation:**
    *   Loads and blends SFT datasets (`--dataset`, `--dataset_probs`) containing prompt-completion pairs.
    *   Creates `SFTDataset` instances for training and evaluation. Handles different input templates and pre-training mode (calculating loss only on completions).
    *   Sets up dataloaders, potentially using packing collation.
7.  **Strategy Preparation:** Prepares the model, optimizer, and scheduler for distributed training.
8.  **Checkpoint Loading:** Optionally loads a previous checkpoint.
9.  **Trainer Initialization:** Creates an `SFTTrainer` instance, passing the model, strategy, optimizer, dataloaders, scheduler, etc.
10. **Training Loop:** Calls `trainer.fit()` to execute the SFT training loop. This involves:
    *   Iterating through the training data (prompt-completion pairs).
    *   Calculating forward passes and computing the standard cross-entropy language modeling loss (predicting the next token in the completion).
    *   Performing backpropagation and optimizer steps.
    *   Running evaluation periodically.
    *   Saving checkpoints.
11. **Final Model Saving:** Saves the final fine-tuned model and tokenizer.

### Key Arguments

*   `--pretrain`: Path or HF name of the base model to fine-tune.
*   `--dataset`, `--dataset_probs`: Dataset(s) containing prompt-completion pairs.
*   `--learning_rate`, `--lr_scheduler`, `--lr_warmup_ratio`, `--adam_betas`, `--l2`: Optimizer/scheduler hyperparameters.
*   `--lora_rank`, `--lora_alpha`, `--lora_dropout`: LoRA configuration.
*   `--max_len`: Maximum sequence length.
*   `--micro_train_batch_size`, `--train_batch_size`: Batch size configuration.
*   `--max_epochs`, `--max_samples`: Training duration control.
*   `--gradient_checkpointing`: Enable gradient checkpointing.
*   `--flash_attn`, `--bf16`, `--load_in_4bit`, `--packing_samples`: Performance/memory optimization flags.
*   `--pretrain_mode`: Flag to only calculate loss on the completion part of the sequence, ignoring the prompt.
*   `--save_path`, `--ckpt_path`, `--load_checkpoint`: Checkpointing configuration.

### Dependencies

*   `torch`, `transformers`
*   `openrlhf.datasets`, `openrlhf.models`, `openrlhf.trainer`, `openrlhf.utils`

### Use Case

Used as a primary step to adapt a pre-trained model to follow instructions, adopt a specific persona, or learn domain-specific knowledge before potentially undergoing further alignment like RLHF.
