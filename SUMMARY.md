# OpenRLHF Repository Summary

This document provides a high-level overview of the OpenRLHF (Open Reinforcement Learning from Human Feedback) repository, summarizing the key components found in its subdirectories. OpenRLHF is designed to facilitate research and implementation of various large language model alignment techniques, including RLHF, preference-based optimization, and safety/harmlessness training.

## Core Components

### 1. CLI (`openrlhf/cli`)

This directory contains the main command-line interface scripts for running different training procedures, evaluations, and utilities. Key scripts include:

*   **Training:** `train_sft.py`, `train_rm.py` (deprecated, use preference trainers), `train_ppo.py`, `train_dpo.py`, `train_kto.py`, `train_kd.py`, `train_prm.py`, `train_ppo_ray.py`, `train_combined_harmlessness.py` (implementation within `train_ppo.py`). These scripts handle various alignment algorithms like Supervised Fine-Tuning, Proximal Policy Optimization, Direct Preference Optimization, Kahneman-Tversky Optimization, Knowledge Distillation, and Process Reward Modeling, with support for DeepSpeed and Ray for distribution.
*   **Evaluation:** `evaluate_gcg_sz.py`, `evaluate_beast.py`, `estimate_rare_harmful_outputs.py`. These scripts evaluate model robustness against adversarial attacks (GCG, BEAST) and estimate probabilities of rare outputs using LPE methods.
*   **Inference & Utilities:** `batch_inference.py`, `interactive_chat.py`, `lora_combiner.py`, `serve_rm.py`. These provide tools for batch generation/scoring, interactive testing, merging LoRA adapters, and serving reward models via an API.

### 2. Datasets (`openrlhf/datasets`)

This component focuses on defining PyTorch `Dataset` classes tailored for different training paradigms. It includes utilities for data processing and collation, including support for packing sequences for efficiency. Key dataset types:

*   `SFTDataset`: For Supervised Fine-Tuning (prompt-response pairs).
*   `RewardDataset`: For preference data (prompt, chosen response, rejected response), used for RM training and DPO/IPO.
*   `UnpairedPreferenceDataset`: For unpaired preference data (prompt, response, desirable/undesirable label), used for KTO.
*   `PromptDataset`: For providing prompts during RL rollouts (e.g., in PPO).
*   `ProcessRewardDataset`: For training Process Reward Models (sequences with step-wise annotations).
*   Utilities: Padding, data preprocessing helpers.

### 3. Evaluation (`openrlhf/evaluation`)

Contains tools and metrics for evaluating language models, particularly focusing on safety and robustness aspects related to alignment.

*   `load_lpe_distributions.py`: Utilities for creating input distributions for Low Probability Estimation (LPE) methods (e.g., uniform, hex character distributions).
*   `harmfulness_metrics.py`: Simple metrics for assessing sequence harmfulness, primarily by checking for specific undesirable token IDs.

### 4. LPE (`openrlhf/lpe`)

Implements methods for Wu and Hilton's Low Probability Estimation (LPE), specifically Iterated Tempered Geometric Importance Sampling (ITGIS) and Metropolis-Hastings Importance Sampling (MHIS). These algorithms are used in `estimate_rare_harmful_outputs.py` to estimate the probability of rare, potentially harmful, model outputs.
*(Note: The specific `SUMMARY.md` for this directory was empty, but its purpose is inferred from usage in evaluation scripts).*

### 5. Models (`openrlhf/models`)

Defines the core neural network architectures and loss functions used in OpenRLHF.

*   **Architectures:**
    *   `Actor`: The primary policy/language model class, wrapping Hugging Face transformers and integrating LoRA, quantization, Flash Attention, DeepSpeed, etc.
    *   `RewardModel`/`CriticModel`: Created via `get_llm_for_sequence_regression`, adapting base LLMs for sequence-level value prediction.
    *   `ActorCustom`, `ActorCritic`: Experimental models for research (e.g., logit modulation, shared actor-critic).
*   **Loss Functions:** Implementations for various objectives:
    *   `GPTLMLoss`: Standard causal language modeling loss.
    *   RL Losses: `PolicyLoss` (PPO Actor), `ValueLoss` (PPO Critic), `REINFORCELoss`, CTL, SIXO, DPG.
    *   Preference Losses: `PairWiseLoss`, `LogExpLoss`, `DPOLoss`, `KTOLoss`.
    *   Other Losses: `KDLoss` (Knowledge Distillation), `PRMLoss` (Process Reward Model).
*   **Utilities:** Functions for reward computation (including KL penalty), log probability calculation, masked operations, handling packed sequences, and RingAttention integration.

### 6. Trainer (`openrlhf/trainer`)

Contains trainer classes that encapsulate the complex logic of different training algorithms, orchestrating model interactions, data handling, optimization, and logging.

*   **RL Trainers:** `BasePPOTrainer` (implements PPO and advanced variants like CTL, SIXO, DPG), `HarmlessnessTrainer`, `CombinedHarmlessnessTrainer` (specialized for safety/adversarial alignment using techniques like REINFORCE and negative sampling).
*   **Preference Trainers:** `DPOTrainer`, `KTOTrainer`, `RewardModelTrainer`.
*   **Other Trainers:** `SFTTrainer`, `KDTrainer`, `ProcessRewardModelTrainer`.
*   All trainers integrate with distributed strategies (DeepSpeed), handle logging, checkpointing, and evaluation loops specific to their algorithms.

### 7. Utils (`openrlhf/utils`)

Provides a collection of general-purpose utility functions supporting various aspects of the library.

*   **Distributed Computing:** `DistributedSampler`, `init_process_group` (for custom process groups), `get_strategy` (DeepSpeed setup).
*   **Data Handling:** `blending_datasets` (loading and mixing multiple datasets), processor functions (`conditional_sft_processor`, `rejection_sampling_processor`, `iterative_dpo_processor`) for specific data transformations.
*   **Model Interaction:** `get_tokenizer`, `remote_rm_fn` (interacting with remote reward models).
*   **Logging:** Standardized logging setup (`init_logger`).
*   **Miscellaneous:** Prompt tiling, reward inspection, run name generation.

### 8. Forecasting Rare Outputs (`openrlhf/forecasting_rare_outputs`)

This module implements the methodology from Jones et al. (2025) "Forecasting Rare Language Model Behaviors" for estimating and forecasting the risk of rare, potentially harmful, model outputs at scale using Extreme Value Theory (EVT). It allows comparing models based on their forecasted worst-query risk `Q_p(n)` for specific behaviors at simulated deployment scales `n`.

*   `behavior_defs.py`: Defines structures for specifying the target harmful behaviors, including attributes like `target_sequence` and `target_keywords` used by specific elicitation methods.
*   `elicitation.py`: Implements functions to estimate `p_elicit`, the probability that a given query elicits the target behavior. Supports methods like:
    *   `logprob_target_sequence`: Calculating probability of a specific target sequence.
    *   `logprob_target_keyword_in_target_seq`: Calculating probability of target keywords within a target sequence.
    *   `repeated_sampling`: Monte Carlo estimation based on generated outputs.
*   `forecasting.py`: Contains the core EVT logic. Implements `fit_gumbel_tail` to fit the linear relationship between elicitation scores (`ψ = -log(-log p_elicit)`) and empirical log survival probabilities based on the top-k scores from an evaluation set. Implements `forecast_worst_query_risk` to predict the `1/n`-th upper quantile `Q_p(n)` using the fitted parameters.
*   `experiment_runner.py`: An executable script to run the end-to-end forecasting experiment for a given model, behavior, and query set. It handles data loading, orchestrates `p_elicit` estimation and Gumbel fitting/forecasting, and saves detailed and summary results.
*   `analysis.py`: A script to load results from multiple experiment runs, generate comparative plots (e.g., `Q_p(n)` vs. `n` for different models), and facilitate analysis.

This summary provides a starting point for navigating the OpenRLHF codebase. Refer to the `SUMMARY.md` files within each subdirectory for more detailed information on specific modules and scripts.
