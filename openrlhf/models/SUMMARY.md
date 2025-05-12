# OpenRLHF Models Summary

This document provides a summary of the model definitions and related utilities found in the `openrlhf/models` directory.

## `model.py`

### Purpose

This file provides core functionalities for loading and constructing specialized language models used in RLHF workflows, specifically Reward Models (RM) and Critic Models. It handles the integration of base Hugging Face transformer models with additional components like value heads and LoRA adapters.

### Key Functions

*   **`get_llm_for_sequence_regression(...)`**:
    *   **Description:** This is the main factory function for creating Reward Models and Critic Models. It takes a base model name/path and configuration options, then returns a model instance suitable for sequence regression tasks (predicting a single value per sequence).
    *   **Workflow:**
        1.  Loads the `AutoConfig` for the specified base model.
        2.  Dynamically determines the appropriate base model class (e.g., `LlamaModel`, `MistralModel`) and the corresponding pretrained base class (e.g., `LlamaPreTrainedModel`) using Hugging Face's model mapping or by inspecting the configuration's `auto_map`.
        3.  Calls internal helper functions (`_get_reward_model` or `_get_critic_model`) to construct a new class that inherits from the base pretrained class and incorporates the base LLM along with a value head.
        4.  Handles DeepSpeed ZeRO-3 integration for distributed model loading (`HfDeepSpeedConfig`).
        5.  Supports BitsAndBytes 4-bit quantization (`load_in_4bit`).
        6.  Loads the pretrained weights into the constructed class.
        7.  Applies LoRA adaptation if `lora_rank > 0` using `peft`. Includes specific handling for data types when using LoRA with 4-bit loading.
        8.  Configures the model for MoE balancing loss if applicable.
        9.  Disables `use_cache` for compatibility.
        10. (Previously) supported packing samples with Flash Attention 2 (currently marked as `NotImplementedError`).
        11. Provides an option (`init_value_head`) to manually initialize the value head weights, which can be necessary under certain DeepSpeed configurations.
    *   **Key Arguments:** `model_name_or_path`, `model_type` ("reward" or "critic"), `bf16`, `load_in_4bit`, `lora_rank`, `lora_alpha`, `target_modules`, `normalize_reward`, `use_flash_attention_2`, `ds_config`, `init_value_head`, `value_head_prefix`.

*   **`_get_reward_model(...)`**:
    *   **Description:** An internal helper function that dynamically creates a `RewardModel` class.
    *   **Structure:**
        *   Inherits from the base pretrained model class provided by `get_llm_for_sequence_regression`.
        *   Initializes the base LLM using the provided configuration.
        *   Adds a linear layer (`value_head`) on top of the base LLM to predict a scalar reward value.
        *   Includes optional reward normalization logic (maintaining running mean/std).
    *   **Forward Pass (`forward`):**
        1.  Calculates `position_ids` based on the `attention_mask`. Handles standard and packed sequence scenarios.
        2.  Passes inputs through the base LLM.
        3.  Passes the last hidden states through the `value_head`.
        4.  Extracts the reward value, typically from the position corresponding to the last non-padding token (or returns all values for packed sequences).
        5.  Applies normalization during inference if enabled.
        6.  Returns the scalar reward (or optionally, reward and base model outputs).

*   **`_get_critic_model(...)`**:
    *   **Description:** Similar to `_get_reward_model`, but creates a `CriticModel` class.
    *   **Structure:** Largely identical to `RewardModel`, inheriting from the base pretrained class and adding a `value_head`.
    *   **Forward Pass (`forward`):**
        1.  Similar to the Reward Model's forward pass, processing inputs through the base LLM and value head.
        2.  Crucially, it accepts an optional `action_mask`. If provided, it selects the value corresponding to the *last token of the action* within the sequence, using the `action_mask` to identify these positions. This is specific to how critics are often used in PPO, evaluating the value of states *before* specific actions. If no `action_mask` is given, it defaults to using the value at the end of the sequence (similar to the reward model).
        3.  Returns the scalar value (or optionally, value and base model outputs).

*   **`_get_reward_model_custom(...)`**:
    *   **Description:** An internal helper function likely intended for constructing reward models based on custom architectures or loading procedures (e.g., specific model names like "OpenAssistant/reward-model-deberta-v3-large-v2"). The implementation details seem specific to handling different reward model types found in the wild.
    *   **Forward Pass (`forward`):** Contains logic to handle different input formats and model-specific requirements, often involving tokenizing queries and answers separately and potentially applying custom logic based on the `rm_name`. *Note: The implementation might be specific to certain research paths.*

### Use Case

This file is central to RLHF implementations (like PPO) that require separate Reward or Critic models. It provides a standardized way to load various transformer architectures and adapt them for sequence-level value prediction, incorporating optimizations like LoRA, 4-bit loading, and DeepSpeed integration.

## `loss.py`

### Purpose

This file defines various loss functions used across different training paradigms within the OpenRLHF framework, including standard supervised learning, reinforcement learning (PPO, REINFORCE), preference-based learning (DPO, KTO, Pairwise), and knowledge distillation (KD).

### Key Loss Functions

*   **`GPTLMLoss`**:
    *   **Description:** Standard Causal Language Modeling loss (Cross-Entropy).
    *   **Functionality:** Calculates the cross-entropy loss between model `logits` and target `labels`, ignoring padding tokens (`IGNORE_INDEX = -100`). Handles logit/label shifting (`logits[..., :-1, :]` vs `labels[..., 1:]`). Includes support for RingAttention, correctly handling label slicing and loss aggregation across the ring.

*   **`PolicyLoss` (PPO Actor Loss)**:
    *   **Description:** Implements the clipped surrogate objective for the PPO actor.
    *   **Functionality:** Calculates the importance sampling ratio (`ratio = exp(log_probs - old_log_probs)`). Computes the unclipped loss (`surr1 = ratio * advantages`) and the clipped loss (`surr2 = clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages`). The final loss is the element-wise minimum of `surr1` and `surr2`, averaged over valid (non-masked) action tokens.

*   **`ValueLoss` (PPO Critic Loss)**:
    *   **Description:** Implements the loss function for the PPO critic.
    *   **Functionality:** Calculates the difference between the predicted `values` and the target `returns`. Optionally clips the predicted values (`values_clipped`) based on `old_values` and `clip_eps`. Computes the squared error loss for both the unclipped (`vf_loss1`) and clipped (`vf_loss2`) values. The final loss is the element-wise maximum of `vf_loss1` and `vf_loss2`, averaged over valid (non-masked) action tokens.

*   **`PairWiseLoss`**:
    *   **Description:** Simple pairwise ranking loss for preference data.
    *   **Functionality:** Calculates `loss = -log_sigmoid(chosen_reward - reject_reward + margin)`. Encourages the chosen reward to be higher than the rejected reward by at least a margin.

*   **`LogExpLoss`**:
    *   **Description:** Another pairwise ranking loss using log(1 + exp(difference)).
    *   **Functionality:** Calculates `loss = log(1 + exp(reject_reward - chosen_reward + margin))`. Also encourages separation between chosen and rejected rewards.

*   **`DPOLoss` (Direct Preference Optimization)**:
    *   **Description:** Implements the DPO loss, directly optimizing a policy based on preference pairs without an explicit reward model.
    *   **Functionality:** Calculates the log-ratio of policy vs. reference probabilities for chosen (`pi_logratios`) and rejected (`ref_logratios`) responses, scaled by `beta`. The loss is `log_sigmoid(ref_logratios - pi_logratios)`. Supports label smoothing and IPO (Identity PPO) loss variant.
    *   **Returns:** DPO loss, chosen rewards estimate, rejected rewards estimate.

*   **`KTOLoss` (Kahneman-Tversky Optimization)**:
    *   **Description:** Implements the KTO loss for unpaired preference data (desirable vs. undesirable examples).
    *   **Functionality:** Calculates policy vs. reference log-ratios for chosen, rejected, and KL-regularization examples, scaled by `beta`. Computes separate loss terms for desirable (`losses_chosen`) and undesirable (`losses_rejected`) examples based on the KTO formulation (`1 - sigmoid(...)` for chosen, `sigmoid(...)` for rejected). Returns weighted combination of these losses, plus KL estimates.

*   **`KDLoss` (Knowledge Distillation)**:
    *   **Description:** Calculates the KL divergence loss for knowledge distillation.
    *   **Functionality:** Computes the KL divergence between the log-softmax of the student `logits` and the log-softmax of the `teacher_logits` (often temperature-scaled, although temperature isn't explicit here). Masked by valid `label` tokens.

*   **`PRMLoss` (Process Reward Model Loss)**:
    *   **Description:** Calculates loss for training a Process Reward Model.
    *   **Functionality:** Identifies target steps/tokens using `placeholder_token_id` or `reward_token_ids`. Computes Binary Cross-Entropy with Logits (BCEWithLogitsLoss) between the model's logits at those specific target positions and the corresponding binary labels in the `labels` tensor. Optionally calculates accuracy.

*   **`REINFORCELoss`**:
    *   **Description:** Implements the basic REINFORCE policy gradient loss.
    *   **Functionality:** Calculates `loss = (masked_mean(-log_probs, action_mask) * final_reward).mean()`. Supports baseline subtraction (`expectation` or `hardcoded`).

*   **Research/Experimental Losses:**
    *   `NegTrainingLoss`: Combines REINFORCE with a term pushing down probabilities of negatively sampled sequences.
    *   `NegREINFORCELoss`: Applies REINFORCE logic to both positive and reweighted negative samples.
    *   `CTLLoss` (Contrastive Trajectory Learning - likely based on specific research): Complex loss involving reweighting based on policy and base model log-probs, potentially incorporating value function estimates.
    *   `MixedCTLValueLoss`: Blends standard Value Loss with a CTL-like term.
    *   `SIXOLoss` (likely based on specific research): Another complex contrastive/reweighting loss, potentially related to self-imitation or off-policy correction.
    *   `DPGLoss` (likely based on specific research): Resembles DPG/TD3 style updates, potentially incorporating gradients from a critic/value function applied to all possible next actions.

### Helper Functions

*   Contains helper functions primarily for calculating log-probability ratios and weights used in the research/experimental losses (CTL, SIXO, DPG), often involving detached computations (`get_positive_and_negative_weights_detached`, etc.).

### Use Case

This file centralizes the diverse loss computations needed for various model alignment and training strategies supported by OpenRLHF. Trainers (`SFTTrainer`, `PPOTrainer`, `DPOTrainer`, etc.) import and utilize the appropriate loss classes from this module based on their configuration.

## `utils.py`

### Purpose

This file contains various utility functions used primarily during the training and evaluation of models within the OpenRLHF framework, particularly focusing on calculations related to RL objectives and handling packed sequences.

### Key Functions

*   **`compute_approx_kl(...)`**:
    *   **Description:** Calculates the approximate KL divergence between two discrete probability distributions represented by their log-probabilities (`log_probs` and `log_probs_base`).
    *   **Functionality:** Implements the standard KL approximation `log_probs - log_probs_base`. Applies an `action_mask` if provided. Includes diagnostic prints if the mean KL is negative.
    *   **Note:** Contains commented-out code for alternative KL estimators (`k2`, `k3` from Schulman's blog), suggesting potential experimental directions, but the default implementation uses the standard `log_probs - log_probs_base` difference.

*   **`compute_reward(...)`**:
    *   **Description:** Computes the final reward signal used in PPO, combining an external reward (`r`) with a KL penalty term.
    *   **Functionality:**
        1.  Calculates the KL divergence using `compute_approx_kl`.
        2.  Computes the KL penalty (`kl_reward = -kl_coef * kl`).
        3.  Optionally clamps the external reward `r` between -10 and 10.
        4.  Creates a `last_reward` tensor where the external reward `r` is placed only at the position corresponding to the last non-masked token in each sequence (identified using `action_mask`).
        5.  Returns the sum of `last_reward` and `kl_reward`, along with the raw `kl` divergence.

*   **`log_probs_from_logits(...)`**:
    *   **Description:** Calculates log-probabilities from model logits, optionally selecting only the log-probabilities corresponding to given label tokens.
    *   **Functionality:**
        1.  Applies `log_softmax` to the input `logits` to get log-probabilities over the entire vocabulary (`log_probs_all_vocab`).
        2.  If `labels` are provided, uses `gather` to select the log-probabilities corresponding to the token indices in `labels`.
        3.  Returns either the log-probabilities for the labels, the log-probabilities for the full vocabulary, or both, based on the `return_type` argument.

*   **`return_or_gather_then_return(...)`**:
    *   **Description:** A helper function used by `log_probs_from_logits` and `log_probs_from_logits_with_modulation` to handle the return logic based on `return_type`.

*   **`log_probs_from_logits_with_modulation(...)`**:
    *   **Description:** Calculates log-probabilities similarly to `log_probs_from_logits`, but incorporates an additional `modulation` term added to the base log-probabilities before the final log-softmax.
    *   **Functionality:** Computes `log_softmax(log_softmax(logits) + modulation)`. Uses `return_or_gather_then_return` to select and return the appropriate log-probabilities.

*   **`masked_mean(...)`**:
    *   **Description:** Computes the mean of a tensor, masking out elements where the corresponding `mask` value is zero.
    *   **Functionality:** Calculates `(tensor * mask).sum() / mask.sum()` either globally or along a specified dimension `dim`.

*   **`masked_normalize(...)`**:
    *   **Description:** Normalizes a tensor (subtract mean, divide by standard deviation) using masked statistics.
    *   **Functionality:** Uses `masked_mean` to calculate the mean and variance, then applies normalization. Ensures variance is non-zero using `clamp(min=eps)`.

*   **`reset_position_ids(...)`**:
    *   **Description:** Generates position IDs specifically for packed sequences, where multiple sequences are concatenated and distinguished by values in the `attention_mask` (e.g., 1 for seq1, 2 for seq2, etc.).
    *   **Functionality:** Iterates through the unique sequence identifiers in the `attention_mask` and assigns position IDs starting from 0 for each identified sequence segment.

*   **`unpacking_samples(...)`**:
    *   **Description:** Takes a tensor containing packed sequences (typically reward/value predictions) and a list of the original sequence lengths (`packed_seqlens`).
    *   **Functionality:** Splits the packed tensor back into a list of tensors, where each tensor corresponds to one of the original sequences.

### Use Case

These utilities are essential components of the RL training loops (especially PPO), providing functions for KL calculation, reward shaping, log-probability extraction, and handling the complexities introduced by packing multiple sequences into single tensors for efficiency.

## `actor.py`

### Purpose

This file defines the `Actor` class, which serves as the primary interface for the policy model in RLHF training (e.g., PPO) and for standard generative inference.

### Class: `Actor`

*   **Inheritance:** `torch.nn.Module`

*   **Initialization (`__init__`)**:
    *   **Model Loading:**
        *   Can accept either a pre-loaded model object or a `model_name_or_path` string.
        *   If a string is provided, it loads a Causal LM using `transformers.AutoModelForCausalLM` (or `liger_kernel.transformers.AutoLigerKernelForCausalLM` if `use_liger_kernel=True`).
        *   Handles standard Hugging Face model loading arguments: `trust_remote_code`, `attn_implementation` (for Flash Attention 2), `quantization_config` (for 4-bit loading via BitsAndBytes), `torch_dtype`, `device_map`.
        *   Integrates with DeepSpeed ZeRO-3 for distributed loading (`HfDeepSpeedConfig`).
    *   **LoRA Integration:**
        *   Applies LoRA using `peft.get_peft_model` if `lora_rank > 0`.
        *   Configures LoRA specifically for `TaskType.CAUSAL_LM`.
        *   Includes specific dtype handling for LoRA layers when using 4-bit quantization.
    *   **MoE Configuration:** Sets `output_router_logits = True` in the model config if detected, potentially for calculating balancing losses elsewhere.
    *   **Caching:** Disables `use_cache` in the model config by default for training compatibility (generation overrides this).
    *   **Packing Samples:** Stores the `packing_samples` flag (logic for applying packing patches seems commented out or removed).
    *   **Temperature:** Stores the generation `temperature`.

*   **Generation (`generate`)**:
    *   **Description:** Wraps the underlying Hugging Face model's `generate` method for producing token sequences.
    *   **Functionality:**
        1.  Constructs a dictionary of generation arguments (`top_k`, `top_p`, `do_sample`, `temperature`, `num_beams`, `max_new_tokens`, `attention_mask`, `eos_token_id`, `pad_token_id`, etc.) based on provided `kwargs`.
        2.  Calls `self.model.generate(**generate_args)`.
        3.  Calls `self.process_sequences` to post-process the output sequences and generate relevant masks.
    *   **Returns:** `sequences`, `attention_mask`, `action_mask`.

*   **Sequence Processing (`process_sequences`)**:
    *   **Description:** Post-processes the raw output sequences from `generate` to create masks needed for RL training.
    *   **Functionality:**
        1.  Identifies the actual end of the generated sequence (before padding or EOS tokens) using the initial `attention_mask` derived from `eos_token_id` and `pad_token_id`.
        2.  **Important:** Includes commented-out code and discussion about a potential issue where manually setting the EOS token at the end of sequences (especially when `max_length` is reached) might distort KL calculations in RL. The current implementation seems to *avoid* explicitly setting the EOS token at the end based on this concern.
        3.  Refines the `attention_mask` to only cover tokens from the start of the prompt up to the identified end of the generated sequence.
        4.  Creates an `action_mask` suitable for RL: it covers the generated tokens (excluding the prompt) and ensures the first generated token is always masked as valid.
    *   **Returns:** `sequences` (potentially unmodified from `generate`), `attention_mask` (refined), `action_mask`.

*   **Forward Pass (`forward`)**:
    *   **Description:** Performs a forward pass through the underlying language model to get logits and compute action log-probabilities.
    *   **Functionality:**
        1.  Calculates `position_ids`. Handles standard sequences (using cumulative sum of `attention_mask`) or packed sequences (using `reset_position_ids` or `convert_ring_attn_params` for RingAttention).
        2.  Calls the underlying `self.model` with `sequences`, `attention_mask` (if not packing), and `position_ids`.
        3.  Ensures logits are `float32`.
        4.  Selects logits corresponding to the generated actions (`logits[:, :-1]`).
        5.  Divides logits by `temperature` if specified.
        6.  Calls `log_probs_from_logits` to compute the log-probabilities of the actual generated sequence tokens (`sequences[:, 1:]`).
        7.  Returns the action log-probabilities (or other outputs based on `return_output`, `return_type`, `return_unnormalized`).

*   **Utility Methods:**
    *   `gradient_checkpointing_enable()`: Enables gradient checkpointing on the underlying model.
    *   `gradient_checkpointing_disable()`: Disables gradient checkpointing.
    *   `print_trainable_parameters()`: Utility to print the number of trainable parameters (useful when using LoRA).

### Use Case

This class is the workhorse policy model for OpenRLHF. It's used in: 
1.  **RL Training (PPO):** The `forward` method computes action log-probabilities needed for the policy loss. The `generate` method (via trainer/experience maker) produces rollouts.
2.  **Supervised Fine-Tuning (SFT, DPO, KTO, etc.):** The `forward` method provides logits used in standard cross-entropy or preference-based losses.
3.  **Inference/Evaluation:** The `generate` method is used to get model outputs for prompts.

## `actor_custom.py`

### Purpose

This file defines custom actor models (`ActorCustom`, `ActorCritic`) that extend the base `Actor` functionality, primarily for experimental research purposes involving different model parameterizations (like logit modulation) and shared actor-critic architectures.

### Class: `NNHead`

*   **Description:** A simple Multi-Layer Perceptron (MLP) head intended to replace a standard linear head, potentially offering more expressive power.
*   **Structure:** Consists of three linear layers with ReLU activations in between (`Linear -> ReLU -> Linear -> ReLU -> Linear`).
*   **Initialization:** Supports optional Xavier initialization and scaling of weights.

### Class: `ActorCustom`

*   **Inheritance:** `torch.nn.Module`
*   **Purpose:** Extends the `Actor` model to support alternative parameterizations, specifically focused on *logit modulation*. Instead of directly outputting logits, this model can output a *modification* that is added to the logits of a base/initial model.
*   **Initialization (`__init__`)**:
    *   Takes both `pretrain_or_model` (for the modification network/head) and `initial_model` (the frozen base model) as input.
    *   **Parameterization Handling:**
        *   `modulation_model`: Loads a full separate model (`pretrain_or_model`) whose `lm_head` output is interpreted as the modulation. Can optionally initialize this head's weights based on the `initial_model`'s head, scaled by `additional_sd_divider`.
        *   `modulation_linear_head` or `modulation_nn_head`: Uses the *frozen* `initial_model`'s hidden states and adds a *trainable* head (`modulation_head`) on top. This head can be a simple linear layer (`modulation_linear_head`) or the `NNHead` (`modulation_nn_head`). Supports initializing the linear head from the base model's head.
    *   Model loading (for `modulation_model` parameterization or if a path is passed) is similar to the base `Actor`, supporting Flash Attention, 4-bit, LoRA, DeepSpeed, etc.
*   **Generation (`generate`)**:
    *   **Description:** Overrides the standard generation to incorporate the modulation logic.
    *   **Functionality:**
        1.  Calls `self.custom_generate` which implements the core logic.
        2.  `custom_generate` iteratively performs:
            a.  Forward pass through the model(s) to get logits (potentially combining base model logits and modulation).
            b.  Applies sampling (top-k, top-p, temperature) to the potentially modulated logits.
            c.  Appends the sampled token to the sequence.
        3.  Handles stopping conditions (`eos_token_id`, `max_new_tokens`).
        4.  Calls `self.process_sequences` for post-processing.
*   **Forward Pass (`forward`)**:
    *   **Description:** Computes logits and optionally log-probabilities, incorporating the specified parameterization.
    *   **Functionality:**
        1.  Gets hidden states from the appropriate model (either the full `self.model` if `parameterization='modulation_model'`, or `self.initial_model` if using a `modulation_head`).
        2.  Calculates the modulation term (output of `self.model.lm_head` or `self.modulation_head`).
        3.  If `return_only_modulation`, returns the modulation directly.
        4.  Otherwise, gets the base logits from `self.initial_model`.
        5.  Adds the modulation to the base logits.
        6.  Computes log-probabilities from the *final modulated logits* using `log_probs_from_logits_with_modulation` (which handles the internal log-softmax correctly).
        7.  Returns log-probabilities or logits based on `return_type`, etc.
*   **Other Methods:** Includes `smc_procedure` (likely for Sequential Monte Carlo experiments, currently seems basic), `process_sequences` (similar to base `Actor`), `get_position_ids`, gradient checkpointing controls, and parameter printing utilities.

### Class: `ActorCritic`

*   **Inheritance:** `torch.nn.Module`
*   **Purpose:** Implements a shared-trunk actor-critic model where the base transformer body is shared, but separate linear heads are used for predicting policy logits (`lm_head`) and value estimates (`value_head`).
*   **Initialization (`__init__`)**:
    *   Loads a base Causal LM using standard procedures (Flash Attention, 4-bit, LoRA, DeepSpeed, etc.).
    *   Adds a separate `value_head` (linear layer) on top of the loaded model's hidden size.
    *   Initializes the `value_head` weights (normal distribution).
*   **Generation (`generate`)**: Similar to the base `Actor`, using the model's `lm_head` for generation logic.
*   **Forward Pass (`forward`)**:
    *   **Description:** Performs a forward pass through the shared trunk and both heads.
    *   **Functionality:**
        1.  Passes inputs through the base model (`self.model.model`).
        2.  Gets logits from the `self.model.lm_head`.
        3.  Gets value predictions from the `self.value_head` applied to the hidden states.
        4.  Calculates action log-probabilities from the logits.
        5.  Extracts the value estimate corresponding to the last token (or based on action mask if critic logic were fully implemented here).
    *   **Returns:** A tuple containing action log-probabilities and value predictions (and optionally base model output).
*   **Other Methods:** Includes `process_sequences`, gradient checkpointing controls, and parameter printing.

### Use Case

These classes are primarily for research exploring alternative RLHF parameterizations: 
*   `ActorCustom`: Used for methods involving modulating a base policy's logits (e.g., related to 'twisting' or modification-based approaches).
*   `ActorCritic`: Used for PPO implementations that share parameters between the actor and critic networks to potentially improve efficiency or sample complexity.

## `ring_attn_utils.py`

### Purpose

This file provides utility functions specifically designed to support the integration and use of RingAttention (`ring_flash_attn` library) within the OpenRLHF framework, particularly when dealing with packed sequences.

### Global Variable

*   **`RING_ATTN_GROUP`**: A global variable intended to hold the PyTorch distributed process group used for RingAttention communication. Initialized to `None`.

### Key Functions

*   **`set_ring_attn_group(group)`**: Sets the global `RING_ATTN_GROUP` variable.
*   **`get_ring_attn_group()`**: Returns the current `RING_ATTN_GROUP`.

*   **`reset_ring_attn_position_ids(start, end, packed_seq_lens)`**:
    *   **Description:** Calculates the correct `position_ids` for a specific chunk (`start` to `end`) of packed sequences distributed across RingAttention ranks.
    *   **Functionality:** Similar to `models.utils.reset_position_ids`, but calculates positions only for the segment relevant to the current rank, considering the original lengths (`packed_seq_lens`) of the sequences packed together.

*   **`update_ring_attn_params(packed_seq_lens, total_seq_len)`**:
    *   **Description:** Prepares and updates the parameters required by the underlying `ring_flash_attn` library for a forward pass.
    *   **Functionality:**
        1.  Calculates the cumulative sequence lengths (`cu_seqlens`) based on `packed_seq_lens` and pads it with 0 at the beginning and `total_seq_len` at the end.
        2.  Imports `update_ring_flash_attn_params` from `ring_flash_attn`.
        3.  Calls `update_ring_flash_attn_params` with the calculated `cu_seqlens` and the global `RING_ATTN_GROUP`.

*   **`convert_ring_attn_params(sequences, attention_mask, packed_seq_lens, ring_attn_group)`**:
    *   **Description:** Adapts input tensors (`sequences`, `attention_mask`) and calculates `position_ids` for use with RingAttention.
    *   **Functionality:**
        1.  Determines the local chunk of the sequence (`start` to `end`) that the current rank is responsible for based on its rank in the `ring_attn_group`.
        2.  Slices the `sequences` and `attention_mask` to get the local chunk.
        3.  Calls `reset_ring_attn_position_ids` to get the correct position IDs for the local chunk.
        4.  Calls `update_ring_attn_params` to inform the RingAttention library about the sequence boundaries.
        5.  Returns the processed local `sequences`, `attention_mask`, and `position_ids`.

*   **`pad_sequences(...)`**:
    *   **Description:** Pads input sequences and related tensors/lists so their total length is divisible by the RingAttention world size.
    *   **Functionality:** Calculates the required padding length (`pad_len`). Appends padding tokens (`pad_token_id`) to `sequences`, padding values (using a new sequence ID) to `attention_mask`, and adjusts the last elements of `num_actions` and `packed_seq_lens` to account for the padding.

*   **`unpad_sequences(...)`**:
    *   **Description:** Removes the padding added by `pad_sequences` from the sequences and related tensors/lists after the RingAttention computation.
    *   **Functionality:** If `pad_len > 0`, it slices the tensors/lists to remove the last `pad_len` elements and adjusts the last elements of `num_actions` and `packed_seq_lens` back.

### Use Case

These utilities are crucial when using RingAttention (especially with packed sequences) to ensure that each rank processes the correct segment of the data, has the appropriate position IDs, and that the underlying `ring_flash_attn` library is correctly configured with sequence boundary information (`cu_seqlens`). The padding/unpadding functions handle the requirement that the total sequence length must be divisible by the ring size.

## `__init__.py`

### Purpose

This file serves as the entry point for the `openrlhf.models` package. It imports and exposes key classes and functions from other modules within the package, making them directly accessible under the `openrlhf.models` namespace.

### Exports

*   **From `actor.py`:**
    *   `Actor`: The primary class for policy models.
*   **From `loss.py`:**
    *   `DPOLoss`: Direct Preference Optimization loss.
    *   `GPTLMLoss`: Standard Causal Language Modeling loss.
    *   `KDLoss`: Knowledge Distillation loss.
    *   `KTOLoss`: Kahneman-Tversky Optimization loss.
    *   `LogExpLoss`: Log-Exponential pairwise ranking loss.
    *   `PairWiseLoss`: Sigmoid-based pairwise ranking loss.
    *   `PolicyLoss`: PPO clipped surrogate objective (actor loss).
    *   `PRMLoss`: Process Reward Model loss.
    *   `ValueLoss`: PPO critic loss.
    *   `VanillaKTOLoss`: Simpler KTO loss variant.
*   **From `model.py`:**
    *   `get_llm_for_sequence_regression`: Factory function for creating Reward and Critic models.

*   **Note:** Does not export `ActorCustom`, `ActorCritic`, or research-specific losses, keeping the main interface focused on standard components.

### Use Case

Allows users and other parts of the OpenRLHF codebase to import essential model and loss components directly, e.g., `from openrlhf.models import Actor, DPOLoss`.
