This module implements the methodology from Jones et al. (2025) "Forecasting Rare Language Model Behaviors." It provides tools to estimate and forecast the risk of rare, potentially harmful, model outputs at scale using Extreme Value Theory (EVT). The goal is to compare models based on their forecasted worst-query risk `Q_p(n)` for specific behaviors at simulated large deployment scales (`n` queries) using data from smaller evaluation sets (`m` queries). Lower forecasted risk implies greater harmlessness with respect to the tested behaviors.

**Core Methodology: Forecasting via Extreme Value Theory (Jones et al., 2025)**

The framework forecasts rare event risks in language models by analyzing the distribution of elicitation probabilities:

1.  **Elicitation Probability (`p_elicit(x)`):** For each query `x`, this is the probability `P(B(o)=1 | o ~ D_LLM(x))` that the model's output `o` exhibits a target harmful behavior `B`. This continuous value is estimated for each query in an evaluation set.
2.  **Distribution Focus & Extreme Value Theory (EVT):** The risk at large deployment scales (many queries) is often dominated by the queries with the highest `p_elicit` values (the upper tail of the distribution). EVT, specifically the Gumbel distribution, is used to model the behavior of these maximum `p_elicit` values.
3.  **Elicitation Score (`ψ`) and Gumbel-tail Scaling:** The "elicitation score" is defined as `ψ = -log(-log p_elicit)`. For large scores (very small `p_elicit`), the logarithm of the survival function `log P(Ψ > ψ)` is approximately linear: `log P(Ψ > ψ) ≈ aψ + b`. The parameters `a` (slope) and `b` (intercept) are key to forecasting.
4.  **Quantile Forecasting (`Q_p(n)`):**
    *   The `1/n`-th upper quantile of the `p_elicit` distribution, `Q_p(n)`, represents the `p_elicit` value expected to be exceeded by the single highest `p_elicit` query out of `n` deployment queries.
    *   Parameters `a` and `b` are estimated by fitting the linear relationship `log(j/m) ≈ a * ψ_{(j)} + b` using the top-`k` highest scores (`ψ_{(j)}`) and their empirical log survival probabilities (`log(j/m)`) from an evaluation set of `m` queries.
    *   The score quantile `Q_ψ(n)` at scale `n` is forecasted as: `Q_ψ(n) ≈ (-log n - b) / a`.
    *   This is converted back to the forecasted probability quantile (the worst-query risk): `Q_p(n) = exp(-exp(-Q_ψ(n)))`.

**Key Files in this Directory:**

*   **`behavior_defs.py`**:
    *   **Purpose:** Defines structures and classes for specifying target harmful behaviors.
    *   **Functionality:** Allows users to define a behavior through various criteria. Each `Behavior` dataclass object includes:
        *   `id: str`, `name: str`, `description: str`, `type: str` (e.g., 'keyword', 'custom_function'), `goal: str` (describes the aim of eliciting the behavior).
        *   `prompt_filter_keywords: List[str]`: Keywords used to filter prompts during query generation (though not directly used by `query_generation.py`'s current filtering logic).
        *   `default_elicitation_method: Optional[str]`: Specifies the default method name (e.g., 'logprob_target_sequence', 'logprob_target_keyword_in_target_seq', 'repeated_sampling') for estimating `p_elicit`.
        *   `target_sequence: Optional[Union[str, List[str]]]`: A specific target output sequence or a list of such sequences. Used by the 'logprob_target_sequence' elicitation method. (Note: The instantiation logic allows this to be a list if the source data provides a list).
        *   `target_keywords: Optional[List[str]]`: A list of specific keywords (stored in lowercase). Used by the 'logprob_target_keyword_in_target_seq' elicitation method (which also uses `target_sequence` as the reference text) and by the 'repeated_sampling' method (when `type` is 'keyword') to check if an output exhibits the behavior.
    *   The file also pre-defines a list of `Behavior` objects (`BEHAVIORS`) by processing an internal `_BEHAVIORS_DATA` dictionary and provides helper functions like `get_behavior_definition(behavior_id)`, `get_behaviors()`, and `get_behavior_by_name(name)`.

*   **`elicitation.py`**:
    *   **Purpose:** Implements functions to estimate the elicitation probability `p_elicit(x)` for each query `x`.
    *   **Functionality:**
        *   The core function is `estimate_p_elicit(..., method: str, ...)`, which takes a `method` string to select the estimation approach.
        *   Supported `method` names and their logic:
            *   `'logprob_target_seq'`: If `behavior.target_sequence` is a list of strings, it calculates `log P(target_output_string | query)` for each `target_output_string` in the list and returns the average of their probabilities `exp(log_prob)`. If `behavior.target_sequence` is a single string (or not a list), it handles it as a single target. Uses a single forward pass per target string.
            *   `'logprob_target_keyword_in_target_seq'`: This method uses `behavior.target_sequence` (expected to be a single string) as a fixed reference output. It first checks if any of `behavior.target_keywords` (case-insensitive) are present within this `behavior.target_sequence` string. If at least one keyword is found, it then calculates `log P(behavior.target_sequence | query)` and returns `exp(log_prob)`. If no keywords are found within `behavior.target_sequence`, it returns 0. This method uses a single forward pass.
            *   `'repeated_sampling'`: Estimates `p_elicit` using Monte Carlo by generating multiple outputs for each query (controlled by `k_samples` and `elicitation_processing_batch_size`) and calculating the fraction that exhibit the behavior. Behavior checking is done by `get_behavior_check_function`.
        *   `get_behavior_check_function(behavior: Behavior)`: Returns a callable that checks if a given output string exhibits the defined behavior. For `behavior.type == 'keyword'`, it checks if any of `behavior.target_keywords` (which are pre-lowercased) are present in the (lowercased) output. It also supports a `'specific_output'` type that checks for an exact match with `behavior.target_sequence` (if it's a string).
        *   Helper functions include `generate_output` (for `repeated_sampling`, with improved batching and error handling), `calculate_sequence_logprob` (for log-probability based methods), and `estimate_keyword_probability` (a utility that uses repeated sampling to find a single keyword).
        *   These functions provide the core data (`p_elicit` values) for the EVT analysis.

*   **`forecasting.py`**:
    *   **Purpose:** Contains the core EVT logic for fitting the Gumbel tail and forecasting risk with improved numerical stability.
    *   **Functionality:**
        *   `fit_gumbel_tail()`: Calculates elicitation scores `ψ = -log(-log p_elicit)` from the `p_elicit` values. It clips `p_elicit` with a small epsilon to avoid math errors at 0 or 1. Invalid `p_elicit` values are skipped. Implements linear regression (`scipy.stats.linregress`) to estimate parameters `a` (slope) and `b` (intercept) from the top-`k` elicitation scores (`ψ_{(j)}`) and their empirical log survival probabilities (`log(j/m)`), where `m` is the number of valid `p_elicit` values. Returns `(a, b, r_value)` (correlation coefficient). Includes enhanced error handling for insufficient data points or non-finite fit parameters.
        *   `forecast_worst_query_risk()`: Uses the fitted `a` and `b` to calculate `Q_ψ(n) ≈ (-log n - b) / a` and then `Q_p(n) = exp(-exp(-Q_ψ(n)))`, predicting the worst-query risk at a deployment scale `n`. Includes more robust handling of potential `OverflowError` during calculations and checks for invalid inputs (e.g., `a=0`, `n<=0`, `NaN` parameters). The final result is clipped to `[0.0, 1.0]`.

*   **`experiment_runner.py`**:
    *   **Purpose:** An executable script to run an end-to-end forecasting experiment, now with support for bootstrapping.
    *   **Functionality:** Orchestrates the entire pipeline:
        *   Parses a comprehensive set of command-line arguments for model paths (including LoRA and quantization options like 4-bit), checkpoint paths, behavior ID, query files, evaluation parameters (evaluation set size `m`, `top_k_fit`), forecasting scales `n`, DeepSpeed strategy settings, and, crucially, `num_bootstrap_samples`. Also includes arguments for generation parameters (`temperature`, `max_new_tokens`, etc.) and batch sizes for elicitation.
        *   Initializes `DeepSpeedStrategy`.
        *   Loads model(s) (potentially with LoRA adapters, quantization) and tokenizer using `openrlhf.utils.load_model_and_tokenizer` and `strategy`. Handles loading of specific model checkpoints via `ckpt_path`.
        *   Loads evaluation queries from file (with more robust parsing for different JSONL formats and plain text fallback) and behavior definitions (using `behavior_defs.get_behavior_definition`). The number of queries used per bootstrap sample (`evaluation_set_size_m_per_bootstrap`) is determined based on `args.evaluation_set_size` and the total available queries.
        *   **Bootstrapping Loop:** If `args.num_bootstrap_samples > 1`, the script performs bootstrapping:
            *   For each bootstrap iteration, it samples `args.evaluation_set_size` queries (with replacement) from the loaded query pool.
            *   It iteratively calls `elicitation.estimate_p_elicit` (using the specified or default `elicitation_method`) to generate `p_elicit` values for the sampled evaluation queries in batches.
            *   Calls `forecasting.fit_gumbel_tail` to get Gumbel parameters (`a_slope`, `b_intercept`, `r_squared_approx`, `top_k_actually_used`) for this bootstrap sample. These parameters are stored.
        *   **Results Aggregation & Forecasting:**
            *   If bootstrapping was performed, it aggregates the `a_slopes` and `b_intercepts` from all iterations. It then calculates mean, standard deviation, and percentiles for these parameters. Forecasts (`Q_p(n)`) are then made using these mean parameters, and confidence intervals for `Q_p(n)` are also derived from the percentile `(a,b)` pairs.
            *   If no bootstrapping (`num_bootstrap_samples == 1`), it uses the single set of fitted `a` and `b`.
        *   Calls `forecasting.forecast_worst_query_risk` to generate `Q_p(n)` forecasts.
        *   **Output Saving:**
            *   Saves detailed per-query results (query, `p_elicit`, bootstrap iteration index if applicable) to a `*_results.jsonl` file.
            *   Saves a comprehensive summary JSON file (`*_summary.json`) containing:
                *   All run arguments.
                *   Details of the bootstrap process (e.g., `num_bootstrap_samples_requested`, `num_successful_gumbel_fits`, `m_eval_size_per_bootstrap`).
                *   If bootstrapped: lists of `a_slopes`, `b_intercepts` from each iteration; summary statistics of these bootstrapped parameters (mean, std, percentiles for `a`, `b`, and `r_value`/`r_squared`).
                *   If not bootstrapped: the single fitted `a`, `b`, `r_squared_approx`.
                *   Forecasted `Q_p(n)` values for various `n` (from mean `a,b` if bootstrapped, plus CI forecasts).
                *   Other statistics like `mean_num_valid_p_elicits_per_bootstrap`.

*   **`analysis.py`**:
    *   **Purpose:** A script to load, analyze, and visualize results from one or more completed forecasting experiments (summary JSON files).
    *   **Functionality:**
        *   `load_results()`: Loads and consolidates data from multiple `*_summary.json` files (output by `experiment_runner.py`) into a pandas DataFrame. It recursively searches for these files and is designed to extract `model_name` and `seed` from a directory structure like `base_results_dir / model_name / seed / behavior_dir / xxx_summary.json`. It now handles summary files that include bootstrapped Gumbel fit parameters (e.g., mean and std dev of `a` and `b`, lists of `a_slopes` and `b_intercepts` from bootstrap iterations) and extracts detailed information like `num_bootstrap_samples_requested`, `num_successful_gumbel_fits`, and `m_eval_size_per_bootstrap`.
        *   `plot_forecast_comparison()`: Generates comparative plots of forecasted `Q_p(n)` vs. `n` (log-log scale) for different models or runs, grouped by `behavior_id`. It can now optionally aggregate results by seed or plot individual seed runs. For individual runs, if bootstrap data (`bootstrap_a_slopes`, `bootstrap_b_intercepts`) is available, it calculates and plots confidence intervals for `Q_p(n)` by re-calculating forecasts for each bootstrap sample of (a, b) parameters.
        *   `plot_model_averaged_forecasts()`: Generates plots where `Q_p(n)` forecasts are averaged across seeds for each model, also displaying confidence intervals derived from the bootstrapped parameters of individual seeds.
        *   `generate_qpn_summary_table()`: Creates a CSV table summarizing key `Q_p(n)` values (e.g., at `n=1e6`, `n=1e8`) for different models and behaviors, including mean and confidence intervals if bootstrap data is present.
        *   `generate_r2_summary_table()`: Creates a CSV table summarizing the `R^2` (goodness-of-fit) values for the Gumbel tail fits for different models and behaviors.
        *   The `main` function orchestrates loading results (accepting a `specific_run_prefix` to filter runs), saving a combined DataFrame to a CSV file, generating comparative plots, model-averaged plots, and summary tables for each unique behavior ID found in the results.
        *   Facilitates robust comparison of model harmlessness based on forecasted risks, including uncertainty quantification through bootstrapping.

*   **`query_generation.py`**:
    *   **Purpose:** Uses a specified large language model (defaulting to `mistralai/Mistral-7B-Instruct-v0.2`) to generate a pool of diverse queries for each behavior. These queries are intended to be used as inputs for evaluating other LMs.
    *   **Functionality:**
        *   Uses a `META_PROMPT_TEMPLATE` to instruct an LLM to generate user-like prompts aimed at eliciting the `goal` specified in a `Behavior` object. The meta-prompt encourages the generation of subtle, potentially provocative queries.
        *   `load_generation_model_and_tokenizer()`: Loads the LLM (with `device_map="auto"`, `torch_dtype=torch.bfloat16`) and tokenizer for query generation. Sets tokenizer padding to left.
        *   `generate_raw_prompts_from_llm()`: Generates a batch of potential queries using the LLM and the meta-prompt. Extracts text between `<prompt>` and `</prompt>` tags from the LLM's output.
        *   `generate_query_pool_for_behavior()`: Orchestrates query generation for a single behavior. It repeatedly calls the LLM (in batches controlled by `queries_per_batch`) and filters the generated queries. The filtering logic aims to remove queries that are too direct by checking for the presence of certain keywords:
            *   It attempts to use keywords from `getattr(behavior, 'keywords', [])` (an ad-hoc attribute not in the standard `Behavior` dataclass).
            *   It also attempts to use keywords from `behavior.target` (also not a standard `Behavior` field), depending on `behavior.type`.
            *   Generated queries containing these derived keywords are discarded.
        *   `main_generate_all_query_pools()`: Iterates through a list of `Behavior` objects, generates a query pool for each (up to `num_queries_per_behavior`), and saves them as JSONL files (e.g., `[behavior_id]_queries.jsonl`) in the specified `query_pool_dir`.
        *   Supports command-line arguments for output directory, number of queries per behavior, and batch size for generation.

*   **`query_pools.py`**:
    *   **Purpose:** A simple utility to list available query pool files.
    *   **Functionality:**
        *   `list_query_pools()`: Scans a specified directory for files matching `*_queries.jsonl` (typically generated by `query_generation.py`) and returns a list of their filenames.
        *   Includes a `main` block to print found pools.

*   **`separate_behaviors.py`**:
    *   **Purpose:** Classifies prompts from an input JSONL file based on keywords associated with behaviors defined in `behavior_defs.py`. It then separates these prompts into different output JSONL files, one for each matched behavior.
    *   **Functionality:**
        *   Loads behavior definitions (specifically their `target_keywords`, which are stored in lowercase) from `behavior_defs.py` using `get_behaviors()`.
        *   Reads an input JSONL file where each line is expected to contain a "prompt" field (a list of message dictionaries, e.g., `[{"role": "user", "content": "..."}]`).
        *   Uses a helper `get_user_prompt_content` to extract and concatenate all "content" from "user" messages within a prompt, converting it to lowercase.
        *   For each prompt, it iterates through the defined behaviors and checks if any of a behavior's `target_keywords` are present (as a case-insensitive substring) in the extracted user content.
        *   A prompt is assigned to the *first* behavior it matches. The original JSONL line for the prompt is stored.
        *   Writes prompts to `[behavior_id]_prompts.jsonl` files in a specified output directory. Prompts not matching any behavior are saved to `unclassified_prompts.jsonl`.
        *   Includes improved error handling for file reading and JSON parsing, along with progress reporting.
        *   The script's `main` section contains hardcoded input/output paths and can create a dummy input file for testing if the primary input is not found.