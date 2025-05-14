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
        *   `target_sequence: Optional[str]`: A specific target output sequence. Used by the 'logprob_target_sequence' elicitation method.
        *   `target_keywords: Optional[List[str]]`: A list of specific keywords. Used by the 'logprob_target_keyword_in_target_seq' elicitation method (which also uses `target_sequence` as the reference text) and by the 'repeated_sampling' method (when `type` is 'keyword') to check if an output exhibits the behavior.
    *   The file also pre-defines a list of `Behavior` objects (`BEHAVIORS`) and provides helper functions like `get_behavior_definition(behavior_id)`.

*   **`elicitation.py`**:
    *   **Purpose:** Implements functions to estimate the elicitation probability `p_elicit(x)` for each query `x`.
    *   **Functionality:**
        *   The core function is `estimate_p_elicit(..., method: str, ...)`, which takes a `method` string to select the estimation approach.
        *   Supported `method` names and their logic:
            *   `'logprob_target_sequence'`: Measures `log P(target_sequence | query)` where `target_sequence` is taken from `Behavior.target_sequence`. Uses a single forward pass.
            *   `'logprob_target_keyword_in_target_seq'`: Estimates the sum of probabilities of `Behavior.target_keywords` appearing within `Behavior.target_sequence` (acting as a fixed reference output), conditioned on the query. Uses a single forward pass.
            *   `'repeated_sampling'`: Estimates `p_elicit` using Monte Carlo by generating multiple outputs for each query and calculating the fraction that exhibit the behavior. Behavior checking is done by `get_behavior_check_function`.
        *   `get_behavior_check_function(behavior: Behavior)`: Returns a callable that checks if a given output string exhibits the defined behavior. For `behavior.type == 'keyword'`, it checks if any of `behavior.target_keywords` are present in the output.
        *   Helper functions include `generate_output` (for `repeated_sampling`) and `calculate_sequence_logprob` (for log-probability based methods).
        *   These functions provide the core data (`p_elicit` values) for the EVT analysis.

*   **`forecasting.py`**:
    *   **Purpose:** Contains the core EVT logic for fitting the Gumbel tail and forecasting risk.
    *   **Functionality:**
        *   `fit_gumbel_tail()`: Calculates elicitation scores `ψ = -log(-log p_elicit)` from the `p_elicit` values. Implements the linear regression to estimate parameters `a` (slope) and `b` (intercept) from the top-`k` elicitation scores (`ψ_{(j)}`) and their empirical log survival probabilities (`log(j/m)`) observed in the evaluation set.
        *   `forecast_worst_query_risk()`: Uses the fitted `a` and `b` to calculate `Q_ψ(n) ≈ (-log n - b) / a` and then `Q_p(n) = exp(-exp(-Q_ψ(n)))`, predicting the worst-query risk at a deployment scale `n`.

*   **`experiment_runner.py`**:
    *   **Purpose:** An executable script to run an end-to-end forecasting experiment.
    *   **Functionality:** Orchestrates the entire pipeline:
        *   Parses command-line arguments for model paths, behavior ID, query files, evaluation parameters, forecasting scales, etc.
        *   Loads model(s), tokenizer, evaluation queries (from file, with sampling), and behavior definitions (using `behavior_defs.get_behavior_definition`).
        *   Iteratively calls `elicitation.estimate_p_elicit` to generate `p_elicit` values for the sampled evaluation queries.
        *   Calls `forecasting.fit_gumbel_tail` to get Gumbel parameters (`a`, `b`, `r_squared`).
        *   Calls `forecasting.forecast_worst_query_risk` to generate `Q_p(n)` forecasts for various specified `n` values.
        *   Saves detailed results (individual `p_elicit` values per query in a JSONL file) and a summary JSON file (containing run arguments, fitted parameters, forecasts, and statistics).

*   **`analysis.py`**:
    *   **Purpose:** A script to load and analyze results from one or more completed experiments (summary JSON files).
    *   **Functionality:**
        *   `load_results()`: Loads and consolidates data from multiple `*_summary.json` files (output by `experiment_runner.py`) into a pandas DataFrame.
        *   `plot_forecast_comparison()`: Generates comparative plots, such as `Q_p(n)` vs. `n` (log-log scale) for different models or runs, grouped by `behavior_id`.
        *   The `main` function orchestrates loading results, saving a combined DataFrame to a CSV file, and generating plots for each unique behavior ID found in the results.
        *   Facilitates comparison of model harmlessness based on forecasted risks.

*   **`query_generation.py`**:
    *   **Purpose:** Uses a specified large language model (e.g., Mistral-7B-Instruct) to generate a pool of diverse queries for each behavior defined in `behavior_defs.py`. These queries are intended to be used as inputs for evaluating other LMs.
    *   **Functionality:**
        *   Uses a `META_PROMPT_TEMPLATE` to instruct an LLM to generate user-like prompts aimed at eliciting the `goal` specified in a `Behavior` object.
        *   `load_generation_model_and_tokenizer()`: Loads the LLM and tokenizer for query generation.
        *   `generate_raw_prompts_from_llm()`: Generates a batch of potential queries using the LLM and the meta-prompt.
        *   `generate_query_pool_for_behavior()`: Orchestrates query generation for a single behavior. It repeatedly calls the LLM and filters the generated queries. The filtering aims to remove queries that are too direct by checking for the presence of certain keywords (derived from `getattr(behavior, 'keywords', [])` which is not a standard `Behavior` field, and `behavior.target`).
        *   `main_generate_all_query_pools()`: Iterates through behaviors, generates a query pool for each, and saves them as JSONL files (e.g., `[behavior_id]_queries.jsonl`).
        *   Supports command-line arguments for output directory and number of queries per behavior.

*   **`query_pools.py`**:
    *   **Purpose:** A simple utility to list available query pool files.
    *   **Functionality:**
        *   `list_query_pools()`: Scans a specified directory for files matching `*_queries.jsonl` (typically generated by `query_generation.py`) and returns a list of their filenames.
        *   Includes a `main` block to print found pools.

*   **`separate_behaviors.py`**:
    *   **Purpose:** Classifies prompts from an input JSONL file based on keywords associated with behaviors defined in `behavior_defs.py`. It then separates these prompts into different output JSONL files, one for each matched behavior.
    *   **Functionality:**
        *   Loads behavior definitions (specifically `target_keywords`) from `behavior_defs.py`.
        *   Reads an input JSONL where each line contains a "prompt" (list of message dicts).
        *   Extracts and lowercases user content from prompts.
        *   For each prompt, it checks if any `target_keywords` (case-insensitive substring match) from any defined behavior are present in the user content.
        *   Appends the original JSONL line to a list corresponding to the first behavior matched.
        *   Writes prompts to `[behavior_id]_prompts.jsonl` files. Prompts not matching any behavior are saved to `unclassified_prompts.jsonl`.
        *   The script includes a `main` section with hardcoded input/output paths and logic to create dummy input data if the specified input file doesn't exist.