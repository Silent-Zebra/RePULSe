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
    *   **Functionality:** Allows users to define a behavior through various criteria, such as the presence of specific keywords/tokens in the output, adherence to a specific format, or evaluation by a custom Python function (`check_output_func`). This is crucial for identifying `B(o)=1`.

*   **`elicitation.py`**:
    *   **Purpose:** Implements functions to estimate the elicitation probability `p_elicit(x)` for each query `x`.
    *   **Functionality:**
        *   `logprob_specific()`: TO BE IMPLEMENTED. Measures the log probability of a specific sequence given the query.
        *   `logprob_keyword()`: TO BE IMPLEMENTED. Measures the log probability of a specific keyword appearing anywhere in the generated output.
        *   `repeated_sampling()`: Estimates `p_elicit` using Monte Carlo estimation by generating multiple outputs for each query and calculating the fraction that exhibit the defined harmful behavior.
        *   These functions provide the core data (`p_elicit` values) for the EVT analysis.

*   **`forecasting.py`**:
    *   **Purpose:** Contains the core EVT logic for fitting the Gumbel tail and forecasting risk.
    *   **Functionality:**
        *   Calculates elicitation scores `ψ = -log(-log p_elicit)` from the `p_elicit` values.
        *   `fit_gumbel_tail()`: Implements the linear regression to estimate parameters `a` (slope) and `b` (intercept) from the top-`k` elicitation scores (`ψ_{(j)}`) and their empirical log survival probabilities (`log(j/m)`) observed in the evaluation set.
        *   `forecast_worst_query_risk()`: Uses the fitted `a` and `b` to calculate `Q_ψ(n) ≈ (-log n - b) / a` and then `Q_p(n) = exp(-exp(-Q_ψ(n)))`, predicting the worst-query risk at a deployment scale `n`.

*   **`experiment_runner.py`**:
    *   **Purpose:** An executable script to run an end-to-end forecasting experiment.
    *   **Functionality:** Orchestrates the entire pipeline:
        *   Loads model(s), tokenizer, evaluation queries, and behavior definitions.
        *   Calls functions from `elicitation.py` to generate `p_elicit` values for all queries.
        *   Calls functions from `forecasting.py` to fit the Gumbel tail and generate `Q_p(n)` forecasts for various `n`.
        *   Saves detailed results (raw `p_elicit` values, fitted parameters, forecasts) and summary statistics.

*   **`analysis.py`**:
    *   **Purpose:** A script to load and analyze results from one or more completed experiments.
    *   **Functionality:**
        *   Loads saved experiment outputs.
        *   Generates comparative plots, such as `Q_p(n)` vs. `n` for different models or behaviors.
        *   Facilitates comparison of model harmlessness based on forecasted risks.