import numpy as np
from scipy.stats import linregress
from typing import List, Tuple

# Extremely small constant to prevent log(0) or log(1) issues
_LOG_EPSILON = 1e-300

def fit_gumbel_tail(p_elicits: List[float], top_k: int = 10) -> Tuple[float, float, float, List[float], List[float], int, List[float], List[float]]:
    """Fits the Gumbel tail linear model (log P(>psi) ~ a*psi + b)
       based on the top_k elicitation scores derived from p_elicits.
       Returns fit parameters, r_value, the x/y data points used for the fit,
       the top 10 scores, and their corresponding log survival probabilities.

    Args:
        p_elicits: List of elicitation probabilities from the evaluation set (size m).
                   Values should be in [0, 1].
        top_k: Number of top quantiles (highest scores) to use for fitting.
               Must be at least 2.

    Returns:
        Tuple (a, b, r_value, top_scores_fitted, log_survival_probs_fitted, actual_top_k_used, top_10_scores, log_survival_probs_top_10) of
        the linear fit parameters (slope, intercept), the correlation coefficient (r_value),
        the top scores used for fitting (x-values), the corresponding log survival probabilities (y-values),
        the actual number of points used for fitting, the top 10 scores (or fewer if not available),
        and the log survival probabilities for these top 10 scores.
        Returns (nan, nan, nan, [], [], 0, [], []) on failure.
    """
    if not isinstance(p_elicits, list) or not p_elicits:
        print("Warning: p_elicits must be a non-empty list.")
        return np.nan, np.nan, np.nan, [], [], 0, [], []

    if top_k < 2:
        print("Warning: top_k must be at least 2 for linear regression.")
        return np.nan, np.nan, np.nan, [], [], 0, [], []

    # Calculate scores: psi = -log(-log p)
    scores = []
    valid_p_count = 0
    for p in p_elicits:
        if p is None or not (0 <= p <= 1):
            print(f"Warning: Skipping invalid p_elicit value {p}.")
            continue

        # Clip p to avoid infinities at exactly 0 or 1
        p_clipped = np.clip(p, _LOG_EPSILON, 1.0 - _LOG_EPSILON)

        try:
            score = -np.log(-np.log(p_clipped))
            scores.append(score)
            valid_p_count += 1
        except ValueError as e:
            print(f"Warning: Skipping p_elicit value {p} due to math error during score calculation: {e}")

    if valid_p_count < 2: # Still need at least 2 for the primary regression, even if top_10 related logic could handle fewer.
        print(f"Warning: Only {valid_p_count} valid scores available after processing. Cannot fit line (minimum 2 required).")
        return np.nan, np.nan, np.nan, [], [], 0, [], []

    m = valid_p_count # Effective number of samples is count of valid scores
    scores.sort(reverse=True)

    # Determine top 10 scores and their log survival probabilities
    num_top_10_to_consider = min(10, len(scores))
    top_10_scores = scores[:num_top_10_to_consider]
    log_survival_probs_top_10 = []
    if num_top_10_to_consider > 0:
        ranks_top_10 = np.arange(1, num_top_10_to_consider + 1)
        log_survival_probs_top_10 = np.log(ranks_top_10 / m).tolist()


    # Regression for Gumbel fit (using actual_top_k)
    actual_top_k_for_regression = min(top_k, len(scores))
    if actual_top_k_for_regression < 2:
        print(f"Warning: Only {actual_top_k_for_regression} point(s) available for regression after selecting top_k={top_k}. Minimum 2 required.")
        # Still return top_10 data if available, even if regression fails
        return np.nan, np.nan, np.nan, [], [], 0, top_10_scores, log_survival_probs_top_10

    top_scores_for_regression = scores[:actual_top_k_for_regression]
    ranks_for_regression = np.arange(1, actual_top_k_for_regression + 1)
    log_survival_probs_for_regression = np.log(ranks_for_regression / m)

    try:
        slope, intercept, r_value, p_value, std_err = linregress(top_scores_for_regression, log_survival_probs_for_regression)

        if not np.isfinite(slope) or not np.isfinite(intercept):
            print("Warning: Linear regression resulted in non-finite parameters (slope or intercept).")
            # Still return top_10 data if available
            return np.nan, np.nan, np.nan, [], [], 0, top_10_scores, log_survival_probs_top_10

        # print(f"Fit Gumbel Tail: a={slope:.4f}, b={intercept:.4f}, R^2={r_value**2:.4f} using top {actual_top_k_for_regression}/{m} points.")
        return slope, intercept, r_value, top_scores_for_regression, log_survival_probs_for_regression.tolist(), actual_top_k_for_regression, top_10_scores, log_survival_probs_top_10
    except ValueError as e:
        print(f"Warning: Linear regression failed. Error: {e}")
        print(f"  Scores (top {actual_top_k_for_regression}): {top_scores_for_regression}")
        print(f"  Log Survival Probs for regression: {log_survival_probs_for_regression}")
        # Still return top_10 data if available
        return np.nan, np.nan, np.nan, [], [], 0, top_10_scores, log_survival_probs_top_10


def forecast_worst_query_risk(a: float, b: float, n: int) -> float:
    """Forecasts the worst-query risk Q_p(n) using fitted Gumbel parameters a, b.

    This is the 1/n upper quantile of the p_elicit distribution.

    Args:
        a: Fitted slope parameter from fit_gumbel_tail.
        b: Fitted intercept parameter from fit_gumbel_tail.
        n: Target deployment scale (number of queries). Must be > 0.

    Returns:
        Forecasted Q_p(n) probability (float between 0 and 1), or nan if forecast
        is not possible (e.g., invalid inputs, numerical issues).
    """
    if np.isnan(a) or np.isnan(b):
        print("Warning: Cannot forecast Q_p(n) due to invalid fit parameters (a or b is NaN).")
        return np.nan
    if a == 0:
        print("Warning: Cannot forecast Q_p(n) because the fitted slope (a) is zero.")
        return np.nan
    if n <= 0:
        print("Warning: Deployment scale n must be positive.")
        return np.nan

    # Forecast score quantile: Q_psi(n) ≈ (-log n - b) / a
    # Use log(1/n) = -log(n)
    try:
        log_n = np.log(n)
        numerator = -log_n - b
        q_psi_n = numerator / a

    except OverflowError:
        # This might happen if n is extremely large, making -log(n) very negative.
        # If numerator overflows negatively, and a is negative -> q_psi_n -> +inf
        # If numerator overflows negatively, and a is positive -> q_psi_n -> -inf
        if a > 0: q_psi_n = -np.inf
        else:     q_psi_n = np.inf
        print(f"Warning: Potential overflow calculating Q_psi(n) numerator for n={n}. Estimated Q_psi(n) as {q_psi_n}.")

    # Convert score quantile back to probability quantile: Q_p(n) = exp(-exp(-Q_psi(n)))
    try:
        # Calculate the inner exponent term: -Q_psi(n)
        neg_q_psi_n = -q_psi_n

        # Calculate the middle exponentiation: exp(-Q_psi(n))
        # Handle potential overflow if -q_psi_n is large positive
        if neg_q_psi_n > 700: # exp(700) is roughly 1e304, close to float limit
            exp_term = float('inf')
        else:
            exp_term = np.exp(neg_q_psi_n)

        # Calculate the outer exponentiation: exp(-exp_term)
        # Handle potential overflow if -exp_term is large positive (never happens as exp_term >= 0)
        # Handle potential underflow if -exp_term is large negative (exp_term -> inf)
        if exp_term == float('inf'):
            q_p_n = 0.0 # exp(-inf) -> 0
        else:
            q_p_n = np.exp(-exp_term)

    except OverflowError:
        q_p_n = 0.0
        print(f"Warning: Overflow converting Q_psi(n)={q_psi_n} back to Q_p(n). Result capped at 0.")
    except ValueError as e:
        print(f"Warning: Math error converting Q_psi(n)={q_psi_n} back to Q_p(n): {e}. Returning NaN.")
        return np.nan

    if np.isnan(q_p_n):
        print(f"Warning: Forecasted Q_p(n) resulted in NaN. Check inputs and fit quality (a={a}, b={b}).")
        return np.nan

    return np.clip(q_p_n, 0.0, 1.0) 