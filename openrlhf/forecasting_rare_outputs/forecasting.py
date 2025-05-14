import numpy as np
from scipy.stats import linregress
import math
from typing import List, Tuple

# Small constant to prevent log(0) or log(1) issues
_LOG_EPSILON = 1e-200

def fit_gumbel_tail(p_elicits: List[float], top_k: int = 10) -> Tuple[float, float, float]:
    """Fits the Gumbel tail linear model (log P(>psi) ~ a*psi + b)
       based on the top_k elicitation scores derived from p_elicits.

    Args:
        p_elicits: List of elicitation probabilities from the evaluation set (size m).
                   Values should be in [0, 1].
        top_k: Number of top quantiles (highest scores) to use for fitting.
               Must be at least 2.

    Returns:
        Tuple (a, b, r_value) of the linear fit parameters (slope, intercept) and
        the correlation coefficient (r_value). Returns (nan, nan, nan) on failure
        (e.g., insufficient valid data points).
    """
    if not isinstance(p_elicits, list) or not p_elicits:
        print("Warning: p_elicits must be a non-empty list.")
        return np.nan, np.nan, np.nan

    if top_k < 2:
        print("Warning: top_k must be at least 2 for linear regression.")
        return np.nan, np.nan, np.nan

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
            # If p_clipped is very close to 1, -log(p_clipped) is near 0.
            # -log(-log(p_clipped)) can then become -log(small_positive) -> large positive.
            # If p_clipped is very close to 0, -log(p_clipped) is large positive.
            # -log(-log(p_clipped)) would involve log(negative) -> error if not clipped.
            # The clipping handles p=0. For p=1, -log(1-_LOG_EPSILON) is ~ -log(_LOG_EPSILON) which is large positive.
            # Then -log(-log(p_clipped)) = -log(large_positive), which is negative.
            score = -math.log(-math.log(p_clipped))
            scores.append(score)
            valid_p_count += 1
        except ValueError as e:
            # This should theoretically not happen with clipping, but catch defensively.
            print(f"Warning: Skipping p_elicit value {p} due to math error during score calculation: {e}")

    if valid_p_count < 2:
        print(f"Warning: Only {valid_p_count} valid scores available after processing. Cannot fit line (minimum 2 required).")
        return np.nan, np.nan, np.nan

    m = valid_p_count # Effective number of samples is count of valid scores

    scores.sort(reverse=True) # Sort scores descending: psi_(1), psi_(2), ...

    # Determine actual number of points to use for regression
    actual_top_k = min(top_k, len(scores))
    if actual_top_k < 2:
        print(f"Warning: Only {actual_top_k} point(s) available for regression after selecting top_k={top_k}. Minimum 2 required.")
        return np.nan, np.nan, np.nan

    top_scores = scores[:actual_top_k]
    # Use ranks j=1 to actual_top_k for survival probability P(>psi) ~ j/m
    ranks = np.arange(1, actual_top_k + 1)

    # Calculate empirical log survival probabilities: log(j/m) for j=1 to actual_top_k
    # Add epsilon to avoid log(0) if m is very small and j=1? No, m >= 2 here.
    log_survival_probs = np.log(ranks / m)

    # Perform linear regression: log_survival_prob ~ a * score + b
    try:
        # linregress expects x then y: scores (psi) are x, log_survival_probs are y
        slope, intercept, r_value, p_value, std_err = linregress(top_scores, log_survival_probs)

        # Check for non-finite results which can happen in edge cases
        if not np.isfinite(slope) or not np.isfinite(intercept):
            print("Warning: Linear regression resulted in non-finite parameters (slope or intercept).")
            return np.nan, np.nan, np.nan

        print(f"Fit Gumbel Tail: a={slope:.4f}, b={intercept:.4f}, R^2={r_value**2:.4f} using top {actual_top_k}/{m} points.")
        return slope, intercept, r_value # a, b, r
    except ValueError as e:
        # Might happen if inputs to linregress are somehow invalid (e.g., all scores identical)
        print(f"Warning: Linear regression failed. Error: {e}")
        print(f"  Scores (top {actual_top_k}): {top_scores}")
        print(f"  Log Survival Probs: {log_survival_probs}")
        return np.nan, np.nan, np.nan


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
        # If slope is zero, the model implies survival probability is constant,
        # which doesn't allow forecasting based on scale n in this way.
        print("Warning: Cannot forecast Q_p(n) because the fitted slope (a) is zero.")
        return np.nan
    if n <= 0:
        print("Warning: Deployment scale n must be positive.")
        return np.nan

    # Forecast score quantile: Q_psi(n) ≈ (-log n - b) / a
    # Use log(1/n) = -log(n)
    try:
        # Use np.log for better handling of large n if needed, though math.log is fine
        log_n = math.log(n)
        # Ensure numerical stability if -log(n) - b is extremely large or small
        numerator = -log_n - b
        q_psi_n = numerator / a

    except OverflowError:
        # This might happen if n is extremely large, making -log(n) very negative.
        # If numerator overflows negatively, and a is negative -> q_psi_n -> +inf
        # If numerator overflows negatively, and a is positive -> q_psi_n -> -inf
        # We'll handle the resulting q_psi_n in the next step.
        # Let's estimate based on signs for large magnitude numerator:
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
            exp_term = math.exp(neg_q_psi_n)

        # Calculate the outer exponentiation: exp(-exp_term)
        # Handle potential overflow if -exp_term is large positive (never happens as exp_term >= 0)
        # Handle potential underflow if -exp_term is large negative (exp_term -> inf)
        if exp_term == float('inf'):
            q_p_n = 0.0 # exp(-inf) -> 0
        else:
            q_p_n = math.exp(-exp_term)

    except OverflowError:
        # This is highly unlikely given the check above, but as a safeguard.
        # If exp(-q_psi_n) somehow overflows, it implies -q_psi_n is large positive,
        # meaning q_psi_n is large negative. exp(-exp(-q_psi_n)) -> exp(-inf) -> 0.
        q_p_n = 0.0
        print(f"Warning: Overflow converting Q_psi(n)={q_psi_n} back to Q_p(n). Result capped at 0.")
    except ValueError as e:
        # e.g., math domain error if inputs are unexpected NaNs slipped through checks
        print(f"Warning: Math error converting Q_psi(n)={q_psi_n} back to Q_p(n): {e}. Returning NaN.")
        return np.nan

    # Final sanity check and clamp the result to [0, 1]
    if np.isnan(q_p_n):
        print(f"Warning: Forecasted Q_p(n) resulted in NaN. Check inputs and fit quality (a={a}, b={b}).")
        return np.nan

    return np.clip(q_p_n, 0.0, 1.0) 