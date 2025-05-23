#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
import pandas as pd

# Import the fit_gumbel_tail and forecast_worst_query_risk functions
from openrlhf.forecasting_rare_outputs.forecasting import fit_gumbel_tail, forecast_worst_query_risk

def load_visualization_data(file_path):
    """Load visualization data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_fit_statistics(x_values, y_values, slope, intercept):
    """
    Calculate comprehensive goodness-of-fit statistics for a linear regression.
    
    Args:
        x_values: List of x values (predictor variable).
        y_values: List of y values (response variable).
        slope: The fitted slope parameter.
        intercept: The fitted intercept parameter.
    
    Returns:
        Dictionary containing various fit statistics.
    """
    if not x_values or not y_values or len(x_values) != len(y_values):
        return {
            "error": "Invalid input data for fit statistics calculation"
        }
    
    n = len(x_values)
    if n < 3:  # Need at least 3 points for meaningful statistics
        return {
            "error": f"Insufficient data points ({n}) for fit statistics calculation. Need at least 3."
        }
    
    # Convert to numpy arrays
    x = np.array(x_values)
    y = np.array(y_values)
    
    # Predicted values
    y_pred = slope * x + intercept
    
    # Residuals
    residuals = y - y_pred
    
    # Sum of squares
    ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
    ss_residual = np.sum(residuals**2)       # Residual sum of squares
    ss_regression = ss_total - ss_residual   # Regression sum of squares
    
    # Degrees of freedom
    df_total = n - 1
    df_regression = 1
    df_residual = n - 2
    
    # Mean squares
    ms_regression = ss_regression / df_regression
    ms_residual = ss_residual / df_residual
    
    # F-statistic
    f_statistic = ms_regression / ms_residual
    f_p_value = 1 - stats.f.cdf(f_statistic, df_regression, df_residual)
    
    # R-squared and adjusted R-squared
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / df_residual) if df_residual > 0 else 0
    
    # Standard errors of coefficients
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean)**2)
    
    se_slope = np.sqrt(ms_residual / x_var) if x_var != 0 else np.nan
    se_intercept = np.sqrt(ms_residual * (1/n + x_mean**2/x_var)) if x_var != 0 else np.nan
    
    # t-statistics and p-values for coefficients
    t_slope = slope / se_slope if se_slope != 0 else np.nan
    t_intercept = intercept / se_intercept if se_intercept != 0 else np.nan
    
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), df_residual)) if not np.isnan(t_slope) else np.nan
    p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df_residual)) if not np.isnan(t_intercept) else np.nan
    
    # Confidence intervals for coefficients (95%)
    t_critical = stats.t.ppf(0.975, df_residual)
    
    ci_slope_lower = slope - t_critical * se_slope if not np.isnan(se_slope) else np.nan
    ci_slope_upper = slope + t_critical * se_slope if not np.isnan(se_slope) else np.nan
    
    ci_intercept_lower = intercept - t_critical * se_intercept if not np.isnan(se_intercept) else np.nan
    ci_intercept_upper = intercept + t_critical * se_intercept if not np.isnan(se_intercept) else np.nan
    
    # Root mean squared error
    rmse = np.sqrt(ms_residual)
    
    # Mean absolute error
    mae = np.mean(np.abs(residuals))
    
    # Durbin-Watson statistic (test for autocorrelation)
    residual_diff_sq = np.sum(np.diff(residuals)**2)
    durbin_watson = residual_diff_sq / ss_residual if ss_residual != 0 else np.nan
    
    # Results dictionary
    fit_stats = {
        "n_points": n,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted_r_squared),
        "f_statistic": float(f_statistic),
        "f_p_value": float(f_p_value),
        "std_error_slope": float(se_slope),
        "std_error_intercept": float(se_intercept),
        "t_statistic_slope": float(t_slope),
        "t_statistic_intercept": float(t_intercept),
        "p_value_slope": float(p_slope),
        "p_value_intercept": float(p_intercept),
        "ci_95_slope": [float(ci_slope_lower), float(ci_slope_upper)],
        "ci_95_intercept": [float(ci_intercept_lower), float(ci_intercept_upper)],
        "rmse": float(rmse),
        "mae": float(mae),
        "durbin_watson": float(durbin_watson),
        "residuals": residuals.tolist(),
        "predicted_values": y_pred.tolist(),
        "sum_squares_total": float(ss_total),
        "sum_squares_regression": float(ss_regression),
        "sum_squares_residual": float(ss_residual)
    }
    
    return fit_stats

def plot_residual_analysis(fig, fit_stats, title_prefix=""):
    """
    Create a 2x2 grid of residual analysis plots:
    1. Residuals vs Fitted Values
    2. Residuals Histogram
    3. Normal Q-Q Plot of Residuals
    4. Residuals vs Order
    
    Args:
        fig: The matplotlib figure to plot on.
        fit_stats: Dictionary of fit statistics from calculate_fit_statistics.
        title_prefix: Optional prefix for plot titles.
    
    Returns:
        The matplotlib figure with the plots.
    """
    if "error" in fit_stats:
        fig.text(0.5, 0.5, f"Error: {fit_stats['error']}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        return fig
    
    residuals = np.array(fit_stats["residuals"])
    predicted = np.array(fit_stats["predicted_values"])
    n = len(residuals)
    
    # 1. Residuals vs Fitted Values
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(predicted, residuals, color='blue', alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{title_prefix}Residuals vs Fitted Values')
    ax1.grid(True, alpha=0.3)
    
    # Add a loess/lowess smoothing line if scipy has it and we have enough points
    try:
        if n >= 10:
            from scipy.stats import loess
            smoothed = loess(predicted, residuals)
            ax1.plot(smoothed.x, smoothed.y, 'r-')
    except (ImportError, AttributeError):
        print("Warning: loess/lowess smoothing line not available.")
    
    # 2. Residuals Histogram
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(residuals, bins=min(10, max(5, n//5)), alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{title_prefix}Histogram of Residuals')
    ax2.grid(True, alpha=0.3)
    
    # Add a normal density curve for comparison
    if n >= 5:
        x = np.linspace(min(residuals), max(residuals), 100)
        mean = np.mean(residuals)
        std = np.std(residuals)
        if std > 0:
            pdf = stats.norm.pdf(x, mean, std)
            # Scale the PDF to match histogram height
            hist_max = np.histogram(residuals, bins=min(10, max(5, n//5)))[0].max()
            pdf_scaled = pdf * (hist_max / pdf.max())
            ax2.plot(x, pdf_scaled, 'r-', linewidth=1)
    
    # 3. Normal Q-Q Plot of Residuals
    ax3 = fig.add_subplot(2, 2, 3)
    stats.probplot(residuals, plot=ax3)
    ax3.set_title(f'{title_prefix}Normal Q-Q Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals vs Order (observation sequence)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(np.arange(1, n+1), residuals, color='blue', alpha=0.6)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Observation Order')
    ax4.set_ylabel('Residuals')
    ax4.set_title(f'{title_prefix}Residuals vs Order')
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig

def plot_linear_regression(ax, psi_values, log_survival_probs, fit_a, fit_b, r_squared, fitted_line_x, fitted_line_y, title=None, k_fit_val=None):
    """Create a scatter plot of log(S(ψ)) vs ψ with the fitted line."""
    
    # Create scatter plot
    ax.scatter(psi_values, log_survival_probs, color='blue', marker='o', label='Data points')
    
    # Plot the fitted line
    ax.plot(fitted_line_x, fitted_line_y, 'r-', label=f'Fitted line: y = {fit_a:.4f}x + {fit_b:.4f}')
        
    # Add labels and title
    ax.set_xlabel('ψ (Elicitation Score)')
    ax.set_ylabel('log S(ψ) = log(rank/m)')
    
    base_title = 'Gumbel Tail Linear Regression'
    if k_fit_val is not None:
        base_title += f' (Refitted with k={k_fit_val})'
    
    if title:
        ax.set_title(f'{title}\n{base_title} | R² = {r_squared:.4f}')
    else:
        ax.set_title(f'{base_title}\nR² = {r_squared:.4f}')
    
    # Add a grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    return ax

def plot_qq(ax, data, title=None):
    """Create a Quantile-Quantile plot comparing empirical quantiles to theoretical Gumbel quantiles."""
    # Extract data
    psi_values = data["all_valid_scores"]["psi_values"]
    gumbel_quantiles = data["all_valid_scores"]["gumbel_quantiles"]
    
    if not psi_values or not gumbel_quantiles:
        ax.text(0.5, 0.5, "Insufficient data for QQ plot", 
                horizontalalignment='center', verticalalignment='center')
        return ax
    
    # Create QQ plot
    ax.scatter(gumbel_quantiles, psi_values, color='blue', marker='o', alpha=0.6)
    
    # Add reference line
    min_val = min(min(psi_values), min(gumbel_quantiles))
    max_val = max(max(psi_values), max(gumbel_quantiles))
    reference_line = np.linspace(min_val, max_val, 100)
    ax.plot(reference_line, reference_line, 'r--', label='Reference Line')
    
    # Add labels and title
    ax.set_xlabel('Theoretical Gumbel Quantiles')
    ax.set_ylabel('Empirical Quantiles (ψ)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Quantile-Quantile Plot')
    
    # Add a grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    return ax

def plot_probability(ax, data, title=None):
    """Create a probability plot comparing empirical CDF to theoretical Gumbel CDF."""
    # Extract data
    psi_values = data["all_valid_scores"]["psi_values"]
    empirical_quantiles = data["all_valid_scores"]["empirical_quantiles"]
    
    if not psi_values or not empirical_quantiles:
        ax.text(0.5, 0.5, "Insufficient data for probability plot", 
                horizontalalignment='center', verticalalignment='center')
        return ax
    
    # Create probability plot
    ax.scatter(psi_values, empirical_quantiles, color='blue', marker='o', alpha=0.6)
    
    # Calculate theoretical Gumbel CDF for the range of psi values
    if psi_values:
        x_range = np.linspace(min(psi_values), max(psi_values), 100)
        # Use standard Gumbel distribution parameters
        theoretical_cdf = stats.gumbel_r.cdf(x_range)
        ax.plot(x_range, theoretical_cdf, 'r-', label='Theoretical Gumbel CDF')
    
    # Add labels and title
    ax.set_xlabel('ψ (Elicitation Score)')
    ax.set_ylabel('Cumulative Probability')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Probability Plot')
    
    # Add a grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    return ax

def plot_p_elicit_histogram(ax, data, title=None):
    """Create a histogram of p_elicit values."""
    # Extract data
    p_elicit_values = data["all_valid_scores"]["p_elicit_values"]
    
    if not p_elicit_values:
        ax.text(0.5, 0.5, "Insufficient data for p_elicit histogram", 
                horizontalalignment='center', verticalalignment='center')
        return ax
    
    # Create histogram
    ax.hist(p_elicit_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add labels and title
    ax.set_xlabel('P_ELICIT')
    ax.set_ylabel('Frequency')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Distribution of Elicitation Probabilities')
    
    # Add a grid for better readability
    ax.grid(True, alpha=0.3)
    
    return ax

def perform_bootstrap_refitting(bootstrap_data, k_fit):
    """
    Refits Gumbel parameters (a, b, r) for each bootstrap sample using the specified k_fit.

    Args:
        bootstrap_data: Dictionary containing bootstrap iteration data, specifically 
                        'top_10_scores' (list of lists of psi values) and 
                        'top_10_log_survival_probs' (list of lists of log survival probabilities).
        k_fit: The number of top points (k_fit <= 10) to use for refitting.

    Returns:
        Tuple: (mean_a, mean_b, mean_r_squared, num_successful_fits)
               Returns (np.nan, np.nan, np.nan, 0) if refitting is not possible.
    """
    if not bootstrap_data or 'top_10_scores' not in bootstrap_data or 'top_10_log_survival_probs' not in bootstrap_data:
        print("Warning: Bootstrap data for refitting is missing or incomplete.")
        return np.nan, np.nan, np.nan, 0

    boot_psis_all_iters = bootstrap_data['top_10_scores']
    boot_log_surv_probs_all_iters = bootstrap_data['top_10_log_survival_probs']

    if not isinstance(boot_psis_all_iters, list) or not isinstance(boot_log_surv_probs_all_iters, list) or len(boot_psis_all_iters) != len(boot_log_surv_probs_all_iters):
        print("Warning: Bootstrap data lists are malformed or have mismatched lengths.")
        return np.nan, np.nan, np.nan, 0

    refit_a_slopes = []
    refit_b_intercepts = []
    refit_r_values = []

    for i in range(len(boot_psis_all_iters)):
        psi_scores_iter = boot_psis_all_iters[i]
        log_surv_probs_iter = boot_log_surv_probs_all_iters[i]

        if not (isinstance(psi_scores_iter, list) and 
                isinstance(log_surv_probs_iter, list) and 
                len(psi_scores_iter) == len(log_surv_probs_iter)):
            # print(f"Skipping bootstrap iteration {i+1} due to malformed data.")
            continue

        # Data is assumed to be sorted: highest score (psi) first.
        # Select top k_fit points. k_fit cannot exceed available points (max 10 here)
        actual_k = min(k_fit, len(psi_scores_iter))

        if actual_k < 2: # Need at least 2 points for linear regression
            # print(f"Skipping bootstrap iteration {i+1}: Not enough points ({actual_k}) for k_fit={k_fit}.")
            continue

        selected_psi = psi_scores_iter[:actual_k]
        selected_log_surv = log_surv_probs_iter[:actual_k]

        # Filter out NaN/inf values
        valid_indices = [idx for idx, (p_val, l_val) in enumerate(zip(selected_psi, selected_log_surv)) 
                         if pd.notna(p_val) and pd.notna(l_val) and np.isfinite(p_val) and np.isfinite(l_val)]
        
        final_selected_psi = [selected_psi[j] for j in valid_indices]
        final_selected_log_surv = [selected_log_surv[j] for j in valid_indices]

        if len(final_selected_psi) < 2:
            # print(f"Skipping bootstrap iteration {i+1} after NaN/inf filter: Not enough valid points.")
            continue

        try:
            slope, intercept, r_value, _, _ = stats.linregress(final_selected_psi, final_selected_log_surv)
            if np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(r_value):
                refit_a_slopes.append(slope)
                refit_b_intercepts.append(intercept)
                refit_r_values.append(r_value)
            # else:
                # print(f"Skipping bootstrap iteration {i+1} due to non-finite linregress results.")
        except ValueError as e:
            # print(f"Linregress error for bootstrap iteration {i+1}: {e}")
            continue
    
    num_successful_fits = len(refit_a_slopes)
    if num_successful_fits > 0:
        mean_a = np.mean(refit_a_slopes)
        mean_b = np.mean(refit_b_intercepts)
        # R-squared is r_value squared. Calculate mean of r_values then square, or square then mean?
        # For consistency with how r_squared is typically reported from a single fit, 
        # and to match original data's `fit_r` which is then squared, let's average r_values then square.
        # However, if we want average of R^2 values, it's np.mean([r**2 for r in refit_r_values])
        # Let's use mean of R-squared values directly for the mean R-squared.
        mean_r_squared = np.mean([r**2 for r in refit_r_values])
        return mean_a, mean_b, mean_r_squared, num_successful_fits
    else:
        return np.nan, np.nan, np.nan, 0

def plot_all_visualizations(data, output_dir, base_filename, k_fit=None):
    """Generate all visualization plots and save them to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine original fit parameters and data points for regression plot
    original_psi_values = data["linear_regression_fit"]["top_scores"]
    original_log_survival_probs = data["linear_regression_fit"]["log_survival_probs"]
    original_fit_a = data["linear_regression_fit"]["fit_a"]
    original_fit_b = data["linear_regression_fit"]["fit_b"]
    original_r_value = data["linear_regression_fit"]["fit_r"]
    original_r_squared = original_r_value**2
    original_fitted_line_x = data["linear_regression_fit"]["fitted_line"]["x"]
    original_fitted_line_y = data["linear_regression_fit"]["fitted_line"]["y"]

    # --- Refitting Logic ---
    current_fit_a = original_fit_a
    current_fit_b = original_fit_b
    current_r_squared = original_r_squared
    # For the linear regression plot, we use the original points that the initial fit was based on if not refitting.
    # If refitting, we'd ideally plot the points used for refitting, but for now, let's be explicit.
    # The `calculate_fit_statistics` will use original_psi_values and original_log_survival_probs for residuals against the chosen line.
    points_for_plot_psi = original_psi_values
    points_for_plot_log_surv = original_log_survival_probs
    line_x_for_plot = original_fitted_line_x
    line_y_for_plot = original_fitted_line_y
    k_fit_display_val = None # For plot title

    if k_fit is not None:
        if "bootstrap_iterations_data" in data:
            bootstrap_refit_data = data.get("bootstrap_iterations_data")
            if bootstrap_refit_data: # Check if data actually exists
                print(f"Attempting refitting with k_fit={k_fit}...")
                mean_a_refit, mean_b_refit, mean_r_sq_refit, num_fits = perform_bootstrap_refitting(bootstrap_refit_data, k_fit)
                
                if num_fits > 0:
                    print(f"Refitting successful ({num_fits} bootstrap samples). Using refitted parameters.")
                    current_fit_a = mean_a_refit
                    current_fit_b = mean_b_refit
                    current_r_squared = mean_r_sq_refit
                    k_fit_display_val = k_fit
                    if points_for_plot_psi and len(points_for_plot_psi) > 0:
                        min_psi = np.min(points_for_plot_psi)
                        max_psi = np.max(points_for_plot_psi)
                        line_x_for_plot = np.array([min_psi, max_psi])
                        line_y_for_plot = current_fit_a * line_x_for_plot + current_fit_b
                else:
                    print(f"Refitting with k_fit={k_fit} failed (0 successful bootstrap fits). Using original Gumbel fit parameters.")
            else:
                print(f"k_fit={k_fit} provided, but 'bootstrap_iterations_data' key found with no data. Using original fit.")
        else: # "bootstrap_iterations_data" not in data
            print(f"k_fit={k_fit} provided, but 'bootstrap_iterations_data' not found in input JSON. Using original fit.")
    # If k_fit is None, original parameters are used by default, no explicit message needed here.
    
    # Calculate fit statistics using the chosen line parameters (current_fit_a, current_fit_b)
    # but against the original set of top-k points that the initial fit was based on.
    fit_stats = calculate_fit_statistics(original_psi_values, original_log_survival_probs, current_fit_a, current_fit_b)
    
    # Adjust base_filename if refitting occurred
    if k_fit_display_val is not None:
        base_filename_adjusted = f"{base_filename}-kfit{k_fit_display_val}"
    else:
        base_filename_adjusted = base_filename

    # Save fit statistics to a JSON file
    fit_stats_path = os.path.join(output_dir, f"{base_filename_adjusted}-fit-statistics.json")
    with open(fit_stats_path, 'w') as f:
        json.dump(fit_stats, f, indent=4)
    
    # 1. Linear Regression Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_linear_regression(ax1, 
                           points_for_plot_psi, 
                           points_for_plot_log_surv, 
                           current_fit_a, 
                           current_fit_b, 
                           current_r_squared, 
                           line_x_for_plot, 
                           line_y_for_plot, 
                           k_fit_val=k_fit_display_val)
    linear_regression_path = os.path.join(output_dir, f"{base_filename_adjusted}-linear-regression.png")
    fig1.tight_layout()
    fig1.savefig(linear_regression_path, dpi=300)
    plt.close(fig1)
    
    # 2. QQ Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_qq(ax2, data)
    qq_plot_path = os.path.join(output_dir, f"{base_filename_adjusted}-qq-plot.png")
    fig2.tight_layout()
    fig2.savefig(qq_plot_path, dpi=300)
    plt.close(fig2)
    
    # 3. Probability Plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_probability(ax3, data)
    probability_plot_path = os.path.join(output_dir, f"{base_filename_adjusted}-probability-plot.png")
    fig3.tight_layout()
    fig3.savefig(probability_plot_path, dpi=300)
    plt.close(fig3)
    
    # 4. P_ELICIT Histogram
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    plot_p_elicit_histogram(ax4, data)
    histogram_path = os.path.join(output_dir, f"{base_filename_adjusted}-p-elicit-histogram.png")
    fig4.tight_layout()
    fig4.savefig(histogram_path, dpi=300)
    plt.close(fig4)
    
    # 5. Residual Analysis Plot
    fig5 = plt.figure(figsize=(15, 12))
    plot_residual_analysis(fig5, fit_stats)
    residual_analysis_path = os.path.join(output_dir, f"{base_filename_adjusted}-residual-analysis.png")
    fig5.tight_layout()
    fig5.savefig(residual_analysis_path, dpi=300)
    plt.close(fig5)
    
    # 6. Combined Plot (2x2 grid)
    fig6 = plt.figure(figsize=(15, 12))
    
    # Linear Regression
    ax6_1 = fig6.add_subplot(2, 2, 1)
    plot_linear_regression(ax6_1, 
                           points_for_plot_psi, 
                           points_for_plot_log_surv, 
                           current_fit_a, 
                           current_fit_b, 
                           current_r_squared, 
                           line_x_for_plot, 
                           line_y_for_plot, 
                           k_fit_val=k_fit_display_val)
    
    # QQ Plot
    ax6_2 = fig6.add_subplot(2, 2, 2)
    plot_qq(ax6_2, data)
    
    # Probability Plot
    ax6_3 = fig6.add_subplot(2, 2, 3)
    plot_probability(ax6_3, data)
    
    # P_ELICIT Histogram
    ax6_4 = fig6.add_subplot(2, 2, 4)
    plot_p_elicit_histogram(ax6_4, data)
    
    combined_path = os.path.join(output_dir, f"{base_filename_adjusted}-combined-plots.png")
    fig6.tight_layout()
    fig6.savefig(combined_path, dpi=300)
    plt.close(fig6)
    
    # Return paths to all generated files
    return {
        "linear_regression": linear_regression_path,
        "qq_plot": qq_plot_path,
        "probability_plot": probability_plot_path,
        "p_elicit_histogram": histogram_path,
        "residual_analysis": residual_analysis_path,
        "fit_statistics": fit_stats_path,
        "combined": combined_path,
        # No sensitivity_analysis or sensitivity_data paths to return
    }

def plot_multi_seed_linear_regression(seeds_data, output_dir, filename, k_fit=None):
    """Create a plot comparing linear regressions across multiple seeds.
    
    Args:
        seeds_data: Dictionary mapping seed IDs to their visualization data.
        output_dir: Directory to save the output plot.
        filename: Filename for the output plot.
        k_fit: The k value used for fitting (optional), affects titling if provided.
        
    Returns:
        Path to the saved plot file.
    """
    # Create a figure large enough to show all seeds clearly
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define a list of colors for different seeds
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    markers = ['o', 's', '^', 'v', 'D', '*', 'x', '+', '<', '>']
    
    for i, (seed, data) in enumerate(seeds_data.items()):
        # Determine original fit parameters and data points
        psi_values_to_plot = data["linear_regression_fit"]["top_scores"]
        log_survival_probs_to_plot = data["linear_regression_fit"]["log_survival_probs"]
        fit_a_to_use = data["linear_regression_fit"]["fit_a"]
        fit_b_to_use = data["linear_regression_fit"]["fit_b"]
        r_value_to_use = data["linear_regression_fit"]["fit_r"]
        fitted_line_x_to_use = data["linear_regression_fit"]["fitted_line"]["x"]
        fitted_line_y_to_use = data["linear_regression_fit"]["fitted_line"]["y"]
        r_squared_to_use = r_value_to_use**2
        effective_k_fit = None

        if k_fit is not None and "bootstrap_iterations_data" in data:
            bootstrap_data_seed = data.get("bootstrap_iterations_data")
            mean_a_refit, mean_b_refit, mean_r_sq_refit, num_fits = perform_bootstrap_refitting(bootstrap_data_seed, k_fit)
            if num_fits > 0:
                fit_a_to_use = mean_a_refit
                fit_b_to_use = mean_b_refit
                r_squared_to_use = mean_r_sq_refit # r_value is not directly available from refit, use r_squared
                effective_k_fit = k_fit
                # Recalculate line for plot based on refitted params and original points' range
                if psi_values_to_plot:
                    min_psi = np.min(psi_values_to_plot)
                    max_psi = np.max(psi_values_to_plot)
                    fitted_line_x_to_use = np.array([min_psi, max_psi])
                    fitted_line_y_to_use = fit_a_to_use * fitted_line_x_to_use + fit_b_to_use
        
        # Get color and marker for this seed (cycle through if more seeds than colors/markers)
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        
        # Plot scatter points with some transparency
        ax.scatter(psi_values_to_plot, log_survival_probs_to_plot, 
                   color=colors[color_idx], marker=markers[marker_idx], 
                   alpha=0.6, label=f'{seed} (points)')
        
        label_suffix = f' (Refit k={effective_k_fit})' if effective_k_fit is not None else ''
        # Plot the fitted line with a different style
        ax.plot(fitted_line_x_to_use, fitted_line_y_to_use, 
                color=colors[color_idx], linestyle='-', linewidth=2,
                label=f'{seed}{label_suffix}: y = {fit_a_to_use:.4f}x + {fit_b_to_use:.4f}, R² = {r_squared_to_use:.4f}')
    
    # Add labels and title
    ax.set_xlabel('ψ (Elicitation Score)')
    ax.set_ylabel('log S(ψ) = log(rank/m)')
    title_str = 'Gumbel Tail Linear Regression - Multiple Seeds'
    if k_fit is not None:
        title_str += f' (Attempted Refit with k={k_fit})'
    ax.set_title(title_str)
    
    # Add a grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add legend with smaller font to accommodate all seeds
    ax.legend(fontsize='small', loc='best')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    
    return plot_path

def generate_input_path(base_path, seed):
    """Generate the input file path for a given seed by replacing the seed placeholder.
    
    Args:
        base_path: The base input path with a seed placeholder.
        seed: The seed ID to insert.
        
    Returns:
        The input file path with the seed inserted.
    """
    # Replace seed placeholder (e.g., "s{seed}") with the actual seed
    if "{seed}" in base_path:
        return base_path.format(seed=seed)
    else:
        # If there's no explicit placeholder, try to replace 's1', 's2', etc.
        for s in range(1, 100):  # Reasonable range for existing seeds
            if f"s{s}" in base_path:
                return base_path.replace(f"s{s}", f"s{seed}")
    
    # If no replacement was possible, warn and return the original path
    print(f"Warning: Could not insert seed '{seed}' into path: {base_path}")
    return base_path

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for assessing Gumbel fit quality")
    parser.add_argument("--input_file", help="Path to the visualization data JSON file (with optional {seed} placeholder)")
    parser.add_argument("--seeds", nargs='+', type=str, help="List of seeds to process (e.g., 's1 s2 s3')")
    parser.add_argument("--output_dir", default="/h/liaidan/OpenRLHF/openrlhf/forecasting_rare_outputs/diagnostic_plots", help="Directory to save the output plots")
    parser.add_argument("--base_filename", default="gumbel-fit", help="Base filename for output plots")
    parser.add_argument("--k_fit", type=int, default=None, help="Number of top points (k <= 10) from bootstrap data to use for refitting linear regression.")
    
    args = parser.parse_args()
    
    if args.seeds:
        # Process multiple seeds
        seeds_data = {}
        seed_list = args.seeds
        
        if not args.input_file:
            parser.error("--input_file is required when using --seeds")
            
        for seed in seed_list:
            # Generate the input path for this seed
            input_path = generate_input_path(args.input_file, seed)
            
            # Check if file exists
            if not os.path.exists(input_path):
                print(f"Warning: Input file for seed {seed} not found: {input_path}")
                continue
                
            print(f"Processing seed {seed}: {input_path}")
            
            # Load visualization data for this seed
            data = load_visualization_data(input_path)
            
            # If k_fit is provided, try to load bootstrap_iterations_data from the summary file
            if args.k_fit is not None:
                summary_file_path = input_path.replace("-visualization-data.json", "-summary.json")
                if os.path.exists(summary_file_path):
                    try:
                        with open(summary_file_path, 'r') as f_summary:
                            summary_data = json.load(f_summary)
                        if "bootstrap_iterations_data" in summary_data:
                            # Check for suitability for refitting
                            boot_data_for_refit_check = summary_data.get("bootstrap_iterations_data")
                            if isinstance(boot_data_for_refit_check, dict) and \
                               "top_10_scores" in boot_data_for_refit_check and \
                               "top_10_log_survival_probs" in boot_data_for_refit_check and \
                               isinstance(boot_data_for_refit_check["top_10_scores"], list) and \
                               isinstance(boot_data_for_refit_check["top_10_log_survival_probs"], list) and \
                               len(boot_data_for_refit_check["top_10_scores"]) > 0:
                                data["bootstrap_iterations_data"] = boot_data_for_refit_check
                                print(f"Loaded bootstrap data for refitting seed {seed} from {summary_file_path}")
                            else:
                                print(f"Warning: Bootstrap data in {summary_file_path} for seed {seed} is not suitable for refitting (e.g., missing 'top_10_scores', run was likely --use_all_queries_no_bootstrap). k_fit={args.k_fit} will be ignored for this seed's individual plots.")
                                # Ensure no refitting happens if k_fit was intended but data is unsuitable
                                if "bootstrap_iterations_data" in data: # Clear any partial/unsuitable data
                                    del data["bootstrap_iterations_data"]
                        else:
                            print(f"Warning: 'bootstrap_iterations_data' not found in {summary_file_path} for seed {seed}. k_fit={args.k_fit} will be ignored for this seed's individual plots.")
                    except Exception as e:
                        print(f"Warning: Could not load or parse {summary_file_path} for seed {seed}: {e}. k_fit={args.k_fit} will be ignored for this seed's individual plots.")
                else:
                    print(f"Warning: Summary file {summary_file_path} not found for seed {seed}. k_fit={args.k_fit} will be ignored for this seed's individual plots.")
            
            # Store data for combined plot
            seeds_data[seed] = data
            
            # Also generate individual plots for this seed
            base_seed_filename = f"{args.base_filename}-{seed}"
            output_files = plot_all_visualizations(data, args.output_dir, base_seed_filename, k_fit=args.k_fit)
            
            print(f"Individual visualizations for seed {seed} saved to {args.output_dir}")
        
        # Generate combined plot for all seeds if we have data
        if seeds_data:
            combined_filename = f"{args.base_filename}-multi-seed-linear-regression.png"
            combined_plot_path = plot_multi_seed_linear_regression(seeds_data, args.output_dir, combined_filename, k_fit=args.k_fit)
            print(f"Combined linear regression plot saved: {combined_plot_path}")
        else:
            print("No valid data found for any seed. Combined plot could not be created.")
            
    else:
        # Original functionality for single input file
        if not args.input_file:
            parser.error("--input_file is required")
            
        # Load visualization data
        data = load_visualization_data(args.input_file)
        
        # If k_fit is provided, try to load bootstrap_iterations_data from the summary file
        if args.k_fit is not None:
            summary_file_path = args.input_file.replace("-visualization-data.json", "-summary.json")
            if os.path.exists(summary_file_path):
                try:
                    with open(summary_file_path, 'r') as f_summary:
                        summary_data = json.load(f_summary)
                    if "bootstrap_iterations_data" in summary_data:
                        # Check for suitability for refitting
                        boot_data_for_refit_check = summary_data.get("bootstrap_iterations_data")
                        if isinstance(boot_data_for_refit_check, dict) and \
                           "top_10_scores" in boot_data_for_refit_check and \
                           "top_10_log_survival_probs" in boot_data_for_refit_check and \
                           isinstance(boot_data_for_refit_check["top_10_scores"], list) and \
                           isinstance(boot_data_for_refit_check["top_10_log_survival_probs"], list) and \
                           len(boot_data_for_refit_check["top_10_scores"]) > 0:
                            data["bootstrap_iterations_data"] = boot_data_for_refit_check
                            print(f"Loaded bootstrap data for refitting from {summary_file_path}")
                        else:
                            print(f"Warning: Bootstrap data in {summary_file_path} is not suitable for refitting (e.g., missing 'top_10_scores', run was likely --use_all_queries_no_bootstrap). k_fit={args.k_fit} will be ignored.")
                             # Ensure no refitting happens if k_fit was intended but data is unsuitable
                            if "bootstrap_iterations_data" in data: # Clear any partial/unsuitable data
                                del data["bootstrap_iterations_data"]
                            effective_k_fit = None # This will prevent refitting attempt in plot_all_visualizations
                    else:
                        print(f"Warning: 'bootstrap_iterations_data' not found in {summary_file_path}. k_fit={args.k_fit} will be ignored.")
                        effective_k_fit = None
                except Exception as e:
                    print(f"Warning: Could not load or parse {summary_file_path}: {e}. k_fit={args.k_fit} will be ignored.")
                    effective_k_fit = None
            else:
                print(f"Warning: Summary file {summary_file_path} not found. k_fit={args.k_fit} will be ignored.")
                effective_k_fit = None
        
        # Determine k_fit to use for plot_all_visualizations
        # If args.k_fit was given, but data["bootstrap_iterations_data"] was removed or never suitable,
        # then k_fit passed to plot_all_visualizations should be None.
        k_to_pass_to_plotter = args.k_fit
        if args.k_fit is not None and not (data.get("bootstrap_iterations_data") and \
                                           data["bootstrap_iterations_data"].get("top_10_scores")):
            k_to_pass_to_plotter = None

        # Generate and save all plots
        output_files = plot_all_visualizations(data, args.output_dir, args.base_filename, k_fit=k_to_pass_to_plotter)
        
        print(f"Visualizations generated and saved to {args.output_dir}:")
        for plot_type, path in output_files.items():
            if path:  # Skip None values
                print(f"  - {plot_type}: {os.path.basename(path)}")

if __name__ == "__main__":
    main() 