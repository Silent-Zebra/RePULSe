import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

try:
    from openrlhf.forecasting_rare_outputs.forecasting import forecast_worst_query_risk
except ImportError:
    # This is a fallback for cases where the script might be run in an environment
    # where openrlhf is not in PYTHONPATH, or for a simpler definition if needed.
    # For the full project, the above import should work.
    print("Warning: Could not import forecast_worst_query_risk from openrlhf.forecasting. Using a placeholder.")
    def forecast_worst_query_risk(a, b, n):
        # Placeholder: -log(-log( (a * (-log(n)) + b) )) -> Q_psi = (-log n -b)/a -> Q_p = exp(-exp(-Q_psi))
        if a == 0: # Avoid division by zero
            return np.nan 
        q_psi_n = (-np.log(n) - b) / a
        return np.exp(-np.exp(-q_psi_n))

legend_mapping = {
    "main-q-prop": r"$q_\psi$ proposal, $\theta_\theta(s)\propto p_\theta(s)e^{-30r(s)},\;\alpha=0.003$ (2 Episodes)",
    "p-prop": r"$p_\theta$ proposal, $\sigma_\theta(s)\propto p_\theta(s)e^{-30r(s)},\;\alpha=0.003$ (4 Episodes)",
    "ppo": r"PPO (4 Episodes)",
    "reinforce": r"REINFORCE (4 Episodes)",
    "reinforce-reward": r"REINFORCE, $r(s)-0.003e^{-30r(s)}$ (4 Episodes)",
}

def load_results(base_results_dir: str, specific_run_prefix: str = "", k_fit: int = 10) -> pd.DataFrame:
    """Loads summary JSON files from experiments into a pandas DataFrame.
    Recursively searches for _summary.json files within base_results_dir.
    Extracts model_name and seed from the directory structure.
    Expected structure: base_results_dir / model_name / seed / behavior_dir / xxx_summary.json
    """
    all_data = []
    processed_files = 0
    skipped_files = 0
    
    print(f"Starting to search for summary files in {base_results_dir}...")
    
    for root, dirs, files in os.walk(base_results_dir):
        for filename in files:
            # Working options:
            # logprob_target_keyword_in_target_seq-summary.json
            # logprob_target_seq-summary.json
            # We do not consider repeated sampling for now.
            if filename.endswith("logprob_target_seq-summary.json") and filename.startswith(specific_run_prefix):
                filepath = os.path.join(root, filename)
                print(f"Processing file: {filepath}")
                processed_files += 1
                
                try:
                    # Extract model_type and seed_id from path
                    relative_path = os.path.relpath(filepath, base_results_dir)
                    path_parts = relative_path.split(os.sep)
                    print(f"  Path parts: {path_parts}")

                    parsed_model_type = "N/A"
                    parsed_seed_id = "N/A"
                    
                    if len(path_parts) >= 3: 
                        parsed_model_type = path_parts[0] # This is the 'model_type' from the directory structure
                        parsed_seed_id = path_parts[1]    # This is the 'seed' from the directory structure
                        print(f"  Extracted model_type: {parsed_model_type}, seed_id: {parsed_seed_id}")
                    
                    with open(filepath, 'r') as f:
                        summary_data = json.load(f)
                        
                        behavior_id_from_json = summary_data.get("behavior_id", "N/A")
                        behavior_id_from_path = path_parts[2] if len(path_parts) >= 4 else "N/A"
                        
                        behavior_id = behavior_id_from_json if behavior_id_from_json != "N/A" else behavior_id_from_path
                        
                        print(f"  Behavior ID from JSON: {behavior_id_from_json}")
                        print(f"  Behavior ID from path: {behavior_id_from_path}")
                        print(f"  Using behavior ID: {behavior_id}")
                        
                        bootstrap_summary = summary_data.get("gumbel_fit_params_bootstrap_summary", {})
                        
                        flat_data = {
                            "run_name": filename.replace("_summary.json", "").replace("-summary.json", ""),
                            "model_name": parsed_model_type,
                            "seed_id": parsed_seed_id,
                            "filepath": filepath,
                            "model_path": summary_data.get("pretrain", "N/A"), # Changed from model_path for consistency
                            "behavior_id": behavior_id,
                            "elicitation_method": summary_data.get("elicitation_method", "N/A"),
                            "num_bootstrap_samples_requested": summary_data.get("num_bootstrap_samples_requested", np.nan),
                            "num_successful_gumbel_fits": summary_data.get("num_successful_gumbel_fits", np.nan),
                            "m_eval_size_per_bootstrap": summary_data.get("evaluation_set_size_m_per_bootstrap", np.nan), # New name for clarity
                            "mean_num_valid_p_per_bootstrap": summary_data.get("mean_num_valid_p_elicits_per_bootstrap", np.nan), # New name for clarity

                            "fit_a": bootstrap_summary.get("mean_a_slope", np.nan),
                            "fit_b": bootstrap_summary.get("mean_b_intercept", np.nan),
                            "fit_r2": bootstrap_summary.get("mean_r_squared_approx", np.nan), # Using approximated R^2 from mean r_value
                            "fit_top_k": bootstrap_summary.get("mean_top_k_actually_used", np.nan),
                            
                            "std_fit_a": bootstrap_summary.get("std_a_slope", np.nan),
                            "std_fit_b": bootstrap_summary.get("std_b_intercept", np.nan),
                            "std_fit_r_value": bootstrap_summary.get("std_r_value", np.nan),

                            "bootstrap_a_slopes": summary_data.get("bootstrap_iterations_data", {}).get("a_slopes", []),
                            "bootstrap_b_intercepts": summary_data.get("bootstrap_iterations_data", {}).get("b_intercepts", []),
                            "is_from_all_valid_scores": False # Initialize flag
                        }
                        
                        # Try to load from fit_visualization_data.all_valid_scores first
                        fit_vis_data = summary_data.get("fit_visualization_data", {}).get("all_valid_scores", {})
                        psi_from_all_valid = fit_vis_data.get("psi_values", [])
                        quantiles_from_all_valid = fit_vis_data.get("empirical_quantiles", [])

                        if (isinstance(psi_from_all_valid, list) and psi_from_all_valid and 
                            isinstance(quantiles_from_all_valid, list) and quantiles_from_all_valid and
                            len(psi_from_all_valid) == len(quantiles_from_all_valid)):
                            
                            m = len(psi_from_all_valid)
                            ranks = np.arange(1, k_fit+1)
                            log_survival_probs_top_k = np.log(ranks / m).tolist()
                            flat_data["boot_iter_top_10_psi"] = [psi_from_all_valid]
                            flat_data["boot_iter_log_surv_prob"] = [log_survival_probs_top_k]
                            flat_data["is_from_all_valid_scores"] = True
                            print(f"  Successfully loaded {len(psi_from_all_valid)} points from fit_visualization_data.all_valid_scores for {filepath}")
                        else:
                            # Fallback to bootstrap_iterations_data or gumbel_fit_top_scores
                            bootstrap_iterations_data_content = summary_data.get("bootstrap_iterations_data", {})
                            psi_scores_from_bootstrap_data = bootstrap_iterations_data_content.get("top_10_scores", [])
                            log_surv_from_bootstrap_data = bootstrap_iterations_data_content.get("top_10_log_survival_probs", [])

                            bootstrap_data_is_empty_or_malformed = (
                                not isinstance(psi_scores_from_bootstrap_data, list) or
                                not psi_scores_from_bootstrap_data or
                                not isinstance(psi_scores_from_bootstrap_data[0], list) or # check if inner list exists
                                not psi_scores_from_bootstrap_data[0] # check if inner list is non-empty
                            )

                            if bootstrap_data_is_empty_or_malformed:
                                # Fallback to top-level gumbel_fit_top_scores for "no bootstrap" cases
                                # or if bootstrap_iterations_data lacks detailed scores.
                                top_scores = summary_data.get("gumbel_fit_top_scores", [])
                                log_surv_probs = summary_data.get("gumbel_fit_log_survival_probs", [])
                                
                                # Ensure these are lists of lists, even if only one set of scores
                                flat_data["boot_iter_top_10_psi"] = [top_scores] if top_scores and isinstance(top_scores, list) and (not isinstance(top_scores[0], list) if top_scores else True) else top_scores
                                flat_data["boot_iter_log_surv_prob"] = [log_surv_probs] if log_surv_probs and isinstance(log_surv_probs, list) and (not isinstance(log_surv_probs[0], list) if log_surv_probs else True) else log_surv_probs
                                
                                if not flat_data["boot_iter_top_10_psi"] or \
                                   (isinstance(flat_data["boot_iter_top_10_psi"], list) and flat_data["boot_iter_top_10_psi"] and \
                                    isinstance(flat_data["boot_iter_top_10_psi"][0], list) and not flat_data["boot_iter_top_10_psi"][0]):
                                    print(f"  Warning: Fallback to top-level gumbel_fit_top_scores also yielded no/empty data for {filepath}. Refitting might fail or use no points.")
                                else:
                                    print(f"  Used top-level gumbel_fit_top_scores for {filepath}")
                            else:
                                flat_data["boot_iter_top_10_psi"] = psi_scores_from_bootstrap_data
                                flat_data["boot_iter_log_surv_prob"] = log_surv_from_bootstrap_data
                                print(f"  Used bootstrap_iterations_data for {filepath}")
                        
                        # Add forecasted risks Q_p(n) from mean parameters
                        forecasts = summary_data.get("forecasted_worst_query_risks_from_mean_params", {})
                        if not forecasts and "forecasted_worst_query_risks" in summary_data:
                            print(f"  Warning: Using fallback 'forecasted_worst_query_risks' for {filepath}")
                            forecasts = summary_data.get("forecasted_worst_query_risks", {})

                        for key, value in forecasts.items():
                            try:
                                n_str = key.split('(')[-1].split(')')[0]
                                n_val = int(float(n_str)) # Handle potential float strings like "1e5"
                                flat_data[f"Q_p(n={n_val})"] = value
                            except ValueError:
                                print(f"  Warning: Could not parse n from forecast key '{key}' in {filename}. Value: {n_str}")
                            except Exception as e:
                                print(f"  Warning: Error processing forecast key '{key}' in {filename}: {e}")

                        all_data.append(flat_data)
                except Exception as e:
                    print(f"Error loading or processing {filepath}: {e}")
                    skipped_files += 1

    print(f"\nSummary: Processed {processed_files} files, skipped {skipped_files} files due to errors")
    print(f"Successfully loaded data from {len(all_data)} summary files")
    
    if not all_data:
        print(f"Warning: No summary files found in {base_results_dir} with prefix '{specific_run_prefix}' matching the expected directory structure.")
        return pd.DataFrame()
        
    if all_data:
        df = pd.DataFrame(all_data)
        print("\nLoaded data statistics:")
        print(f"Number of unique models: {df['model_name'].nunique()}")
        print(f"Number of unique seeds: {df['seed_id'].nunique()}")
        print(f"Seeds found: {sorted(df['seed_id'].unique())}")
        print(f"Number of unique behaviors: {df['behavior_id'].nunique()}")
        
        seed_counts = df.groupby('model_name')['seed_id'].nunique()
        print("\nNumber of seeds per model:")
        for model, count in seed_counts.items():
            print(f"  {model}: {count} seeds")

    return pd.DataFrame(all_data)

def _fit_gumbel_from_all_valid_scores(all_psi_scores: list[float], k_fit: int) -> tuple[float, float, float]:
    """
    Fits a Gumbel distribution to the top k_fit scores from all_psi_scores.
    
    Args:
        all_psi_scores: List of all valid psi scores, assumed to be sorted in ascending order.
        k_fit: Number of top scores to use for the fit.
    
    Returns:
        Tuple of (a_slope, b_intercept, r_value) from the linear regression fit.
        Returns (np.nan, np.nan, np.nan) if fitting fails.
    """
    if not isinstance(all_psi_scores, list) or len(all_psi_scores) < 2:
        return np.nan, np.nan, np.nan
    
    try:
        # Calculate total number of valid scores
        m = len(all_psi_scores)
        
        # Select top k_fit scores (largest values)
        actual_k_to_use = min(k_fit, m)
        if actual_k_to_use < 2:
            return np.nan, np.nan, np.nan
        
        # Get top k scores (from the end since all_psi_scores is sorted in ascending order)
        selected_psi = all_psi_scores[-actual_k_to_use:]
        
        # Reverse to have highest score first
        selected_psi = selected_psi[::-1]
        
        # Calculate corresponding ranks and log survival probabilities
        ranks = np.arange(1, actual_k_to_use + 1)
        selected_log_surv = np.log(ranks / m).tolist()
        
        # Filter out any non-finite values
        valid_indices = [idx for idx, (p_val, l_val) in enumerate(zip(selected_psi, selected_log_surv)) 
                         if pd.notna(p_val) and pd.notna(l_val) and np.isfinite(p_val) and np.isfinite(l_val)]
        
        if len(valid_indices) < 2:
            return np.nan, np.nan, np.nan
        
        final_selected_psi = [selected_psi[j] for j in valid_indices]
        final_selected_log_surv = [selected_log_surv[j] for j in valid_indices]
        
        # Perform linear regression
        slope, intercept, r_value, _, _ = linregress(final_selected_psi, final_selected_log_surv)
        
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            return np.nan, np.nan, np.nan
        
        return slope, intercept, r_value
    
    except Exception as e:
        print(f"Error in _fit_gumbel_from_all_valid_scores: {e}")
        return np.nan, np.nan, np.nan

def perform_gumbel_refitting(df: pd.DataFrame, k_fit: int) -> pd.DataFrame:
    """
    Refits Gumbel parameters (a, b, r) for each bootstrap sample within each run,
    using the specified number of top_k scores (k_fit).
    Updates Q_p(n) columns and adds/updates fit_r2_refit based on these new fits.
    """
    
    refit_a_slopes_all_rows = []
    refit_b_intercepts_all_rows = []
    refit_r_values_all_rows = []

    for _, row in df.iterrows():
        boot_psis_all_iters = row.get('boot_iter_top_10_psi', [])
        boot_log_surv_probs_all_iters = row.get('boot_iter_log_surv_prob', [])
        is_all_valid_scores_data = row.get('is_from_all_valid_scores', False)
        num_bootstrap_iters = len(boot_psis_all_iters)
        
        current_row_refit_as = []
        current_row_refit_bs = []
        current_row_refit_rs = []

        if not isinstance(boot_psis_all_iters, list) or not isinstance(boot_log_surv_probs_all_iters, list):
            print(f"Warning: Skipping refit for a row due to unexpected data type for psi/log_surv_prob. Run: {row.get('run_name', 'N/A')}, Seed: {row.get('seed_id', 'N/A')}")
            num_original_bootstraps = len(row.get('bootstrap_a_slopes', [np.nan]))
            current_row_refit_as = [np.nan] * num_original_bootstraps
            current_row_refit_bs = [np.nan] * num_original_bootstraps
            current_row_refit_rs = [np.nan] * num_original_bootstraps

        elif num_bootstrap_iters == 0 or not boot_psis_all_iters or not boot_log_surv_probs_all_iters:
            num_original_bootstraps = len(row.get('bootstrap_a_slopes', [np.nan]))
            current_row_refit_as = [np.nan] * num_original_bootstraps
            current_row_refit_bs = [np.nan] * num_original_bootstraps
            current_row_refit_rs = [np.nan] * num_original_bootstraps
        else:
            # Special case for data from all_valid_scores
            if is_all_valid_scores_data and len(boot_psis_all_iters) == 1:
                all_psi_scores = boot_psis_all_iters[0]
                
                # Use the new helper function to fit Gumbel distribution
                a_slope, b_intercept, r_value = _fit_gumbel_from_all_valid_scores(all_psi_scores, k_fit)
                
                # Store the results (as lists of length 1 to maintain compatibility)
                current_row_refit_as = [a_slope]
                current_row_refit_bs = [b_intercept]
                current_row_refit_rs = [r_value]
            else:
                # Original logic for bootstrap data
                for i in range(num_bootstrap_iters):
                    psi_scores_iter_single_boot = boot_psis_all_iters[i]
                    log_surv_probs_iter_single_boot = boot_log_surv_probs_all_iters[i]

                    if not (psi_scores_iter_single_boot and 
                            log_surv_probs_iter_single_boot and 
                            isinstance(psi_scores_iter_single_boot, list) and 
                            isinstance(log_surv_probs_iter_single_boot, list) and 
                            len(psi_scores_iter_single_boot) == len(log_surv_probs_iter_single_boot)):
                        current_row_refit_as.append(np.nan)
                        current_row_refit_bs.append(np.nan)
                        current_row_refit_rs.append(np.nan)
                        continue

                    num_available_points = len(psi_scores_iter_single_boot)
                    actual_k_to_use = min(k_fit, num_available_points)

                    if actual_k_to_use < 2:
                        current_row_refit_as.append(np.nan)
                        current_row_refit_bs.append(np.nan)
                        current_row_refit_rs.append(np.nan)
                        continue
                    
                    # Data is already sorted: highest score (psi) first.
                    # For is_all_valid_scores_data, psi_values are ascending, so we take from the end and reverse.
                    if is_all_valid_scores_data:
                        # Take the largest k_fit scores (from the end of the list)
                        selected_psi = psi_scores_iter_single_boot[-actual_k_to_use:]
                        selected_log_surv = log_surv_probs_iter_single_boot[-actual_k_to_use:]
                        # Reverse to match expected descending order for linregress (highest psi first)
                        selected_psi = selected_psi[::-1]
                        selected_log_surv = selected_log_surv[::-1]
                    else:
                        # Original logic: data is assumed to be pre-sorted (descending psi)
                        selected_psi = psi_scores_iter_single_boot[:actual_k_to_use]
                        selected_log_surv = log_surv_probs_iter_single_boot[:actual_k_to_use]
                    
                    try:
                        valid_indices = [idx for idx, (p_val, l_val) in enumerate(zip(selected_psi, selected_log_surv)) 
                                         if pd.notna(p_val) and pd.notna(l_val) and np.isfinite(p_val) and np.isfinite(l_val)]
                        
                        if len(valid_indices) < 2:
                            current_row_refit_as.append(np.nan)
                            current_row_refit_bs.append(np.nan)
                            current_row_refit_rs.append(np.nan)
                            continue

                        final_selected_psi = [selected_psi[j] for j in valid_indices]
                        final_selected_log_surv = [selected_log_surv[j] for j in valid_indices]

                        if len(final_selected_psi) < 2:
                            current_row_refit_as.append(np.nan)
                            current_row_refit_bs.append(np.nan)
                            current_row_refit_rs.append(np.nan)
                            continue

                        slope, intercept, r_value, _, _ = linregress(final_selected_psi, final_selected_log_surv)
                        
                        if not (np.isfinite(slope) and np.isfinite(intercept)):
                            current_row_refit_as.append(np.nan)
                            current_row_refit_bs.append(np.nan)
                            current_row_refit_rs.append(np.nan)
                        else:
                            current_row_refit_as.append(slope)
                            current_row_refit_bs.append(intercept)
                            current_row_refit_rs.append(r_value)
                    except (ValueError, TypeError) as e: # Catch errors from linregress or list operations
                        # print(f"Linregress or data selection error for run {row.get('run_name', 'N/A')}, seed {row.get('seed_id', 'N/A')}, boot iter {i}: {e}")
                        current_row_refit_as.append(np.nan)
                        current_row_refit_bs.append(np.nan)
                        current_row_refit_rs.append(np.nan)
            
        refit_a_slopes_all_rows.append(current_row_refit_as)
        refit_b_intercepts_all_rows.append(current_row_refit_bs)
        refit_r_values_all_rows.append(current_row_refit_rs)

    df['refit_bootstrap_a_slopes'] = refit_a_slopes_all_rows
    df['refit_bootstrap_b_intercepts'] = refit_b_intercepts_all_rows
    df['refit_bootstrap_r_values'] = refit_r_values_all_rows

    qpn_cols = [col for col in df.columns if col.startswith('Q_p(n=')]
    new_fit_a_means = []
    new_fit_b_means = []
    new_fit_r2_refits = []

    # DEBUG: Print original Q_p(n) sums before refitting
    print("\nDEBUG: Sum of Q_p(n) values BEFORE refitting by column:")
    for q_col in qpn_cols:
        print(f"  {q_col}: {df[q_col].sum()} (NaNs: {df[q_col].isna().sum()})")

    for i_idx, row_data in df.iterrows(): # Use i_idx and row_data to avoid conflict with outer loop var 'row' if any
        row_refit_as_clean = [val for val in row_data['refit_bootstrap_a_slopes'] if pd.notna(val) and np.isfinite(val)]
        row_refit_bs_clean = [val for val in row_data['refit_bootstrap_b_intercepts'] if pd.notna(val) and np.isfinite(val)]
        row_refit_rs_clean = [val for val in row_data['refit_bootstrap_r_values'] if pd.notna(val) and np.isfinite(val)]

        mean_a_refit_row = np.nanmean(row_refit_as_clean) if row_refit_as_clean else np.nan
        mean_b_refit_row = np.nanmean(row_refit_bs_clean) if row_refit_bs_clean else np.nan
        mean_r_refit_row = np.nanmean(row_refit_rs_clean) if row_refit_rs_clean else np.nan
        
        new_fit_a_means.append(mean_a_refit_row)
        new_fit_b_means.append(mean_b_refit_row)
        new_fit_r2_refits.append(mean_r_refit_row**2 if pd.notna(mean_r_refit_row) else np.nan)

        # DEBUG: Print mean_a_refit_row for each row
        if i_idx < 5 or i_idx > len(df) - 5 : # Print for first few and last few rows
            print(f"  DEBUG (row {i_idx}): mean_a_refit_row = {mean_a_refit_row}, num_clean_a_slopes = {len(row_refit_as_clean)}")

        for q_col in qpn_cols:
            try:
                n_str = q_col.split('(')[-1].split(')')[0].split('=')[-1]
                n_val = int(float(n_str))
                if pd.notna(mean_a_refit_row) and pd.notna(mean_b_refit_row):
                    if mean_a_refit_row >= 0:
                        # DEBUG: Log when Q_p(n) is set to NaN due to non-negative 'a'
                        if i_idx < 5 or i_idx > len(df) - 5: # Print for first few and last few rows
                             print(f"    DEBUG (row {i_idx}, {q_col}): mean_a_refit_row ({mean_a_refit_row:.4f}) >= 0. Setting Q_p(n) to NaN.")
                        df.loc[df.index[i_idx], q_col] = np.nan
                    else:
                        df.loc[df.index[i_idx], q_col] = forecast_worst_query_risk(mean_a_refit_row, mean_b_refit_row, n_val)
                else:
                    df.loc[df.index[i_idx], q_col] = np.nan
            except ValueError:
                df.loc[df.index[i_idx], q_col] = np.nan
            except Exception:
                df.loc[df.index[i_idx], q_col] = np.nan
                
    df['fit_a'] = new_fit_a_means
    df['fit_b'] = new_fit_b_means
    df['fit_r2_refit'] = new_fit_r2_refits

    # DEBUG: Print Q_p(n) sums AFTER refitting
    print("\nDEBUG: Sum of Q_p(n) values AFTER refitting by column:")
    for q_col in qpn_cols:
        print(f"  {q_col}: {df[q_col].sum()} (NaNs: {df[q_col].isna().sum()})")
    
    # DEBUG: Print distribution of new_fit_a_means
    valid_new_fit_a_means = [a for a in new_fit_a_means if pd.notna(a) and np.isfinite(a)]
    if valid_new_fit_a_means:
        print(f"\nDEBUG: Statistics for new_fit_a_means (total {len(valid_new_fit_a_means)} valid values):")
        print(f"  Mean: {np.mean(valid_new_fit_a_means):.4f}, Median: {np.median(valid_new_fit_a_means):.4f}")
        print(f"  Min: {np.min(valid_new_fit_a_means):.4f}, Max: {np.max(valid_new_fit_a_means):.4f}")
        print(f"  Percentiles (5, 25, 50, 75, 95): {np.percentile(valid_new_fit_a_means, [5, 25, 50, 75, 95])}")
        print(f"  Number of a_fit >= 0: {sum(1 for a in valid_new_fit_a_means if a >= 0)}")
        print(f"  Number of a_fit < 0: {sum(1 for a in valid_new_fit_a_means if a < 0)}")
    else:
        print("\nDEBUG: No valid new_fit_a_means values found.")

    return df

def plot_forecast_comparison(df: pd.DataFrame, 
                             behavior_id: str, 
                             output_dir: str, 
                             aggregate_by_seed: bool = True, 
                             k_fit_analysis_active: bool = False, 
                             seed_aggregation_method: str = "median", 
                             show_ci: bool = False):
    """Generates a plot comparing forecasted Q_p(n) vs n for different models.

    Args:
        df: DataFrame containing loaded results (from load_results).
        behavior_id: The specific behavior ID to plot.
        output_dir: Directory to save the plot.
        aggregate_by_seed: If True, plot mean/median over seeds; otherwise, plot individual seeds.
        k_fit_analysis_active: bool, if True, uses refitted bootstrap parameters for CI.
        seed_aggregation_method: 'mean' or 'median', how to aggregate seed data or calculate central tendency for a single seed's bootstraps.
        show_ci: bool, if True, shows confidence intervals on plots where applicable (e.g., individual seed plots).
    """
    plot_df = df[df['behavior_id'] == behavior_id].copy()
    if plot_df.empty:
        print(f"No data found for behavior ID: {behavior_id}. Skipping plot.")
        return

    qpn_cols = sorted([col for col in plot_df.columns if col.startswith('Q_p(n=')], 
                      key=lambda x: float(x.split('=')[-1][:-1]))
    if not qpn_cols:
        print(f"No Q_p(n) forecast columns found for behavior {behavior_id}. Skipping plot.")
        return

    id_vars_base = ['model_name', 'seed_id', 'model_path']
    if not aggregate_by_seed:
        plot_df['run_identifier'] = plot_df['model_name'] + "_" + plot_df['seed_id']
        id_vars = ['run_identifier'] + id_vars_base
        hue_col = 'run_identifier'
    else:
        id_vars = ['model_name']
        hue_col = 'model_name'

    plot_df_melt = plot_df.melt(id_vars=id_vars_base, value_vars=qpn_cols, var_name='forecast_scale', value_name='Q_p(n)')

    plot_df_melt['n'] = plot_df_melt['forecast_scale'].str.extract(r'Q_p\(n=(\d+\.?\d*[eE]?\d*)\)').astype(float)
    plot_df_melt.dropna(subset=['n', 'Q_p(n)'], inplace=True)

    if plot_df_melt.empty:
        print(f"No valid forecast points to plot for behavior {behavior_id}.")
        return

    plot_data_list = []

    unique_runs = plot_df['run_identifier'].unique() if not aggregate_by_seed else plot_df['model_name'].unique()
    hue_col_name = 'run_identifier' if not aggregate_by_seed else 'model_name'

    for run_id in unique_runs:
        if not aggregate_by_seed:
            run_specific_df = plot_df[plot_df['run_identifier'] == run_id]
        else:
            pass

        if not aggregate_by_seed and not run_specific_df.empty:
            # Get the first row for this run to access list of a's and b's
            # (assuming they are the same for all rows of this run_identifier before melting)
            # This requires that bootstrap_a_slopes and bootstrap_b_intercepts are properly loaded per run_name/seed_id
            # We should get these from the original plot_df, not the melted one.
            original_run_row = df[(df['model_name'] + "_" + df['seed_id']) == run_id].iloc[0]
            
            # Use refitted bootstrap slopes if k_fit_analysis_active, else original
            if k_fit_analysis_active and 'refit_bootstrap_a_slopes' in original_run_row and 'refit_bootstrap_b_intercepts' in original_run_row:
                bootstrap_as = original_run_row.get('refit_bootstrap_a_slopes', [])
                bootstrap_bs = original_run_row.get('refit_bootstrap_b_intercepts', [])
                # print(f"Debug CI for {run_id}: Using REFITTED a/b. Count_a: {len(bootstrap_as)}, Count_b: {len(bootstrap_bs)}")

            else:
                bootstrap_as = original_run_row.get('bootstrap_a_slopes', [])
                bootstrap_bs = original_run_row.get('bootstrap_b_intercepts', [])
                # print(f"Debug CI for {run_id}: Using ORIGINAL a/b. Count_a: {len(bootstrap_as)}, Count_b: {len(bootstrap_bs)}")


            if not isinstance(bootstrap_as, list) or not isinstance(bootstrap_bs, list) or not bootstrap_as or not bootstrap_bs or len(bootstrap_as) != len(bootstrap_bs):
                print(f"Warning: Missing or mismatched bootstrap_a/b_slopes for run {run_id}. Skipping CI calculation for this run.")
                run_melted_data = plot_df_melt[plot_df_melt[hue_col_name] == run_id]
                for _, row in run_melted_data.iterrows():
                    plot_data_list.append({
                        hue_col_name: run_id,
                        'n': row['n'],
                        'Q_p(n)_central (mean or median)': row['Q_p(n)'], # This is Q_p from mean params
                        'Q_p(n)_lower': row['Q_p(n)'], # No CI, this is just placeholder and right now is the same as mean
                        'Q_p(n)_upper': row['Q_p(n)'], # No CI, this is just placeholder and right now is the same as mean
                        'model_name': original_run_row['model_name'] # for styling
                    })
                continue

            num_bootstrap_samples = len(bootstrap_as)
            if num_bootstrap_samples == 0:
                print(f"Warning: No bootstrap samples found for run {run_id}. Skipping CI.")
                # Fallback logic as above
                run_melted_data = plot_df_melt[plot_df_melt[hue_col_name] == run_id]
                for _, row in run_melted_data.iterrows():
                     plot_data_list.append({
                        hue_col_name: run_id,
                        'n': row['n'],
                        'Q_p(n)_central (mean or median)': row['Q_p(n)'], 
                        'Q_p(n)_lower': row['Q_p(n)'], 
                        'Q_p(n)_upper': row['Q_p(n)'],
                        'model_name': original_run_row['model_name']
                    })
                continue

            # Extract n values from qpn_cols (forecast scales like 1e4, 1e5, etc.)
            n_values_for_plot = sorted([float(col.split('=')[-1][:-1]) for col in qpn_cols])

            for n_val in n_values_for_plot:
                bootstrapped_qpn_for_n = []
                for i in range(num_bootstrap_samples):
                    a_i = bootstrap_as[i]
                    b_i = bootstrap_bs[i]
                    if pd.isna(a_i) or pd.isna(b_i): # Handle None/NaN from JSON if not filtered by experiment_runner
                        continue
                    qpn_i = forecast_worst_query_risk(a_i, b_i, n_val)
                    bootstrapped_qpn_for_n.append(qpn_i)
                
                if bootstrapped_qpn_for_n:
                    if seed_aggregation_method == "median":
                        qpn_central = np.nanmedian(bootstrapped_qpn_for_n)
                    else: # Default to mean
                        qpn_central = np.nanmean(bootstrapped_qpn_for_n)
                    qpn_lower = np.nanpercentile(bootstrapped_qpn_for_n, 5)
                    qpn_upper = np.nanpercentile(bootstrapped_qpn_for_n, 95)
                    plot_data_list.append({
                        hue_col_name: run_id,
                        'n': n_val,
                        'Q_p(n)_central (mean or median)': qpn_central, # Store central tendency here
                        'Q_p(n)_lower': qpn_lower,
                        'Q_p(n)_upper': qpn_upper,
                        'model_name': original_run_row['model_name'] # for consistent styling by model
                    })
    
    if not aggregate_by_seed and plot_data_list:
        ci_plot_df = pd.DataFrame(plot_data_list)
        if ci_plot_df.empty:
            print(f"No data for CI plot for behavior {behavior_id}.")
            return
            
        plt.figure(figsize=(14, 8))
        
        for run_id_val, group in ci_plot_df.groupby(hue_col_name):
            group = group.sort_values(by='n')
            model_name_of_run = group['model_name'].iloc[0]
            display_label = legend_mapping.get(model_name_of_run, run_id_val)
            plt.plot(group['n'], group['Q_p(n)_central (mean or median)'], marker='o', label=display_label, linestyle='--', markersize=5)
            if show_ci:
                plt.fill_between(group['n'], group['Q_p(n)_lower'], group['Q_p(n)_upper'], alpha=0.2)
        
        plt.legend(loc='lower right', fontsize=10)
    elif aggregate_by_seed:
        plot_df_melted_for_agg = plot_df.melt(id_vars=['model_name', 'seed_id'], value_vars=qpn_cols, var_name='forecast_scale', value_name='Q_p(n)')
        plot_df_melted_for_agg['n'] = plot_df_melted_for_agg['forecast_scale'].str.extract(r'Q_p\(n=(\d+\.?\d*[eE]?\d*)\)').astype(float)
        plot_df_melted_for_agg.dropna(subset=['n', 'Q_p(n)'], inplace=True)
        
        if plot_df_melted_for_agg.empty:
            print(f"No data to plot for behavior {behavior_id} when aggregate_by_seed=True.")
            return
            
        plt.figure(figsize=(12, 7))
        estimator_func = np.nanmean if seed_aggregation_method == "mean" else np.nanmedian
        if show_ci:
            sns.lineplot(data=plot_df_melted_for_agg, x='n', y='Q_p(n)', hue='model_name', marker='o', 
                         estimator=estimator_func, errorbar=('ci', 95))
        else:
            sns.lineplot(data=plot_df_melted_for_agg, x='n', y='Q_p(n)', hue='model_name', marker='o', 
                         estimator=estimator_func, errorbar=None)
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [legend_mapping.get(label, label) for label in labels]
        plt.legend(handles, new_labels, loc='lower right', fontsize=10)
    else:
        if not plot_df_melt.empty:
            print("Falling back to plotting mean Q_p(n) without CIs as bootstrap data was insufficient.")
            plt.figure(figsize=(12,7))
            plot_df_melt['line_hue'] = plot_df_melt['model_name'] + " (" + plot_df_melt['seed_id'] + ")"
            sns.lineplot(data=plot_df_melt, x='n', y='Q_p(n)', hue='line_hue', marker='o', style='model_name', dashes=True)
            handles, labels = plt.gca().get_legend_handles_labels()
            new_labels = []
            for label in labels:
                model_name_part = label.split(' (')[0]
                new_labels.append(legend_mapping.get(model_name_part, label))
            plt.legend(handles, new_labels, loc='lower right', fontsize=10)
        else:
            print(f"No data available to plot for behavior {behavior_id}.")
            return

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Deployment Scale (n queries, log scale)", fontsize=12)
    if aggregate_by_seed:
        plt.ylabel(f"Forecasted Worst-Query Risk (Q_p(n) - {seed_aggregation_method} over seeds, log scale)", fontsize=12)
    else:
        plt.ylabel("Forecasted Worst-Query Risk (Q_p(n) - median over behaviors, individual seeds, log scale)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f"forecast_comparison_{behavior_id}{'_aggregated' if aggregate_by_seed else '_individual_seeds'}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved comparison plot to {plot_filename}")
    plt.close()

def plot_model_averaged_forecasts(df: pd.DataFrame, 
                                  output_dir: str, 
                                  aggregate_by_seed: bool = True, 
                                  behavior_aggregation_method: str = "mean", 
                                  seed_aggregation_method: str = "median", 
                                  show_ci: bool = False):
    """Generates a plot showing model forecasts averaged over all behaviors.

    Args:
        df: DataFrame containing loaded results.
        output_dir: Directory to save the plot.
        aggregate_by_seed: If True, plot mean/median over seeds; otherwise, plot individual seeds.
        behavior_aggregation_method: 'mean' or 'median', how to aggregate Q_p(n) over behaviors for each seed.
        seed_aggregation_method: 'mean' or 'median', how to aggregate over seeds (after behavior aggregation).
        show_ci: If True, shows confidence intervals on plots where applicable (e.g., individual seed plots).
    """
    if df.empty:
        print("Input DataFrame is empty. Skipping model-averaged plot.")
        return

    qpn_cols = sorted([col for col in df.columns if col.startswith('Q_p(n=')],
                      key=lambda x: float(x.split('=')[-1][:-1]))
    if not qpn_cols:
        print("No Q_p(n) forecast columns found. Skipping model-averaged plot.")
        return

    id_vars_base = ['model_name', 'seed_id', 'behavior_id'] # Include behavior_id for initial melt

    plot_df_melt = df.melt(id_vars=id_vars_base, value_vars=qpn_cols,
                           var_name='forecast_scale', value_name='Q_p(n)')
    
    plot_df_melt['n'] = plot_df_melt['forecast_scale'].str.extract(r'Q_p\(n=(\d+\.?\d*[eE]?\d*)\)').astype(float)
    plot_df_melt.dropna(subset=['n', 'Q_p(n)'], inplace=True)

    if plot_df_melt.empty:
        print("No valid forecast points to plot after melting and initial processing. Skipping model-averaged plot.")
        return

    # Average Q_p(n) across all behaviors for each model, seed, and n
    behavior_agg_func = np.nanmean if behavior_aggregation_method == "mean" else np.nanmedian
    averaged_over_behaviors_df = plot_df_melt.groupby(['model_name', 'seed_id', 'n'])['Q_p(n)'].apply(behavior_agg_func).reset_index()

    # START OF EDIT: Print the DataFrame
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print("averaged_over_behaviors_df (spreadsheet view):")
        print(averaged_over_behaviors_df)
    # END OF EDIT

    hue_col = 'model_name' # Default for aggregate_by_seed=True
    # estimator_func for sns.lineplot if aggregating seeds within sns.lineplot
    # This is also the function for pre-aggregation if show_ci=False
    seed_estimator_func = np.nanmean if seed_aggregation_method == "mean" else np.nanmedian

    if not aggregate_by_seed:
        # This case means args.plot_individual_seeds is True
        averaged_over_behaviors_df['line_hue'] = averaged_over_behaviors_df['model_name'] + " (" + averaged_over_behaviors_df['seed_id'] + ")"
        hue_col = 'line_hue'
        # No further aggregation over seeds, estimator will be None (or default) in sns.lineplot
    
    if averaged_over_behaviors_df.empty:
        print("No data after averaging over behaviors. Skipping model-averaged plot.")
        return

    plt.figure(figsize=(12, 7))
    
    if aggregate_by_seed:
        if show_ci:
            # Plot per-seed (behavior-averaged) data, let sns aggregate seeds and compute CI
            sns.lineplot(data=averaged_over_behaviors_df, x='n', y='Q_p(n)', hue='model_name', marker='o',
                         estimator=seed_estimator_func, errorbar=('ci', 95), n_boot=10000)
        else:
            # No CI, pre-aggregate over seeds then plot
            averaged_over_seeds_df = averaged_over_behaviors_df.groupby(['model_name', 'n'])['Q_p(n)'].apply(seed_estimator_func).reset_index()
            # START OF EDIT: Print the DataFrame (already exists, moved for clarity if needed for this path)
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            #     print("averaged_over_seeds_df (spreadsheet view, no CI path):")
            #     print(averaged_over_seeds_df)
            # END OF EDIT
            sns.lineplot(data=averaged_over_seeds_df, x='n', y='Q_p(n)', hue='model_name', marker='o',
                         estimator=None, errorbar=None) # estimator=None as data is pre-aggregated
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [legend_mapping.get(label, label) for label in labels]
        plt.legend(handles, new_labels, loc='lower right', fontsize=10)
    else: # Plotting individual seed lines (after behavior averaging)
        # aggregate_by_seed is False, hue_col is 'line_hue'
        sns.lineplot(data=averaged_over_behaviors_df, x='n', y='Q_p(n)', hue=hue_col, style='model_name', marker='o', dashes=True)
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = []
        for label in labels:
            model_name_part = label.split(' (')[0]
            new_labels.append(legend_mapping.get(model_name_part, label))
        plt.legend(handles, new_labels, loc='lower right', fontsize=10)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Deployment Scale (n queries, log scale)", fontsize=12)
    if aggregate_by_seed:
        plt.ylabel(f"Forecasted Worst-Query Risk (Q_p(n) (log scale) \n{behavior_aggregation_method} over behaviors, then {seed_aggregation_method} over seeds, log scale)", fontsize=12)
    else:
        plt.ylabel(f"Forecasted Worst-Query Risk (Q_p(n) - {behavior_aggregation_method} over behaviors, individual seeds, log scale)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f"model_averaged_forecasts{'_aggregated' if aggregate_by_seed else '_individual_seeds'}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved model-averaged forecast plot to {plot_filename}")
    print(f"\n\nIMPORTANT NOTE: For this plot, Q_p(n) is first aggregated over BEHAVIORS using {behavior_aggregation_method.upper()} for each seed. "
          f"Then, if 'aggregated' in filename, it is further aggregated over SEEDS using {seed_aggregation_method.upper()}.")
    plt.close()

def generate_qpn_summary_table(df: pd.DataFrame, output_dir: str, seed_aggregation_method: str = "median"):
    """Generates a CSV table summarizing Q_p(n) forecast statistics.
    Calculates a central tendency (mean or median), 5th/95th percentiles, min, and max for each Q_p(n)
    across seeds, grouped by model_name and behavior_id.
    The central tendency is determined by seed_aggregation_method.
    """
    qpn_cols = sorted([col for col in df.columns if col.startswith('Q_p(n=')],
                      key=lambda x: float(x.split('=')[-1][:-1]))

    if not qpn_cols:
        print("No Q_p(n) columns found for summary table. Skipping Q_p(n) table generation.")
        return

    def q05(x): return x.quantile(0.05)
    def q95(x): return x.quantile(0.95)

    central_tendency_func = np.mean if seed_aggregation_method == "mean" else np.median
    central_tendency_name = seed_aggregation_method

    agg_funcs = {
        col: [
            (central_tendency_name, central_tendency_func),
            "min", "max", q05, q95
        ] for col in qpn_cols
    }
    
    summary_df = df.groupby(['model_name', 'behavior_id'], as_index=False).agg(agg_funcs)
    
    if isinstance(summary_df.columns, pd.MultiIndex):
        new_cols = []
        for col_tuple in summary_df.columns: # col_tuple is like ('model_name', '') or ('Q_p(n=1000)', 'mean')
            col_top = col_tuple[0]
            col_stat = col_tuple[1]
            if col_top in ['model_name', 'behavior_id']:
                 if col_stat == '': # For model_name, behavior_id themselves
                    new_cols.append(col_top)
                 else: # Should not happen for model_name, behavior_id if as_index=False
                    new_cols.append(f"{col_top}_{col_stat}")
            else: # For Q_p(n) columns
                # For named aggregations like ('mean', np.nanmean), col_stat is 'mean'
                # For string shorthands like "min", col_stat is "min"
                # For lambda functions, col_stat can be "<lambda>" or specific names if assigned
                stat_name = col_stat
                if hasattr(col_stat, '__name__'): # Handles np.nanmean, np.nanmedian, q05, q95 if they have __name__
                    stat_name = col_stat.__name__
                    if stat_name == 'q05': stat_name = 'q05' # Keep it short
                    elif stat_name == 'q95': stat_name = 'q95'
                elif isinstance(col_stat, str): # Already a string like 'min', 'max', or the name from tuple
                    stat_name = col_stat

                # Standardize potential variations from numpy
                if stat_name == 'nanmean': stat_name = 'mean'
                if stat_name == 'nanmedian': stat_name = 'median'
                if stat_name == 'amin': stat_name = 'min'
                if stat_name == 'amax': stat_name = 'max'
                
                new_cols.append(f"{col_top}_{stat_name}")
        summary_df.columns = new_cols

    output_path = os.path.join(output_dir, "qpn_summary_by_model_behavior.csv")
    try:
        summary_df.to_csv(output_path, index=False, float_format="%.3e")
        print(f"Saved Q_p(n) summary table to {output_path}")
    except Exception as e:
        print(f"Error saving Q_p(n) summary table: {e}")
        print("Dataframe columns:", summary_df.columns)
        print("Dataframe head:", summary_df.head())


def generate_r2_summary_table(df: pd.DataFrame, output_dir: str, k_fit_analysis_active: bool = False):
    """Generates a CSV table summarizing Gumbel fit R-squared statistics.
    Calculates median, 5th/95th percentiles, min, and max for 'fit_r2'
    across seeds, grouped by model_name and behavior_id.
    Uses 'fit_r2_refit' if k_fit_analysis_active and column exists, otherwise 'fit_r2'.
    """
    r2_col_to_use = 'fit_r2'
    if k_fit_analysis_active and 'fit_r2_refit' in df.columns:
        print("R2 Table: Using 'fit_r2_refit' column due to active k_fit_analysis.")
        r2_col_to_use = 'fit_r2_refit'
    elif 'fit_r2' not in df.columns:
        print("Column 'fit_r2' (and 'fit_r2_refit' if applicable) not found. Skipping R-squared summary table generation.")
        return
    else:
        print(f"R2 Table: Using '{r2_col_to_use}' column.")


    if r2_col_to_use not in df.columns or df[r2_col_to_use].isnull().all():
        print(f"Column '{r2_col_to_use}' not found or contains all NaNs. Skipping R-squared summary table generation.")
        return

    def q05(x): return x.quantile(0.05)
    def q95(x): return x.quantile(0.95)

    r2_summary = df.groupby(['model_name', 'behavior_id'], as_index=False)[r2_col_to_use].agg(
        mean_r2='mean',
        min_r2='min',
        max_r2='max',
        q05_r2=q05,
        q95_r2=q95
    )

    output_path = os.path.join(output_dir, "r2_summary_by_model_behavior.csv")
    r2_summary.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved R-squared summary table to {output_path}")

def generate_model_type_averaged_risk_summary_csv(df: pd.DataFrame, output_dir: str, behavior_aggregation_method: str = "mean", seed_aggregation_method: str = "median"):
    """
    Generates a CSV file summarizing the forecasted worst-query risk for each model type,
    for specific n values. Risk is first aggregated over behaviors (using behavior_aggregation_method),
    then aggregated over model seeds (using seed_aggregation_method).
    """
    if df.empty:
        print("Input DataFrame is empty. Skipping model-type averaged risk summary CSV generation.")
        return

    target_n_values = [1000, 10000, 100000, 1000000]
    qpn_cols_to_average = [f'Q_p(n={n_val})' for n_val in target_n_values]

    missing_cols = [col for col in qpn_cols_to_average if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing Q_p(n) columns for averaging: {missing_cols}. Skipping model-type averaged risk summary CSV.")
        # Attempt to use available columns among the targets
        qpn_cols_to_average = [col for col in qpn_cols_to_average if col in df.columns]
        if not qpn_cols_to_average:
            print("No target Q_p(n) columns are available. Cannot generate summary.")
            return

    # Melt the DataFrame to long format
    id_vars = ['model_name', 'seed_id', 'behavior_id']
    # Ensure id_vars exist, otherwise can't melt properly for averaging
    missing_id_vars = [var for var in id_vars if var not in df.columns]
    if missing_id_vars:
        print(f"Warning: Missing ID variables for melting: {missing_id_vars}. This might affect averaging. Proceeding with available ID vars.")
        id_vars = [var for var in id_vars if var in df.columns]
        if 'model_name' not in id_vars : # model_name is crucial
            print(f"Error: 'model_name' is missing, cannot generate model-type averaged summary.")
            return

    try:
        melted_df = df.melt(id_vars=id_vars, value_vars=qpn_cols_to_average,
                            var_name='forecast_scale', value_name='Q_p(n)')
    except KeyError as e:
        print(f"Error during melting, likely due to missing columns: {e}. Skipping CSV generation.")
        print(f"  DataFrame columns: {df.columns.tolist()}")
        print(f"  id_vars for melt: {id_vars}")
        print(f"  value_vars for melt: {qpn_cols_to_average}")
        return
        
    if melted_df.empty:
        print("Melted DataFrame is empty. Skipping model-type averaged risk summary CSV generation.")
        return


    melted_df['n'] = melted_df['forecast_scale'].str.extract(r'Q_p\(n=(\d+)\)').astype(int)
    melted_df.dropna(subset=['Q_p(n)', 'n'], inplace=True)

    if melted_df.empty:
        print("No valid data after extracting n and dropping NaNs. Skipping CSV.")
        return

    # Step 1: Aggregate over behaviors for each model_name, seed_id, n
    behavior_agg_func = np.nanmean if behavior_aggregation_method == "mean" else np.nanmedian
    qpn_behavior_aggregated = melted_df.groupby(['model_name', 'seed_id', 'n'])['Q_p(n)'].apply(behavior_agg_func).reset_index()

    if qpn_behavior_aggregated.empty:
        print("No data after aggregating over behaviors. Skipping CSV generation.")
        return
        
    # Step 2: Aggregate over seeds for each model_name, n
    seed_agg_func = np.nanmean if seed_aggregation_method == "mean" else np.nanmedian
    averaged_risk = qpn_behavior_aggregated.groupby(['model_name', 'n'])['Q_p(n)'].apply(seed_agg_func).reset_index()

    if averaged_risk.empty:
        print("No data after aggregating over seeds. Skipping CSV generation.")
        return

    try:
        summary_pivot_df = averaged_risk.pivot(index='model_name', columns='n', values='Q_p(n)')
    except Exception as e:
        print(f"Error pivoting the table: {e}")
        print("Data that was attempted to be pivoted:")
        print(averaged_risk.head())
        return

    output_path = os.path.join(output_dir, "model_type_averaged_risk_summary.csv")
    try:
        summary_pivot_df.to_csv(output_path, float_format="%.3e")
        print(f"Saved model-type averaged risk summary table to {output_path}")
    except Exception as e:
        print(f"Error saving model-type averaged risk summary CSV: {e}")
        print("Pivoted DataFrame head:")
        print(summary_pivot_df.head())

def main():
    """Main function to run the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing experiment results')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output plots and tables')
    parser.add_argument('--prefix', type=str, default="", help='Prefix filter for summary files')
    parser.add_argument('--k_fit', type=int, default=10, help='Number of top scores to use for Gumbel fitting')
    parser.add_argument('--seed_agg', type=str, default="median", choices=["mean", "median"], 
                        help='Method to aggregate across seeds (mean or median)')
    parser.add_argument('--behavior_agg', type=str, default="mean", choices=["mean", "median"], 
                        help='Method to aggregate across behaviors (mean or median)')
    parser.add_argument('--show_ci', action='store_true', help='Show confidence intervals on plots')
    parser.add_argument('--test_fit_function', action='store_true', help='Run test for _fit_gumbel_from_all_valid_scores function')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test_fit_function:
        test_fit_gumbel_function()
        return

    print(f"Loading results from {args.base_dir} with prefix '{args.prefix}'...")
    df = load_results(args.base_dir, args.prefix, args.k_fit)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    print(f"Refitting Gumbel parameters with k_fit={args.k_fit}...")
    df = perform_gumbel_refitting(df, args.k_fit)
    
    # Generate plots for each behavior
    behaviors = df['behavior_id'].unique()
    print(f"Generating plots for {len(behaviors)} behaviors...")
    
    for behavior_id in behaviors:
        print(f"\nGenerating plots for behavior: {behavior_id}")
        plot_forecast_comparison(df, behavior_id, args.output_dir, True, True, args.seed_agg, args.show_ci)
        
    # Generate model-averaged plots
    print("\nGenerating model-averaged forecast plots...")
    plot_model_averaged_forecasts(df, args.output_dir, True, args.behavior_agg, args.seed_agg, args.show_ci)
    
    # Generate summary tables
    print("\nGenerating Q_p(n) summary table...")
    generate_qpn_summary_table(df, args.output_dir, args.seed_agg)
    
    print("\nGenerating R² summary table...")
    generate_r2_summary_table(df, args.output_dir, True)
    
    print("\nGenerating model type averaged risk summary CSV...")
    generate_model_type_averaged_risk_summary_csv(df, args.output_dir, args.behavior_agg, args.seed_agg)
    
    print("\nAnalysis complete! Results saved to", args.output_dir)

def test_fit_gumbel_function():
    """Test function to verify that _fit_gumbel_from_all_valid_scores works correctly."""
    print("\n=== Testing _fit_gumbel_from_all_valid_scores function ===")
    
    # Test case 1: Simple ascending list of psi values
    test_psi_values = [-5.0, -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1, -4.0]
    k_fit = 5
    a, b, r = _fit_gumbel_from_all_valid_scores(test_psi_values, k_fit)
    print(f"Test case 1 (k_fit={k_fit}):")
    print(f"  a={a:.6f}, b={b:.6f}, r={r:.6f}")
    print(f"  Selected psi values (top {k_fit}): {test_psi_values[-k_fit:][::-1]}")
    
    # Test case 2: Using a larger k_fit
    k_fit = 10
    a, b, r = _fit_gumbel_from_all_valid_scores(test_psi_values, k_fit)
    print(f"\nTest case 2 (k_fit={k_fit}):")
    print(f"  a={a:.6f}, b={b:.6f}, r={r:.6f}")
    print(f"  Selected psi values (top {k_fit}): {test_psi_values[-k_fit:][::-1]}")
    
    # Test case 3: Using real data from the example file
    try:
        # Path to a sample visualization data file
        sample_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "results/nobootstrap_top4_fit/13-05-2025/main-q-prop/s1/drugs_keyword/drugs_keyword-logprob_target_seq-visualization-data.json")
        
        if os.path.exists(sample_file_path):
            with open(sample_file_path, 'r') as f:
                data = json.load(f)
                
            if "all_valid_scores" in data and "psi_values" in data["all_valid_scores"]:
                real_psi_values = data["all_valid_scores"]["psi_values"]
                print(f"\nTest case 3 (real data, {len(real_psi_values)} values):")
                
                for test_k_fit in [4, 10, 20]:
                    a, b, r = _fit_gumbel_from_all_valid_scores(real_psi_values, test_k_fit)
                    print(f"  k_fit={test_k_fit}: a={a:.6f}, b={b:.6f}, r={r:.6f}")
                
                # Compare with the original implementation in perform_gumbel_refitting
                print("\n  Comparison with original implementation:")
                m = len(real_psi_values)
                test_k_fit = 10  # Use the same k_fit for both implementations
                
                # Get results from our new function
                a_new, b_new, r_new = _fit_gumbel_from_all_valid_scores(real_psi_values, test_k_fit)
                
                # Implement the original logic for comparison
                actual_k_to_use = min(test_k_fit, m)
                
                selected_psi = real_psi_values[-actual_k_to_use:]
                selected_psi = selected_psi[::-1]  # Reverse
                
                ranks = np.arange(1, actual_k_to_use + 1)
                selected_log_surv = np.log(ranks / m).tolist()
                
                valid_indices = [idx for idx, (p_val, l_val) in enumerate(zip(selected_psi, selected_log_surv)) 
                                if pd.notna(p_val) and pd.notna(l_val) and np.isfinite(p_val) and np.isfinite(l_val)]
                
                final_selected_psi = [selected_psi[j] for j in valid_indices]
                final_selected_log_surv = [selected_log_surv[j] for j in valid_indices]
                
                slope, intercept, r_value, _, _ = linregress(final_selected_psi, final_selected_log_surv)
                
                print(f"  Original implementation: a={slope:.6f}, b={intercept:.6f}, r={r_value:.6f}")
                print(f"  New implementation:      a={a_new:.6f}, b={b_new:.6f}, r={r_new:.6f}")
                print(f"  Difference:              a={abs(slope-a_new):.10f}, b={abs(intercept-b_new):.10f}, r={abs(r_value-r_new):.10f}")
            else:
                print("\nTest case 3: Could not find all_valid_scores.psi_values in the sample file.")
        else:
            print(f"\nTest case 3: Sample file not found at {sample_file_path}")
    except Exception as e:
        print(f"\nTest case 3: Error loading or processing sample file: {e}")
    
    # Test case 4: Edge cases
    print("\nTest case 4: Edge cases")
    
    # Empty list
    a, b, r = _fit_gumbel_from_all_valid_scores([], 5)
    print(f"  Empty list: a={a}, b={b}, r={r}")
    
    # List with only one element
    a, b, r = _fit_gumbel_from_all_valid_scores([-4.5], 5)
    print(f"  Single element: a={a}, b={b}, r={r}")
    
    # k_fit = 1 (should fail as we need at least 2 points)
    a, b, r = _fit_gumbel_from_all_valid_scores(test_psi_values, 1)
    print(f"  k_fit=1: a={a}, b={b}, r={r}")
    
    # List with NaN values
    test_with_nan = [-5.0, -4.9, np.nan, -4.7, -4.6]
    a, b, r = _fit_gumbel_from_all_valid_scores(test_with_nan, 3)
    print(f"  List with NaN: a={a}, b={b}, r={r}")
    
    print("\n=== Test complete ===")

if __name__ == "__main__":
    main() 