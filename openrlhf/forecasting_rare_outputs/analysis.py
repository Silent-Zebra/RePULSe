import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Attempt to import forecast_worst_query_risk, handle if module not found for standalone execution
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

def load_results(base_results_dir: str, specific_run_prefix: str = "") -> pd.DataFrame:
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
            # Check for both naming conventions: '_summary.json' and '-summary.json'
            # Options:
            # logprob_target_keyword_in_target_seq-summary.json
            # logprob_target_seq-summary.json
            # repeated_sampling-summary.json
            if filename.endswith("logprob_target_seq-summary.json") and filename.startswith(specific_run_prefix):
                filepath = os.path.join(root, filename)
                print(f"Processing file: {filepath}")
                processed_files += 1
                
                try:
                    # Extract model_type and seed_id from path
                    # base_results_dir is the "dated directory" (e.g., .../results/10-05-2025/)
                    # Expected structure: dated_directory / model_type / seed / behavior_directory / xxx_summary.json
                    relative_path = os.path.relpath(filepath, base_results_dir)
                    path_parts = relative_path.split(os.sep)
                    print(f"  Path parts: {path_parts}")

                    # Initialize parsed model_type and seed_id
                    parsed_model_type = "N/A"
                    parsed_seed_id = "N/A"
                    
                    # If path is deep enough (e.g., model_type/seed/...), extract them.
                    # For example:
                    # - model_type/seed/file.json -> len(path_parts) is 3
                    # - model_type/seed/behavior_dir/file.json -> len(path_parts) is 4
                    if len(path_parts) >= 3: 
                        parsed_model_type = path_parts[0] # This is the 'model_type' from the directory structure
                        parsed_seed_id = path_parts[1]    # This is the 'seed' from the directory structure
                        print(f"  Extracted model_type: {parsed_model_type}, seed_id: {parsed_seed_id}")
                    
                    with open(filepath, 'r') as f:
                        summary_data = json.load(f)
                        
                        # Extract behavior_id - try both from the JSON and from the path if available
                        behavior_id_from_json = summary_data.get("behavior_id", "N/A")
                        behavior_id_from_path = path_parts[2] if len(path_parts) >= 4 else "N/A"
                        
                        # Use JSON behavior_id if available, otherwise try from path
                        behavior_id = behavior_id_from_json if behavior_id_from_json != "N/A" else behavior_id_from_path
                        
                        print(f"  Behavior ID from JSON: {behavior_id_from_json}")
                        print(f"  Behavior ID from path: {behavior_id_from_path}")
                        print(f"  Using behavior ID: {behavior_id}")
                        
                        # Extract data based on the new bootstrapped summary.json structure
                        bootstrap_summary = summary_data.get("gumbel_fit_params_bootstrap_summary", {})
                        
                        flat_data = {
                            "run_name": filename.replace("_summary.json", "").replace("-summary.json", ""),
                            "model_name": parsed_model_type,
                            "seed_id": parsed_seed_id,
                            "filepath": filepath,
                            "model_path": summary_data.get("pretrain", "N/A"), # Changed from model_path for consistency
                            "behavior_id": behavior_id,
                            "elicitation_method": summary_data.get("elicitation_method", "N/A"),
                            # Bootstrap specific metrics
                            "num_bootstrap_samples_requested": summary_data.get("num_bootstrap_samples_requested", np.nan),
                            "num_successful_gumbel_fits": summary_data.get("num_successful_gumbel_fits", np.nan),
                            "m_eval_size_per_bootstrap": summary_data.get("evaluation_set_size_m_per_bootstrap", np.nan), # New name for clarity
                            "mean_num_valid_p_per_bootstrap": summary_data.get("mean_num_valid_p_elicits_per_bootstrap", np.nan), # New name for clarity

                            # Mean Gumbel fit parameters from bootstrap
                            "fit_a": bootstrap_summary.get("mean_a_slope", np.nan),
                            "fit_b": bootstrap_summary.get("mean_b_intercept", np.nan),
                            "fit_r2": bootstrap_summary.get("mean_r_squared_approx", np.nan), # Using approximated R^2 from mean r_value
                            "fit_top_k": bootstrap_summary.get("mean_top_k_actually_used", np.nan),
                            
                            # Standard deviations of Gumbel fit parameters if needed for advanced analysis (optional for now)
                            "std_fit_a": bootstrap_summary.get("std_a_slope", np.nan),
                            "std_fit_b": bootstrap_summary.get("std_b_intercept", np.nan),
                            "std_fit_r_value": bootstrap_summary.get("std_r_value", np.nan),

                            # Raw bootstrap parameters for CI calculations
                            "bootstrap_a_slopes": summary_data.get("bootstrap_iterations_data", {}).get("a_slopes", []),
                            "bootstrap_b_intercepts": summary_data.get("bootstrap_iterations_data", {}).get("b_intercepts", []),
                        }
                        
                        # Add forecasted risks Q_p(n) from mean parameters
                        forecasts = summary_data.get("forecasted_worst_query_risks_from_mean_params", {})
                        if not forecasts and "forecasted_worst_query_risks" in summary_data:
                            # Fallback for potentially old files during transition, can be removed later
                            print(f"  Warning: Using fallback 'forecasted_worst_query_risks' for {filepath}")
                            forecasts = summary_data.get("forecasted_worst_query_risks", {})

                        for key, value in forecasts.items():
                            try:
                                n_str = key.split('(')[-1].split(')')[0]
                                n_val = int(float(n_str)) # Handle potential float strings like "1e5"
                                flat_data[f"Q_p(n={n_val})"] = value
                            except ValueError: # Handle cases where n_str might not be a simple int/float
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
        
    # Before returning, print some statistics about the loaded data
    if all_data:
        df = pd.DataFrame(all_data)
        print("\nLoaded data statistics:")
        print(f"Number of unique models: {df['model_name'].nunique()}")
        print(f"Number of unique seeds: {df['seed_id'].nunique()}")
        print(f"Seeds found: {sorted(df['seed_id'].unique())}")
        print(f"Number of unique behaviors: {df['behavior_id'].nunique()}")
        
        # Print seed count by model
        seed_counts = df.groupby('model_name')['seed_id'].nunique()
        print("\nNumber of seeds per model:")
        for model, count in seed_counts.items():
            print(f"  {model}: {count} seeds")

    return pd.DataFrame(all_data)

def plot_forecast_comparison(df: pd.DataFrame, behavior_id: str, output_dir: str, aggregate_by_seed: bool = True):
    """Generates a plot comparing forecasted Q_p(n) vs n for different models.

    Args:
        df: DataFrame containing loaded results (from load_results).
        behavior_id: The specific behavior ID to plot.
        output_dir: Directory to save the plot.
    """
    # TODO: Implement plotting logic
    # Example using seaborn/matplotlib:

    # Filter data for the specific behavior
    plot_df = df[df['behavior_id'] == behavior_id].copy()
    if plot_df.empty:
        print(f"No data found for behavior ID: {behavior_id}. Skipping plot.")
        return

    # Extract Q_p(n=...) columns and melt for plotting
    qpn_cols = sorted([col for col in plot_df.columns if col.startswith('Q_p(n=')], 
                      key=lambda x: float(x.split('=')[-1][:-1])) # Sort by n value
    if not qpn_cols:
        print(f"No Q_p(n) forecast columns found for behavior {behavior_id}. Skipping plot.")
        return

    id_vars_base = ['model_name', 'seed_id', 'model_path']
    if not aggregate_by_seed:
        # If not aggregating, use a more unique identifier for hue if seed_id is not enough (e.g. model_name + seed_id)
        plot_df['run_identifier'] = plot_df['model_name'] + "_" + plot_df['seed_id']
        id_vars = ['run_identifier'] + id_vars_base
        hue_col = 'run_identifier'
    else:
        id_vars = ['model_name'] # Aggregating by model_name
        hue_col = 'model_name'

    plot_df_melt = plot_df.melt(id_vars=id_vars_base, value_vars=qpn_cols, var_name='forecast_scale', value_name='Q_p(n)')

    # Extract n from the 'forecast_scale' column name
    plot_df_melt['n'] = plot_df_melt['forecast_scale'].str.extract(r'Q_p\(n=(\d+\.?\d*[eE]?\d*)\)').astype(float)
    plot_df_melt.dropna(subset=['n', 'Q_p(n)'], inplace=True)

    if plot_df_melt.empty:
        print(f"No valid forecast points to plot for behavior {behavior_id}.")
        return

    # Prepare a list to collect data for plotting, including CIs
    plot_data_list = []

    unique_runs = plot_df['run_identifier'].unique() if not aggregate_by_seed else plot_df['model_name'].unique()
    hue_col_name = 'run_identifier' if not aggregate_by_seed else 'model_name'

    for run_id in unique_runs:
        if not aggregate_by_seed:
            run_specific_df = plot_df[plot_df['run_identifier'] == run_id]
        else:
            # If aggregating by seed, we'd typically average/median the mean parameters first
            # For now, let's assume aggregate_by_seed=True uses the existing seaborn CI over seeds.
            # The complex CI logic is for aggregate_by_seed=False.
            pass

        if not aggregate_by_seed and not run_specific_df.empty:
            # Get the first row for this run to access list of a's and b's
            # (assuming they are the same for all rows of this run_identifier before melting)
            # This requires that bootstrap_a_slopes and bootstrap_b_intercepts are properly loaded per run_name/seed_id
            # We should get these from the original plot_df, not the melted one.
            original_run_row = df[(df['model_name'] + "_" + df['seed_id']) == run_id].iloc[0]
            
            bootstrap_as = original_run_row.get('bootstrap_a_slopes', [])
            bootstrap_bs = original_run_row.get('bootstrap_b_intercepts', [])

            if not isinstance(bootstrap_as, list) or not isinstance(bootstrap_bs, list) or not bootstrap_as or not bootstrap_bs or len(bootstrap_as) != len(bootstrap_bs):
                print(f"Warning: Missing or mismatched bootstrap_a/b_slopes for run {run_id}. Skipping CI calculation for this run.")
                # Fallback to plotting the mean Q_p(n) from melted data for this run
                run_melted_data = plot_df_melt[plot_df_melt[hue_col_name] == run_id]
                for _, row in run_melted_data.iterrows():
                    plot_data_list.append({
                        hue_col_name: run_id,
                        'n': row['n'],
                        'Q_p(n)_mean': row['Q_p(n)'], # This is Q_p from mean params
                        'Q_p(n)_lower': row['Q_p(n)'], # No CI, so lower/upper are same as mean
                        'Q_p(n)_upper': row['Q_p(n)'],
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
                        'Q_p(n)_mean': row['Q_p(n)'], 
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
                    qpn_mean = np.nanmean(bootstrapped_qpn_for_n) # Changed from nanmedian
                    qpn_lower = np.nanpercentile(bootstrapped_qpn_for_n, 5)
                    qpn_upper = np.nanpercentile(bootstrapped_qpn_for_n, 95)
                    plot_data_list.append({
                        hue_col_name: run_id,
                        'n': n_val,
                        'Q_p(n)_mean': qpn_mean, # Changed from Q_p(n)_median
                        'Q_p(n)_lower': qpn_lower,
                        'Q_p(n)_upper': qpn_upper,
                        'model_name': original_run_row['model_name'] # for consistent styling by model
                    })
    
    if not aggregate_by_seed and plot_data_list:
        ci_plot_df = pd.DataFrame(plot_data_list)
        if ci_plot_df.empty:
            print(f"No data for CI plot for behavior {behavior_id}.")
            return
            
        plt.figure(figsize=(14, 8)) # Adjusted size for potentially more complex plot
        
        # Plotting each run identifier (model_seed) with its CI
        for run_id_val, group in ci_plot_df.groupby(hue_col_name):
            group = group.sort_values(by='n')
            model_name_of_run = group['model_name'].iloc[0]
            display_label = legend_mapping.get(model_name_of_run, run_id_val)
            plt.plot(group['n'], group['Q_p(n)_mean'], marker='o', label=display_label, linestyle='--', markersize=5) # Changed from Q_p(n)_median
            plt.fill_between(group['n'], group['Q_p(n)_lower'], group['Q_p(n)_upper'], alpha=0.2)
        
        plt.legend(loc='lower right', fontsize=10)
    elif aggregate_by_seed:
        # Fallback to original seaborn plotting if aggregating seeds (uses Q_p(n) from mean params)
        # Or, one could aggregate the CIs, but that's more complex.
        # For now, let seaborn handle CIs across seeds based on the Q_p(n) from mean params.
        plot_df_melted_for_agg = plot_df.melt(id_vars=['model_name', 'seed_id'], value_vars=qpn_cols, var_name='forecast_scale', value_name='Q_p(n)')
        plot_df_melted_for_agg['n'] = plot_df_melted_for_agg['forecast_scale'].str.extract(r'Q_p\(n=(\d+\.?\d*[eE]?\d*)\)').astype(float)
        plot_df_melted_for_agg.dropna(subset=['n', 'Q_p(n)'], inplace=True)
        
        if plot_df_melted_for_agg.empty:
            print(f"No data to plot for behavior {behavior_id} when aggregate_by_seed=True.")
            return
            
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=plot_df_melted_for_agg, x='n', y='Q_p(n)', hue='model_name', marker='o', 
                     estimator=np.mean, errorbar=None)
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [legend_mapping.get(label, label) for label in labels]
        plt.legend(handles, new_labels, loc='lower right', fontsize=10)
    else:
        # This case might be hit if not aggregate_by_seed but plot_data_list is empty
        print(f"No data processed for plotting for behavior {behavior_id}. This might be due to missing bootstrap data or other issues.")
        # Optionally, fall back to the old plotting method without CIs if desired.
        # For now, just return if no CI data was generated.
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

    # Customize plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Deployment Scale (n queries, log scale)", fontsize=12)
    plt.ylabel("Forecasted Worst-Query Risk (Q_p(n), log scale)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"forecast_comparison_{behavior_id}{'_aggregated' if aggregate_by_seed else '_individual_seeds'}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved comparison plot to {plot_filename}")
    plt.close()

def plot_model_averaged_forecasts(df: pd.DataFrame, output_dir: str, aggregate_by_seed: bool = True):
    """Generates a plot showing model forecasts averaged over all behaviors.

    Args:
        df: DataFrame containing loaded results.
        output_dir: Directory to save the plot.
        aggregate_by_seed: If True, plot mean over seeds; otherwise, plot individual seeds.
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
    if aggregate_by_seed:
        # Group by model_name, seed_id, and n, then average Q_p(n) over behaviors
        # Then, the lineplot will take the mean of these per-seed averages for each model
        averaged_df = plot_df_melt.groupby(['model_name', 'seed_id', 'n'])['Q_p(n)'].mean().reset_index()
        hue_col = 'model_name'
        estimator = np.mean
    else:
        # Group by model_name, seed_id, and n, then average Q_p(n) over behaviors
        averaged_df = plot_df_melt.groupby(['model_name', 'seed_id', 'n'])['Q_p(n)'].mean().reset_index()
        averaged_df['line_hue'] = averaged_df['model_name'] + " (" + averaged_df['seed_id'] + ")"
        hue_col = 'line_hue'
        estimator = None # Plot individual lines

    if averaged_df.empty:
        print("No data after averaging over behaviors. Skipping model-averaged plot.")
        return

    plt.figure(figsize=(12, 7))
    
    if estimator: # Using mean for aggregated view
        sns.lineplot(data=averaged_df, x='n', y='Q_p(n)', hue=hue_col, marker='o',
                     estimator=estimator, errorbar=None)
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [legend_mapping.get(label, label) for label in labels]
        plt.legend(handles, new_labels, loc='lower right', fontsize=10)
    else: # Plotting individual lines (already averaged over behavior)
        sns.lineplot(data=averaged_df, x='n', y='Q_p(n)', hue=hue_col, style='model_name', marker='o', dashes=True)
        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = []
        for label in labels:
            model_name_part = label.split(' (')[0]
            new_labels.append(legend_mapping.get(model_name_part, label))
        plt.legend(handles, new_labels, loc='lower right', fontsize=10)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Deployment Scale (n queries, log scale)", fontsize=12)
    plt.ylabel("Forecasted Worst-Query Risk (Q_p(n) averaged over behaviors, log scale)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f"model_averaged_forecasts{'_aggregated' if aggregate_by_seed else '_individual_seeds'}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved model-averaged forecast plot to {plot_filename}")
    plt.close()

def generate_qpn_summary_table(df: pd.DataFrame, output_dir: str):
    """Generates a CSV table summarizing Q_p(n) forecast statistics.
    Calculates median, 5th/95th percentiles, min, and max for each Q_p(n)
    across seeds, grouped by model_name and behavior_id.
    """
    qpn_cols = sorted([col for col in df.columns if col.startswith('Q_p(n=')],
                      key=lambda x: float(x.split('=')[-1][:-1]))

    if not qpn_cols:
        print("No Q_p(n) columns found for summary table. Skipping Q_p(n) table generation.")
        return

    # Define aggregation functions
    def q05(x): return x.quantile(0.05)
    def q95(x): return x.quantile(0.95)

    agg_funcs = {col: ["mean", "min", "max", q05, q95] for col in qpn_cols} # Removed "median"
    
    # Group by model and behavior, then aggregate
    summary_df = df.groupby(['model_name', 'behavior_id'], as_index=False).agg(agg_funcs)

    # Flatten MultiIndex columns
    # Original columns from agg: ('Q_p(n=10000)', 'median'), ('Q_p(n=10000)', 'amin'), etc.
    # New columns: 'Q_p(n=10000)_median', 'Q_p(n=10000)_min', etc.
    
    # Correctly access the levels for renaming based on Pandas version behavior with as_index=False
    if isinstance(summary_df.columns, pd.MultiIndex):
         # When as_index=False, groupby columns are regular columns.
         # The value columns are MultiIndex.
        new_cols = []
        for col_top, col_stat in summary_df.columns:
            if col_top in ['model_name', 'behavior_id']: # These are not part of the MultiIndex from agg
                 if col_stat == '': # pandas can add an empty string as the second level for non-aggregated columns
                    new_cols.append(col_top)
                 else: # Should not happen if model_name/behavior_id are correctly handled by as_index=False
                    new_cols.append(f"{col_top}_{col_stat}")
            else:
                new_cols.append(f"{col_top}_{col_stat}")
        summary_df.columns = new_cols
    else: # If columns are already flat (e.g. older pandas or simpler structure)
        # This case might occur if only one agg func or one qpn_col (less likely)
        # For safety, ensure renaming only happens if structure is as expected
        pass # Columns should already be 'model_name', 'behavior_id', 'Q_p(n=...)_median', ...

    # Clean up column names if groupby columns were part of the MultiIndex (if as_index=True was used)
    # This part might need adjustment based on how pandas structures columns with as_index=False
    # With as_index=False, 'model_name' and 'behavior_id' are already flat.
    # The aggregated columns are MultiIndex. We need to flatten those.
    
    # Let's rebuild columns more explicitly
    processed_columns = []
    # Handle the columns that were used for grouping first
    if 'model_name' in df.columns: processed_columns.append('model_name')
    if 'behavior_id' in df.columns: processed_columns.append('behavior_id')

    # Now, handle the aggregated Q_p(n) columns
    # The agg_funcs dictionary keys are the original Q_p(n) column names
    # The values are lists of functions, like [np.median, np.min, ...]
    # Pandas creates column names like ('Q_p(n=...)', 'median'), ('Q_p(n=...)', 'amin') for these
    
    # Iterate through the original columns in summary_df to build new names
    # This assumes summary_df starts with 'model_name', 'behavior_id', then the aggregated multi-index columns
    
    flat_cols = ['model_name', 'behavior_id']
    for q_col in qpn_cols: # These are the original names like 'Q_p(n=10000)'
        for stat_func in agg_funcs[q_col]:
            stat_name = stat_func.__name__ if hasattr(stat_func, '__name__') else str(stat_func)
            if stat_name == '<lambda>': # For q05, q95 if they don't get a proper name
                if '0.05' in str(stat_func): stat_name = 'q05'
                elif '0.95' in str(stat_func): stat_name = 'q95'
            elif stat_name == 'amin': stat_name = 'min' # numpy.min's name
            elif stat_name == 'amax': stat_name = 'max' # numpy.max's name
            
            flat_cols.append(f"{q_col}_{stat_name}")
    
    # After aggregation, summary_df columns would be ('model_name',''), ('behavior_id',''), ('Q_p(n=...)','median'), etc.
    # We need to ensure we select the correct part.
    # The .agg() result with as_index=False will have 'model_name' and 'behavior_id' as simple columns.
    # The other columns will be MultiIndex: (('Q_p(n=10000)', 'median'), ('Q_p(n=10000)', 'amin'), ...)
    # So we need to join these multi-level column names.
    
    current_cols = summary_df.columns.tolist()
    new_column_names = []
    for col in current_cols:
        if isinstance(col, tuple):
            if col[0] in ['model_name', 'behavior_id'] and col[1] == '': # For grouping cols if they end up in tuple
                new_column_names.append(col[0])
            else:
                stat_name = col[1]
                if stat_name == 'amin': stat_name = 'min'
                elif stat_name == 'amax': stat_name = 'max'
                new_column_names.append(f"{col[0]}_{stat_name}")
        else: # Already flat, like 'model_name'
            new_column_names.append(col)
    summary_df.columns = new_column_names

    output_path = os.path.join(output_dir, "qpn_summary_by_model_behavior.csv")
    try:
        summary_df.to_csv(output_path, index=False, float_format="%.3e")
        print(f"Saved Q_p(n) summary table to {output_path}")
    except Exception as e:
        print(f"Error saving Q_p(n) summary table: {e}")
        print("Dataframe columns:", summary_df.columns)
        print("Dataframe head:", summary_df.head())


def generate_r2_summary_table(df: pd.DataFrame, output_dir: str):
    """Generates a CSV table summarizing Gumbel fit R-squared statistics.
    Calculates median, 5th/95th percentiles, min, and max for 'fit_r2'
    across seeds, grouped by model_name and behavior_id.
    """
    if 'fit_r2' not in df.columns:
        print("Column 'fit_r2' not found. Skipping R-squared summary table generation.")
        return

    def q05(x): return x.quantile(0.05)
    def q95(x): return x.quantile(0.95)

    r2_summary = df.groupby(['model_name', 'behavior_id'], as_index=False)['fit_r2'].agg(
        mean_r2='mean', # Changed from median_r2='median'
        min_r2='min',
        max_r2='max',
        q05_r2=q05,
        q95_r2=q95
    )

    output_path = os.path.join(output_dir, "r2_summary_by_model_behavior.csv")
    r2_summary.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved R-squared summary table to {output_path}")

def generate_model_type_averaged_risk_summary_csv(df: pd.DataFrame, output_dir: str):
    """
    Generates a CSV file summarizing the forecasted worst-query risk for each model type,
    averaged across all model seeds and behaviors, for specific n values.
    """
    if df.empty:
        print("Input DataFrame is empty. Skipping model-type averaged risk summary CSV generation.")
        return

    target_n_values = [1000, 10000, 100000, 1000000]
    qpn_cols_to_average = [f'Q_p(n={n_val})' for n_val in target_n_values]

    # Check if all target Q_p(n) columns exist
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

    # Extract n from 'forecast_scale'
    melted_df['n'] = melted_df['forecast_scale'].str.extract(r'Q_p\(n=(\d+)\)').astype(int)
    melted_df.dropna(subset=['Q_p(n)', 'n'], inplace=True) # Drop rows where Q_p(n) or n is NaN

    if melted_df.empty:
        print("No valid data after extracting n and dropping NaNs. Skipping CSV.")
        return

    # Average Q_p(n) across all seeds and behaviors for each model_name and n
    # Group by model_name and n, then calculate mean Q_p(n)
    averaged_risk = melted_df.groupby(['model_name', 'n'])['Q_p(n)'].mean().reset_index()

    if averaged_risk.empty:
        print("No data after averaging. Skipping CSV generation.")
        return

    # Pivot the table to get model_name as index and n values as columns
    try:
        summary_pivot_df = averaged_risk.pivot(index='model_name', columns='n', values='Q_p(n)')
    except Exception as e:
        print(f"Error pivoting the table: {e}")
        print("Data that was attempted to be pivoted:")
        print(averaged_risk.head())
        return

    # Rename columns for clarity if desired, e.g., Q_p(n=1000) -> 1000
    # summary_pivot_df.columns = [f'Avg_Q_p(n={col})' for col in summary_pivot_df.columns]
    # Or keep it simple as just the n values, which is what pivot will produce for 'n' column.

    output_path = os.path.join(output_dir, "model_type_averaged_risk_summary.csv")
    try:
        summary_pivot_df.to_csv(output_path, float_format="%.3e")
        print(f"Saved model-type averaged risk summary table to {output_path}")
    except Exception as e:
        print(f"Error saving model-type averaged risk summary CSV: {e}")
        print("Pivoted DataFrame head:")
        print(summary_pivot_df.head())

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot Jones et al. forecasting results.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing the experiment summary (.json) files.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plots and analysis results. Defaults to results_dir.")
    parser.add_argument("--run_prefix", type=str, default="",
                        help="Only load summary files starting with this prefix.")
    # Add arguments to select specific behaviors or models for analysis if needed
    parser.add_argument("--plot_individual_seeds", action="store_true",
                        help="If set, plots will show individual seed runs instead of aggregating them.")
    parser.add_argument("--filter_eval_size", type=int, default=None,
                        help="If set, only include data points with this exact m_eval_size value.")

    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    # The main results_dir argument should now be the base directory for all models/seeds
    # e.g., openrlhf/forecasting_rare_outputs/results/13-03-2025/
    print(f"Loading results from base directory: {args.results_dir} with prefix '{args.run_prefix}'")
    results_df = load_results(args.results_dir, args.run_prefix)

    if results_df.empty:
        print("No results loaded. Exiting analysis.")
        return
        
    # Filter by evaluation size if specified
    # Ensure we are using the correct column name for filtering, which is now 'm_eval_size_per_bootstrap'
    eval_size_column = 'm_eval_size_per_bootstrap' 
    if eval_size_column not in results_df.columns and 'm_eval_size' in results_df.columns:
        # Fallback for older data that might still use 'm_eval_size'
        print(f"Warning: Column '{eval_size_column}' not found, falling back to 'm_eval_size' for filtering.")
        eval_size_column = 'm_eval_size'
    elif eval_size_column not in results_df.columns:
        print(f"Error: Cannot filter by evaluation size, column '{eval_size_column}' (or fallback 'm_eval_size') not found in loaded data. Skipping filtering.")
        # Proceed without filtering by eval size if column is missing
        pass # Or return, or raise error, depending on desired strictness
    else:
        default_eval_size_filter = 100 # Default from experiment_runner args
        eval_size_filter_to_use = args.filter_eval_size if args.filter_eval_size is not None else default_eval_size_filter
        
        # Check if the column contains valid numbers for comparison
        if pd.api.types.is_numeric_dtype(results_df[eval_size_column]):
            original_len = len(results_df)
            results_df = results_df[results_df[eval_size_column] >= eval_size_filter_to_use].copy() # Use .copy() to avoid SettingWithCopyWarning
            
            if len(results_df) < original_len:
                print(f"\nFiltered data to only include rows with {eval_size_column} >= {eval_size_filter_to_use}")
                print(f"Original dataset: {original_len} records")
                print(f"Filtered dataset: {len(results_df)} records")
                
                if results_df.empty:
                    print(f"Warning: No records found with {eval_size_column} >= {eval_size_filter_to_use}")
                    # available_sizes = sorted(results_df[eval_size_column].dropna().unique()) # This would be on the already empty df
                    # To show available sizes, we'd need the original df before filtering, which is complex here.
                    # For simplicity, just state that no records were found.
                    return # Stop if filtering made it empty
            else:
                print(f"\nAll records already meet {eval_size_column} >= {eval_size_filter_to_use} criterion, or no filtering applied due to args.")
        else:
            print(f"Warning: Column '{eval_size_column}' is not numeric. Skipping filtering by evaluation size.")

    if results_df.empty: # Check again if results_df became empty after potential filtering
        print("No results remaining after filtering. Exiting analysis.")
        return

    # Print information about unique behavior_ids after filtering
    unique_behaviors_filtered = sorted(results_df['behavior_id'].unique())
    print(f"\nFound {len(unique_behaviors_filtered)} unique behavior_ids after filtering:")
    for behavior in unique_behaviors_filtered:
        behavior_count = len(results_df[results_df['behavior_id'] == behavior])
        print(f"  - {behavior}: {behavior_count} records")

    # Save combined results to CSV
    combined_csv_path = os.path.join(output_dir, "combined_forecasting_results.csv")
    results_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined results table to {combined_csv_path}")

    # Generate summary tables
    generate_qpn_summary_table(results_df, output_dir)
    generate_r2_summary_table(results_df, output_dir)
    generate_model_type_averaged_risk_summary_csv(results_df, output_dir)

    # Generate plots for each unique behavior found
    unique_behaviors = results_df['behavior_id'].unique()
    print(f"Found results for behaviors: {list(unique_behaviors)}")
    for behavior in unique_behaviors:
        if pd.notna(behavior):
            plot_forecast_comparison(results_df, behavior, output_dir, aggregate_by_seed=not args.plot_individual_seeds)

    # Generate model-averaged plots
    print("\nGenerating model-averaged forecast plot...")
    plot_model_averaged_forecasts(results_df, output_dir, aggregate_by_seed=not args.plot_individual_seeds)

    print("Analysis complete.")

if __name__ == "__main__":
    main() 