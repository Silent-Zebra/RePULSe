import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
            if (filename.endswith("_summary.json") or filename.endswith("-summary.json")) and filename.startswith(specific_run_prefix):
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
                        
                        flat_data = {
                            "run_name": filename.replace("_summary.json", "").replace("-summary.json", ""), # Original run_name based on file
                            "model_name": parsed_model_type, # Using the parsed model_type
                            "seed_id": parsed_seed_id,       # Using the parsed seed_id
                            "filepath": filepath, # For debugging/reference
                            "model_path": summary_data.get("model_path", "N/A"),
                            "behavior_id": behavior_id,
                            "elicitation_method": summary_data.get("elicitation_method", "N/A"),
                            "m_eval_size": summary_data.get("num_eval_queries_sampled", np.nan),
                            "num_valid_p": summary_data.get("num_valid_p_elicits", np.nan),
                            "fit_a": summary_data.get("gumbel_fit_params", {}).get("a_slope", np.nan),
                            "fit_b": summary_data.get("gumbel_fit_params", {}).get("b_intercept", np.nan),
                            "fit_r2": summary_data.get("gumbel_fit_params", {}).get("r_squared", np.nan),
                            "fit_top_k": summary_data.get("gumbel_fit_params", {}).get("top_k_used", np.nan),
                        }
                        # Add forecasted risks Q_p(n)
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

    plt.figure(figsize=(12, 7)) # Slightly larger figure
    
    if aggregate_by_seed:
        # Plot median across seeds, with error bands for min/max or std dev
        sns.lineplot(data=plot_df_melt, x='n', y='Q_p(n)', hue='model_name', marker='o', 
                     estimator=np.median, errorbar=('ci', 95)) # 'ci' for confidence interval
        legend_title = "Model (Median over seeds)"
    else:
        # Plot all individual seeds
        plot_df_melt['line_hue'] = plot_df_melt['model_name'] + " (" + plot_df_melt['seed_id'] + ")"
        sns.lineplot(data=plot_df_melt, x='n', y='Q_p(n)', hue='line_hue', marker='o', style='model_name', dashes=True)
        legend_title = "Model (Seed)"


    # Customize plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Deployment Scale (n queries, log scale)")
    plt.ylabel("Forecasted Worst-Query Risk (Q_p(n), log scale)")
    plt.title(f"Forecasted Harmfulness Risk Comparison\nBehavior: {behavior_id}")
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Save the plot
    plot_filename = os.path.join(output_dir, f"forecast_comparison_{behavior_id}{'_aggregated' if aggregate_by_seed else '_individual_seeds'}.png")
    plt.savefig(plot_filename)
    print(f"Saved comparison plot to {plot_filename}")
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

    agg_funcs = {col: ["mean","median", "min", "max", q05, q95] for col in qpn_cols}
    
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
        median_r2='median',
        min_r2='min',
        max_r2='max',
        q05_r2=q05,
        q95_r2=q95
    )

    output_path = os.path.join(output_dir, "r2_summary_by_model_behavior.csv")
    r2_summary.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved R-squared summary table to {output_path}")

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
    eval_size_filter = 100 if args.filter_eval_size is None else args.filter_eval_size
    filtered_df = results_df[results_df['m_eval_size'] >= eval_size_filter]
    
    if len(filtered_df) < len(results_df):
        print(f"\nFiltered data to only include rows with m_eval_size = {eval_size_filter}")
        print(f"Original dataset: {len(results_df)} records")
        print(f"Filtered dataset: {len(filtered_df)} records")
        
        # If filtering resulted in empty DataFrame, show what values are available
        if filtered_df.empty:
            print(f"Warning: No records found with m_eval_size = {eval_size_filter}")
            available_sizes = sorted(results_df['m_eval_size'].unique())
            print(f"Available m_eval_size values: {available_sizes}")
            return
            
        # Replace the original DataFrame with the filtered one
        results_df = filtered_df
    else:
        print(f"\nAll records already have m_eval_size = {eval_size_filter}, no filtering needed")
    
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

    # Generate plots for each unique behavior found
    unique_behaviors = results_df['behavior_id'].unique()
    print(f"Found results for behaviors: {list(unique_behaviors)}")
    for behavior in unique_behaviors:
        if pd.notna(behavior):
            plot_forecast_comparison(results_df, behavior, output_dir, aggregate_by_seed=not args.plot_individual_seeds)

    print("Analysis complete.")

if __name__ == "__main__":
    main() 