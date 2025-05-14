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
    for root, dirs, files in os.walk(base_results_dir):
        for filename in files:
            if filename.endswith("_summary.json") and filename.startswith(specific_run_prefix):
                filepath = os.path.join(root, filename)
                try:
                    # Extract model_name and seed from path
                    # Path relative to base_results_dir
                    relative_path = os.path.relpath(filepath, base_results_dir)
                    path_parts = relative_path.split(os.sep)

                    model_name = "N/A"
                    seed_id = "N/A"
                    # behavior_dirname = "N/A" # The actual behavior_id is read from JSON

                    if len(path_parts) >= 3: # model_name/seed/behavior_folder/file.json
                        model_name = path_parts[0]
                        seed_id = path_parts[1]
                        # behavior_dirname = path_parts[2] # This is the folder name for behavior

                    with open(filepath, 'r') as f:
                        summary_data = json.load(f)
                        flat_data = {
                            "run_name": filename.replace("_summary.json", ""), # Original run_name based on file
                            "model_name": model_name,
                            "seed_id": seed_id,
                            "filepath": filepath, # For debugging/reference
                            "model_path": summary_data.get("model_path", "N/A"),
                            "behavior_id": summary_data.get("behavior_id", "N/A"),
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
                                print(f"Warning: Could not parse n from forecast key '{key}' in {filename}. Value: {n_str}")
                            except Exception as e:
                                print(f"Warning: Error processing forecast key '{key}' in {filename}: {e}")

                        all_data.append(flat_data)
                except Exception as e:
                    print(f"Error loading or processing {filepath}: {e}")

    if not all_data:
        print(f"Warning: No summary files found in {base_results_dir} with prefix '{specific_run_prefix}' matching the expected directory structure.")
        return pd.DataFrame()

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

    # Save combined results to CSV
    combined_csv_path = os.path.join(output_dir, "combined_forecasting_results.csv")
    results_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined results table to {combined_csv_path}")

    # Generate plots for each unique behavior found
    unique_behaviors = results_df['behavior_id'].unique()
    print(f"Found results for behaviors: {list(unique_behaviors)}")
    for behavior in unique_behaviors:
        if pd.notna(behavior):
            plot_forecast_comparison(results_df, behavior, output_dir, aggregate_by_seed=not args.plot_individual_seeds)

    print("Analysis complete.")

if __name__ == "__main__":
    main() 