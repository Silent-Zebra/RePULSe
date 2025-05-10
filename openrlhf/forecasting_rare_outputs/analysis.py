import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results(results_dir: str, run_prefix: str = "") -> pd.DataFrame:
    """Loads summary JSON files from experiments into a pandas DataFrame."""
    all_data = []
    for filename in os.listdir(results_dir):
        if filename.endswith("_summary.json") and filename.startswith(run_prefix):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    summary_data = json.load(f)
                    # Flatten relevant data for DataFrame
                    flat_data = {
                        "run_name": filename.replace("_summary.json", ""),
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
                        # Extract n from key like "Q_p(10000)"
                        try:
                            n_str = key.split('(')[-1].split(')')[0]
                            n_val = int(float(n_str))
                            flat_data[f"Q_p(n={n_val})"] = value
                        except:
                            print(f"Warning: Could not parse n from forecast key '{key}' in {filename}")

                    all_data.append(flat_data)
            except Exception as e:
                print(f"Error loading or processing {filename}: {e}")

    if not all_data:
        print(f"Warning: No summary files found in {results_dir} with prefix '{run_prefix}'")
        # Return empty DataFrame with expected columns if possible
        return pd.DataFrame()

    return pd.DataFrame(all_data)

def plot_forecast_comparison(df: pd.DataFrame, behavior_id: str, output_dir: str):
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
    qpn_cols = [col for col in plot_df.columns if col.startswith('Q_p(n=')]
    if not qpn_cols:
        print(f"No Q_p(n) forecast columns found for behavior {behavior_id}. Skipping plot.")
        return

    id_vars = ['run_name', 'model_path'] # Identify runs
    plot_df_melt = plot_df.melt(id_vars=id_vars, value_vars=qpn_cols, var_name='forecast_scale', value_name='Q_p(n)')

    # Extract n from the 'forecast_scale' column name
    plot_df_melt['n'] = plot_df_melt['forecast_scale'].str.extract(r'Q_p\(n=(\d+\.?\d*[eE]?\d*)\)').astype(float)
    plot_df_melt.dropna(subset=['n', 'Q_p(n)'], inplace=True) # Drop rows where parsing failed or Q_p(n) is NaN

    if plot_df_melt.empty:
        print(f"No valid forecast points to plot for behavior {behavior_id}.")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df_melt, x='n', y='Q_p(n)', hue='run_name', marker='o')

    # Customize plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Deployment Scale (n queries, log scale)")
    plt.ylabel("Forecasted Worst-Query Risk (Q_p(n), log scale)")
    plt.title(f"Forecasted Harmfulness Risk Comparison\nBehavior: {behavior_id}")
    plt.legend(title="Model/Run", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(output_dir, f"forecast_comparison_{behavior_id}.png")
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

    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load all results
    print(f"Loading results from: {args.results_dir} with prefix '{args.run_prefix}'")
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
            plot_forecast_comparison(results_df, behavior, output_dir)

    print("Analysis complete.")

if __name__ == "__main__":
    main() 