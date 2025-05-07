import matplotlib
matplotlib.use('PDF') # THIS MUST BE AT THE START OF THE CODE (before other imports)!!!!
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# from plot_utils import *

import torch

import scipy.stats as stats

n_epochs = 80
policy_updates_per_epoch = 5

ylabel_bad = "Log Total Prob of Bad Word"

from scipy.stats import norm
# from scipy import stats

def make_frontier(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7,
    aggregate_seeds=False, alpha_error=0.3, threshold=-5
):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(len(labels)):
        print(labels[i])
        print(results_list[i])
        if isinstance(results_list[i], tuple):
            to_plot_x = results_list[i][0]
            to_plot_y = results_list[i][1]

        else:

            tensor_list = results_list[i]
            x_results = []
            y_results = []
            # y_successes = []
            # y_totals = []
            for t in tensor_list:
                x_results.append(t.cpu().numpy().mean())
                print((t.cpu().numpy() < threshold).sum())
                print(t.cpu().numpy().shape)
                y_results.append((t.cpu().numpy() < threshold).mean())


            to_plot_x = np.stack(x_results)
            to_plot_y = np.stack(y_results)

        if not aggregate_seeds:
            plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i], marker=marker_list[i])

        else: # aggregate_seeds:
            n_samples_x = to_plot_x.shape[0]
            n_samples_y = to_plot_y.shape[0]

            t_value_x = stats.t.ppf(0.975, df=n_samples_x - 1)
            t_value_y = stats.t.ppf(0.975, df=n_samples_y - 1)

            print(to_plot_x)
            print(to_plot_y)
            x_std = np.std(to_plot_x, axis=0, ddof=1)
            y_std = np.std(to_plot_y, axis=0, ddof=1)

            x_conf = t_value_x * x_std / np.sqrt(n_samples_x)
            y_conf = t_value_y * y_std / np.sqrt(n_samples_y)

            print(t_value_x)
            print(t_value_y)

            print(x_conf)
            print(y_conf)

            print(to_plot_x.shape[0])
            print(to_plot_x.shape)

            to_plot_x = to_plot_x.mean(axis=0)
            to_plot_y = to_plot_y.mean(axis=0)

            plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i],
                        marker=marker_list[i])
            plt.errorbar(
                to_plot_x,
                to_plot_y,
                xerr=x_conf,
                yerr=y_conf,
                fmt='',
                ecolor=color_list[i],
                alpha=alpha_error,
                capsize=2,
            )

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    # plt.legend(loc='upper right', bbox_to_anchor=(1, 0.6), fontsize=7)
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)


def make_frontier(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7,
    aggregate_seeds=False, alpha_error=0.3, threshold=-5
):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(len(labels)):
        print(labels[i])
        print(results_list[i])
        if isinstance(results_list[i], tuple):
            to_plot_x = results_list[i][0]
            to_plot_y = results_list[i][1]

        else:

            tensor_list = results_list[i]
            x_results = []
            y_results = []
            for t in tensor_list:
                x_results.append(t.cpu().numpy().mean())
                print((t.cpu().numpy() < threshold).sum())
                print(t.cpu().numpy().shape)
                y_results.append((t.cpu().numpy() < threshold).mean())


            to_plot_x = np.stack(x_results)
            to_plot_y = np.stack(y_results)

        if not aggregate_seeds:
            plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i], marker=marker_list[i])

        else: # aggregate_seeds:
            n_samples_x = to_plot_x.shape[0]
            n_samples_y = to_plot_y.shape[0]

            to_plot_x_mean = to_plot_x.mean(axis=0)
            to_plot_y_mean = to_plot_y.mean(axis=0)

            # --- X-axis: T-Distribution Confidence Intervals ---
            t_value_x = stats.t.ppf(0.975, df=n_samples_x - 1)
            x_std = np.std(to_plot_x, axis=0, ddof=1)
            x_conf = t_value_x * x_std / np.sqrt(n_samples_x)

            # --- Y-axis: Wald Approximation Confidence Intervals ---
            z_value = stats.norm.ppf(0.975)
            y_std = np.std(to_plot_y, axis=0, ddof=1)
            y_conf = z_value * y_std / np.sqrt(n_samples_y)

            print(f"Label: {labels[i]}")
            print("--- X-axis (T-Distribution) ---")
            print(f"n_samples_x: {n_samples_x}")
            print(f"t_value_x: {t_value_x}")
            print(f"x_std: {x_std}")
            print(f"x_conf: {x_conf}")
            print("--- Y-axis (Wald Approximation) ---")
            print(f"n_samples_y: {n_samples_y}")
            print(f"z_value: {z_value}")
            print(f"y_std: {y_std}")
            print(f"y_conf: {y_conf}")
            print("-----------------------------------")

            plt.scatter(to_plot_x_mean, to_plot_y_mean, label=labels[i], c=color_list[i],
                        marker=marker_list[i])
            plt.errorbar(
                to_plot_x_mean,
                to_plot_y_mean,
                xerr=x_conf,
                yerr=y_conf,
                fmt='',
                ecolor=color_list[i],
                alpha=alpha_error,
                capsize=2,
            )

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)



def make_frontier_bootstrap(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7,
    aggregate_seeds=False, alpha_error=0.3, threshold=-5,
    n_bootstrap_draws=1000  # Still needed for Y variable
):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(len(labels)):
        print(f"Processing: {labels[i]}")

        if isinstance(results_list[i], tuple):
            to_plot_x_agg = results_list[i][0]
            to_plot_y_agg = results_list[i][1]
            if not aggregate_seeds:
                plt.scatter(to_plot_x_agg, to_plot_y_agg, label=labels[i], c=color_list[i], marker=marker_list[i])
            else:
                plt.scatter(to_plot_x_agg, to_plot_y_agg, label=labels[i], c=color_list[i], marker=marker_list[i])
                print(
                    f"Warning: aggregate_seeds is True for {labels[i]}, but data is a pre-aggregated tuple. Plotting point without error bars.")
        else:
            tensor_list = results_list[i]
            x_results_per_seed = []
            y_results_per_seed = []

            if not tensor_list:
                print(f"Warning: Empty tensor_list for {labels[i]}. Skipping.")
                continue

            for t_idx, t in enumerate(tensor_list):
                x_results_per_seed.append(t.cpu().numpy().mean())
                y_results_per_seed.append((t.cpu().numpy() < threshold).mean())

            x_values_all_seeds = np.array(x_results_per_seed)
            y_values_all_seeds = np.array(y_results_per_seed)

            if not aggregate_seeds:
                plt.scatter(x_values_all_seeds, y_values_all_seeds, label=labels[i], c=color_list[i],
                            marker=marker_list[i])
            else:  # aggregate_seeds is True
                n_seeds = x_values_all_seeds.shape[0]

                if n_seeds == 0:
                    print(f"Warning: No seed data for {labels[i]} after processing. Skipping.")
                    continue

                x_observed_mean = np.mean(x_values_all_seeds)
                y_observed_mean = np.mean(y_values_all_seeds)

                print(
                    f"  {labels[i]}: Num seeds = {n_seeds}, Observed X-mean = {x_observed_mean:.4f}, Observed Y-mean (Prob Bad) = {y_observed_mean:.4f}")

                x_conf_final = None  # For t-distribution symmetric error
                y_err_bootstrap_final = None  # For bootstrap asymmetric error array

                if n_seeds < 2:
                    print(f"  Warning: Only {n_seeds} seed for {labels[i]}. Plotting mean without error bars.")
                else:
                    alpha_level_for_ci = 0.05  # For a 95% CI

                    # --- X Variable: t-distribution CI ---
                    t_value_x = stats.t.ppf(1 - (alpha_level_for_ci / 2), df=n_seeds - 1)  # Corrected for 0.975
                    x_std_dev = np.std(x_values_all_seeds, ddof=1)
                    x_conf_final = t_value_x * x_std_dev / np.sqrt(n_seeds)  # This is the margin of error
                    print(f"  {labels[i]}: X t-dist ME = +/-{x_conf_final:.4f}")

                    # --- Y Variable: Bootstrap CI ---
                    bootstrap_y_means = []
                    for _ in range(n_bootstrap_draws):
                        resample_indices = np.random.choice(n_seeds, size=n_seeds, replace=True)
                        bootstrap_sample_y = y_values_all_seeds[resample_indices]
                        bootstrap_y_means.append(np.mean(bootstrap_sample_y))

                    y_ci_lower = np.percentile(bootstrap_y_means, (alpha_level_for_ci / 2) * 100)
                    y_ci_upper = np.percentile(bootstrap_y_means, (1 - alpha_level_for_ci / 2) * 100)

                    y_err_bootstrap_final = np.array([[y_observed_mean - y_ci_lower], [y_ci_upper - y_observed_mean]])
                    y_err_bootstrap_final[y_err_bootstrap_final < 0] = 0  # Ensure error deltas are non-negative
                    print(
                        f"  {labels[i]}: Y Bootstrap CI ({((1 - alpha_level_for_ci) * 100):.0f}%) = [{y_ci_lower:.4f}, {y_ci_upper:.4f}]")

                # Plot the observed mean
                plt.scatter(x_observed_mean, y_observed_mean, label=labels[i], c=color_list[i],
                            marker=marker_list[i])

                # Plot error bars if they were computed (i.e., n_seeds >= 2)
                if x_conf_final is not None and y_err_bootstrap_final is not None:
                    plt.errorbar(
                        x_observed_mean,
                        y_observed_mean,
                        xerr=x_conf_final,  # Symmetric error for X
                        yerr=y_err_bootstrap_final,  # Potentially asymmetric error for Y
                        fmt='',
                        ecolor=color_list[i],
                        alpha=alpha_error,
                        capsize=2,
                    )

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)
    print(f"Figure saved to {figname}")



def make_frontier_bootstrap(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7,
    aggregate_seeds=False, alpha_error=0.3, threshold=-5,
    n_bootstrap_draws=1000  # Added parameter for number of bootstrap draws
):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(len(labels)):
        print(f"Processing: {labels[i]}")  # Changed to f-string for clarity
        # print(results_list[i]) # Original print, can be verbose

        if isinstance(results_list[i], tuple):
            # This case seems to assume pre-processed x and y, no aggregation
            # If aggregation is needed here too, this part would also need modification
            to_plot_x_agg = results_list[i][0]  # Assuming these are already aggregated means
            to_plot_y_agg = results_list[i][1]
            # No error bars defined for this case in original or new logic
            # If this branch should also have error bars with bootstrapping, it needs similar logic
            # based on the raw data that would lead to these tuple values.
            # For now, plotting as a single point if not aggregate_seeds, or if aggregate_seeds
            # and it's a tuple, it plots the pre-aggregated point without error bars.
            if not aggregate_seeds:
                plt.scatter(to_plot_x_agg, to_plot_y_agg, label=labels[i], c=color_list[i], marker=marker_list[i])
            else:  # aggregate_seeds is True but data is a pre-aggregated tuple
                # Plotting the point, but can't compute CI without per-seed data
                plt.scatter(to_plot_x_agg, to_plot_y_agg, label=labels[i], c=color_list[i], marker=marker_list[i])
                print(
                    f"Warning: aggregate_seeds is True for {labels[i]}, but data is a pre-aggregated tuple. Plotting point without error bars.")


        else:  # results_list[i] is a list of tensors (per-seed data)
            tensor_list = results_list[i]
            x_results_per_seed = []
            y_results_per_seed = []

            if not tensor_list:  # Handle empty tensor_list
                print(f"Warning: Empty tensor_list for {labels[i]}. Skipping.")
                continue

            for t_idx, t in enumerate(tensor_list):
                x_results_per_seed.append(t.cpu().numpy().mean())
                # print(f"  Seed {t_idx+1} raw bad outputs count: {(t.cpu().numpy() < threshold).sum()}")
                # print(f"  Seed {t_idx+1} raw shape: {t.cpu().numpy().shape}")
                y_results_per_seed.append((t.cpu().numpy() < threshold).mean())

            # Convert lists of per-seed results to numpy arrays
            x_values_all_seeds = np.array(x_results_per_seed)
            y_values_all_seeds = np.array(y_results_per_seed)

            if not aggregate_seeds:
                plt.scatter(x_values_all_seeds, y_values_all_seeds, label=labels[i], c=color_list[i],
                            marker=marker_list[i])
            else:  # aggregate_seeds is True, perform bootstrap
                n_seeds = x_values_all_seeds.shape[0]

                if n_seeds == 0:  # Should be caught by empty tensor_list earlier
                    print(f"Warning: No seed data for {labels[i]} after processing. Skipping error bars.")
                    continue

                # Calculate the observed mean from the original seed data
                x_observed_mean = np.mean(x_values_all_seeds)
                y_observed_mean = np.mean(y_values_all_seeds)

                print(
                    f"  {labels[i]}: Num seeds = {n_seeds}, Observed X-mean = {x_observed_mean:.4f}, Observed Y-mean (Prob Bad) = {y_observed_mean:.4f}")

                if n_seeds < 2:
                    print(f"  Warning: Only {n_seeds} seed for {labels[i]}. Plotting mean without error bars.")
                    x_err_bootstrap = None  # No error bar
                    y_err_bootstrap = None  # No error bar
                else:
                    alpha_level_for_ci = 0.05  # For a 95% CI

                    # Bootstrap for X
                    bootstrap_x_means = []
                    for _ in range(n_bootstrap_draws):
                        resample_indices = np.random.choice(n_seeds, size=n_seeds, replace=True)
                        bootstrap_sample_x = x_values_all_seeds[resample_indices]
                        bootstrap_x_means.append(np.mean(bootstrap_sample_x))

                    # Calculate percentile CI for X
                    x_ci_lower = np.percentile(bootstrap_x_means, (alpha_level_for_ci / 2) * 100)
                    x_ci_upper = np.percentile(bootstrap_x_means, (1 - alpha_level_for_ci / 2) * 100)
                    # xerr for errorbar: [negative_error_delta, positive_error_delta]
                    x_err_bootstrap = np.array([[x_observed_mean - x_ci_lower], [x_ci_upper - x_observed_mean]])
                    # Ensure error deltas are non-negative (can occur if observed mean is outside bootstrap CI)
                    x_err_bootstrap[x_err_bootstrap < 0] = 0

                    # Bootstrap for Y
                    bootstrap_y_means = []
                    for _ in range(n_bootstrap_draws):
                        resample_indices = np.random.choice(n_seeds, size=n_seeds, replace=True)
                        bootstrap_sample_y = y_values_all_seeds[resample_indices]
                        bootstrap_y_means.append(np.mean(bootstrap_sample_y))

                    # Calculate percentile CI for Y
                    y_ci_lower = np.percentile(bootstrap_y_means, (alpha_level_for_ci / 2) * 100)
                    y_ci_upper = np.percentile(bootstrap_y_means, (1 - alpha_level_for_ci / 2) * 100)
                    y_err_bootstrap = np.array([[y_observed_mean - y_ci_lower], [y_ci_upper - y_observed_mean]])
                    y_err_bootstrap[y_err_bootstrap < 0] = 0

                    print(
                        f"  {labels[i]}: X CI ({((1 - alpha_level_for_ci) * 100):.0f}%) = [{x_ci_lower:.4f}, {x_ci_upper:.4f}], Y CI = [{y_ci_lower:.4f}, {y_ci_upper:.4f}]")

                # Plot the observed mean
                plt.scatter(x_observed_mean, y_observed_mean, label=labels[i], c=color_list[i],
                            marker=marker_list[i])

                # Plot error bars if they were computed
                if x_err_bootstrap is not None and y_err_bootstrap is not None:
                    plt.errorbar(
                        x_observed_mean,
                        y_observed_mean,
                        xerr=x_err_bootstrap,
                        yerr=y_err_bootstrap,
                        fmt='',  # No line connecting points, marker is from scatter
                        ecolor=color_list[i],
                        alpha=alpha_error,
                        capsize=2,
                    )

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)
    print(f"Figure saved to {figname}")

do_load = True # False
if do_load:
    n_epochs = 100
    load_prefixes_to_use = [
        [
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s1",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s2",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s3",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s4",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s5",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s6",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s7",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epochs1_schedconstant_alr0.0001_clr0.0001_clossmse_policy_s8",

        ],
        [
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s4",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5",
        ],
        [
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s2",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s3",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s4",
            "rewards_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s5",
        ]
    ]
    # load_prefixes_to_use = [
    #     # [
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
    #     # ],
    #     [
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
    #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
    #         "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s4",
    #         "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5",
    #
    #     ],
    #     # [
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s4",
    #     # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5",
    #     # ],
    #     [
    #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
    #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
    #     ],
    #     [
    #         "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
    #         "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s2",
    #         "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s3",
    #     ],
    #     # [
    #     #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
    #     #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s2",
    #     #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s3",
    #     # ],
    #     # [
    #     #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #     #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
    #     #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
    #     # ]
    #
    # ]

    # labels = [
    #     r"a0.001 alr0.0001_blr3e-05",
    #     r"a",
    #     r"a",
    #     r"a",
    #     r"a",
    #     r"a",
    #     r"a",
    #     r"a",
    #     r"a",
    # ]

    labels = ['_'.join(a[0].split('len20_')[-1].split('_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_')).split('_policy_psi_q_p_s_t')[0] for a in load_prefixes_to_use]

    figname_modifier = "len20_05_06_bootstrap"


    results_list = [[] for i in range(len(load_prefixes_to_use))]

    for i in range(len(load_prefixes_to_use)):

        load_prefixes = load_prefixes_to_use[i]


        for load_prefix in load_prefixes:
            print(load_prefix)
            x = torch.load(f'./info/{load_prefix}')

            results_list[i].append(x)

    # results_list.append((np.array([-0.07,1.67,-0.44,0.46,-0.37]), np.array([0.000348259, 9.95025E-06, 0.001333333, 0.000378109, 0])))
    # labels.append("REINFORCE Baseline")


linestyle_list = ['solid'] * 30

color_list = [
    'xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:purple', 'xkcd:red', 'xkcd:black',  'xkcd:gray',  'xkcd:light brown', 'xkcd:pink',
    'xkcd:gold', 'xkcd:teal', 'xkcd:dark brown', 'xkcd:magenta'
]
marker_list = ["o", "o", "o", "o", "o", "o", "o", "o", "o", "v", "v", "^", "^", "x", "x", "x", "x", "D", "P", "P", "P"]

# xlimlow = 2
# xlimhigh = 10
fontsize = 6

make_frontier_bootstrap(
    xlabel="Average Reward", ylabel=ylabel_bad,
    figname=f"{figname_modifier}_frontier",
    labels=labels, results_list=results_list,
    color_list=color_list, marker_list=marker_list,
    # xlimlow=xlimlow, xlimhigh=xlimhigh,
    fontsize=fontsize, aggregate_seeds=True
)

