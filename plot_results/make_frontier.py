import matplotlib

matplotlib.use('PDF') # THIS MUST BE AT THE START OF THE CODE (before other imports)!!!!
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from plot_utils import *

import torch

import scipy.stats as stats

n_epochs = 80
policy_updates_per_epoch = 5


from scipy.stats import norm
# from scipy import stats


def make_frontier_bootstrap(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7,
    aggregate_seeds=False, alpha_error=0.2, threshold=-5,
    n_bootstrap_draws=5000,  # Added parameter for number of bootstrap draws
    tuple_index=0, # 0 for rewards, 1 for returns (with kl penalty)
    tuple_index_gcg=1,
    compare_to_reference=False,
    gcg_results_list=None,
    gcg_results_type="attack_success",
    ylimlow=None, ylimhigh=None
):
    plt.clf()
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)


    for i in range(len(labels)):
        # print(f"Processing: {labels[i]}")  # Changed to f-string for clarity
        # print(results_list[i]) # Original print, can be verbose



        if isinstance(results_list[i], tuple):
            raise NotImplementedError # hasn't been used/tested in a while
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
                # print(t[0])
                # print(len(t[0]))
                # print(t[0][0].shape)
                if isinstance(t, tuple):
                    t, unmodified_rew = t[tuple_index], t[0]
                    if isinstance(t, list):
                        t = torch.cat(t)
                    if isinstance(unmodified_rew, list):
                        unmodified_rew = torch.cat(unmodified_rew)

                x_results_per_seed.append(t.float().cpu().numpy().mean())
                # print(f"  Seed {t_idx+1} raw bad outputs count: {(t.cpu().numpy() < threshold).sum()}")
                # print(f"  Seed {t_idx+1} raw shape: {t.cpu().numpy().shape}")
                if gcg_results_list is None:
                    y_results_per_seed.append((unmodified_rew.float().cpu().numpy() < threshold).mean())

            if gcg_results_list is not None:
                gcg_list = gcg_results_list[i]
                # print(gcg_list)
                if gcg_results_type == "attack_success":

                    for t_idx, t in enumerate(gcg_list):
                        # print(t)
                        print("proportion of successful attacks")
                        successful_attacks = (np.array(t[tuple_index_gcg]) > 0)
                        prop = successful_attacks.mean()
                        print(prop)
                        y_results_per_seed.append(prop)

                elif gcg_results_type == "log_probs":
                    y_results_per_seed = gcg_list

                else:
                    raise NotImplementedError



            # Convert lists of per-seed results to numpy arrays
            x_values_all_seeds = np.array(x_results_per_seed)
            y_values_all_seeds = np.array(y_results_per_seed)

            # print(x_values_all_seeds)
            # print(x_values_all_seeds.shape)
            # print(y_values_all_seeds.shape)

            if y_values_all_seeds.shape[0] < x_values_all_seeds.shape[0]:
                print("WARNING: IGNORING ADDITIONAL X VALUES")
                x_values_all_seeds = x_values_all_seeds[: y_values_all_seeds.shape[0]]
            elif y_values_all_seeds.shape[0] > x_values_all_seeds.shape[0]:
                print("WARNING: IGNORING ADDITIONAL Y VALUES")
                y_values_all_seeds = y_values_all_seeds[: x_values_all_seeds.shape[0]]


            # print(x_values_all_seeds)
            # print(y_values_all_seeds)


            if compare_to_reference:
                if i == 0:
                    reference_x = x_values_all_seeds
                    reference_y = y_values_all_seeds
                x_values_all_seeds = x_values_all_seeds - reference_x[:x_values_all_seeds.shape[0]]
                y_values_all_seeds = y_values_all_seeds - reference_y[:y_values_all_seeds.shape[0]]


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

                # print(
                #     f"  {labels[i]}: Num seeds = {n_seeds}, Observed X-mean = {x_observed_mean:.3f}, Observed Y-mean (Prob Bad) = {y_observed_mean:.6f}")

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

                    # print(
                    #     f"  {labels[i]}: X CI ({((1 - alpha_level_for_ci) * 100):.0f}%) = [{x_ci_lower:.3f}, {x_ci_upper:.3f}], Y CI = [{y_ci_lower:.6f}, {y_ci_upper:.6f}]")

                    print(
                        f"  {labels[i]}: X = {x_observed_mean:.3f} [{x_ci_lower:.3f}, {x_ci_upper:.3f}], Y = {y_observed_mean:.2f} [{y_ci_lower:.2f}, {y_ci_upper:.2f}]")

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
    if (ylimlow is not None) or (ylimhigh is not None):
        plt.ylim(ylimlow, ylimhigh)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.legend(fontsize=fontsize)
    plt.savefig(figname)
    print(f"Figure saved to {figname}")




figname_modifier = "dummy"
threshold = 0 # Check diff values here too

linestyle_list = ['solid'] * 30


color_list = [
    'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:green', 'xkcd:blue',
    'xkcd:black',  'xkcd:gray',  'xkcd:light brown',
    'xkcd:pink', 'xkcd:gold', 'xkcd:teal', 'xkcd:dark brown', 'xkcd:magenta',
    'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:green', 'xkcd:blue',
    'xkcd:black', 'xkcd:gray', 'xkcd:light brown',
    'xkcd:pink', 'xkcd:gold', 'xkcd:teal', 'xkcd:dark brown', 'xkcd:magenta',
    'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:green', 'xkcd:blue',
    'xkcd:black', 'xkcd:gray', 'xkcd:light brown',
    'xkcd:pink', 'xkcd:gold', 'xkcd:teal', 'xkcd:dark brown', 'xkcd:magenta',
]
marker_list = ["D", "x", "^", "o", "P", "v", "v", "v", "P", "o", "o",
               "P", "o", "o", "P", "^", "^", "P", "v", "v", "v",
               "v", "v", "^", "P", "v", "D", "v", "v", "x", "v", # 22 23 25 27 28 reinf, ours, baseprop, reinftransf ppo
               "v", "v", "v", "^", "^", "x", "x", "x", "x", "D",
               "P", "P", "P", "v", "v", "v", "^", "^", "x", "x", "x", "x", "D", "P", "P", "P"]


# xlimlow = 2
# xlimhigh = 10
fontsize = 11


compare_to_reference = False
if compare_to_reference:
    figname_modifier += "_comparetoref"


do_load = True # False


if do_load:
    n_epochs = 100


    load_prefixes_to_use = [
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epo1_epi4_schconstant_alr0.0001_clr0.0001_clossmse_policy_s1",
        #     1, 10),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 10),
        #
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 10),
        #
        # make_list(
        #     "info_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 10),

        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 10),

        # Stuff from 07-26
        make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epo1_epi2_schconstant_alr0.0001_clr0.0001_clossmse_policy_s1", 1, 10),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 10),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 10),
        make_list("info_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,10),





        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1", 1, 10),
        # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1", 1, 10),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 10),
        #
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
        #
        #
        # # Below did not really work well
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.002_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0002_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0005_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.005_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-50.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        #
        # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",1,10),
        # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",1,10),

        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # # make_list(
        # #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        # #     1, 5),
        # # make_list(
        # #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta3.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        # #     1, 5),
        # # make_list(
        # #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta5.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        # #     1, 5),
        # # make_list(
        # #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        # #     1, 5),
        # # make_list(
        # #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        # #     1, 5),
        # # make_list(
        # #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta3.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        # #     1, 5),
        # # make_list(
        # #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta5.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        # #     1, 5),

        # for x in $(ls /scratch/zhaostep/OpenRLHF/info/rlhfmultismol/ | grep neg_tr | grep info | grep epi1 | grep _s1); do echo make_list\(\"$x\", 1, 5\),; done

        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-10.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-20.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta0.3_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.003rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.005_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),
        # make_list(
        #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-30.0_harml_neg_training_a0.01rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        #     1, 5),
        make_list(
            "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.03_beta-40.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
            1, 5),


    ]

    # load_prefixes_to_use = [
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.1_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1", 1, 10),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1, 10),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.1_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1", 1, 10),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.1_harml_reinforce_a10.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1", 1, 10),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.1_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1, 10),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1", 1, 10),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1", 1, 10),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1, 10),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1, 5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
    #
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #
    #     # make_list("info_eval_rlhf_tu_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1", 1, 10),
    #
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epo1_epi3_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi3_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi3_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi3_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi3_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi3_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #     # make_list(
    #     #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #     #     1, 5),
    #
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.1_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #
    #
    # ]
    #

    # for x in $(ls info/rlhfmultitransform3 | grep epi8 | grep info_eval | grep _s1 | grep -v _s10); do echo make_list\(\"$x\",1,5\),; done
    # load_prefixes_to_use = [
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,20),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,20),
    #
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1", 1, 5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1", 1, 5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1", 1, 10),
    #     # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1", 1, 5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1", 1, 5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1", 1, 20),
    #     # # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1", 1, 5),
    #     #
    #     # # make_list("", 1, 5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-50.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-100.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",1,5),
    #
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi3_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1,5),
    #
    #     make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",1,10),
    #
    #     # make_list(
    #     #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.01_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
    #     #     1, 5),
    #     # make_list(
    #     #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.01_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
    #     #     1, 5),
    #     # make_list(
    #     #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
    #     #     1, 5),
    #
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #
    #     # make_list(
    #     #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.01_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1",
    #     #     1, 5),
    #     # make_list(
    #     #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-20.0_kl0.01_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
    #     #     1, 5),
    #
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #
    #     # for x in $(ls info/rlhfmulti | grep epi4 | grep kl0.01 | grep info_eval | grep _s1 | grep -v _s10); do echo make_list\(\"$x\",1,10\),; done
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #         1, 10),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #         1, 10),
    #     make_list(
    #         "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_s1",
    #         1, 10),
    #     # make_list(
    #     #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
    #     #     1, 10),
    #
    # ]

    # load_prefixes_to_use = [
    #     make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.03_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
    #     make_list(
    #         "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.03_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",
    #         1, 5),
    #
    #     make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1", 1, 5),
    #     make_list(
    #         "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",
    #         1, 5),
    # ]

    # load_prefixes_to_use = [
    #     make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1", 1, 5),
    #     make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1, 5),
    #
    # ]


    # New experiments
    load_prefixes_to_use = [
        make_list(
            "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta0.0_kl0.1_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",
            1, 5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1", 1, 5),
        make_list(
            "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",
            1, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-30.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-30.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 1),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 2) + make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1", 3, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.02_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-20.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-30.0_kl0.1_harml_neg_training_a0.005_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 1) + make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-30.0_harml_neg_training_a0.005_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",2,5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-5.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 2) + make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-5.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1", 3, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-5.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-7.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 5),

        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-05_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-3.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-05_blr3e-06_policy_psi_q_p_s_t_s1", 1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-3.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-5.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-05_blr3e-06_policy_psi_q_p_s_t_s1", 1,5),
        # make_list(
        #     "",
        #     1, 5),



        # # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-3.0_harml_neg_training_a0.01rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,10),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.1_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),

        make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta5.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),



        # make_list(
        #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_reinforce_a0.0003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",
        #     1, 5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_reinforce_a0.001_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_reinforce_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5,),
        # # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_reinforce_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_reinforce_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-0.3_harml_neg_reinforce_a0.0003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-1.0_harml_neg_reinforce_a0.0003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-3.0_harml_neg_reinforce_a0.0003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-3.0_harml_neg_reinforce_a0.001_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s2",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_reinforce_a0.0003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),



        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.5_kl0.1_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.3_kl0.1_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.3_kl0.1_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.5_kl0.1_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,2) + make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-0.5_harml_reinforce_a1.0rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",3,10),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.5_kl0.1_harml_reinforce_a10.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-1.0_kl0.1_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),

        # make_list("info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-0.5_harml_reinforce_a0.5rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),


        # make_list("info_eval_ind_thresh0.0_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta1.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # # # make_list("info_eval_ind_thresh0.0_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta1.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr1e-05_policy_psi_q_p_s_t_s1",1,5),
        # # make_list("info_eval_ind_thresh0.0_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta1.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr1e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_ind_thresh0.0_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta1.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # make_list("info_eval_ind_thresh0.0_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta1.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",1,5),


        # make_list("info_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),
        # TODO later add back the below
        # make_list("info_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",1,5),



    ]

    # load_prefixes_to_use = [
    #     make_list(
    #         "special/info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-0.5_harml_reinforce_a1.0rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi0_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1",
    #         1, 10),
    #
    #     make_list(
    #         "special/info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi0_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1",
    #         1, 10),
    #
    # ]

    # load_prefixes_to_use = [
    #     make_list("special/info_eval_rlhf_Ll3.1BIn_LlGu31B_20misi1_len100_kl0.1_beta-0.5_harml_reinforce_a1.0rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi0_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s6",6,10),
    #     make_list(
    #         "special/info_eval_rlhf_Ll3.1BIn_LlGu31B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi0_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s6",
    #         6, 10),
    #
    # ]

    use_handcrafted_labels = False


    if use_handcrafted_labels:

        labels = [
            r"PPO (4 Episodes)",
            r"REINFORCE (4 Episodes)",
            r"REINFORCE, $r(s) - 3 e^{-0.5 r(s)}$ (4 Episodes)",
            r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-30 r(s)}$, $\alpha = 0.003$ (4 Episodes)",
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-30 r(s)}$, $\alpha = 0.003$ (2 Episodes)",
        ]

    else:
        # labels = ['_'.join(a[0].split('len20_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for a in load_prefixes_to_use]
        labels = ['_'.join(a[0].split('len100_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for a in load_prefixes_to_use]


    # inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    # inds_to_use = [0, 1, 2, 3, 4]
    # inds_to_use = [0, 1, 2, 3]

    inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13]
    # inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7]
    inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    inds_to_use = None

    figname_modifier = "len20_05_15_final"
    figname_modifier = "len20_05_23"
    figname_modifier = "len20_05_23_kl0_1"
    figname_modifier = "len20_05_23_kl0_01"
    figname_modifier = "len20_05_24_kl0_1"
    figname_modifier = "len20_05_25_kl0_01"
    figname_modifier = "len20_05_26_kl0_01"

    figname_modifier = "len20_05_26_kl0_03"
    figname_modifier = "len20_05_27_kl0_03"
    figname_modifier = "len20_05_28_kl0_01"
    figname_modifier = "len20_05_28_kl0_03"
    figname_modifier = "len20_05_28_2_kl0_01"
    figname_modifier = "len20_05_31_kl0_01"
    figname_modifier = "len20_06_01_kl0_01"
    figname_modifier = "len20_06_03_kl0_01"
    figname_modifier = "len20_06_04_kl0_03"
    figname_modifier = "len20_06_05_kl0_03"

    figname_modifier = "len20_07_26_kl0_03"


    figname_modifier = "1B_len100_09_08_kl0_03"

    figname_modifier = "1B_len100_09_08_kl0_1"

    figname_modifier = "1B_len100_09_11_kl0_1"
    figname_modifier = "1B_len100_09_12_kl0_1"

    figname_modifier = "len20_09_13_kl0_03"
    figname_modifier = "len20_09_14_kl0_03"

    figname_modifier = "1B_len100_09_14_kl0_1"

    figname_modifier = "1B_len100_09_14_kl0_1_badonlyeval"
    figname_modifier = "1B_len100_09_14_kl0_1_llamaguardbadonlyeval"
    # figname_modifier = "len20_09_15_kl0_03"

    figname_modifier = "1B_len100_09_15_kl0_1"


    do_gcg = False # True
    # if not do_gcg:
    #     fontsize = 12

    if not use_handcrafted_labels:
        # fontsize = 8
        fontsize = 4

    if do_gcg:
        inds_to_use = [0, 10]
        inds_to_use = [0,1,2,3,4,5,6]
        inds_to_use = None
        # inds_to_use = [0,1,2,3,4,5,6]

        # TODO have to sync this with the results before

        gcg_prefixes = [
            make_list(
                "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epo1_epi4_schconstant_alr0.0001_clr0.0001_clossmse_policy_s1_actor",
                1, 10),

            make_list(
                "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
                1, 10),

            make_list(
                "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
                1, 10),
            make_list(
                "gcg_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
                1, 10),
            # make_list("20prompts_gcg_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor", 1, 10),


            make_list(
                "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
                1, 10),
            # make_list("20prompts_gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",1,10),

            # make_list("gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor", 1, 10),
            # make_list(
            #     "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
            #     1, 10),

            # 07-26 Stuff
            make_list("gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epo1_epi2_schconstant_alr0.0001_clr0.0001_clossmse_policy_s1_actor", 1, 10),
            make_list("gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",1,10),
            make_list("gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor", 1, 10),
            make_list("gcg_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",1,10),


        #     # TODO ensure this matches, and add the neg_tr one as well
        #     make_list(
        #         "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
        #         1, 10),
        #     make_list(
        #         "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
        #         1, 10),
        #     make_list(
        #         "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
        #         1, 10),
        #     # make_list("info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",1,10),
        #     make_list(
        #         "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi8_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
        #         1, 10),

        ]

        # # gcg_prefixes = [
        # #     make_list(
        # #         "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.01_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s6_harml_actor",
        # #         1, 15),
        # #
        # #     make_list("gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.01_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1_harml_actor",
        # #               1,15),
        # #
        # # ]

        gcg_prefixes = [
            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.03_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor", 1, 5),
            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor", 1, 5),
            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor", 1, 5)
        ]

        gcg_prefixes = [
            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta0.0_kl0.1_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor", 1,5),
            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor", 1, 5),
            # make_list(
            #     "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-10.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
            #     1, 2) + make_list(
            #     "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
            #     3, 5),
            # make_list(
            #     "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-30.0_kl0.1_harml_neg_training_a0.005_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
            #     1, 1) + make_list(
            #     "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-30.0_harml_neg_training_a0.005_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
            #     2, 5),
            # make_list(
            #     "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-5.0_kl0.1_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
            #     1, 2) + make_list(
            #     "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-5.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
            #     3, 5),

            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr1e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",1,5),
            make_list(
                "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta0.5_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
                1, 5),
            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",1,5),


            make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-10.0_harml_neg_training_a0.01rta5.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi1_schconstant_alr3e-06_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor", 1, 5),

            # make_list("gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.5_kl0.1_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",1,5),



            make_list(
                "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_beta-0.5_kl0.1_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor",
                1, 2)
            + make_list("rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl0.1_beta-0.5_harml_reinforce_a1.0rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-06_policy_psi_q_p_s_t_s1_harml_actor", 3,5)
            ,
            # TODO CHECK THIS ENSURE CORRECT

        ]


        # for x in result_2025-05-13-00-17_eval_gcg250_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_s*; do tail -n 2 $x; done
        gcg_logprobs_results_list = [
            [-60.90, -54.80, -55.77, -49.74, -60.50, -53.38, -40.51, -47.87, -61.19, -47.33],  # ppo
            # result_2025-05-13-00-15_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epo1_epi4_schconstant_alr0.0001_clr0.0001_clossmse_policy

            [-55.98, -55.18, -50.66, -56.32, -55.45, -62.03, -48.08, -52.74, -61.27, -52.06], # regular reinforce
            # result_2025-05-13-00-20_eval_gc
            # result_2025-05-15-21-52_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.

            [-64.51, -51.60, -58.95, -45.33, -69.47, -57.84, -66.63, -48.76, -58.48, -108.19],
            # reinf with reward transf
            # result_2025-05-15-17-09_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_poli

            [-54.94, -59.82, -51.30, -49.71, -62.02, -72.24, -54.40, -50.12, -64.69, -58.72], # baseprop
            # result_2025-05-13-00-17_eval_gcg250_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_s
            # result_2025-05-15-20-31_eval_gcg250_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_


            [-78.86, -62.08, -124.83, -97.63, -76.06, -2026.25, -59.10, -63.84, -120.35, -684.96],  # ours
            # result_2025-05-15-20-06_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.

        ]

        figname_modifier = "len20_05_15_bootstrap_gcg"
        figname_modifier = "len20_05_23_bootstrap_gcg"
        figname_modifier = "len20_05_23_bootstrap_kl0_1"
        figname_modifier = "len20_05_25_bootstrap_kl0_01"
        figname_modifier = "len20_05_26_bootstrap_gcg_kl0_01"
        figname_modifier = "len20_05_26_bootstrap_gcg_kl0_03"
        figname_modifier = "len20_05_27_bootstrap_gcg_kl0_03"
        figname_modifier = "len20_05_28_bootstrap_gcg_kl0_01"
        figname_modifier = "len20_05_28_bootstrap_gcg_kl0_03"
        figname_modifier = "len20_06_06_bootstrap_gcg_kl0_03"
        figname_modifier = "len20_07_26_bootstrap_gcg_kl0_03"
        figname_modifier = "len20_07_29_20prompts_gcg_kl0_03"
        figname_modifier = "len20_07_29_10prompts_gcg_kl0_03"
        figname_modifier = "1B_len100_09_08_10prompts_gcg_kl0_03"
        figname_modifier = "1B_len100_09_08_10prompts_gcg_kl0_1"
        figname_modifier = "1B_len100_09_11_10prompts_gcg_kl0_1"
        figname_modifier = "1B_len100_09_12_10prompts_gcg_kl0_1"

    if inds_to_use is None:
        pass
    else:
        labels = [labels[i] for i in inds_to_use]
        load_prefixes_to_use = [load_prefixes_to_use[i] for i in inds_to_use]
        marker_list = [marker_list[i] for i in inds_to_use]
        color_list = [color_list[i] for i in inds_to_use]

    results_list = [[] for i in range(len(load_prefixes_to_use))]

    do_load_prefixes(results_list, load_prefixes_to_use, map_location='cpu')

    if do_gcg:
        gcg_results_list = [[] for i in range(len(gcg_prefixes))]
        do_load_prefixes(gcg_results_list, gcg_prefixes, map_location='cpu')


ylabel_bad = f"Total Prob of Bad Output (reward < {threshold})"



if do_gcg:
    ylabel_bad = f"Proportion of GCG Attack Success"


    make_frontier_bootstrap(
        xlabel="Average Return (including KL penalty)", ylabel=ylabel_bad,
        figname=f"{figname_modifier}_frontier_ret",
        labels=labels, results_list=results_list,
        color_list=color_list, marker_list=marker_list,
        # xlimlow=xlimlow, xlimhigh=xlimhigh,
        fontsize=fontsize, aggregate_seeds=True,
        tuple_index=1,
        tuple_index_gcg=0, # 1 originally TODO change back if need the old plots
        compare_to_reference=compare_to_reference,
        threshold=threshold,
        gcg_results_list=gcg_results_list
    )

    # TODO This one below goes better in a table instead...

    ylabel_bad = f"Log Prob of Target Sequence After GCG Attack"

    make_frontier_bootstrap(
        xlabel="Average Reward", ylabel=ylabel_bad,
        figname=f"{figname_modifier}_frontier_logprob",
        labels=labels, results_list=results_list,
        color_list=color_list, marker_list=marker_list,
        ylimlow=-600,
        # xlimhigh=xlimhigh,
        fontsize=fontsize, aggregate_seeds=True,
        tuple_index=1,
        tuple_index_gcg=1,
        compare_to_reference=compare_to_reference,
        threshold=threshold,
        gcg_results_list=gcg_logprobs_results_list,
        gcg_results_type="log_probs"
    )


    raise SystemExit(0)



make_frontier_bootstrap(
    xlabel="Average Reward", ylabel=ylabel_bad,
    figname=f"{figname_modifier}_frontier_rew",
    labels=labels, results_list=results_list,
    color_list=color_list, marker_list=marker_list,
    # xlimlow=xlimlow, xlimhigh=xlimhigh,
    fontsize=fontsize, aggregate_seeds=True,
    tuple_index=0,
    compare_to_reference=compare_to_reference,
    threshold=threshold
)

make_frontier_bootstrap(
    xlabel="Average Return (including KL penalty)", ylabel=ylabel_bad,
    figname=f"{figname_modifier}_frontier_ret",
    labels=labels, results_list=results_list,
    color_list=color_list, marker_list=marker_list,
    # xlimlow=xlimlow, xlimhigh=xlimhigh,
    fontsize=fontsize, aggregate_seeds=True,
    tuple_index=1,
    compare_to_reference=compare_to_reference,
    threshold=threshold
)

