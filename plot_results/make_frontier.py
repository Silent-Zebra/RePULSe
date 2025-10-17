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

load_dir = "./info"

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
    ylimlow=None, ylimhigh=None,
    calculate_cvar=False
):
    plt.clf()
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    all_x = []
    all_y = []

    for i in range(len(labels)):

        # print(f"Processing: {labels[i]}")  # Changed to f-string for clarity
        # print(results_list[i]) # Original print, can be verbose



        if isinstance(results_list[i], tuple):
            raise NotImplementedError # hasn't been used/tested in a while
            # # This case seems to assume pre-processed x and y, no aggregation
            # # If aggregation is needed here too, this part would also need modification
            # to_plot_x_agg = results_list[i][0]  # Assuming these are already aggregated means
            # to_plot_y_agg = results_list[i][1]
            # # No error bars defined for this case in original or new logic
            # # If this branch should also have error bars with bootstrapping, it needs similar logic
            # # based on the raw data that would lead to these tuple values.
            # # For now, plotting as a single point if not aggregate_seeds, or if aggregate_seeds
            # # and it's a tuple, it plots the pre-aggregated point without error bars.
            # if not aggregate_seeds:
            #     plt.scatter(to_plot_x_agg, to_plot_y_agg, label=labels[i], c=color_list[i], marker=marker_list[i])
            # else:  # aggregate_seeds is True but data is a pre-aggregated tuple
            #     # Plotting the point, but can't compute CI without per-seed data
            #     plt.scatter(to_plot_x_agg, to_plot_y_agg, label=labels[i], c=color_list[i], marker=marker_list[i])
            #     print(
            #         f"Warning: aggregate_seeds is True for {labels[i]}, but data is a pre-aggregated tuple. Plotting point without error bars.")


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
                    if calculate_cvar:
                        assert threshold > 0 and threshold < 1
                        rew = unmodified_rew.float().cpu().numpy()
                        sorted_rew = np.sort(rew)

                        # Determine how many samples correspond to the worst alpha fraction
                        n = len(sorted_rew)
                        k = max(1, int(np.floor(threshold * n)))

                        # Take the worst k samples and average them
                        cvar = sorted_rew[:k].mean()

                        y_results_per_seed.append(cvar)
                    else:
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

                all_x.append(x_observed_mean)
                all_y.append(y_observed_mean)

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    if (ylimlow is not None) or (ylimhigh is not None):
        plt.ylim(ylimlow, ylimhigh)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.legend(fontsize=fontsize)


    if "final" in figname:
        if "1B" in figname:
            indices = [1, 2, 3, 4, 5]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="red", alpha=0.3, linestyle='--')

            # plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="black", alpha=0.3, linestyle=":")
            indices = [6, 7]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="black", alpha=0.3, linestyle='-.')
            indices = [8,9]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="teal", alpha=0.3, linestyle='-.')

        else:
            indices = [1,2,3,4,5]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="red", alpha=0.3, linestyle='--')
            indices = [6,7]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="black", alpha=0.3, linestyle=":")
            indices = [8,9,10]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="teal", alpha=0.3, linestyle='-.')
            # indices = [12,13,14,15,16]
            # plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="red", alpha=0.3)
            # indices = [17,18]
            # plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="green", alpha=0.3)

    plt.savefig(figname)
    print(f"Figure saved to {figname}")




figname_modifier = "len20_10_08_kl0_2_epi2_w_epi1_s10_clean"
figname_modifier = "len20_10_08_kl0_2_epi2_w_epi1_gcg"
figname_modifier = "len20_10_09_kl0_2_epi2_w_epi1_s10_final"


figname_modifier = "1B_len100_10_09_kl2"

figname_modifier = "len20_10_09_kl0_2_epi2_s10_final"

figname_modifier = "len20_10_09_kl0_2_epi2_sameepicomparison_s10_final"

figname_modifier = "len20_10_09_kl0_2_epi2_s10_gcg_final"

figname_modifier = "1B_len100_10_09_kl2_v2"

figname_modifier = "1B_len100_10_09_kl2_final"

figname_modifier = "1B_len100_10_09_kl2_v3"
figname_modifier = "1B_len100_10_10_kl2_v2"
figname_modifier = "1B_len100_10_10_kl2_v3"
figname_modifier = "1B_len100_10_10_kl2_v4"


figname_modifier = "1B_len100_10_10_kl2_v5"
figname_modifier = "1B_len100_10_11_kl2"
figname_modifier = "1B_len100_10_11_kl2_v2"
figname_modifier = "1B_len100_10_11_kl2_gcg"
figname_modifier = "1B_len100_10_11_kl2_v3"
figname_modifier = "1B_len100_10_12_kl2_gcg"
figname_modifier = "1B_len100_10_12_kl2"
figname_modifier = "1B_len100_10_12_kl2_v2"
figname_modifier = "1B_len100_10_13_kl2"

figname_modifier = "1B_len100_10_13_kl2_gcg"

figname_modifier = "1B_len100_10_14_kl2"

figname_modifier = "1B_len100_10_14_kl2"
figname_modifier = "1B_len100_10_14_kl2_final"
figname_modifier = "1B_len100_10_14_kl2_gcg_final"



figname_modifier = "1B_len100_10_14_kl2_wepi1"

figname_modifier = "1B_len100_10_14_kl2_epi2_final"

figname_modifier = "1B_len100_10_14_kl2_sameepi_final"

figname_modifier = "len20_10_09_kl0_2_epi2_s10_cvar_final"

figname_modifier = "1B_len100_10_14_kl2_cvar_final"

figname_modifier = "1B_len100_10_14_kl2_gcg_final_v2"



figname_modifier = "1B_len100_10_14_kl2_epi2_final"

figname_modifier = "1B_len100_10_14_kl2_gcg_final"

figname_modifier = "len20_10_14_kl0_2_epi2_s10_gcg_final"




do_1B_experiments = False
if "1B" in figname_modifier:
    do_1B_experiments = True


linestyle_list = ['solid'] * 30


color_list = [
    'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:green', 'xkcd:blue',
    'xkcd:black',  'xkcd:gray',  'xkcd:light brown',
    'xkcd:pink', 'xkcd:gold', 'xkcd:teal', 'xkcd:magenta',
] * 5
marker_list = ["D", "x", "^", "o", "P", "v", "v", "v", "P", "o", "o",
               "P", "o", "o", "P", "^", "^", "P", "v", "v", "v",
               "v", "v", "^", "P", "v", "D", "v", "v", "x", "v", # 22 23 25 27 28 reinf, ours, baseprop, reinftransf ppo
               "v", "v", "v", "^", "^", "x", "x", "x", "x", "D",
               "P", "P", "P", "v", "v", "v", "^", "^", "x", "x", "x", "x", "D", "P", "P", "P"]

marker_list = ["D"] * 12
marker_list.extend(["P"] * 12)
marker_list.extend(["o"] * 12)
marker_list.extend(["v"] * 12)
marker_list.extend(["^"] * 12)
marker_list.extend(["x"] * 12)

# xlimlow = 2
# xlimhigh = 10
fontsize = 9


compare_to_reference = False
if compare_to_reference:
    figname_modifier += "_comparetoref"


do_load = True # False

threshold = -5 # Check diff values here too
if do_1B_experiments:
    threshold = -7

figname_modifier += f"_thr{threshold}"

if "final" in figname_modifier:

    color_list = [
                     'xkcd:light brown',
                     'xkcd:orange', 'xkcd:red', 'xkcd:pink', 'xkcd:magenta', 'xkcd:purple',
                     'xkcd:black', 'xkcd:gray',
                     'xkcd:green', 'xkcd:teal', 'xkcd:blue',
                     # 'xkcd:gold',

                 ] * 2
    marker_list = ["D", "^", "^", "^", "^", "^",
                   "o", "o",
                   "P", "P", "P",
                   # "P", "v", "v", "v", "P", "o", "o",
                   # "P", "o", "o", "P", "^", "^", "P", "v", "v", "v",
                   # "v", "v", "^", "P", "v", "D", "v", "v", "x", "v",
                   ] * 2




if do_load:
    n_epochs = 100

    if "kl0_2" in figname_modifier and "len20" in figname_modifier:
        # load_dir = "./info_10_09"
        # load_dir = "./info_10_10"
        load_prefixes_to_use = [
            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta5.0_policy_ppo_epo1_epi4_schconstant_alr3e-05_clr3e-05_clossmse_policy_s1",
                1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.1_harml_reinforce_a1.0rta1.0_b-0.1_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s5",
                1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.5_harml_reinforce_a0.3rta0.3_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),
            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.3_harml_reinforce_a1.0rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s5",
                1, 10),

            # make_list(
            #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.3_harml_reinforce_a0.3rta0.3_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s5",
            #     1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.5_harml_reinforce_a1.0rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),

            make_list(
                "info_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),
            make_list(
                "info_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),


            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),

            # make_list(
            #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-20.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr1e-05_blr3e-05_policy_psi_q_p_s_t_s1",
            #     1, 10),
            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-20.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr2e-05_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),
            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr1e-05_blr3e-05_policy_psi_q_p_s_t_s5",
                1, 10),


            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta5.0_policy_ppo_epo1_epi2_schconstant_alr3e-05_clr3e-05_clossmse_policy_s1",
                1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.1_harml_reinforce_a1.0rta1.0_b-0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s5",
                1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.5_harml_reinforce_a0.3rta0.3_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),
            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.3_harml_reinforce_a1.0rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s5",
                1, 10),

            # make_list(
            #     "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.3_harml_reinforce_a0.3rta0.3_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s5",
            #     1, 10),

            make_list(
                "info_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.5_harml_reinforce_a1.0rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),

            make_list(
                "info_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),

            make_list(
                "info_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
                1, 10),
            # for x in $(ls info/rlhfmultilen20kl2 | grep neg_tr | grep info | grep _s1); do echo make_list\(\"$x\", 1, 5\),; done


        ]


        if "sameepi" in figname_modifier:
            load_prefixes_to_use = load_prefixes_to_use[11:] + load_prefixes_to_use[8:11]
        elif "epi2" in figname_modifier and "epi1" in figname_modifier:
            pass
        elif "epi2" in figname_modifier:
            load_prefixes_to_use = load_prefixes_to_use[:11]







    elif do_1B_experiments and "kl2" in figname_modifier and "len100" in figname_modifier and "1B" in figname_modifier:
        # load_dir = "./info_10_09"
        # load_dir = "./info_10_10"

        load_prefixes_to_use = [
            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta0.5_policy_ppo_epo1_epi4_schconstant_alr3e-07_clr3e-06_clossmse_policy_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-0.3_harml_reinforce_a3.0rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-1.0_harml_reinforce_a1.0rta1.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-0.3_harml_reinforce_a3.0rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-1.0_harml_reinforce_a1.0rta1.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1",
                1,5),

            make_list(
                "info_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-10.0_harml_neg_training_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-5.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-07_blr3e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-5.0_harml_neg_training_a0.2_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-07_blr3e-07_policy_psi_q_p_s_t_s1",
                1, 5),






            # Epi1/epi2 stuff
            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta0.5_policy_ppo_epo1_epi2_schconstant_alr3e-07_clr3e-06_clossmse_policy_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            # make_list(
            #     "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1",
            #     1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-0.3_harml_reinforce_a3.0rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-1.0_harml_reinforce_a1.0rta1.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-0.3_harml_reinforce_a3.0rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-1.0_harml_reinforce_a1.0rta1.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "info_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-10.0_harml_neg_training_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1",
                1, 5),





        ]

        if "wepi1" in figname_modifier:
            pass
        elif "sameepi" in figname_modifier:
            load_prefixes_to_use = load_prefixes_to_use[10:] + load_prefixes_to_use[8:10]

            # for x in load_prefixes_to_use:
            #     print(x[0])
            # 1/0

        else:
            load_prefixes_to_use = load_prefixes_to_use[:10]



    else:
        raise Exception("Figname does not correspond to any set of data")



    # inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    # inds_to_use = [0, 1, 2, 3, 4]
    # inds_to_use = [0, 1, 2, 3]

    inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13]
    # inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7]
    inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    inds_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    inds_to_use = None


    do_gcg = False
    if "gcg" in figname_modifier:
        do_gcg = True
    # if not do_gcg:
    #     fontsize = 12

    if do_gcg:
        inds_to_use = [0, 10]
        inds_to_use = [0,1,2,3,4,5,6]
        inds_to_use = None
        # inds_to_use = [0,1,2,3,4,5,6]



        if "1B" in figname_modifier:

            load_prefixes_to_use = load_prefixes_to_use

            load_prefixes_to_use = [load_prefixes_to_use[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

            # for x in $(ls /scratch/zhaostep/OpenRLHF/checkpoint/rlhfmultikl20 | grep kl2 | grep gcg | grep s1 ); do echo make_list\(\"$x\", 1, 5\),; done
            gcg_prefixes = [

                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta0.5_policy_ppo_epo1_epi4_schconstant_alr3e-07_clr3e-06_clossmse_policy_s1_actor",
                    1, 5),

                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),
                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-0.3_harml_reinforce_a3.0rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),
                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-1.0_harml_reinforce_a1.0rta1.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),
                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-0.3_harml_reinforce_a3.0rta3.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),
                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-1.0_harml_reinforce_a1.0rta1.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1_harml_actor",
                          1,5),

                make_list(
                    "gcg_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),
                make_list(
                    "gcg_eval_rlhf_baseprop_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-10.0_harml_neg_training_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr1e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),

                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-5.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-07_blr3e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),
                make_list(
                    "gcg_eval_rlhf_Ll3.1BIn_SkReV2Ll3.1B_20misi1_len100_kl2.0_beta-5.0_harml_neg_training_a0.2_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-07_blr3e-07_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 5),

            ]

            gcg_logprobs_results_list = [
                #  for x in $(ls | grep result | grep eval_gcg | grep ppo | grep kl2 ); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-16.56, -16.83, -17.57, -15.90, -16.86],

                #  for x in $(ls | grep result | grep eval_gcg | grep reinf | grep beta0 | grep kl2 | grep 1e-07 ); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-15.99, -16.72, -15.36, -16.42, -16.08],

                #  for x in $(ls | grep result | grep eval_gcg | grep reinf | grep beta-0.3 | grep kl2 | grep a3 | grep 1e-07 ); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-16.06, -16.59, -17.05, -15.96, -17.11],

                #  for x in $(ls | grep result | grep eval_gcg | grep reinf | grep beta-1 | grep kl2 | grep a1 | grep 1e-07 ); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-16.62, -18.50, -16.53, -17.36, -16.08],
                #  for x in $(ls | grep result | grep eval_gcg | grep reinf | grep beta-0.3 | grep kl2 | grep a3 | grep 3e-07 ); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-15.89, -16.41, -16.18, -15.22, -17.14],
                #  for x in $(ls | grep result | grep eval_gcg | grep reinf | grep beta-1 | grep kl2 | grep a1 | grep 3e-07 ); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-15.20, -15.22, -15.74, -17.11, -16.69],

                #  for x in $(ls | grep result | grep eval_gcg | grep neg_tr | grep kl2 | grep a0.1 |  grep basepro); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-19.18, -15.01, -16.03, -14.27, -14.77],
                #  for x in $(ls | grep result | grep eval_gcg | grep neg_tr | grep kl2 | grep a1 |  grep basepro); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-17.38, -13.96, -17.67, -17.22, -16.46],

                #  for x in $(ls | grep result | grep eval_gcg | grep neg_tr | grep beta-5 | grep kl2 | grep a0.1 | grep _alr3e-07 | grep -v basepro); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-17.22, -18.00, -17.55, -16.42, -15.85],
                # for x in $(ls | grep result | grep eval_gcg | grep neg_tr | grep beta-5 | grep kl2 | grep a0.2 | grep _alr3e-07 | grep -v basepro); do tail -n 13 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-16.13, -17.25, -17.03, -19.99, -15.43],

            ]


        else:

            # TODO copy over results
            #  for x in $(ls /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2/ | grep gcg); do \cp -rf  /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2/$x gcginfo/; done

            load_prefixes_to_use = load_prefixes_to_use

            fontsize -= 2

            # for x in $(ls gcginfo | grep _s1); do echo make_list\(\"$x\", 1, 5\),; done
            gcg_prefixes = [
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta5.0_policy_ppo_epo1_epi4_schconstant_alr3e-05_clr3e-05_clossmse_policy_s1_actor",
                    1, 10),

                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.1_harml_reinforce_a1.0rta1.0_b-0.1_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.5_harml_reinforce_a0.3rta0.3_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.3_harml_reinforce_a1.0rta1.0_b-0.3_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-0.5_harml_reinforce_a1.0rta1.0_b-0.5_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),

                make_list(
                    "gcg_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-20.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr2e-05_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),
                make_list(
                    "gcg_eval_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr1e-05_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor",
                    1, 10),

            ]

            gcg_logprobs_results_list = [
                # for x in $(ls | grep result | grep eval_gcg | grep ppo | grep kl0.2 ); do tail -n 3 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'
                [-11.62, -10.73, -12.45, -11.24, -11.41, -11.07, -11.63, -11.25, -13.39, -12.22],


                # for x in $(ls | grep result | grep eval_gcg | grep neg_tr | grep beta-30 | grep kl0.2 | grep -v basepro); do tail -n 3 $x; done
                # result_2025-10-07-00-53_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_kl0.2_beta-30.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi2_schconstant_alr1e-05_blr3e-05_policy_psi_q_p_s_t_s1_harml_actor.txt
                [ -50.34, -51.41, -49.90, -45.96, -49.65, -12.73, -19.52, -14.63, -15.07, -13.01],  #

                # for x in $(ls | grep result | grep eval_gcg | grep neg_tr | grep beta-20 | grep kl0.2 | grep -v basepro); do tail -n 3 $x; done
                [-11.34, -12.43, -13.23, -13.80, -9.73, -10.21, -12.78 , -10.95, -11.91, -10.14],


                # for x in $(ls | grep result | grep eval_gcg | grep basepro | grep kl0.2 ); do tail -n 3 "$x" | head -n 1 | grep "Average Log Prob of Target After Attack" | awk -F': ' '{print $2}'; done | paste -sd, - | sed 's/^/[/' | sed 's/$/]/'

            ]



        # # for x in result_2025-05-13-00-17_eval_gcg250_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_s*; do tail -n 2 $x; done
        # gcg_logprobs_results_list = [
        #     [-60.90, -54.80, -55.77, -49.74, -60.50, -53.38, -40.51, -47.87, -61.19, -47.33],  # ppo
        #     # result_2025-05-13-00-15_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta33.333_kl0.03_policy_ppo_epo1_epi4_schconstant_alr0.0001_clr0.0001_clossmse_policy
        #
        #     [-55.98, -55.18, -50.66, -56.32, -55.45, -62.03, -48.08, -52.74, -61.27, -52.06], # regular reinforce
        #     # result_2025-05-13-00-20_eval_gc
        #     # result_2025-05-15-21-52_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta0.0_kl0.03_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.
        #
        #     [-64.51, -51.60, -58.95, -45.33, -69.47, -57.84, -66.63, -48.76, -58.48, -108.19],
        #     # reinf with reward transf
        #     # result_2025-05-15-17-09_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-0.5_kl0.03_harml_reinforce_a3.0_policy_psi_q_p_s_t_ctl_epo1_epi4_schconstant_alr0.0_blr0.0001_poli
        #
        #     [-54.94, -59.82, -51.30, -49.71, -62.02, -72.24, -54.40, -50.12, -64.69, -58.72], # baseprop
        #     # result_2025-05-13-00-17_eval_gcg250_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi4_s
        #     # result_2025-05-15-20-31_eval_gcg250_rlhf_baseprop_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.003_policy_psi_q_
        #
        #
        #     [-78.86, -62.08, -124.83, -97.63, -76.06, -2026.25, -59.10, -63.84, -120.35, -684.96],  # ours
        #     # result_2025-05-15-20-06_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.
        #     [-78.86, -62.08, -124.83, -97.63, -76.06, -2026.25, -59.10, -63.84, -120.35, -684.96],  # ours
        #     # result_2025-05-15-20-06_eval_gcg250_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-30.0_kl0.03_harml_neg_training_a0.
        #
        # ]

    calculate_cvar = False
    if "cvar" in figname_modifier:
        calculate_cvar = True
        threshold = 0.0001
        if "1B" in figname_modifier:
            threshold = 0.002
            fontsize -= 1

    use_handcrafted_labels = False

    if "final" in figname_modifier:
        use_handcrafted_labels = True

        if "sameepi" in figname_modifier:
            more_eps_str = " (2 Episodes)"
            less_eps_str = " (2 Episodes)"
        else:
            more_eps_str = " (4 Episodes)"
            less_eps_str = " (2 Episodes)"



        if "kl0_2" in figname_modifier and "len20" in figname_modifier:
            labels = [
                # All baselr3e-5
                r"PPO" + more_eps_str,
                r"REINFORCE" + more_eps_str,
                r"REINFORCE, $r(s) - e^{-0.1 r(s)}$" + more_eps_str,
                r"REINFORCE, $r(s) - 0.3 e^{-0.5 r(s)}$" + more_eps_str,
                r"REINFORCE, $r(s) - e^{-0.3 r(s)}$" + more_eps_str,
                r"REINFORCE, $r(s) - e^{-0.5 r(s)}$" + more_eps_str,
                r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-30 r(s)}$, $\alpha = 0.1$" + more_eps_str,
                r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-30 r(s)}$, $\alpha = 1$" + more_eps_str,
                r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.1$" + less_eps_str,
                # alr3e-05
                r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-20 r(s)}$, $\alpha = 0.1$" + less_eps_str,
                # alr2e-05
                r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-30 r(s)}$, $\alpha = 0.1$" + less_eps_str,
                # alr1e-05
            ]
        elif "kl2" in figname_modifier and "len100" and "1B" in figname_modifier:
            if "sameepi" in figname_modifier:
                fontsize -= 2

            labels = [
                # All baselr3e-5
                r"PPO" + more_eps_str,
                r"REINFORCE" + more_eps_str,
                r"REINFORCE, $r(s) - 3 e^{-0.3 r(s)}$, lr 1e-7" + more_eps_str,
                r"REINFORCE, $r(s) - 1 e^{-r(s)}$, lr 1e-7" + more_eps_str,
                r"REINFORCE, $r(s) - 3 e^{-0.3 r(s)}$, lr 3e-7" + more_eps_str,
                r"REINFORCE, $r(s) - 1 e^{-r(s)}$, lr 3e-7" + more_eps_str,
                r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.1$" + more_eps_str,
                r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 1$" + more_eps_str,
                r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-5 r(s)}$, $\alpha = 0.1$" + less_eps_str,
                r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.1$" + less_eps_str,
                # alr3e-05
                # r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-20 r(s)}$, $\alpha = 0.1$" + less_eps_str,
                # # alr2e-05
                # r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-30 r(s)}$, $\alpha = 0.1$" + less_eps_str,
                # # alr1e-05
            ]

        else:
            raise NotImplementedError


    else:
        # labels = ['_'.join(a[0].split('len20_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for a in load_prefixes_to_use]
        labels = [
            '_'.join(a[0].split('len100_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0]
            for a in load_prefixes_to_use]

    if not use_handcrafted_labels:
        # fontsize = 8
        fontsize = 4

    if inds_to_use is None:
        pass
    else:
        labels = [labels[i] for i in inds_to_use]
        load_prefixes_to_use = [load_prefixes_to_use[i] for i in inds_to_use]
        marker_list = [marker_list[i] for i in inds_to_use]
        color_list = [color_list[i] for i in inds_to_use]

    results_list = [[] for i in range(len(load_prefixes_to_use))]



    do_load_prefixes(results_list, load_prefixes_to_use, map_location='cpu', load_dir=load_dir)

    if do_gcg:
        gcg_results_list = [[] for i in range(len(gcg_prefixes))]
        do_load_prefixes(gcg_results_list, gcg_prefixes, map_location='cpu', load_dir=load_dir)


ylabel_bad = f"Total Prob of Bad Output (reward < {threshold})"
if calculate_cvar:
    ylabel_bad = f"CVaR (average reward of worst {threshold * 100}% samples)"


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
        tuple_index_gcg=1, # 1 originally TODO change back if need the old plots
        compare_to_reference=compare_to_reference,
        threshold=threshold,
        gcg_results_list=gcg_results_list,
        calculate_cvar=calculate_cvar
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
    threshold=threshold,
    calculate_cvar=calculate_cvar
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
    threshold=threshold,
    calculate_cvar=calculate_cvar
)

