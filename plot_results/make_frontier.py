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


from scipy.stats import norm
# from scipy import stats

load_dir = "./info"

def make_frontier_bootstrap(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7, legendfontsize=7,
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

        if isinstance(results_list[i], tuple):
            raise NotImplementedError # hasn't been used/tested in a while


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
                print("Warning: compare_to_reference not tested in quite a while")
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
    plt.legend(fontsize=legendfontsize)


    if "final" in figname:
        if "1B" in figname:
            indices = [1, 2, 3, 4, 5]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="red", alpha=0.3, linestyle='--')
            indices = [6, 7]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="black", alpha=0.3, linestyle=':')
            indices = [8,9]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="teal", alpha=0.3, linestyle='-.')

        else:
            indices = [1,2,3,4,5]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="red", alpha=0.3, linestyle='--')
            indices = [6,7]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="black", alpha=0.3, linestyle=":")
            indices = [8,9,10]
            plt.plot([all_x[i] for i in indices], [all_y[i] for i in indices], color="teal", alpha=0.3, linestyle='-.')


    plt.savefig(figname)
    print(f"Figure saved to {figname}")



# Comment out/select as needed
figname_modifier = "1B_len100_10_18_kl2_epi2_heldout2"
figname_modifier = "len20_10_18_kl0_2_epi2_s10_heldout2"

figname_modifier = "len20_10_23_kl0_2_epi2_s10_final"
figname_modifier = "len20_10_23_kl0_2_epi2_s10_gcg_final"
figname_modifier = "len20_10_23_kl0_2_epi2_s10_cvar_final"
figname_modifier = "len20_10_23_kl0_2_epi2_s10_sameepi_final"
#
figname_modifier = "1B_len100_10_23_kl2_epi2_final"
figname_modifier = "1B_len100_10_23_kl2_gcg_final"
figname_modifier = "1B_len100_10_23_kl2_cvar_final"
figname_modifier = "1B_len100_10_23_kl2_sameepi_final"



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
legendfontsize = 9

compare_to_reference = False
if compare_to_reference:
    figname_modifier += "_comparetoref"


threshold = -5
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
                   # "v", "v", "^", "P", "v", "D", "v", "v", "x", "v",
                   ] * 2


if "heldout2" in figname_modifier:
    load_dir = "./info_heldout2"

if "kl0_2" in figname_modifier and "len20" in figname_modifier:
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
    else:
        load_prefixes_to_use = load_prefixes_to_use[:10]



else:
    raise Exception("Figname does not correspond to any set of data")

inds_to_use = None


do_gcg = False
if "gcg" in figname_modifier:
    do_gcg = True
# if not do_gcg:
#     fontsize = 12

if do_gcg:
    inds_to_use = None

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

    else:

        # TODO copy over results
        #  for x in $(ls /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2/ | grep gcg); do \cp -rf  /mfs1/u/stephenzhao/OpenRLHF/checkpoint/rlhfmultilen20kl2/$x gcginfo/; done

        load_prefixes_to_use = load_prefixes_to_use

        legendfontsize -= 2

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


calculate_cvar = False
if "cvar" in figname_modifier:
    calculate_cvar = True
    threshold = 0.0001
    if "1B" in figname_modifier:
        threshold = 0.002
        legendfontsize -= 1

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
            r"PPO, no reward transformation" + more_eps_str,
            r"REINFORCE, no reward transformation" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - e^{-0.1 r(s)}$" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - 0.3 e^{-0.5 r(s)}$" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - e^{-0.3 r(s)}$" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - e^{-0.5 r(s)}$" + more_eps_str,
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
            legendfontsize -= 2

        labels = [
            r"PPO, no reward transformation" + more_eps_str,
            r"REINFORCE, no reward transformation" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - 3 e^{-0.3 r(s)}$, lr 1e-7" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - 1 e^{-r(s)}$, lr 1e-7" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - 3 e^{-0.3 r(s)}$, lr 3e-7" + more_eps_str,
            r"REINFORCE, $r'(s) = r(s) - 1 e^{-r(s)}$, lr 3e-7" + more_eps_str,
            r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.1$" + more_eps_str,
            r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 1$" + more_eps_str,
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-5 r(s)}$, $\alpha = 0.1$" + less_eps_str,
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-5 r(s)}$, $\alpha = 0.2$" + less_eps_str,
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
        fontsize=fontsize, legendfontsize=legendfontsize,
        aggregate_seeds=True,
        tuple_index=1,
        tuple_index_gcg=1, # 1 originally TODO change back if need the old plots
        compare_to_reference=compare_to_reference,
        threshold=threshold,
        gcg_results_list=gcg_results_list,
        calculate_cvar=calculate_cvar
    )

    raise SystemExit(0)



make_frontier_bootstrap(
    xlabel="Average Reward", ylabel=ylabel_bad,
    figname=f"{figname_modifier}_frontier_rew",
    labels=labels, results_list=results_list,
    color_list=color_list, marker_list=marker_list,
    # xlimlow=xlimlow, xlimhigh=xlimhigh,
    fontsize=fontsize, legendfontsize=legendfontsize,
    aggregate_seeds=True,
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
    fontsize=fontsize, legendfontsize=legendfontsize,
    aggregate_seeds=True,
    tuple_index=1,
    compare_to_reference=compare_to_reference,
    threshold=threshold,
    calculate_cvar=calculate_cvar
)

