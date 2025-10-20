import matplotlib

matplotlib.use('PDF')  # THIS MUST BE AT THE START OF THE CODE (before other imports)!!!!
import matplotlib.pyplot as plt

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import datetime
import copy
import scipy.stats as stats

from plot_utils import make_list, do_load_prefixes


def plot_with_conf_bounds(ax, record, x_range, label, **kwargs):
    avg = record.mean(axis=0)
    stdev = np.std(record, axis=0, ddof=1)

    t_value = stats.t.ppf(0.975, df=record.shape[0] - 1)

    conf_bound = t_value * stdev / np.sqrt(record.shape[0])

    upper_conf_bound = avg + conf_bound
    lower_conf_bound = avg - conf_bound

    ax.plot(x_range, avg, label=label, **kwargs)
    ax.fill_between(x_range, lower_conf_bound, upper_conf_bound, alpha=0.3, **kwargs)

    return avg[-1], conf_bound[-1]


# Comment out/select as needed
figname_modifier = "toyrlhf_kl10_10_18_final"
figname_modifier = "toyrlhf_10_18_final"


if "final" in figname_modifier:

    if "kl10" in figname_modifier:

        labels = [
            # r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.01$",
            # r"PPO",
            r"REINFORCE, no reward transformation",
            r"REINFORCE, $r'(s) = r(s) - e^{- r(s)}$",
            r"REINFORCE, $r'(s) = r(s) - 10 e^{- r(s)}$",
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{- r(s)}$, $\alpha = 100$",
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 100$",

        ]
        fontsize = 11

        # for x in $(ls /h/319/stephenzhao/OpenRLHF/info/toyrlhfmultikl1 | grep "kl10\.0" | grep analytic | grep _s2 ); do echo make_list\(\"$x\", 1, 5\),; done
        # for x in $(ls info | grep "kl10\.0" | grep analytic | grep _s2 ); do echo make_list\(\"$x\", 1, 5\),; done
        #
        load_prefixes_to_use = [

            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl10.0_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr1e-05_policy_psi_q_p_s_t_s2",
                1, 3),


            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl10.0_beta-1.0_harml_reinforce_a1.0rta1.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr1e-05_policy_psi_q_p_s_t_s2",
                1, 3),

            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl10.0_beta-1.0_harml_reinforce_a10.0rta10.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr1e-05_policy_psi_q_p_s_t_s2",
                1, 3),

            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl10.0_beta-1.0_harml_neg_training_a100.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr0.0003_blr1e-05_policy_psi_q_p_s_t_s2",
                1, 3),

            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl10.0_beta-10.0_harml_neg_training_a100.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr0.0003_blr1e-05_policy_psi_q_p_s_t_s2",
                1, 3),

        ]


    elif "kl1" in figname_modifier:
        labels = [
            # r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.01$",
            # r"PPO",
            r"REINFORCE",
            r"REINFORCE, $r'(s) = r(s) - e^{- r(s)}$",
            r"REINFORCE, $r'(s) = r(s) - 10 e^{- r(s)}$",
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{- r(s)}$, $\alpha = 100$",
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 100$",

        ]
        fontsize = 11


        # for x in $(ls /h/319/stephenzhao/OpenRLHF/info/toyrlhfmultikl1 | grep "kl1\.0" | grep analytic | grep _s2 ); do echo make_list\(\"$x\", 1, 5\),; done
        # for x in $(ls | grep "kl1\.0" | grep analytic | grep _s2 ); do echo make_list\(\"$x\", 1, 5\),; done
        #
        load_prefixes_to_use = [
            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl1.0_beta0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s2",
                1, 5),
            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl1.0_beta-1.0_harml_reinforce_a10.0rta10.0_b-1.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s2",
                1, 5),
            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_kl1.0_beta-10.0_harml_neg_training_a0.3_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s2",
                1, 5),

        ]

        labels = ['_'.join(a[0].split('len2_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for
                  a in load_prefixes_to_use]
        fontsize = 5

    elif "kl" not in figname_modifier:

        labels = [
            r"PPO, no reward transformation",
            r"REINFORCE, no reward transformation",
            r"REINFORCE, $r'(s) = r(s) - e^{- r(s)}$",
            r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.01$",
            r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.01$",
        ]
        fontsize = 11
        load_prefixes_to_use = [
            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta10000_kl0.0_policy_ppo_epo1_epi10_schconstant_alr0.0001_clr3e-05_clossmse_policy_s1",
                1, 5),

            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta-10.0_kl0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
                1, 5),
            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta-1.0_kl0.0_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "analyticlogprob_rewsample_rlhf_baseprop_di_To_thmaisa_len2_beta-10.0_kl0.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
                1, 5),

            make_list(
                "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta-10.0_kl0.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
                1, 5),

        ]


    else:
        raise NotImplementedError



results_list = [[] for i in range(len(load_prefixes_to_use))]

do_load_prefixes(results_list, load_prefixes_to_use)

if "final" not in figname_modifier:

    labels = ['_'.join(a[0].split('len2_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for
              a in load_prefixes_to_use]
    fontsize = 5



x_range = np.arange(51) * 10 * 500

color_list_for_variances = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red',
                            'xkcd:light purple', 'xkcd:dark grey', 'xkcd:light brown', 'xkcd:light lime green',
                            'xkcd:light navy blue', 'xkcd:light indigo', 'xkcd:olive yellow', 'xkcd:peach',
                            'xkcd:light lavender', 'xkcd:bright pink']
color_list_for_fqs = [
    'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:green', 'xkcd:blue',
    'xkcd:black',  'xkcd:gray',  'xkcd:light brown',
    'xkcd:pink', 'xkcd:gold', 'xkcd:teal', 'xkcd:magenta',
] * 5
linestyle_list = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5)), (0, (1, 1))] * 5


def plot_results_over_time(index_to_use=0, plot_name="logprobbad", ylabel=r"Log Total Probability of Bad Output"):
    fig, ax1 = plt.subplots()

    for i in range(len(results_list)):
        np_results = np.stack([x[index_to_use] for x in results_list[i]])
        # print(np_results)
        print(np_results.shape)
        plot_with_conf_bounds(
            ax1, np_results, x_range, label=labels[i],
            color=color_list_for_fqs[i],
            linestyle=linestyle_list[i],
        )
    ax1.set_xlabel("Number of Samples", fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    # ax1.set_ylim(top=15)
    # Adjust tick colors
    # ax1.tick_params(axis="y", colors=color_list_for_fqs[0])
    # ax2.tick_params(axis="y", colors=color_list_for_variances[0])
    ax1.tick_params(axis='both', labelsize=fontsize)
    # Combine legends
    # fig.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.45, 0.5))
    plt.legend(fontsize=fontsize)
    # plt.legend()
    plt.tight_layout()
    figname = f"./{figname_modifier}_{plot_name}.pdf"
    plt.savefig(figname)
    plt.clf()


plot_results_over_time(index_to_use=0, plot_name="logprobbad", ylabel=r"Log Total Probability of Bad Output")

plot_results_over_time(index_to_use=1, plot_name="rew", ylabel=r"Average Reward")

plot_results_over_time(index_to_use=-1, plot_name="untransformed_ret", ylabel=r"Average Return")


raise SystemExit(0)




