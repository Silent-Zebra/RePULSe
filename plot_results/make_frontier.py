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

import numpy as np
from scipy import stats
from scipy.special import logit, expit  # logit = log(p / (1 - p)), expit = inverse

def logit_t_ci(successes, totals, confidence=0.95, epsilon=0.5):
    """
    Compute a t-distribution-based confidence interval over per-seed probabilities,
    using a logit transform to keep the interval bounded in [0, 1].

    Args:
        successes (list or np.array): number of successes per seed
        totals (list or np.array): number of trials per seed
        confidence (float): confidence level (e.g., 0.95)
        epsilon (float): pseudo-count to avoid p=0 or p=1 (default: 0.5)

    Returns:
        mean_p: mean estimated probability
        lower_bound: lower bound of confidence interval
        upper_bound: upper bound of confidence interval
    """
    successes = np.array(successes)
    totals = np.array(totals)

    # Smoothed proportions to avoid 0 and 1
    smoothed_probs = (successes + epsilon) / (totals + 2 * epsilon)

    # Transform to logit space
    logits = logit(smoothed_probs)

    # Compute mean and CI in logit space
    n = len(logits)
    mean_logit = logits.mean()
    std_logit = logits.std(ddof=1)

    t_value = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    margin = t_value * std_logit / np.sqrt(n)

    lower_logit = mean_logit - margin
    upper_logit = mean_logit + margin

    # Transform back to probability space
    mean_p = expit(mean_logit)
    lower_p = expit(lower_logit)
    upper_p = expit(upper_logit)

    return mean_p, lower_p, upper_p


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
            y_successes = []
            y_totals = []
            for t in tensor_list:
                # print(np.array(d[dictkey_x]).shape)
                # Take last result
                # print(t.shape)
                x_results.append(t.cpu().numpy().mean())
                print((t.cpu().numpy() < threshold).sum())
                print(t.cpu().numpy().shape)
                # 1/0
                y_results.append((t.cpu().numpy() < threshold).mean())
                y_successes.append((t.cpu().numpy() < threshold).sum())
                y_totals.append(t.cpu().shape[0])

            mean_p, lower_p, upper_p = logit_t_ci(y_successes, y_totals)

            to_plot_x = np.stack(x_results)
            # to_plot_y = np.stack(y_results)
            to_plot_y = mean_p

            print(mean_p)
            print(lower_p)
            print(upper_p)
            1/0


        # print(to_plot_x.shape)
        # print(to_plot_y.shape)

        # print(to_plot.shape)
        if not aggregate_seeds:
            plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i], marker=marker_list[i])

        else: # aggregate_seeds:
            n_samples_x = to_plot_x.shape[0]
            # n_samples_y = to_plot_y.shape[0]


            t_value_x = stats.t.ppf(0.975, df=n_samples_x - 1)
            # t_value_y = stats.t.ppf(0.975, df=n_samples_y - 1)

            print(to_plot_x)
            print(to_plot_y)
            x_std = np.std(to_plot_x, axis=0, ddof=1)
            # y_std = np.std(to_plot_y, axis=0, ddof=1)

            x_conf = t_value_x * x_std / np.sqrt(n_samples_x)
            # y_conf = t_value_y * y_std / np.sqrt(n_samples_y)
            y_conf = wilson_score_interval()

            print(t_value_x)
            print(t_value_y)

            print(x_conf)
            print(y_conf)

            print(to_plot_x.shape[0])
            print(to_plot_x.shape)
            # 1 / 0

            to_plot_x = to_plot_x.mean(axis=0)
            to_plot_y = to_plot_y.mean(axis=0)
            # to_plot_x = np.stack(x_results).mean(axis=1).mean(axis=0)
            # to_plot_y = np.stack(y_results).mean(axis=1).mean(axis=0)

            # print(to_plot_x.shape)
            # print(to_plot_y.shape)

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


do_load = True # False
if do_load:
    n_epochs = 100
    load_prefixes_to_use = [
        # [
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
        # ],
        [
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s4",
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5",

        ],
        # [
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s4",
        # "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5",
        # ],
        [
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
        ],
        [
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1",
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s2",
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s3",
        ],
        # [
        #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s1",
        #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s2",
        #     "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_s3",
        # ],
        [
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
            "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
        ]

    ]

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

    figname_modifier = "len20_05_03"


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

make_frontier(
    xlabel="Average Reward", ylabel=ylabel_bad,
    figname=f"{figname_modifier}_frontier",
    labels=labels, results_list=results_list,
    color_list=color_list, marker_list=marker_list,
    # xlimlow=xlimlow, xlimhigh=xlimhigh,
    fontsize=fontsize, aggregate_seeds=True
)

