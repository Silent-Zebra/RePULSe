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

n_epochs = 80
policy_updates_per_epoch = 5

ylabel_bad = "Log Total Prob of Bad Word"


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
                # print(np.array(d[dictkey_x]).shape)
                # Take last result
                # print(t.shape)
                x_results.append(t.cpu().numpy().mean())
                y_results.append((t.cpu().numpy() < threshold).mean())

            to_plot_x = np.stack(x_results)
            to_plot_y = np.stack(y_results)

        # print(to_plot_x.shape)
        # print(to_plot_y.shape)

        # print(to_plot.shape)
        if not aggregate_seeds:
            plt.scatter(to_plot_x, to_plot_y, label=labels[i], c=color_list[i], marker=marker_list[i])

        else: # aggregate_seeds:
            n_samples_x = to_plot_x.shape[0]
            n_samples_y = to_plot_y.shape[0]

            z_score = 1.96  # For 95% confidence
            x_std = np.std(to_plot_x, axis=0, ddof=1)
            y_std = np.std(to_plot_y, axis=0, ddof=1)
            x_conf = z_score * x_std / np.sqrt(n_samples_x)
            y_conf = z_score * y_std / np.sqrt(n_samples_y)

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
        [
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.001_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
        ],
        [
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
        ],
        [
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s1",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s2",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s3",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s4",
        "rewards_eval_toy_rlhf_Sm13In_remodev3lav2_20misi1_len20_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5",
        ],
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

