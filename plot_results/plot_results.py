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


load_prefixes_to_use = [
    make_list(
        "analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta-10.0_kl0.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1",
        1, 5),
    make_list("analyticlogprob_rewsample_rlhf_baseprop_di_To_thmaisa_len2_beta-10.0_kl0.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0003_blr0.0001_policy_psi_q_p_s_t_s1", 1, 5),
    make_list("analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta10000_kl0.0_policy_ppo_epo1_epi10_schconstant_alr0.0001_clr3e-05_clossmse_policy_s1", 1, 5),
    make_list("analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta-10.0_kl0.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1", 1, 5),
    make_list("analyticlogprob_rewsample_rlhf_di_To_thmaisa_len2_beta-1.0_kl0.0_harml_reinforce_a1.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr0.0_blr0.0001_policy_psi_q_p_s_t_s1", 1, 5),
]

results_list = [[] for i in range(len(load_prefixes_to_use))]

do_load_prefixes(results_list, load_prefixes_to_use)

labels = ['_'.join(a[0].split('len2_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for
          a in load_prefixes_to_use]

labels = [
r"RePULSe ($q_\xi$), $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.01$",
r"$p_\theta$ proposal, $\sigma_\theta(s) \propto p_\theta(s) e^{-10 r(s)}$, $\alpha = 0.01$",
r"PPO",
r"REINFORCE",
r"REINFORCE, $r(s) - e^{- r(s)}$",
]

# print(results_list[0])
# print(results_list[0][2])
# 1/0

x_range = np.arange(51) * 10 * 500

fig, ax1 = plt.subplots()

color_list_for_variances = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red', 'xkcd:light purple', 'xkcd:dark grey', 'xkcd:light brown', 'xkcd:light lime green', 'xkcd:light navy blue', 'xkcd:light indigo', 'xkcd:olive yellow', 'xkcd:peach', 'xkcd:light lavender', 'xkcd:bright pink' ]
color_list_for_fqs = ['xkcd:blue', 'xkcd:green', 'xkcd:orange', 'xkcd:red', 'xkcd:purple', 'xkcd:black', 'xkcd:brown', 'xkcd:lime green', 'xkcd:navy blue', 'xkcd:indigo', 'xkcd:dark yellow', 'xkcd:dark peach', 'xkcd:lavender', 'xkcd:hot pink']
linestyle_list = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5)), (0, (1, 1))] * 5

fontsize = 13

for i in range(len(results_list)):
    np_results = np.stack([x[0] for x in results_list[i]])
    # print(np_results)
    print(np_results.shape)
    plot_with_conf_bounds(
        ax1, np_results, x_range, label=labels[i],
        color=color_list_for_fqs[i],
        linestyle=linestyle_list[i],
    )

ax1.set_xlabel("Number of Samples", fontsize=fontsize)
ax1.set_ylabel(r"Log Total Probability of Bad Output", fontsize=fontsize)
ax1.set_ylim(top=15)

# Adjust tick colors
# ax1.tick_params(axis="y", colors=color_list_for_fqs[0])
# ax2.tick_params(axis="y", colors=color_list_for_variances[0])
ax1.tick_params(axis='both', labelsize=fontsize)


# Combine legends
# fig.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.45, 0.5))
plt.legend(fontsize=fontsize)

# plt.legend()
plt.tight_layout()

figname = "./toyrlhf_logprobbad.pdf"
plt.savefig(figname)


plt.clf()

fig, ax1 = plt.subplots()

for i in range(len(results_list)):
    np_results = np.stack([x[1] for x in results_list[i]])
    # print(np_results)
    print(np_results.shape)
    plot_with_conf_bounds(
        ax1, np_results, x_range, label=labels[i],
        color=color_list_for_fqs[i],
        linestyle=linestyle_list[i],
    )

ax1.set_xlabel("Number of Samples", fontsize=fontsize)
ax1.set_ylabel(r"Average Reward", fontsize=fontsize)
ax1.tick_params(axis='both', labelsize=fontsize)

# Adjust tick colors
# ax1.tick_params(axis="y", colors=color_list_for_fqs[0])
# ax2.tick_params(axis="y", colors=color_list_for_variances[0])

# Combine legends
# fig.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.45, 0.5))
plt.legend(fontsize=fontsize)

# plt.legend()
plt.tight_layout()

figname = "./toyrlhf_rew.pdf"
plt.savefig(figname)


raise SystemExit(0)



# ax2 = ax1.twinx()

f_q_record_ctl_withneg = np.stack([x['F_q'] for x in results_ctl_withneg])
f_q_record_ctl_noneg = np.stack([x['F_q'] for x in results_ctl_noneg])
var_record_ctl_withneg = np.stack([x['Grad_variances'] for x in results_ctl_withneg])
var_record_ctl_noneg = np.stack([x['Grad_variances'] for x in results_ctl_noneg])

plot_with_conf_bounds(
    ax1, f_q_record_ctl_withneg, x_range, label=f"CTL: $E_q[\log (\sigma(s) / q(s))]$",
    color=color_list_for_fqs[0],
    linestyle=linestyle_list[0],
)
plot_with_conf_bounds(
    ax1, f_q_record_ctl_noneg, x_range, label=f"CTL (no second term): $E_q[\log (\sigma(s) / q(s))]$",
    color=color_list_for_fqs[1],
    linestyle=linestyle_list[0],
)
ax1.set_xlabel("Number of Samples")
ax1.set_ylabel(r"Lower Bound Estimate $(E_q[\log (\sigma(s) / q(s))])$")

# Adjust tick colors
# ax1.tick_params(axis="y", colors=color_list_for_fqs[0])
# ax2.tick_params(axis="y", colors=color_list_for_variances[0])

# Combine legends
fig.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.45, 0.5))

# plt.legend()
plt.tight_layout()

figname = "./toy_rlhf.pdf"
plt.savefig(figname)

raise SystemExit(0)


