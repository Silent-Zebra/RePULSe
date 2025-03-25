import torch
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt


load_dir = "../info"

metrics = ["f_q_estimates", "rewards", "kl_vals", "entropy"]


def populate_lists(load_prefixes):
    data_dict = {metric: [] for metric in metrics}

    for i in range(len(load_prefixes)):
        for metric in metrics:
            data_dict[metric].append([])

        for j in range(len(load_prefixes[i])):
            prefix = load_prefixes[i][j]
            loaded_data = torch.load(f"{load_dir}/{prefix}")

            for metric in metrics:
                data_dict[metric][i].append(loaded_data[metrics.index(metric)])

    return data_dict


def get_last_avg_and_conf_bound(f_q_estimates_list, i):
    f_q_estimates = np.stack(f_q_estimates_list[i], axis=0)
    f_q_estimates = f_q_estimates.mean(axis=-1)  # Average over all samples for each seed
    z_score = 1.96
    f_q_avg = f_q_estimates.mean(axis=0)  # Average over seeds
    f_q_stdev = np.std(f_q_estimates, axis=0, ddof=1)  # stdev across seeds
    conf_bound_f_q = z_score * f_q_stdev / np.sqrt(f_q_avg.shape[0])
    last_avg_f_q = f_q_avg[-1]
    last_conf_bound_f_q = conf_bound_f_q[-1]
    return last_avg_f_q, last_conf_bound_f_q


def make_table(load_prefixes, twist_learn_method_names, proposal_names, fig_name_modifier):
    print(f"----------Making table for {fig_name_modifier}----------")

    data_dict = populate_lists(load_prefixes)

    output_latex = []

    for i in range(len(load_prefixes)):
        metric_values = []

        for metric in metrics:
            last_avg, last_conf_bound = get_last_avg_and_conf_bound(data_dict[metric], i)
            metric_values.append(f"${last_avg:.1f} \pm {last_conf_bound:.1f}$")

        twist_learn_method_name = twist_learn_method_names[i]
        proposal_name = proposal_names[i]

        midrule = " \midrule"
        if i == 3:
            midrule = " \midrule \midrule"
        elif i == len(load_prefixes) - 1:
            midrule = ""

        tabularnewline = r"\tabularnewline"
        prop_and_twist = f"{proposal_name} & {twist_learn_method_name}"

        output_latex.append(f"{prop_and_twist} & {' & '.join(metric_values)} {tabularnewline} {midrule}")

    for x in output_latex:
        print(x)


load_prefixes_policy = [
    [
        "f_q_rew_kltoprior_ent_toy_rlhf_ctl_epochs1_lrscheduleconstant_actorlr3e-05_adambetas0.9_0.999_policy_seed1",
    ],
    [
        "f_q_rew_kltoprior_ent_toy_rlhf_ppo_epochs1_lrscheduleconstant_actorlr3e-05_criticlr1e-05_criticlossmse_adambetas0.9_0.999_policy_seed1",
    ],
    #         "f_q_rew_kltoprior_ent_toy_rlhf_ppo_epochs1_lrscheduleconstant_actorlr3e-05_criticlr0.0001_criticlossmse_adambetas0.9_0.999_policy_seed1"
    [
        "f_q_rew_kltoprior_ent_toy_rlhf_dpg_epochs1_lrscheduleconstant_actorlr3e-05_adambetas0.9_0.999_policy_seed1",
    ],
]
load_prefixes_linear = [
    [
        "f_q_rew_kltoprior_ent_toy_rlhf_ctl_epochs1_lrscheduleconstant_actorlr3e-05_adambetas0.9_0.999_policy_seed1",
    ],
    [
        "f_q_rew_kltoprior_ent_toy_rlhf_ctl_epochs1_lrscheduleconstant_actorlr0.0003_adambetas0.9_0.999_policy_seed1",

    ],
    [
        "f_q_rew_kltoprior_ent_toy_rlhf_ctl_epochs1_lrscheduleconstant_actorlr3e-05_adambetas0.9_0.999_policy_seed1",
    ],
]


twist_learn_method_names = [
    "Contrastive",
    "RL",
    "SIXO",
    "FUDGE",
    "--",
    "--",

]

# All constant LR for now
proposal_names = [
    "Twisted",
    "Twisted",
    "Twisted",
    "Twisted",
    "DPG",
    r"PPO (Original (TRL), 4 Steps, Shared Actor Critic, Adam $\beta_2=0.999$, No Dropout)",
]

fig_name_modifier = "03-24"

make_table(load_prefixes, twist_learn_method_names, proposal_names, fig_name_modifier)

