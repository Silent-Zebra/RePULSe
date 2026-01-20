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

from plot_utils import make_list, do_load_prefixes, generate_labels_from_prefixes

# Color and linestyle lists (defined early for use in plotting functions)
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


def extract_common_suffix(prefix):
    """
    Extracts the common suffix from any of the three prefix formats:
    - analytic_kls_toxicity_rlhf_...
    - analyticlogprob_rewsample_base_rlhf_...
    - analyticlogprob_rewsample_sampling_rlhf_...
    
    Returns the common suffix part (e.g., "rlhf_di_To_thmaisa_len1_...")
    """
    # Try to match analytic_kls_toxicity_ prefix
    if prefix.startswith("analytic_kls_toxicity_"):
        return prefix[len("analytic_kls_toxicity_"):]
    
    # Try to match analyticlogprob_rewsample_base_ prefix
    if prefix.startswith("analyticlogprob_rewsample_base_"):
        return prefix[len("analyticlogprob_rewsample_base_"):]
    
    # Try to match analyticlogprob_rewsample_sampling_ prefix
    if prefix.startswith("analyticlogprob_rewsample_sampling_"):
        return prefix[len("analyticlogprob_rewsample_sampling_"):]
    
    # Try to match old format analyticlogprob_rewsample_ (without base/sampling)
    if prefix.startswith("analyticlogprob_rewsample_"):
        return prefix[len("analyticlogprob_rewsample_"):]
    
    # If no match, return the original (for backward compatibility)
    return prefix


def build_prefix(common_suffix, prefix_type):
    """
    Builds the full prefix with the appropriate format.
    
    Args:
        common_suffix: The common suffix part (e.g., "rlhf_di_To_thmaisa_len1_...")
        prefix_type: One of "kls_toxicity", "logprob_base", "logprob_sampling"
    
    Returns:
        Full prefix string with appropriate format
    """
    if prefix_type == "kls_toxicity":
        return f"analytic_kls_toxicity_{common_suffix}"
    elif prefix_type == "logprob_base":
        return f"analyticlogprob_rewsample_base_{common_suffix}"
    elif prefix_type == "logprob_sampling":
        return f"analyticlogprob_rewsample_sampling_{common_suffix}"
    else:
        raise ValueError(f"Unknown prefix_type: {prefix_type}")


def transform_prefixes_for_file_type(load_prefixes_to_use, file_type_suffix):
    """
    Transforms a list of prefix lists to the appropriate file type format.
    
    Args:
        load_prefixes_to_use: List of lists of prefixes (from make_list calls)
        file_type_suffix: Either "base" or "sampling"
    
    Returns:
        Transformed list of lists of prefixes
    """
    transformed = []
    for prefix_list in load_prefixes_to_use:
        new_prefix_list = []
        for old_prefix in prefix_list:
            # Extract common suffix from each prefix individually
            common_suffix = extract_common_suffix(old_prefix)
            
            # Build new prefix with the target type
            prefix_type = f"logprob_{file_type_suffix}"
            new_prefix = build_prefix(common_suffix, prefix_type)
            new_prefix_list.append(new_prefix)
        
        transformed.append(new_prefix_list)
    
    return transformed


def transform_prefixes_for_kl(load_prefixes_to_use, file_type_suffix=None):
    """
    Transforms prefixes for KL divergence files.
    
    Args:
        load_prefixes_to_use: List of lists of prefixes
        file_type_suffix: Optional "base" or "sampling" (KL files may not have this suffix)
    
    Returns:
        Transformed list of lists of prefixes for KL files
    """
    transformed = []
    for prefix_list in load_prefixes_to_use:
        new_prefix_list = []
        for old_prefix in prefix_list:
            # Extract common suffix from each prefix individually
            common_suffix = extract_common_suffix(old_prefix)
            
            # Build KL prefix (without base/sampling suffix for now)
            new_prefix = build_prefix(common_suffix, "kls_toxicity")
            new_prefix_list.append(new_prefix)
        
        transformed.append(new_prefix_list)
    
    return transformed


def plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                           index_to_use=0, plot_name="logprobbad", 
                           ylabel=r"Log Total Probability of Bad Output",
                           file_type_suffix=""):
    """
    Generic function to plot results over time with confidence bounds.
    
    Args:
        results_list: List of lists of loaded data
        labels: List of labels for each series
        x_range: X-axis range
        fontsize: Font size for labels
        figname_modifier: Base name for the output file
        index_to_use: Index into the data tuple to plot
        plot_name: Name for the plot file
        ylabel: Y-axis label
        file_type_suffix: Suffix to add to filename ("base", "sampling", or "")
    """
    fig, ax1 = plt.subplots()

    for i in range(len(results_list)):
        if len(results_list[i]) == 0:
            continue
        np_results = np.stack([x[index_to_use] for x in results_list[i]])
        print(np_results.shape)
        plot_with_conf_bounds(
            ax1, np_results, x_range, label=labels[i],
            color=color_list_for_fqs[i],
            linestyle=linestyle_list[i],
        )
    ax1.set_xlabel("Number of Samples", fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    
    # Add suffix to filename if provided
    if file_type_suffix:
        figname = f"./{figname_modifier}_{file_type_suffix}_{plot_name}.pdf"
    else:
        figname = f"./{figname_modifier}_{plot_name}.pdf"
    plt.savefig(figname)
    plt.clf()


def process_file_type(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range, fontsize):
    """
    Process one file type (base or sampling) and generate all standard plots.
    
    Args:
        file_type_suffix: Either "base" or "sampling"
        load_prefixes_to_use: List of lists of prefixes
        labels: List of labels
        figname_modifier: Figure name modifier
        x_range: X-axis range
        fontsize: Font size
    """
    # Transform prefixes for this file type
    transformed_prefixes = transform_prefixes_for_file_type(load_prefixes_to_use, file_type_suffix)
    
    # Load data
    results_list = [[] for i in range(len(transformed_prefixes))]
    do_load_prefixes(results_list, transformed_prefixes)
    
    # Generate plots
    try:
        plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                              index_to_use=0, plot_name="logprobbad",
                              ylabel=r"Log Total Probability of Bad Output",
                              file_type_suffix=file_type_suffix)
    except:
        print(f"Failed to generate logprobbad plot for {file_type_suffix}")

    try:
        plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                              index_to_use=-2, plot_name="rew",
                              ylabel=r"Average Reward",
                              file_type_suffix=file_type_suffix)
    except:
        print(f"Failed to generate rew plot for {file_type_suffix}")

    try:
        plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                              index_to_use=-1, plot_name="untransformed_ret",
                              ylabel=r"Average Return",
                              file_type_suffix=file_type_suffix)
    except:
        print(f"Failed to generate untransformed_ret plot for {file_type_suffix}")

    return results_list


def plot_kl_divergences(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range, fontsize):
    """
    Plot KL divergence metrics from analytic_kls_toxicity files.
    
    Args:
        file_type_suffix: Either "base" or "sampling" (for filename suffix)
        load_prefixes_to_use: List of lists of prefixes
        labels: List of labels
        figname_modifier: Figure name modifier
        x_range: X-axis range
        fontsize: Font size
    """
    # Transform prefixes for KL files
    transformed_prefixes = transform_prefixes_for_kl(load_prefixes_to_use, file_type_suffix)
    
    # Load KL data
    kl_results_list = [[] for i in range(len(transformed_prefixes))]
    do_load_prefixes(kl_results_list, transformed_prefixes)
    
    # Check if we have any data
    has_data = any(len(kl_results_list[i]) > 0 for i in range(len(kl_results_list)))
    if not has_data:
        print(f"Warning: No KL divergence data found for {file_type_suffix}, skipping KL plots")
        return
    
    # Plot KL(sigma|q) = KL(target|proposal) from index 0
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=0, plot_name="kl_sigma_q", 
                          ylabel=r"KL($\sigma$|q) = KL(target|proposal)",
                          file_type_suffix=file_type_suffix)
    
    # Plot KL(q|sigma) = KL(proposal|target) from index 1
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=1, plot_name="kl_q_sigma", 
                          ylabel=r"KL(q|$\sigma$) = KL(proposal|target)",
                          file_type_suffix=file_type_suffix)


# Comment out/select as needed
figname_modifier = "toyrlhf_kl10_10_18_final"
figname_modifier = "toyrlhf_10_18_final"
# figname_modifier = "toyrepulse_01_18"
figname_modifier = "toyrepulse_01_19"


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



if "final" not in figname_modifier:

    if "repulse" in figname_modifier:

        load_prefixes_to_use = [
            # make_list(
            #     "analyticlogprob_rewsample_base_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.1_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_s1",
            #     1, 5),


            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.0001_cfsepnn_after_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_s2",
            #     1, 10),

            # for x in $(ls info/toyrepulse/ | grep analytic_kl | grep beta-10 | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done

            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count10.0_s2",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count1.0_s2",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_s2",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_s2",
                1, 10),

        ]

        # Use the same naming code from make_frontier.py
        labels = generate_labels_from_prefixes(load_prefixes_to_use)
        fontsize = 6

    else:

        labels = ['_'.join(a[0].split('len2_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for
                  a in load_prefixes_to_use]
        fontsize = 5



x_range = np.arange(51) * 10 * 500


# Process both base and sampling file types
for file_type_suffix in ["base", "sampling"]:
    print(f"\nProcessing {file_type_suffix} files...")
    process_file_type(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range, fontsize)

plot_kl_divergences("sampling", load_prefixes_to_use, labels, figname_modifier, x_range, fontsize)


raise SystemExit(0)
