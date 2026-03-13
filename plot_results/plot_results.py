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
import torch

from plot_utils import (
    make_list, do_load_prefixes, generate_labels_from_prefixes, to_scalar,
    compute_global_logZ_from_iwae_bounds, compute_approx_kl_from_f_q_g_q,
    generate_visual_style_from_prefixes,
    MARKER_NO_BONUS, MARKER_CFN, MARKER_MIXTURE, MARKER_EXACT_COUNT,
    MARKER_CTL, MARKER_CTLN, MARKER_LOSS_UNKNOWN,
)
from make_frontier import make_frontier_exact_kl_bootstrap

# Color and linestyle lists (defined early for use in plotting functions)
color_list_for_variances = ['xkcd:light blue', 'xkcd:light green', 'xkcd:light orange', 'xkcd:light red',
                            'xkcd:light purple', 'xkcd:dark grey', 'xkcd:light brown', 'xkcd:light lime green',
                            'xkcd:light navy blue', 'xkcd:light indigo', 'xkcd:olive yellow', 'xkcd:peach',
                            'xkcd:light lavender', 'xkcd:bright pink']
color_list_for_fqs = [
    'xkcd:green', 'xkcd:blue', 'xkcd:red', 'xkcd:orange',  'xkcd:purple',
    'xkcd:black',  'xkcd:gray',  'xkcd:light brown',
    'xkcd:pink', 'xkcd:gold', 'xkcd:teal', 'xkcd:magenta',
] * 5

linestyle_list = ['solid', 'dashed', 'dotted', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5)), (0, (1, 1))] * 5
marker_list_default = ["D", "x", "^", "o", "P", "v", "s", "p", "h", "*", "d", "8"] * 5


# MARKER_NO_BONUS, MARKER_CFN, MARKER_MIXTURE, MARKER_EXACT_COUNT are imported from plot_utils


def plot_with_conf_bounds(ax, record, x_range, label, **kwargs):
    avg = np.nanmean(record, axis=0)
    stdev = np.nanstd(record, axis=0, ddof=1)

    t_value = stats.t.ppf(0.975, df=record.shape[0] - 1)

    conf_bound = t_value * stdev / np.sqrt(record.shape[0])

    upper_conf_bound = avg + conf_bound
    lower_conf_bound = avg - conf_bound

    # Filter to non-NaN positions so lines connect across gaps (stretched)
    valid = ~np.isnan(avg)
    if not np.any(valid):
        return np.nan, np.nan
    x_valid = x_range[valid]
    avg_valid = avg[valid]
    upper_valid = upper_conf_bound[valid]
    lower_valid = lower_conf_bound[valid]

    ax.plot(x_valid, avg_valid, label=label, **kwargs)
    ax.fill_between(x_valid, lower_valid, upper_valid, alpha=0.3, **kwargs)

    return avg_valid[-1], conf_bound[valid][-1]


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


def plot_results_over_time(results_list, labels, x_range=None, fontsize=7, figname_modifier="",
                           index_to_use=0, plot_name="logprobbad",
                           ylabel=r"Log Total Probability of Bad Output",
                           file_type_suffix="", load_prefixes_to_use=None, output_dir=None):
    """
    Generic function to plot results over time with confidence bounds.

    Args:
        results_list: List of lists of loaded data
        labels: List of labels for each series
        x_range: X-axis range (optional, will be computed as simple integer indices if None)
        fontsize: Font size for labels
        figname_modifier: Base name for the output file
        index_to_use: Index into the data tuple to plot
        plot_name: Name for the plot file
        ylabel: Y-axis label
        file_type_suffix: Suffix to add to filename ("base", "sampling", or "")
        load_prefixes_to_use: Optional list of lists of prefixes (used for semantic styling)
    """
    fig, ax1 = plt.subplots()

    # First pass: determine max timesteps across all series if x_range not provided
    max_timesteps = 0
    if x_range is None:
        for i in range(len(results_list)):
            if len(results_list[i]) == 0:
                continue
            filtered_results = []
            for x in results_list[i]:
                if isinstance(x, tuple) and len(x) > index_to_use:
                    filtered_results.append(x[index_to_use])
                elif not isinstance(x, tuple):
                    filtered_results.append(x)
            if len(filtered_results) > 0:
                # Get shape of first result to determine timesteps
                first_result = filtered_results[0]
                if hasattr(first_result, 'shape'):
                    if len(first_result.shape) > 0:
                        max_timesteps = max(max_timesteps, first_result.shape[0])
                    else:
                        max_timesteps = max(max_timesteps, 1)
                elif isinstance(first_result, (list, tuple)):
                    max_timesteps = max(max_timesteps, len(first_result))
                else:
                    max_timesteps = max(max_timesteps, 1)

        if max_timesteps == 0:
            print("Warning: Could not determine number of timesteps from data, using default")
            max_timesteps = 51
        else:
            print(f"Detected max time steps: {max_timesteps}")

        x_range = np.arange(max_timesteps)


    # Generate semantic styling if prefixes are available, else fall back to index-based lists
    if load_prefixes_to_use is not None:
        semantic_colors, semantic_markers, semantic_linestyles = generate_visual_style_from_prefixes(load_prefixes_to_use)
    else:
        semantic_colors, semantic_markers, semantic_linestyles = color_list_for_fqs, marker_list_default, linestyle_list

    # Second pass: plot each series
    for i in range(len(results_list)):
        if len(results_list[i]) == 0:
            continue
        # Handle backward compatibility: check if index exists in tuple
        filtered_results = []
        for x in results_list[i]:
            if isinstance(x, tuple) and len(x) > index_to_use:
                value = x[index_to_use]
                # Skip None values
                if value is not None:
                    filtered_results.append(value)
            elif not isinstance(x, tuple):
                # Handle non-tuple data (shouldn't happen but be safe)
                if x is not None:
                    filtered_results.append(x)
        if len(filtered_results) == 0:
            continue
        np_results = _truncate_and_stack(filtered_results, context=f"series '{labels[i]}', plot '{plot_name}'")
        print(np_results.shape)

        # Ensure x_range matches the actual data length (must be np.array for boolean indexing in plot_with_conf_bounds)
        actual_timesteps = np_results.shape[1] if len(np_results.shape) > 1 else 1
        if len(x_range) != actual_timesteps:
            if actual_timesteps > len(x_range):
                x_range_adjusted = np.arange(actual_timesteps)
            else:
                x_range_adjusted = np.asarray(x_range[:actual_timesteps])
        else:
            x_range_adjusted = np.asarray(x_range)

        plot_with_conf_bounds(
            ax1, np_results, x_range_adjusted, label=labels[i],
            color=semantic_colors[i],
            linestyle=semantic_linestyles[i],
        )
    ax1.set_xlabel("Time Steps", fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.tick_params(axis='both', labelsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    
    # Add suffix to filename if provided
    if file_type_suffix:
        basename = f"{figname_modifier}_{file_type_suffix}_{plot_name}.pdf"
    else:
        basename = f"{figname_modifier}_{plot_name}.pdf"
    if output_dir is not None:
        figname = os.path.join(output_dir, basename)
    else:
        figname = f"./{basename}"
    plt.savefig(figname)
    plt.clf()


def process_file_type(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=7):
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
    # Pass transformed_prefixes (which preserve the parameter encoding) for auto-detection
    try:
        plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                              index_to_use=0, plot_name="logprobbad",
                              ylabel=r"Log Total Probability of Bad Output",
                              file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes)
    except:
        print(f"Failed to generate logprobbad plot for {file_type_suffix}")

    try:
        plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                              index_to_use=4, plot_name="rew",
                              ylabel=r"Average Reward",
                              file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes)
    except:
        print(f"Failed to generate rew plot for {file_type_suffix}")

    try:
        plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                              index_to_use=5, plot_name="untransformed_ret",
                              ylabel=r"Average Return",
                              file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes)
    except:
        print(f"Failed to generate untransformed_ret plot for {file_type_suffix}")
    
    # Plot threshold-based log probability of bad output (if available)
    try:
        # Check if data has threshold-based results (10 elements for base, 11 for sampling)
        has_threshold_data = False
        for result_group in results_list:
            if len(result_group) > 0 and len(result_group[0]) >= 10:
                has_threshold_data = True
                break
        
        if has_threshold_data:
            # For base: threshold data starts at index 6, for sampling: at index 7
            threshold_index = 7 if file_type_suffix == "sampling" else 6
            plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                                  index_to_use=threshold_index, plot_name="logprobbad_threshold",
                                  ylabel=r"Log Total Probability of Bad Output (Threshold-based)",
                                  file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes)
    except Exception as e:
        print(f"Failed to generate logprobbad_threshold plot for {file_type_suffix}: {e}")
    
    # Plot exploration bonus values (only for sampling actor, and only if available)
    if file_type_suffix == "sampling":
        try:
            # Check if data has 7 elements (with bonus) or 6 elements (without bonus)
            # Only plot if we have bonus data
            has_bonus_data = False
            for result_group in results_list:
                if len(result_group) > 0 and len(result_group[0]) >= 7:
                    has_bonus_data = True
                    break
            
            if has_bonus_data:
                plot_results_over_time(results_list, labels, x_range, fontsize, figname_modifier,
                                      index_to_use=6, plot_name="bonus",
                                      ylabel=r"Average Exploration Bonus",
                                      file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes)
        except Exception as e:
            print(f"Failed to generate bonus plot for {file_type_suffix}: {e}")

    return results_list


def plot_kl_divergences(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=7, n_frontiers=0, frontier_legendfontsize=None):
    """
    Plot KL divergence metrics from analytic_kls_toxicity files.

    Args:
        file_type_suffix: Either "base" or "sampling" (for filename suffix)
        load_prefixes_to_use: List of lists of prefixes
        labels: List of labels
        figname_modifier: Figure name modifier
        x_range: X-axis range
        fontsize: Font size
        n_frontiers: Number of frontier plots at evenly-spaced training checkpoints (0 to skip)
        frontier_legendfontsize: Legend font size for frontier plots (defaults to fontsize if None)
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

    # Save to subdirectory (consistent with sample-based f_q plots)
    output_dir = _make_output_dir(figname_modifier)

    # Plot KL(sigma|q) = KL(target|proposal) from index 0
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=0, plot_name="kl_sigma_q",
                          ylabel=r"KL($\sigma$|q) = KL(target|proposal)",
                          file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes,
                          output_dir=output_dir)

    # Plot KL(q|sigma) = KL(proposal|target) from index 1
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=1, plot_name="kl_q_sigma",
                          ylabel=r"KL(q|$\sigma$) = KL(proposal|target)",
                          file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes,
                          output_dir=output_dir)

    # Frontier plots at training checkpoints (use original prefixes for semantic styling)
    if n_frontiers > 0:
        _generate_kl_frontier_plots(
            kl_results_list, labels, load_prefixes_to_use, output_dir,
            n_frontiers=n_frontiers, fontsize=fontsize,
            frontier_legendfontsize=frontier_legendfontsize,
        )


def load_heldout_over_time_files(load_prefixes_to_use, load_dir="./info", map_location='cpu', threshold=-5):
    """
    Load heldout_over_time_* files (3-tuple or 4-tuple: list of reward tensors, list of return tensors, f_q_mean_list, optional target_samples_logprob_list).
    Convert to (reward_means, return_means, f_q_means, prob_bad_output, target_samples_logprob) per file for plotting.
    
    Args:
        load_prefixes_to_use: List of lists of filenames (e.g. heldout_over_time_OpenRLHF_xxx_s1)
        load_dir: Directory to load from
        map_location: Device for tensors
        threshold: Reward threshold for computing probability of bad output (default -5)
    
    Returns:
        List of lists of (reward_means, return_means, f_q_means, prob_bad_output, target_samples_logprob) as numpy arrays, one per seed per experiment.
    """
    loaded_data = []
    for prefix_list in load_prefixes_to_use:
        exp_data = []
        for fn in prefix_list:
            path = os.path.join(load_dir, fn)
            try:
                data = torch.load(path, map_location=map_location)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
            if not isinstance(data, (tuple, list)) or len(data) < 3:
                print(f"Warning: Expected at least 3-tuple for {path}, got {type(data)}. Skipping.")
                continue
            heldout_reward_list, heldout_return_list, f_q_mean_list = data[:3]
            # Check if 4th element (target_samples_logprob_list) exists
            target_samples_logprob_list = None
            if len(data) >= 4:
                target_samples_logprob_list = data[3]
            
            # Compute means per time point from tensors
            reward_means = np.array([t.float().mean().item() for t in heldout_reward_list])
            return_means = np.array([t.float().mean().item() for t in heldout_return_list])
            f_q_means = np.array(f_q_mean_list) if not isinstance(f_q_mean_list, np.ndarray) else f_q_mean_list
            # Compute probability of bad output (rewards < threshold) per time point
            prob_bad_output = np.array([(t.float().cpu().numpy() < threshold).mean() for t in heldout_reward_list])
            
            # Handle target_samples_logprob_list (can be list of floats or None)
            if target_samples_logprob_list is not None:
                target_samples_logprob = np.array(target_samples_logprob_list) if not isinstance(target_samples_logprob_list, np.ndarray) else target_samples_logprob_list
            else:
                target_samples_logprob = None
            
            exp_data.append((reward_means, return_means, f_q_means, prob_bad_output, target_samples_logprob))
        loaded_data.append(exp_data)

    return loaded_data


def plot_heldout_over_time(load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=7, load_dir="./info", threshold=-5):
    """
    Load heldout_over_time_* files and plot reward mean, return mean, f_q mean, probability of bad output, and target samples log probability over time.
    """
    loaded_data = load_heldout_over_time_files(load_prefixes_to_use, load_dir=load_dir, threshold=threshold)

    has_data = any(len(exp_data) > 0 for exp_data in loaded_data)
    if not has_data:
        print("Warning: No heldout_over_time data found, skipping plots")
        return

    # Plot reward mean over time
    plot_results_over_time(loaded_data, labels, x_range, fontsize, figname_modifier,
                          index_to_use=0, plot_name="heldout_reward",
                          ylabel=r"Heldout reward (mean)", load_prefixes_to_use=load_prefixes_to_use)
    # Plot return mean over time
    plot_results_over_time(loaded_data, labels, x_range, fontsize, figname_modifier,
                          index_to_use=1, plot_name="heldout_return",
                          ylabel=r"Heldout return (mean)", load_prefixes_to_use=load_prefixes_to_use)
    # Plot f_q mean over time
    plot_results_over_time(loaded_data, labels, x_range, fontsize, figname_modifier,
                          index_to_use=2, plot_name="heldout_f_q",
                          ylabel=r"$f_q$ (mean)", load_prefixes_to_use=load_prefixes_to_use)
    # Plot probability of bad output over time
    plot_results_over_time(loaded_data, labels, x_range, fontsize, figname_modifier,
                          index_to_use=3, plot_name="heldout_prob_bad_output",
                          ylabel=f"Probability of bad output (reward < {threshold})", load_prefixes_to_use=load_prefixes_to_use)
    # Plot target samples log probability over time (if available)
    # Check if any experiment has target_samples_logprob data
    has_target_logprob = False
    for exp_data in loaded_data:
        for seed_data in exp_data:
            if len(seed_data) >= 5 and seed_data[4] is not None:
                has_target_logprob = True
                break
        if has_target_logprob:
            break
    
    if has_target_logprob:
        plot_results_over_time(loaded_data, labels, x_range, fontsize, figname_modifier,
                              index_to_use=4, plot_name="heldout_target_samples_logprob",
                              ylabel=r"Log probability of target samples (logsumexp)", load_prefixes_to_use=load_prefixes_to_use)


def _reconstruct_aligned_agg_from_per_prompt(per_prompt_list):
    """Reconstruct an aligned aggregated list from per-prompt data.

    per_prompt_list is a list-of-lists: outer = timesteps, inner = one value per prompt
    (tensor or None). Returns a list of length len(per_prompt_list) where each entry is
    either the concatenated non-None tensors for that timestep, or None.
    """
    aligned = []
    for per_prompt in per_prompt_list:
        valid = [x for x in per_prompt if x is not None]
        aligned.append(torch.cat(valid) if valid else None)
    return aligned


def _extract_f_q_g_q_from_loaded(data, path=""):
    """Extract (f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list) from loaded data.

    Handles both v1 (tuple/list of 4) and v2 (dict with 'version': 2) formats.
    Returns the tuple, or None if format is unrecognized.
    """
    if isinstance(data, dict) and data.get("version", 1) >= 2:
        # v2 dict format: extract aggregated lists
        f_q = data.get("f_q_estimates_list", [])
        g_q = data.get("g_q_estimates_list", [])
        iwae_lbs = data.get("iwae_lbs_list", [])
        iwae_ubs = data.get("iwae_ubs_list", [])

        # Fix misaligned aggregated lists from old saves that skipped appending None.
        # The per-prompt lists are always aligned, so reconstruct from them.
        if len(f_q) != len(g_q):
            g_q_per_prompt = data.get("g_q_by_prompt_fixed")
            if g_q_per_prompt is not None and len(g_q_per_prompt) == len(f_q):
                print(f"Note: Reconstructing aligned g_q from per-prompt data "
                      f"(f_q has {len(f_q)} entries, g_q had {len(g_q)})")
                g_q = _reconstruct_aligned_agg_from_per_prompt(g_q_per_prompt)
            else:
                print(f"Warning: f_q ({len(f_q)}) and g_q ({len(g_q)}) list lengths differ "
                      f"and per-prompt data unavailable for reconstruction in {path}")
        if len(f_q) != len(iwae_lbs):
            iwae_lbs_per_prompt = data.get("iwae_lbs_by_prompt_fixed")
            if iwae_lbs_per_prompt is not None and len(iwae_lbs_per_prompt) == len(f_q):
                print(f"Note: Reconstructing aligned iwae_lbs from per-prompt data")
                iwae_lbs = []
                for per_prompt in iwae_lbs_per_prompt:
                    valid = [x for x in per_prompt if x is not None]
                    iwae_lbs.append(torch.tensor(valid) if valid else None)
        if len(f_q) != len(iwae_ubs):
            iwae_ubs_per_prompt = data.get("iwae_ubs_by_prompt_fixed")
            if iwae_ubs_per_prompt is not None and len(iwae_ubs_per_prompt) == len(f_q):
                print(f"Note: Reconstructing aligned iwae_ubs from per-prompt data")
                iwae_ubs = []
                for per_prompt in iwae_ubs_per_prompt:
                    valid = [x for x in per_prompt if x is not None]
                    iwae_ubs.append(torch.tensor(valid) if valid else None)

        return (f_q, g_q, iwae_lbs, iwae_ubs)
    elif isinstance(data, (tuple, list)) and len(data) >= 4:
        return data[:4]
    else:
        print(f"Warning: Unrecognized format for {path}, got {type(data)}. Skipping.")
        return None


def v2_data_to_aggregated(all_v2_data):
    """Convert v2 loaded data (from _load_v2_f_q_g_q_files) to the aggregated format
    used by plot_f_q_g_q_kl_divergences.

    Returns:
        List of lists of tuples (f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list),
        same format as load_f_q_g_q_files_over_time.
    """
    loaded_data = []
    for exp_data in all_v2_data:
        exp_agg = []
        for v2_data in exp_data:
            extracted = _extract_f_q_g_q_from_loaded(v2_data)
            if extracted is not None:
                exp_agg.append(extracted)
        loaded_data.append(exp_agg)
    return loaded_data


def load_f_q_g_q_files_over_time(load_prefixes_to_use, load_dir="./info", map_location='cpu'):
    """
    Load f_q/g_q/IWAE bound files and extract time-series data.

    Args:
        load_prefixes_to_use: List of lists of filenames; inner list = one experiment,
                              each filename = one seed (full basename, e.g. f_q_g_q_iwae_bounds_OpenRLHF_..._s1)
        load_dir: Directory to load files from
        map_location: Device to load tensors to (default 'cpu')

    Returns:
        List of lists, where each inner list contains loaded data tuples per seed.
        Each tuple is (f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list)
    """
    loaded_data = []

    for prefix_list in load_prefixes_to_use:
        exp_data = []
        for fn in prefix_list:
            path = os.path.join(load_dir, fn)
            try:
                data = torch.load(path, map_location=map_location)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
            extracted = _extract_f_q_g_q_from_loaded(data, path)
            if extracted is None:
                continue
            f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list = extracted
            exp_data.append((f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list))
        loaded_data.append(exp_data)

    return loaded_data


def compute_approx_kl_over_time(loaded_data, global_logZ):
    """
    Compute approximate KL divergence estimates over time from loaded f_q/g_q data.
    
    Args:
        loaded_data: List of experiments, where each experiment is a list of seeds.
                     Each seed contains a tuple (f_q_estimates_list, g_q_estimates_list, 
                     iwae_lbs_list, iwae_ubs_list)
        global_logZ: Global log Z value (float)
    
    Returns:
        List of lists of tuples, where each tuple is (kl_sigma_q_array, kl_q_sigma_array)
        per seed, compatible with plot_results_over_time format
    """
    results_list = []
    
    for exp_data in loaded_data:
        row = []

        for f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list in exp_data:
            if len(f_q_estimates_list) == 0 or len(g_q_estimates_list) == 0:
                continue

            print(len(iwae_lbs_list))
            # Compute KL for each timestep; insert NaN where f_q or g_q is None
            # (can occur in fixed trajectory setting without target samples for some steps).
            # NaN preserves correct timestep indexing so plots have proper x-axis spacing.
            T = len(f_q_estimates_list)
            kl_sigma_q_per_t = np.full(T, np.nan)
            kl_q_sigma_per_t = np.full(T, np.nan)

            for t in range(T):
                if f_q_estimates_list[t] is None or t >= len(g_q_estimates_list) or g_q_estimates_list[t] is None:
                    continue
                f_q_t = to_scalar(f_q_estimates_list[t])
                g_q_t = to_scalar(g_q_estimates_list[t])

                kl_sigma_q_t, kl_q_sigma_t = compute_approx_kl_from_f_q_g_q(
                    f_q_t, g_q_t, global_logZ
                )
                kl_sigma_q_per_t[t] = kl_sigma_q_t
                kl_q_sigma_per_t[t] = kl_q_sigma_t

            if np.all(np.isnan(kl_sigma_q_per_t)):
                continue
            row.append((kl_sigma_q_per_t, kl_q_sigma_per_t))
        results_list.append(row)
    
    return results_list


def extract_f_q_over_time(loaded_data):
    """
    Extract f_q estimates over time from loaded f_q/g_q data, including all timesteps
    (even those where g_q is None).

    Args:
        loaded_data: List of experiments, where each experiment is a list of seeds.
                     Each seed contains a tuple (f_q_estimates_list, g_q_estimates_list,
                     iwae_lbs_list, iwae_ubs_list)

    Returns:
        List of lists of np.arrays (one per seed), compatible with plot_results_over_time format.
        Each array has shape (T,) with the scalar f_q at each timestep.
    """
    results_list = []
    for exp_data in loaded_data:
        row = []
        for f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list in exp_data:
            if len(f_q_estimates_list) == 0:
                continue
            f_q_per_t = []
            for t in range(len(f_q_estimates_list)):
                if f_q_estimates_list[t] is None:
                    continue
                f_q_per_t.append(to_scalar(f_q_estimates_list[t]))
            if len(f_q_per_t) == 0:
                continue
            row.append(np.array(f_q_per_t))
        results_list.append(row)
    return results_list


def _generate_kl_frontier_plots(
    kl_results_list, labels, load_prefixes_to_use, output_dir,
    n_frontiers=1, fontsize=7, frontier_legendfontsize=None,
    filename_prefix="", skip_individual_timestep_plots=False,
    skip_combined_frontier=False, skip_time_avg_halves=False,
):
    """Generate KL frontier plots (individual per-timestep + combined) from KL results.

    Args:
        kl_results_list: Standard format — kl_results_list[exp_i] is a list of
            (kl_sigma_q_array, kl_q_sigma_array) tuples, one per seed.
            Each array has shape (T,).
        labels: List of labels for each experiment.
        load_prefixes_to_use: List of lists of prefixes (used for semantic styling).
        output_dir: Directory to save PDF plots.
        n_frontiers: Number of frontier plots at evenly-spaced training checkpoints.
        fontsize: Font size for plots.
        frontier_legendfontsize: Legend font size (defaults to fontsize if None).
    """
    if n_frontiers <= 0:
        return

    print(f"\nGenerating {n_frontiers} frontier plot(s)...")
    semantic_colors, semantic_markers, semantic_linestyles = generate_visual_style_from_prefixes(load_prefixes_to_use)
    effective_frontier_legendfontsize = frontier_legendfontsize if frontier_legendfontsize is not None else fontsize

    # Determine max trajectory length across all experiments and seeds
    # Each element is a tuple/list; index 0 = kl_sigma_q, index 1 = kl_q_sigma (may have extra elements)
    max_T = 0
    for exp_data in kl_results_list:
        for seed_item in exp_data:
            kl_sigma_q_arr, kl_q_sigma_arr = seed_item[0], seed_item[1]
            max_T = max(max_T, len(kl_sigma_q_arr), len(kl_q_sigma_arr))

    if max_T == 0:
        print("Warning: max_T=0, skipping frontier plots")
        return

    frontier_indices = [round((max_T - 1) * i / n_frontiers)
                        for i in range(1, n_frontiers + 1)]
    # Deduplicate while preserving order
    seen = set()
    unique_frontier_indices = []
    for idx in frontier_indices:
        if idx not in seen:
            seen.add(idx)
            unique_frontier_indices.append(idx)
    frontier_indices = unique_frontier_indices

    n_exp = len(kl_results_list)
    all_frontier_results = {}  # t_idx -> frontier_results list

    for t_idx in frontier_indices:
        frontier_results = []  # frontier_results[exp_i] = list of (kl_sigma_q, kl_q_sigma) per seed
        for exp_data in kl_results_list:
            seed_data = []
            for seed_item in exp_data:
                kl_sigma_q_arr, kl_q_sigma_arr = seed_item[0], seed_item[1]
                if t_idx >= len(kl_sigma_q_arr) or t_idx >= len(kl_q_sigma_arr):
                    continue  # seed hasn't reached this timestep
                # make_frontier_exact_kl_bootstrap expects t[0] = kl_sigma_q, t[1] = kl_q_sigma
                seed_data.append((np.array([kl_sigma_q_arr[t_idx]]), np.array([kl_q_sigma_arr[t_idx]])))
            frontier_results.append(seed_data)
        all_frontier_results[t_idx] = frontier_results

        # Individual frontier plot for this time step
        if not skip_individual_timestep_plots:
            make_frontier_exact_kl_bootstrap(
                xlabel=r"KL($\sigma$|q)", ylabel=r"KL(q|$\sigma$)",
                figname=os.path.join(output_dir, f"{filename_prefix}frontier_kl_t{t_idx}.pdf"),
                labels=labels, results_list=frontier_results,
                color_list=semantic_colors, marker_list=semantic_markers,
                aggregate_seeds=True, fontsize=fontsize,
                legendfontsize=effective_frontier_legendfontsize,
            )

    # Combined frontier plot: all time steps in one figure
    if len(frontier_indices) > 1 and not skip_combined_frontier:
        print(f"Generating combined frontier plot across {len(frontier_indices)} time steps...")
        combined_results = []
        combined_labels = []
        combined_colors = []
        combined_markers = []
        combined_sizes = []
        combined_groups = []

        max_frontier_t = max(frontier_indices)
        max_marker_size = 50  # matplotlib scatter 's' (area in points^2); default is ~36
        for t_pos, t_idx in enumerate(frontier_indices):
            marker_size = t_idx / max_frontier_t * max_marker_size
            for exp_i in range(n_exp):
                combined_results.append(all_frontier_results[t_idx][exp_i])
                combined_labels.append(labels[exp_i])
                combined_colors.append(semantic_colors[exp_i])
                combined_markers.append(semantic_markers[exp_i])
                combined_sizes.append(marker_size)
                combined_groups.append(exp_i)

        make_frontier_exact_kl_bootstrap(
            xlabel=r"KL($\sigma$|q)", ylabel=r"KL(q|$\sigma$)",
            figname=os.path.join(output_dir, f"{filename_prefix}frontier_kl_combined.pdf"),
            labels=combined_labels, results_list=combined_results,
            color_list=combined_colors, marker_list=combined_markers,
            aggregate_seeds=True, fontsize=fontsize,
            legendfontsize=effective_frontier_legendfontsize,
            size_list=combined_sizes,
            connect_groups=combined_groups,
        )

    # Time-averaged frontier plots: full, first 50%, last 50%
    def _make_time_avg_frontier(frac_start, frac_end, suffix, description):
        """Build and plot a time-averaged frontier over a fractional slice of timesteps."""
        print(f"Generating {description} frontier plot...")
        avg_results = []
        for exp_data in kl_results_list:
            seed_data = []
            for seed_item in exp_data:
                kl_sigma_q_arr, kl_q_sigma_arr = seed_item[0], seed_item[1]
                T = len(kl_sigma_q_arr)
                t_start = int(round(T * frac_start))
                t_end = int(round(T * frac_end))
                if t_end <= t_start:
                    continue
                avg_sigma_q = np.nanmean(kl_sigma_q_arr[t_start:t_end])
                avg_q_sigma = np.nanmean(kl_q_sigma_arr[t_start:t_end])
                if np.isnan(avg_sigma_q) or np.isnan(avg_q_sigma):
                    continue
                seed_data.append((np.array([avg_sigma_q]), np.array([avg_q_sigma])))
            avg_results.append(seed_data)

        if any(len(sd) > 0 for sd in avg_results):
            make_frontier_exact_kl_bootstrap(
                xlabel=r"KL($\sigma$|q)", ylabel=r"KL(q|$\sigma$)",
                figname=os.path.join(output_dir, f"{filename_prefix}frontier_kl_{suffix}.pdf"),
                labels=labels, results_list=avg_results,
                color_list=semantic_colors, marker_list=semantic_markers,
                aggregate_seeds=True, fontsize=fontsize,
                legendfontsize=effective_frontier_legendfontsize,
            )
        else:
            print(f"Warning: No data for {description} frontier plot")

    _make_time_avg_frontier(0.0, 1.0, "time_avg", "time-averaged (all steps)")
    if not skip_time_avg_halves:
        _make_time_avg_frontier(0.0, 0.5, "time_avg_first_half", "time-averaged (first 50%)")
        _make_time_avg_frontier(0.5, 1.0, "time_avg_last_half", "time-averaged (last 50%)")


def plot_f_q_g_q_kl_divergences(load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=7, load_dir="./info", n_frontiers=1, frontier_legendfontsize=None, preloaded_data=None):
    """
    Plot approximate KL divergence metrics over time from f_q/g_q/IWAE bound files.

    Args:
        load_prefixes_to_use: List of lists of prefixes for f_q_g_q files
        labels: List of labels for each series
        figname_modifier: Figure name modifier
        x_range: X-axis range
        fontsize: Font size
        load_dir: Directory to load files from
        n_frontiers: Number of frontier plots to generate at evenly-spaced training checkpoints
        preloaded_data: Optional pre-loaded data in the same format as load_f_q_g_q_files_over_time returns.
                        If provided, skips loading from disk.
    """
    # Load f_q/g_q files (or use preloaded data)
    if preloaded_data is not None:
        loaded_data = preloaded_data
    else:
        loaded_data = load_f_q_g_q_files_over_time(load_prefixes_to_use, load_dir=load_dir)

    # Check if we have any data
    has_data = any(len(exp_data) > 0 for exp_data in loaded_data)
    if not has_data:
        print(f"Warning: No f_q/g_q data found, skipping KL plots")
        return

    # Create output subfolder for plots
    output_dir = _make_output_dir(figname_modifier)

    # Plot f_q over time (includes all timesteps, even those without g_q)
    f_q_results_list = extract_f_q_over_time(loaded_data)
    has_f_q = any(len(row) > 0 for row in f_q_results_list)
    if has_f_q:
        plot_results_over_time(f_q_results_list, labels, x_range, fontsize, figname_modifier,
                              index_to_use=0, plot_name="f_q",
                              ylabel=r"$f_q$ (ELBO)",
                              file_type_suffix="", load_prefixes_to_use=load_prefixes_to_use,
                              output_dir=output_dir)
    else:
        print("Warning: No f_q data found, skipping f_q plot")

    # Compute global log Z (needed for KL plots and frontiers)
    try:
        global_logZ = compute_global_logZ_from_iwae_bounds(loaded_data)
        print(f"Global log Z: {global_logZ}")
    except ValueError as e:
        print(f"Warning: Failed to compute global log Z: {e}, skipping KL plots")
        return

    # Compute KL estimates over time
    # kl_results_list[exp_i] = list of (kl_sigma_q_array, kl_q_sigma_array) per seed
    kl_results_list = compute_approx_kl_over_time(loaded_data, global_logZ)

    # Check if we have any results
    has_results = any(len(row) > 0 for row in kl_results_list)
    if not has_results:
        print(f"Warning: No KL results computed, skipping KL plots")
        return

    # Plot KL(sigma|q) = KL(target|proposal) from index 0
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=0, plot_name="kl_sigma_q",
                          ylabel=r"KL($\sigma$|q) = KL(target|proposal)",
                          file_type_suffix="", load_prefixes_to_use=load_prefixes_to_use,
                          output_dir=output_dir)

    # Plot KL(q|sigma) = KL(proposal|target) from index 1
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=1, plot_name="kl_q_sigma",
                          ylabel=r"KL(q|$\sigma$) = KL(proposal|target)",
                          file_type_suffix="", load_prefixes_to_use=load_prefixes_to_use,
                          output_dir=output_dir)

    # Frontier plots at training checkpoints
    _generate_kl_frontier_plots(
        kl_results_list, labels, load_prefixes_to_use, output_dir,
        n_frontiers=n_frontiers, fontsize=fontsize,
        frontier_legendfontsize=frontier_legendfontsize,
    )


TRUNCATE_DROP_THRESHOLD = 0.75  # Drop arrays shorter than this fraction of the reference max length


def _truncate_and_stack(arrays, context="", global_max_len=None):
    """Stack 1D arrays: drop those below TRUNCATE_DROP_THRESHOLD of max length, truncate the rest to shortest remaining.

    If global_max_len is provided, it is used instead of the local max for computing the drop threshold.
    """
    assert len(arrays) > 0, f"_truncate_and_stack called with empty list{' (' + context + ')' if context else ''}"
    max_len = max(len(a) for a in arrays)
    ref_max = global_max_len if global_max_len is not None else max_len
    threshold = ref_max * TRUNCATE_DROP_THRESHOLD
    kept = []
    for i, a in enumerate(arrays):
        if len(a) < threshold:
            ctx = f" ({context})" if context else ""
            print(f"Warning: dropping incomplete data at index {i} (length {len(a)} < {TRUNCATE_DROP_THRESHOLD:.0%} of ref_max {ref_max}){ctx}")
        else:
            kept.append(a)
    assert len(kept) > 0, (
        f"All arrays were incomplete and dropped{' (' + context + ')' if context else ''}! "
        f"Lengths: {[len(a) for a in arrays]}"
    )
    min_len = min(len(a) for a in kept)
    if min_len < max_len:
        ctx = f" ({context})" if context else ""
        print(f"Truncating {len(kept)} arrays from max {max_len} to min {min_len}{ctx}")
    return np.stack([a[:min_len] for a in kept])


def _truncate_and_stack_2d(matrices, context="", global_max_cols=None):
    """Stack 2D arrays: drop those below TRUNCATE_DROP_THRESHOLD of max cols, truncate the rest to shortest remaining.

    If global_max_cols is provided, it is used instead of the local max for computing the drop threshold.
    """
    assert len(matrices) > 0, f"_truncate_and_stack_2d called with empty list{' (' + context + ')' if context else ''}"
    max_cols = max(m.shape[1] for m in matrices)
    ref_max = global_max_cols if global_max_cols is not None else max_cols
    threshold = ref_max * TRUNCATE_DROP_THRESHOLD
    kept = []
    for i, m in enumerate(matrices):
        if m.shape[1] < threshold:
            ctx = f" ({context})" if context else ""
            print(f"Warning: dropping incomplete 2D data at index {i} (cols {m.shape[1]} < {TRUNCATE_DROP_THRESHOLD:.0%} of ref_max {ref_max}){ctx}")
        else:
            kept.append(m)
    assert len(kept) > 0, (
        f"All matrices were incomplete and dropped{' (' + context + ')' if context else ''}! "
        f"Column counts: {[m.shape[1] for m in matrices]}"
    )
    min_cols = min(m.shape[1] for m in kept)
    if min_cols < max_cols:
        ctx = f" ({context})" if context else ""
        print(f"Truncating {len(kept)} matrices from max {max_cols} cols to min {min_cols}{ctx}")
    return np.stack([m[:, :min_cols] for m in kept])


def _make_output_dir(figname_modifier):
    """Create and return an output directory under figs/<figname_modifier>/."""
    output_dir = os.path.join("figs", figname_modifier)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}/")
    return output_dir


def _sanitize_for_filename(text, max_len=50):
    """Sanitize a string for use in filenames."""
    import re as _re
    sanitized = _re.sub(r'[^\w\s-]', '', text)
    sanitized = _re.sub(r'\s+', '_', sanitized)
    return sanitized[:max_len]


def _load_and_reduce_v2_file(path):
    """Load a single v2 f_q/g_q file and pre-reduce per-sample tensors to scalars.

    The raw files store full (N,)-shaped tensors per prompt per timestep for f_q and g_q,
    but the plotting code only needs the mean (scalar). Pre-reducing at load time avoids
    keeping large tensors in memory and speeds up downstream computation.

    Also drops aggregated backward-compat keys that the multiprompt plotting path doesn't use.
    """
    data = torch.load(path, map_location='cpu')
    if not (isinstance(data, dict) and data.get("version", 1) >= 2):
        return None

    # Pre-reduce f_q_by_prompt_fixed: list-of-lists of tensors -> list-of-lists of scalars
    for key in ("f_q_by_prompt_fixed", "g_q_by_prompt_fixed", "f_q_by_prompt_random"):
        if key in data:
            data[key] = [
                [to_scalar(x) if x is not None else None for x in timestep_list]
                for timestep_list in data[key]
            ]

    return data


def _load_v2_f_q_g_q_files(load_prefixes_to_use, load_dir="./info"):
    """Load v2 format f_q/g_q files. Returns list of experiments, each is list of v2 dicts.

    Files are loaded in parallel and per-sample tensors are pre-reduced to scalars.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build flat list of (exp_index, seed_index, path) for parallel loading
    load_tasks = []
    for exp_i, prefix_list in enumerate(load_prefixes_to_use):
        for seed_j, fn in enumerate(prefix_list):
            load_tasks.append((exp_i, seed_j, fn, os.path.join(load_dir, fn)))

    # Load files in parallel
    results = {}  # (exp_i, seed_j) -> data
    with ThreadPoolExecutor() as executor:
        future_to_key = {
            executor.submit(_load_and_reduce_v2_file, path): (exp_i, seed_j, fn)
            for exp_i, seed_j, fn, path in load_tasks
        }
        for future in as_completed(future_to_key):
            exp_i, seed_j, fn = future_to_key[future]
            try:
                data = future.result()
            except Exception as e:
                print(f"Warning: Failed to load {fn}: {e}")
                continue
            if data is None:
                print(f"Warning: {fn} is not v2 format, skipping for multiprompt plotting")
                continue
            results[(exp_i, seed_j)] = data

    # Reconstruct list-of-lists structure, preserving seed order
    all_v2_data = []
    loaded_seed_indices = []  # loaded_seed_indices[exp_i] = list of original seed indices that loaded
    for exp_i, prefix_list in enumerate(load_prefixes_to_use):
        exp_data = []
        exp_seeds = []
        for seed_j in range(len(prefix_list)):
            if (exp_i, seed_j) in results:
                exp_data.append(results[(exp_i, seed_j)])
                exp_seeds.append(seed_j)
        all_v2_data.append(exp_data)
        loaded_seed_indices.append(exp_seeds)
    return all_v2_data, loaded_seed_indices


def _get_prompts_with_targets(target_samples_path):
    """Load target samples file and return set of prompt texts that have at least 1 target sample."""
    raw = torch.load(target_samples_path, map_location='cpu')
    if isinstance(raw, dict) and raw.get("version", 1) >= 2:
        prompt_texts = raw["prompt_texts"]
        samples_by_prompt = raw["samples_by_prompt"]
        prompts_with_targets = set()
        for prompt_text, samples in zip(prompt_texts, samples_by_prompt):
            if len(samples) > 0:
                prompts_with_targets.add(prompt_text)
        print(f"Target samples: {len(prompts_with_targets)}/{len(prompt_texts)} prompts have target samples")
        return prompts_with_targets
    else:
        raise ValueError("Target samples must be v2 format for multiprompt plotting")


def _infer_prompts_with_targets_from_v2(all_v2_data):
    """Infer which prompts have target data from v2 IWAE bounds.

    Returns set of prompt texts that have at least one non-None IWAE bound
    in any experiment/seed/timestep.
    """
    prompt_texts = None
    prompts_with_bounds = set()

    for exp_data in all_v2_data:
        for v2_data in exp_data:
            if prompt_texts is None:
                prompt_texts = v2_data.get("prompt_texts_fixed")
            if prompt_texts is None:
                continue
            for key in ("iwae_lbs_by_prompt_fixed", "iwae_ubs_by_prompt_fixed"):
                for t_list in v2_data.get(key, []):
                    for p, val in enumerate(t_list):
                        if val is not None and p < len(prompt_texts):
                            prompts_with_bounds.add(prompt_texts[p])

    if prompt_texts is not None:
        print(f"Inferred from v2 data: {len(prompts_with_bounds)}/{len(prompt_texts)} prompts have IWAE bounds")
    return prompts_with_bounds


def _compute_global_per_prompt_log_Z(all_v2_data, prompts_with_targets):
    """
    Compute per-prompt log Z from IWAE bounds pooled across ALL experiments, seeds, and timesteps.

    Takes the max LB and min UB across all experiments/seeds/timesteps for each prompt,
    giving the tightest bounds and thus the best midpoint estimate. This ensures a single
    fixed log Z per prompt that is consistent across experiments.

    Returns dict: prompt_index -> log_Z_p (float), only for prompts in prompts_with_targets
    that have non-None IWAE bounds in at least one experiment.
    """
    # Collect all IWAE bounds per prompt across all experiments and seeds
    all_lbs_by_prompt = {}  # prompt_index -> list of all LB values
    all_ubs_by_prompt = {}  # prompt_index -> list of all UB values
    prompt_texts = None

    for exp_data in all_v2_data:
        for v2_data in exp_data:
            if prompt_texts is None:
                prompt_texts = v2_data["prompt_texts_fixed"]
            iwae_lbs_by_prompt = v2_data.get("iwae_lbs_by_prompt_fixed", [])
            iwae_ubs_by_prompt = v2_data.get("iwae_ubs_by_prompt_fixed", [])
            T = len(iwae_lbs_by_prompt)

            for p in range(len(prompt_texts)):
                for t in range(T):
                    if p < len(iwae_lbs_by_prompt[t]) and iwae_lbs_by_prompt[t][p] is not None:
                        all_lbs_by_prompt.setdefault(p, []).append(iwae_lbs_by_prompt[t][p])
                    if t < len(iwae_ubs_by_prompt) and p < len(iwae_ubs_by_prompt[t]) and iwae_ubs_by_prompt[t][p] is not None:
                        all_ubs_by_prompt.setdefault(p, []).append(iwae_ubs_by_prompt[t][p])

    if prompt_texts is None:
        return {}

    log_Z_by_prompt = {}
    for p in range(len(prompt_texts)):
        if prompt_texts[p] not in prompts_with_targets:
            continue

        lbs = all_lbs_by_prompt.get(p, [])
        ubs = all_ubs_by_prompt.get(p, [])

        if not lbs or not ubs:
            continue

        max_lb = max(lbs)
        min_ub = min(ubs)
        log_Z_p = (max_lb + min_ub) / 2.0
        log_Z_by_prompt[p] = log_Z_p


    return log_Z_by_prompt


def _compute_per_prompt_kl_over_time(v2_data, log_Z_by_prompt):
    """
    Compute per-prompt KL divergences over time.

    Returns:
        dict: prompt_index -> (kl_q_sigma_array, kl_sigma_q_array) where arrays have shape (T,).
              Entries are NaN where data is unavailable.
    """
    f_q_by_prompt = v2_data.get("f_q_by_prompt_fixed", [])
    g_q_by_prompt = v2_data.get("g_q_by_prompt_fixed", [])

    T = len(f_q_by_prompt)
    result = {}

    for p, log_Z_p in log_Z_by_prompt.items():
        kl_q_sigma = np.full(T, np.nan)
        kl_sigma_q = np.full(T, np.nan)

        for t in range(T):
            if p < len(f_q_by_prompt[t]) and f_q_by_prompt[t][p] is not None:
                # Values are pre-reduced to scalars during loading
                kl_q_sigma[t] = log_Z_p - f_q_by_prompt[t][p]

            if (t < len(g_q_by_prompt) and p < len(g_q_by_prompt[t])
                    and g_q_by_prompt[t][p] is not None):
                kl_sigma_q[t] = g_q_by_prompt[t][p] - log_Z_p

        result[p] = (kl_q_sigma, kl_sigma_q)

    return result


def _compute_random_f_q_over_time(v2_data):
    """
    Compute average f_q over time for the random prompt set.

    Returns:
        np.array of shape (T_random,) with mean f_q per timestep, or None if no data.
    """
    f_q_by_prompt_random = v2_data.get("f_q_by_prompt_random", [])
    if not f_q_by_prompt_random:
        return None

    avg_f_q = []
    for t in range(len(f_q_by_prompt_random)):
        # Values are pre-reduced to scalars during loading
        per_prompt_means = [x for x in f_q_by_prompt_random[t] if x is not None]
        if per_prompt_means:
            avg_f_q.append(np.mean(per_prompt_means))
        else:
            avg_f_q.append(np.nan)
    return np.array(avg_f_q)


def _extract_x_range(load_prefixes_to_use, n_timesteps):
    """Create x_range as simple integer timestep indices."""
    return np.arange(n_timesteps)


def _load_v1_as_summary_kl(prefix_list, load_dir, global_logZ):
    """Load v1 format f_q/g_q files and compute summary KL trajectories.

    For v1 files (aggregated across prompts), computes KL directly from
    the aggregate f_q/g_q values using the given global log Z.

    Returns list of (kl_q_sigma_array, kl_sigma_q_array) tuples, one per seed.
    """
    seed_results = []
    for fn in prefix_list:
        path = os.path.join(load_dir, fn)
        try:
            data = torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"Warning: Failed to load v1 file {path}: {e}")
            continue
        extracted = _extract_f_q_g_q_from_loaded(data, path)
        if extracted is None:
            continue
        f_q_list, g_q_list, iwae_lbs_list, iwae_ubs_list = extracted
        T = len(f_q_list)
        kl_q_sigma = np.full(T, np.nan)
        kl_sigma_q = np.full(T, np.nan)
        for t in range(T):
            if f_q_list[t] is not None:
                kl_q_sigma[t] = global_logZ - to_scalar(f_q_list[t])
            if t < len(g_q_list) and g_q_list[t] is not None:
                kl_sigma_q[t] = to_scalar(g_q_list[t]) - global_logZ
        seed_results.append((kl_q_sigma, kl_sigma_q))
    return seed_results


def plot_f_q_g_q_kl_divergences_multiprompt(
    load_prefixes_to_use, labels, figname_modifier,
    target_samples_path,
    x_range=None, fontsize=7, load_dir="./info",
    individual_prompt_plots=True,
    random_f_q_ylim_low=None,
    n_frontiers=1,
    frontier_legendfontsize=None,
):
    """
    Plot per-prompt and summary KL divergence metrics from multiprompt v2 f_q/g_q data.

    Generates:
    1. Per-prompt KL plots (two per prompt) in subfolder figs/<figname_modifier>/
    2. Summary plots: mean KL across prompts per seed, with CI over seeds
    3. Average f_q plot for random prompt set
    """
    # Load v2 files
    all_v2_data, loaded_seed_indices = _load_v2_f_q_g_q_files(load_prefixes_to_use, load_dir)

    has_data = any(len(exp_data) > 0 for exp_data in all_v2_data)
    if not has_data:
        print("Warning: No v2 f_q/g_q data found, skipping multiprompt KL plots")
        return

    # Load target samples to identify prompts with actual target samples
    if target_samples_path is not None:
        prompts_with_targets = _get_prompts_with_targets(target_samples_path)
    else:
        prompts_with_targets = _infer_prompts_with_targets_from_v2(all_v2_data)

    # Create output subfolder for plots
    per_prompt_dir = _make_output_dir(figname_modifier)

    # Generate semantic styling for all experiments
    semantic_colors, semantic_markers, semantic_linestyles = generate_visual_style_from_prefixes(load_prefixes_to_use)

    # Compute global per-prompt log Z from IWAE bounds pooled across ALL experiments/seeds/timesteps
    print("\nComputing global per-prompt log Z from all experiments...")
    global_log_Z = _compute_global_per_prompt_log_Z(all_v2_data, prompts_with_targets)
    print(f"Computed log Z for {len(global_log_Z)} prompts")

    # For each experiment/seed, compute per-prompt KL using the shared global log Z
    # Structure: all_kl_data[exp_i][seed_j] = {prompt_idx: (kl_q_sigma, kl_sigma_q)}
    all_kl_data = []
    all_random_fq = []  # all_random_fq[exp_i][seed_j] = np.array or None
    # Grab prompt_texts from first available v2 data (same across seeds for fixed set)
    prompt_texts = None

    for exp_i, exp_data in enumerate(all_v2_data):
        exp_kl = []
        exp_random_fq = []
        n_total_seeds = len(load_prefixes_to_use[exp_i])
        loaded_set = set(loaded_seed_indices[exp_i])
        loaded_idx = 0
        for seed_j in range(n_total_seeds):
            if seed_j not in loaded_set:
                print(f"Exp {exp_i + 1} seed {seed_j + 1}: not found")
                continue

            v2_data = exp_data[loaded_idx]
            loaded_idx += 1

            if prompt_texts is None:
                prompt_texts = v2_data["prompt_texts_fixed"]

            kl_by_prompt = _compute_per_prompt_kl_over_time(v2_data, global_log_Z)
            random_fq = _compute_random_f_q_over_time(v2_data)

            exp_kl.append(kl_by_prompt)
            exp_random_fq.append(random_fq)

            print(f"Exp {exp_i + 1} seed {seed_j + 1}: {len(kl_by_prompt)} prompts with KL, "
                  f"random f_q timesteps: {len(random_fq) if random_fq is not None else 0}")

        all_kl_data.append(exp_kl)
        all_random_fq.append(exp_random_fq)

    if prompt_texts is None:
        print("Warning: No prompt texts found, skipping multiprompt plots")
        return

    # Determine common prompt indices (present across all seeds of first v2 experiment)
    # Use first experiment with v2 data as reference
    ref_kl = {}
    for exp_kl in all_kl_data:
        if exp_kl:
            ref_kl = exp_kl[0]
            break
    common_prompt_indices = sorted(ref_kl.keys())
    T = len(next(iter(ref_kl.values()))[0]) if ref_kl else 0

    if T == 0:
        print("Warning: No timesteps found, skipping multiprompt plots")
        return

    # Compute global max timesteps across ALL experiments and seeds (for truncation thresholds)
    global_max_T_fixed = 0
    global_max_T_random = 0
    for exp_kl, exp_rfq in zip(all_kl_data, all_random_fq):
        for kl_by_prompt in exp_kl:
            for p, (kl_q_s, kl_s_q) in kl_by_prompt.items():
                global_max_T_fixed = max(global_max_T_fixed, len(kl_q_s))
        for rfq in exp_rfq:
            if rfq is not None:
                global_max_T_random = max(global_max_T_random, len(rfq))
    print(f"Global max timesteps: fixed={global_max_T_fixed}, random={global_max_T_random}")

    if x_range is None:
        x_range = _extract_x_range(load_prefixes_to_use, global_max_T_fixed)

    # ---- 1. Per-prompt plots (subfolder) ----
    if not individual_prompt_plots:
        print("\nSkipping per-prompt KL plots (individual_prompt_plots=False)")
    else:
        print(f"\nGenerating per-prompt KL plots for {len(common_prompt_indices)} prompts...")
    for p in common_prompt_indices if individual_prompt_plots else []:
        prompt_text = prompt_texts[p]
        sanitized = _sanitize_for_filename(prompt_text)

        for kl_idx, (kl_name, kl_ylabel) in enumerate([
            ("kl_q_sigma", r"KL(q|$\sigma_p$)"),
            ("kl_sigma_q", r"KL($\sigma_p$|q)"),
        ]):
            fig, ax = plt.subplots()
            for exp_i in range(len(all_kl_data)):
                # Collect this prompt's KL across seeds for this experiment
                seed_trajectories = []
                for seed_j in range(len(all_kl_data[exp_i])):
                    kl_by_prompt = all_kl_data[exp_i][seed_j]
                    if p in kl_by_prompt:
                        traj = kl_by_prompt[p][kl_idx]
                        if not np.all(np.isnan(traj)):
                            seed_trajectories.append(traj)
                if not seed_trajectories:
                    continue
                np_results = _truncate_and_stack(seed_trajectories, context=f"prompt {p} {kl_name}, series '{labels[exp_i]}'", global_max_len=global_max_T_fixed)
                x_range_adj = x_range[:np_results.shape[1]] if len(x_range) > np_results.shape[1] else x_range
                plot_with_conf_bounds(ax, np_results, x_range_adj, label=labels[exp_i],
                                      color=semantic_colors[exp_i],
                                      linestyle=semantic_linestyles[exp_i])

            ax.set_xlabel("Time Steps", fontsize=fontsize)
            ax.set_ylabel(kl_ylabel, fontsize=fontsize)
            ax.set_title(f"Prompt {p}: {prompt_text[:80]}", fontsize=max(fontsize - 1, 5))
            ax.tick_params(axis='both', labelsize=fontsize)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()
            figname = os.path.join(per_prompt_dir, f"prompt_{p:03d}_{sanitized}_{kl_name}.pdf")
            plt.savefig(figname)
            plt.clf()
            plt.close(fig)

    # ---- 2. Summary plots (mean KL across prompts per seed, CI over seeds) ----
    print("\nGenerating summary KL plots...")

    # Pre-compute summary trajectories for both KL directions:
    # all_summary_kl[kl_idx][exp_i] = list of mean_traj arrays (one per seed)
    kl_names_and_ylabels = [
        ("kl_q_sigma", r"KL(q|$\sigma$) mean over prompts"),
        ("kl_sigma_q", r"KL($\sigma$|q) mean over prompts"),
    ]
    all_summary_kl = [[] for _ in range(len(kl_names_and_ylabels))]

    for kl_idx, (kl_name, _) in enumerate(kl_names_and_ylabels):
        for exp_i in range(len(all_kl_data)):
            seed_means = []
            for seed_j in range(len(all_kl_data[exp_i])):
                kl_by_prompt = all_kl_data[exp_i][seed_j]
                # Collect KL arrays for all prompts, compute mean across prompts per timestep
                per_prompt_arrays = []
                for p in common_prompt_indices:
                    if p in kl_by_prompt:
                        traj = kl_by_prompt[p][kl_idx]
                        per_prompt_arrays.append(traj)
                if per_prompt_arrays:
                    # nanmean to handle prompts with partial g_q data
                    stacked = _truncate_and_stack(per_prompt_arrays, context=f"prompts in seed {seed_j}, {kl_name}")  # (n_prompts, T)
                    mean_traj = np.nanmean(stacked, axis=0)  # (T,)
                    seed_means.append(mean_traj)
            all_summary_kl[kl_idx].append(seed_means)

    # ---- Handle v1-only experiments (e.g., mixeval files) ----
    v1_exp_indices = [i for i, exp_data in enumerate(all_v2_data) if len(exp_data) == 0]
    if v1_exp_indices and global_log_Z:
        # Use mean of v2 per-prompt log Z as reference for v1 experiments
        ref_logZ = float(np.mean(list(global_log_Z.values())))
        print(f"\nProcessing {len(v1_exp_indices)} v1 experiment(s) using ref log Z = {ref_logZ:.4f}")
        for exp_i in v1_exp_indices:
            seed_results = _load_v1_as_summary_kl(
                load_prefixes_to_use[exp_i], load_dir, ref_logZ)
            if seed_results:
                # kl_q_sigma = index 0, kl_sigma_q = index 1
                all_summary_kl[0][exp_i] = [r[0] for r in seed_results]
                all_summary_kl[1][exp_i] = [r[1] for r in seed_results]
                # Update global max with v1 data
                for r in seed_results:
                    global_max_T_fixed = max(global_max_T_fixed, len(r[0]), len(r[1]))
                print(f"  Exp {exp_i + 1} ({labels[exp_i]}): {len(seed_results)} seeds loaded")
        print(f"Global max timesteps (after v1): fixed={global_max_T_fixed}")

    # Plot summary time series (same as before, using stored trajectories)
    for kl_idx, (kl_name, kl_ylabel) in enumerate(kl_names_and_ylabels):
        fig, ax = plt.subplots()
        for exp_i in range(len(all_summary_kl[kl_idx])):
            if not all_summary_kl[kl_idx][exp_i]:
                continue
            np_results = _truncate_and_stack(all_summary_kl[kl_idx][exp_i], context=f"seeds for summary {kl_name}, series '{labels[exp_i]}'", global_max_len=global_max_T_fixed)  # (n_seeds, T)
            x_range_adj = x_range[:np_results.shape[1]] if len(x_range) > np_results.shape[1] else x_range
            plot_with_conf_bounds(ax, np_results, x_range_adj, label=labels[exp_i],
                                  color=semantic_colors[exp_i],
                                  linestyle=semantic_linestyles[exp_i])

        ax.set_xlabel("Time Steps", fontsize=fontsize)
        ax.set_ylabel(kl_ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        figname = os.path.join(per_prompt_dir, f"summary_{kl_name}.pdf")
        plt.savefig(figname)
        plt.clf()
        plt.close(fig)

    # ---- 2b. Frontier plots at training checkpoints ----
    # Convert all_summary_kl[kl_idx][exp_i][seed_j] -> standard kl_results_list format
    # all_summary_kl[0] = kl_q_sigma, all_summary_kl[1] = kl_sigma_q
    mp_kl_results_list = []
    for exp_i in range(len(all_summary_kl[0])):
        seed_tuples = []
        n_seeds = len(all_summary_kl[0][exp_i])
        for seed_j in range(n_seeds):
            kl_q_sigma = all_summary_kl[0][exp_i][seed_j] if seed_j < len(all_summary_kl[0][exp_i]) else None
            kl_sigma_q = all_summary_kl[1][exp_i][seed_j] if seed_j < len(all_summary_kl[1][exp_i]) else None
            if kl_q_sigma is not None and kl_sigma_q is not None:
                seed_tuples.append((kl_sigma_q, kl_q_sigma))
        mp_kl_results_list.append(seed_tuples)

    _generate_kl_frontier_plots(
        mp_kl_results_list, labels, load_prefixes_to_use, per_prompt_dir,
        n_frontiers=n_frontiers, fontsize=fontsize,
        frontier_legendfontsize=frontier_legendfontsize,
    )

    # ---- 2b2. "Average of per-prompt time-averages" frontier ----
    # Different from the time-avg frontier in 2b: that one averages across prompts at each timestep
    # first, then averages over time. This one averages over time per prompt first, then averages
    # across prompts — giving equal weight to each prompt regardless of how many non-NaN timesteps it has.
    print("\nGenerating 'average of per-prompt time-averages' frontier...")
    avg_of_avg_results = []
    for exp_i in range(len(all_kl_data)):
        seed_data = []
        for seed_j in range(len(all_kl_data[exp_i])):
            kl_by_prompt = all_kl_data[exp_i][seed_j]
            per_prompt_time_avg_q_sigma = []
            per_prompt_time_avg_sigma_q = []
            for p in common_prompt_indices:
                if p not in kl_by_prompt:
                    continue
                kl_q_sigma_arr, kl_sigma_q_arr = kl_by_prompt[p]
                avg_q_sigma = np.nanmean(kl_q_sigma_arr)
                avg_sigma_q = np.nanmean(kl_sigma_q_arr)
                if not np.isnan(avg_q_sigma):
                    per_prompt_time_avg_q_sigma.append(avg_q_sigma)
                if not np.isnan(avg_sigma_q):
                    per_prompt_time_avg_sigma_q.append(avg_sigma_q)
            if per_prompt_time_avg_q_sigma and per_prompt_time_avg_sigma_q:
                seed_data.append((
                    np.array([np.mean(per_prompt_time_avg_sigma_q)]),
                    np.array([np.mean(per_prompt_time_avg_q_sigma)]),
                ))
        avg_of_avg_results.append(seed_data)

    if any(len(sd) > 0 for sd in avg_of_avg_results):
        semantic_colors_aoa, semantic_markers_aoa, _ = generate_visual_style_from_prefixes(load_prefixes_to_use)
        make_frontier_exact_kl_bootstrap(
            xlabel=r"KL($\sigma$|q)", ylabel=r"KL(q|$\sigma$)",
            figname=os.path.join(per_prompt_dir, "frontier_kl_avg_of_prompt_avgs.pdf"),
            labels=labels, results_list=avg_of_avg_results,
            color_list=semantic_colors_aoa, marker_list=semantic_markers_aoa,
            aggregate_seeds=True, fontsize=fontsize,
            legendfontsize=frontier_legendfontsize if frontier_legendfontsize is not None else fontsize,
        )
    else:
        print("Warning: No data for 'average of per-prompt time-averages' frontier")

    # ---- 2c. Per-prompt frontier plots ----
    if individual_prompt_plots:
        print(f"\nGenerating per-prompt frontier plots for {len(common_prompt_indices)} prompts...")
        for p in common_prompt_indices:
            per_prompt_kl_results = []
            has_any_data = False
            for exp_i in range(len(all_kl_data)):
                seed_tuples = []
                for seed_j in range(len(all_kl_data[exp_i])):
                    kl_by_prompt = all_kl_data[exp_i][seed_j]
                    if p in kl_by_prompt:
                        kl_q_sigma, kl_sigma_q = kl_by_prompt[p]
                        if not (np.all(np.isnan(kl_q_sigma)) and np.all(np.isnan(kl_sigma_q))):
                            # Note order swap: _generate_kl_frontier_plots expects (kl_sigma_q, kl_q_sigma)
                            seed_tuples.append((kl_sigma_q, kl_q_sigma))
                            has_any_data = True
                per_prompt_kl_results.append(seed_tuples)

            if not has_any_data:
                continue

            sanitized = _sanitize_for_filename(prompt_texts[p])
            prefix = f"prompt_{p:03d}_{sanitized}_"

            _generate_kl_frontier_plots(
                per_prompt_kl_results, labels, load_prefixes_to_use, per_prompt_dir,
                n_frontiers=n_frontiers, fontsize=fontsize,
                frontier_legendfontsize=frontier_legendfontsize,
                filename_prefix=prefix,
                skip_individual_timestep_plots=True,
                skip_combined_frontier=True,
                skip_time_avg_halves=True,
            )
    else:
        print("\nSkipping per-prompt frontier plots (individual_prompt_plots=False)")

    # ---- 3. Random f_q average plot ----
    print("\nGenerating random f_q plot...")
    has_random = any(
        any(rfq is not None for rfq in exp_random_fq)
        for exp_random_fq in all_random_fq
    )
    if has_random:
        fig, ax = plt.subplots()
        for exp_i in range(len(all_random_fq)):
            seed_trajectories = [rfq for rfq in all_random_fq[exp_i] if rfq is not None]
            if not seed_trajectories:
                continue
            # Drop seeds <80% of max length, then truncate to common length
            np_results = _truncate_and_stack(seed_trajectories, context=f"random f_q, series '{labels[exp_i]}'", global_max_len=global_max_T_random)
            x_range_random = _extract_x_range(load_prefixes_to_use, np_results.shape[1])
            plot_with_conf_bounds(ax, np_results, x_range_random, label=labels[exp_i],
                                  color=semantic_colors[exp_i],
                                  linestyle=semantic_linestyles[exp_i])

        ax.set_xlabel("Time Steps", fontsize=fontsize)
        ax.set_ylabel(r"$f_q$ (random prompts, mean)", fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        if random_f_q_ylim_low is not None:
            ax.set_ylim(bottom=random_f_q_ylim_low)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        figname = os.path.join(per_prompt_dir, f"random_f_q.pdf")
        plt.savefig(figname)
        plt.clf()
        plt.close(fig)
    else:
        print("No random f_q data found, skipping random f_q plot")

    # ---- 4. Heatmaps (prompts x time, one per KL direction per experiment) ----
    skip_heatmaps = True  # TODO: set to False to re-enable heatmaps
    if not individual_prompt_plots or skip_heatmaps:
        print("\nSkipping KL heatmaps")
    else:
        print("\nGenerating KL heatmaps...")
        for kl_idx, (kl_name, kl_label) in enumerate([
            ("kl_q_sigma", r"KL(q||$\sigma_p$)"),
            ("kl_sigma_q", r"KL($\sigma_p$||q)"),
        ]):
            for exp_i in range(len(all_kl_data)):
                # Average across seeds for this experiment
                seed_matrices = []
                for seed_j in range(len(all_kl_data[exp_i])):
                    kl_by_prompt = all_kl_data[exp_i][seed_j]
                    rows = []
                    for p in common_prompt_indices:
                        if p in kl_by_prompt:
                            rows.append(kl_by_prompt[p][kl_idx])
                        else:
                            rows.append(np.full(T, np.nan))
                    seed_matrices.append(_truncate_and_stack(rows, context=f"heatmap prompts, seed {seed_j}, {kl_name}"))  # (n_prompts, T)
                if not seed_matrices:
                    continue
                # Mean across seeds: (n_prompts, T)
                heatmap_data = np.nanmean(_truncate_and_stack_2d(seed_matrices, context=f"heatmap seeds, {kl_name}, series '{labels[exp_i]}'", global_max_cols=global_max_T_fixed), axis=0)

                # Sort prompts by starting KL (first timestep, ascending) for visual clarity
                starting_kl_per_prompt = heatmap_data[:, 0]
                sort_order = np.argsort(starting_kl_per_prompt)
                heatmap_sorted = heatmap_data[sort_order]
                sorted_indices = [common_prompt_indices[i] for i in sort_order]

                n_prompts = heatmap_sorted.shape[0]
                fig_height = max(4, n_prompts * 0.08)
                fig, ax = plt.subplots(figsize=(8, fig_height))
                im = ax.imshow(heatmap_sorted, aspect='auto', origin='lower',
                               extent=[x_range[0], x_range[min(T - 1, len(x_range) - 1)],
                                       -0.5, n_prompts - 0.5])
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(kl_label, fontsize=fontsize)
                cbar.ax.tick_params(labelsize=fontsize)

                ax.set_xlabel("Time Steps", fontsize=fontsize)
                ax.set_ylabel("Prompt (sorted by mean KL)", fontsize=fontsize)
                ax.tick_params(axis='both', labelsize=fontsize)

                # Label every Nth prompt on y-axis to avoid clutter
                label_every = max(1, n_prompts // 20)
                ytick_positions = list(range(0, n_prompts, label_every))
                ytick_labels = [f"{sorted_indices[i]}" for i in ytick_positions]
                ax.set_yticks(ytick_positions)
                ax.set_yticklabels(ytick_labels, fontsize=max(fontsize - 2, 4))

                if len(all_kl_data) > 1:
                    ax.set_title(f"{labels[exp_i]}", fontsize=fontsize)

                plt.tight_layout()
                suffix = f"_exp{exp_i}" if len(all_kl_data) > 1 else ""
                figname = os.path.join(per_prompt_dir, f"heatmap_{kl_name}{suffix}.pdf")
                plt.savefig(figname)
                plt.clf()
                plt.close(fig)

    print(f"\nDone. All plots saved to {per_prompt_dir}/")

    return all_v2_data


# Default: no target samples path (set in specific block to enable multiprompt plotting)
target_samples_path = None

# Comment out/select as needed
figname_modifier = "toyrlhf_kl10_10_18_final"
figname_modifier = "toyrlhf_10_18_final"
# figname_modifier = "toyrepulse_01_18"
figname_modifier = "toyrepulse_01_19_v2"
figname_modifier = "toyrepulse_01_20"
figname_modifier = "toyrepulse_01_20_v2"



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

            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count10.0_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count1.0_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_s2",
            #     1, 10),


            # # for x in $(ls info/toyrepulse2/ | grep analytic_kl | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs1024_s2",
            # #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs16_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs64_s2",
            #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs1024_s2",
            # #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs16_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs64_s2",
            #     1, 10),
            # # # make_list(
            # # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs1024_s2",
            # # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs16_s2",
            # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs256_s2",
            # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count3.0_tbs64_s2",
            # #     1, 10),
            # # # make_list(
            # # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs1024_s2",
            # # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs16_s2",
            # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi4_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs64_s2",
            # #     1, 10),

            # for x in $(ls info/toyrepulse3/ | grep analytic_kl | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs1024_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs16_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs64_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count10.0_tbs1024_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count10.0_tbs16_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count10.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count10.0_tbs64_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count30.0_tbs1024_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count30.0_tbs16_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count30.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count30.0_tbs64_s2",
            #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_tbs16_s2",
            # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            # #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_tbs64_s2",
            # #     1, 10),

            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs1024_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs16_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs64_s2",
            #     1, 10),

            # for x in $(ls info/toyrepulse3/ | grep analytic_kl | grep bs256 | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),

            # for x in $(ls info/toyrepulse4/ | grep analytic_kl | grep bs256 | grep -v analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_tbs256 | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.003_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_count100.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr0.0001_policy_psi_q_p_s_t_count100.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr0.0001_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.03_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-20.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_count100.0_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-20.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr3e-05_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
            #     1, 10),

            # for x in $(ls info/toyrepulse4coin/ | grep analytic_kl | grep bs256 | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd16_cflr0.001_cfsepnn_after_tbs256_s2",
            #     1, 10),
            # # make_list(
            # #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfhis0.01_cfsepnn_after_tbs256_s2",
            # #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfsepnn_after_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfsepnn_after_pri_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfsepnn_after_tbs256_s2",
            #     1, 10),
            # Below is decent
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfsepnn_before_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfsepnn_before_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfus64_cfsepnn_after_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_fpis0.01_cfsepnn_after_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.001_cfsepnn_after_tbs256_s2",
            #     1, 10),
            # Below is decent
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.001_cfsepnn_after_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.001_cfsepnn_before_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cfsepnn_after_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cfsepnn_before_firstonline_tbs256_s2",
            #     1, 10),

            # for x in $(ls info/toyrepulse4coin/ | grep analytic_kl | grep bs256 | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cfllp_after_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cfllp_before_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cfllq_after_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cfllq_before_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cflsib_after_firstonline_tbs256_s2",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cflsib_before_firstonline_tbs256_s2",
            #     1, 10),



            # 01-22
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_reinforce_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0_blr3e-05_policy_psi_q_p_s_t_tbs256_s2",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_count100.0_tbs256_s2",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_tbs256_s2",
                1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0001_cfus64_cfsepnn_after_tbs256_s2",
            #     1, 10),

            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu64_cfs_bf_fo_tb256_s3",
            #     1, 10),

            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cfsepnn_before_firstonline_tbs256_s2",
                1, 10),
            #
            # # for x in $(ls info/toyrepulse4cointest/ | grep analytic_kl | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
            # # make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu64_cfs_af_fo_tb256_s2", 1,10),

            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu4_cfs_bf_fo_tb256_s3",
                1, 10) + make_list("analytic_kls_toxicity_rlhf_di_To_2_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu4_cfs_bf_fo_tb256_s2",2,2,),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu4_cfsn_bf_fo_tb256_s3",
                1, 10),


            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu16_cfs_bf_fo_tb256_s3",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu16_cfsn_bf_fo_tb256_s3",
            #     1, 10),

            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfu64_cfs_bf_fo_tb256_s3",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cflsib_after_firstonline_tbs256_s3",
            #     1, 10),
            # make_list(
            #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_epo1_epi1_hepi5_schconstant_alr0.0001_blr0.0001_policy_psi_q_p_s_t_cfn100.0_cfd64_cflr0.0003_cflsib_before_firstonline_tbs256_s3",
            #     1, 10),

        ]
        figname_modifier = "toyrepulse_01_21_b256_coinflip_clean"
        figname_modifier = "toyrepulse_01_21_b256_coinflip_v2"
        figname_modifier = "toyrepulse_01_21_b256_coinflip_compare64"
        figname_modifier = "toyrepulse_01_21_b256_coinflip_comparerest"

        load_prefixes_to_use = [
            # for x in $(ls info/toyrepulseIsmallbatch/ | grep analytic_kl | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_I_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfsn_bf_fo_tb256_s3",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_I_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfsn_bf_fo_tb4_s3",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_I_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_cf100.0_cd64_cfr0.0003_cfsn_bf_fo_tb64_s3",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_I_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_tb256_s3",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_I_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_tb4_s3",
                1, 10),
            make_list(
                "analytic_kls_toxicity_rlhf_di_To_I_l1_kl0.0_b-10.0_hlnt_a0.01_ppq_ctl_ep1_e1_he5_scc_al0.0001_bl0.0001_ppq_tb64_s3",
                1, 10),

        ]
        figname_modifier = "toyrepulse_01_22_b4"


        # Use the same naming code from make_frontier.py
        labels = generate_labels_from_prefixes(load_prefixes_to_use)
        fontsize = 6

    else:

        labels = ['_'.join(a[0].split('len2_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0] for
                  a in load_prefixes_to_use]
        fontsize = 5



# x_range is now computed dynamically from data, but can be overridden if needed
# x_range = np.arange(51) * 10 * 500  # Uncomment to use custom x_range



load_prefixes_to_use = [
    # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0003_cfsn_bf_fo_tb5_s2", 1,5),
    make_list(
        "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2",
        1, 10),
    make_list(
        "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s2",
        1, 10),

    # make_list(
    #     "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2",
    #     1, 5),
    # make_list(
    #     "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2",
    #     1, 5),
    # make_list(
    #     "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2",
    #     1, 5),
    # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0003_cfu4_cfsn_af_fo_tb5_s1",1,5),
    # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0003_cfu4_cfsn_bf_fo_tb5_s1",1,5),
]
figname_modifier = "toy_len2_01_28_kl_div_approx_1000_clean"

# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/toyrepulse2p2len4/ | grep f_q | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
#
# load_prefixes_to_use = [
#     make_list(
#         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3",
#         1, 10),
#     make_list(
#         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s3",
#         1, 10),    ]
# figname_modifier = "toy_len4_01_28_kl_div_approx_1000"
#
#
# # load_prefixes_to_use = [
# #     make_list(
# #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l1_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_c3.0_tb5_s2",
# #         1, 5),
# #     make_list(
# #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l1_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2",
# #         1, 5),
# #     make_list(
# #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l1_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s2",
# #         1, 5),
# #
# # ]
# # figname_modifier = "toy_len1_01_28_kl_div_approx"
#
# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/thismanl1/ | grep analyt | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
# load_prefixes_to_use = [
#     # make_list(
#     #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_c3.0_tb5_s2",
#     #     1, 10),
#     make_list(
#         "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_tb5_s2",
#         1, 10),
#     # make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_c100.0_tb5_s3", 1,10),
#     make_list(
#         "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_c10.0_tb5_s3",
#         1, 10),
#     # make_list(
#     #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_c7.0_tb5_s3",
#     #     1, 10),
#         make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_c30.0_tb5_s3", 1,10),
# make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
# make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.0003_cfu64_cfsn_af_fo_tb5_s3", 1,10),
# make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_l1_kl0.0_b10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.0003_cfu64_cfsn_bf_fo_tb5_s3", 1,10),
# ]
# figname_modifier = "toy_len1_01_28_tm_beta10_kl_div_exact"
#
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/toyrepulse2p2len4/ | grep f_q | grep al3e-05 | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
load_prefixes_to_use = [
#     # make_list(
#     #     "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al0.0001_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3",
#     #     1, 10),
#     # make_list(
#     #     "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al0.0001_bl0.0_ppq_tb5_s3",
#     #     1, 10),
#     make_list(
#         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3",
#         1, 10),
#     make_list(
#         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s3",
#         1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),

# for x in $(ls /scratch/zhaostep/OpenRLHF/info/toyrepulse2p2len4/ | grep f_q | grep he40 | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he40_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he40_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he40_scc_al1e-05_bl0.0_ppq_tb5_s3", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he40_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he40_scc_al3e-05_bl0.0_ppq_cf5.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he40_scc_al3e-05_bl0.0_ppq_tb5_s3", 1,10),
]
figname_modifier = "toy_len4_01_28_2p2_kl_div_approx_v2"


load_prefixes_to_use = [

# for x in $(ls /scratch/zhaostep/OpenRLHF/info/toy2p2len4/ | grep f_q | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,9),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s3", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,6) + make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 8,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s3", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_s-1.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s3", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s3", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l4_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s3", 1,10),
]
figname_modifier = "toy_len4_01_31_2p2_b10_kl_div_approx_w1e-5"

#
load_prefixes_to_use = [
# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/murder50/ | grep f_q | grep _s1 | grep al3e-05); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s1", 1,10),
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/murder50/ | grep f_q | grep _s2 | grep al1e-05); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.0001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.0001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr1e-05_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),

# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/murder50/ | grep f_q | grep _s1 | grep -E 'al1e-06|al3e-06'); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-06_bl0.0_ppq_tb5_s1", 1,10),

# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/murder50/ | grep f_q | grep _s1 | grep al3e-06); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb5_s1", 1,10),
]
figname_modifier = "len50_m_02_01_b-10_kl_div_approx_v2"


load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/murderp50/ | grep f_q | grep _s1); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb5_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_Sm13In_remodev3lav2_H_l50_kl0.0_b5.0_hlnt_a0.0_ppq_ctl_ep1_e1_he5_scc_al3e-05_bl0.0_ppq_tb5_s1", 1,10),

]
figname_modifier = "len50_m_02_01_b5_kl_div_approx"

load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/cryp/ | grep f_q | grep _s1); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_t_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb500_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_t_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb500_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_t_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb500_s1", 1,10),

]
figname_modifier = "len20_cryp_02_02_b-20_kl_div_approx"

load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/friendcar/ | grep he20 | grep f_q | grep _s1); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_H_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb500_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_H_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb500_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_H_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb500_s1", 1,10),

]
figname_modifier = "len20_friendcar_02_02_b-20_kl_div_approx"
load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/plunger/ | grep he20 | grep f_q | grep _s1); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_W_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb500_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_W_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb500_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_W_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb500_s1", 1,10),
]
figname_modifier = "len20_plunger_02_02_b-20_kl_div_approx"






load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/drinkwater/ | grep f_q | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_D_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb5_s2", 1,10),

]
figname_modifier = "len20_drinkwater_02_04_b-20_1e-5_kl_div_approx"


# load_prefixes_to_use = [
# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/dis/  | grep f_q | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),
#
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-06_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-06_bl0.0_ppq_tb5_s2", 1,10),
#
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf0.1_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb5_s2", 1,10),
#
# ]
# figname_modifier = "len20_dis_02_04_b-20_combined_kl_div_approx"




# load_prefixes_to_use = [
# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/smtest/ | grep f_q | grep _s2); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc7.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-06_bl0.0_ppq_tb5_s2", 1,10),
#
# ]
# figname_modifier = "len20_homeless_02_04_b-10_kl_div_approx"


load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/repulsedis/ | grep heldout | grep he20 | grep _s1 | grep bl1e-05); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("heldout_over_time_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.2_b-20.0_hlnt_a0.1_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl1e-05_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("heldout_over_time_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.2_b-20.0_hlnt_a0.1_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl1e-05_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
make_list("heldout_over_time_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.2_b-20.0_hlnt_a0.1_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl1e-05_ppq_tb5_s1", 1,10),
]
threshold = -5
figname_modifier = "len20_repulse_dis_02_05_b-20"

load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/repulsedis2/ | grep heldout | grep he20 | grep _s1 | grep bl1e-05); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("heldout_over_time_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.2_b-20.0_hlnt_a0.1_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl1e-05_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
# make_list("heldout_over_time_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.2_b-20.0_hlnt_a0.1_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl1e-05_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s1", 1,10),
make_list("heldout_over_time_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.2_b-20.0_hlnt_a0.1_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl1e-05_ppq_tb5_s1", 1,10),
# make_list("heldout_over_time_rlhf_rc7.0_Sm13In_remodev3lav2_T_l20_kl0.2_b-20.0_hlnt_a0.1_ppq_ctl_ep1_e1_he2_fs2_scc_al1e-05_bl1e-05_ppq_tb5_s1", 1,10),
]
threshold = -5
figname_modifier = "len20_repulse_dis_02_05_b-20_nobonusintargetforp"


load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitdis/ | grep he20 | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs50_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs50_scc_al3e-06_bl0.0_ppq_tb5_s2", 1,10),
]
threshold = -5
# figname_modifier = "len20_noit_dis_02_05_b-20_v2"
figname_modifier = "len20_noit_dis_02_05_b-20_v3"

load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/multitesttoy/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he5_fs20_scc_al1e-05_bl0.0_ppq_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he5_fs20_scc_al3e-05_bl0.0_ppq_tb200_s2", 1,10),

# for x in $(ls info/multitesttoy/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_tb200_s2", 1,15),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,15),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf15.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf7.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf5.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),

# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al0.0001_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al0.0001_bl0.0_ppq_cf20.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al0.0001_bl0.0_ppq_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al1e-05_bl0.0_ppq_cf20.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al1e-05_bl0.0_ppq_tb200_s2", 1,10),

]
threshold = -5
# figname_modifier = "len20_dis_02_08_b-20_multiprompt"
figname_modifier = "len20_toymultiprompt_02_10_b-20_lr3e-05_v6"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_miprAL_tsa18.pt"


load_prefixes_to_use = [
# for x in $(ls info/ifat/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_tb5_s2", 1,10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
]

figname_modifier = "len20_ifat_02_10_b-20_lr3e-05"
target_samples_path = None
# target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_I_tsa20.pt"

# load_prefixes_to_use = [
# # for x in $(ls info/ustupid/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_tb5_s2", 1,10),
#
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
#
# ]
# figname_modifier = "len20_ustupid_02_10_b-20_lr3e-05"
# target_samples_path = None

load_prefixes_to_use = [
# for x in $(ls info/multitest/ | grep he4 | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,1),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s1", 1,1),

# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_wu100_tb250_s1", 1,10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_wu800_tb250_s1", 1,10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_wu1200_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_wu1600_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_wu200_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_wu40_tb250_s1", 1,10),

]
threshold = -5
figname_modifier = "len20_multitest_02_11_b-20_v2"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_20misi1_tsa50.pt"

load_prefixes_to_use = [
# for x in $(ls info/multitest/ | grep he4 | grep -v cd64 | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf100.0_cd256_cfr0.0001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf100.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf100.0_cd256_cfr0.001_cfsn_af_fo_wu800_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf100.0_cd256_cfr0.01_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf100.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf100.0_cd256_cfr0.001_cfsn_af_fo_wu800_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf100.0_cd256_cfr0.01_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf100.0_cd256_cfr1e-05_cfsn_af_fo_tb250_s1", 1,10),

# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s1", 1,10),

# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd1024_cfr0.0001_cfsn_af_fo_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd1024_cfr0.001_cfsn_af_fo_wu800_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd1024_cfr0.01_cfsn_af_fo_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd256_cfr0.0001_cfsn_af_fo_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd256_cfr0.001_cfsn_af_fo_wu800_tb250_s1", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd256_cfr0.01_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd1024_cfr0.0001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd1024_cfr0.001_cfsn_af_fo_wu800_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd1024_cfr0.01_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd256_cfr0.0001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd256_cfr0.001_cfsn_af_fo_wu800_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd256_cfr0.01_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s1", 1,10),


# # for x in $(ls info/multitest/ | grep he4 | grep -v cf100 | grep -v cf30 | grep -v cf10 | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf0.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf1.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf3.0_cd1024_cfr0.0001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf3.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf3.0_cd1024_cfr0.01_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf3.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf1.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf3.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s1", 1,10),

# for x in $(ls info/multitest/ | grep he4 | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf10.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf10.0_cd256_cfr0.0001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf10.0_cd256_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf30.0_cd256_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf3.0_cd1024_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf3.0_cd256_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_tb250_s2", 1,10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf100.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf10.0_cd256_cfr0.0001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf10.0_cd256_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd256_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s2", 1,10),


]
threshold = -5
figname_modifier = "len20_multitest_02_13_b-20_3e-05"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_20misi1_tsa50.pt"
individual_prompt_plots = False


#
# # load_prefixes_to_use = [
# # # for x in $(ls info/multitesttoy/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al0.0001_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al0.0001_bl0.0_ppq_cf20.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al0.0001_bl0.0_ppq_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al1e-05_bl0.0_ppq_cf20.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al1e-05_bl0.0_ppq_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al1e-06_bl0.0_ppq_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf15.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf5.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_cf7.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs50_scc_al3e-05_bl0.0_ppq_tb200_s2", 1,10),
# #
# # ]
# # threshold = -5
# # # figname_modifier = "len20_dis_02_08_b-20_multiprompt"
# # figname_modifier = "len20_toymultiprompt_02_10_b-20_all"
# # target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_miprAL_tsa18.pt"
#
# load_prefixes_to_use = [
# # for x in $(ls info/multitesttoy2/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf7.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1,10),
#
# ]
# threshold = -5
# figname_modifier = "len20_toymultiprompt_02_12_b-20"
# target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_miprAL_tsa18.pt"
# individual_prompt_plots = False


load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitdis2/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done

# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd256_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd512_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_bf_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.01_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd20_cfr0.001_cfsn_bf_fo_tb5_s2", 1,10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),

]
threshold = -5
figname_modifier = "len20_noit_dis_02_13_b-20_1e-05_v5_d256"
target_samples_path = None
individual_prompt_plots = False


load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitifat/ | grep al1e-05 | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_tb5_s2", 1,10),
]
threshold = -5
figname_modifier = "len20_noit_ifat_02_15_b-20_1e-05_v2"
target_samples_path = None
individual_prompt_plots = False



# load_prefixes_to_use = [
# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitustupid/ | grep f_q | grep al1e-05 | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf30.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_tb5_s2", 1,10),
#
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_Y_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_tb5_s2", 1,10),
#
#
# ]
# threshold = -5
# figname_modifier = "len20_noit_ustupid_02_15_b-20_1e-05"
# target_samples_path = None
# individual_prompt_plots = False


# load_prefixes_to_use = [
# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitmultitesttoy/ | grep f_q | grep _s2 | grep rc10 ); do echo make_list\(\"$x\", 1,10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s3", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1,10),
#
# ]
# threshold = -5
# figname_modifier = "len20_noit_multitoy_02_17_b-20_v2"
# target_samples_path = "info/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_miprAL_tsa36.pt"
# individual_prompt_plots = False


load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitmultitest/ | grep -v 0001_ | grep -v 3e-06 | grep -v cf30 | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf10.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf10.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf3.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf15.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf6.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
#
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf10.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf3.0_cd256_cfr0.001_cfsn_af_fo_tb250_s1", 1,10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s1", 1,10),


]
threshold = -5
figname_modifier = "len20_noit_multi_02_17_b-20_1e-5_v4"
target_samples_path = "info/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_20misi1_tsa100.pt"
individual_prompt_plots = False
random_f_q_ylim_low = 150


load_prefixes_to_use = [
# for x in $(ls info/noitmultitesttoy/ | grep f_q | grep _s2 | grep rc10 ); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-06_bl0.0_ppq_tb200_s2", 1,10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),

]
threshold = -5
figname_modifier = "len20_noit_multitoy_02_18_b-20_v6"
target_samples_path = "info/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_miprAL_tsa34.pt"
individual_prompt_plots = False
random_f_q_ylim_low = 150
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitmultitest/ |  grep f_q | grep -v 1e-06 | grep _s2 | grep -v cf2.0_ | grep -v cf1.0_ ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s2", 1,10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),


]
threshold = -5
figname_modifier = "len20_noit_multi_02_18_b-20_v8"
target_samples_path = "info/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_20misi1_tsa100.pt"
individual_prompt_plots = False
random_f_q_ylim_low = 150
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/noitremodevifat/ | grep -v 3e-05 |  grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13_remodev3lav2_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_tb5_s2", 1, 10),

]
threshold = -5
figname_modifier = "len20_noitremodevifat_02_19_b-20"
target_samples_path = None
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4


# load_prefixes_to_use = [
# # for x in $(ls info/ittoxifat/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al1e-05_bl0.0_ppq_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-05_bl0.0_ppq_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_I_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_fs100_scc_al3e-06_bl0.0_ppq_tb5_s2", 1, 10),
#
# ]
# threshold = -5
# figname_modifier = "len20_ittoxifat_02_19_b-20"
# target_samples_path = None
# individual_prompt_plots = False
# random_f_q_ylim_low = None
# n_frontiers = 4

load_prefixes_to_use = [
# for x in $(ls info/ittoxmultitesttoy/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-06_bl0.0_ppq_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),


make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_ittoxmultitesttoy_02_19_b-20_v4"
target_samples_path = "info/target_samples_Sm13In_To_rlhf_l20_b-20.0_rc10.0_miprAL_tsa11.pt"
individual_prompt_plots = True
random_f_q_ylim_low = 150
n_frontiers = 4


load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoy/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoy_02_21_b-20"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_miprAL_tsa12.pt"
individual_prompt_plots = True
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoy2/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-33.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoy_b-33_02_22_v2"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-33.0_rc6.0_miprAL_tsa10.pt"
individual_prompt_plots = True
random_f_q_ylim_low = None
n_frontiers = 4


#
# load_prefixes_to_use = [
# # for x in $(ls info/ittoxmultitesttoyb10/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1, 10),
#
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-10.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
#
#
# ]
# threshold = -5
# figname_modifier = "len20_ittoxmultitesttoy_b-10_02_23_v2"
# target_samples_path = "info/target_samples_Sm13In_To_rlhf_l20_b-10.0_rc10.0_miprAL_tsa14.pt"
# individual_prompt_plots = True
# random_f_q_ylim_low = None
# n_frontiers = 4




load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyb40rc5/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-40.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-40.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-40.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-40.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-40.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-40.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-40.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoyb40rc5_02_24"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-40.0_rc5.0_miprAL_tsa40.pt"
individual_prompt_plots = True
random_f_q_ylim_low = None
n_frontiers = 4


load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyb20rc5/ | grep f_q | grep b-20 | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoyb20rc5_02_24"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc5.0_miprAL_tsa40.pt"
individual_prompt_plots = True
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyb50rc4/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoyb50rc4_02_24"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-50.0_rc4.0_miprAL_tsa40.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyb20rc4/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc4.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoyb20rc4_02_24"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc4.0_miprAL_tsa40.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyb20rc3/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoyb20rc3_02_24"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc3.0_miprAL_tsa40.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyb66rc3/ | grep f_q | grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-66.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-66.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-66.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-66.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-66.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-66.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc3.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-66.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s1", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoyb66rc3_02_24"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-66.0_rc3.0_miprAL_tsa40.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyb50/ | grep f_q  | grep -v 3e-6 | grep -v cf0.5 | grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-50.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoy_b-50_02_23_v4"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-50.0_rc6.0_miprAL_tsa9.pt"
individual_prompt_plots = True
random_f_q_ylim_low = None
n_frontiers = 4



# load_prefixes_to_use = [
# # for x in $(ls info/itremodevmultitesttoyb100rc6/ | grep f_q  | grep -v 3e-6 | grep -v cf0.5 | grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-100.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-100.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-100.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-100.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-100.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1, 10),
#
# ]
# threshold = -5
# figname_modifier = "len20_itremodevmultitesttoyb100rc6_02_25"
# target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-100.0_rc6.0_miprAL_tsa9.pt"
# individual_prompt_plots = True
# random_f_q_ylim_low = None
# n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/noitmultitesttoynochat/ | grep f_q  |  grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),

# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1, 10),

# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
#
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),



]
threshold = -5
figname_modifier = "len20_noit_multitoy_nochat_b-20_02_27_v7"
target_samples_path = "info/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_miprAL_tsa34.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4


#
# load_prefixes_to_use = [
# # for x in $(ls info/ittoxmultitesttoy2/ | grep f_q  |  grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# # make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqh_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixqh_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_tb200_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0003_bl0.0_ppq_tb200_s2", 1, 10),
#
# ]
# threshold = -5
# figname_modifier = "len20_ittoxmultitesttoy2_b-20_02_26_v8"
# target_samples_path = "info/target_samples_Sm13In_To_rlhf_l20_b-20.0_rc10.0_miprAL_tsa16.pt"
# individual_prompt_plots = False
# random_f_q_ylim_low = None
# n_frontiers = 4
#
#
# load_prefixes_to_use = [
# # for x in $(ls /scratch/zhaostep/OpenRLHF/info/ittoxmultitest/ | grep f_q  |  grep _s3 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# # make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_mixqi_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_mixqi_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_mixmx_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_mixqi_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_mixqi_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_mixmx_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s2", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_mixmx_tb250_s1",1,10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
#
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
#
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
#
# # make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s3", 1, 10),
#
#
# ]
# threshold = -5
# figname_modifier = "len20_ittoxmultitest_b-20_02_26_v10"
# target_samples_path = "info/target_samples_Sm13In_To_rlhf_l20_b-20.0_rc10.0_20misi1_tsa20.pt"
# individual_prompt_plots = False
# random_f_q_ylim_low = None
# n_frontiers = 4

load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitesttoyfixed/ | grep f_q  |  grep _s6 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s6", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixqi_tb200_s6", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s6", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s6", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s6", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb200_s6", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixmx_tb200_s6", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s6", 1, 10),

]
threshold = -5
figname_modifier = "len20_itremodevmultitesttoyfixed_b-20_02_28_v3"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_miprAL_tsa12.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4




load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/ittoxmultitesttoypos/ | grep -v al0.0001 | grep -v 3e-06 | grep -v cf0.5 | grep f_q  |  grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_mixeval_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al0.0001_bl0.0_ppq_tb200_s2", 1, 10),


make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixmx_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1, 10),


make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_miprAL_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_mixqi_tb200_s2", 1, 10),


# ls | grep sba | grep -v 3e-6 | grep -v 1e-4 | grep 02-28-22 | grep -v _s   | grep -v a0.5 | grep -v mxqi
]
threshold = -5
figname_modifier = "len20_ittoxmultitesttoypos_b20_03_01_v7_clean"
target_samples_path = "info/target_samples_Sm13In_To_rlhf_l20_b20.0_rc10.0_miprAL_tsa52.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/ittoxmultitestpos/ |  grep f_q  | grep -v 3e-05  |  grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_mixqi_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_mixmx_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13In_To_20misi1_l20_kl0.0_b20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s2", 1, 10),
]
threshold = -5
figname_modifier = "len20_ittoxmultitestpos_b20_03_02_v5"
target_samples_path = "info/target_samples_Sm13In_To_rlhf_l20_b20.0_rc10.0_20misi1_tsa50.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/itremodevmultitestfixed/ |  grep f_q  |  grep _s1 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_mixqi_tb250_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_mixqi_tb250_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_mixmx_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_mixmx_tb250_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s1", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_mixmx_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s1", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s5", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s3", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s4", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_mixqi_tb250_s2", 1, 10),


]
threshold = -5
figname_modifier = "len20_itremodevmultitestfixed_b-20_03_02_v11"
target_samples_path = "info/target_samples_Sm13In_remodev3lav2_rlhf_l20_b-20.0_rc6.0_20misi1_tsa20.pt"
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4




load_prefixes_to_use = [
# for x in $(ls info/probinffixedtrajdistoy/ |  grep -v rew | grep 1e-05  |  grep _s9 ); do echo make_list\(\"$x\", 1, 20\)\,; done
# make_list("analytic_kls_toxicity_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb5_s9", 1, 20),
# make_list("analytic_kls_toxicity_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb5_s9", 1, 20),
#
# # make_list("analytic_kls_toxicity_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_first_tb5_s5", 1, 20),
# # make_list("analytic_kls_toxicity_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_lag10_tb5_s5", 1, 20),
# make_list("analytic_kls_toxicity_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_lag30_tb5_s5", 1, 20),
# # make_list("analytic_kls_toxicity_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_lag50_tb5_s5", 1, 20),


make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb5_s9", 1, 20),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_first_tb5_s9", 1, 20),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_lag10_tb5_s9", 1, 20),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_mixqi_lag30_tb5_s9", 1, 20),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_di_To_thmaisa_l1_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb5_s9", 1, 20),

# One more: try the lag50 also.
# Plot this and then the f_q values as well. Check those.
# Then check the scaled up setting: just 3e-5, do some more alpha values (see if higher was done or not yet), and try the mixture there too.

]
threshold = -5
figname_modifier = "len1_probinffixedtrajdistoy_b-20_03_06_v8"
target_samples_path = None
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4




load_prefixes_to_use = [
# for x in $(ls info/probinffixedtrajdis/ |  grep f_q  |  grep _s9 ); do echo make_list\(\"$x\", 1, 20\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s9", 1, 20),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s9", 1, 20),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb5_s9", 1, 20),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_mixqi_lag10_tb5_s9", 1, 20),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_mixqi_lag30_tb5_s9", 1, 20),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_tb5_s9", 1, 20),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc6.0_Sm13In_remodev3lav2_T_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_tb5_s9", 1, 20),


]
threshold = -5
figname_modifier = "len20_probinffixedtrajdis_b-20_03_05_v9"
target_samples_path = None
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 4



load_prefixes_to_use = [
# for x in $(ls info/probinffixedtrajdislen50/ | grep -v 0.0001 | grep f_q  |  grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_mixqi_lag10_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_mixqi_lag30_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_mixqi_lag10_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_mixqi_lag30_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_mixmx_lag10_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_mixmx_lag30_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al1e-05_bl0.0_ppq_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_mixmx_lag10_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_mixmx_lag30_tb5_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_T_l50_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he10_fs60_scc_al3e-05_bl0.0_ppq_tb5_s2", 1, 10),


]
threshold = -5
figname_modifier = "len50_probinffixedtrajdislen50_b-20_03_07_v2"
target_samples_path = None
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 60



load_prefixes_to_use = [
# for x in $(ls info/probinffixedtrajitremulti2/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_mixqi_lag30_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf4.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixqi_lag10_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixqi_lag30_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf4.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_first_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_lag10_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_lag30_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_lag50_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_lag80_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixmx_first_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixmx_lag30_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc5.0_Sm13In_remodev3lav2_20misi1_l20_kl0.0_b-30.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),

]
threshold = -5
figname_modifier = "len20_probinffixedtrajitremulti_b-30_ctltraj_03_11_v4"
target_samples_path = None
individual_prompt_plots = True
random_f_q_ylim_low = None
n_frontiers = 2




load_prefixes_to_use = [
# for x in $(ls info/probinffixedtrajittox2/ | grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1, 10\)\,; done

make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_mixqi_first_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_mixqi_lag30_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_mixqi_lag50_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixqi_first_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixqi_lag50_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf2.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_first_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_lag10_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_lag30_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixqi_lag50_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al0.0001_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
# make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixmx_first_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_mixmx_lag30_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al1e-05_bl0.0_ppq_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_cf1.0_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixmx_first_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_mixmx_lag30_tb250_s2", 1, 10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc8.0_Sm13In_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctln_ep1_e1_he2_scc_al3e-05_bl0.0_ppq_tb250_s2", 1, 10),


]
threshold = -5
figname_modifier = "len20_probinffixedtrajittox2_b-20_03_12_v6"
target_samples_path = None
individual_prompt_plots = False
random_f_q_ylim_low = None
n_frontiers = 2
frontier_legendfontsize = 5

# TODO remodev and toypos. Then, what next? remodev with it with pos (need to use cap for this)? noit with pos? Noit with mixture (did I do this already?)
# Basically just collect all the results and make a story. If you have a consistent story, good. If not, investigate why this is not the case - why some better in some settings.



random_f_q_ylim_low = random_f_q_ylim_low if 'random_f_q_ylim_low' in vars() else None
n_frontiers = n_frontiers if 'n_frontiers' in vars() else 1
frontier_legendfontsize = frontier_legendfontsize if 'frontier_legendfontsize' in vars() else None
labels = generate_labels_from_prefixes(load_prefixes_to_use)


# # --- Debug: check number of timesteps per seed ---
# print("\n=== Checking number of timesteps per experiment/seed ===")
# for exp_i, prefix_list in enumerate(load_prefixes_to_use):
#     print(f"\nExp {exp_i} ({labels[exp_i]}):")
#     for seed_j, fn in enumerate(prefix_list):
#         path = os.path.join("./info", fn)
#         try:
#             data = torch.load(path, map_location='cpu')
#         except Exception as e:
#             print(f"  seed {seed_j}: FAILED to load ({e})")
#             continue
#         if isinstance(data, dict) and data.get("version", 1) >= 2:
#             n_f_q_fixed = len(data.get("f_q_by_prompt_fixed", []))
#             n_g_q_fixed = len(data.get("g_q_by_prompt_fixed", []))
#             n_f_q_random = len(data.get("f_q_by_prompt_random", []))
#             n_iwae_lbs = len(data.get("iwae_lbs_by_prompt_fixed", []))
#             # Sample sizes: check first timestep's per-prompt tensors
#             f_q_fixed = data.get("f_q_by_prompt_fixed", [])
#             sample_sizes_str = ""
#             if f_q_fixed and len(f_q_fixed) > 0:
#                 first_ts = f_q_fixed[0]  # list of per-prompt values
#                 sizes = [len(x) if hasattr(x, '__len__') else 1 for x in first_ts if x is not None]
#                 sample_sizes_str = f", samples/prompt(t=0)={sizes}"
#             print(f"  seed {seed_j}: v2, f_q_fixed={n_f_q_fixed}, g_q_fixed={n_g_q_fixed}, f_q_random={n_f_q_random}, iwae_lbs={n_iwae_lbs}{sample_sizes_str}")
#         elif isinstance(data, (tuple, list)) and len(data) >= 4:
#             # Sample sizes: check first timestep tensor
#             f_q_sample = data[0][0] if len(data[0]) > 0 else None
#             g_q_sample = data[1][0] if len(data[1]) > 0 else None
#             f_q_n = len(f_q_sample) if f_q_sample is not None and hasattr(f_q_sample, '__len__') else (1 if f_q_sample is not None else 0)
#             g_q_n = len(g_q_sample) if g_q_sample is not None and hasattr(g_q_sample, '__len__') else (1 if g_q_sample is not None else 0)
#             print(f"  seed {seed_j}: v1, f_q={len(data[0])} (n={f_q_n}), g_q={len(data[1])} (n={g_q_n}), iwae_lbs={len(data[2])}, iwae_ubs={len(data[3])}")
#         else:
#             print(f"  seed {seed_j}: unknown format ({type(data)})")
# raise Exception("Debug stop after printing timestep counts")

# Check prefix type for routing
use_f_q_g_q = False
use_heldout_over_time = False
if len(load_prefixes_to_use) > 0 and len(load_prefixes_to_use[0]) > 0:
    first_prefix = load_prefixes_to_use[0][0]

    if isinstance(first_prefix, str):
        if first_prefix.startswith("f_q_g_q_iwae_bounds"):
            use_f_q_g_q = True
        elif first_prefix.startswith("heldout_over_time_"):
            use_heldout_over_time = True


if use_heldout_over_time:
    # Plot heldout reward/return/f_q means over time (from heldout_over_time_* files)
    print("\nPlotting heldout and f_q over time...")

    plot_heldout_over_time(load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=fontsize, threshold=threshold)
elif use_f_q_g_q:
    # Try multiprompt v2 path first (works with or without target_samples_path)
    print("\nPlotting multiprompt per-prompt KL divergences from v2 f_q/g_q files...")
    all_v2_data = plot_f_q_g_q_kl_divergences_multiprompt(
        load_prefixes_to_use, labels, figname_modifier,
        target_samples_path=target_samples_path,
        x_range=None, fontsize=fontsize,
        individual_prompt_plots=individual_prompt_plots,
        random_f_q_ylim_low=random_f_q_ylim_low,
        n_frontiers=n_frontiers,
        frontier_legendfontsize=frontier_legendfontsize)
    # Also run aggregated path for backward compat (uses global logZ, not per-prompt)
    # Reuse already-loaded v2 data by converting to aggregated format
    print("\nPlotting approximate KL divergences from f_q/g_q files (aggregated)...")
    aggregated_data = v2_data_to_aggregated(all_v2_data) if all_v2_data is not None else None
    plot_f_q_g_q_kl_divergences(load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=fontsize, n_frontiers=n_frontiers, frontier_legendfontsize=frontier_legendfontsize, preloaded_data=aggregated_data)
else:
    # Process both base and sampling file types
    for file_type_suffix in ["base", "sampling"]:
        print(f"\nProcessing {file_type_suffix} files...")
        # x_range will be computed dynamically from data (can pass custom x_range if needed)
        process_file_type(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=fontsize)

    # x_range will be computed dynamically from data (can pass custom x_range if needed)
    plot_kl_divergences("sampling", load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=fontsize, n_frontiers=n_frontiers, frontier_legendfontsize=frontier_legendfontsize)


raise SystemExit(0)
