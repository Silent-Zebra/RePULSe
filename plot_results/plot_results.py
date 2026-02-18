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

from plot_utils import make_list, do_load_prefixes, generate_labels_from_prefixes, to_scalar, compute_global_logZ_from_iwae_bounds, compute_approx_kl_from_f_q_g_q

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


def plot_results_over_time(results_list, labels, x_range=None, fontsize=7, figname_modifier="",
                           index_to_use=0, plot_name="logprobbad", 
                           ylabel=r"Log Total Probability of Bad Output",
                           file_type_suffix="", load_prefixes_to_use=None):
    """
    Generic function to plot results over time with confidence bounds.
    
    Args:
        results_list: List of lists of loaded data
        labels: List of labels for each series
        x_range: X-axis range (optional, will be computed from data if None)
        fontsize: Font size for labels
        figname_modifier: Base name for the output file
        index_to_use: Index into the data tuple to plot
        plot_name: Name for the plot file
        ylabel: Y-axis label
        file_type_suffix: Suffix to add to filename ("base", "sampling", or "")
        load_prefixes_to_use: Optional list of lists of prefixes (used to extract parameters for samples per time step)
    """
    import re
    fig, ax1 = plt.subplots()

    # Calculate samples per time step from prefixes if available (needed for both x_range creation and extension)
    samples_per_timestep = 5000  # Default fallback
    if load_prefixes_to_use is not None and len(load_prefixes_to_use) > 0:

        # Try to extract parameters from the first prefix of the first series
        first_prefix = None
        for prefix_list in load_prefixes_to_use:
            if len(prefix_list) > 0:
                first_prefix = prefix_list[0]
                break
        
        if first_prefix is not None:
            # Extract harmlessness_training_num_episodes from _hepi pattern
            hepi_match = re.search(r'_he(\d+)', first_prefix) or re.search(r'_hepi(\d+)', first_prefix)
            harmlessness_training_num_episodes = int(hepi_match.group(1)) if hepi_match else None
            
            # Extract batch_size from _tbs or _tb pattern
            tbs_match = re.search(r'_tbs(\d+)', first_prefix) or re.search(r'_tb(\d+)', first_prefix)
            batch_size = int(tbs_match.group(1)) if tbs_match else None
            
            # Extract fit_steps from _fs pattern (another multiplier on samples_per_timestep)
            fs_match = re.search(r'_fs(\d+)', first_prefix)
            fit_steps = int(fs_match.group(1)) if fs_match else None
            
            # Calculate samples per time step if both parameters found
            if harmlessness_training_num_episodes is not None and batch_size is not None:
                samples_per_timestep = harmlessness_training_num_episodes * batch_size
                print(f"Auto-detected samples per fit step: {harmlessness_training_num_episodes} (episodes) * {batch_size} (batch_size) = {samples_per_timestep}")
            elif harmlessness_training_num_episodes is not None or batch_size is not None:
                print(f"Warning: Could not extract both parameters from prefix '{first_prefix}'. Using default 5000 samples per timestep.")
                print(f"  Found harmlessness_training_num_episodes: {harmlessness_training_num_episodes}, batch_size: {batch_size}")
            else:
                raise NotImplementedError

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
                        # Scalar, treat as single timestep
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
        
        # Create x_range: use indices scaled by calculated or default interval
        x_range = np.arange(max_timesteps) * samples_per_timestep


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
        
        # Ensure x_range matches the actual data length
        actual_timesteps = np_results.shape[1] if len(np_results.shape) > 1 else 1
        if len(x_range) != actual_timesteps:
            # Adjust x_range to match actual data length
            if actual_timesteps > len(x_range):
                # Extend x_range
                interval = x_range[1] - x_range[0] if len(x_range) > 1 else samples_per_timestep
                x_range_adjusted = np.arange(actual_timesteps) * interval
            else:
                # Truncate x_range
                x_range_adjusted = x_range[:actual_timesteps]
        else:
            x_range_adjusted = x_range
        
        plot_with_conf_bounds(
            ax1, np_results, x_range_adjusted, label=labels[i],
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


def plot_kl_divergences(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=7):
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
                          file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes)
    
    # Plot KL(q|sigma) = KL(proposal|target) from index 1
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=1, plot_name="kl_q_sigma", 
                          ylabel=r"KL(q|$\sigma$) = KL(proposal|target)",
                          file_type_suffix=file_type_suffix, load_prefixes_to_use=transformed_prefixes)


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
        return (f_q, g_q, iwae_lbs, iwae_ubs)
    elif isinstance(data, (tuple, list)) and len(data) >= 4:
        return data[:4]
    else:
        print(f"Warning: Unrecognized format for {path}, got {type(data)}. Skipping.")
        return None


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
            # Compute KL for each timestep
            T = len(f_q_estimates_list)
            kl_sigma_q_per_t = []
            kl_q_sigma_per_t = []
            
            for t in range(T):
                f_q_t = to_scalar(f_q_estimates_list[t])
                g_q_t = to_scalar(g_q_estimates_list[t])
                
                kl_sigma_q_t, kl_q_sigma_t = compute_approx_kl_from_f_q_g_q(
                    f_q_t, g_q_t, global_logZ
                )
                kl_sigma_q_per_t.append(kl_sigma_q_t)
                kl_q_sigma_per_t.append(kl_q_sigma_t)
            
            row.append((np.array(kl_sigma_q_per_t), np.array(kl_q_sigma_per_t)))
        results_list.append(row)
    
    return results_list


def plot_f_q_g_q_kl_divergences(load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=7, load_dir="./info"):
    """
    Plot approximate KL divergence metrics over time from f_q/g_q/IWAE bound files.
    
    Args:
        load_prefixes_to_use: List of lists of prefixes for f_q_g_q files
        labels: List of labels for each series
        figname_modifier: Figure name modifier
        x_range: X-axis range
        fontsize: Font size
        load_dir: Directory to load files from
    """
    # Load f_q/g_q files
    loaded_data = load_f_q_g_q_files_over_time(load_prefixes_to_use, load_dir=load_dir)
    
    # Check if we have any data
    has_data = any(len(exp_data) > 0 for exp_data in loaded_data)
    if not has_data:
        print(f"Warning: No f_q/g_q data found, skipping KL plots")
        return
    
    # Compute global log Z
    try:
        global_logZ = compute_global_logZ_from_iwae_bounds(loaded_data)
        print(f"Global log Z: {global_logZ}")
    except ValueError as e:
        print(f"Warning: Failed to compute global log Z: {e}, skipping KL plots")
        return
    
    # Compute KL estimates over time
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
                          file_type_suffix="", load_prefixes_to_use=load_prefixes_to_use)
    
    # Plot KL(q|sigma) = KL(proposal|target) from index 1
    plot_results_over_time(kl_results_list, labels, x_range, fontsize, figname_modifier,
                          index_to_use=1, plot_name="kl_q_sigma", 
                          ylabel=r"KL(q|$\sigma$) = KL(proposal|target)",
                          file_type_suffix="", load_prefixes_to_use=load_prefixes_to_use)


def _truncate_and_stack(arrays, context=""):
    """Stack 1D arrays: drop those <50% of max length, then truncate the rest to the shortest remaining."""
    assert len(arrays) > 0, f"_truncate_and_stack called with empty list{' (' + context + ')' if context else ''}"
    max_len = max(len(a) for a in arrays)
    threshold = max_len * 0.5
    kept = []
    for i, a in enumerate(arrays):
        if len(a) < threshold:
            ctx = f" ({context})" if context else ""
            print(f"Warning: dropping incomplete data at index {i} (length {len(a)} < 50% of max {max_len}){ctx}")
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


def _truncate_and_stack_2d(matrices, context=""):
    """Stack 2D arrays: drop those with cols <50% of max, then truncate the rest to the shortest remaining."""
    assert len(matrices) > 0, f"_truncate_and_stack_2d called with empty list{' (' + context + ')' if context else ''}"
    max_cols = max(m.shape[1] for m in matrices)
    threshold = max_cols * 0.5
    kept = []
    for i, m in enumerate(matrices):
        if m.shape[1] < threshold:
            ctx = f" ({context})" if context else ""
            print(f"Warning: dropping incomplete 2D data at index {i} (cols {m.shape[1]} < 50% of max {max_cols}){ctx}")
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

    # Drop aggregated keys not used by multiprompt plotting
    for key in ("f_q_estimates_list", "g_q_estimates_list", "iwae_lbs_list", "iwae_ubs_list",
                "prompt_texts_random_per_timepoint"):
        data.pop(key, None)

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
    for exp_i, prefix_list in enumerate(load_prefixes_to_use):
        exp_data = []
        for seed_j in range(len(prefix_list)):
            if (exp_i, seed_j) in results:
                exp_data.append(results[(exp_i, seed_j)])
        all_v2_data.append(exp_data)
    return all_v2_data


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
    """Extract x_range from prefixes using the same pattern as plot_results_over_time."""
    import re as _re
    samples_per_timestep = 5000  # Default fallback
    first_prefix = None
    for prefix_list in load_prefixes_to_use:
        if len(prefix_list) > 0:
            first_prefix = prefix_list[0]
            break
    if first_prefix is not None:
        hepi_match = _re.search(r'_he(\d+)', first_prefix) or _re.search(r'_hepi(\d+)', first_prefix)
        tbs_match = _re.search(r'_tbs(\d+)', first_prefix) or _re.search(r'_tb(\d+)', first_prefix)
        harmlessness_training_num_episodes = int(hepi_match.group(1)) if hepi_match else None
        batch_size = int(tbs_match.group(1)) if tbs_match else None
        if harmlessness_training_num_episodes is not None and batch_size is not None:
            samples_per_timestep = harmlessness_training_num_episodes * batch_size
            print(f"Auto-detected samples per timestep: {harmlessness_training_num_episodes} * {batch_size} = {samples_per_timestep}")
    return np.arange(n_timesteps) * samples_per_timestep


def plot_f_q_g_q_kl_divergences_multiprompt(
    load_prefixes_to_use, labels, figname_modifier,
    target_samples_path,
    x_range=None, fontsize=7, load_dir="./info",
    individual_prompt_plots=True,
    random_f_q_ylim_low=None,
):
    """
    Plot per-prompt and summary KL divergence metrics from multiprompt v2 f_q/g_q data.

    Generates:
    1. Per-prompt KL plots (two per prompt) in subfolder figs/<figname_modifier>/
    2. Summary plots: mean KL across prompts per seed, with CI over seeds
    3. Average f_q plot for random prompt set
    """
    # Load v2 files
    all_v2_data = _load_v2_f_q_g_q_files(load_prefixes_to_use, load_dir)

    has_data = any(len(exp_data) > 0 for exp_data in all_v2_data)
    if not has_data:
        print("Warning: No v2 f_q/g_q data found, skipping multiprompt KL plots")
        return

    # Load target samples to identify prompts with actual target samples
    prompts_with_targets = _get_prompts_with_targets(target_samples_path)

    # Create output subfolder for per-prompt plots
    per_prompt_dir = os.path.join("figs", figname_modifier)
    os.makedirs(per_prompt_dir, exist_ok=True)
    print(f"Per-prompt plots will be saved to: {per_prompt_dir}/")

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
        for seed_j, v2_data in enumerate(exp_data):
            if prompt_texts is None:
                prompt_texts = v2_data["prompt_texts_fixed"]

            kl_by_prompt = _compute_per_prompt_kl_over_time(v2_data, global_log_Z)
            random_fq = _compute_random_f_q_over_time(v2_data)

            exp_kl.append(kl_by_prompt)
            exp_random_fq.append(random_fq)

            print(f"Exp {exp_i} seed {seed_j}: {len(kl_by_prompt)} prompts with KL, "
                  f"random f_q timesteps: {len(random_fq) if random_fq is not None else 0}")

        all_kl_data.append(exp_kl)
        all_random_fq.append(exp_random_fq)

    if prompt_texts is None:
        print("Warning: No prompt texts found, skipping multiprompt plots")
        return

    # Determine common prompt indices (present across all seeds of first experiment)
    # Use first experiment's first seed as reference
    ref_kl = all_kl_data[0][0] if all_kl_data[0] else {}
    common_prompt_indices = sorted(ref_kl.keys())
    T = len(next(iter(ref_kl.values()))[0]) if ref_kl else 0

    if T == 0:
        print("Warning: No timesteps found, skipping multiprompt plots")
        return

    if x_range is None:
        x_range = _extract_x_range(load_prefixes_to_use, T)

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
                np_results = _truncate_and_stack(seed_trajectories, context=f"prompt {p} {kl_name}, series '{labels[exp_i]}'")
                x_range_adj = x_range[:np_results.shape[1]] if len(x_range) > np_results.shape[1] else x_range
                plot_with_conf_bounds(ax, np_results, x_range_adj, label=labels[exp_i],
                                      color=color_list_for_fqs[exp_i],
                                      linestyle=linestyle_list[exp_i])

            ax.set_xlabel("Number of Samples", fontsize=fontsize)
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
    for kl_idx, (kl_name, kl_ylabel) in enumerate([
        ("kl_q_sigma", r"KL(q|$\sigma$) mean over prompts"),
        ("kl_sigma_q", r"KL($\sigma$|q) mean over prompts"),
    ]):
        # Build results_list compatible with plot_results_over_time:
        # results_list[exp_i] = list of seeds, each seed is a tuple where index kl_idx gives array (T,)
        summary_results_list = []
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
            summary_results_list.append(seed_means)

        fig, ax = plt.subplots()
        for exp_i in range(len(summary_results_list)):
            if not summary_results_list[exp_i]:
                continue
            np_results = _truncate_and_stack(summary_results_list[exp_i], context=f"seeds for summary {kl_name}, series '{labels[exp_i]}'")  # (n_seeds, T)
            x_range_adj = x_range[:np_results.shape[1]] if len(x_range) > np_results.shape[1] else x_range
            plot_with_conf_bounds(ax, np_results, x_range_adj, label=labels[exp_i],
                                  color=color_list_for_fqs[exp_i],
                                  linestyle=linestyle_list[exp_i])

        ax.set_xlabel("Number of Samples", fontsize=fontsize)
        ax.set_ylabel(kl_ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        figname = os.path.join(per_prompt_dir, f"summary_{kl_name}.pdf")
        plt.savefig(figname)
        plt.clf()
        plt.close(fig)

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
            # Pad to same length (different seeds may have different T_random)
            max_T = max(len(t) for t in seed_trajectories)
            padded = np.full((len(seed_trajectories), max_T), np.nan)
            for j, traj in enumerate(seed_trajectories):
                padded[j, :len(traj)] = traj
            x_range_random = _extract_x_range(load_prefixes_to_use, max_T)
            plot_with_conf_bounds(ax, padded, x_range_random, label=labels[exp_i],
                                  color=color_list_for_fqs[exp_i],
                                  linestyle=linestyle_list[exp_i])

        ax.set_xlabel("Number of Samples", fontsize=fontsize)
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
    if not individual_prompt_plots:
        print("\nSkipping KL heatmaps (individual_prompt_plots=False)")
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
                heatmap_data = np.nanmean(_truncate_and_stack_2d(seed_matrices, context=f"heatmap seeds, {kl_name}, series '{labels[exp_i]}'"), axis=0)

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

                ax.set_xlabel("Number of Samples", fontsize=fontsize)
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
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-05_bl0.0_ppq_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al1e-06_bl0.0_ppq_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-05_bl0.0_ppq_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-06_bl0.0_ppq_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-07_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-07_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb200_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_miprAL_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he10_fs100_scc_al3e-07_bl0.0_ppq_tb200_s2", 1,10),

]
threshold = -5
figname_modifier = "len20_noit_multitoy_02_18_b-20"
target_samples_path = "info/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_miprAL_tsa34.pt"
individual_prompt_plots = False
random_f_q_ylim_low = 150



load_prefixes_to_use = [
# for x in $(ls /scratch/zhaostep/OpenRLHF/info/noitmultitest/ |  grep f_q | grep _s2 ); do echo make_list\(\"$x\", 1,10\)\,; done
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al0.0001_bl0.0_ppq_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.3_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-05_bl0.0_ppq_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al1e-06_bl0.0_ppq_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-05_bl0.0_ppq_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_cf0.5_cd64_cfr0.001_cfsn_af_fo_tb250_s2", 1,10),
make_list("f_q_g_q_iwae_bounds_OpenRLHF_rlhf_rc10.0_Sm13_To_20misi1_l20_kl0.0_b-20.0_hlnt_a0.0_ppq_ctl_ep1_e1_he4_scc_al3e-06_bl0.0_ppq_tb250_s2", 1,10),

]
threshold = -5
figname_modifier = "len20_noit_multi_02_18_b-20"
target_samples_path = "info/target_samples_Sm13_To_rlhf_l20_b-20.0_rc10.0_20misi1_tsa100.pt"
individual_prompt_plots = False
random_f_q_ylim_low = 150


random_f_q_ylim_low = random_f_q_ylim_low if 'random_f_q_ylim_low' in vars() else None
labels = generate_labels_from_prefixes(load_prefixes_to_use)


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
    if target_samples_path is not None:
        # Multiprompt v2 path: per-prompt KL plots + summary + random f_q
        print("\nPlotting multiprompt per-prompt KL divergences from v2 f_q/g_q files...")
        plot_f_q_g_q_kl_divergences_multiprompt(
            load_prefixes_to_use, labels, figname_modifier,
            target_samples_path=target_samples_path,
            x_range=None, fontsize=fontsize,
            individual_prompt_plots=individual_prompt_plots,
            random_f_q_ylim_low=random_f_q_ylim_low)
    else:
        # Aggregated path (original): single global log Z
        print("\nPlotting approximate KL divergences from f_q/g_q files...")
        plot_f_q_g_q_kl_divergences(load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=fontsize)
else:
    # Process both base and sampling file types
    for file_type_suffix in ["base", "sampling"]:
        print(f"\nProcessing {file_type_suffix} files...")
        # x_range will be computed dynamically from data (can pass custom x_range if needed)
        process_file_type(file_type_suffix, load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=fontsize)

    # x_range will be computed dynamically from data (can pass custom x_range if needed)
    plot_kl_divergences("sampling", load_prefixes_to_use, labels, figname_modifier, x_range=None, fontsize=fontsize)


raise SystemExit(0)
