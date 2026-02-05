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
        np_results = np.stack(filtered_results)
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
            if not isinstance(data, (tuple, list)) or len(data) < 4:
                print(f"Warning: Expected 4-tuple for {path}, got {type(data)}. Skipping.")
                continue
            f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list = data[:4]
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
figname_modifier = "len20_repulse_dis_02_04_b-20"


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
    # Plot approximate KL divergences from f_q/g_q files
    print("\nPlotting approximate KL divergences from f_q/g_q files...")
    # x_range will be computed dynamically from data (can pass custom x_range if needed)
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
