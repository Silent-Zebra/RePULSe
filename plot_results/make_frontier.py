import matplotlib

matplotlib.use('PDF') # THIS MUST BE AT THE START OF THE CODE (before other imports)!!!!
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from plot_utils import *

import torch

import scipy.stats as stats


from scipy.stats import norm

from plot_utils import generate_labels_from_prefixes, to_scalar, compute_global_logZ_from_iwae_bounds, compute_approx_kl_from_f_q_g_q
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
                    # Note: This code works with both 6-element (old) and 7-element (new with bonus) tuples
                    # tuple_index selects reward (0) or return (1) data, t[0] is always the first element
                    # Bonus values at index 6 are not used in frontier plots
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
                        f"  {labels[i]}: X = {x_observed_mean:.3f} [{x_ci_lower:.3f}, {x_ci_upper:.3f}], Y = {y_observed_mean:.2f} [{y_ci_lower:.2f}, {y_ci_upper:.2f}], n_seeds={n_seeds}")

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


def make_frontier_exact_kl_bootstrap(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7, legendfontsize=7,
    aggregate_seeds=False, alpha_error=0.2,
    n_bootstrap_draws=5000,  # Added parameter for number of bootstrap draws
    compare_to_reference=False,
    ylimlow=None, ylimhigh=None,
    alpha_list=None,
    size_list=None,
    connect_groups=None,
    title=None,
):
    """
    Plot KL divergence on two axes (KL(sigma_p || q) vs KL(q || sigma_p)),
    using exact/analytic KL values per seed.

    results_list should contain tuples of (kl_sigma_q_list, kl_q_sigma_list, metrics_list)
    where kl_sigma_q_list and kl_q_sigma_list are lists of KL values per prompt/evaluation.

    If connect_groups is provided, it should be a list of group IDs (one per series).
    Series in the same group share a legend entry and are connected by lines.
    """
    plt.clf()
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize + 1)

    all_x = []
    all_y = []

    # For connect_groups: find the last series index per group *that has data*
    # (for legend deduplication) and track plotted coordinates per group (for connecting lines)
    if connect_groups is not None:
        last_in_group = {}
        for idx in range(len(connect_groups)):
            has_data = (isinstance(results_list[idx], list) and len(results_list[idx]) > 0) or (
                isinstance(results_list[idx], tuple))
            if has_data:
                last_in_group[connect_groups[idx]] = idx
        group_plotted_coords = {}  # group_id -> list of (x, y)

    for i in range(len(labels)):

        if isinstance(results_list[i], tuple):
            raise NotImplementedError # hasn't been used/tested in a while

        else:  # results_list[i] is a list of tuples (per-seed data)
            tuple_list = results_list[i]
            x_values_all_seeds, y_values_all_seeds = _parse_kl_seed_data(tuple_list, labels[i])
            if x_values_all_seeds is None:
                continue

            if compare_to_reference:
                print("Warning: compare_to_reference not tested in quite a while")
                if i == 0:
                    reference_x = x_values_all_seeds
                    reference_y = y_values_all_seeds
                x_values_all_seeds = x_values_all_seeds - reference_x[:x_values_all_seeds.shape[0]]
                y_values_all_seeds = y_values_all_seeds - reference_y[:y_values_all_seeds.shape[0]]

            point_alpha = alpha_list[i] if alpha_list is not None else 1.0

            # Determine legend label: deduplicate when using connect_groups
            # (only the last series in each group gets a legend entry, so the legend
            # marker/color matches the full-opacity point)
            if connect_groups is not None and i != last_in_group[connect_groups[i]]:
                use_label = '_nolegend_'
            else:
                use_label = labels[i]

            point_size = size_list[i] if size_list is not None else None
            scatter_kwargs = dict(label=use_label, c=color_list[i], marker=marker_list[i], alpha=point_alpha)
            if point_size is not None:
                scatter_kwargs['s'] = point_size

            if not aggregate_seeds:
                plt.scatter(x_values_all_seeds, y_values_all_seeds, **scatter_kwargs)
            else:  # aggregate_seeds is True, perform bootstrap
                n_seeds = x_values_all_seeds.shape[0]

                if n_seeds == 0:  # Should be caught by empty tuple_list earlier
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

                    print(
                        f"  {labels[i]}: X = {x_observed_mean:.3f} [{x_ci_lower:.3f}, {x_ci_upper:.3f}], Y = {y_observed_mean:.3f} [{y_ci_lower:.3f}, {y_ci_upper:.3f}], n_seeds={n_seeds}")

                # Plot the observed mean
                agg_scatter_kwargs = dict(label=use_label, c=color_list[i],
                                         marker=marker_list[i], alpha=point_alpha, zorder=3)
                if point_size is not None:
                    agg_scatter_kwargs['s'] = point_size
                plt.scatter(x_observed_mean, y_observed_mean, **agg_scatter_kwargs)

                # Plot error bars if they were computed
                if x_err_bootstrap is not None and y_err_bootstrap is not None:
                    plt.errorbar(
                        x_observed_mean,
                        y_observed_mean,
                        xerr=x_err_bootstrap,
                        yerr=y_err_bootstrap,
                        fmt='',  # No line connecting points, marker is from scatter
                        ecolor=color_list[i],
                        alpha=alpha_error * point_alpha,
                        capsize=2,
                        zorder=2,
                    )

                all_x.append(x_observed_mean)
                all_y.append(y_observed_mean)

                # Track coordinates for connecting lines
                if connect_groups is not None:
                    gid = connect_groups[i]
                    if gid not in group_plotted_coords:
                        group_plotted_coords[gid] = []
                    group_plotted_coords[gid].append((x_observed_mean, y_observed_mean))

    # Draw connecting lines between points in the same group
    if connect_groups is not None:
        for gid, coords in group_plotted_coords.items():
            if len(coords) > 1:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                # Use the color of the first series in this group
                group_color = None
                for idx in range(len(connect_groups)):
                    if connect_groups[idx] == gid:
                        group_color = color_list[idx]
                        break
                plt.plot(xs, ys, color=group_color, linewidth=1, linestyle='-', alpha=0.5, zorder=1)

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    if (ylimlow is not None) or (ylimhigh is not None):
        plt.ylim(ylimlow, ylimhigh)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.legend(fontsize=legendfontsize)

    plt.savefig(figname)
    print(f"Figure saved to {figname}")


def make_frontier_exact_kl_individual(
    xlabel, ylabel, figname, labels, results_list,
    color_list, marker_list, xlimlow=None, xlimhigh=None, fontsize=7, legendfontsize=7,
    ylimlow=None, ylimhigh=None,
    size_list=None,
    connect_groups=None,
    line_alpha=0.25,
):
    """
    Individual-seed version of the combined KL frontier plot.

    Instead of bootstrap-aggregated means with CIs, each seed is plotted as a separate point
    with a per-seed marker (from SEED_MARKERS). Seeds of the same setting share a color;
    one legend entry per setting.

    If connect_groups is provided, each seed's trajectory across time steps within the same
    group is connected by a low-opacity line.

    The results_list / connect_groups structure is the same as make_frontier_exact_kl_bootstrap's
    combined frontier: one entry per (time_step, experiment) pair.
    """
    plt.clf()
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    # Reorganize data: group by (experiment, seed) to enable per-seed trajectories
    # connect_groups maps each entry in results_list to an experiment index.
    # Within each experiment, entries appear once per time step (in order).
    # Each entry's results_list[i] is a list of per-seed tuples.

    # First, figure out unique experiments and their time-ordered entries
    if connect_groups is not None:
        # Collect entries per experiment in order
        exp_entries = {}  # exp_id -> list of indices into results_list
        for idx, gid in enumerate(connect_groups):
            if gid not in exp_entries:
                exp_entries[gid] = []
            exp_entries[gid].append(idx)
    else:
        # No grouping: each entry is its own experiment
        exp_entries = {i: [i] for i in range(len(results_list))}

    # Track which settings have been added to legend
    legend_added = set()

    for exp_id, entry_indices in exp_entries.items():
        # Determine the setting color from the first entry
        first_idx = entry_indices[0]
        setting_color = color_list[first_idx]
        setting_label = labels[first_idx]

        # Parse all time steps for this experiment: list of (x_per_seed, y_per_seed) arrays
        timestep_data = []
        for idx in entry_indices:
            tuple_list = results_list[idx]
            if isinstance(tuple_list, tuple):
                raise NotImplementedError
            x_vals, y_vals = _parse_kl_seed_data(tuple_list, labels[idx])
            timestep_data.append((x_vals, y_vals))

        # Determine the number of seeds (use the max across time steps)
        n_seeds = max((len(xv) for xv, yv in timestep_data if xv is not None), default=0)
        if n_seeds == 0:
            continue

        # Plot each seed
        for seed_idx in range(n_seeds):
            seed_marker = SEED_MARKERS[seed_idx % len(SEED_MARKERS)]

            # Collect this seed's (x, y) across time steps
            seed_xs = []
            seed_ys = []
            for t_pos, (x_vals, y_vals) in enumerate(timestep_data):
                if x_vals is None or seed_idx >= len(x_vals):
                    continue
                sx, sy = x_vals[seed_idx], y_vals[seed_idx]
                seed_xs.append(sx)
                seed_ys.append(sy)

                # Determine point size
                entry_idx = entry_indices[t_pos]
                point_size = size_list[entry_idx] if size_list is not None else None

                # Only first seed, first time step of this experiment gets a legend entry
                if exp_id not in legend_added and seed_idx == 0 and t_pos == 0:
                    use_label = setting_label
                    legend_added.add(exp_id)
                else:
                    use_label = '_nolegend_'

                scatter_kwargs = dict(c=setting_color, marker=seed_marker, alpha=0.8, label=use_label)
                if point_size is not None:
                    scatter_kwargs['s'] = point_size
                plt.scatter(sx, sy, **scatter_kwargs)

            # Connect this seed's trajectory across time steps
            if len(seed_xs) > 1:
                plt.plot(seed_xs, seed_ys, color=setting_color, linewidth=0.8,
                         linestyle='-', alpha=line_alpha, zorder=1)

    if (xlimlow is not None) or (xlimhigh is not None):
        plt.xlim(xlimlow, xlimhigh)
    if (ylimlow is not None) or (ylimhigh is not None):
        plt.ylim(ylimlow, ylimhigh)
    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.legend(fontsize=legendfontsize)

    plt.savefig(figname)
    print(f"Figure saved to {figname}")


def _parse_kl_seed_data(tuple_list, label):
    """Parse per-seed KL data from a tuple_list into x/y arrays.

    Each element of tuple_list should be a tuple: (kl_sigma_q_list, kl_q_sigma_list, ...).
    Returns (x_values_all_seeds, y_values_all_seeds) as numpy arrays, where
    x = mean KL(sigma||q) per seed and y = mean KL(q||sigma) per seed.
    Returns (None, None) if the list is empty or has no valid data.
    """
    x_results_per_seed = []
    y_results_per_seed = []

    if not tuple_list:
        print(f"Warning: Empty tuple_list for {label}. Skipping.")
        return None, None

    for t_idx, t in enumerate(tuple_list):
        if not isinstance(t, tuple) or len(t) < 2:
            print(f"Warning: Expected tuple with at least 2 elements for {label}, seed {t_idx+1}. Got {type(t)}. Skipping.")
            continue

        kl_sigma_q_list = t[0]
        kl_q_sigma_list = t[1]

        if isinstance(kl_sigma_q_list, list):
            kl_sigma_q_array = np.array(kl_sigma_q_list)
        elif isinstance(kl_sigma_q_list, torch.Tensor):
            kl_sigma_q_array = kl_sigma_q_list.float().cpu().numpy()
        else:
            kl_sigma_q_array = np.array([kl_sigma_q_list])

        if isinstance(kl_q_sigma_list, list):
            kl_q_sigma_array = np.array(kl_q_sigma_list)
        elif isinstance(kl_q_sigma_list, torch.Tensor):
            kl_q_sigma_array = kl_q_sigma_list.float().cpu().numpy()
        else:
            kl_q_sigma_array = np.array([kl_q_sigma_list])

        x_results_per_seed.append(kl_sigma_q_array.mean())
        y_results_per_seed.append(kl_q_sigma_array.mean())

    if not x_results_per_seed:
        return None, None

    x_values = np.array(x_results_per_seed)
    y_values = np.array(y_results_per_seed)

    # Truncate to matching lengths
    min_len = min(len(x_values), len(y_values))
    if len(x_values) != len(y_values):
        print(f"WARNING: x/y length mismatch for {label}, truncating to {min_len}")
    return x_values[:min_len], y_values[:min_len]


def load_f_q_g_q_iwae_and_compute_approx_kl(load_dir, f_q_load_prefixes_to_use, map_location='cpu'):
    """
    Load f_q/g_q/IWAE bound files, compute a single global log Z (median over all
    per-seed and per-experiment estimates), then build results_list of (kl_sigma_q, kl_q_sigma)
    per seed in the format expected by make_frontier_exact_kl_bootstrap.

    f_q_load_prefixes_to_use: list of lists of filenames; inner list = one experiment,
    each filename = one seed (full basename, e.g. f_q_g_q_iwae_bounds_OpenRLHF_..._s1).
    """
    # Phase 1: load all data
    cached = []  # cached[i] = list of (f_q_list, g_q_list, iwae_lbs_list, iwae_ubs_list) for each seed

    for i in range(len(f_q_load_prefixes_to_use)):
        prefix_list = f_q_load_prefixes_to_use[i]
        exp_data = []
        for fn in prefix_list:
            path = os.path.join(load_dir, fn)
            try:
                data = torch.load(path, map_location=map_location)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
            # Handle both v1 (tuple/list) and v2 (dict) formats
            if isinstance(data, dict) and data.get("version", 1) >= 2:
                f_q_estimates_list = data.get("f_q_estimates_list", [])
                g_q_estimates_list = data.get("g_q_estimates_list", [])
                iwae_lbs_list = data.get("iwae_lbs_list", [])
                iwae_ubs_list = data.get("iwae_ubs_list", [])
                # Prefer per-prompt averaged IWAE bounds (aggregate is not meaningful
                # in multi-prompt settings — see mean_of_per_prompt_bounds docstring).
                iwae_lbs_bp = data.get("iwae_lbs_by_prompt_fixed")
                if iwae_lbs_bp is not None and len(iwae_lbs_bp) > 0:
                    iwae_lbs_list = mean_of_per_prompt_bounds(iwae_lbs_bp)
                iwae_ubs_bp = data.get("iwae_ubs_by_prompt_fixed")
                if iwae_ubs_bp is not None and len(iwae_ubs_bp) > 0:
                    iwae_ubs_list = mean_of_per_prompt_bounds(iwae_ubs_bp)
            elif isinstance(data, (tuple, list)) and len(data) >= 4:
                f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list = data[:4]
            else:
                print(f"Warning: Unrecognized format for {path}, got {type(data)}. Skipping.")
                continue
            exp_data.append((f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list))
        cached.append(exp_data)

    if not cached or all(len(exp_data) == 0 for exp_data in cached):
        return [[] for _ in range(len(f_q_load_prefixes_to_use))]

    # Compute global log Z using extracted utility
    try:
        global_logZ = compute_global_logZ_from_iwae_bounds(cached)
    except ValueError:
        return [[] for _ in range(len(f_q_load_prefixes_to_use))]

    print(f"Global log Z: {global_logZ}")

    # Phase 2: build results_list using global_logZ for every seed
    results_list = []
    for i in range(len(f_q_load_prefixes_to_use)):
        exp_data = cached[i]
        row = []
        for f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list in exp_data:
            if len(f_q_estimates_list) == 0 or len(g_q_estimates_list) == 0:
                continue
            f_q_last = to_scalar(f_q_estimates_list[-1])
            g_q_last = to_scalar(g_q_estimates_list[-1])

            kl_sigma_q_seed, kl_q_sigma_seed = compute_approx_kl_from_f_q_g_q(
                f_q_last, g_q_last, global_logZ
            )
            row.append((np.array([kl_sigma_q_seed]), np.array([kl_q_sigma_seed])))
        results_list.append(row)
    return results_list


def make_frontier_approx_kl_bootstrap(
    load_dir, f_q_load_prefixes_to_use, labels, color_list, marker_list,
    xlabel, ylabel, figname,
    xlimlow=None, xlimhigh=None, fontsize=7, legendfontsize=7,
    aggregate_seeds=False, alpha_error=0.2,
    n_bootstrap_draws=5000,
    compare_to_reference=False,
    ylimlow=None, ylimhigh=None,
    map_location='cpu',
):
    """
    Plot approx KL frontier from f_q/g_q/IWAE bound files. Loads those files, computes
    a global log Z (median of per-seed and per-experiment estimates), then KL(q||sigma)
    and KL(sigma||q) per seed and forwards to make_frontier_exact_kl_bootstrap.
    """
    results_list = load_f_q_g_q_iwae_and_compute_approx_kl(
        load_dir, f_q_load_prefixes_to_use, map_location=map_location
    )
    make_frontier_exact_kl_bootstrap(
        xlabel=xlabel,
        ylabel=ylabel,
        figname=figname,
        labels=labels,
        results_list=results_list,
        color_list=color_list,
        marker_list=marker_list,
        xlimlow=xlimlow,
        xlimhigh=xlimhigh,
        fontsize=fontsize,
        legendfontsize=legendfontsize,
        aggregate_seeds=aggregate_seeds,
        alpha_error=alpha_error,
        n_bootstrap_draws=n_bootstrap_draws,
        compare_to_reference=compare_to_reference,
        ylimlow=ylimlow,
        ylimhigh=ylimhigh,
    )


if __name__ == "__main__":

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
    figname_modifier = "1B_len100_10_23_kl2_sameepi_final"
    figname_modifier = "1B_len100_10_23_kl2_cvar_final"


    figname_modifier = "toy_len1_01_07_kl_div"
    figname_modifier = "toy_len1_epi10_01_07_kl_div"

    figname_modifier = "toy_len1_01_11_kl_div"
    figname_modifier = "toy_len1_01_12_kl_div"
    figname_modifier = "toy_len1_01_13_kl_div"
    figname_modifier = "toy_len1_01_14_kl_div"
    figname_modifier = "toy_len1_01_14_kl_div_fixed_combined"
    figname_modifier = "toy_len1_01_14_kl_div_fixed_combined2"
    figname_modifier = "toy_len1_01_15_kl_div_fixed_combined"
    figname_modifier = "toy_len1_01_16_kl_div_fixed_combined"
    figname_modifier = "toy_len1_01_17_kl_div_wnn"


    figname_modifier = "toy_len1_01_27_kl_div_approx"
    figname_modifier = "toy_len1_01_27_kl_div_approx_400"
    figname_modifier = "toy_len1_01_27_kl_div_approx_400_v2"


    do_1B_experiments = False
    if "1B" in figname_modifier:
        do_1B_experiments = True


    linestyle_list = ['solid'] * 30


    color_list = [
        'xkcd:green', 'xkcd:blue', 'xkcd:red', 'xkcd:orange',  'xkcd:purple',
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
        if "kl_div" in figname_modifier:

            if "approx" not in figname_modifier:
                # Load KL divergence data from analytic_kls_toxicity files
                # Option 1: Manually specify your KL divergence file prefixes (recommended)
                # Uncomment and modify the section below:
                #
                kl_load_prefixes_to_use = [
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_s1",
                        1, 10),
                    # make_list(
                    #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn_s1",
                    #     1, 10),
                    # make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd16_cflr0.0001_cfus1024_s1",1,5),
                    # make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd1024_cflr0.0001_cfus1024_s1",1,10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_count_s1",
                        1, 10),


                    # 0 CFUS (or 0 CF lr) is just testing the effect of essentially just adding noise to the reward model (random, unguided noise)
                    # for x in $(ls info/exploretoyrlhfmultifixedcombined/ | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
                    # make_list(
                    #     "after_not_before_analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus64_s3",
                    #     1, 10),
                    make_list(
                        "after_not_before_analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.003_cfus64_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_after_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_before_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_after_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_before_s3",
                        1, 10),
                    # make_list(
                    #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus64_before_s3",
                    #     1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.003_after_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.003_before_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.003_cfus4_after_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.003_cfus4_before_s3",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.003_cfus64_before_s3",
                        1, 10),
                    # make_list(
                    #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.0_after_s3",
                    #     1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.0_cfus0_after_s3",
                        1, 10),

                    # for x in $(ls info/exploretoyrlhfmultifixedcombined/ | grep firstonl | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_after_firstonline_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_before_firstonline_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_after_firstonline_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_before_firstonline_s2",
                        1, 10),

                    # for x in $(ls info/exploretoyrlhfmultifixedcombined2/ | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_after_pri_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_before_pri_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_after_firstonline_pri_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_after_pri_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_before_firstonline_pri_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_before_pri_s2",
                        1, 10),

                ]

                kl_load_prefixes_to_use = [
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_s1",
                        1, 10),
                    # make_list(
                    #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn_s1",
                    #     1, 10),
                    # make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd16_cflr0.0001_cfus1024_s1",1,5),
                    # make_list("analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd1024_cflr0.0001_cfus1024_s1",1,10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_count_s1",
                        1, 10),

                    # 0 CFUS (or 0 CF lr) is just testing the effect of essentially just adding noise to the reward model (random, unguided noise)
                    # for x in $(ls info/exploretoyrlhfmultifixedcombined/ | grep _s3); do echo make_list\(\"$x\", 1,10\)\,; done
                    # make_list(
                    #     "after_not_before_analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus64_s3",
                    #     1, 10),
                    make_list(
                        "after_not_before_analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.003_cfus64_s3",
                        1, 10),

                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.0_cfus0_after_s3",
                        1, 10),

                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_after_s3",
                        1, 10),




                    # for x in $(ls info/exploretoyrlhfmultifixedcombined/ | grep firstonl | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_after_firstonline_s2",
                        1, 10),




                    # for x in $(ls info/exploretoyrlhfmultifixedcombined2/ | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_after_pri_s2",
                        1, 10),

                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_after_s3",
                        1, 10),

                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_after_firstonline_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfus4_after_firstonline_pri_s2",
                        1, 10),

                    # for x in $(ls info/exploretoyrlhfmultifixedcombined3/ | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_after_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfbias_after_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfhis0.001_after_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfhis0.001_fpis0.1_after_s2",
                        1, 10),
                    # make_list(
                    #     "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfhis0.1_fpis0.1_after_s2",
                    #     1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_fpis0.001_after_s2",
                        1, 10),

                    # for x in $(ls info/explorecoinflipnn/ | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.0001_cfsepnn_after_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.0001_cfus64_cfsepnn_after_s2",
                        1, 10),
                    make_list(
                        "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi5_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn3.0_cfd64_cflr0.001_cfsepnn_after_s2",
                        1, 10),

                ]

                # kl_load_prefixes_to_use = [
                #     make_list(
                #         "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_s1",
                #         1, 10),
                #     make_list(
                #         "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_cfn_s1",
                #         1, 10),
                #     make_list(
                #         "analytic_kls_toxicity_rlhf_di_To_thmaisa_len1_kl0.0_beta-1.0_harml_neg_training_a0.0_policy_psi_q_p_s_t_ctl_epo1_epi10_schconstant_alr3e-05_blr0.0_policy_psi_q_p_s_t_count_s1",
                #         1, 10),
                # ]
                load_prefixes_to_use = kl_load_prefixes_to_use

                marker_list = ["D", "o", "P", "v", "x", "x", "x", "^", "^", "^", "o", "P", "v", "v", "v", "P", "o", "o",
                               "P", "o", "o", "P", "^", "^", "P", "v", "v", "v",
                               "v", "v", "^", "P", "v", "D", "v", "v", "x", "v",
                               # 22 23 25 27 28 reinf, ours, baseprop, reinftransf ppo
                               "v", "v", "v", "^", "^", "x", "x", "x", "x", "D",
                               "P", "P", "P", "v", "v", "v", "^", "^", "x", "x", "x", "x", "D", "P", "P", "P"]
            else:
                # approx KL stuff
                # for x in $(ls info/toyrepulse2p2v2/ | grep f_q_g_q | grep he20 | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done

                kl_load_prefixes_to_use = [
                    make_list(
                        "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l1_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_c3.0_tb5_s2",
                        1, 10),
                    make_list(
                        "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l1_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2",
                        1, 10),
                    make_list(
                        "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l1_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s2",
                        1, 10),

                ]
                figname_modifier = "toy_len1_01_27_kl_div_approx_400_fixed"

                # # for x in $(ls /scratch/zhaostep/OpenRLHF/info/toyrepulse2p2len2/ | grep f_q_g_q | grep he20 | grep s2); do echo make_list\(\"$x\", 1,10\)\,; done
                # kl_load_prefixes_to_use = [
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0001_cfsn_af_fo_tb5_s2",
                #         1, 10),
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0001_cfsn_bf_fo_tb5_s2",
                #         1, 10),
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0003_cfsn_af_fo_tb5_s2",
                #         1, 10),
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.0003_cfsn_bf_fo_tb5_s2",
                #         1, 10),
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_af_fo_tb5_s2",
                #         1, 10),
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_cfsn_bf_fo_tb5_s2",
                #         1, 10),
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_cf3.0_cd64_cfr0.001_fp0.01_cfsn_af_fo_tb5_s2",
                #         1, 10),
                #     make_list(
                #         "f_q_g_q_iwae_bounds_OpenRLHF_rlhf_di_To_2_l2_kl0.0_b-1.0_hlnt_a0.0_ppq_ctl_ep1_e1_he20_scc_al3e-05_bl0.0_ppq_tb5_s2",
                #         1, 10),
                # ]
                # figname_modifier = "toy_len2_01_27_kl_div_approx_400"
                #
                #
                # for x in $(ls /scratch/zhaostep/OpenRLHF/info/toyrepulse2p2len2/ | grep f_q_g_q | grep 001 | grep af | grep he20 | grep s2); do echo make_list\(\"$x\", 1,5\)\,; done
                kl_load_prefixes_to_use = [
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
            threshold = 0.003
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
        if "kl_div" in figname_modifier:
            labels = generate_labels_from_prefixes(kl_load_prefixes_to_use)
        else:

            labels = [
                '_'.join(a[0].split('len100_')[-1].split('_policy_psi_q_p_s_t_ctl_epo1_')).split('_policy_psi_q_p_s_t')[0]
                for a in load_prefixes_to_use]

    if not use_handcrafted_labels:
        # fontsize = 8
        fontsize = 6
        legendfontsize = 6

    if inds_to_use is None:
        pass
    else:
        labels = [labels[i] for i in inds_to_use]
        load_prefixes_to_use = [load_prefixes_to_use[i] for i in inds_to_use]
        marker_list = [marker_list[i] for i in inds_to_use]
        color_list = [color_list[i] for i in inds_to_use]

    # Opt-in semantic styling: override hardcoded color/marker/linestyle lists
    # based on experiment prefix properties (loss type, bonus type, LR, CFN alpha).
    # Set to False to use the hardcoded lists above instead.
    use_semantic_styling = True
    if use_semantic_styling:
        from plot_utils import generate_visual_style_from_prefixes
        color_list, marker_list, linestyle_list = generate_visual_style_from_prefixes(load_prefixes_to_use)

    # if "kl_div" in figname_modifier:
    #     fontsize = 10
    #     labels = []
    #     for x in load_prefixes_to_use:
    #         if "cfn" in x[0]:
    #             labels.append("Coin Flip Net Pseudo-count")
    #         elif "count" in x[0]:
    #             labels.append("Exact Count")
    #         else:
    #             labels.append("No Exploration Bonus")




    # Below is for new exploration based experiments (toy env for now only with analytic kls)
    # Optionally plot KL divergence on two axes
    # Set this flag to True to enable KL divergence plotting
    do_kl_plot = False
    # You can also enable it by adding "kl_div" to figname_modifier
    if "kl_div" in figname_modifier:
        do_kl_plot = True




    if do_kl_plot:

        # # Option 2: Auto-convert from load_prefixes_to_use (may need adjustment)
        # # This assumes your files follow the pattern: analytic_kls_toxicity_<rest_of_name>
        # # If Option 1 is used above, comment out the section below:
        # kl_load_prefixes_to_use = []
        # # Convert existing prefixes to KL divergence file prefixes
        # for prefix_list in load_prefixes_to_use:
        #     kl_prefix_list = []
        #     for prefix in prefix_list:
        #         # Extract the part after "info_eval_" or similar pattern
        #         # Adjust this pattern based on your actual file naming convention
        #         if "info_eval_" in prefix:
        #             # Replace "info_eval_" with "analytic_kls_toxicity_"
        #             kl_prefix = prefix.replace("info_eval_", "analytic_kls_toxicity_")
        #         elif prefix.startswith("analytic_kls_toxicity_"):
        #             # Already a KL file prefix
        #             kl_prefix = prefix
        #         else:
        #             # Try to construct it - adjust based on your naming pattern
        #             kl_prefix = f"analytic_kls_toxicity_{prefix}"
        #         kl_prefix_list.append(kl_prefix)
        #     kl_load_prefixes_to_use.append(kl_prefix_list)

        # Ensure we have the same number of KL prefixes as labels
        if len(kl_load_prefixes_to_use) != len(labels):
            print(
                f"Warning: Number of KL prefixes ({len(kl_load_prefixes_to_use)}) doesn't match number of labels ({len(labels)}).")
            print("Using first min(len(kl_load_prefixes_to_use), len(labels)) entries.")
            n_use = min(len(kl_load_prefixes_to_use), len(labels))
            kl_load_prefixes_to_use = kl_load_prefixes_to_use[:n_use]
            kl_labels = labels[:n_use]
            if use_semantic_styling:
                kl_color_list, kl_marker_list, _ = generate_visual_style_from_prefixes(kl_load_prefixes_to_use)
            else:
                kl_color_list = color_list[:n_use]
                kl_marker_list = marker_list[:n_use]
        else:
            kl_labels = labels
            if use_semantic_styling:
                kl_color_list, kl_marker_list, _ = generate_visual_style_from_prefixes(kl_load_prefixes_to_use)
            else:
                kl_color_list = color_list
                kl_marker_list = marker_list

        # Load KL divergence results
        kl_results_list = [[] for i in range(len(kl_load_prefixes_to_use))]
        do_load_prefixes(kl_results_list, kl_load_prefixes_to_use, map_location='cpu', load_dir=load_dir)

        if "approx" in figname_modifier:

            make_frontier_approx_kl_bootstrap(
                xlabel="KL(sigma_p || q)",
                ylabel="KL(q || sigma_p)",
                figname=f"{figname_modifier}_frontier_kl",
                load_dir=load_dir,
                f_q_load_prefixes_to_use=kl_load_prefixes_to_use,
                labels=kl_labels,
                color_list=kl_color_list,
                marker_list=kl_marker_list,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                aggregate_seeds=True,
                compare_to_reference=compare_to_reference,

            )

        else:

            # Plot KL divergence on two axes
            make_frontier_exact_kl_bootstrap(
                xlabel="KL(sigma_p || q)",
                ylabel="KL(q || sigma_p)",
                figname=f"{figname_modifier}_frontier_kl",
                labels=kl_labels,
                results_list=kl_results_list,
                color_list=kl_color_list,
                marker_list=kl_marker_list,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                aggregate_seeds=True,
                compare_to_reference=compare_to_reference,
            )

            # Plot bar chart of top tokens (avg over time + final step)
            for final_only, suffix in [(False, "_top_tokens_bar_avg"), (True, "_top_tokens_bar_final")]:
                plot_top_tokens_bar_chart(
                    figname=f"{figname_modifier}{suffix}",
                    labels=kl_labels,
                    results_list=kl_results_list,
                    color_list=kl_color_list,
                    fontsize=fontsize,
                    legendfontsize=legendfontsize,
                    n_bootstrap_draws=5000,
                    n_top_tokens=10,
                    final_only=final_only,
                )

            # Lollipop chart of absolute log probabilities (avg over time + final step w/ individual seeds)
            for final_only, suffix in [(False, "_top_tokens_lollipop_avg"), (True, "_top_tokens_lollipop_final")]:
                plot_top_tokens_lollipop(
                    figname=f"{figname_modifier}{suffix}",
                    labels=kl_labels,
                    results_list=kl_results_list,
                    color_list=kl_color_list,
                    fontsize=fontsize,
                    legendfontsize=legendfontsize,
                    n_bootstrap_draws=5000,
                    n_top_tokens=10,
                    final_only=final_only,
                    figname_individual=f"{figname_modifier}_top_tokens_lollipop_final_individual" if final_only else None,
                )

            # Lollipop chart for intersection of top-q tokens across settings (+ individual seeds)
            plot_top_q_intersection_lollipop(
                figname=f"{figname_modifier}_top_q_lollipop_final",
                labels=kl_labels,
                results_list=kl_results_list,
                color_list=kl_color_list,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                n_bootstrap_draws=5000,
                figname_individual=f"{figname_modifier}_top_q_lollipop_final_individual",
            )

            # Ranked lollipop chart (+ individual seeds)
            plot_top_q_ranked_lollipop(
                figname=f"{figname_modifier}_top_q_ranked_lollipop_final",
                labels=kl_labels,
                results_list=kl_results_list,
                color_list=kl_color_list,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                n_bootstrap_draws=5000,
                n_ranks=10,
                figname_individual=f"{figname_modifier}_top_q_ranked_lollipop_final_individual",
            )

            # Lollipop chart of top token log q over time
            plot_top_tokens_lollipop_over_time(
                figname=f"{figname_modifier}_top_tokens_lollipop_over_time",
                labels=kl_labels,
                results_list=kl_results_list,
                color_list=kl_color_list,
                n_frontiers=4,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                n_bootstrap_draws=5000,
                n_top_tokens=10,
            )

            # Cumulative sample counts over time for top target tokens
            plot_sample_counts_over_time(
                figname=f"{figname_modifier}_sample_counts_over_time",
                labels=kl_labels,
                results_list=kl_results_list,
                color_list=kl_color_list,
                n_frontiers=4,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                n_bootstrap_draws=5000,
                n_top_tokens=10,
            )

            # Coverage curve: fraction of top-K target tokens discovered over time
            plot_coverage_curve(
                figname=f"{figname_modifier}_coverage_curve",
                labels=kl_labels,
                results_list=kl_results_list,
                color_list=kl_color_list,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                n_bootstrap_draws=5000,
                n_top_tokens=10,
            )

            # Vocab coverage curve: fraction of all tokens discovered over time
            plot_vocab_coverage_curve(
                figname=f"{figname_modifier}_vocab_coverage_curve",
                labels=kl_labels,
                results_list=kl_results_list,
                color_list=kl_color_list,
                fontsize=fontsize,
                legendfontsize=legendfontsize,
                n_bootstrap_draws=5000,
            )

            # Visitation heatmaps: 2D grid of per-token visit counts (incremental + cumulative)
            plot_visitation_heatmaps(
                figname_prefix=f"{figname_modifier}_visitation_heatmap",
                labels=kl_labels,
                results_list=kl_results_list,
                n_frontiers=4,
                fontsize=fontsize,
            )

        raise SystemExit(0)


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
            tuple_index_gcg=1,
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
