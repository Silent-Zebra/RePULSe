"""Stage 2: plot functions. Each takes cached setting data (data.load) plus a list of run keys,
and returns a matplotlib figure. Drawing details reproduce the legacy figures
(make_frontier.make_frontier_exact_kl_bootstrap, plot_utils.plot_top_tokens_lollipop,
plot_utils.plot_vocab_coverage_curve); bootstrap CIs use a fixed seed so outputs are deterministic.
"""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

from . import data, style
from .stats import bootstrap_ci


# ---------------------------------------------------------------------------
# KL frontier at the final evaluation step
# ---------------------------------------------------------------------------

FRONTIER_TEXT = {
    "kl_q_sigma": dict(title="KL Frontier (final step)", ylabel=r"KL(q|$\sigma$)"),
    "elbo": dict(title=r"KL($\sigma$|q) vs ELBO ($= \log Z - D_{KL}(q\|\sigma)$)",
                 ylabel="ELBO (random prompts, mean)"),
}


def frontier(setting, setting_data, run_keys, legend_ncol=2):
    style.apply_rcparams()
    text = FRONTIER_TEXT[setting.y_metric]
    points, _ = data.frontier_points(setting_data, run_keys, setting.y_metric)
    fig, ax = plt.subplots(figsize=style.SIZES["panel"])
    handles, labels = [], []
    for run_index, run_key in enumerate(run_keys):
        run = setting.run(run_key)
        x_values, y_values = points[run_key]
        x_mean, x_lower, x_upper = bootstrap_ci(x_values, seed=2 * run_index)
        y_mean, y_lower, y_upper = bootstrap_ci(y_values, seed=2 * run_index + 1)
        run_color = style.color(run)
        handle = ax.scatter(x_mean, y_mean, c=[run_color], marker=style.marker(run), zorder=3)
        ax.errorbar(x_mean, y_mean, xerr=[[max(x_mean - x_lower, 0)], [max(x_upper - x_mean, 0)]],
                    yerr=[[max(y_mean - y_lower, 0)], [max(y_upper - y_mean, 0)]],
                    fmt="", ecolor=run_color, alpha=0.2, capsize=2, zorder=2)
        handles.append(handle)
        labels.append(style.label(run))
    ax.set_xlabel(r"KL($\sigma$|q)", fontsize=style.FONTSIZE)
    ax.set_ylabel(text["ylabel"], fontsize=style.FONTSIZE)
    ax.set_title(text["title"], fontsize=style.FONTSIZE + 1)
    ax.tick_params(labelsize=style.FONTSIZE)
    fig.tight_layout()
    style.legend_below(ax, handles, labels, ncol=legend_ncol, anchor_y=-0.2)
    return fig


# ---------------------------------------------------------------------------
# Lollipop: final log q on the highest-target-density tokens (toy settings)
# ---------------------------------------------------------------------------

def lollipop(setting, setting_data, run_keys, n_top=20, rows=None):
    """rows: optional list of legend rows (lists of run keys); the target/prior entries are
    prepended to the first two rows. Default: CTL runs in one row, RLOO runs in the other."""
    style.apply_rcparams()
    target_logprob, base_logprob = setting_data["target_logprob"], setting_data["base_logprob"]
    top_tokens = np.argsort(-target_logprob, kind="stable")[:n_top]
    n_runs = len(run_keys)
    dot_spacing = 0.12
    total_width = dot_spacing * (n_runs - 1)
    run_offsets = np.linspace(-total_width / 2, total_width / 2, n_runs) if n_runs > 1 else np.zeros(1)
    bar_half_width = total_width / 2 + dot_spacing * 0.6 if n_runs > 1 else 0.15
    token_positions = np.arange(n_top)

    fig, ax = plt.subplots(figsize=style.SIZES["wide"])
    for position, token in zip(token_positions, top_tokens):
        ax.plot([position - bar_half_width, position + bar_half_width], [target_logprob[token]] * 2,
                color="black", linewidth=2, zorder=3, solid_capstyle="butt",
                label=r"$\sigma$ (target)" if position == 0 else None)
    run_handles = {}
    for run_index, run_key in enumerate(run_keys):
        run = setting.run(run_key)
        q_logprob = setting_data["runs"][run_key]["q_logprob_final"][:, top_tokens].astype(float)  # (n_seeds, n_top)
        token_stats = [bootstrap_ci(q_logprob[:, token_index], seed=1000 * run_index + token_index)
                       for token_index in range(n_top)]
        mean, lower, upper = (np.array([stat[i] for stat in token_stats]) for i in range(3))
        run_handles[run_key] = ax.errorbar(token_positions + run_offsets[run_index], mean,
                                           yerr=[mean - lower, upper - mean], fmt=style.marker(run),
                                           color=style.color(run), markersize=5, capsize=3, linewidth=1, zorder=4)
    if base_logprob is not None:
        for position, token in zip(token_positions, top_tokens):
            ax.plot([position - bar_half_width, position + bar_half_width], [base_logprob[token]] * 2,
                    color="gold", linewidth=2, zorder=10, solid_capstyle="butt",
                    label=r"$p$ (base)" if position == 0 else None)
    ax.set_xlabel("Token", fontsize=style.FONTSIZE)
    ax.set_ylabel("Log Probability", fontsize=style.FONTSIZE)
    ax.set_title(f"Top {n_top} Target Tokens: Log Probability of q (final step)", fontsize=style.FONTSIZE + 1)
    ax.set_xticks(token_positions)
    ax.set_xticklabels([""] * n_top)
    ax.tick_params(axis="y", labelsize=style.FONTSIZE)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # legend laid out in rows (matplotlib fills legends column by column, so interleave)
    handle_by_label = dict(zip(*ax.get_legend_handles_labels()[::-1]))
    reference_entries = [(handle_by_label[r"$\sigma$ (target)"], r"$\sigma$ (target)")]
    if base_logprob is not None:
        reference_entries.append((handle_by_label[r"$p$ (base)"], r"$p$ (base)"))
    if rows is None:
        rows = [[run_key for run_key in run_keys if setting.run(run_key).algo == algo] for algo in ("CTL", "RLOO")]
    legend_rows = [[reference_entries[row_index]] if row_index < len(reference_entries) else []
                   for row_index in range(len(rows))]
    for row_index, row in enumerate(rows):
        legend_rows[row_index] += [(run_handles[run_key], style.label(setting.run(run_key), long=True))
                                   for run_key in row]
    ncol = max(len(legend_row) for legend_row in legend_rows)
    column_major = [legend_row[column] for column in range(ncol) for legend_row in legend_rows
                    if column < len(legend_row)]
    style.legend_below(ax, [handle for handle, _ in column_major], [label for _, label in column_major],
                       ncol=ncol, anchor_y=-0.15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Vocabulary coverage over training (toy settings)
# ---------------------------------------------------------------------------

def coverage(setting, setting_data, run_keys):
    style.apply_rcparams()
    fig, ax = plt.subplots(figsize=style.SIZES["coverage"])
    handles, labels = [], []
    for run_index, run_key in enumerate(run_keys):
        run = setting.run(run_key)
        curves = [list(curve) for curve in setting_data["runs"][run_key]["coverage"]]
        n_steps = max(len(curve) for curve in curves)
        curves = np.array([curve + [curve[-1]] * (n_steps - len(curve)) for curve in curves])  # pad with last value
        step_stats = [bootstrap_ci(curves[:, step][~np.isnan(curves[:, step])], seed=1000 * run_index + step)
                      for step in range(n_steps)]
        mean, lower, upper = (np.array([stat[i] for stat in step_stats]) for i in range(3))
        steps = np.arange(n_steps)
        (handle,) = ax.plot(steps, mean, color=style.color(run), linewidth=1.5, linestyle=style.linestyle(run))
        ax.fill_between(steps, lower, upper, color=style.color(run), alpha=0.15)
        handles.append(handle)
        labels.append(style.label(run))
    ax.set_xlabel("Evaluation Step", fontsize=style.FONTSIZE)
    ax.set_ylabel("Fraction of Vocab\nTokens Discovered", fontsize=style.FONTSIZE)
    ax.set_title("Vocab Coverage: Fraction of All Tokens Sampled by q", fontsize=style.FONTSIZE + 1)
    y_min, y_max = ax.get_ylim()
    margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.05
    ax.set_ylim(max(0, y_min - margin), y_max + margin)
    ax.tick_params(labelsize=style.FONTSIZE)
    ax.grid(alpha=0.3, linestyle="--")
    style.legend_below(ax, handles, labels, ncol=2, anchor_y=-0.22)
    fig.tight_layout()
    return fig
