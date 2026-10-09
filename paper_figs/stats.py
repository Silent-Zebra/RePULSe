"""Statistics: bootstrap confidence intervals and significance tests across seeds."""
import csv
import itertools
import os

import numpy as np


def bootstrap_ci(values, n_draws=5000, alpha=0.05, seed=0):
    """Mean and percentile-bootstrap CI of the mean over seeds (as in the legacy plots, but with a
    fixed seed so figures are reproducible). Returns (mean, lower, upper)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = values.mean()
    if len(values) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    resampled_means = values[rng.integers(0, len(values), (n_draws, len(values)))].mean(axis=1)
    return (mean, np.percentile(resampled_means, 100 * alpha / 2),
            np.percentile(resampled_means, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Significance tests for the claims made in the paper
# ---------------------------------------------------------------------------

def compare(reference, compared, n_boot=100000, seed=0):
    """Two-sided tests for a difference in means between two independent groups of seeds:
    Welch's t-test, a bootstrap of the difference (each group resampled independently),
    and a permutation test (exact enumeration when feasible, else Monte Carlo)."""
    from scipy import stats as scipy_stats
    reference, compared = np.asarray(reference, float), np.asarray(compared, float)
    n_ref, n_cmp = len(reference), len(compared)
    rng = np.random.default_rng(seed)
    diff = reference.mean() - compared.mean()

    welch_p = scipy_stats.ttest_ind(reference, compared, equal_var=False).pvalue

    boot_diffs = (reference[rng.integers(0, n_ref, (n_boot, n_ref))].mean(1)
                  - compared[rng.integers(0, n_cmp, (n_boot, n_cmp))].mean(1))
    boot_p = min(1.0, 2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean()))

    pooled = np.concatenate([reference, compared])
    assignments = list(itertools.combinations(range(len(pooled)), n_ref))  # indices assigned to "reference"
    if len(assignments) <= 200000:
        in_reference = np.zeros((len(assignments), len(pooled)), bool)
        in_reference[np.arange(len(assignments))[:, None], np.array(assignments)] = True
        perm_diffs = (pooled * in_reference).sum(1) / n_ref - (pooled * ~in_reference).sum(1) / n_cmp
        perm_kind = "exact"
    else:
        perm_diffs = []
        for _ in range(n_boot):
            shuffled = rng.permutation(pooled)
            perm_diffs.append(shuffled[:n_ref].mean() - shuffled[n_ref:].mean())
        perm_diffs = np.array(perm_diffs)
        perm_kind = "MC"
    perm_p = (np.abs(perm_diffs) >= abs(diff) - 1e-12).mean()

    return dict(diff=diff, welch_p=welch_p, boot_p=boot_p, boot_ci=tuple(np.percentile(boot_diffs, [2.5, 97.5])),
                perm_p=perm_p, perm_kind=perm_kind)


# Claims: (description, figure path in figures.FIGURES, axis, reference run, compared run).
# axis "x" = KL(sigma|q); "y" = the figure's y-axis (KL(q|sigma), or the ELBO where higher is better).
# "diff" = mean(reference) - mean(compared): positive means the compared run has lower KL / lower ELBO.
CLAIMS = [
    ("Fig 4a: CTL tempering vs CTL, KL(s|q)", "main/toy_mc_frontier.pdf", "x", "ctl", "ctl_temp"),
    ("Fig 4a: CTL tempering vs CTL, KL(q|s)", "main/toy_mc_frontier.pdf", "y", "ctl", "ctl_temp"),
    ("Fig 4a: RLOO tempering vs RLOO, KL(s|q)", "main/toy_mc_frontier.pdf", "x", "rloo", "rloo_temp"),
    ("Fig 4a: RLOO tempering vs RLOO, KL(q|s)", "main/toy_mc_frontier.pdf", "y", "rloo", "rloo_temp"),
    ("Fig 4b: CTL tempering vs CTL, KL(s|q)", "main/small_mc_frontier.pdf", "x", "ctl", "ctl_temp"),
    ("Fig 4b: RLOO tempering vs RLOO, KL(s|q)", "main/small_mc_frontier.pdf", "x", "rloo", "rloo_temp"),
    ("Fig 4c: CTL tempering vs CTL, KL(s|q)", "main/medium_mc_frontier.pdf", "x", "ctl", "ctl_temp"),
    ("Fig 4c: RLOO tempering vs RLOO, KL(s|q)", "main/medium_mc_frontier.pdf", "x", "rloo", "rloo_temp"),
    ("Fig 5a: CTL exploration vs CTL, KL(s|q)", "main/toy_htf_frontier.pdf", "x", "ctl", "ctl_expl"),
    ("Fig 5a: RLOO exploration vs RLOO, KL(s|q)", "main/toy_htf_frontier.pdf", "x", "rloo", "rloo_expl"),
    ("Fig 5b: CTL exploration vs CTL, KL(s|q)", "main/small_htf_frontier.pdf", "x", "ctl", "ctl_expl"),
    ("Fig 5b: RLOO exploration vs RLOO, KL(s|q)", "main/small_htf_frontier.pdf", "x", "rloo", "rloo_expl"),
    ("Fig 5b: CTL exploration vs CTL, KL(q|s)", "main/small_htf_frontier.pdf", "y", "ctl", "ctl_expl"),
    ("Fig 5b: CTL tempering vs exploration, KL(s|q)", "main/small_htf_frontier.pdf", "x", "ctl_temp", "ctl_expl"),
    ("Fig 5b: RLOO tempering vs exploration, KL(s|q)", "main/small_htf_frontier.pdf", "x", "rloo_temp", "rloo_expl"),
    ("Fig 5c: CTL exploration vs CTL, KL(s|q)", "main/medium_htf_frontier.pdf", "x", "ctl", "ctl_expl"),
    ("Fig 5c: RLOO exploration vs RLOO, KL(s|q)", "main/medium_htf_frontier.pdf", "x", "rloo", "rloo_expl"),
    ("Fig 5c: CTL tempering vs CTL, KL(s|q)", "main/medium_htf_frontier.pdf", "x", "ctl", "ctl_temp"),
    ("Fig 5c: RLOO tempering vs RLOO, KL(s|q)", "main/medium_htf_frontier.pdf", "x", "rloo", "rloo_temp"),
    ("Fig 6: entropy vs CTL, KL(s|q)", "appendix/small_mc_extra.pdf", "x", "ctl", "ctl_ent"),
    ("Fig 6: entropy vs CTL, ELBO", "appendix/small_mc_extra.pdf", "y", "ctl", "ctl_ent"),
    ("Fig 6: mixture vs CTL, KL(s|q)", "appendix/small_mc_extra.pdf", "x", "ctl", "ctl_mix"),
    ("Fig 6: mixture vs CTL, ELBO", "appendix/small_mc_extra.pdf", "y", "ctl", "ctl_mix"),
    ("Fig 6: tempering+exploration vs tempering, KL(s|q)", "appendix/small_mc_extra.pdf", "x", "ctl_temp", "ctl_temp_expl"),
    ("Fig 6: tempering+exploration vs tempering, ELBO", "appendix/small_mc_extra.pdf", "y", "ctl_temp", "ctl_temp_expl"),
    ("Fig 7: tempering+exploration vs exploration, KL(s|q)", "appendix/medium_htf_extra.pdf", "x", "ctl_expl", "ctl_temp_expl"),
    ("Fig 7: tempering+exploration vs exploration, KL(q|s)", "appendix/medium_htf_extra.pdf", "y", "ctl_expl", "ctl_temp_expl"),
    ("Fig 7: DPG exploration vs DPG, KL(s|q)", "appendix/medium_htf_extra.pdf", "x", "dpg", "dpg_expl"),
    ("Fig 7: DPG exploration vs DPG, KL(q|s)", "appendix/medium_htf_extra.pdf", "y", "dpg", "dpg_expl"),
    ("Fig 7: DPG tempering vs DPG, KL(s|q)", "appendix/medium_htf_extra.pdf", "x", "dpg", "dpg_temp"),
    ("Fig 7: DPG tempering vs DPG, KL(q|s)", "appendix/medium_htf_extra.pdf", "y", "dpg", "dpg_temp"),
    ("Fig 7: CTL vs DPG, KL(s|q)", "appendix/medium_htf_extra.pdf", "x", "dpg", "ctl"),
    ("Fig 7: CTL vs DPG, KL(q|s)", "appendix/medium_htf_extra.pdf", "y", "dpg", "ctl"),
    ("Fig 7: CTL tempering vs DPG tempering, KL(s|q)", "appendix/medium_htf_extra.pdf", "x", "dpg_temp", "ctl_temp"),
    ("Fig 7: CTL tempering vs DPG tempering, KL(q|s)", "appendix/medium_htf_extra.pdf", "y", "dpg_temp", "ctl_temp"),
    ("Fig 7: CTL exploration vs DPG exploration, KL(s|q)", "appendix/medium_htf_extra.pdf", "x", "dpg_expl", "ctl_expl"),
    ("Fig 7: CTL exploration vs DPG exploration, KL(q|s)", "appendix/medium_htf_extra.pdf", "y", "dpg_expl", "ctl_expl"),
    ("Fig 7: CTL(U) vs CTL, KL(s|q)", "appendix/medium_htf_extra.pdf", "x", "ctl", "ctl_u"),
    ("Fig 7: CTL(U) vs CTL, KL(q|s)", "appendix/medium_htf_extra.pdf", "y", "ctl", "ctl_u"),
    ("Fig 8: entropy vs CTL, KL(s|q)", "appendix/small_htf_extra.pdf", "x", "ctl", "ctl_ent"),
    ("Fig 8: entropy vs CTL, KL(q|s)", "appendix/small_htf_extra.pdf", "y", "ctl", "ctl_ent"),
    ("Fig 8: entropy vs exploration, KL(s|q)", "appendix/small_htf_extra.pdf", "x", "ctl_expl", "ctl_ent"),
    ("Fig 8: entropy vs exploration, KL(q|s)", "appendix/small_htf_extra.pdf", "y", "ctl_expl", "ctl_ent"),
    ("Fig 8: mixture vs CTL, KL(s|q)", "appendix/small_htf_extra.pdf", "x", "ctl", "ctl_mix"),
    ("Fig 8: mixture vs CTL, KL(q|s)", "appendix/small_htf_extra.pdf", "y", "ctl", "ctl_mix"),
]


def _format_p(p_value):
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def significance_table(out_dir):
    """Run every claim's tests on the final-step values shown in its figure; write
    significance_results.md and .csv to out_dir."""
    from . import data
    from .experiments import SETTINGS
    from .figures import FIGURES
    figures_by_path = {figure.path: figure for figure in FIGURES}
    setting_data_cache, rows = {}, []
    for claim_index, (description, figure_path, axis, reference_run, compared_run) in enumerate(CLAIMS):
        figure = figures_by_path[figure_path]
        setting = SETTINGS[figure.setting]
        setting_data = setting_data_cache.setdefault(figure.setting, data.load(figure.setting))
        points, _ = data.frontier_points(setting_data, list(figure.runs), setting.y_metric)
        axis_index = 0 if axis == "x" else 1
        reference_values, compared_values = points[reference_run][axis_index], points[compared_run][axis_index]
        result = compare(reference_values, compared_values, seed=claim_index)
        rows.append(dict(claim=description, n=f"{len(reference_values)}/{len(compared_values)}",
                         diff=result["diff"], welch_p=result["welch_p"], boot_p=result["boot_p"],
                         boot_ci_lo=result["boot_ci"][0], boot_ci_hi=result["boot_ci"][1],
                         perm_p=result["perm_p"], perm_kind=result["perm_kind"]))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "significance_results.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Two-sided tests at the final evaluation step, across seeds. diff = mean(reference) - mean(compared).",
             "", "| Claim | n | diff | Welch p | Bootstrap p | Bootstrap 95% CI of diff | Permutation p |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['claim']} | {row['n']} | {row['diff']:.3f} | {_format_p(row['welch_p'])} | "
                     f"{_format_p(row['boot_p'])} | [{row['boot_ci_lo']:.3f}, {row['boot_ci_hi']:.3f}] | "
                     f"{_format_p(row['perm_p'])} ({row['perm_kind']}) |")
    with open(os.path.join(out_dir, "significance_results.md"), "w") as file:
        file.write("\n".join(lines) + "\n")
    return rows
