"""Stage 1: load raw result files from info/ and compute per-seed series, cached per setting.

The computations here are ported from plot_results/plot_results.py and plot_results/plot_utils.py
with the same logic (function names in comments refer to the originals). The main structural
change is that log Z is estimated once per setting, from the runs with in_logz_pool=True, so that
every figure of a setting uses the same estimate.

Cached per run (lists over loaded seeds, each a 1D array over evaluation steps):
  kl_sigma_q, kl_q_sigma            (all settings)
  elbo                              (multiprompt settings with random-prompt evaluations)
  coverage                          (toy: fraction of the vocabulary sampled at least once by q)
  q_logprob_final                   (toy: (n_seeds, |V|) final log q over the vocabulary)
and per setting: target_logprob / base_logprob (toy), log Z per prompt and bound diagnostics.
"""
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .experiments import SETTINGS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_DIR = os.path.join(REPO, "info")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

TRUNCATE_DROP_THRESHOLD = 0.75  # as in plot_results.py


# ---------------------------------------------------------------------------
# Small helpers (ported)
# ---------------------------------------------------------------------------

def to_scalar(value):
    """Mean over all elements (plot_utils.to_scalar)."""
    return float(np.asarray(value).ravel().mean())


def _truncate_and_stack(arrays, context=""):
    """Stack 1D arrays: drop those shorter than TRUNCATE_DROP_THRESHOLD of the longest,
    truncate the rest to the shortest remaining (plot_results._truncate_and_stack)."""
    assert len(arrays) > 0, f"empty list ({context})"
    max_len = max(len(array) for array in arrays)
    kept = [array for array in arrays if len(array) >= max_len * TRUNCATE_DROP_THRESHOLD]
    if len(kept) < len(arrays):
        print(f"  Warning: dropped {len(arrays) - len(kept)} incomplete arrays ({context})")
    min_len = min(len(array) for array in kept)
    return np.stack([np.asarray(array[:min_len], dtype=float) for array in kept])


# ---------------------------------------------------------------------------
# Toy settings: analytic KL files
# ---------------------------------------------------------------------------

def _load_analytic_run(run):
    """Load analytic_kls_toxicity files for one run. Each file holds a tuple
    (kl_sigma_q over steps, kl_q_sigma over steps, metrics_list over steps, ...)."""
    result = {"seeds": [], "kl_sigma_q": [], "kl_q_sigma": [], "coverage": [], "q_logprob_final": [],
              "target_logprob_final": [], "base_logprob_final": []}
    for seed, filename in zip(run.seeds, run.files()):
        path = os.path.join(INFO_DIR, filename)
        if not os.path.exists(path):
            print(f"  Warning: missing {filename}")
            continue
        saved = torch.load(path, map_location="cpu", weights_only=False)
        result["seeds"].append(seed)
        result["kl_sigma_q"].append(np.asarray(saved[0], dtype=float))
        result["kl_q_sigma"].append(np.asarray(saved[1], dtype=float))
        metrics_per_step = saved[2]
        # vocab coverage over steps (plot_utils.plot_vocab_coverage_curve)
        coverage = []
        for metrics in metrics_per_step:
            counts = metrics.get("cumulative_q_sample_counts") if isinstance(metrics, dict) else None
            coverage.append(np.nan if counts is None else (counts > 0).sum().item() / len(counts))
        result["coverage"].append(np.asarray(coverage, dtype=float))
        # final-step full-vocabulary log probs (plot_utils._collect_top_token_log_probs, final_only=True)
        final_metrics = metrics_per_step[-1]
        result["q_logprob_final"].append(final_metrics["log_probs_q_full"].float().numpy())
        result["target_logprob_final"].append(final_metrics["log_probs_target_full"].float().numpy())
        if final_metrics.get("log_probs_base_full") is not None:
            result["base_logprob_final"].append(final_metrics["log_probs_base_full"].float().numpy())
        del saved, metrics_per_step
    result["q_logprob_final"] = np.stack(result["q_logprob_final"]).astype(np.float32)
    return result


def build_analytic(setting):
    runs, target_logprobs, base_logprobs = {}, [], []
    for run in setting.runs:
        print(f"  loading {setting.key}/{run.key}")
        run_data = _load_analytic_run(run)
        target_logprobs += run_data.pop("target_logprob_final")
        base_logprobs += run_data.pop("base_logprob_final")
        runs[run.key] = run_data
    # the target and prior are shared by all runs; pool as in _compute_target_means_for_tokens
    return {"setting": setting.key, "kind": setting.kind, "runs": runs,
            "target_logprob": np.mean(np.stack(target_logprobs), axis=0),
            "base_logprob": np.mean(np.stack(base_logprobs), axis=0) if base_logprobs else None}


# ---------------------------------------------------------------------------
# Larger settings: multiprompt v2 f_q/g_q files
# ---------------------------------------------------------------------------

_HELDOUT_TO_STANDARD = {  # plot_results._remap_heldout_to_standard_keys (subset used here)
    "f_q_by_prompt_heldout": "f_q_by_prompt_fixed",
    "g_q_by_prompt_heldout": "g_q_by_prompt_fixed",
    "iwae_lbs_by_prompt_heldout": "iwae_lbs_by_prompt_fixed",
    "iwae_ubs_by_prompt_heldout": "iwae_ubs_by_prompt_fixed",
    "f_q_by_prompt_random_heldout": "f_q_by_prompt_random",
    "prompt_texts_heldout": "prompt_texts_fixed",
}
_PER_STEP_KEYS = ("f_q_by_prompt_fixed", "g_q_by_prompt_fixed", "iwae_lbs_by_prompt_fixed",
                  "iwae_ubs_by_prompt_fixed", "f_q_by_prompt_random")


def _reduce(values_per_step):
    """[[value or None per prompt] per step] -> same structure with scalars (to_scalar)."""
    return [[to_scalar(value) if value is not None else None for value in step_values]
            for step_values in values_per_step]


def _load_v2_file(path, split):
    """Load one v2 file, select the evaluation split, and reduce every per-sample entry to a
    scalar (plot_results._load_and_reduce_v2_file; g_q is also reduced here since only its mean
    is used for the KL computation)."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(saved, dict) and saved.get("version", 1) >= 2):
        raise ValueError(f"{path} is not v2 format")
    if split == "heldout":
        saved = {standard_key: saved[heldout_key] for heldout_key, standard_key in _HELDOUT_TO_STANDARD.items()
                 if heldout_key in saved}
    reduced = {"prompt_texts_fixed": list(saved["prompt_texts_fixed"])}
    for key in _PER_STEP_KEYS:
        reduced[key] = _reduce(saved.get(key, []))
    return reduced


def _load_v2_run(run, split):
    """Returns (seeds that were found, list of reduced v2 dicts, one per found seed)."""
    paths = [(seed, os.path.join(INFO_DIR, filename)) for seed, filename in zip(run.seeds, run.files())]
    present = [(seed, path) for seed, path in paths if os.path.exists(path)]
    for seed, path in paths:
        if not os.path.exists(path):
            print(f"  Warning: missing {os.path.basename(path)}")
    with ThreadPoolExecutor() as executor:
        loaded = list(executor.map(lambda seed_and_path: _load_v2_file(seed_and_path[1], split), present))
    return [seed for seed, _ in present], loaded


def _seed_averaged_bounds(pool_runs):
    """For each (run, step, prompt), average the IWAE LB (and UB) over seeds; collect these per
    prompt (plot_results._collect_seed_averaged_iwae_bounds_per_prompt).
    pool_runs: list over runs of lists over seeds of reduced v2 dicts."""

    def seed_values(per_seed_data, key, step, prompt):
        return [seed_data[key][step][prompt] for seed_data in per_seed_data
                if step < len(seed_data[key]) and prompt < len(seed_data[key][step])
                and seed_data[key][step][prompt] is not None]

    lower_bounds, upper_bounds, prompt_texts = {}, {}, None
    for per_seed_data in pool_runs:
        if not per_seed_data:
            continue
        if prompt_texts is None:
            prompt_texts = per_seed_data[0]["prompt_texts_fixed"]
        n_steps = max(max(len(seed_data["iwae_lbs_by_prompt_fixed"]), len(seed_data["iwae_ubs_by_prompt_fixed"]))
                      for seed_data in per_seed_data)
        for prompt in range(len(prompt_texts)):
            for step in range(n_steps):
                lower = seed_values(per_seed_data, "iwae_lbs_by_prompt_fixed", step, prompt)
                upper = seed_values(per_seed_data, "iwae_ubs_by_prompt_fixed", step, prompt)
                if lower:
                    lower_bounds.setdefault(prompt, []).append(float(np.mean(lower)))
                if upper:
                    upper_bounds.setdefault(prompt, []).append(float(np.mean(upper)))
    return prompt_texts, lower_bounds, upper_bounds


def _prompts_with_bounds(pool_runs):
    """Texts of the prompts that have at least one IWAE bound
    (plot_results._infer_prompts_with_targets_from_v2)."""
    result, prompt_texts = set(), None
    for per_seed_data in pool_runs:
        for seed_data in per_seed_data:
            prompt_texts = prompt_texts or seed_data["prompt_texts_fixed"]
            for key in ("iwae_lbs_by_prompt_fixed", "iwae_ubs_by_prompt_fixed"):
                for step_values in seed_data[key]:
                    for prompt, value in enumerate(step_values):
                        if value is not None and prompt < len(prompt_texts):
                            result.add(prompt_texts[prompt])
    return result


def estimate_log_z(pool_runs):
    """Per-prompt log Z = midpoint of (max seed-averaged LB, min seed-averaged UB), pooled over the
    pool runs' (run, step) cells (plot_results._compute_global_per_prompt_log_Z)."""
    prompts_with_bounds = _prompts_with_bounds(pool_runs)
    prompt_texts, lower_bounds, upper_bounds = _seed_averaged_bounds(pool_runs)
    log_z, bounds = {}, {}
    for prompt, text in enumerate(prompt_texts):
        if text not in prompts_with_bounds or not lower_bounds.get(prompt) or not upper_bounds.get(prompt):
            continue
        max_lower, min_upper = max(lower_bounds[prompt]), min(upper_bounds[prompt])
        log_z[prompt] = (max_lower + min_upper) / 2.0
        bounds[text] = (max_lower, min_upper)
    return prompt_texts, log_z, bounds


def _per_prompt_kl(seed_data, log_z):
    """plot_results._compute_per_prompt_kl_over_time: KL(q|s) = logZ - f_q, KL(s|q) = g_q - logZ.
    Returns {prompt index: (KL(q|s) over steps, KL(s|q) over steps)}."""
    f_q, g_q = seed_data["f_q_by_prompt_fixed"], seed_data["g_q_by_prompt_fixed"]
    n_steps = len(f_q)
    result = {}
    for prompt, prompt_log_z in log_z.items():
        kl_q_sigma, kl_sigma_q = np.full(n_steps, np.nan), np.full(n_steps, np.nan)
        for step in range(n_steps):
            if prompt < len(f_q[step]) and f_q[step][prompt] is not None:
                kl_q_sigma[step] = prompt_log_z - f_q[step][prompt]
            if step < len(g_q) and prompt < len(g_q[step]) and g_q[step][prompt] is not None:
                kl_sigma_q[step] = g_q[step][prompt] - prompt_log_z
        result[prompt] = (kl_q_sigma, kl_sigma_q)
    return result


def _random_f_q(seed_data):
    """plot_results._compute_random_f_q_over_time: mean f_q over the random prompts per step (ELBO)."""
    values_per_step = seed_data["f_q_by_prompt_random"]
    if not values_per_step:
        return None
    return np.array([np.mean([value for value in step_values if value is not None])
                     if any(value is not None for value in step_values) else np.nan
                     for step_values in values_per_step])


def build_multiprompt(setting):
    loaded = {}
    for run in setting.runs:
        print(f"  loading {setting.key}/{run.key} ({setting.split} split)")
        loaded[run.key] = _load_v2_run(run, setting.split)
    pool_runs = [loaded[run.key][1] for run in setting.runs if run.in_logz_pool]
    prompt_texts, log_z, bounds = estimate_log_z(pool_runs)
    prompts = sorted(log_z)
    print(f"  log Z estimated for {len(prompts)} prompts from {len(pool_runs)} pool runs")
    runs = {}
    for run in setting.runs:
        seeds, per_seed_data = loaded[run.key]
        run_data = {"seeds": seeds, "kl_sigma_q": [], "kl_q_sigma": [], "elbo": []}
        for seed_data in per_seed_data:
            assert seed_data["prompt_texts_fixed"] == prompt_texts, f"prompt set differs for {run.key}"
            kl_by_prompt = _per_prompt_kl(seed_data, log_z)
            # mean over prompts per step (summary trajectories in plot_f_q_g_q_kl_divergences_multiprompt)
            for index, name in ((0, "kl_q_sigma"), (1, "kl_sigma_q")):
                stacked = _truncate_and_stack([kl_by_prompt[prompt][index] for prompt in prompts],
                                              context=f"{run.key} {name}")
                with np.errstate(all="ignore"):
                    run_data[name].append(np.nanmean(stacked, axis=0))
            run_data["elbo"].append(_random_f_q(seed_data))
        if all(elbo is None for elbo in run_data["elbo"]):
            run_data["elbo"] = None
        runs[run.key] = run_data
    gaps = np.array([upper - lower for lower, upper in bounds.values()])
    return {"setting": setting.key, "kind": setting.kind, "split": setting.split, "runs": runs,
            "log_z": {prompt_texts[prompt]: value for prompt, value in log_z.items()}, "logz_bounds": bounds,
            "logz_gap": {"mean": float(gaps.mean()), "min": float(gaps.min()), "max": float(gaps.max()),
                         "n_crossed": int((gaps < 0).sum()), "n_prompts": int(len(gaps))}}


# ---------------------------------------------------------------------------
# Cache and access
# ---------------------------------------------------------------------------

def _cache_path(setting_key):
    return os.path.join(CACHE_DIR, f"{setting_key}.pkl")


def build(setting_key):
    setting = SETTINGS[setting_key]
    print(f"Building {setting_key} ...")
    setting_data = build_analytic(setting) if setting.kind == "analytic" else build_multiprompt(setting)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(setting_key), "wb") as file:
        pickle.dump(setting_data, file)
    print(f"  cached to {_cache_path(setting_key)}")
    return setting_data


def load(setting_key, rebuild=False):
    if rebuild or not os.path.exists(_cache_path(setting_key)):
        return build(setting_key)
    with open(_cache_path(setting_key), "rb") as file:
        return pickle.load(file)


def series_pair(setting_data, run_key, y_metric):
    """Per-seed (x = KL(sigma|q), y) series, truncated to a common length when y is the ELBO
    (as in the ELBO frontier construction of plot_f_q_g_q_kl_divergences_multiprompt)."""
    run_data = setting_data["runs"][run_key]
    if y_metric == "elbo":
        pairs = []
        for x_series, y_series in zip(run_data["kl_sigma_q"], run_data["elbo"]):
            length = min(len(x_series), len(y_series))
            pairs.append((x_series[:length], y_series[:length]))
        return pairs
    return list(zip(run_data["kl_sigma_q"], run_data[y_metric]))


def frontier_points(setting_data, run_keys, y_metric, step=None):
    """Per-seed (x, y) at one evaluation step, for each run (plot_results._generate_kl_frontier_plots).
    step=None means the final step: n_steps - 1, with n_steps the longest series over the given runs
    and seeds; seeds that did not reach that step are skipped. Returns ({run key: (xs, ys)}, step)."""
    pairs_by_run = {run_key: series_pair(setting_data, run_key, y_metric) for run_key in run_keys}
    if step is None:
        n_steps = max(max(len(x_series), len(y_series))
                      for pairs in pairs_by_run.values() for x_series, y_series in pairs)
        step = n_steps - 1
    points = {}
    for run_key, pairs in pairs_by_run.items():
        reached = [(x_series, y_series) for x_series, y_series in pairs
                   if step < len(x_series) and step < len(y_series)]
        points[run_key] = (np.array([x_series[step] for x_series, _ in reached]),
                           np.array([y_series[step] for _, y_series in reached]))
    return points, step
