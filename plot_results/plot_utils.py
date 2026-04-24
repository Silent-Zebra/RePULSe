import torch
import re
from collections import Counter
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# Marker constants for bonus types (legacy, kept for backward compatibility)
MARKER_NO_BONUS = "x"
MARKER_CFN = "P"  # filled plus — used for CTL + CFN (non-annealed)
MARKER_CFN_REINF = "1"  # tri_down (unfilled) — used for REINF + CFN (non-annealed)
MARKER_MIXTURE = "^"  # triangle up — used for CTL + mixture (non-annealed)
MARKER_MIXTURE_REINF = "v"  # triangle down — used for REINF + mixture (non-annealed)
MARKER_EXACT_COUNT = "D"
MARKER_ENTROPY = "d"  # diamond (thin) — used for CTL + entropy
MARKER_ENTROPY_REINF = "H"  # hexagon — used for REINF + entropy
MARKER_ENTROPY_ANNEALED = "p"  # pentagon — used for CTL + annealed entropy
MARKER_ENTROPY_ANNEALED_REINF = "8"  # octagon — used for REINF + annealed entropy
MARKER_CFN_ANNEALED = "X"  # filled X — used for CTL + annealed CFN (changed from "^" which now denotes mixture)
MARKER_CFN_ANNEALED_REINF = ">"  # right triangle — used for REINF + annealed CFN (changed from "v" which now denotes REINF+mixture)
MARKER_BETA_ANNEALED = "h"  # hexagon2 — used for CTL + annealed beta
MARKER_BETA_ANNEALED_REINF = "<"  # left-pointing triangle — used for REINF + annealed beta

# Marker constants for loss types (used by semantic styling)
MARKER_CTL = "o"
MARKER_CTLN = "x"
MARKER_REINF = "s"  # square
MARKER_LOSS_UNKNOWN = "D"
MARKER_EXACT_COUNT = "*"

# Markers cycled across seeds in individual-seed plots
SEED_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


# Global final-plots flag. When enabled via set_final_plots(True):
#   - figure heights shrink by 25% (applied via _scale_h inside every figsize= call)
#   - token-name x-tick labels on top-token plots are hidden (via _final_plots_enabled)
_FINAL_PLOTS = False


def set_final_plots(enabled):
    """Toggle the final-plots mode. When enabled, all figure heights created
    via plt.subplots/plt.figure in plot_utils are scaled by 0.75, matplotlib's
    default figsize height is likewise reduced so callers without an explicit
    figsize also shrink, and per-token x-tick labels on top-token plots are
    suppressed for a cleaner final look."""
    global _FINAL_PLOTS
    _FINAL_PLOTS = bool(enabled)
    # Scale matplotlib's default so plt.subplots() calls without explicit
    # figsize also pick up the reduced height.
    default_w, default_h = 6.4, 4.8  # matplotlib defaults
    height_factor = 0.75 if _FINAL_PLOTS else 1.0
    plt.rcParams['figure.figsize'] = [default_w, default_h * height_factor]


def _scale_h(h):
    """Scale a figure height by the current final-plots factor (0.75 when enabled)."""
    return h * (0.75 if _FINAL_PLOTS else 1.0)


def _final_plots_enabled():
    """Return True if final-plots mode is active."""
    return _FINAL_PLOTS


def _maybe_hide_token_labels(tick_labels):
    """Return the passed x-tick labels unchanged, or a list of empty strings
    when final-plots mode is on (so tick marks remain but token names are
    suppressed for a less cluttered final-version x-axis)."""
    if _FINAL_PLOTS:
        return ['' for _ in tick_labels]
    return tick_labels


def _token_label(token_id, tokenizer=None):
    """Format a token ID as a display label for plot axes."""
    if tokenizer is None:
        return str(token_id)
    decoded = tokenizer.decode([token_id])
    if not decoded.strip():
        return repr(decoded)
    return decoded


def make_list(name, first_seed, last_seed):
    add_back_actor = ""
    if name[-12:] == "_harml_actor":
        add_back_actor = "_harml_actor"
        name = name[:-12]
    elif name[-6:] == "_actor":
        add_back_actor = "_actor"
        name = name[:-6]
    if name[-1] != "s":
        name = name[:-1]
    return [
        f"{name}{i}{add_back_actor}"
        for i in range(first_seed, last_seed + 1)
    ]


def do_load_prefixes(results_list, load_prefixes_to_use, load_dir="./info", map_location=None):

    for i in range(len(load_prefixes_to_use)):

        load_prefixes = load_prefixes_to_use[i]

        for load_prefix in load_prefixes:
            # print(load_prefix)
            try:
                if map_location is None:
                    x = torch.load(f'{load_dir}/{load_prefix}')
                else:
                    x = torch.load(f'{load_dir}/{load_prefix}', map_location=map_location)
                results_list[i].append(x)
            except Exception as e:
                print(f"Warning: Failed to load {load_prefix}")
                print(e)


def _parse_experiment_properties(prefix):
    """
    Extract semantic properties from a prefix string for visual styling.

    Every detector fires independently, so combined-setting runs (e.g. CFN + entropy,
    entropy + annealed beta) carry data for every component. generate_visual_style_from_prefixes
    uses this to blend colors from all detected modifications.

    Returns a dict with:
        loss_type: "CTL", "CTLN", "RLOO", or None
        modifications: ordered list of detected modifications. Detection priority
            (preserved from the legacy first-match chain): cfn, exact_count, mixture,
            entropy, beta_annealed.
        bonus_type: legacy field = first non-beta modification, or "none". Kept so
            downstream code that branches on bonus_type (marker selection, label
            generation) continues to work.
        mixture_variant: "mixture", "q_independent", "q_half", or None (populated
            whenever mixture is present, even as a secondary modification)
        mixture_other_model: "best", "first", "lag", or None (populated whenever
            mixture is present)
        mixture_lag_steps: int or None (populated when mixture_other_model == "lag")
        learning_rate: float or None (sampling actor LR from _al pattern)
        cfn_alpha: float or None (bonus_alpha from _cf<value>; start value if
            annealed; populated whenever cfn is present)
        cfn_annealed: bool (populated whenever cfn is present)
        entropy_alpha: float or None (start value if annealed; populated whenever
            entropy is present)
        entropy_annealed: bool (populated whenever entropy is present)
        beta_annealed: bool
        beta_anneal_start: float or None
    """
    # Loss type
    if "_ctln_" in prefix:
        loss_type = "CTLN"
    elif "_ctl_" in prefix:
        loss_type = "CTL"
    elif "_reinf_" in prefix:
        loss_type = "RLOO"
    else:
        loss_type = None

    # Independent modification detection.
    # The exact_count fuzzy regex `_c<digits>` is suppressed when cfn is present:
    # a cfn prefix can contain `_c<digits>` positions that aren't genuine
    # exact_count tokens (e.g. the `_c10` in a `_ctl_..._cf10` run has `f` not a
    # digit after `_c`, so only explicit `_count` substrings were the real signal
    # under the legacy if/elif chain).
    has_cfn = ("cfn" in prefix) or bool(re.search(r'_cf([\d.]+)', prefix))
    has_exact_count_explicit = "_count" in prefix
    has_exact_count_fuzzy = bool(re.search(r'_c([\d.]+)(?![a-z])', prefix))
    has_exact_count = has_exact_count_explicit or (has_exact_count_fuzzy and not has_cfn)
    has_mixture = "_mix" in prefix
    has_entropy = bool(re.search(r'_entb([\d.e-]+)', prefix))

    # CFN alpha / annealing — populated whenever CFN is detected
    cfn_alpha = None
    cfn_annealed = False
    if has_cfn:
        cf_anneal_match = re.search(r'_cf([\d.]+(?:e[+-]?\d+)?)to([\d.]+(?:e[+-]?\d+)?)', prefix)
        if cf_anneal_match:
            cfn_alpha = float(cf_anneal_match.group(1))  # start value
            cfn_annealed = True
        else:
            cf_match = re.search(r'_cf([\d.]+)', prefix)
            if cf_match:
                cfn_alpha = float(cf_match.group(1))

    # Entropy alpha / annealing — populated whenever entropy is detected
    entropy_alpha = None
    entropy_annealed = False
    if has_entropy:
        entb_anneal_match = re.search(r'_entb([\d.e-]+)to([\d.e-]+)', prefix)
        if entb_anneal_match:
            entropy_alpha = float(entb_anneal_match.group(1))  # start value
            entropy_annealed = True
        else:
            entb_match = re.search(r'_entb([\d.e-]+)', prefix)
            if entb_match:
                entropy_alpha = float(entb_match.group(1))

    # Mixture variant / strategy — populated whenever mixture is detected
    mixture_variant = None
    mixture_other_model = None
    mixture_lag_steps = None
    if has_mixture:
        mix_variant_map = {"mx": "mixture", "qi": "q_independent", "qh": "q_half"}
        mix_match = re.search(r'_mix([a-z]+)', prefix)
        if mix_match:
            mixture_variant = mix_variant_map.get(mix_match.group(1), mix_match.group(1))
        else:
            mixture_variant = "unknown"

        # Other-model strategy: _first or _lag<N> (absent means "best")
        if "_first" in prefix:
            mixture_other_model = "first"
        else:
            lag_match = re.search(r'_lag(\d+)', prefix)
            if lag_match:
                mixture_other_model = "lag"
                mixture_lag_steps = int(lag_match.group(1))
            else:
                mixture_other_model = "best"

    # Beta annealing: _s<start>_b<end> pattern (start_target_dist_beta -> target_dist_beta)
    beta_annealed = False
    beta_anneal_start = None
    beta_anneal_match = re.search(r'_s(-?[\d.]+)_b(-?[\d.]+)', prefix)
    if beta_anneal_match:
        beta_annealed = True
        beta_anneal_start = float(beta_anneal_match.group(1))

    # Ordered list of modifications. Priority (matches the legacy first-match chain):
    # cfn > exact_count > mixture > entropy > beta_annealed. The first non-beta entry
    # is what the legacy bonus_type field tracks.
    modifications = []
    if has_cfn:
        modifications.append("cfn")
    if has_exact_count:
        modifications.append("exact_count")
    if has_mixture:
        modifications.append("mixture")
    if has_entropy:
        modifications.append("entropy")
    if beta_annealed:
        modifications.append("beta_annealed")

    non_beta_mods = [m for m in modifications if m != "beta_annealed"]
    bonus_type = non_beta_mods[0] if non_beta_mods else "none"

    # Sanity: legacy bonus_type must agree with the ordered modifications list, or
    # the refactor has drifted.
    if non_beta_mods:
        assert bonus_type == non_beta_mods[0], \
            f"bonus_type ({bonus_type}) inconsistent with modifications ({modifications})"
    else:
        assert bonus_type == "none", \
            f"bonus_type ({bonus_type}) should be 'none' when no non-beta modifications (modifications={modifications})"

    # Learning rate (sampling actor LR)
    al_match = re.search(r'_al([\d.e-]+)', prefix)
    learning_rate = float(al_match.group(1)) if al_match else None

    return {
        "loss_type": loss_type,
        "bonus_type": bonus_type,
        "modifications": modifications,
        "mixture_variant": mixture_variant,
        "mixture_other_model": mixture_other_model,
        "mixture_lag_steps": mixture_lag_steps,
        "learning_rate": learning_rate,
        "cfn_alpha": cfn_alpha,
        "cfn_annealed": cfn_annealed,
        "entropy_alpha": entropy_alpha,
        "entropy_annealed": entropy_annealed,
        "beta_annealed": beta_annealed,
        "beta_anneal_start": beta_anneal_start,
    }


def generate_visual_style_from_prefixes(load_prefixes_to_use):
    """
    Generate semantically consistent (color, marker, linestyle) lists from prefix lists.

    Visual encoding:
        - Color hue: per-modification colormap
            - No bonus (baselines): Greys
            - CFN: Blues colormap, shade encodes alpha value (lighter=smaller, darker=larger)
            - Entropy bonus: Reds colormap, shade encodes alpha value
            - Beta-annealed: Greens colormap, shade encodes start-beta value
            - Mixture (q_independent): Purples; other mixture variants: RdPu
            - Exact count: cm.YlGnBu, shade encodes LR rank
        - Combined-setting runs: the final color is the RGBA average of every
          detected modification's color, so a run that is (e.g.) CFN + entropy
          lands visually between the Blues and Reds components. Single-mod runs
          are unchanged (mean of a length-1 list is the value itself).
        - Marker shape: selected from the legacy priority tree
          (beta_annealed > exact_count > entropy_annealed > entropy > cfn_annealed
          > cfn > mixture > loss_type fallback).
        - Line style: cycles within same-hue group; the hue key includes every
          detected modification so combined-setting runs form their own group.

    Args:
        load_prefixes_to_use: List of lists of prefixes (one inner list per experiment)

    Returns:
        (color_list, marker_list, linestyle_list) — parallel lists, one entry per experiment.
        color_list entries are RGBA tuples (floats in [0, 1], alpha forced to 1.0).
    """
    # 1. Parse all experiments
    props_list = []
    for prefix_list in load_prefixes_to_use:
        first_prefix = prefix_list[0] if prefix_list else ""
        props_list.append(_parse_experiment_properties(first_prefix))

    # 2. Collect unique LRs (excluding None), sorted ascending
    unique_lrs = sorted(set(p["learning_rate"] for p in props_list if p["learning_rate"] is not None))
    if len(unique_lrs) <= 1:
        lr_position = {lr: 0.5 for lr in unique_lrs}
    else:
        lr_position = {lr: i / (len(unique_lrs) - 1) for i, lr in enumerate(unique_lrs)}

    # 3. Linestyle options for cycling within same-hue groups
    linestyle_options = ["solid", "dashed", "dotted", "dashdot", (5, (10, 3)), (0, (3, 5, 1, 5)), (0, (1, 1))]

    # 4a. CFN alpha -> shade within a single colormap (Blues).
    #     Shade encodes alpha rank: lighter = smaller alpha, darker = larger alpha.
    #     LR differentiation comes from linestyle (handled by hue_group_counter below).
    CFN_CMAP = cm.Blues
    CFN_SHADE_LO, CFN_SHADE_HI = 0.30, 0.90
    unique_alphas = sorted(set(p["cfn_alpha"] for p in props_list if p["cfn_alpha"] is not None))
    if len(unique_alphas) <= 1:
        cfn_alpha_shade = {a: (CFN_SHADE_LO + CFN_SHADE_HI) / 2 for a in unique_alphas}
    else:
        cfn_alpha_shade = {a: CFN_SHADE_LO + (CFN_SHADE_HI - CFN_SHADE_LO) * i / (len(unique_alphas) - 1)
                           for i, a in enumerate(unique_alphas)}

    # 4a2. Entropy alpha -> shade within a single colormap (Reds).
    #      Mirrors the CFN scheme: shade encodes alpha rank (lighter = smaller alpha,
    #      darker = larger alpha). Keeping entropy on a single hue (red) makes it
    #      visually identifiable as a single "category" across runs, cleanly distinct
    #      from CFN (blue), mixture qi (purple), beta-annealed (green) and base (grey).
    ENTROPY_CMAP = cm.Reds
    ENTROPY_SHADE_LO, ENTROPY_SHADE_HI = 0.30, 0.90
    unique_entropy_alphas = sorted(set(p["entropy_alpha"] for p in props_list if p["entropy_alpha"] is not None))
    if len(unique_entropy_alphas) <= 1:
        entropy_alpha_shade = {a: (ENTROPY_SHADE_LO + ENTROPY_SHADE_HI) / 2 for a in unique_entropy_alphas}
    else:
        entropy_alpha_shade = {
            a: ENTROPY_SHADE_LO + (ENTROPY_SHADE_HI - ENTROPY_SHADE_LO) * i / (len(unique_entropy_alphas) - 1)
            for i, a in enumerate(unique_entropy_alphas)
        }

    # 4a3. Beta annealing: Greens colormap, shade encodes start beta value.
    BETA_ANNEAL_CMAP = cm.Greens
    BETA_ANNEAL_SHADE_LO, BETA_ANNEAL_SHADE_HI = 0.30, 0.90
    unique_beta_starts = sorted(set(
        p["beta_anneal_start"] for p in props_list if p["beta_annealed"] and p["beta_anneal_start"] is not None
    ))
    if len(unique_beta_starts) <= 1:
        beta_start_shade = {s: (BETA_ANNEAL_SHADE_LO + BETA_ANNEAL_SHADE_HI) / 2 for s in unique_beta_starts}
    else:
        beta_start_shade = {
            s: BETA_ANNEAL_SHADE_LO + (BETA_ANNEAL_SHADE_HI - BETA_ANNEAL_SHADE_LO) * i / (len(unique_beta_starts) - 1)
            for i, s in enumerate(unique_beta_starts)
        }

    # 4b. Mixture lag_steps -> color: assign a fixed shade per (variant, lag_steps) key spread
    #     across 0.25-0.9 so different lag values are clearly distinguishable.
    #     qi variants use a single orange-red colormap; other variants use a purple-pink colormap.
    #     Shade is spread by rank among all mixture keys, NOT by LR (LR uses linestyle instead).
    unique_mixture_keys = sorted(
        set((p["mixture_variant"], p["mixture_lag_steps"])
            for p in props_list if "mixture" in p["modifications"]),
        key=lambda x: (x[0] or "", x[1] if x[1] is not None else -1),
    )
    MIXTURE_SHADE_LO, MIXTURE_SHADE_HI = 0.55, 0.90
    n_mix = len(unique_mixture_keys)
    mixture_shade_map = {}
    mixture_cmap_map = {}
    for i, (variant, lag) in enumerate(unique_mixture_keys):
        # Single mixture key uses the midpoint (mirrors CFN/entropy single-alpha behavior)
        # so the color reads as a solid, saturated purple instead of a very light tint.
        if n_mix == 1:
            shade = (MIXTURE_SHADE_LO + MIXTURE_SHADE_HI) / 2
        else:
            shade = MIXTURE_SHADE_LO + (MIXTURE_SHADE_HI - MIXTURE_SHADE_LO) * i / (n_mix - 1)
        mixture_shade_map[(variant, lag)] = shade
        # q_independent (the primary mixture variant shown in the main frontier comparison)
        # gets Purples — maximally distinct from CFN's Blues, base's Greys, beta's Greens
        # and entropy's Reds. Other mixture variants use RdPu (red-purple) to stay in the
        # same hue family while remaining distinguishable from qi.
        if variant == "q_independent":
            mixture_cmap_map[(variant, lag)] = cm.Purples
        else:
            mixture_cmap_map[(variant, lag)] = cm.RdPu

    # 5. Colormap and shade range per bonus category
    #    Shade range [lo, hi] samples the colormap avoiding very light/very dark ends.
    shade_range_default = (0.4, 0.90)

    # Loss type -> marker
    loss_marker = {
        "CTL": MARKER_CTL,
        "CTLN": MARKER_CTLN,
        "RLOO": MARKER_REINF,
        None: MARKER_LOSS_UNKNOWN,
    }

    # 6. Compute a "color hue key" for each experiment so we can cycle linestyles
    #    within groups that share the same hue. The key includes every detected
    #    modification (with its params), so combined-setting runs get their own
    #    group (distinct from single-mod runs of any component).
    def _color_hue_key(props):
        parts = []
        for mod in props["modifications"]:
            if mod == "beta_annealed":
                parts.append(("beta_annealed", props["beta_anneal_start"]))
            elif mod == "cfn":
                parts.append(("cfn", props["cfn_alpha"], props["cfn_annealed"]))
            elif mod == "entropy":
                parts.append(("entropy", props["entropy_alpha"], props["entropy_annealed"]))
            elif mod == "mixture":
                parts.append(("mixture", props["mixture_variant"], props["mixture_lag_steps"]))
            elif mod == "exact_count":
                parts.append(("exact_count",))
        return tuple(parts) if parts else ("none",)

    hue_group_counter = {}  # hue_key -> running count of experiments seen

    # Per-modification color lookup. Each modification contributes one RGBA tuple;
    # the experiment's final color is the RGBA average across every detected mod.
    # Single-mod runs are unchanged (mean of a length-1 list is the value itself).
    def _color_for_mod(mod, props, t):
        if mod == "cfn":
            shade = cfn_alpha_shade.get(props["cfn_alpha"], (CFN_SHADE_LO + CFN_SHADE_HI) / 2)
            return CFN_CMAP(shade)
        elif mod == "entropy":
            shade = entropy_alpha_shade.get(
                props["entropy_alpha"], (ENTROPY_SHADE_LO + ENTROPY_SHADE_HI) / 2)
            return ENTROPY_CMAP(shade)
        elif mod == "beta_annealed":
            shade = beta_start_shade.get(
                props["beta_anneal_start"], (BETA_ANNEAL_SHADE_LO + BETA_ANNEAL_SHADE_HI) / 2)
            return BETA_ANNEAL_CMAP(shade)
        elif mod == "mixture":
            mix_key = (props["mixture_variant"], props["mixture_lag_steps"])
            base_shade = mixture_shade_map.get(mix_key, 0.6)
            # Window = 35% of the gap between adjacent mixture keys, so LR ticks never
            # overlap with neighbouring mixture key colours.
            mix_spacing = (MIXTURE_SHADE_HI - MIXTURE_SHADE_LO) / max(n_mix - 1, 1)
            lr_half_window = mix_spacing * 0.35
            # t in [0,1] -> adjustment in [-lr_half_window, +lr_half_window]
            shade = np.clip(base_shade + lr_half_window * (2 * t - 1), 0.10, 0.95)
            cmap = mixture_cmap_map.get(mix_key, cm.cool)
            return cmap(shade)
        elif mod == "exact_count":
            lo, hi = shade_range_default
            shade = lo + t * (hi - lo)
            return cm.YlGnBu(shade)
        else:
            return cm.Greys(0.5)

    # 7. Build output lists
    color_list = []
    marker_list = []
    linestyle_list = []

    for props in props_list:
        # LR rank position for within-type shade adjustment
        if props["learning_rate"] is not None and props["learning_rate"] in lr_position:
            t = lr_position[props["learning_rate"]]
        else:
            t = 0.5

        modifications = props["modifications"]
        if modifications:
            # Blend (RGBA average) across every detected modification.
            component_colors = [_color_for_mod(m, props, t) for m in modifications]
            blended = np.clip(np.mean(component_colors, axis=0), 0.0, 1.0)
            # Force alpha = 1.0 to avoid fp drift from averaging.
            color_list.append((float(blended[0]), float(blended[1]), float(blended[2]), 1.0))
        else:
            # Baseline (no modifications): Greys with LR-based shade.
            lo, hi = shade_range_default
            shade = lo + t * (hi - lo)
            color_list.append(cm.Greys(shade))

        # Marker: beta_annealed -> hexagon; exact_count -> star; annealed entropy -> pentagon;
        # entropy -> diamond; annealed CFN -> filled X; non-annealed CFN -> filled plus;
        # mixture -> triangle; otherwise by loss type.
        # Non-annealed CFN and mixture get their own markers so they don't collide with
        # the generic loss_type marker ("o" for CTL), which would otherwise make base
        # CTL indistinguishable from CTL+CFN and CTL+mixture in the frontier plots.
        if props["beta_annealed"]:
            if props["loss_type"] == "RLOO":
                marker_list.append(MARKER_BETA_ANNEALED_REINF)
            else:
                marker_list.append(MARKER_BETA_ANNEALED)
        elif props["bonus_type"] == "exact_count":
            marker_list.append(MARKER_EXACT_COUNT)
        elif props.get("entropy_annealed"):
            if props["loss_type"] == "RLOO":
                marker_list.append(MARKER_ENTROPY_ANNEALED_REINF)
            else:
                marker_list.append(MARKER_ENTROPY_ANNEALED)
        elif props["bonus_type"] == "entropy":
            if props["loss_type"] == "RLOO":
                marker_list.append(MARKER_ENTROPY_REINF)
            else:
                marker_list.append(MARKER_ENTROPY)
        elif props["cfn_annealed"]:
            if props["loss_type"] == "RLOO":
                marker_list.append(MARKER_CFN_ANNEALED_REINF)
            else:
                marker_list.append(MARKER_CFN_ANNEALED)
        elif props["bonus_type"] == "cfn":
            if props["loss_type"] == "RLOO":
                marker_list.append(MARKER_CFN_REINF)
            else:
                marker_list.append(MARKER_CFN)
        elif props["bonus_type"] == "mixture":
            if props["loss_type"] == "RLOO":
                marker_list.append(MARKER_MIXTURE_REINF)
            else:
                marker_list.append(MARKER_MIXTURE)
        else:
            marker_list.append(loss_marker.get(props["loss_type"], MARKER_LOSS_UNKNOWN))

        # Linestyle: cycle within each color-hue group so same-hue experiments
        # are distinguishable even without markers (e.g. in time-series plots)
        hue_key = _color_hue_key(props)
        idx = hue_group_counter.get(hue_key, 0)
        hue_group_counter[hue_key] = idx + 1
        linestyle_list.append(linestyle_options[idx % len(linestyle_options)])

    return color_list, marker_list, linestyle_list


def generate_method_linestyles_from_prefixes(load_prefixes_to_use):
    """Generate per-method linestyles so different methods (CFN vs entropy vs mixture
    vs annealed beta vs baseline) get visually distinct line patterns.

    Unlike generate_visual_style_from_prefixes (which cycles linestyles within each
    hue group — i.e., across runs that share the same method/params), this assigns
    one linestyle per unique modification signature, so CFN is distinct from entropy,
    etc., even though each already has a different color.

    The method key is the sorted tuple of detected modifications (plus a flag for
    CFN annealing so "CFN" and "annealed CFN" get different styles).

    Returns:
        linestyle_list: list parallel to load_prefixes_to_use with a matplotlib
        linestyle per experiment.
    """
    linestyle_options = ["solid", "dashed", "dotted", "dashdot", (5, (10, 3)),
                         (0, (3, 5, 1, 5)), (0, (1, 1)), (0, (5, 1)), (0, (1, 3))]
    method_to_ls = {}
    linestyle_list = []
    for prefix_list in load_prefixes_to_use:
        first_prefix = prefix_list[0] if prefix_list else ""
        props = _parse_experiment_properties(first_prefix)
        # Method key: sorted modifications, plus annealing flags that should get
        # their own linestyle (annealed CFN vs. non-annealed CFN).
        key = (
            tuple(sorted(props["modifications"])),
            props.get("cfn_annealed", False),
            props.get("entropy_annealed", False),
        )
        if key not in method_to_ls:
            method_to_ls[key] = linestyle_options[len(method_to_ls) % len(linestyle_options)]
        linestyle_list.append(method_to_ls[key])
    return linestyle_list


def _label_parts_for_mod(mod, prefix, props, final_plots=False):
    """Return the label fragments contributed by a single modification.

    Combined-setting runs concatenate the fragments from every detected modification
    (in detection priority order), so the legend shows each component. Parts that
    are shared across all modifications (LR (q), batch_size, base LR, beta annealing)
    are emitted once by the caller rather than per modification, to avoid duplication.

    When ``final_plots`` is True, mixture collapses to the bare "Mixture" tag
    (variant/other-model details dropped) for a more compact legend.
    """
    if mod == "cfn":
        parts = ["CFN"]

        # Extract bonus_alpha from _cf pattern. With annealing: _cf{start}to{end}{schedule}
        cf_anneal_match = re.search(r'_cf([\d.]+(?:e[+-]?\d+)?)to([\d.]+(?:e[+-]?\d+)?)(linear|log)', prefix)
        if cf_anneal_match:
            schedule_suffix = "" if final_plots else f" ({cf_anneal_match.group(3)})"
            parts.append(r"$\alpha$: " + cf_anneal_match.group(1) + r"$\to$" + cf_anneal_match.group(2) + schedule_suffix)
        else:
            cf_match = re.search(r'_cf([\d.]+)', prefix)
            if cf_match:
                parts.append(r"$\alpha$=" + cf_match.group(1))

        # Additional CFN detection details — commented out to keep the legend short.
        # Preserved in case we want to bring them back later.
        # paren_parts = []
        # # Extract coin_flip_dim from _cd pattern
        # cd_match = re.search(r'_cd(\d+)', prefix)
        # if cd_match:
        #     paren_parts.append(f"d{cd_match.group(1)}")
        # # Extract cfus/cfu (updates) - support both old and new
        # cfus_match = re.search(r'_cfus(\d+)', prefix) or re.search(r'_cfu(\d+)', prefix)
        # paren_parts.append(f"{cfus_match.group(1)} upd." if cfus_match else "1 upd.")
        # # Check for "after"/"before" or new abbreviations "af"/"bf"
        # if "after" in prefix or "_af" in prefix:
        #     paren_parts.append("after")
        # elif "before" in prefix or "_bf" in prefix:
        #     paren_parts.append("before")
        # if "firstonline" in prefix or "_fo" in prefix:
        #     paren_parts.append("online")
        # if "pri" in prefix or "_pr" in prefix:
        #     paren_parts.append("pri")
        # if paren_parts:
        #     parts[0] = f"CFN ({', '.join(paren_parts)})"
        #
        # # Extract cfhis/cfh (coin flip head init std) - support both old and new
        # cfhis_match = re.search(r'_cfhis([\d.e-]+)', prefix) or re.search(r'_cfh([\d.e-]+)', prefix)
        # if cfhis_match:
        #     parts.append(f"head_std={cfhis_match.group(1)}")
        #
        # # Extract fpis/fp (frozen prior init std) - support both old and new
        # fpis_match = re.search(r'_fpis([\d.e-]+)', prefix) or re.search(r'_fp([\d.e-]+)', prefix)
        # if fpis_match:
        #     parts.append(f"prior_std={fpis_match.group(1)}")
        #
        # # Check for coin_flip_linear_bias (old _cfbias or new _cfb)
        # if "_cfbias" in prefix or "_cfb" in prefix:
        #     parts.append("bias")
        #
        # # Extract cflr/cfr (coin flip learning rate) - support both old and new
        # cflr_match = re.search(r'_cflr([\d.e-]+)', prefix) or re.search(r'_cfr([\d.e-]+)', prefix)
        # if cflr_match:
        #     parts.append(f"{cflr_match.group(1)} LR (CF)")
        #
        # # Architecture
        # if "sepnn" in prefix or "_cfsn" in prefix:
        #     parts.append("Sep. NN")
        # elif "cflsib" in prefix or "_cfs" in prefix:
        #     parts.append("Lin. Static Base")
        # elif "cfllq" in prefix or "_cfq" in prefix:
        #     parts.append("Lin. on q")
        # elif "cfllp" in prefix or "_cfl" in prefix:
        #     parts.append("Lin. on p")

        return parts

    elif mod == "exact_count":
        parts = ["EC"]
        # Extract bonus_alpha (encoded as _count or _c followed by value).
        # With annealing: _c{start}to{end}{schedule} (e.g., _c10to0linear)
        c_anneal_match = re.search(r'_c([\d.]+(?:e[+-]?\d+)?)to([\d.]+(?:e[+-]?\d+)?)(linear|log)', prefix)
        if c_anneal_match:
            schedule_suffix = "" if final_plots else f" ({c_anneal_match.group(3)})"
            parts.append(f"alpha={c_anneal_match.group(1)}->{c_anneal_match.group(2)}{schedule_suffix}")
        else:
            count_match = re.search(r'_count([\d.]+)', prefix) or re.search(r'_c([\d.]+)', prefix)
            if count_match:
                parts.append(f"alpha={count_match.group(1)}")
        return parts

    elif mod == "mixture":
        if final_plots:
            return ["Mixture"]

        mix_variant_map = {"mx": "mixture", "qi": "q_independent", "qh": "q_half"}
        mix_match = re.search(r'_mix([a-z]+)', prefix)
        mix_variant = mix_variant_map.get(mix_match.group(1), mix_match.group(1)) if mix_match else "unknown"

        # Other-model strategy: _first, _lag<N>, or absent (= best, omitted for brevity)
        other_model = props["mixture_other_model"]
        if other_model == "first":
            other_str = ", first"
        elif other_model == "lag":
            lag_steps = props["mixture_lag_steps"]
            other_str = f", lag={lag_steps}" if lag_steps is not None else ", lag"
        else:
            other_str = ""

        return [f"Mixture ({mix_variant}{other_str})"]

    elif mod == "entropy":
        parts = ["Entropy"]
        # With annealing: _entb{start}to{end}{schedule} (e.g., _entb0.1to0linear)
        entb_anneal_match = re.search(r'_entb([\d.e-]+)to([\d.e-]+)(linear|log)', prefix)
        if entb_anneal_match:
            schedule_suffix = "" if final_plots else f" ({entb_anneal_match.group(3)})"
            parts.append(r"$\alpha$: " + entb_anneal_match.group(1) + r"$\to$" + entb_anneal_match.group(2) + schedule_suffix)
        else:
            entb_match = re.search(r'_entb([\d.e-]+)', prefix)
            if entb_match:
                parts.append(r"$\alpha$=" + entb_match.group(1))
        return parts

    return []


def generate_labels_from_prefixes(load_prefixes_to_use, final_plots=False):
    """
    Generate labels from prefix lists based on the naming logic.

    Mirrors the `info_name_str` composition style: each modification contributes a
    label fragment (via `_label_parts_for_mod`), and combined-setting runs
    concatenate the fragments from every detected modification in detection order
    (cfn > exact_count > mixture > entropy). Shared fields (beta annealing,
    LR (q), batch size, base LR, mixeval tag, info_eval beta) are appended once
    at the end so they don't duplicate across modifications. Beta annealing is
    emitted *before* the LR fragments in both modes.

    When ``final_plots`` is True, produce a compact label suitable for final
    figures: the "No Bonus" tag and batch-size fragment are dropped, "LR (q)"
    shortens to "LR", annealed-beta gets a leading "Tempered " marker, and the
    mixture fragment collapses to "Mixture".

    Supports both old and new abbreviated naming conventions.

    Args:
        load_prefixes_to_use: List of lists of prefixes (each inner list contains prefixes for one series)
        final_plots: If True, use the compact label format described above.
            Defaults to False (existing behavior preserved, except for the
            beta-annealing ordering change noted above).

    Returns:
        List of label strings, one for each prefix list
    """
    labels = []
    for a in load_prefixes_to_use:
        prefix = a[0]
        props = _parse_experiment_properties(prefix)
        loss_type_str = props["loss_type"]  # "CTL", "CTLN", "RLOO", or None

        # Per-modification fragments. Beta annealing is still handled once in
        # the shared tail (just before the LR fragments) so it doesn't duplicate
        # across multiple modifications.
        non_beta_mods = [m for m in props["modifications"] if m != "beta_annealed"]
        label_parts = []
        if not non_beta_mods:
            if not final_plots:
                label_parts.append("No Bonus")
        else:
            for mod in non_beta_mods:
                label_parts.extend(
                    _label_parts_for_mod(mod, prefix, props, final_plots=final_plots)
                )

        # Shared tail — appended once regardless of modification count.
        # Beta annealing is emitted before the LR fragments so the tempered
        # schedule reads as a property of the run rather than a trailing note.
        if props["beta_annealed"]:
            beta_match = re.search(r'_s(-?[\d.]+)_b(-?[\d.]+)', prefix)
            if beta_match:
                beta_label = r"$\beta$: " + beta_match.group(1) + r"$\to$" + beta_match.group(2)
                if final_plots:
                    beta_label = "Tempered " + beta_label
                label_parts.append(beta_label)

        # LR fragments are suppressed entirely in final_plots mode so the legend
        # stays compact (just the method + key hyperparameters like alpha/beta).
        if not final_plots:
            al_match = re.search(r'_al([\d.e-]+)', prefix)
            if al_match:
                label_parts.append(f"{al_match.group(1)} LR (q)")

            tbs_match = re.search(r'_tbs(\d+)', prefix) or re.search(r'_tb(\d+)', prefix)
            if tbs_match:
                label_parts.append(f"batch={tbs_match.group(1)}")

            # Append base actor LR if non-zero (encoded as _bl followed by value)
            bl_match = re.search(r'_bl([\d.e-]+)', prefix)
            if bl_match:
                bl_val = float(bl_match.group(1))
                if bl_val != 0.0:
                    label_parts.append(f"{bl_match.group(1)} LR (p)")

        # When a separate reweighting beta (_sb<value>) is present, distinguish the two betas
        # with subscripts: β_q for target_dist_beta (used to train q) and β_p for the separate
        # reweighting beta (used for σ/p reweighting).
        sb_match = re.search(r'_sb(-?[\d.]+)', prefix)
        if sb_match:
            sb_val = sb_match.group(1)
            if props["beta_annealed"]:
                # Rename the existing β: start→end label to β_q: start→end
                # (also works when a "Tempered " prefix has been prepended).
                for i, part in enumerate(label_parts):
                    if r"$\beta$:" in part:
                        label_parts[i] = part.replace(r"$\beta$:", r"$\beta_q$:")
                        break
            # β_q is clear from context; only show the distinct separate reweighting beta.
            label_parts.append(r"$\beta_p$=" + sb_val)

        # Prepend loss type label.
        # When a reward transform is present (rt<alpha>_b<beta> in prefix), the harmlessness
        # training uses REINFORCE with a transformed reward — label that explicitly instead of CTL.
        rt_match = re.search(r'rt([\d.e-]+)_b([-\d.e]+)', prefix)
        if rt_match:
            rt_alpha = rt_match.group(1)
            rt_beta = rt_match.group(2)
            label_parts.insert(0, f"REINFORCE (reward transform, alpha={rt_alpha}, beta={rt_beta})")
        elif loss_type_str is not None:
            label_parts.insert(0, loss_type_str)

        # Check for mixeval prefix
        if prefix.startswith("f_q_g_q_iwae_bounds_mixeval"):
            label_parts.append("(mixeval)")

        # For info_eval prefixes, include the beta value.
        # Prefixes use abbreviated _b<value> (e.g. _b-10.0); _beta<value> also supported.
        # Use negative lookahead to avoid matching _bl (base LR) or other _b<letter> patterns.
        if prefix.startswith("info_eval") and not sb_match:
            beta_match = re.search(r'_beta([-\d.]+)', prefix) or re.search(r'_b(?![a-zA-Z])([-\d.]+)', prefix)
            if beta_match:
                label_parts.append(r"$\beta$=" + beta_match.group(1))

        labels.append(", ".join(label_parts))

    return labels


def to_scalar(x):
    """
    Convert list/tensor to a single float (mean over all elements).

    Args:
        x: Input that can be converted to numpy array (list, tensor, array, etc.)

    Returns:
        float: Mean value over all elements
    """
    return float(np.asarray(x).ravel().mean())


def mean_of_per_prompt_bounds(per_prompt_list):
    """Compute mean of per-prompt IWAE bounds at each timestep.

    In multi-prompt settings, the aggregate IWAE bounds (computed by concatenating
    samples across prompts into one logsumexp) are not meaningful — each prompt has
    its own target distribution and log Z. This function returns the mean of the
    per-prompt bounds at each timestep, which is a proper average of per-prompt log Z
    estimates.

    Args:
        per_prompt_list: List of steps, each a list of per-prompt scalar values (or Nones).

    Returns:
        List of mean values (float), or None for timesteps with no valid values.
    """
    result = []
    for per_prompt in per_prompt_list:
        valid = [float(x) for x in per_prompt if x is not None]
        result.append(sum(valid) / len(valid) if valid else None)
    return result


def compute_global_logZ_from_iwae_bounds(loaded_data):
    """
    Compute global log Z from IWAE bounds across all experiments and seeds.
    
    Args:
        loaded_data: List of experiments, where each experiment is a list of seeds.
                     Each seed contains a tuple (f_q_estimates_list, g_q_estimates_list, 
                     iwae_lbs_list, iwae_ubs_list)
    
    Returns:
        float: Median global log Z value across all seed and experiment estimates
    """
    all_logZ_estimates = []
    
    for exp_data in loaded_data:
        # Per-seed log Z
        for f_q_estimates_list, g_q_estimates_list, iwae_lbs_list, iwae_ubs_list in exp_data:
            # Filter out None entries (can occur when g_q/IWAE are unavailable for some timesteps,
            # e.g. in the fixed trajectory setting without target samples)
            valid_lbs = [x for x in iwae_lbs_list if x is not None]
            valid_ubs = [x for x in iwae_ubs_list if x is not None]
            if len(valid_lbs) == 0 or len(valid_ubs) == 0:
                continue
            lb_per_t = [to_scalar(x) for x in valid_lbs]
            ub_per_t = [to_scalar(x) for x in valid_ubs]
            logZ_seed = (max(lb_per_t) + min(ub_per_t)) / 2.0
            all_logZ_estimates.append(logZ_seed)

        # Per-experiment log Z (average lb/ub per timestep across seeds, then midpoint)
        if len(exp_data) == 0:
            continue
        T = len(exp_data[0][2])  # Length of iwae_lbs_list
        lb_per_t_raw = [
            [to_scalar(exp_data[s][2][t]) for s in range(len(exp_data))
             if t < len(exp_data[s][2]) and exp_data[s][2][t] is not None]
            for t in range(T)
        ]
        ub_per_t_raw = [
            [to_scalar(exp_data[s][3][t]) for s in range(len(exp_data))
             if t < len(exp_data[s][3]) and exp_data[s][3][t] is not None]
            for t in range(T)
        ]
        lb_per_t = [np.mean(vals) for vals in lb_per_t_raw if len(vals) > 0]
        ub_per_t = [np.mean(vals) for vals in ub_per_t_raw if len(vals) > 0]
        if lb_per_t and ub_per_t:
            all_logZ_estimates.append((max(lb_per_t) + min(ub_per_t)) / 2.0)
    
    if not all_logZ_estimates:
        raise ValueError("No log Z estimates found in loaded data")
    
    global_logZ = float(np.median(all_logZ_estimates))
    return global_logZ


def compute_approx_kl_from_f_q_g_q(f_q_estimates, g_q_estimates, global_logZ):
    """
    Compute approximate KL divergence estimates from f_q, g_q, and global log Z.
    
    Args:
        f_q_estimates: f_q estimate(s) - can be scalar, array, or tensor
        g_q_estimates: g_q estimate(s) - can be scalar, array, or tensor  
        global_logZ: Global log Z value (float)
    
    Returns:
        tuple: (kl_sigma_q, kl_q_sigma) where:
            - kl_sigma_q = g_q - global_logZ (KL(sigma_p || q))
            - kl_q_sigma = global_logZ - f_q (KL(q || sigma_p))
            Both have the same shape as the inputs
    """
    # Convert to numpy arrays for consistent handling
    if isinstance(f_q_estimates, torch.Tensor):
        f_q_array = f_q_estimates.float().cpu().numpy()
    else:
        f_q_array = np.asarray(f_q_estimates)
    
    if isinstance(g_q_estimates, torch.Tensor):
        g_q_array = g_q_estimates.float().cpu().numpy()
    else:
        g_q_array = np.asarray(g_q_estimates)
    
    # Compute KL divergences
    kl_q_sigma = global_logZ - f_q_array
    kl_sigma_q = g_q_array - global_logZ
    
    return kl_sigma_q, kl_q_sigma


def _collect_top_token_log_probs(labels, results_list, n_top_tokens=10, final_only=False):
    """
    Extract per-token log probabilities (target, q, and base) from results_list.

    Args:
        final_only: If True, only use the last metrics dict per seed (final evaluation timestep)
                    instead of averaging across all timesteps.

    Returns:
        (token_log_probs_target_by_setting, token_log_probs_q_by_setting,
         token_log_probs_base_by_setting, top_n_tokens)
        where:
            token_log_probs_{target,q,base}_by_setting: dict of token_id -> {setting_idx: [per-seed averages]}
            top_n_tokens: list of token IDs ranked by average target log prob (descending)
        Returns (None, None, None, None) if no token data is found.
    """
    token_log_probs_target_by_setting = {}  # token_id -> {setting_idx: [per-seed averages]}
    token_log_probs_q_by_setting = {}  # token_id -> {setting_idx: [per-seed averages]}
    token_log_probs_base_by_setting = {}  # token_id -> {setting_idx: [per-seed averages]}

    for setting_idx in range(len(labels)):
        tuple_list = results_list[setting_idx]

        if not tuple_list:
            print(f"Warning: Empty tuple_list for {labels[setting_idx]}. Skipping.")
            continue

        for t_idx, t in enumerate(tuple_list):
            # t should be a tuple: (kl_sigma_q_list, kl_q_sigma_list, metrics_list)
            if not isinstance(t, tuple) or len(t) < 3:
                print(f"Warning: Expected tuple with at least 3 elements for {labels[setting_idx]}, seed {t_idx+1}. Skipping.")
                continue

            metrics_list = t[2]  # List of metrics dicts (one per prompt)

            if not isinstance(metrics_list, list):
                print(f"Warning: metrics_list should be a list for {labels[setting_idx]}, seed {t_idx+1}. Got {type(metrics_list)}. Skipping.")
                continue

            # Collect log probs for this seed
            seed_token_log_probs = {}  # token_id -> {'target': [values], 'q': [values], 'base': [values]}

            metrics_to_use = [metrics_list[-1]] if final_only else metrics_list
            for metrics_dict in metrics_to_use:
                if not isinstance(metrics_dict, dict):
                    continue

                # Prefer full-vocab tensors (new format) over dict entries (old format)
                log_probs_target_full = metrics_dict.get('log_probs_target_full', None)
                log_probs_q_full = metrics_dict.get('log_probs_q_full', None)
                log_probs_base_full = metrics_dict.get('log_probs_base_full', None)

                if log_probs_target_full is not None and log_probs_q_full is not None:
                    # New format: full-vocab tensors — iterate over all tokens
                    import torch
                    n_vocab = len(log_probs_target_full)
                    for token_id in range(n_vocab):
                        if token_id not in seed_token_log_probs:
                            seed_token_log_probs[token_id] = {'target': [], 'q': [], 'base': []}
                        seed_token_log_probs[token_id]['target'].append(log_probs_target_full[token_id].item())
                        seed_token_log_probs[token_id]['q'].append(log_probs_q_full[token_id].item())
                        if log_probs_base_full is not None:
                            seed_token_log_probs[token_id]['base'].append(log_probs_base_full[token_id].item())
                else:
                    # Old format: dict entries for tracked tokens only
                    tracked_tokens = metrics_dict.get('all_tracked_tokens', metrics_dict.get('top_10_target_tokens', []))
                    log_probs_target = metrics_dict.get('log_probs_target', {})
                    log_probs_q = metrics_dict.get('log_probs_q', {})

                    for token_id in tracked_tokens:
                        if token_id not in seed_token_log_probs:
                            seed_token_log_probs[token_id] = {'target': [], 'q': [], 'base': []}

                        if token_id in log_probs_target:
                            seed_token_log_probs[token_id]['target'].append(log_probs_target[token_id])

                        if token_id in log_probs_q:
                            seed_token_log_probs[token_id]['q'].append(log_probs_q[token_id])

            # Average across prompts for this seed, then store
            for token_id, probs_dict in seed_token_log_probs.items():
                target_values = probs_dict['target']
                q_values = probs_dict['q']
                base_values = probs_dict.get('base', [])

                if len(target_values) > 0 and len(q_values) > 0:
                    seed_avg_target = np.mean(target_values)
                    seed_avg_q = np.mean(q_values)

                    if token_id not in token_log_probs_target_by_setting:
                        token_log_probs_target_by_setting[token_id] = {}
                    if setting_idx not in token_log_probs_target_by_setting[token_id]:
                        token_log_probs_target_by_setting[token_id][setting_idx] = []
                    token_log_probs_target_by_setting[token_id][setting_idx].append(seed_avg_target)

                    if token_id not in token_log_probs_q_by_setting:
                        token_log_probs_q_by_setting[token_id] = {}
                    if setting_idx not in token_log_probs_q_by_setting[token_id]:
                        token_log_probs_q_by_setting[token_id][setting_idx] = []
                    token_log_probs_q_by_setting[token_id][setting_idx].append(seed_avg_q)

                    if len(base_values) > 0:
                        seed_avg_base = np.mean(base_values)
                        if token_id not in token_log_probs_base_by_setting:
                            token_log_probs_base_by_setting[token_id] = {}
                        if setting_idx not in token_log_probs_base_by_setting[token_id]:
                            token_log_probs_base_by_setting[token_id][setting_idx] = []
                        token_log_probs_base_by_setting[token_id][setting_idx].append(seed_avg_base)

    # Find top N tokens by average log probability under target distribution across all settings
    token_avg_log_probs_target = {}
    for token_id, setting_dict in token_log_probs_target_by_setting.items():
        all_values = []
        for values in setting_dict.values():
            all_values.extend(values)
        if all_values:
            token_avg_log_probs_target[token_id] = np.mean(all_values)

    if len(token_avg_log_probs_target) == 0:
        print("Warning: No token data found.")
        return None, None, None, None

    sorted_tokens = sorted(token_avg_log_probs_target.items(), key=lambda x: x[1], reverse=True)
    top_n_tokens = [token_id for token_id, _ in sorted_tokens[:n_top_tokens]]

    return token_log_probs_target_by_setting, token_log_probs_q_by_setting, token_log_probs_base_by_setting, top_n_tokens


def _bootstrap_mean_ci(values, n_bootstrap_draws=5000, alpha=0.05):
    """Compute mean and bootstrap confidence interval for an array of values.

    Returns (mean, ci_lower, ci_upper).  If fewer than 2 values, CI equals the mean.
    """
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean_val = np.mean(values)
    if len(values) < 2:
        return mean_val, mean_val, mean_val
    bootstrap_means = []
    for _ in range(n_bootstrap_draws):
        resample = values[np.random.choice(len(values), size=len(values), replace=True)]
        bootstrap_means.append(np.mean(resample))
    ci_lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    return mean_val, ci_lower, ci_upper


def _compute_target_means_for_tokens(token_ids, target_by_setting):
    """Compute pooled target mean per token across all settings/seeds.

    Returns:
        np.array of shape (len(token_ids),), NaN where no data.
    """
    target_means = np.full(len(token_ids), np.nan)
    for idx, token_id in enumerate(token_ids):
        all_target_vals = []
        for vals in target_by_setting.get(token_id, {}).values():
            all_target_vals.extend(vals)
        if all_target_vals:
            target_means[idx] = np.mean(all_target_vals)
    return target_means


def _draw_target_dashes(ax, x_positions, target_means, dash_half_width, linewidth=2, label_text=None,
                        color='black', linestyle='-'):
    """Draw horizontal dashes for a reference series at each position.

    Args:
        label_text: Legend label for the dashes. Defaults to "$\\sigma$ (target)" if None.
        color: Line color (default 'black').
        linestyle: Line style (default '-' solid).
    """
    if label_text is None:
        label_text = r"$\sigma$ (target)"
    target_label_added = False
    for pos_idx in range(len(x_positions)):
        if np.isnan(target_means[pos_idx]):
            continue
        label = label_text if not target_label_added else None
        ax.plot(
            [x_positions[pos_idx] - dash_half_width, x_positions[pos_idx] + dash_half_width],
            [target_means[pos_idx], target_means[pos_idx]],
            color=color, linewidth=linewidth, linestyle=linestyle, solid_capstyle='butt',
            label=label, zorder=3,
        )
        target_label_added = True


def _has_cumulative_q_sample_counts(results_list):
    """Check if any entry in results_list contains cumulative_q_sample_counts."""
    for setting_data in results_list:
        for t in setting_data:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list) and len(t[2]) > 0:
                last = t[2][-1]
                if isinstance(last, dict) and 'cumulative_q_sample_counts' in last:
                    return True
    return False


def _compute_top_tokens_per_setting_and_grouped(q_by_setting, n_settings, n_top_tokens):
    """For each setting, compute mean log q per token across seeds, sort, pick top N.

    Returns:
        top_tokens_per_setting: dict of setting_idx -> list of (token_id, mean_q) sorted descending
        grouped: list of (setting_idx, token_id) tuples, ordered by setting then by rank
    """
    top_tokens_per_setting = {}
    for setting_idx in range(n_settings):
        token_avg_q = {}
        for token_id, setting_dict in q_by_setting.items():
            vals = setting_dict.get(setting_idx, [])
            if vals:
                token_avg_q[token_id] = np.mean(vals)
        sorted_tokens = sorted(token_avg_q.items(), key=lambda x: x[1], reverse=True)
        top_tokens_per_setting[setting_idx] = sorted_tokens[:n_top_tokens]

    grouped = []
    for setting_idx in range(n_settings):
        for token_id, _ in top_tokens_per_setting.get(setting_idx, []):
            grouped.append((setting_idx, token_id))

    return top_tokens_per_setting, grouped


def _collect_rank_data(labels, results_list, n_ranks):
    """Collect per-setting, per-seed, per-rank log probs at the final timestep.

    For each setting and seed, sorts tracked tokens by log_probs_q descending and
    records log prob under q and target at each rank. Uses torch.topk when full-vocab
    tensors are available, otherwise falls back to dict-based sorting.

    Returns:
        rank_data: dict of setting_idx -> list of (q_by_rank, target_by_rank, ids_by_rank) per seed.
            None if no data found.
        n_ranks_actual: actual number of ranks available. 0 if no data found.
    """
    rank_data = {i: [] for i in range(len(labels))}

    for setting_idx in range(len(labels)):
        tuple_list = results_list[setting_idx]
        if not tuple_list:
            continue

        for t in tuple_list:
            if not isinstance(t, tuple) or len(t) < 3:
                continue
            metrics_list = t[2]
            if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                continue

            last_metrics = metrics_list[-1]
            if not isinstance(last_metrics, dict):
                continue

            log_probs_q_full = last_metrics.get('log_probs_q_full', None)
            log_probs_target_full = last_metrics.get('log_probs_target_full', None)

            if log_probs_q_full is not None and log_probs_target_full is not None:
                top_q_vals, top_q_ids = torch.topk(log_probs_q_full, k=n_ranks)
                q_by_rank = top_q_vals.tolist()
                ids_by_rank = top_q_ids.tolist()
                target_by_rank = [log_probs_target_full[tid].item() for tid in ids_by_rank]
            else:
                log_probs_q = last_metrics.get('log_probs_q', {})
                log_probs_target = last_metrics.get('log_probs_target', {})
                if not log_probs_q:
                    continue

                sorted_by_q = sorted(log_probs_q.items(), key=lambda x: x[1], reverse=True)

                q_by_rank = []
                target_by_rank = []
                ids_by_rank = []
                for token_id, log_q_val in sorted_by_q[:n_ranks]:
                    q_by_rank.append(log_q_val)
                    target_by_rank.append(log_probs_target.get(token_id, np.nan))
                    ids_by_rank.append(token_id)

            if q_by_rank:
                rank_data[setting_idx].append((np.array(q_by_rank), np.array(target_by_rank), ids_by_rank))

    has_data = any(len(seeds) > 0 for seeds in rank_data.values())
    if not has_data:
        return None, 0

    max_ranks = 0
    for seeds in rank_data.values():
        for q_by_rank, _, _ in seeds:
            max_ranks = max(max_ranks, len(q_by_rank))
    n_ranks_actual = min(n_ranks, max_ranks)

    return rank_data, n_ranks_actual


def plot_top_tokens_bar_chart(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
    final_only=False,
    precomputed_token_data=None,
    tokenizer=None,
):
    """
    Plot bar chart of log probability differences (q - target) for top N tokens under target distribution.

    Identifies top tokens by their target distribution probabilities, then plots log_probs_q - log_probs_target
    for those tokens. For each token, shows bars for each setting, with confidence intervals.

    results_list should contain tuples of (kl_sigma_q_list, kl_q_sigma_list, metrics_list)
    where metrics_list contains dicts with 'log_probs_q_full' and 'log_probs_target_full' tensors
    (or legacy dicts 'log_probs_q'/'log_probs_target' for old saved data).
    """
    plt.clf()

    if precomputed_token_data is not None:
        target_by_setting, q_by_setting, base_by_setting, all_sorted_tokens = precomputed_token_data
        top_n_tokens = all_sorted_tokens[:n_top_tokens]
    else:
        target_by_setting, q_by_setting, base_by_setting, top_n_tokens = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=final_only)
    if top_n_tokens is None:
        print("Warning: Cannot create bar chart.")
        return

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)
    has_base_data = len(base_by_setting) > 0

    plt.figure(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

    # Number of bar groups: one per setting + optionally one for base
    n_groups = n_settings + (1 if has_base_data else 0)
    bar_width = 0.25
    x_positions = np.arange(n_tokens)
    group_offsets = np.linspace(-bar_width * (n_groups - 1) / 2,
                                 bar_width * (n_groups - 1) / 2,
                                 n_groups)

    # Collect means and CIs for the difference (q - target)
    means = np.zeros((n_tokens, n_settings))
    ci_lowers = np.zeros((n_tokens, n_settings))
    ci_uppers = np.zeros((n_tokens, n_settings))

    for token_idx, token_id in enumerate(top_n_tokens):
        for setting_idx in range(n_settings):
            target_values = np.array(target_by_setting.get(token_id, {}).get(setting_idx, []))
            q_values = np.array(q_by_setting.get(token_id, {}).get(setting_idx, []))

            if len(target_values) > 0 and len(q_values) > 0:
                min_len = min(len(target_values), len(q_values))
                diff_values = q_values[:min_len] - target_values[:min_len]
            else:
                diff_values = np.array([])

            means[token_idx, setting_idx], ci_lowers[token_idx, setting_idx], ci_uppers[token_idx, setting_idx] = \
                _bootstrap_mean_ci(diff_values, n_bootstrap_draws)

    # Collect means and CIs for the difference (base - target) — pooled across settings
    base_means = np.full(n_tokens, np.nan)
    base_ci_lowers = np.full(n_tokens, np.nan)
    base_ci_uppers = np.full(n_tokens, np.nan)
    if has_base_data:
        for token_idx, token_id in enumerate(top_n_tokens):
            all_base_vals = []
            for vals in base_by_setting.get(token_id, {}).values():
                all_base_vals.extend(vals)
            all_target_vals = []
            for vals in target_by_setting.get(token_id, {}).values():
                all_target_vals.extend(vals)
            if all_base_vals and all_target_vals:
                min_len = min(len(all_base_vals), len(all_target_vals))
                diff_values = np.array(all_base_vals[:min_len]) - np.array(all_target_vals[:min_len])
            else:
                diff_values = np.array([])
            base_means[token_idx], base_ci_lowers[token_idx], base_ci_uppers[token_idx] = \
                _bootstrap_mean_ci(diff_values, n_bootstrap_draws)

    # Plot bars for each setting (q - target)
    for setting_idx in range(n_settings):
        x_pos = x_positions + group_offsets[setting_idx]
        plt.bar(x_pos, means[:, setting_idx], bar_width,
                label=labels[setting_idx],
                color=color_list[setting_idx],
                alpha=0.7)

        for token_idx in range(n_tokens):
            if not np.isnan(means[token_idx, setting_idx]):
                mean_val = means[token_idx, setting_idx]
                lower_err = mean_val - ci_lowers[token_idx, setting_idx]
                upper_err = ci_uppers[token_idx, setting_idx] - mean_val
                plt.errorbar(x_pos[token_idx], mean_val,
                           yerr=[[lower_err], [upper_err]],
                           fmt='none', color='black', capsize=3, linewidth=1)

    # Plot bars for base model (base - target)
    if has_base_data:
        x_pos_base = x_positions + group_offsets[-1]
        plt.bar(x_pos_base, base_means, bar_width,
                label=r"$p$ (base)",
                color='gray',
                alpha=0.7)
        for token_idx in range(n_tokens):
            if not np.isnan(base_means[token_idx]):
                mean_val = base_means[token_idx]
                lower_err = mean_val - base_ci_lowers[token_idx]
                upper_err = base_ci_uppers[token_idx] - mean_val
                plt.errorbar(x_pos_base[token_idx], mean_val,
                           yerr=[[lower_err], [upper_err]],
                           fmt='none', color='black', capsize=3, linewidth=1)

    plt.xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
    plt.ylabel('Log Probability Difference (model - target)', fontsize=fontsize)
    time_label = " (final step)" if final_only else " (avg over time)"
    plt.title(f'Top {n_top_tokens} Target Tokens: Log Prob Difference (model - target){time_label}', fontsize=fontsize+1)
    plt.xticks(x_positions, _maybe_hide_token_labels([_token_label(token_id, tokenizer) for token_id in top_n_tokens]), fontsize=fontsize-1)
    plt.legend(fontsize=legendfontsize)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(figname)
    print(f"Bar chart saved to {figname}")


def plot_top_tokens_lollipop(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
    final_only=False,
    precomputed_token_data=None,
    tokenizer=None,
    figname_individual=None,
    marker_list=None,
):
    """
    Lollipop/dumbbell chart showing absolute log probabilities under target and q for top N tokens.

    For each token:
      - A shared black horizontal dash shows the target distribution log probability.
      - For each setting, a colored dot shows the mean log q, with a bootstrap CI error bar.
      - A thin vertical line connects each q dot to the target dash.

    This complements plot_top_tokens_bar_chart (which shows differences) by letting you see
    the absolute scale of both distributions per token.
    """
    if precomputed_token_data is not None:
        target_by_setting, q_by_setting, base_by_setting, all_sorted_tokens = precomputed_token_data
        top_n_tokens = all_sorted_tokens[:n_top_tokens]
    else:
        target_by_setting, q_by_setting, base_by_setting, top_n_tokens = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=final_only)
    if top_n_tokens is None:
        print("Warning: Cannot create lollipop chart.")
        return

    # Print per-seed log probs for the top target tokens at the final timestep
    for setting_idx in range(len(labels)):
        tuple_list = results_list[setting_idx]
        if not tuple_list:
            continue
        print(f"\n[{labels[setting_idx]}] Per-seed log probs at top-{n_top_tokens} target tokens (final timestep):")
        for seed_j, t in enumerate(tuple_list):
            if not isinstance(t, tuple) or len(t) < 3:
                continue
            metrics_list = t[2]
            if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                continue
            last_metrics = metrics_list[-1]
            if not isinstance(last_metrics, dict):
                continue
            log_probs_q_full = last_metrics.get('log_probs_q_full', None)
            log_probs_target_full = last_metrics.get('log_probs_target_full', None)
            log_probs_q_dict = last_metrics.get('log_probs_q', {})
            log_probs_target_dict = last_metrics.get('log_probs_target', {})
            parts = []
            for token_id in top_n_tokens:
                if log_probs_q_full is not None and token_id < len(log_probs_q_full):
                    q_val = log_probs_q_full[token_id].item()
                else:
                    q_val = log_probs_q_dict.get(token_id, float('nan'))
                if log_probs_target_full is not None and token_id < len(log_probs_target_full):
                    t_val = log_probs_target_full[token_id].item()
                else:
                    t_val = log_probs_target_dict.get(token_id, float('nan'))
                parts.append(f"{token_id}(q={q_val:.2f}, σ={t_val:.2f})")
            print(f"  Seed {seed_j + 1}: {', '.join(parts)}")

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    fig, ax = plt.subplots(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

    # Horizontal spacing: settings are offset within each token position
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    # Compute target mean per token (pooled across all settings/seeds — target is shared)
    target_means = _compute_target_means_for_tokens(top_n_tokens, target_by_setting)

    # Compute base model mean per token (pooled across all settings/seeds — base is shared)
    base_means = np.full(n_tokens, np.nan)
    has_base_data = len(base_by_setting) > 0
    if has_base_data:
        for token_idx, token_id in enumerate(top_n_tokens):
            all_base_vals = []
            for vals in base_by_setting.get(token_id, {}).values():
                all_base_vals.extend(vals)
            if all_base_vals:
                base_means[token_idx] = np.mean(all_base_vals)

    # Draw target markers: one horizontal dash per token spanning the offset range
    dash_half_width = (total_width / 2 + dot_spacing * 0.6) if n_settings > 1 else 0.15
    _draw_target_dashes(ax, x_positions, target_means, dash_half_width)

    # Draw base model markers (if available): dashed horizontal line per token
    if has_base_data:
        base_label_added = False
        for token_idx in range(n_tokens):
            if np.isnan(base_means[token_idx]):
                continue
            label = r"$p$ (base)" if not base_label_added else None
            ax.plot(
                [x_positions[token_idx] - dash_half_width, x_positions[token_idx] + dash_half_width],
                [base_means[token_idx], base_means[token_idx]],
                color='gray', linewidth=2, linestyle='--', solid_capstyle='butt', label=label, zorder=3,
            )
            base_label_added = True

    # Draw q dots + connecting lines for each setting
    for setting_idx in range(n_settings):
        q_means = np.full(n_tokens, np.nan)
        q_ci_lo = np.full(n_tokens, np.nan)
        q_ci_hi = np.full(n_tokens, np.nan)

        for token_idx, token_id in enumerate(top_n_tokens):
            q_values = np.array(q_by_setting.get(token_id, {}).get(setting_idx, []))
            q_means[token_idx], q_ci_lo[token_idx], q_ci_hi[token_idx] = \
                _bootstrap_mean_ci(q_values, n_bootstrap_draws)

        x_pos = x_positions + setting_offsets[setting_idx]

        # q dots with CI error bars
        lower_err = q_means - q_ci_lo
        upper_err = q_ci_hi - q_means
        # Mask NaN for errorbar
        valid = ~np.isnan(q_means)
        if valid.any():
            marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.errorbar(
                x_pos[valid], q_means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt=marker, color=color_list[setting_idx], markersize=5,
                capsize=3, linewidth=1, label=labels[setting_idx], zorder=4,
            )

    ax.set_xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    time_label = " (final step)" if final_only else " (avg over time)"
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Log Probability of q{time_label}', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_maybe_hide_token_labels([_token_label(token_id, tokenizer) for token_id in top_n_tokens]), fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Lollipop chart saved to {figname}")

    # --- Individual-seed plot (reuses precomputed data) ---
    if figname_individual is not None:
        fig_ind, ax_ind = plt.subplots(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

        _draw_target_dashes(ax_ind, x_positions, target_means, dash_half_width)

        for setting_idx in range(n_settings):
            label_added = False
            for token_idx, token_id in enumerate(top_n_tokens):
                q_values = q_by_setting.get(token_id, {}).get(setting_idx, [])
                if not q_values:
                    continue
                x_base = x_positions[token_idx] + setting_offsets[setting_idx]
                for seed_j, val in enumerate(q_values):
                    marker = SEED_MARKERS[seed_j % len(SEED_MARKERS)]
                    label = labels[setting_idx] if not label_added else None
                    ax_ind.scatter(
                        x_base, val,
                        color=color_list[setting_idx], s=20, alpha=0.7,
                        marker=marker, label=label, zorder=4,
                    )
                    label_added = True

        ax_ind.set_xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
        ax_ind.set_ylabel('Log Probability', fontsize=fontsize)
        ax_ind.set_title(f'Top {n_top_tokens} Target Tokens: Log Prob (individual seeds, final step)', fontsize=fontsize + 1)
        ax_ind.set_xticks(x_positions)
        ax_ind.set_xticklabels(_maybe_hide_token_labels([_token_label(token_id, tokenizer) for token_id in top_n_tokens]), fontsize=fontsize - 1)
        ax_ind.tick_params(axis='y', labelsize=fontsize)
        ax_ind.legend(fontsize=legendfontsize)
        ax_ind.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        plt.savefig(figname_individual)
        plt.clf()
        plt.close(fig_ind)
        print(f"Individual lollipop chart saved to {figname_individual}")


def _collect_token_log_probs_at_timestep(results_list, setting_idx, timestep_idx, token_ids):
    """Extract per-seed log probs (target and q) for given tokens at a specific timestep index.

    Returns:
        (target_by_token, q_by_token) where each is dict of token_id -> list of per-seed values.
    """
    target_by_token = {tid: [] for tid in token_ids}
    q_by_token = {tid: [] for tid in token_ids}

    tuple_list = results_list[setting_idx]
    if not tuple_list:
        return target_by_token, q_by_token

    for t in tuple_list:
        if not isinstance(t, tuple) or len(t) < 3:
            continue
        metrics_list = t[2]
        if not isinstance(metrics_list, list) or len(metrics_list) == 0:
            continue
        if timestep_idx >= len(metrics_list):
            continue
        metrics_dict = metrics_list[timestep_idx]
        if not isinstance(metrics_dict, dict):
            continue
        # Prefer full-vocab tensors (new format) over dict entries (old format)
        log_probs_target_full = metrics_dict.get('log_probs_target_full', None)
        log_probs_q_full = metrics_dict.get('log_probs_q_full', None)
        if log_probs_target_full is not None and log_probs_q_full is not None:
            for tid in token_ids:
                if tid < len(log_probs_target_full):
                    target_by_token[tid].append(log_probs_target_full[tid].item())
                if tid < len(log_probs_q_full):
                    q_by_token[tid].append(log_probs_q_full[tid].item())
        else:
            log_probs_target = metrics_dict.get('log_probs_target', {})
            log_probs_q = metrics_dict.get('log_probs_q', {})
            for tid in token_ids:
                if tid in log_probs_target:
                    target_by_token[tid].append(log_probs_target[tid])
                if tid in log_probs_q:
                    q_by_token[tid].append(log_probs_q[tid])

    return target_by_token, q_by_token


def plot_top_tokens_lollipop_over_time(
    figname, labels, results_list,
    color_list, n_frontiers=4,
    fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
    marker_list=None,
):
    """
    Lollipop chart showing log q for top target tokens at multiple timesteps, with lines connecting
    the same token across time.

    Token selection is based on the final timestep (same as the _final lollipop plot).
    For each of n_frontiers evenly-spaced timesteps, dots show the mean log q (across seeds)
    for each token/setting, with bootstrap CIs. Dots for the same token+setting are connected
    by lines across timesteps. Target log prob is shown as a black horizontal dash (constant).

    Timestep progression is shown via marker alpha (lighter = earlier, darker = later).
    """
    # Select top tokens using final timestep
    if precomputed_token_data is not None:
        target_by_setting, q_by_setting, _, all_sorted_tokens = precomputed_token_data
        top_n_tokens = all_sorted_tokens[:n_top_tokens]
    else:
        target_by_setting, q_by_setting, _, top_n_tokens = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=True)
    if top_n_tokens is None:
        print("Warning: Cannot create over-time lollipop chart.")
        return

    # Determine max trajectory length across all settings/seeds
    max_T = 0
    for setting_idx in range(len(labels)):
        for t in results_list[setting_idx]:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list):
                max_T = max(max_T, len(t[2]))

    if max_T == 0:
        print("Warning: max_T=0, skipping over-time lollipop.")
        return

    if n_frontiers <= 0:
        return

    # Compute evenly-spaced timestep indices (same logic as _generate_kl_frontier_plots)
    frontier_indices = [round((max_T - 1) * i / n_frontiers) for i in range(1, n_frontiers + 1)]
    seen = set()
    unique_frontier_indices = []
    for idx in frontier_indices:
        if idx not in seen:
            seen.add(idx)
            unique_frontier_indices.append(idx)
    frontier_indices = unique_frontier_indices
    n_times = len(frontier_indices)

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    fig, ax = plt.subplots(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

    # Horizontal spacing: settings offset within each token position
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    # Compute target mean per token (from final timestep data, already collected)
    target_means = np.full(n_tokens, np.nan)
    for token_idx, token_id in enumerate(top_n_tokens):
        all_target_vals = []
        for vals in target_by_setting.get(token_id, {}).values():
            all_target_vals.extend(vals)
        if all_target_vals:
            target_means[token_idx] = np.mean(all_target_vals)

    # Draw target markers: one horizontal dash per token
    dash_half_width = (total_width / 2 + dot_spacing * 0.6) if n_settings > 1 else 0.15
    target_label_added = False
    for token_idx in range(n_tokens):
        if np.isnan(target_means[token_idx]):
            continue
        label = r"$\sigma$ (target)" if not target_label_added else None
        ax.plot(
            [x_positions[token_idx] - dash_half_width, x_positions[token_idx] + dash_half_width],
            [target_means[token_idx], target_means[token_idx]],
            color='black', linewidth=2, solid_capstyle='butt', label=label, zorder=3,
        )
        target_label_added = True

    # Alpha progression: lighter for earlier timesteps, darker for later
    alphas = np.linspace(0.25, 1.0, n_times)

    # Collect q means at each timestep, then draw dots + connecting lines
    # q_over_time[setting_idx][token_idx][time_i] = mean q log prob
    q_over_time = np.full((n_settings, n_tokens, n_times), np.nan)
    q_ci_lo_over_time = np.full((n_settings, n_tokens, n_times), np.nan)
    q_ci_hi_over_time = np.full((n_settings, n_tokens, n_times), np.nan)

    for time_i, t_idx in enumerate(frontier_indices):
        for setting_idx in range(n_settings):
            _, q_by_token = _collect_token_log_probs_at_timestep(
                results_list, setting_idx, t_idx, top_n_tokens)
            for token_idx, token_id in enumerate(top_n_tokens):
                q_vals = np.array(q_by_token[token_id])
                if len(q_vals) > 0:
                    m, lo, hi = _bootstrap_mean_ci(q_vals, n_bootstrap_draws)
                    q_over_time[setting_idx, token_idx, time_i] = m
                    q_ci_lo_over_time[setting_idx, token_idx, time_i] = lo
                    q_ci_hi_over_time[setting_idx, token_idx, time_i] = hi

    # Draw dots for each setting
    for setting_idx in range(n_settings):
        x_base = x_positions + setting_offsets[setting_idx]

        # Dots with CI at each timestep
        setting_label_added = False
        for time_i in range(n_times):
            means = q_over_time[setting_idx, :, time_i]
            lo = q_ci_lo_over_time[setting_idx, :, time_i]
            hi = q_ci_hi_over_time[setting_idx, :, time_i]
            lower_err = means - lo
            upper_err = hi - means
            valid = ~np.isnan(means)
            if not valid.any():
                continue

            label = labels[setting_idx] if not setting_label_added else None
            marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.errorbar(
                x_base[valid], means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt=marker, color=color_list[setting_idx], markersize=4,
                capsize=2, linewidth=0.8, alpha=alphas[time_i],
                label=label, zorder=4,
            )
            setting_label_added = True

    # Add timestep annotation
    time_str = ", ".join(str(idx) for idx in frontier_indices)
    ax.annotate(f"Timesteps: {time_str} (light→dark)", xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=fontsize - 1, color='gray')

    ax.set_xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Log q Over Time', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_maybe_hide_token_labels([_token_label(token_id, tokenizer) for token_id in top_n_tokens]), fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Over-time lollipop chart saved to {figname}")


def _collect_sample_counts_at_timestep(results_list, setting_idx, timestep_idx, token_ids):
    """Extract per-seed cumulative sample counts for given tokens at a specific timestep.

    Returns:
        dict of token_id -> list of per-seed count values.
    """
    counts_by_token = {tid: [] for tid in token_ids}

    tuple_list = results_list[setting_idx]
    if not tuple_list:
        return counts_by_token

    for t in tuple_list:
        if not isinstance(t, tuple) or len(t) < 3:
            continue
        metrics_list = t[2]
        if not isinstance(metrics_list, list) or len(metrics_list) == 0:
            continue
        if timestep_idx >= len(metrics_list):
            continue
        metrics_dict = metrics_list[timestep_idx]
        if not isinstance(metrics_dict, dict):
            continue
        counts = metrics_dict.get('cumulative_q_sample_counts', None)
        if counts is None:
            continue
        for tid in token_ids:
            if tid < len(counts):
                counts_by_token[tid].append(counts[tid].item())

    return counts_by_token


def plot_sample_counts_over_time(
    figname, labels, results_list,
    color_list, n_frontiers=4,
    fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
    marker_list=None,
):
    """
    Plot cumulative sample counts over time for the top tokens under the target distribution.

    Token selection is based on the final timestep (top N by target log prob).
    At each of n_frontiers evenly-spaced timesteps, dots show the mean cumulative count
    (across seeds) with bootstrap CIs. Alpha progresses from light (early) to dark (late).
    """
    # Select top tokens using final timestep target log prob
    if precomputed_token_data is not None:
        target_by_setting, _, _, all_sorted_tokens = precomputed_token_data
        top_n_tokens = all_sorted_tokens[:n_top_tokens]
    else:
        target_by_setting, _, _, top_n_tokens = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=True)
    if top_n_tokens is None:
        print("Warning: Cannot create sample counts over time plot.")
        return

    if not _has_cumulative_q_sample_counts(results_list):
        print("Warning: No cumulative_q_sample_counts found in data. Skipping counts plot.")
        return

    # Determine max trajectory length
    max_T = 0
    for setting_idx in range(len(labels)):
        for t in results_list[setting_idx]:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list):
                max_T = max(max_T, len(t[2]))

    if max_T == 0 or n_frontiers <= 0:
        return

    # Compute evenly-spaced timestep indices
    frontier_indices = [round((max_T - 1) * i / n_frontiers) for i in range(1, n_frontiers + 1)]
    seen = set()
    unique_frontier_indices = []
    for idx in frontier_indices:
        if idx not in seen:
            seen.add(idx)
            unique_frontier_indices.append(idx)
    frontier_indices = unique_frontier_indices
    n_times = len(frontier_indices)

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    fig, ax = plt.subplots(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    # Alpha progression
    alphas = np.linspace(0.25, 1.0, n_times)

    # Collect counts at each timestep
    counts_over_time = np.full((n_settings, n_tokens, n_times), np.nan)
    counts_ci_lo = np.full((n_settings, n_tokens, n_times), np.nan)
    counts_ci_hi = np.full((n_settings, n_tokens, n_times), np.nan)

    for time_i, t_idx in enumerate(frontier_indices):
        for setting_idx in range(n_settings):
            counts_by_token = _collect_sample_counts_at_timestep(
                results_list, setting_idx, t_idx, top_n_tokens)
            for token_idx, token_id in enumerate(top_n_tokens):
                vals = np.array(counts_by_token[token_id])
                if len(vals) > 0:
                    m, lo, hi = _bootstrap_mean_ci(vals, n_bootstrap_draws)
                    counts_over_time[setting_idx, token_idx, time_i] = m
                    counts_ci_lo[setting_idx, token_idx, time_i] = lo
                    counts_ci_hi[setting_idx, token_idx, time_i] = hi

    # Draw dots for each setting
    for setting_idx in range(n_settings):
        x_base = x_positions + setting_offsets[setting_idx]

        setting_label_added = False
        for time_i in range(n_times):
            means = counts_over_time[setting_idx, :, time_i]
            lo = counts_ci_lo[setting_idx, :, time_i]
            hi = counts_ci_hi[setting_idx, :, time_i]
            lower_err = means - lo
            upper_err = hi - means
            valid = ~np.isnan(means)
            if not valid.any():
                continue

            label = labels[setting_idx] if not setting_label_added else None
            marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.errorbar(
                x_base[valid], means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt=marker, color=color_list[setting_idx], markersize=4,
                capsize=2, linewidth=0.8, alpha=alphas[time_i],
                label=label, zorder=4,
            )
            setting_label_added = True

    # Timestep annotation
    time_str = ", ".join(str(idx) for idx in frontier_indices)
    ax.annotate(f"Timesteps: {time_str} (light\u2192dark)", xy=(0.02, 0.98),
                xycoords='axes fraction', fontsize=fontsize - 1, color='gray',
                verticalalignment='top')

    ax.set_xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
    ax.set_ylabel('Cumulative Sample Count', fontsize=fontsize)
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Cumulative q Sample Counts Over Time', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_maybe_hide_token_labels([_token_label(token_id, tokenizer) for token_id in top_n_tokens]), fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Sample counts over time chart saved to {figname}")


def plot_sample_counts_final_individual(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """
    Plot cumulative sample counts at the final timestep with individual seed points.

    Same token selection as plot_sample_counts_over_time (top N by target log prob at final step),
    but only shows the last timestep and plots each seed as a separate dot instead of
    bootstrap-aggregated means.
    """
    if precomputed_token_data is not None:
        _, _, _, all_sorted_tokens = precomputed_token_data
        top_n_tokens = all_sorted_tokens[:n_top_tokens]
    else:
        _, _, _, top_n_tokens = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=True)
    if top_n_tokens is None:
        print("Warning: Cannot create sample counts final individual plot.")
        return

    if not _has_cumulative_q_sample_counts(results_list):
        print("Warning: No cumulative_q_sample_counts found. Skipping sample counts final individual plot.")
        return

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    fig, ax = plt.subplots(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    for setting_idx in range(n_settings):
        tuple_list = results_list[setting_idx]
        if not tuple_list:
            continue

        # Collect per-seed, per-token counts: list of (seed_j, token_idx, count)
        seed_points = []
        for seed_j, t in enumerate(tuple_list):
            if not isinstance(t, tuple) or len(t) < 3:
                continue
            metrics_list = t[2]
            if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                continue
            last_metrics = metrics_list[-1]
            if not isinstance(last_metrics, dict):
                continue
            counts = last_metrics.get('cumulative_q_sample_counts', None)
            if counts is None:
                continue
            for token_idx, token_id in enumerate(top_n_tokens):
                if token_id < len(counts):
                    seed_points.append((seed_j, token_idx, counts[token_id].item()))

        if not seed_points:
            continue

        label_added = False
        for seed_j, token_idx, count_val in seed_points:
            x_base = x_positions[token_idx] + setting_offsets[setting_idx]
            marker = SEED_MARKERS[seed_j % len(SEED_MARKERS)]
            label = labels[setting_idx] if not label_added else None
            ax.scatter(
                x_base, count_val,
                color=color_list[setting_idx], s=20, alpha=0.7,
                marker=marker, label=label, zorder=4,
            )
            label_added = True

    ax.set_xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
    ax.set_ylabel('Cumulative Sample Count', fontsize=fontsize)
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Final Cumulative q Sample Counts (individual seeds)', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_maybe_hide_token_labels([_token_label(token_id, tokenizer) for token_id in top_n_tokens]), fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Sample counts final individual chart saved to {figname}")


def plot_coverage_curve(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
    precomputed_token_data=None,
    linestyle_list=None,
):
    """
    Plot the fraction of top-K target tokens discovered (sampled at least once by q)
    over training time.

    X-axis: evaluation step index.
    Y-axis: fraction of top-K target tokens with cumulative_q_sample_counts > 0.
    One line per setting with bootstrap CI shading across seeds.
    Coverage is monotonically non-decreasing since counts are cumulative.
    """
    # Select top tokens using final timestep target log prob
    if precomputed_token_data is not None:
        _, _, _, all_sorted_tokens = precomputed_token_data
        top_n_tokens = all_sorted_tokens[:n_top_tokens]
    else:
        _, _, _, top_n_tokens = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=True)
    if top_n_tokens is None:
        print("Warning: Cannot create coverage curve.")
        return

    # Check if any data has cumulative_q_sample_counts
    has_counts = False
    for setting_data in results_list:
        for t in setting_data:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list) and len(t[2]) > 0:
                last = t[2][-1]
                if isinstance(last, dict) and 'cumulative_q_sample_counts' in last:
                    has_counts = True
                    break
        if has_counts:
            break
    if not has_counts:
        print("Warning: No cumulative_q_sample_counts found. Skipping coverage curve.")
        return

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    fig, ax = plt.subplots()

    for setting_idx in range(n_settings):
        tuple_list = results_list[setting_idx]
        if not tuple_list:
            continue

        # For each seed, compute coverage at each evaluation step
        seed_curves = []
        for t in tuple_list:
            if not isinstance(t, tuple) or len(t) < 3:
                continue
            metrics_list = t[2]
            if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                continue

            curve = []
            for metrics_dict in metrics_list:
                if not isinstance(metrics_dict, dict):
                    curve.append(np.nan)
                    continue
                counts = metrics_dict.get('cumulative_q_sample_counts', None)
                if counts is None:
                    curve.append(np.nan)
                    continue
                n_discovered = sum(1 for tid in top_n_tokens if tid < len(counts) and counts[tid].item() > 0)
                curve.append(n_discovered / n_tokens)
            seed_curves.append(curve)

        if not seed_curves:
            continue

        # Pad shorter curves with their last value (seeds may have different lengths)
        max_len = max(len(c) for c in seed_curves)
        for c in seed_curves:
            last_val = c[-1] if c else np.nan
            while len(c) < max_len:
                c.append(last_val)

        # Bootstrap CI at each timestep
        timesteps = np.arange(max_len)
        means = np.full(max_len, np.nan)
        ci_lo = np.full(max_len, np.nan)
        ci_hi = np.full(max_len, np.nan)

        for t_idx in range(max_len):
            vals = np.array([c[t_idx] for c in seed_curves])
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                means[t_idx], ci_lo[t_idx], ci_hi[t_idx] = _bootstrap_mean_ci(valid_vals, n_bootstrap_draws)

        valid = ~np.isnan(means)
        if valid.any():
            ls = linestyle_list[setting_idx] if linestyle_list is not None else 'solid'
            ax.plot(timesteps[valid], means[valid], color=color_list[setting_idx],
                    label=labels[setting_idx], linewidth=1.5, linestyle=ls)
            ax.fill_between(timesteps[valid], ci_lo[valid], ci_hi[valid],
                            color=color_list[setting_idx], alpha=0.15)

    ax.set_xlabel('Evaluation Step', fontsize=fontsize)
    ax.set_ylabel(f'Fraction of Top-{n_top_tokens} Target Tokens Discovered', fontsize=fontsize)
    ax.set_title(f'Coverage: Fraction of Top-{n_top_tokens} Target Tokens Sampled by q', fontsize=fontsize + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Coverage curve saved to {figname}")


def plot_vocab_coverage_curve(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    linestyle_list=None,
):
    """
    Plot the fraction of ALL vocab tokens discovered (sampled at least once by q)
    over training time.

    X-axis: evaluation step index.
    Y-axis: fraction of vocab tokens with cumulative_q_sample_counts > 0.
    One line per setting with bootstrap CI shading across seeds.
    """
    # Check if any data has cumulative_q_sample_counts and determine n_vocab
    n_vocab = None
    has_counts = False
    for setting_data in results_list:
        for t in setting_data:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list) and len(t[2]) > 0:
                last = t[2][-1]
                if isinstance(last, dict) and 'cumulative_q_sample_counts' in last:
                    has_counts = True
                    n_vocab = len(last['cumulative_q_sample_counts'])
                    break
        if has_counts:
            break
    if not has_counts or n_vocab is None:
        print("Warning: No cumulative_q_sample_counts found. Skipping vocab coverage curve.")
        return

    n_settings = len(labels)

    fig, ax = plt.subplots()

    for setting_idx in range(n_settings):
        tuple_list = results_list[setting_idx]
        if not tuple_list:
            continue

        seed_curves = []
        for t in tuple_list:
            if not isinstance(t, tuple) or len(t) < 3:
                continue
            metrics_list = t[2]
            if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                continue

            curve = []
            for metrics_dict in metrics_list:
                if not isinstance(metrics_dict, dict):
                    curve.append(np.nan)
                    continue
                counts = metrics_dict.get('cumulative_q_sample_counts', None)
                if counts is None:
                    curve.append(np.nan)
                    continue
                n_discovered = (counts > 0).sum().item()
                curve.append(n_discovered / n_vocab)
            seed_curves.append(curve)

        if not seed_curves:
            continue

        # Pad shorter curves with their last value
        max_len = max(len(c) for c in seed_curves)
        for c in seed_curves:
            last_val = c[-1] if c else np.nan
            while len(c) < max_len:
                c.append(last_val)

        timesteps = np.arange(max_len)
        means = np.full(max_len, np.nan)
        ci_lo = np.full(max_len, np.nan)
        ci_hi = np.full(max_len, np.nan)

        for t_idx in range(max_len):
            vals = np.array([c[t_idx] for c in seed_curves])
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                means[t_idx], ci_lo[t_idx], ci_hi[t_idx] = _bootstrap_mean_ci(valid_vals, n_bootstrap_draws)

        valid = ~np.isnan(means)
        if valid.any():
            ls = linestyle_list[setting_idx] if linestyle_list is not None else 'solid'
            ax.plot(timesteps[valid], means[valid], color=color_list[setting_idx],
                    label=labels[setting_idx], linewidth=1.5, linestyle=ls)
            ax.fill_between(timesteps[valid], ci_lo[valid], ci_hi[valid],
                            color=color_list[setting_idx], alpha=0.15)

    ax.set_xlabel('Evaluation Step', fontsize=fontsize)
    ax.set_ylabel('Fraction of Vocab Tokens Discovered', fontsize=fontsize)
    ax.set_title(f'Vocab Coverage: Fraction of All Tokens Sampled by q (n_vocab={n_vocab})', fontsize=fontsize + 1)
    # Dynamic y-axis: scale to data range with a small margin
    y_lo, y_hi = ax.get_ylim()
    margin = (y_hi - y_lo) * 0.05 if y_hi > y_lo else 0.05
    ax.set_ylim(max(0, y_lo - margin), y_hi + margin)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Vocab coverage curve saved to {figname}")


def _extract_target_token_ids(target_samples_path):
    """Load target samples file and extract unique token IDs sorted by descending frequency.

    Handles both v1 (list of lists) and v2 (dict with samples_by_prompt) formats.

    Args:
        target_samples_path: Path to the target samples .pt file.

    Returns:
        (sorted_token_ids, token_counts): tuple of
            sorted_token_ids: 1D numpy array of unique token IDs, sorted by descending count
            token_counts: dict mapping token_id -> count in target sequences
    """
    raw = torch.load(target_samples_path, map_location='cpu')

    # Collect all token IDs from all sequences
    all_tokens = []
    if isinstance(raw, dict) and raw.get("version", 1) >= 2:
        for samples_list in raw["samples_by_prompt"]:
            for seq in samples_list:
                all_tokens.extend(seq if isinstance(seq, list) else seq.tolist())
    elif isinstance(raw, list):
        for prompt_samples in raw:
            for seq in prompt_samples:
                all_tokens.extend(seq if isinstance(seq, list) else seq.tolist())
    else:
        raise ValueError(f"Unexpected target samples format: {type(raw)}")

    # Count occurrences of each token
    token_counts = Counter(all_tokens)

    # Sort by descending count
    sorted_token_ids = np.array([tid for tid, _ in token_counts.most_common()])
    print(f"Target sequences contain {len(sorted_token_ids)} unique token IDs "
          f"(total tokens: {len(all_tokens)}, top-5 counts: "
          f"{[token_counts[tid] for tid in sorted_token_ids[:5]]})")

    return sorted_token_ids, dict(token_counts)


def plot_vocab_coverage_from_history(
    figname, labels, counts_results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    marker_list=None, linestyle_list=None,
):
    """Plot fraction of all vocab tokens discovered over time from token_counts_history files.

    Args:
        counts_results_list: List (settings) of list (seeds) of list-of-tensors
            (one (n_vocab,) tensor per fit_step).
        marker_list: Optional list of marker styles, one per setting.
        linestyle_list: Optional list of linestyles, one per setting.
    """
    # Determine n_vocab from first available data
    n_vocab = None
    for setting_data in counts_results_list:
        for seed_data in setting_data:
            if isinstance(seed_data, list) and len(seed_data) > 0:
                n_vocab = len(seed_data[0])
                break
        if n_vocab is not None:
            break
    if n_vocab is None:
        print(f"No token counts data found, skipping {figname}")
        return

    n_settings = len(labels)
    fig, ax = plt.subplots()

    for setting_idx in range(n_settings):
        setting_data = counts_results_list[setting_idx]
        if not setting_data:
            continue

        seed_curves = []
        for seed_data in setting_data:
            if not isinstance(seed_data, list) or len(seed_data) == 0:
                continue
            curve = []
            for counts_tensor in seed_data:
                n_discovered = (counts_tensor > 0).sum().item()
                curve.append(n_discovered / n_vocab)
            seed_curves.append(curve)

        if not seed_curves:
            continue

        # Pad shorter curves with their last value
        max_len = max(len(c) for c in seed_curves)
        for c in seed_curves:
            last_val = c[-1] if c else np.nan
            while len(c) < max_len:
                c.append(last_val)

        timesteps = np.arange(max_len)
        means = np.full(max_len, np.nan)
        ci_lo = np.full(max_len, np.nan)
        ci_hi = np.full(max_len, np.nan)

        for t_idx in range(max_len):
            vals = np.array([c[t_idx] for c in seed_curves])
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                means[t_idx], ci_lo[t_idx], ci_hi[t_idx] = _bootstrap_mean_ci(valid_vals, n_bootstrap_draws)

        valid = ~np.isnan(means)
        if valid.any():
            marker = marker_list[setting_idx] if marker_list is not None else None
            ls = linestyle_list[setting_idx] if linestyle_list is not None else 'solid'
            ax.plot(timesteps[valid], means[valid], color=color_list[setting_idx],
                    label=labels[setting_idx], linewidth=1.5,
                    marker=marker, markersize=4, markevery=max(1, len(timesteps[valid]) // 10),
                    linestyle=ls)
            ax.fill_between(timesteps[valid], ci_lo[valid], ci_hi[valid],
                            color=color_list[setting_idx], alpha=0.15)

    ax.set_xlabel('Fit Step', fontsize=fontsize)
    ax.set_ylabel('Fraction of Vocab Tokens Discovered', fontsize=fontsize)
    ax.set_title(f'Vocab Coverage: Fraction of All Tokens Sampled by q (n_vocab={n_vocab})', fontsize=fontsize + 1)
    y_lo, y_hi = ax.get_ylim()
    margin = (y_hi - y_lo) * 0.05 if y_hi > y_lo else 0.05
    ax.set_ylim(max(0, y_lo - margin), y_hi + margin)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Vocab coverage curve (from history) saved to {figname}")


def plot_target_token_counts_over_time(
    figname, labels, counts_results_list,
    target_token_ids, color_list,
    n_frontiers=4, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=None,
    tokenizer=None,
    marker_list=None,
):
    """Plot cumulative q-sample counts over time for tokens from target sequences.

    Analog of plot_sample_counts_over_time but using token_counts_history files.
    Tokens are sorted by their frequency in the target sequences (descending).

    Args:
        counts_results_list: List (settings) of list (seeds) of list-of-tensors
            (one (n_vocab,) tensor per fit_step).
        target_token_ids: 1D numpy array of unique token IDs sorted by descending
            target-sequence frequency.
        n_top_tokens: If set, only show the top N most frequent target tokens.
    """
    if n_top_tokens is not None:
        target_token_ids = target_token_ids[:n_top_tokens]

    n_tokens = len(target_token_ids)
    if n_tokens == 0:
        print(f"No target tokens, skipping {figname}")
        return

    n_settings = len(labels)

    # Determine max trajectory length
    max_T = 0
    for setting_data in counts_results_list:
        for seed_data in setting_data:
            if isinstance(seed_data, list):
                max_T = max(max_T, len(seed_data))

    if max_T == 0 or n_frontiers <= 0:
        return

    # Evenly-spaced timestep indices
    frontier_indices = [round((max_T - 1) * i / n_frontiers) for i in range(1, n_frontiers + 1)]
    seen = set()
    unique_frontier_indices = []
    for idx in frontier_indices:
        if idx not in seen:
            seen.add(idx)
            unique_frontier_indices.append(idx)
    frontier_indices = unique_frontier_indices
    n_times = len(frontier_indices)

    fig, ax = plt.subplots(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    alphas = np.linspace(0.25, 1.0, n_times)

    # Collect counts at each timestep
    counts_over_time = np.full((n_settings, n_tokens, n_times), np.nan)
    counts_ci_lo = np.full((n_settings, n_tokens, n_times), np.nan)
    counts_ci_hi = np.full((n_settings, n_tokens, n_times), np.nan)

    for time_i, t_idx in enumerate(frontier_indices):
        for setting_idx in range(n_settings):
            setting_data = counts_results_list[setting_idx]
            # Collect per-seed counts for each target token at this timestep
            per_token_seeds = {tid: [] for tid in target_token_ids}
            for seed_data in setting_data:
                if not isinstance(seed_data, list) or t_idx >= len(seed_data):
                    continue
                counts_tensor = seed_data[t_idx]
                for tid in target_token_ids:
                    if tid < len(counts_tensor):
                        per_token_seeds[tid].append(counts_tensor[tid].item())

            for token_idx, tid in enumerate(target_token_ids):
                vals = np.array(per_token_seeds[tid])
                if len(vals) > 0:
                    m, lo, hi = _bootstrap_mean_ci(vals, n_bootstrap_draws)
                    counts_over_time[setting_idx, token_idx, time_i] = m
                    counts_ci_lo[setting_idx, token_idx, time_i] = lo
                    counts_ci_hi[setting_idx, token_idx, time_i] = hi

    # Draw dots for each setting
    for setting_idx in range(n_settings):
        x_base = x_positions + setting_offsets[setting_idx]
        color = color_list[setting_idx]

        setting_label_added = False
        for time_i in range(n_times):
            means_t = counts_over_time[setting_idx, :, time_i]
            lo_t = counts_ci_lo[setting_idx, :, time_i]
            hi_t = counts_ci_hi[setting_idx, :, time_i]
            lower_err = means_t - lo_t
            upper_err = hi_t - means_t
            valid = ~np.isnan(means_t)
            if not valid.any():
                continue

            label = labels[setting_idx] if not setting_label_added else None
            marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.scatter(x_base[valid], means_t[valid], color=color, s=16, zorder=4,
                       alpha=alphas[time_i], label=label, marker=marker)
            ax.errorbar(x_base[valid], means_t[valid],
                        yerr=[lower_err[valid], upper_err[valid]],
                        fmt='', ecolor=color, alpha=alphas[time_i] * 0.2,
                        capsize=2, linewidth=0.8, zorder=3)
            setting_label_added = True

    # Timestep annotation
    time_str = ", ".join(str(idx) for idx in frontier_indices)
    ax.annotate(f"Fit steps: {time_str} (light\u2192dark)", xy=(0.02, 0.98),
                xycoords='axes fraction', fontsize=fontsize - 1, color='gray',
                verticalalignment='top')

    ax.set_xlabel('Token (sorted by target-sequence frequency)' if tokenizer is not None else
                  'Token ID (sorted by target-sequence frequency)', fontsize=fontsize)
    ax.set_ylabel('Cumulative q-Sample Count', fontsize=fontsize)
    ax.set_title(f'Target Token Visitation: Cumulative q Sample Counts Over Time ({n_tokens} tokens)',
                 fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_maybe_hide_token_labels([_token_label(int(tid), tokenizer) for tid in target_token_ids]),
                       fontsize=max(3, fontsize - 2), rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Target token counts over time chart saved to {figname}")


def plot_target_token_counts_final_individual(
    figname, labels, counts_results_list,
    target_token_ids, color_list,
    fontsize=7, legendfontsize=7,
    n_top_tokens=None,
    tokenizer=None,
):
    """Plot cumulative q-sample counts at final timestep with individual seed markers.

    Analog of plot_sample_counts_final_individual but using token_counts_history files.

    Args:
        counts_results_list: List (settings) of list (seeds) of list-of-tensors.
        target_token_ids: 1D numpy array of unique token IDs sorted by descending
            target-sequence frequency.
        n_top_tokens: If set, only show the top N most frequent target tokens.
    """
    if n_top_tokens is not None:
        target_token_ids = target_token_ids[:n_top_tokens]

    n_tokens = len(target_token_ids)
    if n_tokens == 0:
        print(f"No target tokens, skipping {figname}")
        return

    n_settings = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, n_tokens * 0.675), _scale_h(5)))

    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    for setting_idx in range(n_settings):
        setting_data = counts_results_list[setting_idx]
        if not setting_data:
            continue

        label_added = False
        for seed_j, seed_data in enumerate(setting_data):
            if not isinstance(seed_data, list) or len(seed_data) == 0:
                continue
            # Final timestep
            final_counts = seed_data[-1]
            marker = SEED_MARKERS[seed_j % len(SEED_MARKERS)]
            color = color_list[setting_idx]

            for token_idx, tid in enumerate(target_token_ids):
                if tid >= len(final_counts):
                    continue
                x_base = x_positions[token_idx] + setting_offsets[setting_idx]
                label = labels[setting_idx] if not label_added else None
                ax.scatter(x_base, final_counts[tid].item(),
                           color=color, s=20, alpha=0.7,
                           marker=marker, label=label, zorder=4)
                label_added = True

    ax.set_xlabel('Token (sorted by target-sequence frequency)' if tokenizer is not None else
                  'Token ID (sorted by target-sequence frequency)', fontsize=fontsize)
    ax.set_ylabel('Cumulative q-Sample Count', fontsize=fontsize)
    ax.set_title(f'Target Token Visitation: Final Cumulative q Sample Counts (individual seeds, {n_tokens} tokens)',
                 fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(_maybe_hide_token_labels([_token_label(int(tid), tokenizer) for tid in target_token_ids]),
                       fontsize=max(3, fontsize - 2), rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Target token counts final individual chart saved to {figname}")


def _plot_visitation_2d(
    figname_prefix, labels, results_list,
    coords, color_list,
    coord_axis_labels, method_name,
    fontsize=7, legendfontsize=7,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """
    Shared implementation for 2D visitation scatter plots (PCA or t-SNE).

    For each setting, produces a 2D scatter plot showing:
      1. Target distribution density as yellow dots with alpha proportional to target probability.
      2. Visited tokens as circles with size/darkness proportional to cumulative visitation count.
      3. Top-N target distribution tokens as stars with darkness proportional to target probability.

    One PDF per setting: {figname_prefix}_setting{idx}.pdf

    Args:
        coords: (vocab_size, 2) numpy array of 2D coordinates per token.
        coord_axis_labels: Pair of strings for x and y axis labels, e.g. ("PC1", "PC2").
        method_name: Short name for the method, e.g. "PCA" or "t-SNE" (used in title and print).
    """
    # Get top target tokens and their log probs (precomputed with n_top_tokens=999999 covers all tokens)
    if precomputed_token_data is not None:
        target_by_setting, _, _, all_sorted_tokens = precomputed_token_data
        top_n_tokens_list = all_sorted_tokens[:n_top_tokens]
    else:
        target_by_setting, _, _, top_n_tokens_list = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=True)
    if top_n_tokens_list is None:
        print("Warning: No token data for PCA visitation plot.")
        return

    # Compute average target log prob for ALL tokens (across all settings and seeds)
    all_token_target_log_probs = {}
    for token_id, setting_dict in target_by_setting.items():
        all_values = []
        for values in setting_dict.values():
            all_values.extend(values)
        if all_values:
            all_token_target_log_probs[token_id] = np.mean(all_values)

    # Build per-token alpha for the top-N target tokens (for the star overlay)
    top_token_target_log_probs = {t: all_token_target_log_probs[t]
                                  for t in top_n_tokens_list if t in all_token_target_log_probs}
    target_alphas = {}
    if top_token_target_log_probs:
        log_probs_arr = np.array([top_token_target_log_probs[t] for t in top_n_tokens_list
                                  if t in top_token_target_log_probs])
        # Shift so max is 0 for numerical stability, then exponentiate
        probs_arr = np.exp(log_probs_arr - log_probs_arr.max())
        # Normalize to [0.2, 1.0] for alpha (so even the lowest-prob token is visible)
        if probs_arr.max() > probs_arr.min():
            alpha_arr = 0.2 + 0.8 * (probs_arr - probs_arr.min()) / (probs_arr.max() - probs_arr.min())
        else:
            alpha_arr = np.ones_like(probs_arr)
        idx = 0
        for token_id in top_n_tokens_list:
            if token_id in top_token_target_log_probs:
                target_alphas[token_id] = alpha_arr[idx]
                idx += 1

    # Check for cumulative_q_sample_counts
    n_vocab = None
    has_counts = False
    for setting_data in results_list:
        for t in setting_data:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list) and len(t[2]) > 0:
                last = t[2][-1]
                if isinstance(last, dict) and 'cumulative_q_sample_counts' in last:
                    has_counts = True
                    n_vocab = len(last['cumulative_q_sample_counts'])
                    break
        if has_counts:
            break
    if not has_counts or n_vocab is None:
        print(f"Warning: No cumulative_q_sample_counts found. Skipping {method_name} visitation plot.")
        return

    assert coords.shape[0] >= n_vocab, (
        f"{method_name} coords have {coords.shape[0]} tokens but data has {n_vocab} vocab entries"
    )

    n_settings = len(labels)

    for setting_idx in range(n_settings):
        tuple_list = results_list[setting_idx]
        if not tuple_list:
            continue

        # Average final cumulative counts across seeds
        seed_counts = []
        for t in tuple_list:
            if not isinstance(t, tuple) or len(t) < 3:
                continue
            metrics_list = t[2]
            if not isinstance(metrics_list, list) or len(metrics_list) == 0:
                continue
            last_metrics = metrics_list[-1]
            if not isinstance(last_metrics, dict):
                continue
            counts = last_metrics.get('cumulative_q_sample_counts', None)
            if counts is None:
                continue
            counts_np = np.array([counts[i].item() if hasattr(counts[i], 'item') else float(counts[i])
                                  for i in range(min(len(counts), n_vocab))])
            if len(counts_np) < n_vocab:
                counts_np = np.concatenate([counts_np, np.zeros(n_vocab - len(counts_np))])
            seed_counts.append(counts_np)

        if not seed_counts:
            continue

        avg_counts = np.mean(seed_counts, axis=0)

        # --- Build the figure ---
        fig, ax = plt.subplots(1, 1, figsize=(10, _scale_h(8)))

        # 1. Target distribution density as yellow dots with alpha proportional to probability
        # Build a full-vocab log prob array; tokens not in target_by_setting get -inf
        bg_log_probs = np.full(n_vocab, -np.inf)
        for token_id, lp in all_token_target_log_probs.items():
            if token_id < n_vocab:
                bg_log_probs[token_id] = lp
        # Convert to relative probabilities (shift max to 0, exponentiate)
        finite_mask = np.isfinite(bg_log_probs)
        if finite_mask.any():
            bg_max = bg_log_probs[finite_mask].max()
            bg_probs = np.where(finite_mask, np.exp(bg_log_probs - bg_max), 0.0)
            # Map to alpha in [0.02, 0.7] using log scale for better dynamic range
            bg_log_scaled = np.where(bg_probs > 0, np.log1p(bg_probs * 1000), 0.0)  # log(1 + p*1000)
            bg_log_max = bg_log_scaled.max()
            if bg_log_max > 0:
                bg_alpha = 0.02 + 0.68 * (bg_log_scaled / bg_log_max)
            else:
                bg_alpha = np.full(n_vocab, 0.02)
            base_yellow = np.array(mcolors.to_rgba('#FFD700'))  # gold yellow
            bg_rgba = np.tile(base_yellow, (n_vocab, 1))
            bg_rgba[:, 3] = bg_alpha
            ax.scatter(coords[:n_vocab, 0], coords[:n_vocab, 1],
                       s=3, c=bg_rgba, zorder=1, rasterized=True)
            # Proxy artist for legend with full-opacity gold
            ax.scatter([], [], s=15, c='#FFD700', alpha=1.0, label=r'$\sigma$ density')
        else:
            # Fallback: uniform faint yellow if no target data
            ax.scatter(coords[:n_vocab, 0], coords[:n_vocab, 1],
                       s=1, c='#FFFACD', alpha=0.05, zorder=1, rasterized=True)

        # 2. Visited tokens: fixed size, alpha via compositing formula so overlapping
        #    dots visually sum — k visits looks the same as k overlapping single-visit dots.
        #    alpha(k) = 1 - (1 - alpha_base)^k, with alpha_base chosen so max count -> ~0.9
        visited_mask = avg_counts > 0
        if visited_mask.any():
            visited_ids = np.where(visited_mask)[0]
            visited_counts = avg_counts[visited_ids]
            max_count = visited_counts.max()

            # Choose alpha_base so that max_count maps to ~0.9 opacity
            # alpha_base = 1 - (1 - 0.9)^(1/max_count) = 1 - 0.1^(1/max_count)
            alpha_base = 1.0 - 0.1 ** (1.0 / max_count) if max_count > 0 else 0.5
            alphas = 1.0 - (1.0 - alpha_base) ** visited_counts
            # Rescale from [0, 1] to [0.15, 1] so even single-visit tokens are visible
            alphas = 0.15 + 0.85 * alphas

            setting_color = color_list[setting_idx] if setting_idx < len(color_list) else 'blue'
            base_rgba = np.array(mcolors.to_rgba(setting_color))
            rgba_array = np.tile(base_rgba, (len(visited_ids), 1))
            rgba_array[:, 3] = alphas

            ax.scatter(coords[visited_ids, 0], coords[visited_ids, 1],
                       s=10, c=rgba_array, marker='o', zorder=2,
                       edgecolors='none', linewidths=0,
                       label=f'q visits ({labels[setting_idx]})', rasterized=True)

        # 3. Top target tokens as stars
        top_ids_in_range = [t for t in top_n_tokens_list if t < n_vocab and t in target_alphas]
        if top_ids_in_range:
            top_coords = coords[top_ids_in_range]
            top_alpha_vals = np.array([target_alphas[t] for t in top_ids_in_range])

            base_rgba_target = np.array(mcolors.to_rgba('red'))
            rgba_target = np.tile(base_rgba_target, (len(top_ids_in_range), 1))
            rgba_target[:, 3] = top_alpha_vals

            ax.scatter(top_coords[:, 0], top_coords[:, 1],
                       s=80, c=rgba_target, marker='*', zorder=3,
                       edgecolors='darkred', linewidths=0.3,
                       label=f'Top {n_top_tokens} target tokens')

            # Annotate top tokens with their decoded string
            for i, token_id in enumerate(top_ids_in_range):
                token_str = _token_label(token_id, tokenizer)
                ax.annotate(token_str, (top_coords[i, 0], top_coords[i, 1]),
                            fontsize=max(fontsize - 2, 4), color='darkred',
                            textcoords='offset points', xytext=(4, 4),
                            alpha=float(top_alpha_vals[i]),
                            zorder=4)

        ax.set_xlabel(coord_axis_labels[0], fontsize=fontsize)
        ax.set_ylabel(coord_axis_labels[1], fontsize=fontsize)
        ax.set_title(f'Token Visitation in {method_name} Embedding Space: {labels[setting_idx]}', fontsize=fontsize + 1)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.legend(fontsize=legendfontsize, loc='best')
        plt.tight_layout()

        figname = f"{figname_prefix}_setting{setting_idx}.pdf"
        plt.savefig(figname, dpi=150)
        plt.clf()
        plt.close(fig)
        print(f"{method_name} visitation plot saved to {figname}")


def plot_visitation_pca(
    figname_prefix, labels, results_list,
    pca_coords, color_list,
    fontsize=7, legendfontsize=7,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """Wrapper around _plot_visitation_2d for PCA coordinates."""
    _plot_visitation_2d(
        figname_prefix=figname_prefix, labels=labels, results_list=results_list,
        coords=pca_coords, color_list=color_list,
        coord_axis_labels=("PC1", "PC2"), method_name="PCA",
        fontsize=fontsize, legendfontsize=legendfontsize,
        n_top_tokens=n_top_tokens, precomputed_token_data=precomputed_token_data,
        tokenizer=tokenizer,
    )


def plot_visitation_tsne(
    figname_prefix, labels, results_list,
    tsne_coords, color_list,
    fontsize=7, legendfontsize=7,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """Wrapper around _plot_visitation_2d for t-SNE coordinates."""
    _plot_visitation_2d(
        figname_prefix=figname_prefix, labels=labels, results_list=results_list,
        coords=tsne_coords, color_list=color_list,
        coord_axis_labels=("t-SNE 1", "t-SNE 2"), method_name="t-SNE",
        fontsize=fontsize, legendfontsize=legendfontsize,
        n_top_tokens=n_top_tokens, precomputed_token_data=precomputed_token_data,
        tokenizer=tokenizer,
    )


def plot_visitation_heatmaps(
    figname_prefix, labels, results_list,
    n_frontiers=4,
    fontsize=7,
):
    """
    For each setting, produce a figure with (n_frontiers + 1) heatmap subplots showing
    token visitation counts on a 2D grid (reshaped to roughly sqrt(n_vocab) x sqrt(n_vocab)).

    The first n_frontiers subplots show incremental counts (new visits since the previous
    frontier timestep). The final subplot shows the cumulative count at the last timestep.
    Counts are averaged across seeds.

    One PDF is saved per setting: {figname_prefix}_setting{idx}.pdf
    """
    import math

    # Check if any data has cumulative_q_sample_counts and determine n_vocab
    n_vocab = None
    has_counts = False
    for setting_data in results_list:
        for t in setting_data:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list) and len(t[2]) > 0:
                last = t[2][-1]
                if isinstance(last, dict) and 'cumulative_q_sample_counts' in last:
                    has_counts = True
                    n_vocab = len(last['cumulative_q_sample_counts'])
                    break
        if has_counts:
            break
    if not has_counts or n_vocab is None:
        print("Warning: No cumulative_q_sample_counts found. Skipping visitation heatmaps.")
        return

    # Determine max trajectory length
    max_T = 0
    for setting_idx in range(len(labels)):
        for t in results_list[setting_idx]:
            if isinstance(t, tuple) and len(t) >= 3 and isinstance(t[2], list):
                max_T = max(max_T, len(t[2]))

    if max_T == 0 or n_frontiers <= 0:
        return

    # Compute evenly-spaced frontier timestep indices
    frontier_indices = [round((max_T - 1) * i / n_frontiers) for i in range(1, n_frontiers + 1)]
    seen = set()
    unique_frontier_indices = []
    for idx in frontier_indices:
        if idx not in seen:
            seen.add(idx)
            unique_frontier_indices.append(idx)
    frontier_indices = unique_frontier_indices
    n_times = len(frontier_indices)

    # Grid dimensions: reshape n_vocab into roughly square grid
    grid_h = int(math.ceil(math.sqrt(n_vocab)))
    grid_w = int(math.ceil(n_vocab / grid_h))
    n_pad = grid_h * grid_w - n_vocab  # tokens to pad with NaN

    n_settings = len(labels)

    for setting_idx in range(n_settings):
        tuple_list = results_list[setting_idx]
        if not tuple_list:
            continue

        # Collect cumulative counts at each frontier timestep, averaged across seeds
        # Also need counts at one step before the first frontier for the first diff
        # We use timestep 0 as the "previous" for the first frontier
        # Timesteps to collect: [0] + frontier_indices (but 0 may overlap with first frontier)
        all_timesteps = [0] + frontier_indices
        # Remove duplicates while preserving order
        seen_ts = set()
        unique_all_timesteps = []
        for ts in all_timesteps:
            if ts not in seen_ts:
                seen_ts.add(ts)
                unique_all_timesteps.append(ts)
        all_timesteps = unique_all_timesteps

        # avg_counts[ts] = average count vector across seeds at timestep ts
        avg_counts = {}
        for ts in all_timesteps:
            seed_counts = []
            for t in tuple_list:
                if not isinstance(t, tuple) or len(t) < 3:
                    continue
                metrics_list = t[2]
                if not isinstance(metrics_list, list) or ts >= len(metrics_list):
                    continue
                metrics_dict = metrics_list[ts]
                if not isinstance(metrics_dict, dict):
                    continue
                counts = metrics_dict.get('cumulative_q_sample_counts', None)
                if counts is None:
                    continue
                seed_counts.append(np.array([counts[i].item() for i in range(min(len(counts), n_vocab))]))
            if seed_counts:
                # Pad to n_vocab if needed
                padded = []
                for sc in seed_counts:
                    if len(sc) < n_vocab:
                        sc = np.concatenate([sc, np.zeros(n_vocab - len(sc))])
                    padded.append(sc)
                avg_counts[ts] = np.mean(padded, axis=0)

        if not avg_counts:
            continue

        # Build incremental diffs
        diff_grids = []
        prev_ts = 0
        for time_i, ts in enumerate(frontier_indices):
            curr = avg_counts.get(ts, np.zeros(n_vocab))
            prev = avg_counts.get(prev_ts, np.zeros(n_vocab)) if ts != prev_ts else np.zeros(n_vocab)
            diff = curr - prev
            diff_grids.append((diff, ts, prev_ts))
            prev_ts = ts

        # Cumulative counts at the final timestep
        final_ts = frontier_indices[-1]
        cumulative = avg_counts.get(final_ts, np.zeros(n_vocab))

        # Colormap: light gray (count >= 1) to black (max count).
        # Zero-count cells are masked and shown as white, giving a clear visual
        # boundary between "never visited" (white) and "visited at least once" (light gray).
        from matplotlib.colors import LinearSegmentedColormap
        gray_colors = plt.cm.Greys(np.linspace(0.2, 1.0, 256))
        cmap = LinearSegmentedColormap.from_list('gray_nonzero', gray_colors)
        cmap.set_bad(color='white')

        def _mask_zeros(data):
            """Return a masked array where zeros (and NaNs) are masked."""
            masked = np.ma.array(data, mask=(data == 0) | np.isnan(data))
            return masked

        # --- Incremental (new visitations) PDF ---
        n_diff_panels = len(diff_grids)
        fig_diff, axes_diff = plt.subplots(1, n_diff_panels, figsize=(3.5 * n_diff_panels, _scale_h(3.5)))
        if n_diff_panels == 1:
            axes_diff = [axes_diff]

        diff_max = max(np.max(d[0]) for d in diff_grids) if diff_grids else 1
        if diff_max == 0:
            diff_max = 1

        for panel_i, (data, ts, prev_ts_val) in enumerate(diff_grids):
            ax = axes_diff[panel_i]
            padded_data = np.concatenate([data, np.full(n_pad, np.nan)]) if n_pad > 0 else data.copy()
            grid = _mask_zeros(padded_data.reshape(grid_h, grid_w))

            im = ax.imshow(grid, cmap=cmap, aspect='equal', vmin=1, vmax=diff_max,
                           interpolation='nearest')

            if prev_ts_val == 0 and panel_i == 0:
                ax.set_title(f'Steps 0\u2013{ts}', fontsize=fontsize)
            else:
                ax.set_title(f'Steps {prev_ts_val}\u2013{ts}', fontsize=fontsize)

            ax.set_xticks([])
            ax.set_yticks([])
            fig_diff.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig_diff.suptitle(f'New Token Visitations \u2014 {labels[setting_idx]}', fontsize=fontsize + 2)
        plt.tight_layout()

        diff_figname = f"{figname_prefix}_incremental_setting{setting_idx}.pdf"
        plt.savefig(diff_figname)
        plt.clf()
        plt.close(fig_diff)
        print(f"Incremental visitation heatmap saved to {diff_figname}")

        # --- Cumulative PDF ---
        fig_cum, ax_cum = plt.subplots(figsize=(4, _scale_h(4)))

        cum_max = np.max(cumulative) if np.max(cumulative) > 0 else 1
        padded_cum = np.concatenate([cumulative, np.full(n_pad, np.nan)]) if n_pad > 0 else cumulative.copy()
        grid_cum = _mask_zeros(padded_cum.reshape(grid_h, grid_w))

        im_cum = ax_cum.imshow(grid_cum, cmap=cmap, aspect='equal', vmin=1, vmax=cum_max,
                               interpolation='nearest')
        ax_cum.set_title(f'Cumulative (step {final_ts})', fontsize=fontsize)
        ax_cum.set_xticks([])
        ax_cum.set_yticks([])
        fig_cum.colorbar(im_cum, ax=ax_cum, fraction=0.046, pad=0.04)

        fig_cum.suptitle(f'Cumulative Token Visitations \u2014 {labels[setting_idx]}', fontsize=fontsize + 2)
        plt.tight_layout()

        cum_figname = f"{figname_prefix}_cumulative_setting{setting_idx}.pdf"
        plt.savefig(cum_figname)
        plt.clf()
        plt.close(fig_cum)
        print(f"Cumulative visitation heatmap saved to {cum_figname}")


def _format_intersection_lollipop_axes(ax, x_positions, grouped, top_tokens_per_setting,
                                       n_settings, n_positions, color_list, tokenizer, fontsize,
                                       legendfontsize, title_suffix=""):
    """Apply shared formatting for intersection lollipop charts: colored tick labels, separators, labels."""
    tick_labels = []
    tick_colors = []
    for owner_setting, token_id in grouped:
        tick_labels.append(_token_label(token_id, tokenizer))
        tick_colors.append(color_list[owner_setting])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(_maybe_hide_token_labels(tick_labels), fontsize=max(fontsize - 2, 4), rotation=90)
    for tick_label, color in zip(ax.get_xticklabels(), tick_colors):
        tick_label.set_color(color)

    # Add setting separator lines between groups
    pos = 0
    for setting_idx in range(n_settings):
        n_tokens_in_group = len(top_tokens_per_setting.get(setting_idx, []))
        pos += n_tokens_in_group
        if setting_idx < n_settings - 1 and pos < n_positions:
            ax.axvline(x=pos - 0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

    n_top = max(len(v) for v in top_tokens_per_setting.values()) if top_tokens_per_setting else 0
    ax.set_xlabel('Token (colored by owning setting)' if tokenizer is not None else 'Token ID (colored by owning setting)', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Per-Setting Top-{n_top} Tokens by Log q{title_suffix}', fontsize=fontsize + 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_top_q_intersection_lollipop(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
    figname_individual=None,
    marker_list=None,
):
    """
    Lollipop chart showing per-setting top tokens ranked by average log q (across seeds).

    Each setting gets its own top-N tokens (by mean log q across seeds at the final timestep).
    Tokens are grouped by setting: all top-N for setting 1, then all top-N for setting 2, etc.
    Within each group, tokens are ordered left-to-right by decreasing average log q.
    For each token, a black dash shows the target log prob and colored dots (with bootstrap CIs)
    show each setting's log q.

    If figname_individual is provided, also produces an individual-seed version (per-seed scatter
    dots instead of bootstrap CIs) using the same precomputed data.
    """
    # Collect all token data at the final timestep
    if precomputed_token_data is not None:
        target_by_setting, q_by_setting, base_by_setting, _ = precomputed_token_data
    else:
        target_by_setting, q_by_setting, base_by_setting, _ = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens=999999, final_only=True)
    if q_by_setting is None:
        print("Warning: No token data found. Cannot create top-q lollipop.")
        return

    n_settings = len(labels)

    top_tokens_per_setting, grouped = _compute_top_tokens_per_setting_and_grouped(
        q_by_setting, n_settings, n_top_tokens)

    if not grouped:
        print("Warning: No tokens found. Skipping top-q lollipop.")
        return

    n_positions = len(grouped)
    x_positions = np.arange(n_positions)
    grouped_token_ids = [token_id for _, token_id in grouped]
    target_means = _compute_target_means_for_tokens(grouped_token_ids, target_by_setting)

    # Compute base model mean per token (pooled across all settings/seeds — base is shared)
    has_base_data = len(base_by_setting) > 0
    base_means = np.full(n_positions, np.nan)
    if has_base_data:
        for pos_idx, token_id in enumerate(grouped_token_ids):
            all_base_vals = []
            for vals in base_by_setting.get(token_id, {}).values():
                all_base_vals.extend(vals)
            if all_base_vals:
                base_means[pos_idx] = np.mean(all_base_vals)

    dash_half_width = 0.35
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])

    # --- Aggregate (bootstrap CI) plot ---
    fig, ax = plt.subplots(figsize=(max(8, n_positions * 0.45), _scale_h(5)))

    _draw_target_dashes(ax, x_positions, target_means, dash_half_width, linewidth=1.5)

    # Draw base model markers (if available): dashed horizontal line per token
    if has_base_data:
        base_label_added = False
        for pos_idx in range(n_positions):
            if np.isnan(base_means[pos_idx]):
                continue
            label = r"$p$ (base)" if not base_label_added else None
            ax.plot(
                [x_positions[pos_idx] - dash_half_width, x_positions[pos_idx] + dash_half_width],
                [base_means[pos_idx], base_means[pos_idx]],
                color='gray', linewidth=1.5, linestyle='--', solid_capstyle='butt', label=label, zorder=3,
            )
            base_label_added = True

    setting_label_added = [False] * n_settings
    for setting_idx in range(n_settings):
        q_means = np.full(n_positions, np.nan)
        q_ci_lo = np.full(n_positions, np.nan)
        q_ci_hi = np.full(n_positions, np.nan)

        for pos_idx, (owner_setting, token_id) in enumerate(grouped):
            q_values = np.array(q_by_setting.get(token_id, {}).get(setting_idx, []))
            q_means[pos_idx], q_ci_lo[pos_idx], q_ci_hi[pos_idx] = \
                _bootstrap_mean_ci(q_values, n_bootstrap_draws)

        x_pos = x_positions + setting_offsets[setting_idx]
        lower_err = q_means - q_ci_lo
        upper_err = q_ci_hi - q_means
        valid = ~np.isnan(q_means)
        if valid.any():
            label = labels[setting_idx] if not setting_label_added[setting_idx] else None
            marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.errorbar(
                x_pos[valid], q_means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt=marker, color=color_list[setting_idx], markersize=4,
                capsize=2, linewidth=0.8, label=label, zorder=4,
            )
            setting_label_added[setting_idx] = True

    _format_intersection_lollipop_axes(
        ax, x_positions, grouped, top_tokens_per_setting,
        n_settings, n_positions, color_list, tokenizer, fontsize, legendfontsize,
        title_suffix=" (final step)")
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Top-q lollipop chart saved to {figname}")

    # --- Individual-seed plot (reuses precomputed data) ---
    if figname_individual is not None:
        fig_ind, ax_ind = plt.subplots(figsize=(max(8, n_positions * 0.45), _scale_h(5)))

        _draw_target_dashes(ax_ind, x_positions, target_means, dash_half_width, linewidth=1.5)

        # Draw base model markers (if available)
        if has_base_data:
            base_label_added_ind = False
            for pos_idx in range(n_positions):
                if np.isnan(base_means[pos_idx]):
                    continue
                label = r"$p$ (base)" if not base_label_added_ind else None
                ax_ind.plot(
                    [x_positions[pos_idx] - dash_half_width, x_positions[pos_idx] + dash_half_width],
                    [base_means[pos_idx], base_means[pos_idx]],
                    color='gray', linewidth=1.5, linestyle='--', solid_capstyle='butt', label=label, zorder=3,
                )
                base_label_added_ind = True

        for setting_idx in range(n_settings):
            label_added = False
            for pos_idx, (owner_setting, token_id) in enumerate(grouped):
                q_values = q_by_setting.get(token_id, {}).get(setting_idx, [])
                if not q_values:
                    continue
                x_base = x_positions[pos_idx] + setting_offsets[setting_idx]
                for seed_j, val in enumerate(q_values):
                    marker = SEED_MARKERS[seed_j % len(SEED_MARKERS)]
                    label = labels[setting_idx] if not label_added else None
                    ax_ind.scatter(
                        x_base, val,
                        color=color_list[setting_idx], s=15, alpha=0.7,
                        marker=marker, label=label, zorder=4,
                    )
                    label_added = True

        _format_intersection_lollipop_axes(
            ax_ind, x_positions, grouped, top_tokens_per_setting,
            n_settings, n_positions, color_list, tokenizer, fontsize, legendfontsize,
            title_suffix=" (individual seeds, final step)")
        plt.tight_layout()

        plt.savefig(figname_individual)
        plt.clf()
        plt.close(fig_ind)
        print(f"Individual top-q lollipop chart saved to {figname_individual}")


def plot_top_q_ranked_lollipop(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_ranks=10,
    figname_individual=None,
    marker_list=None,
):
    """
    Lollipop chart of rank-ordered log probabilities under q, with corresponding target log probs.

    For each setting and seed, at the final evaluation timestep, sorts tracked tokens by
    log_probs_q descending and records the log prob under q and under target at each rank
    (1st highest, 2nd highest, etc.). Then averages across seeds with bootstrap CIs.

    Unlike the other top-token plots, this does NOT track specific token IDs — it tracks
    rank positions. The token at rank k may differ across seeds; what matters is "how much
    probability does q place on its k-th most likely token, and what does target think of
    that same token?"

    Both q and target get CIs (target varies across seeds because different seeds pick
    different tokens at each rank).

    If figname_individual is provided, also produces an individual-seed version (per-seed
    scatter dots instead of bootstrap CIs) using the same precomputed rank data.
    """
    rank_data, n_ranks_actual = _collect_rank_data(labels, results_list, n_ranks)
    if rank_data is None or n_ranks_actual == 0:
        print("Warning: No rank data found. Cannot create ranked lollipop chart.")
        return

    # Print per-seed token IDs at each rank for diagnostics
    for setting_idx in range(len(labels)):
        seeds = rank_data[setting_idx]
        if not seeds:
            continue
        print(f"\n[{labels[setting_idx]}] Per-seed top-{n_ranks_actual} token IDs under q (final timestep):")
        for seed_j, (q_by_rank, tgt_by_rank, ids_by_rank) in enumerate(seeds):
            ids_str = ", ".join(f"{tid}(q={q:.2f}, σ={t:.2f})"
                                for tid, q, t in zip(ids_by_rank[:n_ranks_actual],
                                                      q_by_rank[:n_ranks_actual],
                                                      tgt_by_rank[:n_ranks_actual]))
            print(f"  Seed {seed_j + 1}: {ids_str}")

    x_positions = np.arange(n_ranks_actual)
    dot_spacing = 0.12
    n_settings = len(labels)
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])

    from matplotlib.lines import Line2D

    # --- Aggregate (bootstrap CI) plot ---
    fig, ax = plt.subplots()

    for setting_idx in range(n_settings):
        seeds = rank_data[setting_idx]
        if not seeds:
            continue

        q_means = np.full(n_ranks_actual, np.nan)
        q_ci_lo = np.full(n_ranks_actual, np.nan)
        q_ci_hi = np.full(n_ranks_actual, np.nan)
        tgt_means = np.full(n_ranks_actual, np.nan)
        tgt_ci_lo = np.full(n_ranks_actual, np.nan)
        tgt_ci_hi = np.full(n_ranks_actual, np.nan)

        for rank in range(n_ranks_actual):
            q_vals = np.array([s[0][rank] for s in seeds if rank < len(s[0])])
            tgt_vals = np.array([s[1][rank] for s in seeds if rank < len(s[1]) and not np.isnan(s[1][rank])])

            q_means[rank], q_ci_lo[rank], q_ci_hi[rank] = _bootstrap_mean_ci(q_vals, n_bootstrap_draws)
            tgt_means[rank], tgt_ci_lo[rank], tgt_ci_hi[rank] = _bootstrap_mean_ci(tgt_vals, n_bootstrap_draws)

        x_pos = x_positions + setting_offsets[setting_idx]

        # Vertical connecting lines (target to q)
        for rank in range(n_ranks_actual):
            if np.isnan(q_means[rank]) or np.isnan(tgt_means[rank]):
                continue
            ax.plot(
                [x_pos[rank], x_pos[rank]],
                [tgt_means[rank], q_means[rank]],
                color=color_list[setting_idx], linewidth=1, alpha=0.4, zorder=1,
            )

        # Target markers with CI (diamond shape, same color but hollow)
        valid_tgt = ~np.isnan(tgt_means)
        if valid_tgt.any():
            tgt_lower = tgt_means[valid_tgt] - tgt_ci_lo[valid_tgt]
            tgt_upper = tgt_ci_hi[valid_tgt] - tgt_means[valid_tgt]
            ax.errorbar(
                x_pos[valid_tgt], tgt_means[valid_tgt],
                yerr=[tgt_lower, tgt_upper],
                fmt='D', color=color_list[setting_idx], markersize=4,
                markerfacecolor='none', markeredgewidth=1.2,
                capsize=2, linewidth=0.8, zorder=3,
            )

        # q markers with CI (filled setting-specific marker)
        valid_q = ~np.isnan(q_means)
        if valid_q.any():
            q_lower = q_means[valid_q] - q_ci_lo[valid_q]
            q_upper = q_ci_hi[valid_q] - q_means[valid_q]
            q_marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.errorbar(
                x_pos[valid_q], q_means[valid_q],
                yerr=[q_lower, q_upper],
                fmt=q_marker, color=color_list[setting_idx], markersize=5,
                capsize=3, linewidth=1, zorder=4,
            )

    # Build consolidated legend: one entry per setting (colored line+marker), plus target marker-type key
    legend_handles = []
    if marker_list is None:
        legend_handles.append(Line2D([], [], marker='o', color='black', markersize=5,
                                     linestyle='None', label='q'))
    legend_handles.append(Line2D([], [], marker='D', color='black', markersize=4,
                                 markerfacecolor='none', markeredgewidth=1.2,
                                 linestyle='None', label=r'$\sigma$ (target)'))
    for setting_idx in range(n_settings):
        if not rank_data[setting_idx]:
            continue
        if marker_list is not None:
            legend_handles.append(Line2D([], [], color=color_list[setting_idx], linewidth=2,
                                         marker=marker_list[setting_idx], markersize=5,
                                         label=labels[setting_idx]))
        else:
            legend_handles.append(Line2D([], [], color=color_list[setting_idx], linewidth=2,
                                         label=labels[setting_idx]))

    ax.set_xlabel('Rank under q', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Top {n_ranks_actual} Ranked Tokens under q: Log Prob (target vs q)', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(r + 1) for r in range(n_ranks_actual)], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(handles=legend_handles, fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Ranked lollipop chart saved to {figname}")

    # --- Individual-seed plot (reuses precomputed rank_data) ---
    if figname_individual is not None:
        fig_ind, ax_ind = plt.subplots()

        for setting_idx in range(n_settings):
            seeds = rank_data[setting_idx]
            if not seeds:
                continue

            x_pos = x_positions + setting_offsets[setting_idx]

            for seed_j, (q_by_rank, tgt_by_rank, ids_by_rank) in enumerate(seeds):
                n_this = min(n_ranks_actual, len(q_by_rank))
                marker = SEED_MARKERS[seed_j % len(SEED_MARKERS)]

                # q dots
                ax_ind.scatter(
                    x_pos[:n_this], q_by_rank[:n_this],
                    color=color_list[setting_idx], s=20, alpha=0.7,
                    marker=marker, zorder=4,
                )

                # target dots (hollow version of same marker)
                valid_tgt = ~np.isnan(tgt_by_rank[:n_this])
                if valid_tgt.any():
                    ax_ind.scatter(
                        x_pos[:n_this][valid_tgt], tgt_by_rank[:n_this][valid_tgt],
                        color=color_list[setting_idx], s=18, alpha=0.7,
                        marker=marker, facecolors='none', linewidths=1, zorder=3,
                    )

        legend_handles_ind = []
        legend_handles_ind.append(Line2D([], [], marker='o', color='black', markersize=5,
                                     linestyle='None', label='q (filled)'))
        legend_handles_ind.append(Line2D([], [], marker='o', color='black', markersize=4,
                                     markerfacecolor='none', markeredgewidth=1.2,
                                     linestyle='None', label=r'$\sigma$ (target, hollow)'))
        for setting_idx in range(n_settings):
            if not rank_data[setting_idx]:
                continue
            legend_handles_ind.append(Line2D([], [], color=color_list[setting_idx], linewidth=2,
                                         label=labels[setting_idx]))

        ax_ind.set_xlabel('Rank under q', fontsize=fontsize)
        ax_ind.set_ylabel('Log Probability', fontsize=fontsize)
        ax_ind.set_title(f'Top {n_ranks_actual} Ranked Tokens under q: Log Prob (individual seeds)', fontsize=fontsize + 1)
        ax_ind.set_xticks(x_positions)
        ax_ind.set_xticklabels([str(r + 1) for r in range(n_ranks_actual)], fontsize=fontsize - 1)
        ax_ind.tick_params(axis='y', labelsize=fontsize)
        ax_ind.legend(handles=legend_handles_ind, fontsize=legendfontsize)
        ax_ind.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        plt.savefig(figname_individual)
        plt.clf()
        plt.close(fig_ind)
        print(f"Individual ranked lollipop chart saved to {figname_individual}")


def plot_max_sis_weight_over_time(
    figname, labels, sis_weights_results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
):
    """
    Plot the mean per-prompt max self-normalized importance weight over training episodes.

    Each entry in sis_weights_results_list[setting_idx] is a loaded SIS weights history
    (list of tensors of shape (num_prompts, samples_per_prompt), one per episode) for one seed.
    For each episode, we compute the max weight within each prompt, then average across prompts.
    This avoids saturation at 1.0 that occurs with the global max when many prompts are present.
    Aggregated across seeds with bootstrap CI.
    """
    n_settings = len(labels)

    fig, ax = plt.subplots()

    for setting_idx in range(n_settings):
        seed_data = sis_weights_results_list[setting_idx]

        if not seed_data:
            continue

        # Each seed's data is a list of tensors (one per episode)
        seed_curves = []
        for weights_history in seed_data:
            if not isinstance(weights_history, list) or len(weights_history) == 0:
                continue
            # Mean of per-prompt max weights (avoids saturation from global max)
            curve = [w.max(dim=-1).values.mean().item() for w in weights_history]
            seed_curves.append(curve)

        if not seed_curves:
            continue

        # Pad shorter curves with their last value
        max_len = max(len(c) for c in seed_curves)
        for c in seed_curves:
            last_val = c[-1] if c else np.nan
            while len(c) < max_len:
                c.append(last_val)

        timesteps = np.arange(max_len)
        means = np.full(max_len, np.nan)
        ci_lo = np.full(max_len, np.nan)
        ci_hi = np.full(max_len, np.nan)

        for t_idx in range(max_len):
            vals = np.array([c[t_idx] for c in seed_curves])
            valid_vals = vals[~np.isnan(vals)]
            if len(valid_vals) > 0:
                means[t_idx], ci_lo[t_idx], ci_hi[t_idx] = _bootstrap_mean_ci(valid_vals, n_bootstrap_draws)

        valid = ~np.isnan(means)
        if valid.any():
            ax.plot(timesteps[valid], means[valid], color=color_list[setting_idx],
                    label=labels[setting_idx], linewidth=1.5)
            ax.fill_between(timesteps[valid], ci_lo[valid], ci_hi[valid],
                            color=color_list[setting_idx], alpha=0.15)

    ax.set_xlabel('Training Episode', fontsize=fontsize)
    ax.set_ylabel('Mean Per-Prompt Max SIS Weight', fontsize=fontsize)
    ax.set_title('Mean Per-Prompt Max Self-Normalized Importance Weight Over Time', fontsize=fontsize + 1)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Max SIS weight plot saved to {figname}")


def plot_sis_weight_histogram(
    figname, labels, sis_weights_results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000, n_bins=8,
):
    """
    Grouped bar chart of per-prompt max self-normalized importance weight distribution.

    Uses the final timestep from each seed. For each prompt, computes the max weight
    across samples, then bins prompts by that max weight. Bars show the fraction of
    prompts in each bin, averaged across seeds with bootstrap CIs.
    """
    n_settings = len(labels)

    # Determine samples_per_prompt from first available tensor
    samples_per_prompt = None
    for seed_data in sis_weights_results_list:
        for weights_history in seed_data:
            if isinstance(weights_history, list) and len(weights_history) > 0:
                samples_per_prompt = weights_history[0].shape[-1]
                break
        if samples_per_prompt is not None:
            break
    if samples_per_prompt is None:
        print(f"Warning: No SIS weight data found, skipping histogram")
        return

    # Bin edges from uniform (1/n) to fully concentrated (1.0)
    bin_lo = 1.0 / samples_per_prompt
    bin_edges = np.linspace(bin_lo, 1.0, n_bins + 1)

    # Collect per-bin fractions: fractions[bin_idx][setting_idx] = list of per-seed fractions
    fractions = [[[] for _ in range(n_settings)] for _ in range(n_bins)]

    for setting_idx in range(n_settings):
        seed_data = sis_weights_results_list[setting_idx]
        for weights_history in seed_data:
            if not isinstance(weights_history, list) or len(weights_history) == 0:
                continue
            w = weights_history[-1]  # Final timestep: (num_prompts, samples_per_prompt)
            max_weights = w.max(dim=-1).values.numpy()  # (num_prompts,)
            counts, _ = np.histogram(max_weights, bins=bin_edges)
            seed_fractions = counts / max_weights.shape[0]
            for bin_idx in range(n_bins):
                fractions[bin_idx][setting_idx].append(seed_fractions[bin_idx])

    # Compute means and CIs
    means = np.full((n_bins, n_settings), np.nan)
    ci_lo = np.full((n_bins, n_settings), np.nan)
    ci_hi = np.full((n_bins, n_settings), np.nan)
    for bin_idx in range(n_bins):
        for setting_idx in range(n_settings):
            vals = np.array(fractions[bin_idx][setting_idx])
            if len(vals) > 0:
                means[bin_idx, setting_idx], ci_lo[bin_idx, setting_idx], ci_hi[bin_idx, setting_idx] = \
                    _bootstrap_mean_ci(vals, n_bootstrap_draws)

    # Plot grouped bars
    fig, ax = plt.subplots()
    bar_width = 0.8 / n_settings
    x_positions = np.arange(n_bins)
    group_offsets = np.linspace(-bar_width * (n_settings - 1) / 2,
                                 bar_width * (n_settings - 1) / 2,
                                 n_settings)

    for setting_idx in range(n_settings):
        x_pos = x_positions + group_offsets[setting_idx]
        ax.bar(x_pos, means[:, setting_idx], bar_width,
               label=labels[setting_idx],
               color=color_list[setting_idx],
               alpha=0.7)
        for bin_idx in range(n_bins):
            if not np.isnan(means[bin_idx, setting_idx]):
                mean_val = means[bin_idx, setting_idx]
                lower_err = mean_val - ci_lo[bin_idx, setting_idx]
                upper_err = ci_hi[bin_idx, setting_idx] - mean_val
                ax.errorbar(x_pos[bin_idx], mean_val,
                            yerr=[[lower_err], [upper_err]],
                            fmt='none', color='black', capsize=2, linewidth=0.8)

    # X-tick labels: bin ranges
    tick_labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(n_bins)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=fontsize - 1)

    ax.set_xlabel('Per-Prompt Max SIS Weight', fontsize=fontsize)
    ax.set_ylabel('Fraction of Prompts', fontsize=fontsize)
    ax.set_title('Distribution of Per-Prompt Max Normalized SIS Weight (Final Step)', fontsize=fontsize + 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"SIS weight histogram saved to {figname}")


def plot_sis_weight_histogram_over_time(
    figname, labels, sis_weights_results_list,
    color_list, n_frontiers=4,
    fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000, n_bins=8,
):
    """
    Per-prompt max SIS weight distribution over time, shown as dots with alpha progression.

    Like plot_sis_weight_histogram but at n_frontiers evenly-spaced timesteps.
    Each timestep is a set of dots (one per bin) with increasing opacity (light=early, dark=late).
    Settings are horizontally offset within each bin. Bootstrap CIs shown as error bars.
    """
    n_settings = len(labels)

    # Determine samples_per_prompt and max trajectory length from data
    samples_per_prompt = None
    max_T = 0
    for seed_data in sis_weights_results_list:
        for weights_history in seed_data:
            if isinstance(weights_history, list) and len(weights_history) > 0:
                if samples_per_prompt is None:
                    samples_per_prompt = weights_history[0].shape[-1]
                max_T = max(max_T, len(weights_history))
    if samples_per_prompt is None or max_T == 0:
        print("Warning: No SIS weight data found, skipping histogram over time")
        return

    # Bin edges
    bin_lo = 1.0 / samples_per_prompt
    bin_edges = np.linspace(bin_lo, 1.0, n_bins + 1)

    # Compute evenly-spaced timestep indices
    frontier_indices = [round((max_T - 1) * i / n_frontiers) for i in range(1, n_frontiers + 1)]
    seen = set()
    unique_frontier_indices = []
    for idx in frontier_indices:
        if idx not in seen:
            seen.add(idx)
            unique_frontier_indices.append(idx)
    frontier_indices = unique_frontier_indices
    n_times = len(frontier_indices)

    # Alpha progression: light (early) to dark (late)
    alphas = np.linspace(0.25, 1.0, n_times)

    # Collect bin fractions: shape (n_settings, n_bins, n_times)
    means = np.full((n_settings, n_bins, n_times), np.nan)
    ci_lo = np.full((n_settings, n_bins, n_times), np.nan)
    ci_hi = np.full((n_settings, n_bins, n_times), np.nan)

    for time_i, t_idx in enumerate(frontier_indices):
        for setting_idx in range(n_settings):
            # Collect per-seed bin fractions at this timestep
            bin_fractions_by_seed = [[] for _ in range(n_bins)]
            for weights_history in sis_weights_results_list[setting_idx]:
                if not isinstance(weights_history, list) or len(weights_history) == 0:
                    continue
                # Clamp t_idx to available range
                actual_t = min(t_idx, len(weights_history) - 1)
                w = weights_history[actual_t]
                max_weights = w.max(dim=-1).values.numpy()
                counts, _ = np.histogram(max_weights, bins=bin_edges)
                seed_fractions = counts / max_weights.shape[0]
                for bin_idx in range(n_bins):
                    bin_fractions_by_seed[bin_idx].append(seed_fractions[bin_idx])

            for bin_idx in range(n_bins):
                vals = np.array(bin_fractions_by_seed[bin_idx])
                if len(vals) > 0:
                    means[setting_idx, bin_idx, time_i], ci_lo[setting_idx, bin_idx, time_i], \
                        ci_hi[setting_idx, bin_idx, time_i] = _bootstrap_mean_ci(vals, n_bootstrap_draws)

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, n_bins * 1.2), _scale_h(5)))

    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_bins)

    for setting_idx in range(n_settings):
        x_base = x_positions + setting_offsets[setting_idx]
        setting_label_added = False

        for time_i in range(n_times):
            m = means[setting_idx, :, time_i]
            lo = ci_lo[setting_idx, :, time_i]
            hi = ci_hi[setting_idx, :, time_i]
            lower_err = m - lo
            upper_err = hi - m
            valid = ~np.isnan(m)
            if not valid.any():
                continue

            label = labels[setting_idx] if not setting_label_added else None
            ax.scatter(
                x_base[valid], m[valid],
                color=color_list[setting_idx], s=25,
                alpha=alphas[time_i], label=label, zorder=4,
            )
            ax.errorbar(
                x_base[valid], m[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt='none', color=color_list[setting_idx],
                capsize=2, linewidth=0.8, alpha=alphas[time_i] * 0.5,
                zorder=3,
            )
            setting_label_added = True

    # Timestep annotation
    time_str = ", ".join(str(idx) for idx in frontier_indices)
    ax.annotate(f"Timesteps: {time_str} (light\u2192dark)", xy=(0.02, 0.98),
                xycoords='axes fraction', fontsize=fontsize - 1, color='gray',
                verticalalignment='top')

    # X-tick labels: bin ranges
    tick_labels = [f"{bin_edges[i]:.2f}\u2013{bin_edges[i+1]:.2f}" for i in range(n_bins)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=fontsize - 1)

    ax.set_xlabel('Per-Prompt Max SIS Weight', fontsize=fontsize)
    ax.set_ylabel('Fraction of Prompts', fontsize=fontsize)
    ax.set_title('Distribution of Per-Prompt Max SIS Weight Over Time', fontsize=fontsize + 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"SIS weight histogram over time saved to {figname}")


def _extract_per_seed_at_timestep(data, n_settings, t_idx=-1):
    """For each setting, extract per-seed 1D arrays at a given timestep.

    Flattens across prompts within each seed.

    Args:
        data: List (settings) of list (seeds) of list (T timesteps) of
              list (P prompts) of 1D numpy array (n_samples,) or None.
        n_settings: Number of settings.
        t_idx: Timestep index to extract. Negative indices work as usual
               (default -1 = final timestep). Clamped to valid range per seed.

    Returns:
        List (settings) of (list of 1D arrays, one per seed) or None if no data.
    """
    all_settings = []
    for setting_idx in range(n_settings):
        seed_list = data[setting_idx] if setting_idx < len(data) else []
        if not seed_list:
            all_settings.append(None)
            continue
        seed_arrays = []
        for seed_data in seed_list:
            if not seed_data:
                continue
            # Resolve and clamp index
            if t_idx < 0:
                actual_t = max(0, len(seed_data) + t_idx)
            else:
                actual_t = min(t_idx, len(seed_data) - 1)
            t_data = seed_data[actual_t]  # list (P) of arrays
            vals = []
            for p_data in t_data:
                if p_data is not None:
                    vals.extend(np.asarray(p_data).ravel().tolist())
            if vals:
                seed_arrays.append(np.array(vals))
        all_settings.append(seed_arrays if seed_arrays else None)
    return all_settings


def _extract_final_per_seed(data, n_settings):
    """For each setting, extract per-seed 1D arrays at the final timestep.

    Convenience wrapper around _extract_per_seed_at_timestep with t_idx=-1.
    """
    return _extract_per_seed_at_timestep(data, n_settings, t_idx=-1)


def _pad_to_length(arr, n):
    """Pad a 1D array with NaN to length n."""
    if len(arr) < n:
        return np.concatenate([arr, np.full(n - len(arr), np.nan)])
    return arr


def plot_g_q_lollipop(figname, labels, g_q_per_sample_data, color_list, fontsize=7, legendfontsize=7,
                      n_bootstrap_draws=5000, marker_list=None):
    """
    Scatter plot of final-timestep g_q values for each target sequence, one color per setting.

    Target sequences (flattened across prompts) are sorted by descending mean g_q across settings,
    so the sequences q covers worst (highest g_q) appear on the left. Multiple settings are shown
    as offset colored dots with translucent bootstrap confidence intervals.

    g_q(x) = log p0(x) + beta*r(x) - log q(x); lower = q covers x better.

    Args:
        g_q_per_sample_data: List (settings) of list (seeds) of list (T timesteps) of
                             list (P prompts) of 1D numpy array (n_samples,) of g_q values.
        n_bootstrap_draws: Number of bootstrap resamples for CI computation.
    """
    n_settings = len(labels)

    # For each setting, extract per-seed 1D arrays at final timestep
    per_seed = _extract_final_per_seed(g_q_per_sample_data, n_settings)

    # Compute mean across seeds for sorting
    final_g_q_mean = []
    for seed_arrays in per_seed:
        if seed_arrays is None:
            final_g_q_mean.append(None)
            continue
        max_n = max(len(a) for a in seed_arrays)
        padded = [_pad_to_length(a, max_n) for a in seed_arrays]
        final_g_q_mean.append(np.nanmean(np.stack(padded, axis=0), axis=0))

    valid_settings = [(i, arr) for i, arr in enumerate(final_g_q_mean) if arr is not None]
    if not valid_settings:
        print(f"No data for g_q lollipop plot, skipping {figname}")
        return

    n_samples = max(len(arr) for _, arr in valid_settings)
    # Sort by descending mean g_q (worst-covered sequences on the left)
    all_arrs = np.full((len(valid_settings), n_samples), np.nan)
    for plot_idx, (_, arr) in enumerate(valid_settings):
        all_arrs[plot_idx, :len(arr)] = arr
    sort_order = np.argsort(np.nanmean(all_arrs, axis=0))[::-1]

    x_positions = np.arange(n_samples)
    n_valid = len(valid_settings)
    x_offsets = np.linspace(-0.25, 0.25, n_valid) if n_valid > 1 else np.array([0.0])

    def _compute_sorted_ci(seed_arrays):
        """Bootstrap mean + CI per sample position, sorted by sort_order."""
        if seed_arrays is None:
            return (np.full(n_samples, np.nan), np.full(n_samples, np.nan),
                    np.full(n_samples, np.nan))
        padded = np.stack([_pad_to_length(a, n_samples)[sort_order] for a in seed_arrays])
        means = np.full(n_samples, np.nan)
        ci_lo = np.full(n_samples, np.nan)
        ci_hi = np.full(n_samples, np.nan)
        for j in range(n_samples):
            col = padded[:, j]
            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                means[j], ci_lo[j], ci_hi[j] = _bootstrap_mean_ci(valid_col, n_bootstrap_draws)
        return means, ci_lo, ci_hi

    fig, ax = plt.subplots()
    for plot_idx, (setting_idx, _) in enumerate(valid_settings):
        seed_arrays = per_seed[setting_idx]
        color = color_list[setting_idx]
        x = x_positions + x_offsets[plot_idx]

        means, ci_lo, ci_hi = _compute_sorted_ci(seed_arrays)
        valid = ~np.isnan(means)
        if np.any(valid):
            lo_err = means[valid] - ci_lo[valid]
            hi_err = ci_hi[valid] - means[valid]
            marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.scatter(x[valid], means[valid],
                       color=color, s=12, zorder=4,
                       marker=marker,
                       label=labels[setting_idx])
            ax.errorbar(x[valid], means[valid], yerr=[lo_err, hi_err],
                        fmt='none', ecolor=color, alpha=0.2,
                        capsize=3, linewidth=1, zorder=3)

    ax.set_xlabel('Target Sequence Index (sorted by mean g_q, worst first)', fontsize=fontsize)
    ax.set_ylabel(r'$g_q(x) = \log p_0(x) + \beta r(x) - \log q(x)$', fontsize=fontsize)
    ax.set_title('g_q at Final Timestep Per Target Sequence (lower = better coverage)', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(np.arange(1, n_samples + 1), fontsize=max(4, fontsize - 2))
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"g_q lollipop plot saved to {figname}")


def plot_two_series_lollipop(figname, labels, series1_name, series2_name,
                              series1_data, series2_data,
                              color_list, fontsize=7, legendfontsize=7,
                              n_bootstrap_draws=5000, figname_individual=None,
                              series3_name=None, series3_data=None,
                              sort_by_series3=False, marker_list=None,
                              n_top_tokens=None):
    """
    Lollipop plot comparing a per-seed quantity (series1) against fixed references (series2,
    and optionally series3) across target sequences.

    Series2 (e.g. log p) is treated as fixed across seeds and drawn as black horizontal dashes.
    Series3 (e.g. log sigma), if provided, is drawn as gray dashed horizontal lines.
    Series1 (e.g. log q) varies across seeds and is drawn as colored dots with bootstrap CIs.
    Sequences are sorted by descending mean of the sort series (series3 if sort_by_series3
    and series3 is provided, otherwise series2).

    If figname_individual is provided, also produces an individual-seed version with a
    different marker per seed.

    Args:
        series1_name: Short label for the per-seed quantity (colored markers), e.g. r'$\\log q$'.
        series2_name: Short label for the first fixed reference (black dashes), e.g. r'$\\log p$'.
        series1_data, series2_data: List (settings) of list (seeds) of list (T timesteps) of
                                    list (P prompts) of 1D numpy array (n_samples,) or None.
        n_bootstrap_draws: Number of bootstrap resamples for confidence intervals.
        figname_individual: If provided, save an individual-seed scatter plot to this path.
        series3_name: Short label for an optional second fixed reference (gray dashes).
        series3_data: Same structure as series2_data. If provided, drawn as gray dashed lines.
        sort_by_series3: If True and series3_data is provided, sort x-axis by descending
                         series3 mean instead of series2.
    """
    n_settings = len(labels)

    per_seed_s1 = _extract_final_per_seed(series1_data, n_settings)
    per_seed_s2 = _extract_final_per_seed(series2_data, n_settings)
    per_seed_s3 = _extract_final_per_seed(series3_data, n_settings) if series3_data is not None else None

    # Determine n_samples and sort order from the sort series
    valid_s2_settings = [(i, arrs) for i, arrs in enumerate(per_seed_s2) if arrs is not None]
    if not valid_s2_settings:
        print(f"No series2 data for two-series lollipop plot, skipping {figname}")
        return

    n_samples = max(len(a) for _, arrs in valid_s2_settings for a in arrs)

    # Decide which series to sort by: series3 if requested and available, else series2
    if sort_by_series3 and per_seed_s3 is not None:
        valid_sort_settings = [(i, arrs) for i, arrs in enumerate(per_seed_s3) if arrs is not None]
        sort_series_name = series3_name
    else:
        valid_sort_settings = valid_s2_settings
        sort_series_name = series2_name

    # Sort by descending mean of the chosen sort series (averaged across all settings and seeds)
    all_sort_padded = []
    for _, arrs in valid_sort_settings:
        for a in arrs:
            all_sort_padded.append(_pad_to_length(a, n_samples))
    sort_order = np.argsort(np.nanmean(np.stack(all_sort_padded), axis=0))[::-1]

    # In final-plots mode, restrict the x-axis to the top n_top_tokens entries
    # (by the sort series). Leaves the individual-seed plot untouched.
    _truncate_main = _final_plots_enabled() and n_top_tokens is not None and n_top_tokens < n_samples
    if _truncate_main:
        sort_order_main = sort_order[:n_top_tokens]
        n_samples_main = n_top_tokens
    else:
        sort_order_main = sort_order
        n_samples_main = n_samples

    # Compute series2 reference values: mean across all settings/seeds (fixed quantity)
    all_s2_padded = []
    for _, arrs in valid_s2_settings:
        for a in arrs:
            all_s2_padded.append(_pad_to_length(a, n_samples))
    s2_ref_full = np.nanmean(np.stack(all_s2_padded), axis=0)
    s2_ref = s2_ref_full[sort_order_main]

    # Compute series3 reference values if provided
    s3_ref = None
    s3_ref_full = None
    if per_seed_s3 is not None:
        valid_s3_settings = [(i, arrs) for i, arrs in enumerate(per_seed_s3) if arrs is not None]
        if valid_s3_settings:
            all_s3_padded = []
            for _, arrs in valid_s3_settings:
                for a in arrs:
                    all_s3_padded.append(_pad_to_length(a, n_samples))
            s3_ref_full = np.nanmean(np.stack(all_s3_padded), axis=0)
            s3_ref = s3_ref_full[sort_order_main]

    x_positions = np.arange(n_samples_main)
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    dash_half_width = (total_width / 2 + dot_spacing * 0.6) if n_settings > 1 else 0.15

    def _compute_sorted_ci(seed_arrays, so, n):
        """Bootstrap mean + CI per sample position, sorted by so (length n)."""
        if seed_arrays is None:
            return (np.full(n, np.nan), np.full(n, np.nan),
                    np.full(n, np.nan))
        padded = np.stack([_pad_to_length(a, n_samples)[so] for a in seed_arrays])
        means = np.full(n, np.nan)
        ci_lo = np.full(n, np.nan)
        ci_hi = np.full(n, np.nan)
        for j in range(n):
            col = padded[:, j]
            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                means[j], ci_lo[j], ci_hi[j] = _bootstrap_mean_ci(valid_col, n_bootstrap_draws)
        return means, ci_lo, ci_hi

    # --- Main plot (bootstrap CI) ---
    # In final-plots mode, size the main figure to match the KL frontier plot
    # (which uses matplotlib's default figsize, scaled by _scale_h when final_plots is on).
    if _final_plots_enabled():
        fig, ax = plt.subplots(figsize=plt.rcParams['figure.figsize'])
    else:
        fig, ax = plt.subplots()

    if s3_ref is not None:
        # When both references present: series3 = black solid (primary), series2 = gray dashed
        _draw_target_dashes(ax, x_positions, s3_ref, dash_half_width, linewidth=2,
                            label_text=series3_name)
        _draw_target_dashes(ax, x_positions, s2_ref, dash_half_width, linewidth=2,
                            label_text=series2_name, color='dimgray', linestyle='--')
    else:
        # Single reference: series2 = black solid
        _draw_target_dashes(ax, x_positions, s2_ref, dash_half_width, linewidth=2,
                            label_text=series2_name)

    # Series1: colored dots with bootstrap CI, one color per setting
    for setting_idx in range(n_settings):
        s1_seeds = per_seed_s1[setting_idx]
        if s1_seeds is None:
            continue
        color = color_list[setting_idx]
        x = x_positions + setting_offsets[setting_idx]

        s1_means, s1_ci_lo, s1_ci_hi = _compute_sorted_ci(s1_seeds, sort_order_main, n_samples_main)

        valid1 = ~np.isnan(s1_means)
        if np.any(valid1):
            lo1 = s1_means[valid1] - s1_ci_lo[valid1]
            hi1 = s1_ci_hi[valid1] - s1_means[valid1]
            marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.scatter(x[valid1], s1_means[valid1], color=color, s=20, zorder=4,
                       marker=marker,
                       label=labels[setting_idx])
            ax.errorbar(x[valid1], s1_means[valid1], yerr=[lo1, hi1],
                        fmt='none', ecolor=color, alpha=0.2,
                        capsize=3, linewidth=1, zorder=3)

    all_ref_names = [series2_name] + ([series3_name] if series3_name else [])
    title_refs = ', '.join(all_ref_names)
    if _final_plots_enabled():
        ax.set_xlabel('Target Sequence', fontsize=fontsize)
        ax.set_ylabel('Log Probability', fontsize=fontsize)
        ax.set_title('Exact Target Sequences: Log Probability of q (final step)',
                     fontsize=fontsize + 1)
    else:
        ax.set_xlabel(f'Target Sequence Index (sorted by descending {sort_series_name})',
                      fontsize=fontsize)
        ax.set_ylabel('Log probability', fontsize=fontsize)
        ax.set_title(f'{series1_name} vs {title_refs} at Final Timestep (per target sequence)',
                     fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    if _final_plots_enabled():
        # Hide the per-sequence numbers; the "Target Sequence" xlabel suffices.
        ax.set_xticklabels(['' for _ in x_positions])
    else:
        ax.set_xticklabels(np.arange(1, n_samples_main + 1), fontsize=max(4, fontsize - 2))
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Lollipop plot saved to {figname}")

    # --- Individual-seed plot ---
    if figname_individual is not None:
        fig_ind, ax_ind = plt.subplots()

        # Use the full (untruncated) sort order for the individual-seed plot.
        x_positions_ind = np.arange(n_samples)
        s2_ref_ind = s2_ref_full[sort_order]
        s3_ref_ind = s3_ref_full[sort_order] if s3_ref_full is not None else None

        if s3_ref_ind is not None:
            _draw_target_dashes(ax_ind, x_positions_ind, s3_ref_ind, dash_half_width, linewidth=2,
                                label_text=series3_name)
            _draw_target_dashes(ax_ind, x_positions_ind, s2_ref_ind, dash_half_width, linewidth=2,
                                label_text=series2_name, color='dimgray', linestyle='--')
        else:
            _draw_target_dashes(ax_ind, x_positions_ind, s2_ref_ind, dash_half_width, linewidth=2,
                                label_text=series2_name)

        # Series1: per-seed scatter with different markers
        for setting_idx in range(n_settings):
            s1_seeds = per_seed_s1[setting_idx]
            if s1_seeds is None:
                continue
            color = color_list[setting_idx]
            label_added = False
            for seed_j, seed_arr in enumerate(s1_seeds):
                sorted_arr = _pad_to_length(seed_arr, n_samples)[sort_order]
                marker = SEED_MARKERS[seed_j % len(SEED_MARKERS)]
                valid = ~np.isnan(sorted_arr)
                if not np.any(valid):
                    continue
                x = x_positions_ind[valid] + setting_offsets[setting_idx]
                label = labels[setting_idx] if not label_added else None
                ax_ind.scatter(x, sorted_arr[valid],
                               color=color, s=20, alpha=0.7,
                               marker=marker, label=label, zorder=4)
                label_added = True

        ax_ind.set_xlabel(f'Target Sequence Index (sorted by descending {sort_series_name})',
                          fontsize=fontsize)
        ax_ind.set_ylabel('Log probability', fontsize=fontsize)
        ax_ind.set_title(f'{series1_name} vs {title_refs} at Final Timestep (individual seeds)',
                         fontsize=fontsize + 1)
        ax_ind.set_xticks(x_positions_ind)
        ax_ind.set_xticklabels(np.arange(1, n_samples + 1), fontsize=max(4, fontsize - 2))
        ax_ind.tick_params(axis='y', labelsize=fontsize)
        ax_ind.legend(fontsize=legendfontsize)
        ax_ind.grid(alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        plt.savefig(figname_individual)
        plt.clf()
        plt.close(fig_ind)
        print(f"Individual two-series lollipop plot saved to {figname_individual}")


def plot_two_series_lollipop_over_time(
    figname, labels, series1_name, series2_name,
    series1_data, series2_data,
    color_list, n_frontiers=4,
    fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    series3_name=None, series3_data=None,
    sort_by_series3=False,
    marker_list=None,
):
    """
    Over-time version of plot_two_series_lollipop with alpha progression.

    Series2 and series3 (fixed references like log p, log sigma) are drawn once.
    Series1 (e.g. log q) is drawn at n_frontiers evenly-spaced timesteps with
    increasing opacity (light=early, dark=late).

    Args:
        Same as plot_two_series_lollipop, plus:
        n_frontiers: Number of evenly-spaced timesteps to show.
    """
    n_settings = len(labels)

    # Determine max trajectory length from series1_data
    max_T = 0
    for setting_data in series1_data:
        for seed_data in setting_data:
            if seed_data:
                max_T = max(max_T, len(seed_data))
    if max_T == 0:
        print(f"No series1 data for over-time lollipop plot, skipping {figname}")
        return

    # Compute evenly-spaced frontier indices (deduplicated)
    frontier_indices = [round((max_T - 1) * i / n_frontiers) for i in range(1, n_frontiers + 1)]
    seen = set()
    unique_frontier_indices = []
    for idx in frontier_indices:
        if idx not in seen:
            seen.add(idx)
            unique_frontier_indices.append(idx)
    frontier_indices = unique_frontier_indices
    n_times = len(frontier_indices)
    alphas = np.linspace(0.25, 1.0, n_times)

    # Extract fixed references at final timestep
    per_seed_s2 = _extract_final_per_seed(series2_data, n_settings)
    per_seed_s3 = _extract_final_per_seed(series3_data, n_settings) if series3_data is not None else None

    # Determine n_samples and sort order from the sort series (same logic as static version)
    valid_s2_settings = [(i, arrs) for i, arrs in enumerate(per_seed_s2) if arrs is not None]
    if not valid_s2_settings:
        print(f"No series2 data for over-time lollipop plot, skipping {figname}")
        return

    n_samples = max(len(a) for _, arrs in valid_s2_settings for a in arrs)

    if sort_by_series3 and per_seed_s3 is not None:
        valid_sort_settings = [(i, arrs) for i, arrs in enumerate(per_seed_s3) if arrs is not None]
        sort_series_name = series3_name
    else:
        valid_sort_settings = valid_s2_settings
        sort_series_name = series2_name

    all_sort_padded = []
    for _, arrs in valid_sort_settings:
        for a in arrs:
            all_sort_padded.append(_pad_to_length(a, n_samples))
    sort_order = np.argsort(np.nanmean(np.stack(all_sort_padded), axis=0))[::-1]

    # Compute fixed reference values
    all_s2_padded = []
    for _, arrs in valid_s2_settings:
        for a in arrs:
            all_s2_padded.append(_pad_to_length(a, n_samples))
    s2_ref = np.nanmean(np.stack(all_s2_padded), axis=0)[sort_order]

    s3_ref = None
    if per_seed_s3 is not None:
        valid_s3_settings = [(i, arrs) for i, arrs in enumerate(per_seed_s3) if arrs is not None]
        if valid_s3_settings:
            all_s3_padded = []
            for _, arrs in valid_s3_settings:
                for a in arrs:
                    all_s3_padded.append(_pad_to_length(a, n_samples))
            s3_ref = np.nanmean(np.stack(all_s3_padded), axis=0)[sort_order]

    x_positions = np.arange(n_samples)
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    dash_half_width = (total_width / 2 + dot_spacing * 0.6) if n_settings > 1 else 0.15

    def _compute_sorted_ci(seed_arrays):
        """Bootstrap mean + CI per sample position, sorted by sort_order."""
        if seed_arrays is None:
            return (np.full(n_samples, np.nan), np.full(n_samples, np.nan),
                    np.full(n_samples, np.nan))
        padded = np.stack([_pad_to_length(a, n_samples)[sort_order] for a in seed_arrays])
        means = np.full(n_samples, np.nan)
        ci_lo = np.full(n_samples, np.nan)
        ci_hi = np.full(n_samples, np.nan)
        for j in range(n_samples):
            col = padded[:, j]
            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                means[j], ci_lo[j], ci_hi[j] = _bootstrap_mean_ci(valid_col, n_bootstrap_draws)
        return means, ci_lo, ci_hi

    # --- Plot ---
    fig, ax = plt.subplots()

    # Draw fixed references (full opacity, drawn once)
    if s3_ref is not None:
        _draw_target_dashes(ax, x_positions, s3_ref, dash_half_width, linewidth=2,
                            label_text=series3_name)
        _draw_target_dashes(ax, x_positions, s2_ref, dash_half_width, linewidth=2,
                            label_text=series2_name, color='dimgray', linestyle='--')
    else:
        _draw_target_dashes(ax, x_positions, s2_ref, dash_half_width, linewidth=2,
                            label_text=series2_name)

    # Series1 at each frontier timestep with alpha progression
    for time_i, t_idx in enumerate(frontier_indices):
        per_seed_s1 = _extract_per_seed_at_timestep(series1_data, n_settings, t_idx=t_idx)

        for setting_idx in range(n_settings):
            s1_seeds = per_seed_s1[setting_idx]
            if s1_seeds is None:
                continue
            color = color_list[setting_idx]
            x = x_positions + setting_offsets[setting_idx]

            s1_means, s1_ci_lo, s1_ci_hi = _compute_sorted_ci(s1_seeds)

            valid1 = ~np.isnan(s1_means)
            if np.any(valid1):
                lo1 = s1_means[valid1] - s1_ci_lo[valid1]
                hi1 = s1_ci_hi[valid1] - s1_means[valid1]
                # Only add label for the last timestep (full opacity)
                label = labels[setting_idx] if time_i == n_times - 1 else None
                marker = marker_list[setting_idx] if marker_list is not None else 'o'
                ax.scatter(x[valid1], s1_means[valid1], color=color, s=20,
                           alpha=alphas[time_i], zorder=4, label=label,
                           marker=marker)
                ax.errorbar(x[valid1], s1_means[valid1], yerr=[lo1, hi1],
                            fmt='none', ecolor=color, alpha=alphas[time_i] * 0.5,
                            capsize=3, linewidth=1, zorder=3)

    # Timestep annotation
    time_str = ", ".join(str(idx) for idx in frontier_indices)
    ax.annotate(f"Timesteps: {time_str} (light\u2192dark)", xy=(0.02, 0.98),
                xycoords='axes fraction', fontsize=fontsize - 1, color='gray',
                verticalalignment='top')

    all_ref_names = [series2_name] + ([series3_name] if series3_name else [])
    title_refs = ', '.join(all_ref_names)
    ax.set_xlabel(f'Target Sequence Index (sorted by descending {sort_series_name})',
                  fontsize=fontsize)
    ax.set_ylabel('Log probability', fontsize=fontsize)
    ax.set_title(f'{series1_name} vs {title_refs} Over Time (per target sequence)',
                 fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(np.arange(1, n_samples + 1), fontsize=max(4, fontsize - 2))
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Over-time lollipop plot saved to {figname}")


def _collect_f_q_sample_rank_data(n_settings, log_q_data, log_p_data, log_phi_data, n_ranks):
    """For each setting/seed, rank f_q samples per-prompt by log_q and concatenate results.

    Ranking is done independently within each prompt (so prompt identity is preserved in
    the rank ordering), then the per-prompt top-k lists are concatenated. The total number
    of rank positions is P * min(n_ranks, n_samples_per_prompt), where P is the number
    of prompts with data.

    Args:
        log_q_data, log_p_data, log_phi_data: List (settings) of list (seeds) of
            list (T timesteps) of list (P prompts) of 1D numpy array (n_samples,) or None.
        n_ranks: Maximum number of top samples to keep per prompt.

    Returns:
        rank_data: dict setting_idx -> list of (q_by_rank, p_by_rank, phi_by_rank) arrays,
                   one tuple per seed. Length = P * min(n_ranks, n_samples_per_prompt).
        n_ranks_actual: int, length of the rank arrays (consistent across settings).
        prompt_boundaries: list of int, x-axis positions where each new prompt begins,
                           for drawing dividers in the plot.
    """
    rank_data = {i: [] for i in range(n_settings)}

    for setting_idx in range(n_settings):
        q_seeds = log_q_data[setting_idx] if setting_idx < len(log_q_data) else []
        p_seeds = log_p_data[setting_idx] if setting_idx < len(log_p_data) else []
        phi_seeds = log_phi_data[setting_idx] if setting_idx < len(log_phi_data) else []

        n_seeds = max(len(q_seeds), len(p_seeds), len(phi_seeds))
        for seed_j in range(n_seeds):
            q_seed = q_seeds[seed_j] if seed_j < len(q_seeds) else []
            p_seed = p_seeds[seed_j] if seed_j < len(p_seeds) else []
            phi_seed = phi_seeds[seed_j] if seed_j < len(phi_seeds) else []

            # Take the final timestep from each seed's data
            q_t = q_seed[-1] if q_seed else None
            p_t = p_seed[-1] if p_seed else None
            phi_t = phi_seed[-1] if phi_seed else None

            if q_t is None:
                continue

            # Per-prompt top-k, then concatenate across prompts
            q_all, p_all, phi_all = [], [], []
            n_prompts = len(q_t)
            for prompt_idx in range(n_prompts):
                q_arr = np.asarray(q_t[prompt_idx]) if q_t[prompt_idx] is not None else None
                p_arr = np.asarray(p_t[prompt_idx]) if (p_t is not None and prompt_idx < len(p_t)
                                                         and p_t[prompt_idx] is not None) else None
                phi_arr = np.asarray(phi_t[prompt_idx]) if (phi_t is not None and prompt_idx < len(phi_t)
                                                              and phi_t[prompt_idx] is not None) else None

                if q_arr is None or len(q_arr) == 0:
                    continue

                # n_ranks=None means no cap: take all samples.
                k = len(q_arr) if n_ranks is None else min(n_ranks, len(q_arr))
                top_idx = np.argsort(q_arr)[::-1][:k]

                q_all.append(q_arr[top_idx])
                p_all.append(p_arr[top_idx] if p_arr is not None else np.full(k, np.nan))
                phi_all.append(phi_arr[top_idx] if phi_arr is not None else np.full(k, np.nan))

            if not q_all:
                continue

            rank_data[setting_idx].append((
                np.concatenate(q_all),
                np.concatenate(p_all),
                np.concatenate(phi_all),
            ))

    n_ranks_actual = max(
        (len(tup[0]) for seeds in rank_data.values() for tup in seeds),
        default=0
    )

    # Compute prompt boundary x-positions from the first available seed of the first setting
    # (all settings/seeds should have the same prompt structure).
    prompt_boundaries = []
    for seeds in rank_data.values():
        if seeds:
            q_seed_example = None
            for setting_idx in range(n_settings):
                q_seeds = log_q_data[setting_idx] if setting_idx < len(log_q_data) else []
                if q_seeds and q_seeds[0]:
                    q_t_ex = q_seeds[0][-1]  # final timestep of first seed
                    if q_t_ex:
                        pos = 0
                        for prompt_idx, q_arr in enumerate(q_t_ex):
                            if q_arr is not None and len(q_arr) > 0:
                                if prompt_idx > 0:
                                    prompt_boundaries.append(pos)
                                k = len(q_arr) if n_ranks is None else min(n_ranks, len(q_arr))
                                pos += k
                        break
                if prompt_boundaries:
                    break
            break

    return rank_data, n_ranks_actual, prompt_boundaries


def plot_top_q_samples_ranked_lollipop(
    figname, labels,
    log_q_data, log_p_data, log_phi_data,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_ranks=10,
    log_phi_label=None,
    marker_list=None,
):
    """
    Lollipop chart of top-n q-drawn samples ranked by log_q, showing log_q, log_p, and log_phi.

    For each setting and seed, flattens f_q samples across prompts at the final evaluation
    timestep, sorts them by log_q (log probability under the proposal q) descending, and
    records (log_q, log_p, log p + β·r) at each rank. Averages across seeds with bootstrap CIs.

    Three markers per rank per setting:
      - Filled circle: log q  (proposal log probability)
      - Hollow square: log p  (prior log probability)
      - Hollow diamond: log_phi_label  (defaults to log tilde sigma)

    Args:
        log_q_data, log_p_data, log_phi_data: List (settings) of list (seeds) of
            list (T timesteps) of list (P prompts) of 1D numpy array (n_samples,) or None.
            log_phi_data should contain the target values (log_p + beta*r, or normalized log sigma).
        log_phi_label: Legend label for the diamond markers. Defaults to
            r'$\\log \\tilde{\\sigma} = \\log p + \\beta r$'.
    """
    if log_phi_label is None:
        log_phi_label = r'$\log \tilde{\sigma} = \log p + \beta r$'
    n_settings = len(labels)
    rank_data, n_ranks_actual, prompt_boundaries = _collect_f_q_sample_rank_data(
        n_settings, log_q_data, log_p_data, log_phi_data, n_ranks)

    if n_ranks_actual == 0:
        print(f"No rank data for top-q samples ranked lollipop, skipping {figname}")
        return

    x_positions = np.arange(n_ranks_actual)
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots()

    for setting_idx in range(n_settings):
        seeds = rank_data[setting_idx]
        if not seeds:
            continue

        q_means = np.full(n_ranks_actual, np.nan)
        q_ci_lo = np.full(n_ranks_actual, np.nan)
        q_ci_hi = np.full(n_ranks_actual, np.nan)
        p_means = np.full(n_ranks_actual, np.nan)
        p_ci_lo = np.full(n_ranks_actual, np.nan)
        p_ci_hi = np.full(n_ranks_actual, np.nan)
        phi_means = np.full(n_ranks_actual, np.nan)
        phi_ci_lo = np.full(n_ranks_actual, np.nan)
        phi_ci_hi = np.full(n_ranks_actual, np.nan)

        for rank in range(n_ranks_actual):
            q_vals = np.array([s[0][rank] for s in seeds if rank < len(s[0])])
            p_vals = np.array([s[1][rank] for s in seeds
                               if rank < len(s[1]) and not np.isnan(s[1][rank])])
            phi_vals = np.array([s[2][rank] for s in seeds
                                 if rank < len(s[2]) and not np.isnan(s[2][rank])])

            q_means[rank], q_ci_lo[rank], q_ci_hi[rank] = _bootstrap_mean_ci(q_vals, n_bootstrap_draws)
            if len(p_vals) > 0:
                p_means[rank], p_ci_lo[rank], p_ci_hi[rank] = _bootstrap_mean_ci(p_vals, n_bootstrap_draws)
            if len(phi_vals) > 0:
                phi_means[rank], phi_ci_lo[rank], phi_ci_hi[rank] = _bootstrap_mean_ci(phi_vals, n_bootstrap_draws)

        x_pos = x_positions + setting_offsets[setting_idx]
        color = color_list[setting_idx]

        # log phi: hollow diamond with CI
        valid_phi = ~np.isnan(phi_means)
        if valid_phi.any():
            phi_lo = phi_means[valid_phi] - phi_ci_lo[valid_phi]
            phi_hi = phi_ci_hi[valid_phi] - phi_means[valid_phi]
            ax.scatter(x_pos[valid_phi], phi_means[valid_phi], color=color, s=16, zorder=3,
                       marker='D', facecolors='none', edgecolors=color, linewidths=1.2)
            ax.errorbar(x_pos[valid_phi], phi_means[valid_phi],
                        yerr=[phi_lo, phi_hi],
                        fmt='none', ecolor=color, alpha=0.2,
                        capsize=2, linewidth=0.8, zorder=2)

        # log p: hollow square with CI
        valid_p = ~np.isnan(p_means)
        if valid_p.any():
            p_lo = p_means[valid_p] - p_ci_lo[valid_p]
            p_hi = p_ci_hi[valid_p] - p_means[valid_p]
            ax.scatter(x_pos[valid_p], p_means[valid_p], color=color, s=16, zorder=3,
                       marker='s', facecolors='none', edgecolors=color, linewidths=1.2)
            ax.errorbar(x_pos[valid_p], p_means[valid_p],
                        yerr=[p_lo, p_hi],
                        fmt='none', ecolor=color, alpha=0.2,
                        capsize=2, linewidth=0.8, zorder=2)

        # log q: filled setting-specific marker with CI
        valid_q = ~np.isnan(q_means)
        if valid_q.any():
            q_lo = q_means[valid_q] - q_ci_lo[valid_q]
            q_hi = q_ci_hi[valid_q] - q_means[valid_q]
            q_marker = marker_list[setting_idx] if marker_list is not None else 'o'
            ax.scatter(x_pos[valid_q], q_means[valid_q], color=color, s=20, zorder=4,
                       marker=q_marker)
            ax.errorbar(x_pos[valid_q], q_means[valid_q],
                        yerr=[q_lo, q_hi],
                        fmt='none', ecolor=color, alpha=0.2,
                        capsize=3, linewidth=1, zorder=3)

    # Legend: per-setting colored lines + marker-type key
    legend_handles = []
    if marker_list is None:
        legend_handles.append(Line2D([], [], marker='o', color='black', markersize=5,
                                     linestyle='None', label=r'$\log q$'))
    legend_handles.append(Line2D([], [], marker='s', color='black', markersize=4,
                                 markerfacecolor='none', markeredgewidth=1.2,
                                 linestyle='None', label=r'$\log p$'))
    legend_handles.append(Line2D([], [], marker='D', color='black', markersize=4,
                                 markerfacecolor='none', markeredgewidth=1.2,
                                 linestyle='None', label=log_phi_label))
    for setting_idx in range(n_settings):
        if not rank_data[setting_idx]:
            continue
        if marker_list is not None:
            legend_handles.append(Line2D([], [], color=color_list[setting_idx], linewidth=2,
                                         marker=marker_list[setting_idx], markersize=5,
                                         label=labels[setting_idx]))
        else:
            legend_handles.append(Line2D([], [], color=color_list[setting_idx], linewidth=2,
                                         label=labels[setting_idx]))

    # Vertical dividers between prompts
    for boundary_x in prompt_boundaries:
        ax.axvline(boundary_x - 0.5, color='black', linewidth=0.7, linestyle=':', alpha=0.5)

    n_prompts_shown = len(prompt_boundaries) + 1
    xlabel = (f'Rank within prompt (top {n_ranks} per prompt, {n_prompts_shown} prompt(s); '
              f'dashed lines separate prompts)')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Top {n_ranks} q Samples per Prompt by Log q: log q, log p, log p + β·r',
                 fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(r + 1) for r in range(n_ranks_actual)], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(handles=legend_handles, fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Top-q samples ranked lollipop saved to {figname}")


def compute_iwae_vs_n_curves(f_qs_by_prompt, g_qs_by_prompt, n_values, n_bootstrap=None, rng_seed=42):
    """Compute IWAE lower and upper bounds as a function of N *total* samples, averaged over prompts.

    N is the total sample count for both bounds:
      LB(N): uses N proposal samples from q.
        LB(N) = log(1/N * sum_{i=1}^{N} w_i)   where w_i = p0(x)*phi(x)/q(x)
      UB(N): uses 1 exact target sample + (N-1) proposal samples from q.
        UB(N) = log(1/N * (w_target + sum_{i=1}^{N-1} w_i))

    Both are on the same x-axis in terms of total samples used, so at N=4 the UB uses
    1 exact target + 3 proposal samples, and the LB uses 4 proposal samples.

    For LB: M independent draws each pick N proposal samples (without replacement) from the saved
    pool and compute IWAE LB; results are averaged. M = number of available target samples.

    For UB: iterates over every target sample exactly once. For each target sample m, one
    independent draw of (N-1) proposal samples is made; the M resulting UB values are averaged.
    This gives every target sample equal weight with no random target selection.

    Results are averaged across all valid prompts.

    Args:
        f_qs_by_prompt: list of numpy arrays of shape (N_max,), one per prompt.  Each entry is a
            vector of log-importance-weights log(p0(x)*phi(x)/q(x)) for samples from q.
        g_qs_by_prompt: list of numpy arrays of shape (M,), one per prompt.  M target samples'
            log-weights; all M are used (each bootstrap draw picks one at random for the UB).
        n_values: list of int total sample counts to evaluate (must all be >= 1).
        n_bootstrap: number of independent draws used to estimate E[LB(N)] and E[UB(N)] at each N.
            Each draw uses exactly N samples (x-axis value). Defaults to None, which sets it to
            the number of available target samples M (so every target sample is represented in the
            UB estimate). For N=1 UB, averages directly over all M target samples exactly.
        rng_seed: base integer seed; each prompt uses rng_seed + prompt_index for independence.

    Returns:
        lb_curve: numpy array of shape (len(n_values),), mean LB across valid prompts.
        ub_curve: numpy array of shape (len(n_values),), mean UB across valid prompts.
        Both are None if no valid prompts are available.
    """
    from scipy.special import logsumexp

    lb_curves_per_prompt = []
    ub_curves_per_prompt = []

    for p_idx, (f_qs, g_qs) in enumerate(zip(f_qs_by_prompt, g_qs_by_prompt)):
        if f_qs is None or len(f_qs) == 0 or g_qs is None or len(g_qs) == 0:
            continue
        f_qs = np.asarray(f_qs, dtype=float)
        g_qs = np.asarray(g_qs, dtype=float)
        N_max = len(f_qs)
        M = len(g_qs)   # number of available target samples
        n_boot = M if n_bootstrap is None else n_bootstrap
        rng = np.random.default_rng(rng_seed + p_idx)

        lb_at_n = []
        ub_at_n = []
        for N in n_values:
            # ---- LB: N proposal samples ----
            if N >= N_max:
                lb_val = float(logsumexp(f_qs) - np.log(N_max))
            else:
                lb_boot = []
                for _ in range(n_boot):
                    idx = rng.choice(N_max, size=N, replace=False)
                    lb_boot.append(float(logsumexp(f_qs[idx]) - np.log(N)))
                lb_val = float(np.mean(lb_boot))

            # ---- UB: 1 exact target sample (drawn from all M) + (N-1) proposal samples ----
            n_prop_ub = N - 1
            if n_prop_ub <= 0:
                # N=1: UB = each target sample's log-weight averaged over all M.
                # No proposals needed; exact mean over all available target samples.
                ub_val = float(np.mean(g_qs))
            elif n_prop_ub >= N_max:
                # All N_max proposals available; still average over all M target samples.
                ub_boot = []
                for m in range(M):
                    all_w = np.concatenate([[g_qs[m]], f_qs])
                    ub_boot.append(float(logsumexp(all_w) - np.log(N_max + 1)))
                ub_val = float(np.mean(ub_boot))
            else:
                # One draw of (N-1) proposal samples per target sample; average over all M.
                ub_boot = []
                for m in range(M):
                    idx = rng.choice(N_max, size=n_prop_ub, replace=False)
                    all_w = np.concatenate([[g_qs[m]], f_qs[idx]])
                    ub_boot.append(float(logsumexp(all_w) - np.log(N)))
                ub_val = float(np.mean(ub_boot))

            lb_at_n.append(lb_val)
            ub_at_n.append(ub_val)

        lb_curves_per_prompt.append(np.array(lb_at_n))
        ub_curves_per_prompt.append(np.array(ub_at_n))

    if not lb_curves_per_prompt:
        return None, None

    return np.mean(lb_curves_per_prompt, axis=0), np.mean(ub_curves_per_prompt, axis=0)
