import torch
import re
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


# Marker constants for bonus types (legacy, kept for backward compatibility)
MARKER_NO_BONUS = "x"
MARKER_CFN = "P"
MARKER_MIXTURE = "^"
MARKER_EXACT_COUNT = "D"

# Marker constants for loss types (used by semantic styling)
MARKER_CTL = "o"
MARKER_CTLN = "x"
MARKER_LOSS_UNKNOWN = "D"
MARKER_EXACT_COUNT = "*"

# Markers cycled across seeds in individual-seed plots
SEED_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


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

    Returns a dict with:
        loss_type: "CTL", "CTLN", or None
        bonus_type: "cfn", "exact_count", "mixture", or "none"
        mixture_variant: "mixture", "q_independent", "q_half", or None (only set when bonus_type == "mixture")
        mixture_other_model: "best", "first", "lag", or None (only set when bonus_type == "mixture")
        mixture_lag_steps: int or None (only set when mixture_other_model == "lag")
        learning_rate: float or None (sampling actor LR from _al pattern)
        cfn_alpha: float or None (bonus_alpha from _cf<value> pattern)
    """
    # Loss type
    if "_ctln_" in prefix:
        loss_type = "CTLN"
    elif "_ctl_" in prefix:
        loss_type = "CTL"
    else:
        loss_type = None

    # Bonus type (same order of specificity as generate_labels_from_prefixes)
    if "cfn" in prefix or re.search(r'_cf([\d.]+)', prefix):
        bonus_type = "cfn"
    elif "_count" in prefix or re.search(r'_c([\d.]+)(?![a-z])', prefix):
        bonus_type = "exact_count"
    elif "_mix" in prefix:
        bonus_type = "mixture"
    else:
        bonus_type = "none"

    # Mixture variant (only when bonus_type is mixture)
    mixture_variant = None
    mixture_other_model = None
    mixture_lag_steps = None
    if bonus_type == "mixture":
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

    # Learning rate (sampling actor LR)
    al_match = re.search(r'_al([\d.e-]+)', prefix)
    learning_rate = float(al_match.group(1)) if al_match else None

    # CFN alpha (bonus_alpha)
    cfn_alpha = None
    if bonus_type == "cfn":
        cf_match = re.search(r'_cf([\d.]+)', prefix)
        if cf_match:
            cfn_alpha = float(cf_match.group(1))

    return {
        "loss_type": loss_type,
        "bonus_type": bonus_type,
        "mixture_variant": mixture_variant,
        "mixture_other_model": mixture_other_model,
        "mixture_lag_steps": mixture_lag_steps,
        "learning_rate": learning_rate,
        "cfn_alpha": cfn_alpha,
    }


def generate_visual_style_from_prefixes(load_prefixes_to_use):
    """
    Generate semantically consistent (color, marker, linestyle) lists from prefix lists.

    Visual encoding:
        - Color hue: bonus/experiment type
            - No bonus (baselines): Greys
            - CFN: blue / green / purple depending on alpha value
            - Mixture (q_independent): Oranges
            - Mixture (mixture / other): Reds
            - Exact count: Teals (cm.YlGnBu)
        - Color shade: learning rate rank (lighter=smaller LR, darker=larger LR; mid if only one)
        - Marker shape: loss type (o=CTL, s=CTLN, D=unknown)
        - Line style: CFN alpha value (solid=non-CFN or single alpha; distinct styles per unique alpha)

    Args:
        load_prefixes_to_use: List of lists of prefixes (one inner list per experiment)

    Returns:
        (color_list, marker_list, linestyle_list) — parallel lists, one entry per experiment
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

    # 4a. CFN alpha -> color family: cycle through blue, green, purple for distinct alphas
    unique_alphas = sorted(set(p["cfn_alpha"] for p in props_list if p["cfn_alpha"] is not None))
    cfn_alpha_cmaps = [cm.Blues, cm.Greens, cm.Purples, cm.BuGn, cm.GnBu, cm.BuPu, cm.YlGnBu, cm.PuBuGn]
    if unique_alphas:
        cfn_alpha_cmap = {a: cfn_alpha_cmaps[i % len(cfn_alpha_cmaps)] for i, a in enumerate(unique_alphas)}
    else:
        cfn_alpha_cmap = {}

    # 4b. Mixture lag_steps -> color: assign a fixed shade per (variant, lag_steps) key spread
    #     across 0.25-0.9 so different lag values are clearly distinguishable.
    #     qi variants use a single orange-red colormap; other variants use a purple-pink colormap.
    #     Shade is spread by rank among all mixture keys, NOT by LR (LR uses linestyle instead).
    unique_mixture_keys = sorted(
        set((p["mixture_variant"], p["mixture_lag_steps"])
            for p in props_list if p["bonus_type"] == "mixture"),
        key=lambda x: (x[0] or "", x[1] if x[1] is not None else -1),
    )
    MIXTURE_SHADE_LO, MIXTURE_SHADE_HI = 0.25, 0.90
    n_mix = len(unique_mixture_keys)
    mixture_shade_map = {}
    mixture_cmap_map = {}
    for i, (variant, lag) in enumerate(unique_mixture_keys):
        shade = MIXTURE_SHADE_LO if n_mix == 1 else MIXTURE_SHADE_LO + (MIXTURE_SHADE_HI - MIXTURE_SHADE_LO) * i / (n_mix - 1)
        mixture_shade_map[(variant, lag)] = shade
        if variant == "q_independent":
            mixture_cmap_map[(variant, lag)] = cm.YlOrRd
        else:
            mixture_cmap_map[(variant, lag)] = cm.RdPu

    # 5. Colormap and shade range per bonus category
    #    Shade range [lo, hi] samples the colormap avoiding very light/very dark ends.
    shade_range_default = (0.4, 0.90)

    # Loss type -> marker
    loss_marker = {
        "CTL": MARKER_CTL,
        "CTLN": MARKER_CTLN,
        None: MARKER_LOSS_UNKNOWN,
    }

    # 6. Compute a "color hue key" for each experiment so we can cycle linestyles
    #    within groups that share the same hue. This ensures experiments with
    #    identical colors (same bonus type / CFN alpha / mixture variant) are
    #    still distinguishable in time-series plots where markers aren't shown.
    def _color_hue_key(props):
        bonus = props["bonus_type"]
        if bonus == "cfn":
            return ("cfn", props["cfn_alpha"])
        elif bonus == "mixture":
            return ("mixture", props["mixture_variant"], props["mixture_lag_steps"])
        else:
            return (bonus,)

    hue_group_counter = {}  # hue_key -> running count of experiments seen

    # 7. Build output lists
    color_list = []
    marker_list = []
    linestyle_list = []

    for props in props_list:
        # Determine colormap based on bonus type (and variant / alpha)
        bonus = props["bonus_type"]
        if bonus == "none":
            cmap = cm.Greys
        elif bonus == "cfn":
            cmap = cfn_alpha_cmap.get(props["cfn_alpha"], cm.Blues)
        elif bonus == "mixture":
            mix_key = (props["mixture_variant"], props["mixture_lag_steps"])
            cmap = mixture_cmap_map.get(mix_key, cm.Oranges)
        elif bonus == "exact_count":
            cmap = cm.YlGnBu
        else:
            cmap = cm.Greys

        # Shade by LR rank within the type's shade range.
        # For mixture: base shade is set by mixture key rank; LR adjusts within a narrow
        # window around the base so both dimensions are visible.
        # For all others: shade spans the full default range by LR rank.
        if props["learning_rate"] is not None and props["learning_rate"] in lr_position:
            t = lr_position[props["learning_rate"]]
        else:
            t = 0.5

        if bonus == "mixture":
            mix_key = (props["mixture_variant"], props["mixture_lag_steps"])
            base_shade = mixture_shade_map.get(mix_key, 0.6)
            # Window = 35% of the gap between adjacent mixture keys, so LR ticks never
            # overlap with neighbouring mixture key colours.
            mix_spacing = (MIXTURE_SHADE_HI - MIXTURE_SHADE_LO) / max(n_mix - 1, 1)
            lr_half_window = mix_spacing * 0.35
            # t in [0,1] -> adjustment in [-lr_half_window, +lr_half_window]
            shade = np.clip(base_shade + lr_half_window * (2 * t - 1), 0.10, 0.95)
        else:
            lo, hi = shade_range_default
            shade = lo + t * (hi - lo)
        color_list.append(cmap(shade))

        # Marker: exact_count always gets a star; otherwise determined by loss type
        if props["bonus_type"] == "exact_count":
            marker_list.append(MARKER_EXACT_COUNT)
        else:
            marker_list.append(loss_marker.get(props["loss_type"], MARKER_LOSS_UNKNOWN))

        # Linestyle: cycle within each color-hue group so same-hue experiments
        # are distinguishable even without markers (e.g. in time-series plots)
        hue_key = _color_hue_key(props)
        idx = hue_group_counter.get(hue_key, 0)
        hue_group_counter[hue_key] = idx + 1
        linestyle_list.append(linestyle_options[idx % len(linestyle_options)])

    return color_list, marker_list, linestyle_list


def generate_labels_from_prefixes(load_prefixes_to_use):
    """
    Generate labels from prefix lists based on the naming logic.
    
    This function extracts information from prefixes to create human-readable labels.
    It handles different run types: "Exact Count", "Coin Flip Net", and "No Exploration Bonus".
    For all run types, LR (q) (sampling actor / actor learning rate from _al) is included when present.
    For Coin Flip Net runs, it also extracts coin flip LR, updates, head_std, prior_std, etc.
    
    Supports both old and new abbreviated naming conventions.
    
    Args:
        load_prefixes_to_use: List of lists of prefixes (each inner list contains prefixes for one series)
    
    Returns:
        List of label strings, one for each prefix list
    """
    labels = []
    for a in load_prefixes_to_use:
        prefix = a[0]
        # Use shared parsing for loss type and bonus type detection
        props = _parse_experiment_properties(prefix)
        loss_type_str = props["loss_type"]  # "CTL", "CTLN", or None

        # Map bonus_type to run_type string used below
        _bonus_to_run_type = {
            "cfn": "Coin Flip Net",
            "exact_count": "Exact Count",
            "mixture": "Mixture",
            "none": "No Exploration Bonus",
        }
        run_type = _bonus_to_run_type[props["bonus_type"]]
        
        # If it's a coin flip net, extract additional parameters
        if run_type == "Coin Flip Net":
            # Build parenthetical descriptor: CFN (d<dim>, <n> upd., after/before, online, pri)
            paren_parts = []

            # Extract coin_flip_dim from _cd pattern
            cd_match = re.search(r'_cd(\d+)', prefix)
            if cd_match:
                paren_parts.append(f"d{cd_match.group(1)}")

            # Extract cfus/cfu (updates) - support both old and new
            cfus_match = re.search(r'_cfus(\d+)', prefix) or re.search(r'_cfu(\d+)', prefix)
            paren_parts.append(f"{cfus_match.group(1)} upd." if cfus_match else "1 upd.")

            # Check for "after"/"before" or new abbreviations "af"/"bf"
            if "after" in prefix or "_af" in prefix:
                paren_parts.append("after")
            elif "before" in prefix or "_bf" in prefix:
                paren_parts.append("before")

            if "firstonline" in prefix or "_fo" in prefix:
                paren_parts.append("online")
            if "pri" in prefix or "_pr" in prefix:
                paren_parts.append("pri")

            run_type_str = f"CFN ({', '.join(paren_parts)})" if paren_parts else "CFN"
            label_parts = [run_type_str]

            # Extract bonus_alpha from _cf pattern (encoded as _cf followed by value)
            cf_match = re.search(r'_cf([\d.]+)', prefix)
            if cf_match:
                label_parts.append(f"alpha={cf_match.group(1)}")

            # Extract cfhis/cfh (coin flip head init std) - support both old and new
            cfhis_match = re.search(r'_cfhis([\d.e-]+)', prefix) or re.search(r'_cfh([\d.e-]+)', prefix)
            if cfhis_match:
                label_parts.append(f"head_std={cfhis_match.group(1)}")

            # Extract fpis/fp (frozen prior init std) - support both old and new
            fpis_match = re.search(r'_fpis([\d.e-]+)', prefix) or re.search(r'_fp([\d.e-]+)', prefix)
            if fpis_match:
                label_parts.append(f"prior_std={fpis_match.group(1)}")

            # Check for coin_flip_linear_bias (old _cfbias or new _cfb)
            if "_cfbias" in prefix or "_cfb" in prefix:
                label_parts.append("bias")

            # Extract sampling actor LR from _al (actor_learning_rate in harmlessness = sampling_actor)
            al_match = re.search(r'_al([\d.e-]+)', prefix)
            if al_match:
                label_parts.append(f"{al_match.group(1)} LR (q)")

            # Extract cflr/cfr (learning rate) - support both old and new
            cflr_match = re.search(r'_cflr([\d.e-]+)', prefix) or re.search(r'_cfr([\d.e-]+)', prefix)
            if cflr_match:
                label_parts.append(f"{cflr_match.group(1)} LR (CF)")

            # Extract batch_size (encoded as _tbs or _tb followed by value) - support both old and new
            tbs_match = re.search(r'_tbs(\d+)', prefix) or re.search(r'_tb(\d+)', prefix)
            if tbs_match:
                label_parts.append(f"batch={tbs_match.group(1)}")

            # Architecture
            if "sepnn" in prefix or "_cfsn" in prefix:
                label_parts.append("Sep. NN")
            elif "cflsib" in prefix or "_cfs" in prefix:
                label_parts.append("Lin. Static Base")
            elif "cfllq" in prefix or "_cfq" in prefix:
                label_parts.append("Lin. on q")
            elif "cfllp" in prefix or "_cfl" in prefix:
                label_parts.append("Lin. on p")

        elif run_type == "Exact Count":
            label_parts = ["EC"]

            # Extract bonus_alpha (encoded as _count or _c followed by value)
            count_match = re.search(r'_count([\d.]+)', prefix) or re.search(r'_c([\d.]+)', prefix)
            if count_match:
                label_parts.append(f"alpha={count_match.group(1)}")

            # Extract LR (q) / sampling actor LR from _al
            al_match = re.search(r'_al([\d.e-]+)', prefix)
            if al_match:
                label_parts.append(f"{al_match.group(1)} LR (q)")

            # # Extract num_episodes (encoded as _epi or _e followed by value) - support both old and new
            # epi_match = re.search(r'_epi(\d+)', prefix) or re.search(r'_e(\d+)', prefix)
            # if epi_match:
            #     label_parts.append(f"ep={epi_match.group(1)}")

            # Extract batch_size (encoded as _tbs or _tb followed by value) - support both old and new
            tbs_match = re.search(r'_tbs(\d+)', prefix) or re.search(r'_tb(\d+)', prefix)
            if tbs_match:
                label_parts.append(f"batch={tbs_match.group(1)}")

        elif run_type == "Mixture":
            # Extract mixture optimization variant from _mix abbreviation
            mix_variant_map = {"mx": "mixture", "qi": "q_independent", "qh": "q_half"}
            mix_match = re.search(r'_mix([a-z]+)', prefix)
            mix_variant = mix_variant_map.get(mix_match.group(1), mix_match.group(1)) if mix_match else "unknown"

            # Extract other-model strategy: _first, _lag<N>, or absent (= best)
            other_model = props["mixture_other_model"]
            if other_model == "first":
                other_str = ", first"
            elif other_model == "lag":
                lag_steps = props["mixture_lag_steps"]
                other_str = f", lag={lag_steps}" if lag_steps is not None else ", lag"
            else:
                # "best" is the original default; omit to keep labels short for backward compat
                other_str = ""

            label_parts = [f"Mixture ({mix_variant}{other_str})"]

            # Extract LR (q) / sampling actor LR from _al
            al_match = re.search(r'_al([\d.e-]+)', prefix)
            if al_match:
                label_parts.append(f"{al_match.group(1)} LR (q)")

            # Extract batch_size (encoded as _tbs or _tb followed by value) - support both old and new
            tbs_match = re.search(r'_tbs(\d+)', prefix) or re.search(r'_tb(\d+)', prefix)
            if tbs_match:
                label_parts.append(f"batch={tbs_match.group(1)}")

        else:
            label_parts = ["No Bonus"]

            # Extract LR (q) / sampling actor LR from _al
            al_match = re.search(r'_al([\d.e-]+)', prefix)
            if al_match:
                label_parts.append(f"{al_match.group(1)} LR (q)")

            # # Extract num_episodes (encoded as _epi or _e followed by value) - support both old and new
            # epi_match = re.search(r'_epi(\d+)', prefix) or re.search(r'_e(\d+)', prefix)
            # if epi_match:
            #     label_parts.append(f"ep={epi_match.group(1)}")

            # Extract batch_size (encoded as _tbs or _tb followed by value) - support both old and new
            tbs_match = re.search(r'_tbs(\d+)', prefix) or re.search(r'_tb(\d+)', prefix)
            if tbs_match:
                label_parts.append(f"batch={tbs_match.group(1)}")

        # Append base actor LR if non-zero (encoded as _bl followed by value)
        bl_match = re.search(r'_bl([\d.e-]+)', prefix)
        if bl_match:
            bl_val = float(bl_match.group(1))
            if bl_val != 0.0:
                label_parts.append(f"{bl_match.group(1)} LR (p)")

        # Prepend CTL/CTLN loss type if detected
        if loss_type_str is not None:
            label_parts.insert(0, loss_type_str)

        # Check for mixeval prefix
        if prefix.startswith("f_q_g_q_iwae_bounds_mixeval"):
            label_parts.append("(mixeval)")

        # For info_eval prefixes, include the beta value.
        # Prefixes use abbreviated _b<value> (e.g. _b-10.0); _beta<value> also supported.
        # Use negative lookahead to avoid matching _bl (base LR) or other _b<letter> patterns.
        if prefix.startswith("info_eval"):
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
    plt.xticks(x_positions, [_token_label(token_id, tokenizer) for token_id in top_n_tokens], fontsize=fontsize-1)
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

    fig, ax = plt.subplots()

    # Horizontal spacing: settings are offset within each token position
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    # Compute target mean per token (pooled across all settings/seeds — target is shared)
    target_means = np.full(n_tokens, np.nan)
    for token_idx, token_id in enumerate(top_n_tokens):
        all_target_vals = []
        for vals in target_by_setting.get(token_id, {}).values():
            all_target_vals.extend(vals)
        if all_target_vals:
            target_means[token_idx] = np.mean(all_target_vals)

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
            ax.errorbar(
                x_pos[valid], q_means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt='o', color=color_list[setting_idx], markersize=5,
                capsize=3, linewidth=1, label=labels[setting_idx], zorder=4,
            )

    ax.set_xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    time_label = " (final step)" if final_only else " (avg over time)"
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Log Prob (target vs q vs base){time_label}', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_token_label(token_id, tokenizer) for token_id in top_n_tokens], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Lollipop chart saved to {figname}")


def plot_top_tokens_lollipop_individual(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """
    Individual-seed version of plot_top_tokens_lollipop (final step only).

    Instead of bootstrap-aggregated means with CIs, each seed is plotted as a separate dot.
    Seeds share the same color/marker per setting; one legend entry per setting.
    """
    if precomputed_token_data is not None:
        target_by_setting, q_by_setting, _, all_sorted_tokens = precomputed_token_data
        top_n_tokens = all_sorted_tokens[:n_top_tokens]
    else:
        target_by_setting, q_by_setting, _, top_n_tokens = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens, final_only=True)
    if top_n_tokens is None:
        print("Warning: Cannot create individual lollipop chart.")
        return

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    fig, ax = plt.subplots()

    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    # Compute target mean per token (pooled across all settings/seeds — target is shared)
    target_means = np.full(n_tokens, np.nan)
    for token_idx, token_id in enumerate(top_n_tokens):
        all_target_vals = []
        for vals in target_by_setting.get(token_id, {}).values():
            all_target_vals.extend(vals)
        if all_target_vals:
            target_means[token_idx] = np.mean(all_target_vals)

    # Draw target markers
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

    # Draw individual seed dots for each setting, with per-seed markers
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
                ax.scatter(
                    x_base, val,
                    color=color_list[setting_idx], s=20, alpha=0.7,
                    marker=marker, label=label, zorder=4,
                )
                label_added = True

    ax.set_xlabel('Token' if tokenizer is not None else 'Token ID', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Log Prob (individual seeds, final step)', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_token_label(token_id, tokenizer) for token_id in top_n_tokens], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Individual lollipop chart saved to {figname}")


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

    fig, ax = plt.subplots(figsize=(max(8, n_tokens * 0.9), 5))

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
            ax.errorbar(
                x_base[valid], means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt='o', color=color_list[setting_idx], markersize=4,
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
    ax.set_xticklabels([_token_label(token_id, tokenizer) for token_id in top_n_tokens], fontsize=fontsize - 1)
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

    fig, ax = plt.subplots(figsize=(max(8, n_tokens * 0.9), 5))

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
            ax.errorbar(
                x_base[valid], means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt='o', color=color_list[setting_idx], markersize=4,
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
    ax.set_xticklabels([_token_label(token_id, tokenizer) for token_id in top_n_tokens], fontsize=fontsize - 1)
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
        print("Warning: No cumulative_q_sample_counts found. Skipping sample counts final individual plot.")
        return

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    fig, ax = plt.subplots(figsize=(max(8, n_tokens * 0.9), 5))

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
    ax.set_xticklabels([_token_label(token_id, tokenizer) for token_id in top_n_tokens], fontsize=fontsize - 1)
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
            ax.plot(timesteps[valid], means[valid], color=color_list[setting_idx],
                    label=labels[setting_idx], linewidth=1.5)
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
            ax.plot(timesteps[valid], means[valid], color=color_list[setting_idx],
                    label=labels[setting_idx], linewidth=1.5)
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


def plot_visitation_pca(
    figname_prefix, labels, results_list,
    pca_coords, color_list,
    fontsize=7, legendfontsize=7,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """
    For each setting, produce a 2D scatter plot in PCA embedding space showing:
      1. Target distribution density as yellow dots with alpha proportional to target probability.
      2. Visited tokens as circles with size/darkness proportional to cumulative visitation count.
      3. Top-N target distribution tokens as stars with darkness proportional to target probability.

    One PDF per setting: {figname_prefix}_setting{idx}.pdf

    Args:
        pca_coords: (vocab_size, 2) numpy array of 2D PCA coordinates per token.
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
        print("Warning: No cumulative_q_sample_counts found. Skipping PCA visitation plot.")
        return

    assert pca_coords.shape[0] >= n_vocab, (
        f"PCA coords have {pca_coords.shape[0]} tokens but data has {n_vocab} vocab entries"
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

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
            ax.scatter(pca_coords[:n_vocab, 0], pca_coords[:n_vocab, 1],
                       s=3, c=bg_rgba, zorder=1, rasterized=True)
            # Proxy artist for legend with full-opacity gold
            ax.scatter([], [], s=15, c='#FFD700', alpha=1.0, label=r'$\sigma$ density')
        else:
            # Fallback: uniform faint yellow if no target data
            ax.scatter(pca_coords[:n_vocab, 0], pca_coords[:n_vocab, 1],
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

            ax.scatter(pca_coords[visited_ids, 0], pca_coords[visited_ids, 1],
                       s=10, c=rgba_array, marker='o', zorder=2,
                       edgecolors='none', linewidths=0,
                       label=f'q visits ({labels[setting_idx]})', rasterized=True)

        # 3. Top target tokens as stars
        top_ids_in_range = [t for t in top_n_tokens_list if t < n_vocab and t in target_alphas]
        if top_ids_in_range:
            top_coords = pca_coords[top_ids_in_range]
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

        ax.set_xlabel('PC1', fontsize=fontsize)
        ax.set_ylabel('PC2', fontsize=fontsize)
        ax.set_title(f'Token Visitation in PCA Embedding Space: {labels[setting_idx]}', fontsize=fontsize + 1)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.legend(fontsize=legendfontsize, loc='best')
        plt.tight_layout()

        figname = f"{figname_prefix}_setting{setting_idx}.pdf"
        plt.savefig(figname, dpi=150)
        plt.clf()
        plt.close(fig)
        print(f"PCA visitation plot saved to {figname}")


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
        fig_diff, axes_diff = plt.subplots(1, n_diff_panels, figsize=(3.5 * n_diff_panels, 3.5))
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
        fig_cum, ax_cum = plt.subplots(figsize=(4, 4))

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


def plot_top_q_intersection_lollipop(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """
    Lollipop chart showing per-setting top tokens ranked by average log q (across seeds).

    Each setting gets its own top-N tokens (by mean log q across seeds at the final timestep).
    Tokens are grouped by setting: all top-N for setting 1, then all top-N for setting 2, etc.
    Within each group, tokens are ordered left-to-right by decreasing average log q.
    For each token, a black dash shows the target log prob and colored dots (with bootstrap CIs)
    show each setting's log q.
    """
    # Collect all token data at the final timestep
    if precomputed_token_data is not None:
        target_by_setting, q_by_setting, _, _ = precomputed_token_data
    else:
        target_by_setting, q_by_setting, _, _ = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens=999999, final_only=True)
    if q_by_setting is None:
        print("Warning: No token data found. Cannot create top-q lollipop.")
        return

    n_settings = len(labels)

    # For each setting, compute mean log q per token (across seeds), sort, pick top N
    top_tokens_per_setting = {}
    for setting_idx in range(n_settings):
        token_avg_q = {}
        for token_id, setting_dict in q_by_setting.items():
            vals = setting_dict.get(setting_idx, [])
            if vals:
                token_avg_q[token_id] = np.mean(vals)
        sorted_tokens = sorted(token_avg_q.items(), key=lambda x: x[1], reverse=True)
        top_tokens_per_setting[setting_idx] = sorted_tokens[:n_top_tokens]
        top_ids = [tid for tid, _ in top_tokens_per_setting[setting_idx]]

    # Build grouped token list: all for setting 0, then all for setting 1, ...
    # Each entry is (setting_idx, token_id)
    grouped = []
    for setting_idx in range(n_settings):
        for token_id, _ in top_tokens_per_setting.get(setting_idx, []):
            grouped.append((setting_idx, token_id))

    if not grouped:
        print("Warning: No tokens found. Skipping top-q lollipop.")
        return

    n_positions = len(grouped)
    fig, ax = plt.subplots(figsize=(max(8, n_positions * 0.45), 5))

    x_positions = np.arange(n_positions)

    # For each position, compute target mean
    target_means = np.full(n_positions, np.nan)
    for pos_idx, (owner_setting, token_id) in enumerate(grouped):
        all_target_vals = []
        for vals in target_by_setting.get(token_id, {}).values():
            all_target_vals.extend(vals)
        if all_target_vals:
            target_means[pos_idx] = np.mean(all_target_vals)

    # Draw target dashes
    dash_half_width = 0.35
    target_label_added = False
    for pos_idx in range(n_positions):
        if np.isnan(target_means[pos_idx]):
            continue
        label = r"$\sigma$ (target)" if not target_label_added else None
        ax.plot(
            [x_positions[pos_idx] - dash_half_width, x_positions[pos_idx] + dash_half_width],
            [target_means[pos_idx], target_means[pos_idx]],
            color='black', linewidth=1.5, solid_capstyle='butt', label=label, zorder=3,
        )
        target_label_added = True

    # Draw q dots for each setting at every position
    setting_label_added = [False] * n_settings
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])

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
            ax.errorbar(
                x_pos[valid], q_means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt='o', color=color_list[setting_idx], markersize=4,
                capsize=2, linewidth=0.8, label=label, zorder=4,
            )
            setting_label_added[setting_idx] = True

    # Colored tick labels matching the owner setting
    tick_labels = []
    tick_colors = []
    for owner_setting, token_id in grouped:
        tick_labels.append(_token_label(token_id, tokenizer))
        tick_colors.append(color_list[owner_setting])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=max(fontsize - 2, 4), rotation=90)
    for tick_label, color in zip(ax.get_xticklabels(), tick_colors):
        tick_label.set_color(color)

    # Add setting separator lines between groups
    pos = 0
    for setting_idx in range(n_settings):
        n_tokens_in_group = len(top_tokens_per_setting.get(setting_idx, []))
        pos += n_tokens_in_group
        if setting_idx < n_settings - 1 and pos < n_positions:
            ax.axvline(x=pos - 0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_xlabel('Token (colored by owning setting)' if tokenizer is not None else 'Token ID (colored by owning setting)', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Per-Setting Top-{n_top_tokens} Tokens by Log q (final step)', fontsize=fontsize + 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Top-q lollipop chart saved to {figname}")


def plot_top_q_intersection_lollipop_individual(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_top_tokens=10,
    precomputed_token_data=None,
    tokenizer=None,
):
    """
    Individual-seed version of plot_top_q_intersection_lollipop.

    Per-setting top-N tokens by mean log q, but instead of bootstrap CIs, each seed
    is plotted as a separate dot. Same color/marker per setting; one legend entry per setting.
    """
    # Collect all token data at the final timestep
    if precomputed_token_data is not None:
        target_by_setting, q_by_setting, _, _ = precomputed_token_data
    else:
        target_by_setting, q_by_setting, _, _ = _collect_top_token_log_probs(
            labels, results_list, n_top_tokens=999999, final_only=True)
    if q_by_setting is None:
        print("Warning: No token data found. Cannot create individual top-q lollipop.")
        return

    n_settings = len(labels)

    # For each setting, compute mean log q per token (across seeds), sort, pick top N
    top_tokens_per_setting = {}
    for setting_idx in range(n_settings):
        token_avg_q = {}
        for token_id, setting_dict in q_by_setting.items():
            vals = setting_dict.get(setting_idx, [])
            if vals:
                token_avg_q[token_id] = np.mean(vals)
        sorted_tokens = sorted(token_avg_q.items(), key=lambda x: x[1], reverse=True)
        top_tokens_per_setting[setting_idx] = sorted_tokens[:n_top_tokens]

    # Build grouped token list: all for setting 0, then all for setting 1, ...
    grouped = []
    for setting_idx in range(n_settings):
        for token_id, _ in top_tokens_per_setting.get(setting_idx, []):
            grouped.append((setting_idx, token_id))

    if not grouped:
        print("Warning: No tokens found. Skipping individual top-q lollipop.")
        return

    n_positions = len(grouped)
    fig, ax = plt.subplots(figsize=(max(8, n_positions * 0.45), 5))

    x_positions = np.arange(n_positions)

    # Target means
    target_means = np.full(n_positions, np.nan)
    for pos_idx, (owner_setting, token_id) in enumerate(grouped):
        all_target_vals = []
        for vals in target_by_setting.get(token_id, {}).values():
            all_target_vals.extend(vals)
        if all_target_vals:
            target_means[pos_idx] = np.mean(all_target_vals)

    # Draw target dashes
    dash_half_width = 0.35
    target_label_added = False
    for pos_idx in range(n_positions):
        if np.isnan(target_means[pos_idx]):
            continue
        label = r"$\sigma$ (target)" if not target_label_added else None
        ax.plot(
            [x_positions[pos_idx] - dash_half_width, x_positions[pos_idx] + dash_half_width],
            [target_means[pos_idx], target_means[pos_idx]],
            color='black', linewidth=1.5, solid_capstyle='butt', label=label, zorder=3,
        )
        target_label_added = True

    # Draw individual seed dots for each setting at every position
    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])

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
                ax.scatter(
                    x_base, val,
                    color=color_list[setting_idx], s=15, alpha=0.7,
                    marker=marker, label=label, zorder=4,
                )
                label_added = True

    # Colored tick labels matching the owner setting
    tick_labels = []
    tick_colors = []
    for owner_setting, token_id in grouped:
        tick_labels.append(_token_label(token_id, tokenizer))
        tick_colors.append(color_list[owner_setting])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=max(fontsize - 2, 4), rotation=90)
    for tick_label, color in zip(ax.get_xticklabels(), tick_colors):
        tick_label.set_color(color)

    # Add setting separator lines between groups
    pos = 0
    for setting_idx in range(n_settings):
        n_tokens_in_group = len(top_tokens_per_setting.get(setting_idx, []))
        pos += n_tokens_in_group
        if setting_idx < n_settings - 1 and pos < n_positions:
            ax.axvline(x=pos - 0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_xlabel('Token (colored by owning setting)' if tokenizer is not None else 'Token ID (colored by owning setting)', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Per-Setting Top-{n_top_tokens} Tokens by Log q (individual seeds, final step)', fontsize=fontsize + 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Individual top-q lollipop chart saved to {figname}")


def plot_top_q_ranked_lollipop(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_ranks=10,
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
    """
    # Collect per-setting, per-seed, per-rank log probs
    # rank_data[setting_idx] = list of (q_by_rank, target_by_rank) per seed
    #   where q_by_rank[k] = log_q of the (k+1)-th highest token under q
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

            # Use only the last metrics dict (final evaluation timestep)
            last_metrics = metrics_list[-1]
            if not isinstance(last_metrics, dict):
                continue

            # Prefer full-vocab tensors (new format) over dict entries (old format)
            log_probs_q_full = last_metrics.get('log_probs_q_full', None)
            log_probs_target_full = last_metrics.get('log_probs_target_full', None)

            if log_probs_q_full is not None and log_probs_target_full is not None:
                import torch
                top_q_vals, top_q_ids = torch.topk(log_probs_q_full, k=n_ranks)
                q_by_rank = top_q_vals.tolist()
                ids_by_rank = top_q_ids.tolist()
                target_by_rank = [log_probs_target_full[tid].item() for tid in ids_by_rank]
            else:
                log_probs_q = last_metrics.get('log_probs_q', {})
                log_probs_target = last_metrics.get('log_probs_target', {})
                if not log_probs_q:
                    continue

                # Sort tracked tokens by log_probs_q descending
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

    # Check that we have data
    has_data = any(len(seeds) > 0 for seeds in rank_data.values())
    if not has_data:
        print("Warning: No rank data found. Cannot create ranked lollipop chart.")
        return

    # Determine max ranks available across all settings/seeds
    max_ranks = 0
    for seeds in rank_data.values():
        for q_by_rank, _, _ in seeds:
            max_ranks = max(max_ranks, len(q_by_rank))
    n_ranks_actual = min(n_ranks, max_ranks)
    if n_ranks_actual == 0:
        print("Warning: No ranks available. Skipping ranked lollipop chart.")
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

    fig, ax = plt.subplots()

    x_positions = np.arange(n_ranks_actual)
    dot_spacing = 0.12
    n_settings = len(labels)
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])

    from matplotlib.lines import Line2D

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
            # Collect values at this rank across seeds (some seeds may have fewer ranks)
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

        # q markers with CI (filled circle)
        valid_q = ~np.isnan(q_means)
        if valid_q.any():
            q_lower = q_means[valid_q] - q_ci_lo[valid_q]
            q_upper = q_ci_hi[valid_q] - q_means[valid_q]
            ax.errorbar(
                x_pos[valid_q], q_means[valid_q],
                yerr=[q_lower, q_upper],
                fmt='o', color=color_list[setting_idx], markersize=5,
                capsize=3, linewidth=1, zorder=4,
            )

    # Build consolidated legend: one entry per setting (colored line), plus marker-type key
    legend_handles = []
    # Marker-type entries (general, in black)
    legend_handles.append(Line2D([], [], marker='o', color='black', markersize=5,
                                 linestyle='None', label='q'))
    legend_handles.append(Line2D([], [], marker='D', color='black', markersize=4,
                                 markerfacecolor='none', markeredgewidth=1.2,
                                 linestyle='None', label=r'$\sigma$ (target)'))
    # Per-setting entries (colored line segment)
    for setting_idx in range(n_settings):
        if not rank_data[setting_idx]:
            continue
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


def plot_top_q_ranked_lollipop_individual(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_ranks=10,
):
    """
    Individual-seed version of plot_top_q_ranked_lollipop.

    For each setting and seed, at the final timestep, sorts tokens by log q descending
    and plots the log prob under q and target at each rank. Each seed is a separate dot
    instead of bootstrap-aggregated means with CIs.
    """
    # Collect per-setting, per-seed, per-rank log probs (same as aggregated version)
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
                import torch
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
        print("Warning: No rank data found. Cannot create individual ranked lollipop chart.")
        return

    max_ranks = 0
    for seeds in rank_data.values():
        for q_by_rank, _, _ in seeds:
            max_ranks = max(max_ranks, len(q_by_rank))
    n_ranks_actual = min(n_ranks, max_ranks)
    if n_ranks_actual == 0:
        print("Warning: No ranks available. Skipping individual ranked lollipop chart.")
        return

    fig, ax = plt.subplots()

    x_positions = np.arange(n_ranks_actual)
    dot_spacing = 0.12
    n_settings = len(labels)
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])

    from matplotlib.lines import Line2D

    for setting_idx in range(n_settings):
        seeds = rank_data[setting_idx]
        if not seeds:
            continue

        x_pos = x_positions + setting_offsets[setting_idx]

        for seed_j, (q_by_rank, tgt_by_rank, ids_by_rank) in enumerate(seeds):
            n_this = min(n_ranks_actual, len(q_by_rank))
            marker = SEED_MARKERS[seed_j % len(SEED_MARKERS)]

            # q dots
            ax.scatter(
                x_pos[:n_this], q_by_rank[:n_this],
                color=color_list[setting_idx], s=20, alpha=0.7,
                marker=marker, zorder=4,
            )

            # target dots (hollow version of same marker)
            valid_tgt = ~np.isnan(tgt_by_rank[:n_this])
            if valid_tgt.any():
                ax.scatter(
                    x_pos[:n_this][valid_tgt], tgt_by_rank[:n_this][valid_tgt],
                    color=color_list[setting_idx], s=18, alpha=0.7,
                    marker=marker, facecolors='none', linewidths=1, zorder=3,
                )

    # Build consolidated legend
    legend_handles = []
    legend_handles.append(Line2D([], [], marker='o', color='black', markersize=5,
                                 linestyle='None', label='q (filled)'))
    legend_handles.append(Line2D([], [], marker='o', color='black', markersize=4,
                                 markerfacecolor='none', markeredgewidth=1.2,
                                 linestyle='None', label=r'$\sigma$ (target, hollow)'))
    for setting_idx in range(n_settings):
        if not rank_data[setting_idx]:
            continue
        legend_handles.append(Line2D([], [], color=color_list[setting_idx], linewidth=2,
                                     label=labels[setting_idx]))

    ax.set_xlabel('Rank under q', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Top {n_ranks_actual} Ranked Tokens under q: Log Prob (individual seeds)', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(r + 1) for r in range(n_ranks_actual)], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(handles=legend_handles, fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Individual ranked lollipop chart saved to {figname}")


def plot_max_sis_weight_over_time(
    figname, labels, sis_weights_results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
):
    """
    Plot the max self-normalized importance weight over training episodes.

    Each entry in sis_weights_results_list[setting_idx] is a loaded SIS weights history
    (list of tensors of shape (num_prompts, samples_per_prompt), one per episode) for one seed.
    For each episode, max_weight = tensor.max().item().
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
            curve = [w.max().item() for w in weights_history]
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
    ax.set_ylabel('Max Normalized SIS Weight', fontsize=fontsize)
    ax.set_title('Max Self-Normalized Importance Weight Over Time', fontsize=fontsize + 1)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Max SIS weight plot saved to {figname}")
