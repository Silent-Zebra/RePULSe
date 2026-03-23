import torch
import re
import numpy as np
import matplotlib.cm as cm
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
    Extract per-token log probabilities (target and q) from results_list.

    Args:
        final_only: If True, only use the last metrics dict per seed (final evaluation timestep)
                    instead of averaging across all timesteps.

    Returns:
        (token_log_probs_target_by_setting, token_log_probs_q_by_setting, top_n_tokens)
        where:
            token_log_probs_{target,q}_by_setting: dict of token_id -> {setting_idx: [per-seed averages]}
            top_n_tokens: list of token IDs ranked by average target log prob (descending)
        Returns (None, None, None) if no token data is found.
    """
    token_log_probs_target_by_setting = {}  # token_id -> {setting_idx: [per-seed averages]}
    token_log_probs_q_by_setting = {}  # token_id -> {setting_idx: [per-seed averages]}

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
            seed_token_log_probs = {}  # token_id -> {'target': [values], 'q': [values]}

            metrics_to_use = [metrics_list[-1]] if final_only else metrics_list
            for metrics_dict in metrics_to_use:
                if not isinstance(metrics_dict, dict):
                    continue

                # Prefer full-vocab tensors (new format) over dict entries (old format)
                log_probs_target_full = metrics_dict.get('log_probs_target_full', None)
                log_probs_q_full = metrics_dict.get('log_probs_q_full', None)

                if log_probs_target_full is not None and log_probs_q_full is not None:
                    # New format: full-vocab tensors — iterate over all tokens
                    import torch
                    n_vocab = len(log_probs_target_full)
                    for token_id in range(n_vocab):
                        if token_id not in seed_token_log_probs:
                            seed_token_log_probs[token_id] = {'target': [], 'q': []}
                        seed_token_log_probs[token_id]['target'].append(log_probs_target_full[token_id].item())
                        seed_token_log_probs[token_id]['q'].append(log_probs_q_full[token_id].item())
                else:
                    # Old format: dict entries for tracked tokens only
                    tracked_tokens = metrics_dict.get('all_tracked_tokens', metrics_dict.get('top_10_target_tokens', []))
                    log_probs_target = metrics_dict.get('log_probs_target', {})
                    log_probs_q = metrics_dict.get('log_probs_q', {})

                    for token_id in tracked_tokens:
                        if token_id not in seed_token_log_probs:
                            seed_token_log_probs[token_id] = {'target': [], 'q': []}

                        if token_id in log_probs_target:
                            seed_token_log_probs[token_id]['target'].append(log_probs_target[token_id])

                        if token_id in log_probs_q:
                            seed_token_log_probs[token_id]['q'].append(log_probs_q[token_id])

            # Average across prompts for this seed, then store
            for token_id, probs_dict in seed_token_log_probs.items():
                target_values = probs_dict['target']
                q_values = probs_dict['q']

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
        return None, None, None

    sorted_tokens = sorted(token_avg_log_probs_target.items(), key=lambda x: x[1], reverse=True)
    top_n_tokens = [token_id for token_id, _ in sorted_tokens[:n_top_tokens]]

    print(f"\nTop {n_top_tokens} tokens (by average log prob under target): {top_n_tokens}")

    return token_log_probs_target_by_setting, token_log_probs_q_by_setting, top_n_tokens


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

    target_by_setting, q_by_setting, top_n_tokens = _collect_top_token_log_probs(
        labels, results_list, n_top_tokens, final_only=final_only)
    if top_n_tokens is None:
        print("Warning: Cannot create bar chart.")
        return

    n_settings = len(labels)
    n_tokens = len(top_n_tokens)

    bar_width = 0.25
    x_positions = np.arange(n_tokens)
    setting_offsets = np.linspace(-bar_width * (n_settings - 1) / 2,
                                   bar_width * (n_settings - 1) / 2,
                                   n_settings)

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

    # Plot bars
    for setting_idx in range(n_settings):
        x_pos = x_positions + setting_offsets[setting_idx]
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

    plt.xlabel('Token ID', fontsize=fontsize)
    plt.ylabel('Log Probability Difference (q - target)', fontsize=fontsize)
    time_label = " (final step)" if final_only else " (avg over time)"
    plt.title(f'Top {n_top_tokens} Target Tokens: Log Prob Difference (q - target){time_label}', fontsize=fontsize+1)
    plt.xticks(x_positions, [str(token_id) for token_id in top_n_tokens], fontsize=fontsize-1)
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
    target_by_setting, q_by_setting, top_n_tokens = _collect_top_token_log_probs(
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

    ax.set_xlabel('Token ID', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    time_label = " (final step)" if final_only else " (avg over time)"
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Log Prob (target vs q){time_label}', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(token_id) for token_id in top_n_tokens], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Lollipop chart saved to {figname}")


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
    target_by_setting, q_by_setting, top_n_tokens = _collect_top_token_log_probs(
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

    ax.set_xlabel('Token ID', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Top {n_top_tokens} Target Tokens: Log q Over Time', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(token_id) for token_id in top_n_tokens], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Over-time lollipop chart saved to {figname}")


def plot_top_q_intersection_lollipop(
    figname, labels, results_list,
    color_list, fontsize=7, legendfontsize=7,
    n_bootstrap_draws=5000,
    n_top_tokens=10,
):
    """
    Lollipop chart for top tokens ranked by average log prob under q across all settings.

    Uses final-timestep data. Collects all tokens, computes average log q across all
    settings and seeds, sorts descending, and picks the top n_top_tokens.
    Shows target log prob as a shared reference dash + per-setting q dots with CIs.
    """
    # Collect all token data at the final timestep
    target_by_setting, q_by_setting, _ = _collect_top_token_log_probs(
        labels, results_list, n_top_tokens=999999, final_only=True)
    if q_by_setting is None:
        print("Warning: No token data found. Cannot create top-q lollipop.")
        return

    # Compute average log q across all settings/seeds for each token
    token_avg_q = {}
    for token_id, setting_dict in q_by_setting.items():
        all_q_vals = []
        for vals in setting_dict.values():
            all_q_vals.extend(vals)
        if all_q_vals:
            token_avg_q[token_id] = np.mean(all_q_vals)

    if not token_avg_q:
        print("Warning: No log prob data. Skipping top-q lollipop.")
        return

    # Sort by average q log prob (descending) and pick top N
    sorted_tokens = sorted(token_avg_q.items(), key=lambda x: x[1], reverse=True)
    token_list = [token_id for token_id, _ in sorted_tokens[:n_top_tokens]]

    n_settings = len(labels)
    n_tokens = len(token_list)
    print(f"\nTop-q tokens ({n_tokens}), sorted by avg log q: {token_list}")

    fig, ax = plt.subplots()

    dot_spacing = 0.12
    total_width = dot_spacing * (n_settings - 1)
    setting_offsets = np.linspace(-total_width / 2, total_width / 2, n_settings) if n_settings > 1 else np.array([0.0])
    x_positions = np.arange(n_tokens)

    # Target mean per token (pooled across all settings/seeds)
    target_means = np.full(n_tokens, np.nan)
    for token_idx, token_id in enumerate(token_list):
        all_target_vals = []
        for vals in target_by_setting.get(token_id, {}).values():
            all_target_vals.extend(vals)
        if all_target_vals:
            target_means[token_idx] = np.mean(all_target_vals)

    # Draw target dashes
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

    # Draw q dots for each setting
    for setting_idx in range(n_settings):
        q_means = np.full(n_tokens, np.nan)
        q_ci_lo = np.full(n_tokens, np.nan)
        q_ci_hi = np.full(n_tokens, np.nan)

        for token_idx, token_id in enumerate(token_list):
            q_values = np.array(q_by_setting.get(token_id, {}).get(setting_idx, []))
            q_means[token_idx], q_ci_lo[token_idx], q_ci_hi[token_idx] = \
                _bootstrap_mean_ci(q_values, n_bootstrap_draws)

        x_pos = x_positions + setting_offsets[setting_idx]

        # q dots with CI error bars
        lower_err = q_means - q_ci_lo
        upper_err = q_ci_hi - q_means
        valid = ~np.isnan(q_means)
        if valid.any():
            ax.errorbar(
                x_pos[valid], q_means[valid],
                yerr=[lower_err[valid], upper_err[valid]],
                fmt='o', color=color_list[setting_idx], markersize=5,
                capsize=3, linewidth=1, label=labels[setting_idx], zorder=4,
            )

    ax.set_xlabel('Token ID', fontsize=fontsize)
    ax.set_ylabel('Log Probability', fontsize=fontsize)
    ax.set_title(f'Top {n_tokens} Tokens by Avg Log q (final step)', fontsize=fontsize + 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(token_id) for token_id in token_list], fontsize=fontsize - 1)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=legendfontsize)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(figname)
    plt.clf()
    plt.close(fig)
    print(f"Top-q lollipop chart saved to {figname}")


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
