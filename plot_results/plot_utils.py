import torch
import re
import numpy as np
import matplotlib.cm as cm


# Marker constants for bonus types (legacy, kept for backward compatibility)
MARKER_NO_BONUS = "x"
MARKER_CFN = "P"
MARKER_MIXTURE = "^"
MARKER_EXACT_COUNT = "D"

# Marker constants for loss types (used by semantic styling)
MARKER_CTL = "o"
MARKER_CTLN = "x"
MARKER_LOSS_UNKNOWN = "D"


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
    cfn_alpha_cmaps = [cm.Blues, cm.Greens, cm.Purples]
    if unique_alphas:
        cfn_alpha_cmap = {a: cfn_alpha_cmaps[i % len(cfn_alpha_cmaps)] for i, a in enumerate(unique_alphas)}
    else:
        cfn_alpha_cmap = {}

    # 4b. Mixture lag_steps -> color family: cycle through orange-ish colormaps for distinct lag values.
    #     None/"best" gets one colormap, each unique lag value gets another.
    unique_mixture_keys = sorted(
        set((p["mixture_variant"], p["mixture_lag_steps"])
            for p in props_list if p["bonus_type"] == "mixture"),
        key=lambda x: (x[0] or "", x[1] if x[1] is not None else -1),
    )
    # Colormaps for q_independent variants; first is for best/no-lag, rest cycle for lag values
    qi_cmaps = [cm.Oranges, cm.YlOrRd, cm.OrRd, cm.Reds]
    # Colormaps for other mixture variants
    other_mix_cmaps = [cm.RdPu, cm.PuRd, cm.pink, cm.hot]
    mixture_cmap_map = {}
    qi_idx, other_idx = 0, 0
    for variant, lag in unique_mixture_keys:
        if variant == "q_independent":
            mixture_cmap_map[(variant, lag)] = qi_cmaps[qi_idx % len(qi_cmaps)]
            qi_idx += 1
        else:
            mixture_cmap_map[(variant, lag)] = other_mix_cmaps[other_idx % len(other_mix_cmaps)]
            other_idx += 1

    # 5. Colormap and shade range per bonus category
    #    Shade range [lo, hi] samples the colormap avoiding very light/very dark ends.
    shade_range_default = (0.45, 0.85)

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

        # Shade based on LR rank
        lo, hi = shade_range_default
        if props["learning_rate"] is not None and props["learning_rate"] in lr_position:
            t = lr_position[props["learning_rate"]]
        else:
            t = 0.5
        shade = lo + t * (hi - lo)
        color_list.append(cmap(shade))

        # Marker from loss type
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

        # Prepend CTL/CTLN loss type if detected
        if loss_type_str is not None:
            label_parts.insert(0, loss_type_str)

        # Check for mixeval prefix
        if prefix.startswith("f_q_g_q_iwae_bounds_mixeval"):
            label_parts.append("(mixeval)")

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
