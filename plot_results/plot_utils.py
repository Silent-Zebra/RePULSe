import torch
import re
import numpy as np


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
        # Detect CTL vs CTLN (ctl_nosecondterm) from prefix
        if "_ctln_" in prefix:
            loss_type_str = "CTLN"
        elif "_ctl_" in prefix:
            loss_type_str = "CTL"
        else:
            loss_type_str = None

        # Determine training run type (support both old and new abbreviations)
        # Check patterns in order of specificity
        # 1. Coin flip: old _cfn or new _cf followed by number
        if "cfn" in prefix or re.search(r'_cf([\d.]+)', prefix):
            run_type = "Coin Flip Net"
        # 2. Exact count: old _count or new _c followed by number (not _cf, _cl, _ct)
        elif "_count" in prefix or re.search(r'_c([\d.]+)(?![a-z])', prefix):
            # The negative lookahead (?![a-z]) ensures _c is not followed by a letter (like _cf, _cl)
            run_type = "Exact Count"
        # 3. Mixture proposal: _mix followed by variant abbreviation
        elif "_mix" in prefix:
            run_type = "Mixture"
        else:
            run_type = "No Exploration Bonus"
        
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
            label_parts = [f"Mixture ({mix_variant})"]

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
            if len(iwae_lbs_list) == 0 or len(iwae_ubs_list) == 0:
                continue
            lb_per_t = [to_scalar(iwae_lbs_list[t]) for t in range(len(iwae_lbs_list))]
            ub_per_t = [to_scalar(iwae_ubs_list[t]) for t in range(len(iwae_ubs_list))]
            logZ_seed = (max(lb_per_t) + min(ub_per_t)) / 2.0
            all_logZ_estimates.append(logZ_seed)
        
        # Per-experiment log Z (average lb/ub per timestep across seeds, then midpoint)
        if len(exp_data) == 0:
            continue
        T = len(exp_data[0][2])  # Length of iwae_lbs_list
        lb_per_t = [
            np.mean([to_scalar(exp_data[s][2][t]) for s in range(len(exp_data)) if t < len(exp_data[s][2])])
            for t in range(T)
        ]
        ub_per_t = [
            np.mean([to_scalar(exp_data[s][3][t]) for s in range(len(exp_data)) if t < len(exp_data[s][3])])
            for t in range(T)
        ]
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
