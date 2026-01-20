import torch
import re


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
    For Coin Flip Net runs, it extracts additional parameters like updates, head_std, prior_std, etc.
    
    Args:
        load_prefixes_to_use: List of lists of prefixes (each inner list contains prefixes for one series)
    
    Returns:
        List of label strings, one for each prefix list
    """
    labels = []
    for a in load_prefixes_to_use:
        prefix = a[0]
        # Determine training run type
        if "count" in prefix:
            run_type = "Exact Count"
        elif "cfn" in prefix:
            run_type = "Coin Flip Net"
        else:
            run_type = "No Exploration Bonus"
        
        # If it's a coin flip net, extract additional parameters
        if run_type == "Coin Flip Net":
            label_parts = [run_type]
            
            # # Extract cfd (dimension)
            # cfd_match = re.search(r'_cfd(\d+)', prefix)
            # if cfd_match:
            #     cfd_num = cfd_match.group(1)
            #     label_parts.append(f"{cfd_num}-d")
            
            # Extract cfus (updates)
            cfus_match = re.search(r'_cfus(\d+)', prefix)
            if cfus_match:
                cfus_num = cfus_match.group(1)
                label_parts.append(f"{cfus_num} updates")
            else:
                label_parts.append("1 update")
            
            # Extract cfhis (coin flip head init std)
            cfhis_match = re.search(r'_cfhis([\d.e-]+)', prefix)
            if cfhis_match:
                cfhis_num = cfhis_match.group(1)
                label_parts.append(f"head_std={cfhis_num}")
            
            # Extract fpis (frozen prior init std)
            fpis_match = re.search(r'_fpis([\d.e-]+)', prefix)
            if fpis_match:
                fpis_num = fpis_match.group(1)
                label_parts.append(f"prior_std={fpis_num}")
            
            # Check for coin_flip_linear_bias
            if "_cfbias" in prefix:
                label_parts.append("with bias")
            
            # Extract cflr (learning rate)
            cflr_match = re.search(r'_cflr([\d.e-]+)', prefix)
            if cflr_match:
                cflr_num = cflr_match.group(1)
                label_parts.append(f"{cflr_num} Coin Flip LR")
            
            # # Check for "after" or "before" in the prefix
            # if "after" in prefix:
            #     label_parts.append("Update After")
            # elif "before" in prefix:
            #     label_parts.append("Update Before")

            if "firstonline" in prefix:
                label_parts.append("First Update Online")
            if "pri" in prefix:
                label_parts.append("Prioritized")
            if "sepnn" in prefix:
                label_parts.append("Sep. NN")

            labels.append(", ".join(label_parts))
        elif run_type == "Exact Count":
            label_parts = [run_type]
            
            # Extract bonus_alpha (encoded as _count followed by value)
            count_match = re.search(r'_count([\d.]+)', prefix)
            if count_match:
                bonus_alpha = count_match.group(1)
                label_parts.append(f"bonus_alpha={bonus_alpha}")
            
            # Extract num_episodes (encoded as _epi followed by value)
            epi_match = re.search(r'_epi(\d+)', prefix)
            if epi_match:
                num_episodes = epi_match.group(1)
                label_parts.append(f"num_episodes={num_episodes}")
            
            # Extract batch_size (encoded as _tbs followed by value)
            tbs_match = re.search(r'_tbs(\d+)', prefix)
            if tbs_match:
                batch_size = tbs_match.group(1)
                label_parts.append(f"batch_size={batch_size}")
            
            labels.append(", ".join(label_parts))
        else:
            label_parts = [run_type]

            # Extract num_episodes (encoded as _epi followed by value)
            epi_match = re.search(r'_epi(\d+)', prefix)
            if epi_match:
                num_episodes = epi_match.group(1)
                label_parts.append(f"num_episodes={num_episodes}")

            # Extract batch_size (encoded as _tbs followed by value)
            tbs_match = re.search(r'_tbs(\d+)', prefix)
            if tbs_match:
                batch_size = tbs_match.group(1)
                label_parts.append(f"batch_size={batch_size}")

            labels.append(", ".join(label_parts))
    
    return labels
