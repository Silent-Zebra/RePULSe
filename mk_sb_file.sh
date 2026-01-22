#!/bin/bash

# Parse --cluster argument
CLUSTER="default"
COMMAND_ARGS=()

# Convert positional parameters to array
ARGS=("$@")

i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [ "${ARGS[$i]}" == "--cluster" ]; then
        if [ $((i+1)) -lt ${#ARGS[@]} ]; then
            CLUSTER="${ARGS[$((i+1))]}"
            i=$((i+2))
        else
            echo "Error: --cluster requires a value"
            exit 1
        fi
    else
        COMMAND_ARGS+=("${ARGS[$i]}")
        i=$((i+1))
    fi
done

# Check if we have any command arguments
if [ ${#COMMAND_ARGS[@]} -eq 0 ]; then
    echo "Error: Please provide the training command"
    exit 1
fi

# Store the full command (without --cluster)
COMMAND="${COMMAND_ARGS[*]}"

# Determine pretrain abbreviation method based on cluster
if [ "$CLUSTER" == "simple" ]; then
    PRETRAIN_LOGIC="simple"
else
    PRETRAIN_LOGIC="split"
fi

# Extract parameters using awk
PARAMS=$(echo "$COMMAND" | awk -v pretrain_logic="$PRETRAIN_LOGIC" '
{
    # Initialize empty variables (union of all files)
    micro_train = train = micro_rollout = rollout = ""
    max_epochs = num_episodes = num_episodes_h = gen_max_len = actor_lr = critic_lr = baseactor_lr = ""
    target_beta = lr_sched = actor_loss = kl = do_harmlessness = rta = rtb = startb = starta = sepb = uniw = ""
    custom_prompt = prompt_data = parameterization = adam_beta2 = rm_type = dup_rollout = pretrain = reward_pretrain = init_head_from_base = ""
    sd_divider = harmloss = harmlossreinbaseline = hlrbval = ""
    save_negdata_threshold = threshold = alpha = only_eval_neg = use_base_as_proposal = ""
    exploration_bonus_sa = exploration_bonus_ba = bonus_alpha = coin_flip_dim = coin_flip_lr = coin_flip_norm_momentum = coin_flip_update_steps = coin_flip_head_init_std = frozen_prior_init_std = coin_flip_linear_bias = coin_flip_architecture = analytic_batch = train_coin_flip_before = coin_flip_first_online = coin_flip_use_prioritization = ""
    
    # Scan through all matches in the string
    for(i=1; i<=NF; i++) {
        if($i == "--micro_train_batch_size") micro_train = $(i+1)
        if($i == "--train_batch_size") train = $(i+1)
        if($i == "--analytic_batch_size") analytic_batch = "_anb"$(i+1)
        if($i == "--micro_rollout_batch_size") micro_rollout = $(i+1)
        if($i == "--rollout_batch_size") rollout = $(i+1)
        if($i == "--max_epochs") max_epochs = $(i+1)
        if($i == "--num_episodes") num_episodes = $(i+1)
        if($i == "--harmlessness_training_num_episodes") num_episodes_h = $(i+1)
        if($i == "--generate_max_len") gen_max_len = $(i+1)
        if($i == "--actor_learning_rate") actor_lr = "_alr"$(i+1)
        if($i == "--critic_learning_rate") critic_lr = "_clr"$(i+1)
        if($i == "--base_actor_learning_rate") baseactor_lr = "_blr"$(i+1)
        if($i ~ /^--target_dist_beta(=|$)/) {
            # Abbreviate target beta
            beta_val = ($i ~ /=/) ? gensub(/^[^=]+=/, "", "g", $i) : $(i+1)
            target_beta = "_b" beta_val
        }
        if($i ~ /^--save_negdata_threshold(=|$)/) save_negdata_threshold = ($i ~ /=/) ? "_savethr" gensub(/^[^=]+=/, "", "g", $i) : "_savethr" $(i+1)
        if($i ~ /^--threshold(=|$)/) {
            # Abbreviate threshold
            thresh_val = ($i ~ /=/) ? gensub(/^[^=]+=/, "", "g", $i) : $(i+1)
            threshold = "_t" thresh_val
        }
        if($i == "--lr_scheduler") lr_sched = $(i+1)
        if($i == "--actor_loss_type") {
            # Abbreviate actor loss type
            loss = $(i+1)
            if(loss == "ctl") actor_loss = "_ctl"
            else actor_loss = "_" substr(loss, 1, 3)
        }
        if($i == "--custom_single_prompt") custom_prompt = "_custom"
        if($i == "--parameterization") {
            # Abbreviate parameterization: take first char of each component
            n = split($(i+1), arr, "_")
            abbrev = ""
            for (j = 1; j <= n; j++) {
                abbrev = abbrev substr(arr[j], 1, 1)
            }
            parameterization = abbrev
        }
        if($i == "--adam_betas") adam_beta2 = "_adambeta2_"$(i+2)
        if($i == "--rm_type") rm_type = $(i+1)
        if($i == "--duplicate_rollout_batch_by") dup_rollout = "_"$(i+1)
        if($i == "--pretrain") {
            if(pretrain_logic == "simple") {
                # Simple: just first 2 chars of path
                abbrev = substr($(i+1), 1, 2)
                pretrain = abbrev
            } else {
                # Split by "-", take first 2 chars of each component
                abbrev = ""
                n = split(gensub(".*/", "", "g", $(i+1)), arr, "-")
                for (j = 1; j <= n; j++) {
                    abbrev = abbrev substr(arr[j], 1, 2)
                }
                pretrain = abbrev
            }
        }
        if($i == "--reward_pretrain") {
            if(pretrain_logic == "simple") {
                # Simple: just first 2 chars of path
                abbrev = substr($(i+1), 1, 2)
                reward_pretrain = abbrev
            } else {
                # Split by "-", take first 2 chars of each component
                abbrev = ""
                n = split(gensub(".*/", "", "g", $(i+1)), arr, "-")
                for (j = 1; j <= n; j++) {
                    abbrev = abbrev substr(arr[j], 1, 2)
                }
                reward_pretrain = abbrev
            }
        }
        if($i ~ /^--custom_prompt(=|$)/) {
            # Extract first character of first word
            if($i ~ /=/) {
                custom_prompt_val = gensub(/^[^=]+=/, "", "g", $i)
            } else {
                custom_prompt_val = $(i+1)
            }
            # Remove quotes from beginning and end
            sub(/^["']/, "", custom_prompt_val)
            sub(/["']$/, "", custom_prompt_val)
            # Get first character of first word
            n = split(custom_prompt_val, words, " ")
            if(n > 0 && words[1] != "") {
                prompt_data = substr(words[1], 1, 1)
            }
        }
        if($i == "--prompt_data" && prompt_data == "") {
            # Simple: just first 2 chars of path
            abbrev = substr($(i+1), 1, 2)
            prompt_data = abbrev
        }
        if($i == "--init_head_from_base") init_head_from_base = "_initheadbase"
        if($i == "--additional_sd_divider") sd_divider = "_sddivider"$(i+1)
        if($i == "--harmlessness_training_loss_type") {
            # Abbreviate harmlessness loss type
            loss_type = $(i+1)
            if(loss_type ~ /neg_training/) harmloss = "_harmlneg"
            else if(loss_type ~ /neg/) harmloss = "_harmlneg"
            else harmloss = "_harml" substr(loss_type, 1, 3)
        }
        if($i == "--reinforce_baseline_type") {
            # Abbreviate baseline type
            baseline = $(i+1)
            if(baseline ~ /expectation/) harmlossreinbaseline = "_exp"
            else harmlossreinbaseline = "_" substr(baseline, 1, 3)
        }
        if($i == "--reinforce_hardcoded_baseline") hlrbval = "_"$(i+1)
        if($i == "--alpha") alpha = "_a"$(i+1)
        if($i == "--init_kl_coef") kl = "_kl"$(i+1)
        if($i == "--only_evaluate_on_neg_data") only_eval_neg = "_onlyevalneg"
        if($i == "--do_harmlessness_training") do_harmlessness = 1
        if($i == "--use_base_as_proposal") use_base_as_proposal = "_baseprop"
        if($i == "--rew_trans_alpha") rta = "_rta"$(i+1)
        if($i == "--rew_trans_beta") rtb = "_rtb"$(i+1)
        if($i == "--start_target_dist_beta") startb = "_start"$(i+1)
        if($i == "--start_alpha") starta = "_start"$(i+1)
        if($i == "--separate_reweighting_beta") sepb = "_sepb"$(i+1)
        if($i == "--uniform_reweight") uniw = "_uniw"
        if($i == "--exploration_bonus_sampling_actor") {
            # Abbreviate exploration bonus type
            bonus_type = $(i+1)
            if(bonus_type ~ /coin_flip/) exploration_bonus_sa = "_expsacf"
            else exploration_bonus_sa = "_expsa" substr(bonus_type, 1, 2)
        }
        if($i == "--exploration_bonus_base_actor") exploration_bonus_ba = "_expba"$(i+1)
        if($i == "--bonus_alpha") bonus_alpha = "_a"$(i+1)
        if($i == "--coin_flip_dim") coin_flip_dim = "_cfd"$(i+1)
        if($i == "--coin_flip_lr") coin_flip_lr = "_cflr"$(i+1)
        if($i == "--coin_flip_normalization_momentum") coin_flip_norm_momentum = "_cfnm"$(i+1)
        if($i == "--coin_flip_update_steps") coin_flip_update_steps = "_cfus"$(i+1)
        if($i == "--coin_flip_head_init_std") coin_flip_head_init_std = "_cfhis"$(i+1)
        if($i == "--frozen_prior_init_std") frozen_prior_init_std = "_fpis"$(i+1)
        if($i == "--coin_flip_linear_bias") coin_flip_linear_bias = "_cfbias"
        if($i == "--coin_flip_architecture") {
            # Abbreviate coin flip architecture
            arch = $(i+1)
            if(arch ~ /separate_nn/) coin_flip_architecture = "_cfarchsnn"
            else if(arch ~ /separate/) coin_flip_architecture = "_cfarchsep"
            else if(arch ~ /linear_head_on_static_initial_base/) coin_flip_architecture = "_cfarchsib"
            else if(arch ~ /linear_head_on_learning_base/) coin_flip_architecture = "_cfarchlp"
            else if(arch ~ /linear_head_on_learning_proposal/) coin_flip_architecture = "_cfarchlq"
            else coin_flip_architecture = "_cfarch" substr(arch, 1, 3)
        }
        if($i == "--train_coin_flip_before") train_coin_flip_before = "_before"
        if($i == "--coin_flip_first_online") coin_flip_first_online = "_firstonline"
        if($i == "--coin_flip_use_prioritization") coin_flip_use_prioritization = "_pri"
    }
    # Build episode string: always include num_episodes, add harmlessness episodes if enabled
    epi_str = "_epi" num_episodes
    if(do_harmlessness && num_episodes_h != "") {
        epi_str = epi_str "_hepi" num_episodes_h
    }
    
    # Shorten lr_sched if it contains "constant"
    if(lr_sched ~ /constant/) lr_sched = gensub(/constant/, "const", "g", lr_sched)
    
    print micro_train "|" train "|" micro_rollout "|" rollout "|" max_epochs "|" epi_str "|" \
          gen_max_len "|" actor_lr "|" critic_lr "|" baseactor_lr "|" target_beta "|" save_negdata_threshold "|" threshold "|" lr_sched "|" \
          actor_loss "|" custom_prompt "|" parameterization "|" adam_beta2 "|" rm_type "|" dup_rollout "|" pretrain "|" \
          reward_pretrain "|" prompt_data "|" init_head_from_base "|" sd_divider "|" harmloss "|" harmlossreinbaseline "|" hlrbval "|" alpha "|" kl "|" only_eval_neg "|" use_base_as_proposal "|" rta "|" rtb "|" startb "|" starta "|" sepb "|" uniw "|" exploration_bonus_sa "|" exploration_bonus_ba "|" bonus_alpha "|" coin_flip_dim "|" coin_flip_lr "|" coin_flip_norm_momentum "|" coin_flip_update_steps "|" coin_flip_head_init_std "|" frozen_prior_init_std "|" coin_flip_linear_bias "|" coin_flip_architecture "|" analytic_batch "|" train_coin_flip_before "|" coin_flip_first_online "|" coin_flip_use_prioritization \
}')

# Read using the special delimiter
IFS='|' read MICRO_TRAIN TRAIN MICRO_ROLLOUT ROLLOUT MAX_EPOCHS EPI_STR GEN_MAX_LEN \
    ACTOR_LR CRITIC_LR BASEACTOR_LR TARGET_BETA SAVE_NEGDATA_THRESH THRESH LR_SCHED ACTOR_LOSS CUSTOM_PROMPT PARAMETERIZATION ADAM_BETA2 RM_TYPE DUP_ROLLOUT PRETRAIN REWARD_PRETRAIN PROMPT_DATA \
    INITHEADBASE SD_DIVIDER HARMLOSS HARMLOSSREINBASELINE HLRBVAL ALPHA KL ONLY_EVAL_NEG BASE_PROP RTA RTB STARTB STARTA SEPB UNIW EXPLORATION_BONUS_SA EXPLORATION_BONUS_BA BONUS_ALPHA COIN_FLIP_DIM COIN_FLIP_LR COIN_FLIP_NORM_MOMENTUM COIN_FLIP_UPDATE_STEPS COIN_FLIP_HEAD_INIT_STD FROZEN_PRIOR_INIT_STD COIN_FLIP_LINEAR_BIAS COIN_FLIP_ARCHITECTURE ANALYTIC_BATCH TRAIN_COIN_FLIP_BEFORE COIN_FLIP_FIRST_ONLINE COIN_FLIP_USE_PRIORITIZATION <<< "$PARAMS"

# Get current date in required format
CURRENT_DATE=$(date +%Y-%m-%d-%H-%M)

# Generate output filename using dcs pattern (shortened)
PATTERN="${CURRENT_DATE}${ONLY_EVAL_NEG}_${PRETRAIN}_${REWARD_PRETRAIN}_${PROMPT_DATA}_${RM_TYPE}${BASE_PROP}${THRESH}${STARTB}${TARGET_BETA}${SEPB}${UNIW}${KL}_len${GEN_MAX_LEN}_${PARAMETERIZATION}${INITHEADBASE}${SD_DIVIDER}_b${MICRO_TRAIN}_${TRAIN}${ANALYTIC_BATCH}_${MICRO_ROLLOUT}_${ROLLOUT}${DUP_ROLLOUT}_epo${MAX_EPOCHS}${EPI_STR}${HARMLOSS}${HARMLOSSREINBASELINE}${HLRBVAL}${STARTA}${ALPHA}${RTA}${RTB}${BASEACTOR_LR}_${ACTOR_LOSS}${ACTOR_LR}${CRITIC_LR}_${LR_SCHED}${CUSTOM_PROMPT}${SAVE_NEGDATA_THRESH}${EXPLORATION_BONUS_SA}${EXPLORATION_BONUS_BA}${BONUS_ALPHA}${COIN_FLIP_DIM}${COIN_FLIP_LR}${COIN_FLIP_NORM_MOMENTUM}${COIN_FLIP_UPDATE_STEPS}${COIN_FLIP_HEAD_INIT_STD}${FROZEN_PRIOR_INIT_STD}${COIN_FLIP_LINEAR_BIAS}${COIN_FLIP_ARCHITECTURE}${TRAIN_COIN_FLIP_BEFORE}${COIN_FLIP_FIRST_ONLINE}${COIN_FLIP_USE_PRIORITIZATION}"
SBATCH_FILE="sbatch_${PATTERN}"
OUTPUT_FILE="result_${PATTERN}_s1.txt"

# Create the sbatch file based on cluster
case "$CLUSTER" in
    "dcs"|"simple")
        cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodelist=overture,quartet[1-5]
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --output=$OUTPUT_FILE
#SBATCH --gres=gpu:1
source /pkgs/anaconda310/etc/profile.d/conda.sh
conda activate openrlhf
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export MAX_JOBS=1
cd ~/OpenRLHF
deepspeed --master_port $(($RANDOM % 1000 + 3000))1 $COMMAND
EOL
        ;;
    "deadline")
        cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --partition=a40
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --export=ALL
#SBATCH --output=$OUTPUT_FILE
#SBATCH --gres=gpu:1
cd ~
ln -s /usr/bin/gcc-10 .local/bin/gcc
ln -s /usr/bin/g++-10 .local/bin/g++
export PATH=\$HOME/.local/bin/:\$PATH
cd ~/OpenRLHF
source newenv/bin/activate
module load cuda-12.3
deepspeed --master_port $(($RANDOM % 1000 + 3000))1 $COMMAND
EOL
        ;;
    "multinode")
        cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --export=ALL
#SBATCH --output=$OUTPUT_FILE
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=4
#SBATCH --exclude=kn003
cd ~
export PATH=\$HOME/.local/bin/:\$PATH
cd ~/projects/aip-rgrosse/zhaostep/OpenRLHF
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load cuda/12.6
module load scipy-stack/2024a
module load gcc arrow/18.1.0
module load opencv/4.12.0
module load rust
source ENV/bin/activate
source ~/.hf_token
deepspeed --master_port $(($RANDOM % 1000 + 3000))1 $COMMAND
EOL
        ;;
    "default"|*)
        cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --export=ALL
#SBATCH --output=$OUTPUT_FILE
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
cd ~
export PATH=\$HOME/.local/bin/:\$PATH
cd ~/projects/aip-rgrosse/zhaostep/OpenRLHF
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load cuda/12.6
module load scipy-stack/2024a
module load gcc arrow/18.1.0
module load opencv/4.12.0
module load rust
source ENV/bin/activate
source ~/.hf_token
deepspeed --master_port $(($RANDOM % 1000 + 3000))1 $COMMAND
EOL
        ;;
esac

# Make the sbatch file executable
chmod +x "$SBATCH_FILE"
echo "Created sbatch file: $SBATCH_FILE"
echo "Output will be written to: $OUTPUT_FILE"

