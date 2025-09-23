#!/bin/bash
if [ "$#" -eq 0 ]; then
    echo "Error: Please provide the training command"
    exit 1
fi

# Store the full command
COMMAND="$*"

# Extract parameters using awk
PARAMS=$(echo "$COMMAND" | awk '
{
    # Initialize empty variables
    micro_train = train = micro_rollout = rollout = ""
    max_epochs = num_episodes = num_episodes_h = gen_max_len = actor_lr = critic_lr = baseactor_lr = ""
    target_beta = lr_sched = actor_loss = kl = do_harmlessness = rta = rtb = startb = sepb = uniw = ""
    custom_prompt = prompt_data = parameterization = adam_beta2 = rm_type = dup_rollout = pretrain = reward_pretrain = init_head_from_base = ""
    sd_divider = harmloss = harmlossreinbaseline = hlrbval = ""
    save_negdata_threshold = threshold = alpha = only_eval_neg = use_base_as_proposal = ""
    
    # Scan through all matches in the string
    for(i=1; i<=NF; i++) {
        if($i == "--micro_train_batch_size") micro_train = $(i+1)
        if($i == "--train_batch_size") train = $(i+1)
        if($i == "--micro_rollout_batch_size") micro_rollout = $(i+1)
        if($i == "--rollout_batch_size") rollout = $(i+1)
        if($i == "--max_epochs") max_epochs = $(i+1)
        if($i == "--num_episodes") num_episodes = $(i+1)
        if($i == "--harmlessness_training_num_episodes") num_episodes_h = $(i+1)
        if($i == "--generate_max_len") gen_max_len = $(i+1)
        if($i == "--actor_learning_rate") actor_lr = $(i+1)
        if($i == "--critic_learning_rate") critic_lr = $(i+1)
        if($i == "--base_actor_learning_rate") baseactor_lr = "_baselr"$(i+1)
        # if($i == "--target_dist_beta") target_beta = "_beta"$(i+1)
        if($i ~ /^--target_dist_beta(=|$)/) target_beta = ($i ~ /=/) ? gensub(/^[^=]+=/, "", "g", $i) : "_beta"$(i+1)
        if($i ~ /^--save_negdata_threshold(=|$)/) save_negdata_threshold = ($i ~ /=/) ? "_savethr" gensub(/^[^=]+=/, "", "g", $i) : "_savethr" $(i+1)
        if($i ~ /^--threshold(=|$)/) threshold = ($i ~ /=/) ? "_thresh" gensub(/^[^=]+=/, "", "g", $i) : "_thresh" $(i+1)
        if($i == "--lr_scheduler") lr_sched = $(i+1)
        if($i == "--actor_loss_type") actor_loss = $(i+1)
        if($i == "--custom_single_prompt") custom_prompt = "_custom"
        if($i == "--parameterization") parameterization = $(i+1)
        if($i == "--adam_betas") adam_beta2 = "_adambeta2_"$(i+2)
        if($i == "--rm_type") rm_type = $(i+1)
        if($i == "--duplicate_rollout_batch_by") dup_rollout = "_"$(i+1)
        if($i == "--pretrain") {
            abbrev = ""
            n = split(gensub(".*/", "", "g", $(i+1)), arr, "-")
            for (j = 1; j <= n; j++) {
                abbrev = abbrev substr(arr[j], 1, 2)  # Append first 2 characters of each word
            }
            pretrain = abbrev
        }
        if($i == "--reward_pretrain") {
            abbrev = ""
            n = split(gensub(".*/", "", "g", $(i+1)), arr, "-")
            for (j = 1; j <= n; j++) {
                abbrev = abbrev substr(arr[j], 1, 2)  # Append first 2 characters of each word
            }
            reward_pretrain = abbrev
        }
        if($i == "--prompt_data") prompt_data = gensub("_.*", "", "g", gensub(".*/", "", "g", $(i+1)))
        if($i == "--init_head_from_base") init_head_from_base = "_initheadbase"
        if($i == "--additional_sd_divider") sd_divider = "_sddivider"$(i+1)
        if($i == "--harmlessness_training_loss_type") harmloss = "_harml"$(i+1)
        if($i == "--reinforce_baseline_type") harmlossreinbaseline = "_"$(i+1)
        if($i == "--reinforce_hardcoded_baseline") hlrbval = "_"$(i+1)
        if($i == "--alpha") alpha = "_alpha"$(i+1)
        if($i == "--init_kl_coef") kl = "_kl"$(i+1)
        if($i == "--only_evaluate_on_neg_data") only_eval_neg = "_onlyevalneg"
        if($i == "--do_harmlessness_training") do_harmlessness = 1
        if($i == "--use_base_as_proposal") use_base_as_proposal = "_baseprop"
        if($i == "--rew_trans_alpha") rta = "_rta"$(i+1)
        if($i == "--rew_trans_beta") rtb = "_rtb"$(i+1)
        if($i == "--start_target_dist_beta") startb = "_start"$(i+1)
        if($i == "--separate_reweighting_beta") sepb = "_sepb"$(i+1)
        if($i == "--uniform_reweight") uniw = "_uniw"
    }
    # Use num_episodes_h if do_harmlessness is set
    episodes_to_use = do_harmlessness ? num_episodes_h : num_episodes
    
        print micro_train "|" train "|" micro_rollout "|" rollout "|" max_epochs "|" episodes_to_use "|" \
              gen_max_len "|" actor_lr "|" critic_lr "|" baseactor_lr "|" target_beta "|" save_negdata_threshold "|" threshold "|" lr_sched "|" \
              actor_loss "|" custom_prompt "|" parameterization "|" adam_beta2 "|" rm_type "|" dup_rollout "|" pretrain "|" \
              reward_pretrain "|" prompt_data "|" init_head_from_base "|" sd_divider "|" harmloss "|" harmlossreinbaseline "|" hlrbval "|" alpha "|" kl "|" only_eval_neg "|" use_base_as_proposal "|" rta "|" rtb "|" startb "|" sepb "|" uniw  \
}')

# Read using the special delimiter
IFS='|' read MICRO_TRAIN TRAIN MICRO_ROLLOUT ROLLOUT MAX_EPOCHS NUM_EPISODES GEN_MAX_LEN \
    ACTOR_LR CRITIC_LR BASEACTOR_LR TARGET_BETA SAVE_NEGDATA_THRESH THRESH LR_SCHED ACTOR_LOSS CUSTOM_PROMPT PARAMETERIZATION ADAM_BETA2 RM_TYPE DUP_ROLLOUT PRETRAIN REWARD_PRETRAIN PROMPT_DATA \
    INITHEADBASE SD_DIVIDER HARMLOSS HARMLOSSREINBASELINE HLRBVAL ALPHA KL ONLY_EVAL_NEG BASE_PROP RTA RTB STARTB SEPB UNIW <<< "$PARAMS"

# echo $PRETRAIN
# PRETRAIN="${PRETRAIN%%/*}"
# echo $PRETRAIN


# Get current date in required format
CURRENT_DATE=$(date +%Y-%m-%d-%H-%M)

# Generate output filename
PATTERN="${CURRENT_DATE}${ONLY_EVAL_NEG}_${PRETRAIN}_${REWARD_PRETRAIN}_${PROMPT_DATA}_${RM_TYPE}${BASE_PROP}${THRESH}${STARTB}${TARGET_BETA}${SEPB}${UNIW}${KL}_len${GEN_MAX_LEN}_${PARAMETERIZATION}${INITHEADBASE}${SD_DIVIDER}_batch${MICRO_TRAIN}_${TRAIN}_${MICRO_ROLLOUT}_${ROLLOUT}${DUP_ROLLOUT}_epo${MAX_EPOCHS}_epi${NUM_EPISODES}${HARMLOSS}${HARMLOSSREINBASELINE}${HLRBVAL}${ALPHA}${RTA}${RTB}${BASEACTOR_LR}_${ACTOR_LOSS}_alr${ACTOR_LR}_clr${CRITIC_LR}_${LR_SCHED}${CUSTOM_PROMPT}${SAVE_NEGDATA_THRESH}"
SBATCH_FILE="sbatch_${PATTERN}"
OUTPUT_FILE="result_${PATTERN}_s1.txt"

# Create the sbatch file
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

# Make the sbatch file executable
chmod +x "$SBATCH_FILE"
echo "Created sbatch file: $SBATCH_FILE"
echo "Output will be written to: $OUTPUT_FILE"
