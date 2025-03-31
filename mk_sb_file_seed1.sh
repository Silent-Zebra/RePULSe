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
    max_epochs = gen_max_len = actor_lr = critic_lr = ""
    target_beta = lr_sched = actor_loss = ""
    custom_prompt = parameterization = adam_beta2 = rm_type = dup_rollout = pretrain = init_head_from_base = ""
    sd_divider = harmloss = harmlossreinbaseline = ""
    save_negdata_threshold = threshold = ""
    
    # Scan through all matches in the string
    for(i=1; i<=NF; i++) {
        if($i == "--micro_train_batch_size") micro_train = $(i+1)
        if($i == "--train_batch_size") train = $(i+1)
        if($i == "--micro_rollout_batch_size") micro_rollout = $(i+1)
        if($i == "--rollout_batch_size") rollout = $(i+1)
        if($i == "--max_epochs") max_epochs = $(i+1)
        if($i == "--generate_max_len") gen_max_len = $(i+1)
        if($i == "--actor_learning_rate") actor_lr = $(i+1)
        if($i == "--critic_learning_rate") critic_lr = $(i+1)
        # if($i == "--target_dist_beta") target_beta = $(i+1)
        if($i ~ /^--target_dist_beta(=|$)/) target_beta = ($i ~ /=/) ? gensub(/^[^=]+=/, "", "g", $i) : $(i+1)
        if($i ~ /^--save_negdata_threshold(=|$)/) save_negdata_threshold = ($i ~ /=/) ? "_savenegdatathresh" gensub(/^[^=]+=/, "", "g", $i) : "_savenegdatathresh" $(i+1)
        if($i ~ /^--threshold(=|$)/) threshold = ($i ~ /=/) ? "_thresh" gensub(/^[^=]+=/, "", "g", $i) : "_thresh" $(i+1)
        if($i == "--lr_scheduler") lr_sched = $(i+1)
        if($i == "--actor_loss_type") actor_loss = $(i+1)
        if($i == "--custom_single_prompt") custom_prompt = "_custom"
        if($i == "--parameterization") parameterization = $(i+1)
        if($i == "--adam_betas") adam_beta2 = "_adambeta2_"$(i+2)
        if($i == "--rm_type") rm_type = $(i+1)
        if($i == "--duplicate_rollout_batch_by") dup_rollout = "_duprollout"$(i+1)
        if($i == "--pretrain") pretrain = $(i+1)
        if($i == "--init_head_from_base") init_head_from_base = "_initheadbase"
        if($i == "--additional_sd_divider") sd_divider = "_sddivider"$(i+1)
        if($i == "--harmlessness_training_loss_type") harmloss = "_harmloss"$(i+1)
        if($i == "--reinforce_baseline_type") harmlossreinbaseline = "_harmlossreinbaseline"$(i+1)
    }
    
    # Print with a special delimiter (|) that wont appear in the values
    if(micro_train != "" && train != "" && micro_rollout != "" && rollout != "" && 
       max_epochs != "" && gen_max_len != "" && actor_lr != "" && critic_lr != "" && 
       target_beta != "" && lr_sched != "" && actor_loss != "")
        print micro_train "|" train "|" micro_rollout "|" rollout "|" max_epochs "|" \
              gen_max_len "|" actor_lr "|" critic_lr "|" target_beta "|" save_negdata_threshold "|" threshold "|" lr_sched "|" \
              actor_loss "|" custom_prompt "|" parameterization "|" adam_beta2 "|" rm_type "|" dup_rollout "|" pretrain "|" init_head_from_base "|" sd_divider "|" harmloss "|" harmlossreinbaseline \

}')

# Read using the special delimiter
IFS='|' read MICRO_TRAIN TRAIN MICRO_ROLLOUT ROLLOUT MAX_EPOCHS GEN_MAX_LEN \
    ACTOR_LR CRITIC_LR TARGET_BETA SAVE_NEGDATA_THRESH THRESH LR_SCHED ACTOR_LOSS CUSTOM_PROMPT PARAMETERIZATION ADAM_BETA2 RM_TYPE DUP_ROLLOUT PRETRAIN INITHEADBASE SD_DIVIDER HARMLOSS HARMLOSSREINBASELINE <<< "$PARAMS"

# Check if required parameters are empty
if [ -z "$MICRO_TRAIN" ] || [ -z "$TRAIN" ] || [ -z "$MICRO_ROLLOUT" ] || [ -z "$ROLLOUT" ] || \
   [ -z "$MAX_EPOCHS" ] || [ -z "$GEN_MAX_LEN" ] || [ -z "$ACTOR_LR" ] || [ -z "$CRITIC_LR" ] || \
   [ -z "$TARGET_BETA" ] || [ -z "$LR_SCHED" ] || [ -z "$ACTOR_LOSS" ] || [ -z "$PARAMETERIZATION" ] || [ -z "$RM_TYPE" ] || [ -z "$PRETRAIN" ]; then
    echo "Error: Missing required parameters"
    exit 1
fi

# echo $PRETRAIN

PRETRAIN="${PRETRAIN%%/*}"

# echo $PRETRAIN


# Get current date in required format
CURRENT_DATE=$(date +%Y-%m-%d-%H-%M)

# Generate output filename
PATTERN="${CURRENT_DATE}_${PRETRAIN}_${RM_TYPE}${THRESH}_beta${TARGET_BETA}_len${GEN_MAX_LEN}_${PARAMETERIZATION}${INITHEADBASE}${SD_DIVIDER}_batch${MICRO_TRAIN}_${TRAIN}_${MICRO_ROLLOUT}_${ROLLOUT}${DUP_ROLLOUT}_ep${MAX_EPOCHS}${HARMLOSS}${HARMLOSSREINBASELINE}_${ACTOR_LOSS}_alr${ACTOR_LR}_clr${CRITIC_LR}_${LR_SCHED}${CUSTOM_PROMPT}${ADAM_BETA2}${SAVE_NEGDATA_THRESH}"
SBATCH_FILE="sbatch_${PATTERN}"
OUTPUT_FILE="result_${PATTERN}_s1.txt"

# Create the sbatch file
cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --partition=a40
#SBATCH --qos=m3
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

# Make the sbatch file executable
chmod +x "$SBATCH_FILE"
echo "Created sbatch file: $SBATCH_FILE"
echo "Output will be written to: $OUTPUT_FILE"
