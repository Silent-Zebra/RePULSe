STEPS=$1
CKPT_PATH=$2
CKPT_ABBREV="${CKPT_PATH##*/}"
MODEL=$3
RM=$4
THRESHOLD=$5
DATA_CSV=$6

# Get current date in required format
CURRENT_DATE=$(date +%Y-%m-%d-%H-%M)

# Generate output filename
SBATCH_FILE="sbatch_${CURRENT_DATE}_eval_gcg${STEPS}_${CKPT_ABBREV}"
OUTPUT_FILE="result_${CURRENT_DATE}_eval_gcg${STEPS}_${CKPT_ABBREV}.txt"

# Create the sbatch file
cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --mem=48G
#SBATCH --time=4:00:00
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
deepspeed --master_port $(($RANDOM % 1000 + 3000))1 --module openrlhf.cli.evaluate_gcg_sz \
    --pretrain "${MODEL}" \
    --ckpt_path ${CKPT_PATH} \
    --file_path data/${DATA_CSV} \
    --atk_success_criteria reward \
    --reward_pretrain "${RM}" \
    --reward_threshold ${THRESHOLD} \
    --rm_exclude_adv_suffix \
    --bf16 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --adam_offload \
    --parameterization policy \
    --max_targets 100 \
    --scenario behaviors \
    --gcg_steps ${STEPS} \
    --gcg_search_width 512 \
    --gcg_topk 256 \
    --gcg_batch_size 512 \
    --gcg_n_replace 1 \
    --gcg_buffer_size 0 \
    --gcg_use_prefix_cache \
    --gcg_filter_ids
EOL

# Make the sbatch file executable
chmod +x "$SBATCH_FILE"
echo "Created sbatch file: $SBATCH_FILE"
echo "Output will be written to: $OUTPUT_FILE"

