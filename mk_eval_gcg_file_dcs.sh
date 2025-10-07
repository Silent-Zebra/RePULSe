STEPS=$1
OUT_LEN=$2
CKPT_PATH=$3
CKPT_ABBREV="${CKPT_PATH##*/}"
MODEL=$4
RM=$5
THRESHOLD=$6
DATA_CSV=$7

# Get current date in required format
CURRENT_DATE=$(date +%Y-%m-%d-%H-%M)

# Generate output filename
SBATCH_FILE="sbatch_${CURRENT_DATE}_eval_gcg${STEPS}_${CKPT_ABBREV}"
OUTPUT_FILE="result_${CURRENT_DATE}_eval_gcg${STEPS}_${CKPT_ABBREV}.txt"

# Create the sbatch file
cat > "$SBATCH_FILE" << EOL
#!/bin/bash
#SBATCH -J s1_$(($RANDOM % 100000))
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodelist=overture,quartet[1-2],quartet[4-5]
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --output=$OUTPUT_FILE
#SBATCH --gres=gpu:1
cd ~
ln -s /usr/bin/gcc-10 .local/bin/gcc
ln -s /usr/bin/g++-10 .local/bin/g++
export PATH=\$HOME/.local/bin/:\$PATH
cd ~/OpenRLHF
source newenv/bin/activate
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
deepspeed --master_port $(($RANDOM % 1000 + 3000))1 --module openrlhf.cli.evaluate_gcg_sz \
    --pretrain "${MODEL}" \
    --ckpt_path ${CKPT_PATH} \
    --file_path data/${DATA_CSV} \
    --atk_success_criteria reward \
    --reward_pretrain "${RM}" \
    --reward_threshold ${THRESHOLD} \
    --rm_exclude_adv_suffix \
    --zero_stage 2 \
    --gradient_checkpointing \
    --parameterization policy \
    --max_targets 100 \
    --max_new_tokens ${OUT_LEN} \
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

