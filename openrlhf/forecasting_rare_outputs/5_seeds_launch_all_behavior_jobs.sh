#!/bin/bash

# --- Configuration ---
JOB_NAME="reinforce"
CHECKPOINT_DIR="/h/liaidan/OpenRLHF/checkpoints"
CHECKPOINT_SPECIFIC="13-05-2025/${JOB_NAME}" # Update this if needed
# CHECKPOINT_SEED="s2" # This will be set by the loop
PROMPT_DIR="openrlhf/forecasting_rare_outputs/split_20k_heldout"
# OUTPUT_BASE_DIR will be defined inside the loop
PRETRAIN_MODEL="HuggingFaceTB/SmolLM-135M-Instruct"
EVALUATION_SET_SIZE=150
ELICITATION_METHOD="logprob_target_seq" # 'logprob_target_seq', 'logprob_target_keyword_in_target_seq', 'repeated_sampling'
NUM_BOOTSTRAP_SAMPLES=20
K_SAMPLES=10000
BATCH_SIZE=1200
TOP_K_FIT=4
FORECAST_SCALES="1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6"
SBATCH_PARTITION="a40,rtx6000,t4v1,t4v2" # Or your partition
SBATCH_QOS="m5" # Or your QOS
SBATCH_TIME="1:00:00"
SBATCH_MEM="1G"
SBATCH_CPUS_PER_TASK=1
SBATCH_GPUS=1
SBATCH_NODES=1

SEEDS=("s1" "s2" "s3" "s4" "s5" "s6" "s7" "s8" "s9" "s10")
# --- End Configuration ---

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE_ROOT="${SCRIPT_DIR}/../.." # Adjust if your script location is different
PROMPT_DIR_ABS="${WORKSPACE_ROOT}/${PROMPT_DIR}"

echo "Workspace Root: ${WORKSPACE_ROOT}"
echo "Prompt Directory (absolute): ${PROMPT_DIR_ABS}"
echo "Looping through seeds: ${SEEDS[*]}"

for CURRENT_CHECKPOINT_SEED in "${SEEDS[@]}"; do
    echo ""
    echo "================================================="
    echo "PROCESSING SEED: ${CURRENT_CHECKPOINT_SEED}"
    echo "================================================="

    CHECKPOINT_SEED="${CURRENT_CHECKPOINT_SEED}" # Set the current seed
    CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_SPECIFIC}/${CHECKPOINT_SEED}"
    CURRENT_OUTPUT_BASE_DIR_REL="openrlhf/forecasting_rare_outputs/results/nobootstrap_top${TOP_K_FIT}_fit/${CHECKPOINT_SPECIFIC}/${CHECKPOINT_SEED}"
    CURRENT_OUTPUT_BASE_DIR_ABS="${WORKSPACE_ROOT}/${CURRENT_OUTPUT_BASE_DIR_REL}"

    echo "  Checkpoint Path: ${CHECKPOINT_PATH}"
    echo "  Output Base Directory (relative): ${CURRENT_OUTPUT_BASE_DIR_REL}"
    echo "  Output Base Directory (absolute): ${CURRENT_OUTPUT_BASE_DIR_ABS}"

    # Create base output directory for the current seed if it doesn\'t exist
    mkdir -p "${CURRENT_OUTPUT_BASE_DIR_ABS}"
    echo "  Ensured output directory exists: ${CURRENT_OUTPUT_BASE_DIR_ABS}"

    # Find all prompt files and loop through them
    find "${PROMPT_DIR_ABS}" -maxdepth 1 -name '*_prompts.jsonl' | while read -r full_query_file; do
        echo "DEBUG: Entering while loop. full_query_file = '${full_query_file}'"
        filename=$(basename "$full_query_file")
        behavior_id="${filename%_prompts.jsonl}"

        # Construct relative path for the query file argument
        relative_query_file="${PROMPT_DIR}/${filename}"

        # Define unique job name for sbatch and output/error files
        sbatch_job_name="${CHECKPOINT_SEED}-${JOB_NAME}"
        output_file="${CURRENT_OUTPUT_BASE_DIR_ABS}/slurm_logs/forecasting_job_${behavior_id}_%j.out"
        error_file="${CURRENT_OUTPUT_BASE_DIR_ABS}/slurm_logs/forecasting_job_${behavior_id}_%j.err"

        # Create SLURM log directory if it doesn\'t exist
        mkdir -p "$(dirname "$output_file")"

        echo "  ------------------------------------"
        echo "  Submitting job for behavior: ${behavior_id} with SEED: ${CHECKPOINT_SEED}"
        echo "    Query file (relative): ${relative_query_file}"
        echo "    Sbatch Job Name: ${sbatch_job_name}"
        echo "    Output file: ${output_file}"
        echo "    Error file: ${error_file}"
        echo "  ------------------------------------"

        # Submit the sbatch job
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${sbatch_job_name}
#SBATCH --output=${output_file}
#SBATCH --error=${error_file}
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --nodes=${SBATCH_NODES}
#SBATCH --gres=gpu:${SBATCH_GPUS}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${SBATCH_CPUS_PER_TASK}
#SBATCH --mem=${SBATCH_MEM}

# --- Environment Setup ---
# Activate your conda environment if needed
# conda activate OpenRLHF
# Or source your environment setup script
source "${WORKSPACE_ROOT}/newenv/bin/activate" 

echo "Starting experiment run for behavior: ${behavior_id}, SEED: ${CHECKPOINT_SEED}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Running on host: \$HOSTNAME"
echo "Allocated GRES: \$SLURM_JOB_GRES"
echo "Python executable: \$(which python)"
echo "Working directory: \$(pwd)" # Should be the workspace root if launched from there

# Navigate to the workspace root before running the command
cd "${WORKSPACE_ROOT}"

deepspeed --master_port \$((\$RANDOM % 1000 + 3000))1 --module openrlhf.forecasting_rare_outputs.experiment_runner \\
    --pretrain "${PRETRAIN_MODEL}" \\
    --ckpt_path "${CHECKPOINT_PATH}" \\
    --behavior_id "${behavior_id}" \\
    --query_file "${relative_query_file}" \\
    --output_dir "${CURRENT_OUTPUT_BASE_DIR_ABS}/${behavior_id}" \\
    --evaluation_set_size ${EVALUATION_SET_SIZE} \\
    --elicitation_method "${ELICITATION_METHOD}" \\
    --num_bootstrap_samples ${NUM_BOOTSTRAP_SAMPLES} \\
    --use_all_queries_no_bootstrap \\
    --k_samples ${K_SAMPLES} \\
    --elicitation_processing_batch_size ${BATCH_SIZE} \\
    --top_k_fit ${TOP_K_FIT} \\
    --forecast_scales "${FORECAST_SCALES}" \\
    --bf16
    # --flash_attn # Uncomment if needed
    # --lora_path "path/to/your/lora_model" # Uncomment if needed

echo "Experiment command finished for behavior: ${behavior_id}, SEED: ${CHECKPOINT_SEED}."
echo "Job finished with exit code \$?."

EOF

        # Optional: Add a small delay between submissions if needed
        # sleep 1
    done # End of behavior_id loop

    echo "All jobs for SEED ${CHECKPOINT_SEED} submitted."
    echo "================================================="
done # End of SEED loop

echo ""
echo "All jobs for all seeds submitted." 