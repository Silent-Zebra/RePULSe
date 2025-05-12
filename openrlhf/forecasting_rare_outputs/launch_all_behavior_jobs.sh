#!/bin/bash

# --- Configuration ---
JOB_NAME="harmless-p1"
CHECKPOINT_DIR="/h/liaidan/OpenRLHF/checkpoints"
CHECKPOINT_SPECIFIC="10-05-2025/${JOB_NAME}" # Update this if needed
CHECKPOINT_SEED="s2" # Update this if needed
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_SPECIFIC}/${CHECKPOINT_SEED}"
PROMPT_DIR="openrlhf/forecasting_rare_outputs/split_20k_heldout"
OUTPUT_BASE_DIR="openrlhf/forecasting_rare_outputs/results/${CHECKPOINT_SPECIFIC}/${CHECKPOINT_SEED}"
PRETRAIN_MODEL="HuggingFaceTB/SmolLM-135M-Instruct"
EVALUATION_SET_SIZE=100
ELICITATION_METHOD="repeated_sampling" # Or 'repeated_sampling', 'logprob_target_seq' etc.
K_SAMPLES=10000
BATCH_SIZE=1200
TOP_K_FIT=10
FORECAST_SCALES="1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6"
SBATCH_ACCOUNT="deadline" # Or your account
SBATCH_PARTITION="a40" # Or your partition
SBATCH_QOS="deadline" # Or your QOS
SBATCH_TIME="12:00:00"
SBATCH_MEM="10G"
SBATCH_CPUS_PER_TASK=4
SBATCH_GPUS=1
SBATCH_NODES=1
# --- End Configuration ---

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE_ROOT="${SCRIPT_DIR}/../.." # Adjust if your script location is different
PROMPT_DIR_ABS="${WORKSPACE_ROOT}/${PROMPT_DIR}"
OUTPUT_BASE_DIR_ABS="${WORKSPACE_ROOT}/${OUTPUT_BASE_DIR}"

echo "Workspace Root: ${WORKSPACE_ROOT}"
echo "Prompt Directory: ${PROMPT_DIR_ABS}"
echo "Output Base Directory: ${OUTPUT_BASE_DIR_ABS}"
echo "Checkpoint Path: ${CHECKPOINT_PATH}"

# Create base output directory if it doesn't exist
mkdir -p "${OUTPUT_BASE_DIR_ABS}"

# Find all prompt files and loop through them
find "${PROMPT_DIR_ABS}" -maxdepth 1 -name '*_prompts.jsonl' | while read -r full_query_file; do
    filename=$(basename "$full_query_file")
    behavior_id="${filename%_prompts.jsonl}"

    # Construct relative path for the query file argument
    relative_query_file="${PROMPT_DIR}/${filename}"

    # Define unique job name and output/error files
    job_name="${JOB_NAME}"
    output_file="${OUTPUT_BASE_DIR_ABS}/slurm_logs/forecasting_job_${behavior_id}_%j.out"
    error_file="${OUTPUT_BASE_DIR_ABS}/slurm_logs/forecasting_job_${behavior_id}_%j.err"

    # Create SLURM log directory if it doesn't exist
    mkdir -p "$(dirname "$output_file")"

    echo "------------------------------------"
    echo "Submitting job for behavior: ${behavior_id}"
    echo "  Query file: ${relative_query_file}"
    echo "  Job Name: ${job_name}"
    echo "  Output file: ${output_file}"
    echo "  Error file: ${error_file}"
    echo "------------------------------------"

    # Submit the sbatch job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${output_file}
#SBATCH --error=${error_file}
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --account=${SBATCH_ACCOUNT}
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

echo "Starting experiment run for behavior: ${behavior_id}"
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
    --output_dir "${OUTPUT_BASE_DIR_ABS}/${behavior_id}" \\
    --evaluation_set_size ${EVALUATION_SET_SIZE} \\
    --elicitation_method "${ELICITATION_METHOD}" \\
    --k_samples ${K_SAMPLES} \\
    --elicitation_processing_batch_size ${BATCH_SIZE} \\
    --top_k_fit ${TOP_K_FIT} \\
    --forecast_scales "${FORECAST_SCALES}" \\
    --bf16
    # --flash_attn # Uncomment if needed
    # --lora_path "path/to/your/lora_model" # Uncomment if needed

echo "Experiment command finished for behavior: ${behavior_id}."
echo "Job finished with exit code \$?."

EOF

    # Optional: Add a small delay between submissions if needed
    sleep 2
done

echo "All jobs submitted." 