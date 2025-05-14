#!/bin/bash
#SBATCH --job-name=forecast_exp
#SBATCH --output=forecasting_job_%j.out
#SBATCH --error=forecasting_job_%j.err
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=concerto[1-3],overture,quartet[1-3,5],dgx1,sonata[1-2],bulbasaur,charmander,squirtle
#SBATCH --ntasks-per-node=1         # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=32G                   # Memory per node (e.g., 32GB)
#SBATCH --time=01-00:00:00

# --- Environment Setup ---
. /mfs1/u/$USER/envs/OpenRLHF

echo "Starting experiment run..."
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $HOSTNAME"
echo "Allocated GRES: $SLURM_JOB_GRES"
echo "Python executable: $(which python)"

CHECKPOINT_DIR="/mfs1/u/aidanl/openrlhf/checkpoints"
CHECKPOINT_SPECIFIC="13-05-2025/p-prop"
CHECKPOINT_SEED="s1"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_SPECIFIC}/${CHECKPOINT_SEED}"
BEHAVIOR_ID="insult_generation"

deepspeed --master_port $(($RANDOM % 1000 + 3000))1 --module openrlhf.forecasting_rare_outputs.experiment_runner \
    --pretrain "HuggingFaceTB/SmolLM-135M-Instruct" \
    --ckpt_path "${CHECKPOINT_PATH}" \
    --behavior_id "${BEHAVIOR_ID}" \
    --query_file "openrlhf/forecasting_rare_outputs/split_20k_heldout/${BEHAVIOR_ID}_prompts.jsonl" \
    --output_dir "openrlhf/forecasting_rare_outputs/results/${CHECKPOINT_SPECIFIC}/${CHECKPOINT_SEED}" \
    --evaluation_set_size 100 \
    --elicitation_method "repeated_sampling" \
    --k_samples 10000 \
    --elicitation_processing_batch_size 1200 \
    --top_k_fit 10 \
    --forecast_scales "1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6" \
    --bf16 
    # --flash_attn # Uncomment if your model/setup supports Flash Attention
    # --lora_path "path/to/your/lora_model" # Uncomment and set if using LoRA
# 'logprob_target_seq', 'logprob_target_keyword_in_target_seq', 'repeated_sampling'
echo "Experiment command finished."
echo "Job finished with exit code $?." 