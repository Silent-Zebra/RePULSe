#!/bin/bash
#SBATCH --job-name=forecast_exp
#SBATCH --output=forecasting_job_%j.out
#SBATCH --error=forecasting_job_%j.err
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=overture,quartet[1-2],quartet[4-5],concerto[1-3],bulbasaur,squirtle,charmander
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

CHECKPOINT_DIR="/mfs1/u/aidanl/openrlhf/checkpoints/harmlessness"
CHECKPOINT_NAME="toy_rlhf_Sm13In_remodev3lav2_miprAL_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_nosecondterm_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5_harml_actor"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"

deepspeed --master_port 35996 --module openrlhf.forecasting_rare_outputs.experiment_runner \
    --pretrain "HuggingFaceTB/SmolLM-135M-Instruct" \
    --ckpt_path "${CHECKPOINT_PATH}" \
    --behavior_id "drugs_keyword" \
    --query_file "openrlhf/forecasting_rare_outputs/query_pools/drugs_keyword_queries.jsonl" \
    --evaluation_set_size 100 \
    --max_queries 100 \
    --elicitation_method "repeated_sampling" \
    --k_samples 200 \
    --top_k_fit 6 \
    --forecast_scales "1e3,1e4,1e5,1e6" \
    --bf16 
    # --flash_attn # Uncomment if your model/setup supports Flash Attention
    # --lora_path "path/to/your/lora_model" # Uncomment and set if using LoRA

echo "Experiment command finished."
echo "Job finished with exit code $?." 