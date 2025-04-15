#!/bin/bash
#SBATCH -J s1_18111
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=overture,concerto1,concerto2,concerto3
#SBATCH --export=ALL
#SBATCH --output=behaviors_gcg_results_100prompts.txt

# Load the environment
. /mfs1/u/$USER/envs/openrlhf

CHECKPOINT_DIR="/mfs1/u/aidanl/openrlhf/checkpoints/toyrlhfmulti"
CHECKPOINT_NAME="toy_rlhf_Sm13In_remodev3lav2_cutopr_len20_beta-10.0_harml_neg_training_a0.5_policy_psi_q_p_s_t_ctl_epochs1_schedconstant_alr0.0001_blr1e-06_policy_psi_q_p_s_t_s1_actor"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"

deepspeed --master_port 33892 --module openrlhf.cli.evaluate_gcg \
    --pretrain "HuggingFaceTB/SmolLM-135M-Instruct" \
    --load_checkpoint \
    --ckpt_path "${CHECKPOINT_PATH}" \
    --flash_attn \
    --bf16 \
    --zero_stage 2 \
    --gradient_checkpointing \
    --adam_offload \
    --parameterization policy_psi_q_p_s_t \
    --max_targets 100 \
    --scenario behaviors \
    --file_path data/bad_outputs.csv \
    --gcg_steps 10

# the scenario argument can be "behaviors" or "strings"
# the advbench_file_path argument can be the path to a csv file containing the harmful behaviors or strings
# Optional: the max_targets argument can be used to limit the number of targets processed from the advbench file instead of all 520 prompts.

# Optional: you can add these to the deepspeed command to change the GCG adversarial attack parameters
    # --gcg_steps 500 \
    # --gcg_search_width 512 \
    # --gcg_topk 256 \
    # --gcg_batch_size 64 \
    # --gcg_n_replace 1 \
    # --gcg_buffer_size 0 \
    # --gcg_use_mellowmax \
    # --gcg_mellowmax_alpha 1.0 \
    # --gcg_early_stop \
    # --gcg_use_prefix_cache \
    # --gcg_filter_ids \
    # --seed 42 \