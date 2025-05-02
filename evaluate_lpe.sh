#!/bin/bash
#SBATCH -J lpe_harmlessness
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=concerto1,concerto2,concerto3,overture
#SBATCH --export=ALL
#SBATCH --output=lpe_harmlessness_%j.txt
#SBATCH --time=02:00:00

# Load the environment
. /mfs1/u/$USER/envs/openrlhf

# === Model and Checkpoint Setup ===
CHECKPOINT_DIR="/mfs1/u/aidanl/openrlhf/checkpoints/harmlessness"
CHECKPOINT_NAME="toy_rlhf_Sm13In_remodev3lav2_miprAL_len20_beta-10.0_harml_neg_training_a0.01_policy_psi_q_p_s_t_ctl_nosecondterm_epochs1_schedconstant_alr0.0001_blr3e-05_policy_psi_q_p_s_t_s5_harml_actor"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"

# === LPE Parameters ===
METHOD="itgis"                # "itgis" or "mhis"
DISTRIBUTION="uniform"        # "uniform" or "hex"
SEQ_LEN=20
TARGET_WORD="kill"         # <-- REPLACE with your actual target word
TEMPERATURE=5
N_SAMPLES=65536               # 2^16=65536
BATCH_SIZE=256
OUTPUT_DIR="./lpe_results/${METHOD}"
OUTPUT_FILE="${OUTPUT_DIR}/results_${TARGET_WORD}_${METHOD}_${DISTRIBUTION}_temp${TEMPERATURE}_${SLURM_JOB_ID}.json"

# Run the LPE estimation script
deepspeed --master_port 40002 --module openrlhf.cli.estimate_rare_harmful_outputs \
    --pretrain "HuggingFaceTB/SmolLM-135M-Instruct" \
    --load_checkpoint \
    --ckpt_path "${CHECKPOINT_PATH}" \
    --bf16 True \
    --zero_stage 2 \
    --parameterization policy_psi_q_p_s_t \
    --distribution_name $DISTRIBUTION \
    --sequence_length $SEQ_LEN \
    --target_word $TARGET_WORD \
    --method $METHOD \
    --temperature $TEMPERATURE \
    --n_samples $N_SAMPLES \
    --batch_size $BATCH_SIZE \
    --output_file $OUTPUT_FILE \
    --show_progress \
    --use_argmax

echo "LPE estimation finished. Results saved to $OUTPUT_FILE" 