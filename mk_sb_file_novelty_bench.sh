#!/bin/bash

# Usage: bash mk_sb_file_novelty_bench.sh <MODEL> <SPLIT>
#   MODEL: e.g. meta-llama/Llama-3.2-1B-Instruct
#   SPLIT: e.g. curated
#
# Generates two sbatch files:
#   sbatch_infpart_* -- inference.py + partition.py  (1 GPU, 48G, 1h)
#   sbatch_score_*   -- score.py + summarize.py      (4 GPUs, 96G, 1h; 27B reward model)

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL> <SPLIT>"
    echo "  MODEL: e.g. meta-llama/Llama-3.2-1B-Instruct"
    echo "  SPLIT: e.g. curated"
    exit 1
fi

MODEL="$1"
SPLIT="$2"

# Hard-coded pipeline parameters (edit here to change)
NUM_GEN=10
PATIENCE=0.8
SAMPLING=regenerate
PART_ALG=classifier

# Build model abbreviation: basename, split on '-', first 2 chars of each component
# (matches the convention in mk_sb_file.sh:109-116)
MODEL_BASE="${MODEL##*/}"
IFS='-' read -ra PARTS <<< "$MODEL_BASE"
MODEL_ABBREV=""
for p in "${PARTS[@]}"; do
    MODEL_ABBREV="${MODEL_ABBREV}${p:0:2}"
done

# Patience abbreviation: strip the dot (0.8 -> 08)
PATIENCE_ABBREV="${PATIENCE/./}"

CURRENT_DATE=$(date +%Y-%m-%d-%H-%M)
STEM="novbench_${MODEL_ABBREV}_${SPLIT}_n${NUM_GEN}_p${PATIENCE_ABBREV}_${CURRENT_DATE}"

# Emit the common aip-rgrosse sbatch header + env setup.
# $MODEL and $SPLIT are baked in at generation time; $EVAL_DIR is a runtime shell var.
make_header() {
    local job_name="$1"
    local output_file="$2"
    local gpus="$3"
    local mem="$4"
    local time="$5"
    cat << EOL
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --export=ALL
#SBATCH --output=${output_file}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=${gpus}
#SBATCH --cpus-per-task=4
#SBATCH --exclude=kn001
cd ~
export PATH=\$HOME/.local/bin/:\$PATH
cd ~/projects/aip-rgrosse/zhaostep/novelty-bench
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5
module load cuda/12.6
module load scipy-stack/2024a
module load gcc arrow/18.1.0
module load opencv/4.12.0
module load rust
source ~/projects/aip-rgrosse/zhaostep/OpenRLHF/ENV2/bin/activate
source ~/.hf_token
export HF_HOME=/scratch/zhaostep/hf
set -e
EVAL_DIR="results/${SPLIT}/${MODEL}"
EOL
}

# Stage 1: inference + partition (partition is fast; deberta-v3-large is small)
INFPART_FILE="sbatch_infpart_${STEM}"
INFPART_OUT="result_infpart_${STEM}.txt"
{
    make_header "nbip_$(($RANDOM % 100000))" "$INFPART_OUT" "1" "48G" "1:00:00"
    cat << EOL

python src/inference.py \\
  --mode transformers \\
  --model "${MODEL}" \\
  --data "${SPLIT}" \\
  --eval-dir "\${EVAL_DIR}" \\
  --sampling ${SAMPLING} \\
  --num-generations ${NUM_GEN}

python src/partition.py \\
  --eval-dir "\${EVAL_DIR}" \\
  --alg ${PART_ALG}
EOL
} > "$INFPART_FILE"
chmod +x "$INFPART_FILE"

# Stage 2: score + summarize. 4 GPUs for Skywork-Reward-Gemma-2-27B (~54GB bf16,
# sharded via device_map="auto"). Mem bumped to 96G for safe CPU-side loading.
SCORE_FILE="sbatch_score_${STEM}"
SCORE_OUT="result_score_${STEM}.txt"
{
    make_header "nbs_$(($RANDOM % 100000))" "$SCORE_OUT" "4" "96G" "1:00:00"
    cat << EOL

python src/score.py \\
  --eval-dir "\${EVAL_DIR}" \\
  --patience ${PATIENCE}

python src/summarize.py --eval-dir "\${EVAL_DIR}"
EOL
} > "$SCORE_FILE"
chmod +x "$SCORE_FILE"

echo "Created sbatch files:"
echo "  1. ${INFPART_FILE}"
echo "  2. ${SCORE_FILE}"
echo ""
echo "Outputs will be written to:"
echo "  1. ${INFPART_OUT}"
echo "  2. ${SCORE_OUT}"
echo ""
echo "Suggested submission (chained with dependencies):"
echo "  JID1=\$(sbatch --parsable ${INFPART_FILE})"
echo "  JID2=\$(sbatch --parsable --dependency=afterok:\${JID1} ${SCORE_FILE})"
echo "Or submit each one individually as the previous stage completes."
