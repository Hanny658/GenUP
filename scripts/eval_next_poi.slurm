#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#
### Specify Memory allocate to this job ###
#SBATCH --mem=48G
#
### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=1
#
### Specify number of node to compute ###
#SBATCH --nodes=1
#
### Optional: Specify node to execute the job ###
#SBATCH --nodelist=TC2N07
#
### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ###
#SBATCH --time=06:00:00
#
#SBATCH --job-name=llm4poi_eval
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

# Add cluster-specific SBATCH directives above if needed, for example:
# #SBATCH --partition=gpu
# #SBATCH --account=my_account

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p log

# Load your HPC environment here if needed, for example:
# module load cuda/12.1
# source .venv/bin/activate

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-meta-llama/Meta-Llama-3.1-8B}"
DATASET_ID="${DATASET_ID:-w11wo/FourSquare-TKY-POI}"
# Defaults to the training script's output dir so you can submit this job with the
# same env vars used for training. Override OUTPUT_DIR to evaluate a different run.
OUTPUT_DIR="${OUTPUT_DIR:-$(basename "$MODEL_CHECKPOINT")-$(basename "$DATASET_ID")}"
EVAL_DATASET_ID="${EVAL_DATASET_ID:-$DATASET_ID}"

EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.65}"
EVAL_TOP_K="${EVAL_TOP_K:-50}"
EVAL_TOP_P="${EVAL_TOP_P:-0.92}"
EVAL_TYPICAL_P="${EVAL_TYPICAL_P:-0.95}"
EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.0}"
EVAL_USE_QUANTIZATION="${EVAL_USE_QUANTIZATION:-0}"

if [[ ! -f "$OUTPUT_DIR/.training_complete" ]]; then
    echo "Warning: $OUTPUT_DIR/.training_complete not found; the model in" >&2
    echo "$OUTPUT_DIR may be a mid-training checkpoint. Continuing anyway." >&2
fi

mkdir -p results

eval_quantization_args=()
if [[ "$EVAL_USE_QUANTIZATION" == "1" ]]; then
    eval_quantization_args+=(--use_quantization)
fi

accelerate launch --num_processes=1 src/eval_next_poi.py \
    --model_checkpoint "$OUTPUT_DIR" \
    --dataset_id "$EVAL_DATASET_ID" \
    --apply_liger_kernel_to_llama \
    --temperature "$EVAL_TEMPERATURE" \
    --top_k "$EVAL_TOP_K" \
    --top_p "$EVAL_TOP_P" \
    --typical_p "$EVAL_TYPICAL_P" \
    --repetition_penalty "$EVAL_REPETITION_PENALTY" \
    "${eval_quantization_args[@]}"
