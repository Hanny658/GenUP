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
#SBATCH --job-name=eval_genup
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

# -E so the ERR trap (installed in lib_env.sh) also fires inside functions.
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p log

if [[ ! -f scripts/lib_env.sh ]]; then
    echo "[error] scripts/lib_env.sh not found; submit from the repo root: sbatch scripts/eval_next_poi.slurm" >&2
    exit 1
fi
# shellcheck source=scripts/lib_env.sh
source scripts/lib_env.sh

CONDA_ENV_NAME="${CONDA_ENV_NAME:-genup}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
# Must match what prepare_env_data.slurm used so we find the env + share the HF
# weight cache off $HOME.
GENUP_CACHE_ROOT="${GENUP_CACHE_ROOT:-}"

setup_caches
init_modules
init_conda
# Fails loudly with "run prepare_env_data.slurm first" if the env was never built.
require_conda_env

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
    log_warn "$OUTPUT_DIR/.training_complete not found; the model in $OUTPUT_DIR may be a mid-training checkpoint. Continuing anyway."
fi

mkdir -p results

eval_quantization_args=()
if [[ "$EVAL_USE_QUANTIZATION" == "1" ]]; then
    eval_quantization_args+=(--use_quantization)
fi

log_stage "evaluating $OUTPUT_DIR on $EVAL_DATASET_ID"
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
log_done "evaluation finished for $OUTPUT_DIR"
