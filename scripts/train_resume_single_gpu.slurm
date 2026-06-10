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
### Allow the job to be requeued, and fire SIGUSR1 to the batch step 120s before
### walltime so we can checkpoint-resume across the 6h limit (see AUTO_REQUEUE below).
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#
#SBATCH --job-name=genup_train
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

# Add cluster-specific SBATCH directives above if needed, for example:
# #SBATCH --partition=gpu
# #SBATCH --account=my_account

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Slurm does not create the log/ dir for its own output files, but create it here
# too so anything the script writes (and a manual run) has somewhere to go.
mkdir -p log

init_modules() {
    if ! command -v module >/dev/null 2>&1; then
        if [[ -f /etc/profile.d/modules.sh ]]; then
            # shellcheck disable=SC1091
            source /etc/profile.d/modules.sh
        elif [[ -f /usr/share/Modules/init/bash ]]; then
            # shellcheck disable=SC1091
            source /usr/share/Modules/init/bash
        fi
    fi

    module purge
    module load gcc/14.2.0
    module load cuda/12.8.0
    module load miniconda3/25.5.1
    module load slurm/slurm/24.11
}

init_conda() {
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
}

activate_conda_env() {
    if [[ -n "$CONDA_ENV_PREFIX" ]]; then
        conda activate "$CONDA_ENV_PREFIX"
    else
        conda activate "$CONDA_ENV_NAME"
    fi
}

CONDA_ENV_NAME="${CONDA_ENV_NAME:-genup}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"

init_modules
init_conda
activate_conda_env

# Help the 8B weights load without spiking CPU RAM (carried over from the original
# train_qlora_fsdp.sh recipe). FSDP itself is configured via TrainingArguments.
export FSDP_CPU_RAM_EFFICIENT_LOADING="${FSDP_CPU_RAM_EFFICIENT_LOADING:-1}"

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-meta-llama/Meta-Llama-3.1-8B}"
DATASET_ID="${DATASET_ID:-w11wo/FourSquare-TKY-POI}"
OUTPUT_DIR="${OUTPUT_DIR:-$(basename "$MODEL_CHECKPOINT")-$(basename "$DATASET_ID")}"
# Written by train_sft_qlora_fsdp.py once all epochs finish; presence means there is
# nothing left to train. Run evaluation separately via scripts/eval_next_poi.slurm.
COMPLETION_SENTINEL="$OUTPUT_DIR/.training_complete"

MAX_LENGTH="${MAX_LENGTH:-16384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
# Keep SAVE_STEPS low enough that at least one checkpoint-* is written well within
# the 6h walltime, otherwise a walltime kill discards all progress and every
# resubmission restarts from scratch. Tune up only after timing a real step.
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
# 1 CPU is allocated (--cpus-per-task=1); keep dataloader workers small to avoid
# oversubscribing it.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
# When 1, trap the pre-walltime SIGUSR1 and requeue this job so it resumes from the
# latest checkpoint. Set to 0 to disable and resubmit manually instead.
AUTO_REQUEUE="${AUTO_REQUEUE:-1}"

run_training() {
    torchrun --nproc_per_node=1 src/train_sft_qlora_fsdp.py \
        --model_checkpoint "$MODEL_CHECKPOINT" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --warmup_steps "$WARMUP_STEPS" \
        --num_epochs "$NUM_EPOCHS" \
        --gradient_checkpointing \
        --apply_liger_kernel_to_llama \
        --resume_from_checkpoint \
        --save_steps "$SAVE_STEPS" \
        --save_total_limit "$SAVE_TOTAL_LIMIT" \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_id "$DATASET_ID"
}

if [[ -f "$COMPLETION_SENTINEL" ]]; then
    echo "Training already complete (found $COMPLETION_SENTINEL); skipping."
    echo "Run scripts/eval_next_poi.slurm to evaluate $OUTPUT_DIR."
    exit 0
fi

if [[ "$AUTO_REQUEUE" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
    # On SIGUSR1 (sent 120s before walltime), requeue so the next run resumes from
    # the latest checkpoint. Run training in the background and `wait` so the trap
    # can interrupt it.
    requeue() {
        echo "[requeue] walltime approaching; requeueing job $SLURM_JOB_ID"
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    }
    trap requeue USR1

    run_training &
    TRAIN_PID=$!
    # `wait` returns when training finishes or when the trap fires.
    wait "$TRAIN_PID"
else
    run_training
fi

echo "Training finished. Run scripts/eval_next_poi.slurm to evaluate $OUTPUT_DIR."
