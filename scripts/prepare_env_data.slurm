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
#SBATCH --job-name=env_prep_genup
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

# NOTE: this job only installs packages and calls the OpenAI / Hugging Face APIs;
# it does not need a GPU. If `conda create` / `pip install` fail here, the most
# common reason is that this compute node has no outbound internet -- in that
# case build the env on a login node first (see README) or pick a node with
# network access.

# -E so the ERR trap (installed in lib_env.sh) also fires inside functions.
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p log

# Load shared logging + conda/env helpers. Submit from the repo root so this
# relative path resolves (Slurm runs a spooled copy of this script, so we cannot
# source relative to the script's own location).
if [[ ! -f scripts/lib_env.sh ]]; then
    echo "[error] scripts/lib_env.sh not found; submit from the repo root: sbatch scripts/prepare_env_data.slurm" >&2
    exit 1
fi
# shellcheck source=scripts/lib_env.sh
source scripts/lib_env.sh

DATASET="${DATASET:-tky}"
INPUT_PATH="${INPUT_PATH:-./LLM4POI}"
OUTPUT_PATH="${OUTPUT_PATH:-./data}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genup}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.10}"

RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_PROFILE_GENERATION="${RUN_PROFILE_GENERATION:-1}"
RUN_POI_REASONING="${RUN_POI_REASONING:-0}"
RUN_CREATE_SFT_DATASET="${RUN_CREATE_SFT_DATASET:-1}"

HF_NAMESPACE="${HF_NAMESPACE:-}"
PROFILE_DATASET_ID="${PROFILE_DATASET_ID:-}"
SFT_DATASET_ID="${SFT_DATASET_ID:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"

log_info "job ${SLURM_JOB_ID:-local} on $(hostname) | dataset=$DATASET | install=$RUN_INSTALL profile=$RUN_PROFILE_GENERATION poi=$RUN_POI_REASONING sft=$RUN_CREATE_SFT_DATASET"

init_modules
init_conda

dataset_base_name() {
    case "$1" in
        nyc) echo "FourSquare-NYC" ;;
        ca) echo "Gowalla-CA" ;;
        tky) echo "FourSquare-TKY" ;;
        moscow) echo "FourSquare-Moscow" ;;
        sao_paulo) echo "FourSquare-SaoPaulo" ;;
        *) echo "Unsupported DATASET='$1'. Expected one of: nyc, ca, tky, moscow, sao_paulo." >&2; exit 1 ;;
    esac
}

DATASET_BASE_NAME="$(dataset_base_name "$DATASET")"

if [[ -n "$HF_NAMESPACE" ]]; then
    PROFILE_DATASET_ID="${PROFILE_DATASET_ID:-$HF_NAMESPACE/${DATASET_BASE_NAME}-User-Profiles}"
    SFT_DATASET_ID="${SFT_DATASET_ID:-$HF_NAMESPACE/${DATASET_BASE_NAME}-POI}"
fi

# ---------------------------------------------------------------------------
# Stage: install dependencies
# ---------------------------------------------------------------------------
if [[ "$RUN_INSTALL" == "1" ]]; then
    warn_if_no_internet
    ensure_conda_env
    activate_conda_env
    verify_conda_env
    log_stage "installing python dependencies"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    log_done "dependencies installed"
else
    require_conda_env
fi

# ---------------------------------------------------------------------------
# Stage: generate user profiles (OpenAI)
# ---------------------------------------------------------------------------
if [[ "$RUN_PROFILE_GENERATION" == "1" ]]; then
    log_stage "generating user profiles (dataset=$DATASET)"
    [[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is required for profile generation."
    [[ -n "$PROFILE_DATASET_ID" ]] || die "Set HF_NAMESPACE or PROFILE_DATASET_ID before profile generation."
    train_csv="$INPUT_PATH/$DATASET/preprocessed/train_sample.csv"
    [[ -f "$train_csv" ]] || die "input data not found: $train_csv (set INPUT_PATH or download the LLM4POI dataset)."
    require_hf_auth
    python src/generate_user_profile.py \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --dataset "$DATASET" \
        --dataset_id "$PROFILE_DATASET_ID" \
        --num_workers "$NUM_WORKERS"
    log_done "user profiles generated -> $PROFILE_DATASET_ID"
fi

# ---------------------------------------------------------------------------
# Stage: generate POI reasoning (OpenAI)
# ---------------------------------------------------------------------------
if [[ "$RUN_POI_REASONING" == "1" ]]; then
    log_stage "generating POI reasoning (dataset=$DATASET)"
    [[ -n "${OPENAI_API_KEY:-}" ]] || die "OPENAI_API_KEY is required for POI reasoning generation."
    [[ -n "$SFT_DATASET_ID" ]] || die "Set HF_NAMESPACE or SFT_DATASET_ID before POI reasoning generation."
    require_hf_auth
    python src/generate_poi_reasoning.py \
        --dataset_id "$SFT_DATASET_ID" \
        --output_path "$OUTPUT_PATH" \
        --dataset "$DATASET" \
        --num_workers "$NUM_WORKERS"
    log_done "POI reasoning generated"
fi

# ---------------------------------------------------------------------------
# Stage: build supervised fine-tuning dataset
# ---------------------------------------------------------------------------
if [[ "$RUN_CREATE_SFT_DATASET" == "1" ]]; then
    log_stage "creating SFT dataset (dataset=$DATASET)"
    [[ -n "$SFT_DATASET_ID" ]] || die "Set HF_NAMESPACE or SFT_DATASET_ID before SFT dataset creation."
    require_hf_auth
    python src/create_sft_dataset.py \
        --input_path "$INPUT_PATH" \
        --profiles_path "$OUTPUT_PATH" \
        --dataset "$DATASET" \
        --dataset_id "$SFT_DATASET_ID"
    log_done "SFT dataset created -> $SFT_DATASET_ID"
fi

log_done "all requested env-prep stages complete"
