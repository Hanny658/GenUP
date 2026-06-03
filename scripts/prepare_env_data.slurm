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
#SBATCH --job-name=llm4poi_env_prep
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

# Add cluster-specific SBATCH directives above if needed, for example:
# #SBATCH --partition=cpu
# #SBATCH --account=my_account

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p log

# Load your HPC environment here if needed, for example:
# module load python/3.10

DATASET="${DATASET:-tky}"
INPUT_PATH="${INPUT_PATH:-./LLM4POI}"
OUTPUT_PATH="${OUTPUT_PATH:-./data}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_PROFILE_GENERATION="${RUN_PROFILE_GENERATION:-1}"
RUN_POI_REASONING="${RUN_POI_REASONING:-0}"
RUN_CREATE_SFT_DATASET="${RUN_CREATE_SFT_DATASET:-1}"

HF_NAMESPACE="${HF_NAMESPACE:-}"
PROFILE_DATASET_ID="${PROFILE_DATASET_ID:-}"
SFT_DATASET_ID="${SFT_DATASET_ID:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"

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

if [[ "$RUN_INSTALL" == "1" ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

if [[ "$RUN_PROFILE_GENERATION" == "1" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "OPENAI_API_KEY is required for profile generation." >&2
        exit 1
    fi
    if [[ -z "$PROFILE_DATASET_ID" ]]; then
        echo "Set HF_NAMESPACE or PROFILE_DATASET_ID before profile generation." >&2
        exit 1
    fi
    python src/generate_user_profile.py \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH" \
        --dataset "$DATASET" \
        --dataset_id "$PROFILE_DATASET_ID" \
        --num_workers "$NUM_WORKERS"
fi

if [[ "$RUN_POI_REASONING" == "1" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "OPENAI_API_KEY is required for POI reasoning generation." >&2
        exit 1
    fi
    if [[ -z "$SFT_DATASET_ID" ]]; then
        echo "Set HF_NAMESPACE or SFT_DATASET_ID before POI reasoning generation." >&2
        exit 1
    fi
    python src/generate_poi_reasoning.py \
        --dataset_id "$SFT_DATASET_ID" \
        --output_path "$OUTPUT_PATH" \
        --dataset "$DATASET" \
        --num_workers "$NUM_WORKERS"
fi

if [[ "$RUN_CREATE_SFT_DATASET" == "1" ]]; then
    if [[ -z "$SFT_DATASET_ID" ]]; then
        echo "Set HF_NAMESPACE or SFT_DATASET_ID before SFT dataset creation." >&2
        exit 1
    fi
    python src/create_sft_dataset.py \
        --input_path "$INPUT_PATH" \
        --profiles_path "$OUTPUT_PATH" \
        --dataset "$DATASET" \
        --dataset_id "$SFT_DATASET_ID"
fi
