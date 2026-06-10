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
#SBATCH --job-name=genup_env_prep
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

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

init_modules
init_conda

conda_env_exists() {
    if [[ -n "$CONDA_ENV_PREFIX" ]]; then
        [[ -d "$CONDA_ENV_PREFIX" ]]
    else
        conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"
    fi
}

create_conda_env() {
    if [[ -n "$CONDA_ENV_PREFIX" ]]; then
        conda create -y -p "$CONDA_ENV_PREFIX" "python=$CONDA_PYTHON_VERSION"
    else
        conda create -y -n "$CONDA_ENV_NAME" "python=$CONDA_PYTHON_VERSION"
    fi
}

activate_conda_env() {
    if [[ -n "$CONDA_ENV_PREFIX" ]]; then
        conda activate "$CONDA_ENV_PREFIX"
    else
        conda activate "$CONDA_ENV_NAME"
    fi
}

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
    if ! conda_env_exists; then
        create_conda_env
    fi
    activate_conda_env
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    activate_conda_env
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
