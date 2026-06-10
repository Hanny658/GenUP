# shellcheck shell=bash
# Shared bootstrap + logging helpers for GenUP Slurm jobs.
#
# Source this from a Slurm script AFTER `set -Eeuo pipefail` and AFTER cd-ing to
# the repo root, e.g.:
#
#     set -Eeuo pipefail
#     cd "${SLURM_SUBMIT_DIR:-$PWD}"
#     source scripts/lib_env.sh
#
# Why this file exists: conda and pip write their diagnostics to STDOUT, so when
# `conda create` / `pip install` fails on a node without internet the Slurm .err
# file is EMPTY and the job still leaves the queue -- it looks like a clean run
# even though nothing was built. The helpers below make every stage announce
# itself and make every failure print a located message to STDERR.

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
_ts() { date '+%Y-%m-%d %H:%M:%S'; }

log_stage() { echo "[$(_ts)] [stage] $*"; }
log_done()  { echo "[$(_ts)] [done]  $*"; }
log_info()  { echo "[$(_ts)] [info]  $*"; }
log_warn()  { echo "[$(_ts)] [warn]  $*" >&2; }
# die: print to STDERR and exit non-zero (so it lands in the .err log).
die()       { echo "[$(_ts)] [error] $*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Fail-loud error trap
# --------------------------------------------------------------------------
# Fires on any unhandled non-zero command (commands inside if/while/||/&&/!
# conditions are exempt, as with `set -e`). Requires `set -E` in the caller so
# it also fires for failures inside functions. Prints the failing line and
# command to STDERR, then exits with the original code.
_genup_err_trap() {
    local rc=$?
    echo "[$(_ts)] [error] command failed (exit $rc) at line ${BASH_LINENO[0]:-?}: ${BASH_COMMAND}" >&2
    exit "$rc"
}
trap _genup_err_trap ERR

# --------------------------------------------------------------------------
# Environment bootstrap
# --------------------------------------------------------------------------
init_modules() {
    log_stage "loading environment modules"
    if ! command -v module >/dev/null 2>&1; then
        if [[ -f /etc/profile.d/modules.sh ]]; then
            # shellcheck disable=SC1091
            source /etc/profile.d/modules.sh
        elif [[ -f /usr/share/Modules/init/bash ]]; then
            # shellcheck disable=SC1091
            source /usr/share/Modules/init/bash
        fi
    fi
    command -v module >/dev/null 2>&1 || die "'module' command not available; cannot load HPC modules."

    module purge
    module load gcc/14.2.0
    module load cuda/12.8.0
    module load miniconda3/25.5.1
    module load slurm/slurm/24.11
    log_done "modules loaded"
}

init_conda() {
    log_stage "initialising conda"
    command -v conda >/dev/null 2>&1 || die "conda not found after 'module load miniconda3'."
    local conda_base
    conda_base="$(conda info --base)"
    # conda's activation scripts reference unbound variables (PS1, _CE_CONDA, ...);
    # relax nounset while sourcing so `set -u` does not abort us here.
    set +u
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
    set -u
    log_done "conda initialised (base: $conda_base)"
}

# True if the configured env (by prefix, else by name) exists.
conda_env_exists() {
    if [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
        [[ -d "$CONDA_ENV_PREFIX" ]]
    else
        conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"
    fi
}

activate_conda_env() {
    # conda activate also trips `set -u`; relax it around the call.
    set +u
    if [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
        conda activate "$CONDA_ENV_PREFIX"
    else
        conda activate "$CONDA_ENV_NAME"
    fi
    set -u
}

# Confirm the activated env has a usable python (catches half-built envs).
verify_conda_env() {
    local py
    py="$(command -v python || true)"
    [[ -n "$py" ]] || die "no 'python' on PATH after activating the env."
    log_info "active env: ${CONDA_DEFAULT_ENV:-?} | python: $($py --version 2>&1) ($py)"
}

# Create the env if missing, then VERIFY it really exists. `conda create` can
# fail (no internet, or -- common on HPC -- a full $HOME quota) yet still leave a
# confusing partial state; this turns that into a hard, explained failure.
ensure_conda_env() {
    local target="${CONDA_ENV_PREFIX:-$CONDA_ENV_NAME}"
    if conda_env_exists; then
        log_info "conda env '$target' already exists"
        return 0
    fi
    log_stage "creating conda env '$target' (python=$CONDA_PYTHON_VERSION)"
    # `|| ok=0` keeps conda's own non-zero exit from tripping the ERR trap so we
    # can print a single, actionable message instead of a noisy double trace.
    local ok=1
    if [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
        conda create -y -p "$CONDA_ENV_PREFIX" "python=$CONDA_PYTHON_VERSION" || ok=0
    else
        conda create -y -n "$CONDA_ENV_NAME" "python=$CONDA_PYTHON_VERSION" || ok=0
    fi
    if [[ "$ok" != "1" ]] || ! conda_env_exists; then
        die "conda create failed for '$target'. Common HPC causes (see the conda traceback above):
  * Disk quota exceeded on \$HOME -- conda writes envs+packages under ~/.conda. Free space
    ('conda clean -a -y' then 'rm -rf ~/.conda/pkgs/*'), or set GENUP_CACHE_ROOT=/path/on/scratch
    to put the env and all caches on a roomy filesystem, then rerun.
  * No outbound internet on this compute node -- build the env on a login node instead."
    fi
    log_done "conda env '$target' created"
}

# Require an already-built env (used by train/eval). Gives an actionable message
# instead of the raw 'EnvironmentNameNotFound'.
require_conda_env() {
    local target="${CONDA_ENV_PREFIX:-$CONDA_ENV_NAME}"
    conda_env_exists || die "conda env '$target' not found. Run scripts/prepare_env_data.slurm \
first and confirm its log shows '[done]  conda env ... created' (check the .out file, not just .err)."
    log_stage "activating conda env '$target'"
    activate_conda_env
    verify_conda_env
    log_done "conda env '$target' active"
}

# Redirect conda envs + package cache, pip cache, Hugging Face cache, and TMPDIR
# off the (usually small, quota-limited) home directory onto a roomy filesystem.
# Set GENUP_CACHE_ROOT=/path/on/scratch and this points everything there; leave
# it unset for the original behaviour. Call BEFORE init_conda / ensure_conda_env
# / require_conda_env, and set it consistently for prep AND train/eval so they
# all agree on where the env lives (exporting it in ~/.bashrc is easiest).
setup_caches() {
    local root="${GENUP_CACHE_ROOT:-}"
    [[ -z "$root" ]] && return 0
    mkdir -p "$root/conda_pkgs" "$root/pip" "$root/hf" "$root/tmp" "$root/envs"
    export CONDA_PKGS_DIRS="$root/conda_pkgs"     # where conda downloads/extracts + caches repodata
    export PIP_CACHE_DIR="$root/pip"
    export HF_HOME="$root/hf"                      # umbrella for hub/datasets/transformers caches
    export HF_DATASETS_CACHE="$root/hf/datasets"
    export HUGGINGFACE_HUB_CACHE="$root/hf/hub"
    export TMPDIR="$root/tmp"
    # Put the env under the cache root unless the caller pinned one explicitly.
    if [[ -z "${CONDA_ENV_PREFIX:-}" ]]; then
        export CONDA_ENV_PREFIX="$root/envs/${CONDA_ENV_NAME:-genup}"
    fi
    log_info "GENUP_CACHE_ROOT=$root -> env=$CONDA_ENV_PREFIX; conda/pip/HF/TMP caches redirected off \$HOME"
}

# Non-fatal connectivity probe so a no-internet node is flagged BEFORE a long,
# confusing failure. Only warns; never aborts.
warn_if_no_internet() {
    command -v curl >/dev/null 2>&1 || return 0
    if ! curl -sSf --max-time 10 -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
        log_warn "no outbound HTTPS to pypi.org from this node. conda create / pip install \
will likely fail. Build the env on a login node, or schedule on a node with internet."
    fi
}

# Confirm Hugging Face credentials exist (datasets are pushed via push_to_hub).
# Filesystem/env check only -- no network call -- so it works offline and fails
# BEFORE any paid OpenAI calls rather than after.
require_hf_auth() {
    [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && return 0
    [[ -f "$HOME/.cache/huggingface/token" || -f "$HOME/.huggingface/token" ]] && return 0
    die "No Hugging Face credentials found (scripts push_to_hub). Run 'huggingface-cli login' \
or export HF_TOKEN before profile/SFT generation."
}
