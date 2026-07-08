#!/bin/bash
# =============================================================================
# CoWERA — SLURM submission script (single node, GPU or CPU)
# =============================================================================
# Submit ONE run:
#     sbatch --export=ALL,CONFIG=./Systems/chignolin/config.yml Scripts/slurm/cowera.sbatch
#
# Submit 5 independent replicas as a job array (recommended for statistics —
# each array task gets a distinct run id `fold_${SLURM_ARRAY_TASK_ID}`):
#     sbatch --array=0-4 --export=ALL,CONFIG=./Systems/chignolin/config.yml Scripts/slurm/cowera.sbatch
#
# Resume a run that hit the wall-time (reuses the same output dir, appends):
#     sbatch --export=ALL,CONFIG=./Systems/chignolin/config.yml,RESTART=1 Scripts/slurm/cowera.sbatch
# -----------------------------------------------------------------------------
#SBATCH --job-name=cowera
#SBATCH --nodes=1                 # CoWERA's work mapper is single-node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # CPU cores for OpenMM CPU platform + joblib analysis
#SBATCH --gres=gpu:1             # GPUs to allocate; set to 0 (or delete) for CPU runs
#SBATCH --time=24:00:00          # realistic wall-time; use --restart to continue
#SBATCH --output=cowera_%x_%A_%a.out
set -euo pipefail

# ----------------------------------------------------------------------------
# 0. Inputs (override on the command line via --export)
# ----------------------------------------------------------------------------
: "${CONFIG:?Set CONFIG=<path to config.yml> via --export=ALL,CONFIG=...}"
RESTART="${RESTART:-0}"                       # 1 to resume from checkpoint
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"                               # run_cowera.py needs the repo root as CWD

# ----------------------------------------------------------------------------
# 1. Environment
# ----------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cowera                         # matches env/environment.yml

# Keep BLAS/OpenMP thread pools inside the SLURM allocation (avoids oversubscription
# when the per-walker processes and joblib analysis run together).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ----------------------------------------------------------------------------
# 2. CUDA MPS (only when GPUs are allocated) — start it and ALWAYS stop it.
# ----------------------------------------------------------------------------
HAVE_GPU=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${SLURM_GPUS_ON_NODE:-0}" != "0" ]]; then
    HAVE_GPU=1
fi

stop_mps() {
    if [[ "$HAVE_GPU" == "1" ]]; then
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    fi
}
trap stop_mps EXIT                            # tear down MPS on normal exit or scheduler kill

if [[ "$HAVE_GPU" == "1" ]]; then
    # Pack multiple walkers per physical GPU. Tune the % to GPU memory / walker count
    # (e.g. 20 => ~5 co-resident walkers). Repeat gpu_ids in config.yml accordingly.
    export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/nvidia-mps-$SLURM_JOB_ID"
    export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/nvidia-mps-log-$SLURM_JOB_ID"
    nvidia-cuda-mps-control -d
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20
fi

# ----------------------------------------------------------------------------
# 3. Replica id for job arrays (distinct 'run' per array task)
# ----------------------------------------------------------------------------
RUN_SUFFIX=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_SUFFIX="_${SLURM_ARRAY_TASK_ID}"
    echo "Array task ${SLURM_ARRAY_TASK_ID}: appending run suffix ${RUN_SUFFIX}"
    # NOTE: set a distinct 'run:' per replica. Either template the config so
    # 'run' ends with ${SLURM_ARRAY_TASK_ID}, or pass per-replica config files.
fi

# ----------------------------------------------------------------------------
# 4. Launch
# ----------------------------------------------------------------------------
# gpu_ids in the config must index the SLURM-VISIBLE devices (CUDA_VISIBLE_DEVICES
# is renumbered 0..N-1 by SLURM), NOT physical device ids. For node-local scratch,
# set `scratch_dir: "$TMPDIR"` in the config (trajectories stage to fast local disk
# and are synced back to the output directory on exit).
RESTART_FLAG=""
[[ "$RESTART" == "1" ]] && RESTART_FLAG="--restart"

echo "Launching CoWERA: config=$CONFIG restart=$RESTART gpu=$HAVE_GPU cpus=${SLURM_CPUS_PER_TASK:-1}"
srun python ./Scripts/run_cowera.py --config "$CONFIG" $RESTART_FLAG
