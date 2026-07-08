#!/bin/bash
# =============================================================================
# CoWERA — PBS submission script (single node, GPU)
# =============================================================================
# Usage:
#   qsub -v CONFIG=./Systems/chignolin/config.yml,REPO_ROOT=$PWD Scripts/sample_job_multirun.sh
#   qsub -v CONFIG=...,REPO_ROOT=$PWD,RESTART=1 Scripts/sample_job_multirun.sh   # resume
#
# The repository root must be passed as REPO_ROOT (run_cowera.py resolves
# Scripts/ and Systems/ relative to the working directory).
# -----------------------------------------------------------------------------
#PBS -N CoWERA
#PBS -q gpu2
#PBS -j oe
#PBS -l nodes=1:ppn=16
#PBS -l walltime=48:00:00
set -euo pipefail

# ----------------------------------------------------------------------------
# Site modules (adjust for your cluster)
# ----------------------------------------------------------------------------
module load compilers/parallel_studio_2019.5.075 || true
source /apps/intel_2019update5/intelpython3/bin/mpivars.sh || true
module load compilers/gcc/8.4.0 || true

# ----------------------------------------------------------------------------
# Environment + working directory
# ----------------------------------------------------------------------------
: "${CONFIG:?Set CONFIG=<path to config.yml> via qsub -v CONFIG=...}"
REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:?set REPO_ROOT to the CoWERA repo root}}"
RESTART="${RESTART:-0}"

# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cowera                          # matches env/environment.yml (was: wepy)

export OMP_NUM_THREADS="${PBS_NP:-1}"
export MKL_NUM_THREADS="${PBS_NP:-1}"

# ----------------------------------------------------------------------------
# CUDA MPS — start it, and ALWAYS stop it on exit (normal or scheduler kill)
# ----------------------------------------------------------------------------
stop_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }
trap stop_mps EXIT

nvidia-cuda-mps-control -d
# Tune to GPU memory / number of co-resident walkers (20 => ~5 walkers/GPU).
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20

# ----------------------------------------------------------------------------
# Launch from the repository root (run_cowera.py depends on relative paths)
# ----------------------------------------------------------------------------
cd "$REPO_ROOT"

RESTART_FLAG=""
[[ "$RESTART" == "1" ]] && RESTART_FLAG="--restart"

python ./Scripts/run_cowera.py --config "$CONFIG" $RESTART_FLAG
