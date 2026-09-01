#!/bin/bash
#SBATCH --job-name=cryoWEight_demo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=cryoWEight_demo-%j.log
#
# SLURM submit for the cryoWEight CV-reweight demo on a Flatiron GPU node. Runs the
# whole iterative loop (seed -> reweight -> w_init/w_run -> merge -> reweight ...),
# not a single w_run, so it wraps run.sh rather than calling w_run directly.
#
#   sbatch submit.sh                 # defaults: we_run, 10 runs x 10 WE iterations
#   sbatch submit.sh we_run 5        # 5 runs into ./we_run/
#   N_WORKERS=8 sbatch submit.sh     # override worker count (default = #GPUs, 1 worker/GPU)
#
# Positional args pass straight through to run.sh ([WORK_DIR] [N_ITERATIONS]).

module load cuda

# The env lives on ceph so GPU nodes can see it, but importing openmm/mdtraj/westpa off a
# network FS costs 3-30 s -- and the demo spawns a fresh python PER segment, so runs were
# ~10x slower. Stage from a single prebuilt tarball onto node-local disk (imports -> ~0.4 s):
# copying one big file + untarring on local disk is fast; per-file ceph copies are not.
# Build the tarball once with:  tar -cf $ENV_TAR -C <env-parent-dir> cryoweight_demo
ENV_TAR=/mnt/ceph/users/aojha/cryoweight_demo.tar
SRC_PREFIX=/home/aojha/miniforge3/envs/cryoweight_demo   # prefix baked into the tarball shebangs
if [ -f "$ENV_TAR" ]; then
  LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/cwdemo.${SLURM_JOB_ID:-$$}"
  mkdir -p "$LOCAL_ROOT"
  echo "staging env: $ENV_TAR -> $LOCAL_ROOT"
  cp "$ENV_TAR" "$LOCAL_ROOT/env.tar" && tar -xf "$LOCAL_ROOT/env.tar" -C "$LOCAL_ROOT" && rm -f "$LOCAL_ROOT/env.tar"
  LOCAL_ENV="$LOCAL_ROOT/cryoweight_demo"
  grep -rIl "$SRC_PREFIX" "$LOCAL_ENV/bin" 2>/dev/null | xargs -r sed -i "s|$SRC_PREFIX|$LOCAL_ENV|g"
  export PATH="$LOCAL_ENV/bin:$PATH"
  # OpenMM's baked plugin dir is the build path (/usr/local/openmm/...), which is absent
  # here, so point it at the staged copy or the CUDA/CPU platforms won't register.
  export OPENMM_PLUGIN_DIR="$LOCAL_ENV/lib/plugins"
  trap 'rm -rf "$LOCAL_ROOT"' EXIT
else
  echo "WARN: $ENV_TAR not found; activating ceph env directly (slower per-segment imports)"
  source /mnt/ceph/users/aojha/miniforge3/etc/profile.d/conda.sh
  conda activate cryoweight_demo
fi
python -c "import mdtraj, openmm, westpa" 2>/dev/null \
  || { echo "ERROR: env not usable (python=$(which python))" >&2; exit 1; }

cd "$SLURM_SUBMIT_DIR"
# Workers spread over the GPUs; runseg.sh pins each to CUDA_VISIBLE_DEVICES =
# WM_PROCESS_INDEX % NGPUS. Default 4 workers/GPU keeps both CPUs and GPUs saturated
# (the per-segment cost is partly CPU, so 1 worker/GPU leaves the GPUs idle).
export NGPUS="${SLURM_GPUS_ON_NODE:-4}"
WEST_PLATFORM=CUDA N_WORKERS="${N_WORKERS:-$((NGPUS * 4))}" bash run.sh "$@"
