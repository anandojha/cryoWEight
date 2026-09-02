#!/bin/bash

# Base WESTPA settings
export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename "$WEST_SIM_ROOT")

# Work manager and parallelism 
export WM_WORK_MANAGER=processes        # or “zmq” if that route is selected
export WM_N_WORKERS=32                  # Total number of segment propagators on this node
export OMP_NUM_THREADS=4                # Must match the SLURM --cpus-per-task

# NUMA & CPU binding 
export NUMA_AFFINITY_ENABLED=true
export NUMACTL=$(which numactl)

# Python for map_worker
export PYTHON=$(which python)           # so runseg.sh can use “$PYTHON -m westraj.cli.map_worker”

# NVIDIA MPS (optional but recommended) 
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  if ! pgrep -x nvidia-cuda-mps-control >/dev/null; then
    nvidia-cuda-mps-control -d
  fi
fi

# Tuning knobs for multi‐process GPU sharing
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MPS_CLIENT_PRIORITY=0
