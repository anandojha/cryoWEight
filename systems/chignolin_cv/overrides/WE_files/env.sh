#!/bin/bash
# CPU-friendly env for the local CV demo: no GPU daemon, few workers.
export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename "$WEST_SIM_ROOT")
export WM_WORK_MANAGER=processes
export WM_N_WORKERS=${WM_N_WORKERS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHON=$(which python)
