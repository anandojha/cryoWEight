#!/bin/bash
# CPU-friendly WESTPA environment for the local CV-reweight demo: no GPU daemon,
# few workers. WEST_SIM_ROOT is the per-run directory (run1/, run2/, ...).
export WEST_SIM_ROOT="$PWD"
export SIM_NAME=$(basename "$WEST_SIM_ROOT")
export WM_WORK_MANAGER=${WM_WORK_MANAGER:-processes}
export WM_N_WORKERS=${WM_N_WORKERS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHON=$(which python)

# Platform for the per-segment OpenMM propagator (production.py reads it from
# seg_config.json; runseg.sh patches the json with $WEST_PLATFORM if set).
# Override with: WEST_PLATFORM=CUDA (default CPU, keeps the demo dependency-light).
export WEST_PLATFORM=${WEST_PLATFORM:-CPU}
