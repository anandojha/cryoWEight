#!/bin/bash
# Initialize a WESTPA run from the CV-reweighted basis states (bstates/bstates.txt).
# w_init reads bstates.txt + each bstates/{i:04d}/{bstate.xml,pcoord.init} and calls
# get_pcoord.sh on every basis state.

# Set up simulation environment
source env.sh

# Clean up from previous/failed runs
rm -rf traj_segs seg_logs istates west.h5
mkdir -p seg_logs traj_segs istates

# Pointer to the basis-state file produced by reweight.py
BSTATE_ARGS="--bstate-file $WEST_SIM_ROOT/bstates/bstates.txt"

# Run w_init (threads work-manager is robust for the lightweight init pass)
w_init $BSTATE_ARGS --segs-per-state 4 --work-manager=threads "$@"
