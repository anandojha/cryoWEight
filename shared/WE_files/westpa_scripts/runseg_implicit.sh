#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

ln -sv $WEST_SIM_ROOT/common_files/seg_config.json .
ln -sv $WEST_SIM_ROOT/common_files/build_system.py .
# bstate.pdb, folded.pdb and parent.xml are not staged here. This script returns
# them through WEST_RESTART_RETURN, and the WESTPA HDF5 restart framework unpacks
# that archive into each segment directory before this script runs.
sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/production.py > production.py

# Assign each worker its GPU
: "${WM_PROCESS_INDEX:=0}"
if command -v westraj &>/dev/null; then
  eval "$(westraj map_worker)"
else
  NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$NGPUS" -gt 0 ]; then
    GPU_IDX=$(( WM_PROCESS_INDEX % NGPUS ))
  else
    GPU_IDX=""
  fi
fi
# Restrict this process to its GPU
export CUDA_VISIBLE_DEVICES=$GPU_IDX
# Run the dynamics with OpenMM
python production.py

python $WEST_SIM_ROOT/common_files/cv.py
cat cv.dat > $WEST_PCOORD_RETURN
cp $WEST_SIM_ROOT/common_files/bstate.pdb $WEST_TRAJECTORY_RETURN
cp $WEST_SIM_ROOT/common_files/folded.pdb $WEST_TRAJECTORY_RETURN
cp seg.dcd $WEST_TRAJECTORY_RETURN
cp $WEST_SIM_ROOT/common_files/bstate.pdb $WEST_RESTART_RETURN
cp $WEST_SIM_ROOT/common_files/folded.pdb $WEST_RESTART_RETURN
cp seg.xml $WEST_RESTART_RETURN/parent.xml
cp seg.log $WEST_LOG_RETURN

# Clean up
rm -f cv.dat production.py
