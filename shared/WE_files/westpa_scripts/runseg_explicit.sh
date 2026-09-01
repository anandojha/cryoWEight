#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

ln -sv $WEST_SIM_ROOT/common_files/bstate.pdb .
ln -sv $WEST_SIM_ROOT/common_files/seg_config.json .
ln -sv $WEST_SIM_ROOT/common_files/build_system.py .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/production.py > production.py
  ln -sv $WEST_PARENT_DATA_REF/seg.xml ./parent.xml
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/production.py > production.py
  ln -sv $WEST_PARENT_DATA_REF ./parent.xml
fi

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

# Calculate pcoord with MDAnalysis
python $WEST_SIM_ROOT/common_files/cv.py
cat cv.dat > $WEST_PCOORD_RETURN

# Clean up
rm -f cv.dat production.py
