#!/bin/bash
# Implicit-solvent segment. production.py picks CPU or CUDA from $WEST_PLATFORM; for
# CUDA, each WESTPA worker is pinned to its own GPU (round-robin over the visible ones).

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

# Pin this worker to a GPU when running on CUDA (WESTPA sets WM_PROCESS_INDEX per worker).
if [ "${WEST_PLATFORM:-CPU}" = "CUDA" ]; then
  : "${WM_PROCESS_INDEX:=0}"
  NGPUS=${NGPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}
  [ "${NGPUS:-0}" -ge 1 ] 2>/dev/null || NGPUS=1
  export CUDA_VISIBLE_DEVICES=$(( WM_PROCESS_INDEX % NGPUS ))
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/production.py > production.py

# Stage the files production.py / cv.py read from the segment working dir
# (the implicit runseg previously staged none of these, so every segment crashed).
ln -sf $WEST_SIM_ROOT/common_files/seg_config.json .
ln -sf $WEST_SIM_ROOT/common_files/build_system.py .
ln -sf $WEST_SIM_ROOT/common_files/bstate.pdb .
ln -sf $WEST_SIM_ROOT/common_files/folded.pdb .

# Parent restart -> parent.xml. Initial segments: WEST_PARENT_DATA_REF is the istate,
# a symlink to bstates/{id}/ holding bstate.xml; continuation segs: the parent seg.xml.
P="$WEST_PARENT_DATA_REF"
if   [ -f "$P/parent.xml" ]; then cp "$P/parent.xml" parent.xml
elif [ -f "$P/seg.xml"    ]; then cp "$P/seg.xml"    parent.xml
elif [ -f "$P/bstate.xml" ]; then cp "$P/bstate.xml" parent.xml
elif [ -f "$P"            ]; then cp "$P"            parent.xml
fi

# Run the dynamics with OpenMM (CPU platform)
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
