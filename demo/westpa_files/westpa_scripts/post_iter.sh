#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
    set -x
    env | sort
fi

cd $WEST_SIM_ROOT || exit 1

# Format iteration number
ITER=$(printf "%06d" $WEST_CURRENT_ITER)
TAR=$(($WEST_CURRENT_ITER-3)) # Keep last n iterations and archive the rest
TAR_DIR=$(printf "%06d" $TAR)

# Archive and remove old segment logs
tar -cf seg_logs/$ITER.tar seg_logs/$ITER-*.log
rm -f seg_logs/$ITER-*.log

# Archive and remove older trajectory segments (except the last n iterations)
if [ -d traj_segs/$TAR_DIR ]; then
  tar -cf traj_segs/$TAR_DIR.tar traj_segs/$TAR_DIR
  rm -rf traj_segs/$TAR_DIR
fi
