#!/bin/bash

set -x
cat $WEST_STRUCT_DATA_REF/pcoord.init > $WEST_PCOORD_RETURN
cp $WEST_SIM_ROOT/common_files/folded.pdb $WEST_TRAJECTORY_RETURN
cp $WEST_SIM_ROOT/common_files/bstate.pdb $WEST_TRAJECTORY_RETURN
cp $WEST_STRUCT_DATA_REF/bstate.xml $WEST_TRAJECTORY_RETURN
cp $WEST_SIM_ROOT/common_files/bstate.pdb $WEST_RESTART_RETURN
cp $WEST_SIM_ROOT/common_files/folded.pdb $WEST_RESTART_RETURN
cp $WEST_STRUCT_DATA_REF/bstate.xml $WEST_RESTART_RETURN/parent.xml
