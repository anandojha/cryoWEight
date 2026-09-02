#!/bin/bash
# Run the CV-signal WESTPA replica (implicit chignolin) locally, no slurm/ssh.
# bash run_cv.sh [WORKDIR] [ITERS]
# ITERS = number of outer reweighting iterations (init = run1, then iterate 2..ITERS).
# Each WESTPA run does seg_config/west.cfg sub-iterations (small by default). Adjust
# size in systems/chignolin_cv.xml: n_iterations (WE sub-iters), n_steps_per_segment.
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:-cv_run}"
ITERS="${2:-3}"
rm -rf "$WORK"
python "$REPO/assemble.py" --system chignolin_cv --dest "$WORK"   # data/ scripts/ init_MD/ WE_files/
cd "$WORK"
run() { PYTHONPATH="$REPO" python "$REPO/cryoWEight.py" --system chignolin_cv --local --n-workers 4 "$@"; }
run init
[ "$ITERS" -ge 2 ] && run iterate --range 2 "$ITERS"
echo "done -> WESTPA runs in $WORK/run{1..$ITERS}/ ; reweighted bstates in $WORK/reweight_run*/"
