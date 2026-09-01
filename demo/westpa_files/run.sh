#!/bin/bash
# Launch the WESTPA weighted-ensemble run (w_run) for the current run directory.
# Driven by the top-level demo/run.sh loop; can also be invoked by hand inside a
# run{n}/ folder after init.sh. Extra args (e.g. --n-workers 2) pass through to w_run.

# Make sure environment is set
source env.sh

# Clean up
rm -f west.log

# Run w_run locally with the processes work-manager
w_run --work-manager processes "$@" &> west.log
