#!/bin/bash
# Self-contained cryoWEight-protocol driver for the implicit chignolin CV demo.
#
#   seeding MD -> reweight (run0) -> [ w_init / w_run -> merge -> reweight ] x N
#
# but the reweight targets a CV distribution ([CA-RMSD, Rg], unfolded basin of
# data/image.dcd) instead of cryo-EM images. Each iteration n produces a run{n}/
# folder containing west.h5, traj_segs/, bstates/ and merged_WE/, and seeds
# run{n+1}/bstates from the merged WE ensemble.
#
# WESTPA runs with cryoWEight "adaptive binning": a RecursiveBinMapper whose coarse
# RectilinearBinMapper base grid is refined inside one bin by an MABBinMapper
# (bottleneck: true). The MAB `at:` bottleneck coordinate is recomputed each
# iteration (reweight.py compute_bottleneck -> run{n+1}/bottleneck.txt) and sed-
# patched into the per-run west.cfg copy by stage_run.
#
# Usage:
#   bash run.sh [WORK_DIR] [N_ITERATIONS]
#   bash run.sh we_run 4
#
# Options via environment:
#   WEST_PLATFORM=CPU|CUDA   OpenMM platform for production + reweight (default CPU)
#   N_WORKERS=2              w_run workers (default 2)
#   K_BSTATES=8              basis states drawn per reweight (default 8)
#   SEED_STEPS=5000          seeding-MD steps (default 5000)
#   MAB_AT="5.0 6.0"         run1 MAB bottleneck at: default (RMSD Rg); later runs
#                            read the bottleneck.txt their prior reweight wrote
#   PYTHON=python            python interpreter (default: python on PATH)
#
# Dependency-light: numpy, scipy, mdtraj, openmm, westpa only (see README for the
# explicit list of excluded heavy ML / analysis packages).
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${1:-we_run}"
N_ITER="${2:-10}"

WEST_FILES="$HERE/westpa_files"
COMMON="$WEST_FILES/common_files"
DATA="$WEST_FILES/data"
SEG_CONFIG="$COMMON/seg_config.json"

export WEST_PLATFORM="${WEST_PLATFORM:-CPU}"
N_WORKERS="${N_WORKERS:-2}"
K_BSTATES="${K_BSTATES:-32}"
BIN_WIDTH="${BIN_WIDTH:-0.5}"
SEED_STEPS="${SEED_STEPS:-5000}"
PCOORD_LEN="${PCOORD_LEN:-3}"
CV_SIGMA="${CV_SIGMA:-0.6}"
CV_LO="${CV_LO:-4.0}"
CV_HI="${CV_HI:-6.0}"
IMAGE="${IMAGE:-image.dcd}"            # target dcd in data/ (e.g. image_4_6.dcd); frames in [CV_LO,CV_HI] are the target
# Default MAB bottleneck `at:` [RMSD Rg] for run1 (the target unfolded-basin center).
# run n>=2 instead reads the bottleneck.txt written by run{n-1}'s reweight.
MAB_AT="${MAB_AT:-5.0 6.0}"
MAX_ITERS="${MAX_ITERS:-}"             # WE iterations per run; empty = west.cfg default
START="${START:-1}"                    # first round; >1 continues from run{START-1}/merged_WE
PYTHON="${PYTHON:-python}"

echo "=== cryoWEight CV-reweight demo ==="
echo "work dir   : $WORK_DIR"
echo "iterations : $N_ITER"
echo "platform   : $WEST_PLATFORM"
echo "max bstates: $K_BSTATES  (bin width $BIN_WIDTH A)"
echo "python     : $($PYTHON -c 'import sys; print(sys.executable)')"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
ROOT="$PWD"

# --------------------------------------------------------------------------- #
# Step 0: seeding MD from bstate.pdb -> init/seed.dcd                          #
# --------------------------------------------------------------------------- #
if [ ! -s init/seed.dcd ]; then
  echo "--- step 0: seeding MD (${SEED_STEPS} implicit steps from bstate.pdb) ---"
  "$PYTHON" "$HERE/seed.py" \
    --pdb "$COMMON/bstate.pdb" \
    --seg-config "$SEG_CONFIG" \
    --out "$ROOT/init/seed.dcd" \
    --n-steps "$SEED_STEPS" \
    --platform "$WEST_PLATFORM"
else
  echo "--- step 0: reusing existing init/seed.dcd ---"
fi

# --------------------------------------------------------------------------- #
# Helper: stage a run{n} directory with the WESTPA files                       #
#                                                                              #
# Each run gets its OWN west.cfg (a COPY, not a symlink) with the MAB `at:`     #
# bottleneck coordinate patched to this run's value, so the adaptive bin is     #
# recentered every cryoWEight iteration. Everything else stays a read-only      #
# symlink into westpa_files/.                                                   #
#   stage_run <rundir> "<RMSD> <Rg>"                                            #
# --------------------------------------------------------------------------- #
stage_run () {
  local rundir="$1"
  local bottleneck="$2"          # "RMSD Rg" for the MAB at: line
  mkdir -p "$rundir"
  # Per-run west.cfg copy with the patched bottleneck `at:` coordinate.
  local rmsd rg
  read -r rmsd rg <<< "$bottleneck"
  cp "$WEST_FILES/west.cfg" "$rundir/west.cfg"
  # Replace the single MAB `at:` line (indentation-tolerant) with this run's value.
  sed -i.bak -E "s/^([[:space:]]*)at:.*$/\1at: [${rmsd}, ${rg}]/" "$rundir/west.cfg"
  [ -n "$MAX_ITERS" ] && sed -i.bak -E "s/^([[:space:]]*)max_total_iterations:.*$/\1max_total_iterations: ${MAX_ITERS}/" "$rundir/west.cfg"
  rm -f "$rundir/west.cfg.bak"
  echo "    west.cfg MAB at: [${rmsd}, ${rg}]"
  # Link the remaining shared WESTPA pieces (read-only).
  ln -sf "$WEST_FILES/env.sh"          "$rundir/env.sh"
  ln -sf "$WEST_FILES/init.sh"         "$rundir/init.sh"
  ln -sf "$WEST_FILES/run.sh"          "$rundir/run.sh"
  ln -sfn "$COMMON"                    "$rundir/common_files"
  ln -sfn "$WEST_FILES/westpa_scripts" "$rundir/westpa_scripts"
}

# --------------------------------------------------------------------------- #
# Step 1: reweight (run0) seeding ensemble -> run1/bstates                     #
# --------------------------------------------------------------------------- #
# Round START's bstates come from reweighting the prior ensemble: the seeding ensemble
# for round 1, or run{START-1}'s merged WE ensemble when continuing (START > 1).
if [ "$START" -le 1 ]; then
  SRC_ENS="$ROOT/init/seed.dcd"
  echo "--- step 1: reweight run0 (seeding ensemble) -> run1/bstates ---"
else
  SRC_ENS="$ROOT/run$(( START - 1 ))/merged_WE/traj_all.dcd"
  echo "--- step 1: continue -> reweight run$(( START - 1 )) merged ensemble -> run${START}/bstates ---"
fi
"$PYTHON" "$HERE/reweight.py" \
  --ensemble "$SRC_ENS" \
  --topology "$DATA/topology.pdb" \
  --folded "$COMMON/folded.pdb" \
  --image "$DATA/$IMAGE" \
  --seg-config "$SEG_CONFIG" \
  --out "$ROOT/run${START}/bstates" \
  --bottleneck-out "$ROOT/run${START}/bottleneck.txt" \
  --k "$K_BSTATES" --bin-width "$BIN_WIDTH" --sigma "$CV_SIGMA" \
  --cv-target-lo "$CV_LO" --cv-target-hi "$CV_HI" \
  --pcoord-len "$PCOORD_LEN" --platform "$WEST_PLATFORM" --seed 0

# --------------------------------------------------------------------------- #
# Step 2: iterate w_init / w_run -> merge -> reweight                          #
# --------------------------------------------------------------------------- #
# Pick this run's MAB bottleneck `at:` [RMSD Rg]: the bottleneck.txt the previous
# reweight wrote into run{n}/ (run0 -> run1, run{n-1} -> run{n}), else the MAB_AT
# default for run1.
read_bottleneck () {
  local rundir="$1"
  if [ -s "$rundir/bottleneck.txt" ]; then
    awk 'NR==1{print $1, $2}' "$rundir/bottleneck.txt"
  else
    echo "$MAB_AT"
  fi
}

for (( n=START; n<=N_ITER; n++ )); do
  rundir="$ROOT/run${n}"
  echo ""
  echo "=================== iteration $n / $N_ITER (run${n}) ==================="
  bottleneck="$(read_bottleneck "$rundir")"
  stage_run "$rundir" "$bottleneck"

  # bstates were produced by the previous reweight into run${n}/bstates.
  if [ ! -s "$rundir/bstates/bstates.txt" ]; then
    echo "ERROR: $rundir/bstates/bstates.txt missing; reweight did not run." >&2
    exit 1
  fi

  pushd "$rundir" >/dev/null

  echo "--- w_init (run${n}) ---"
  WM_N_WORKERS="$N_WORKERS" bash init.sh

  echo "--- w_run (run${n}, local, ${N_WORKERS} workers) ---"
  source env.sh
  WEST_PLATFORM="$WEST_PLATFORM" w_run --n-workers "$N_WORKERS" \
    --work-manager processes &> west.log || { tail -40 west.log; exit 1; }
  echo "w_run done; west.h5 + traj_segs/ written."

  echo "--- merge segments -> run${n}/merged_WE ---"
  "$PYTHON" "$HERE/merge.py" \
    --traj-dir "$rundir/traj_segs" \
    --topology "$DATA/topology.pdb" \
    --out-dir "$rundir/merged_WE" \
    --out traj_all.dcd

  popd >/dev/null

  # Reweight the merged WE ensemble -> next run's bstates (unless this is the last).
  if [ "$n" -lt "$N_ITER" ]; then
    next=$(( n + 1 ))
    echo "--- reweight merged run${n} ensemble -> run${next}/bstates ---"
    # Also recomputes the bottleneck from run{n}'s merged WE ensemble and writes it to
    # run{next}/bottleneck.txt, which run{next}'s stage_run seds into its west.cfg at:.
    "$PYTHON" "$HERE/reweight.py" \
      --ensemble "$rundir/merged_WE/traj_all.dcd" \
      --topology "$DATA/topology.pdb" \
      --folded "$COMMON/folded.pdb" \
      --image "$DATA/$IMAGE" \
      --seg-config "$SEG_CONFIG" \
      --out "$ROOT/run${next}/bstates" \
      --bottleneck-out "$ROOT/run${next}/bottleneck.txt" \
      --k "$K_BSTATES" --bin-width "$BIN_WIDTH" --sigma "$CV_SIGMA" \
      --cv-target-lo "$CV_LO" --cv-target-hi "$CV_HI" \
      --pcoord-len "$PCOORD_LEN" --platform "$WEST_PLATFORM" --seed "$n"
  fi
done

echo ""
echo "=== done. run1..run${N_ITER} under $ROOT ==="
echo "each run{n}/ has: west.h5, traj_segs/, bstates/, merged_WE/traj_all.dcd"
