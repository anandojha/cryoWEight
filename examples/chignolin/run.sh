#!/bin/bash
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${1:-run}"
SYS="examples/chignolin"

cd "$(dirname "$0")"
python "$REPO/assemble.py" --system "$SYS" --dest "$WORK"
cd "$WORK"

DCD=$(python -c 'import json; print(json.load(open("scripts/reweight_config.json"))["init_md_dcd"])')
if [ ! -f "init_MD/$DCD" ]; then
  cd init_MD
  cp ../scripts/reweight_config.json .
  python simulation.py
  rm reweight_config.json
  cd ..
fi

python "$REPO/cryoWEight.py" --system "$SYS" --local init
python "$REPO/cryoWEight.py" --system "$SYS" --local iterate --range 2 3
echo "done. free energy landscapes in run*/merged_WE/fes.png"
