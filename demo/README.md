# CV-reweight WESTPA demo (implicit chignolin)

Self-contained, dependency-light cryoWEight. Same loop

```
seeding MD -> reweight (run0) -> [ w_init/w_run -> merge -> reweight ] x N
```

but the reweight targets a CV distribution ([RMSD, Rg] from the unfolded basin of
`data/image.dcd`) instead of cryo-EM images. WE runs use adaptive binning
(`RecursiveBinMapper` + `MABBinMapper`, `at:` bottleneck recomputed each iteration).

## Install

```bash
mamba env create -f environment.yml      # cryoweight_demo: numpy scipy mdtraj openmm westpa (no torch)
conda activate cryoweight_demo
```

## Run

```bash
bash run.sh we_run 1                                  # shakedown: 1 run, checks WESTPA wiring
bash run.sh we_run 10                                 # 10 runs into ./we_run/  (args: WORK_DIR N_RUNS)
WEST_PLATFORM=CUDA N_WORKERS=4 bash run.sh we_run 10  # on a GPU
sbatch submit.sh we_run 10                            # SLURM (Flatiron GPU node)
python plot.py                                  # RMSD ridgelines per round -> plot.png
```

Env overrides:

| Variable | Default | Meaning |
|---|---|---|
| `WEST_PLATFORM` | `CPU` | OpenMM platform (`CPU`/`CUDA`) |
| `N_WORKERS` | `2` | `w_run` workers |
| `K_BSTATES` | `32` | max basis states (occupied-bin cap) |
| `BIN_WIDTH` | `0.5` | CV clustering bin width (A) |
| `SEED_STEPS` | `5000` | seeding-MD steps |
| `CV_SIGMA` | `0.6` | EM bandwidth (A) |
| `CV_LO`/`CV_HI` | `4`/`6` | target RMSD band (A) |
| `MAB_AT` | `5.0 6.0` | run1 MAB `at:`; later runs read `bottleneck.txt` |

`max_total_iterations` (WE iterations per run, default 10) is set in `westpa_files/west.cfg`.

## Output

```
we_run/
  init/seed.dcd                  seeding ensemble (run0 input)
  run{1..N}/
    west.cfg                     per-run copy, MAB at: patched to this run's bottleneck
    bstates/                     bstate.xml + pcoord.init + bstates.txt (reweighted starts)
    west.h5  traj_segs/          WESTPA run + per-segment seg.dcd
    merged_WE/traj_all.dcd       merged ensemble -> next reweight input
    bottleneck.txt               recomputed MAB at: for the next run
```

## Files

```
run.sh           driver loop
reweight.py      CV-reweight + bstate builder (numpy EM; cluster ensemble -> per-bin representatives)
seed.py          seeding MD
merge.py         merge traj_segs/*/*/seg.dcd -> one DCD
cv_reweight.py   numpy EM (same as production)
westpa_files/    west.cfg (Recursive+MAB), env/init/run, westpa_scripts/, common_files/, data/
```
