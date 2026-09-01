# chignolin_cv — exact-replica WESTPA pipeline driven by the CV signal

Same iterative methodology as production cryoWEight, with two changes: the reweighting
uses the **CV distribution** (RMSD+Rg) instead of cryo-EM images, and the solvent is
**implicit** (OBC2, friction 1) so it runs on a laptop/CPU. One full flow:

```
seeding MD ensemble (init_MD/)  ->  reweight to CV target (reweight_run0)
  ->  bstates + weights  ->  w_init  ->  w_run  (WESTPA run 1)
  ->  merge  ->  reweight (reweight_run1)  ->  w_init/w_run (run 2)  ->  ...
```
The reweighted weight is the WESTPA walker's starting weight (`bstates.txt` + `w_init`);
its `bstate.xml` is the walker's starting coordinates. Only `reweight.py` changed
(`‖cv_struct − cv_target‖²` likelihood instead of synthetic images).

## Run locally (no slurm)

```bash
bash run_cv.sh cv_run 3        # workdir cv_run, init + iterate 2..3   (CPU)
```
Outputs land in `cv_run/`: `run1..3/` (WESTPA), `reweight_run*/` (bstates), and each
run's `merged_WE/`.

Knobs (`systems/chignolin_cv.xml`):
- `n_iterations` — WESTPA sub-iterations per run (default 3).
- `seg_config.n_steps_per_segment` — MD steps per segment (2500 = 5 ps).
- `cv_sigma` (reweight_config) — CV bandwidth; `cv_target_rmsd_lo/hi` — target basin.
- `seg_config.platform` — `CPU` (set `CUDA` on a GPU node).

## Run on a GPU cluster

Set `seg_config.platform: CUDA`, drop the CPU `env.sh`/`runseg` overrides (use the GPU
defaults), then drive without `--local` (sbatch via `ssh_host`):
```bash
python cryoWEight.py --system chignolin_cv init
python cryoWEight.py --system chignolin_cv iterate --range 2 N
```

## Status

The CV reweight + bstate building is validated locally (produces `bstates.txt`,
`bstate.xml`, `pcoord.init`, `bottleneck_coordinates.txt`). The WESTPA `w_run` step has
not been run here — bring it up in stages and report errors:
1. one `run_cv.sh cv_run 1` (a single WESTPA run from CV-reweighted bstates),
2. then `cv_run 3` (the full loop).
