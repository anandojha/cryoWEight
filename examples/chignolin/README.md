# chignolin example

A folded prior steered onto unfolded cryo-EM images over 3 iterations of 2 weighted
ensemble subiterations each. Fully self contained, the seeding trajectory ships with the
example. The chignolin benchmark scaled down:

| | full system | example |
|---|---|---|
| WE subiterations | 30 | 2 |
| cryo-EM target `image.dcd` | 28,000 frames | 1,000 frames |
| seeding MD | 500 frames shipped | 200 frames shipped |
| `n_pixel` | 128 | 64 |
| `N_draw` | 20,000 | 100 |
| segment length tau | 20 ps | 10 ps |

## Run

```bash
bash run.sh
```

or step by step from the repository root:

```bash
python assemble.py --system examples/chignolin --dest run_chig
cd run_chig
python ../cryoWEight.py --system examples/chignolin --local init
python ../cryoWEight.py --system examples/chignolin --local iterate --range 2 3
```

To regenerate the seeding trajectory instead of using the shipped one, place the
RCSB structure `1UAO.pdb` in `init_MD/` and delete `init_MD/chignolin.dcd` before
`run.sh`. Every setting is in `config.xml`, including the structures, the solvent model, the
imaging parameters and the convergence block. Set `platform` to `CPU` on a laptop or
`CUDA` on a GPU node. The free energy landscapes appear in `run*/merged_WE/fes.png`.
