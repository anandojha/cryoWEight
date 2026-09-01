# trpcage example

The bring your own system check. Trp cage (PDB 1L2Y) was added with `new_system.py` and
a dozen config edits, nothing else. A folded prior is steered onto unfolded target
frames (CA RMSD above 4 A) over 3 iterations of 2 weighted ensemble subiterations, in
OBC2 implicit solvent. `sigma_sign` is `auto`, so the bottleneck direction is resolved
from the config and the seeding ensemble instead of being set by hand.

## Run

```bash
bash run.sh
```

The script assembles the run tree, generates the 1 ns seeding trajectory, and runs the
three iterations. `make_target_pool.py` rebuilds `data/image.dcd` from a hot unfolding
trajectory, standing in for the reference ensemble a real application would provide.
