# cryoWEight

[![CI](https://github.com/anandojha/cryoWEight/actions/workflows/ci.yml/badge.svg)](https://github.com/anandojha/cryoWEight/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anandojha/cryoWEight/graph/badge.svg?token=zMOvXWY6Kb)](https://codecov.io/gh/anandojha/cryoWEight)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Iterative weighted ensemble simulations reweighted against cryo-EM particle images,
steering the ensemble towards the conformational populations in the images. The
likelihood acts in image space, so states with no overlap in collective variable space
are still reached.

## Install

```bash
bash install.sh
conda activate cryoweight
```

The installer detects the operating system, loads the cuda module on clusters that have
one, and installs the CPU stack on a Mac. `mamba env create -f environment.yml` followed
by `pip install -e .` does the same by hand.

## Run the examples

Step 1, get the code and install.

```bash
git clone https://github.com/anandojha/cryoWEight.git
cd cryoWEight
bash install.sh
conda activate cryoweight
```

Step 2, run the tests.

```bash
cd cryoWEight
python tests/test_cryoweight.py
```

Step 3, run any example from inside the repository.

```bash
cd cryoWEight
bash examples/chignolin/run.sh
```

```bash
cd cryoWEight
bash examples/trpcage/run.sh
```

```bash
cd cryoWEight
bash examples/ntl9/run.sh
```

```bash
cd cryoWEight
bash examples/adk/run.sh
```

Each script assembles the run tree, generates the 1 ns seeding trajectory when one is
not shipped, runs 3 iterations of 2 weighted ensemble subiterations, and writes the
free energy landscapes to `run*/merged_WE/fes.png` inside the example. A GPU is used
when present and the CPU platform otherwise. Run adk on a GPU machine, the other three
run anywhere.

## Configure

Everything is in the system `config.xml`.

| Setting | Keys |
|---|---|
| System | `input_pdb`, `topology_we`, `topology_stripped`, `cv_reference_pdb` |
| Solvent | `solvent_model`, `solvent_dielectric`, `solute_dielectric` |
| MD | `ff_main`, `ff_solvent`, `temperature_K`, `timestep_ps`, `friction_per_ps`, `platform` |
| Seeding MD | `n_steps`, `dcd_report_interval` |
| Weighted ensemble | `n_iterations`, `n_steps_per_segment`, `walkers_per_bin`, `mab_at`, `mab_nbins`, `sigma_sign` |
| Collective variables | `cv_family`, `cv_atom_selection` |
| Imaging | `n_pixel`, `pixel_size`, `snr`, `sigma`, `add_ctf`, `N_draw` |
| Target selection | `select_mode`, `x_lower`, `x_upper`, `x_thresh`, `y_lower`, `y_upper` |
| Convergence | `kl_threshold`, `max_iterations` |

Outer iterations come from the command line, `iterate --range 2 N` for a fixed count or
`iterate --until-converged` for the KL stopping rule.

## Layout

```
cryoWEight.py    driver, --system <name> or --system examples/<name>
assemble.py      builds a run tree from shared/ + templates/ + the system directory
new_system.py    scaffolds a systems/<name>.xml from input structures
cryoweight/      importable package
examples/        <name>/{config.xml,run.sh,data,init_MD}
shared/          files identical across systems
templates/       <rel>.tmpl rendered from config
systems/         <name>.xml and <name>/{overrides,data,init_MD}
tests/           unit tests and one end to end run
```
