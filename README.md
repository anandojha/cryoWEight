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
mamba env create -f environment.yml
conda activate cryoweight
pip install -e .
```

## Run an example

```bash
bash examples/chignolin/run.sh
```

Three examples ship under `examples/`, each 3 iterations of 2 subiterations. chignolin is
self contained, ntl9 and adk generate their 1 ns seeding MD first. Output lands in
`run*/merged_WE/fes.png`.

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

## Validate

```bash
python tests/test_cryoweight.py
python tests/test_end_to_end.py
python tests/test_three_systems.py
```
