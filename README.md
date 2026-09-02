# cryoWEight

[![CI](https://github.com/anandojha/cryoWEight/actions/workflows/ci.yml/badge.svg)](https://github.com/anandojha/cryoWEight/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anandojha/cryoWEight/graph/badge.svg?token=zMOvXWY6Kb)](https://codecov.io/gh/anandojha/cryoWEight)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated and extensible Python framework that couples iterative weighted ensemble
simulations with ensemble reweighting against cryo-EM particle images. Reweighting in
the image space introduces a directional bias towards the target distribution even when
the prior ensemble has no overlap with the target distribution in the collective
variable space.

## Get started

**Step 1: Install**

```bash
git clone https://github.com/anandojha/cryoWEight.git
cd cryoWEight
bash install.sh
conda activate cryoweight
```

The installer detects the operating system, loads the cuda module on clusters that have
one, and installs the CPU stack on a Mac.

**Step 2: Run the tests**

```bash
python tests/test_cryoweight.py
```

**Step 3: Run the examples**

```bash
bash examples/chignolin/run.sh
```

```bash
bash examples/trpcage/run.sh
```

```bash
bash examples/ntl9/run.sh
```

```bash
bash examples/adk/run.sh
```

Each script assembles the run tree, generates the seeding trajectory when one is not
shipped, runs the iterations, and writes the free energy landscapes to
`run*/merged_WE/fes.png` inside the example. A GPU is used when present and the CPU
platform otherwise. Run adk on a GPU machine, the other three run anywhere.

## Configure

Everything is in the system `config.xml`. Start from the config of the example closest to your system and edit. Outer rounds come from the command line, `iterate --range 2 N` for a fixed count or `iterate --until-converged` for the KL stopping rule.

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
