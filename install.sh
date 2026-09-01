#!/bin/bash
set -e

OS=$(uname -s)
ARCH=$(uname -m)
echo "Installing for $OS $ARCH"

if [ "$OS" = "Linux" ] && command -v module >/dev/null 2>&1; then
  module load cuda 2>/dev/null && echo "Loaded the cuda module" || echo "No cuda module, continuing"
fi

if ! command -v mamba >/dev/null 2>&1; then
  echo "Installing mamba into the base environment"
  conda install -n base -c conda-forge mamba --yes
fi

echo "Removing an existing cryoweight environment if present"
conda env remove -n cryoweight --yes 2>/dev/null || true

echo "Creating the cryoweight environment"
mamba env create -f environment.yml --yes

if [ "$OS" = "Linux" ]; then
  echo "Adding the CUDA build of OpenMM"
  mamba install -n cryoweight -c conda-forge "cuda-version>=12.0" --yes || echo "CUDA packages unavailable, the CPU platform still works"
fi

echo "Installing the package"
conda run -n cryoweight pip install -e . --quiet

echo "Verifying the installation"
conda run -n cryoweight python - <<'PY'
import numpy, scipy, matplotlib, mdtraj, tqdm, MDAnalysis, sklearn, pandas, torch, yaml, h5py
import openmm, westpa
from openmm import Platform
names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print("OpenMM", openmm.version.version, "platforms", names)
print("All packages verified")
PY

echo "Done. Activate with: conda activate cryoweight"
if [ "$OS" = "Darwin" ]; then
  echo "This machine has no CUDA. The example run scripts detect that and use the CPU platform."
fi
