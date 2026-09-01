#!/bin/bash
set -e

echo "Installing mamba into base environment..."
conda install -n base -c conda-forge mamba --yes

echo "Removing existing cryoweight environment if it exists..."
conda env remove -n cryoweight --yes 2>/dev/null || true

echo "Creating cryoweight environment..."
mamba env create -f environment.yml --yes

echo "Pinning OpenMM to CUDA 12.1 for compatibility..."
mamba install -n cryoweight -c conda-forge openmm=8.1 cuda-version=12.1 --yes

echo "Installing WESTPA..."
mamba install -n cryoweight -c conda-forge westpa --yes

echo "Verifying installation..."
conda run -n cryoweight python -c "
import numpy
import scipy
import matplotlib
import mdtraj
import tqdm
import MDAnalysis
import openmm
import sklearn
import pandas
import torch
import yaml
import h5py
import westpa
print('All packages verified successfully')
"

echo "cryoweight environment installed successfully! Activate with: conda activate cryoweight"
