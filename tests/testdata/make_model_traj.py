"""Generate the model chignolin trajectory the end to end test runs on.

    python make_model_traj.py <repo root> <out dir>

The pipeline builds an OpenMM system from every basis state, so the model data has to be a
topology that parametrises and geometry that minimises. A twelve atom CA chain satisfies
neither. This runs a short implicit solvent trajectory from the chignolin structure already
committed to the repo, at raised temperature so the frames spread in RMSD and radius of
gyration rather than sitting on top of each other. It is generated data, not production
data, and it is small enough to commit.
"""
import os
import sys

import numpy as np
import mdtraj as md
from openmm import LangevinMiddleIntegrator, Platform
from openmm.app import PDBFile, ForceField, Simulation, DCDReporter, HBonds, NoCutoff
from openmm.unit import kelvin, picosecond, picoseconds, nanometer

REPO, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
START = os.path.join(REPO, "systems", "chignolin_cv", "overrides", "WE_files",
                     "common_files", "folded.pdb")
N_FRAMES = 40
STEPS_PER_FRAME = 500
TEMPERATURE = 450.0     # above folding, so a short run still spreads the ensemble
SEED = 20260829

os.makedirs(OUT, exist_ok=True)
pdb = PDBFile(START)
forcefield = ForceField("amber14-all.xml", "implicit/obc2.xml")
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                                 constraints=HBonds)
integrator = LangevinMiddleIntegrator(TEMPERATURE * kelvin, 1 / picosecond,
                                      0.002 * picoseconds)
integrator.setRandomNumberSeed(SEED)
simulation = Simulation(pdb.topology, system, integrator,
                        Platform.getPlatformByName("CPU"))
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(TEMPERATURE * kelvin, SEED)
dcd = os.path.join(OUT, "model_chignolin.dcd")
simulation.reporters.append(DCDReporter(dcd, STEPS_PER_FRAME))
simulation.step(N_FRAMES * STEPS_PER_FRAME)
del simulation

top = os.path.join(OUT, "model_chignolin.pdb")
PDBFile.writeFile(pdb.topology, pdb.positions, open(top, "w"))

reference = md.load(top)
traj = md.load(dcd, top=top)
ca = reference.topology.select("name CA")
rmsd = md.rmsd(traj, reference, atom_indices=ca) * 10
rg = md.compute_rg(traj.atom_slice(ca)) * 10
print(f"  model_chignolin.pdb  {os.path.getsize(top):>8d} bytes  {reference.n_atoms} atoms")
print(f"  model_chignolin.dcd  {os.path.getsize(dcd):>8d} bytes  {traj.n_frames} frames")
print(f"  RMSD {rmsd.min():5.2f} to {rmsd.max():5.2f} A")
print(f"  Rg   {rg.min():5.2f} to {rg.max():5.2f} A")
# The target for the CV likelihood is the most expanded tail of the same run.
cut = float(np.percentile(rmsd, 70))
sel = traj[rmsd >= cut]
sel.save_dcd(os.path.join(OUT, "model_target.dcd"))
print(f"  model_target.dcd     {os.path.getsize(os.path.join(OUT, 'model_target.dcd')):>8d} bytes  "
      f"{sel.n_frames} frames, RMSD >= {cut:.2f} A")
