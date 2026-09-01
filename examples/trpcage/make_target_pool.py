"""Build the unfolded target pool for the trpcage example.

One nanosecond of hot implicit solvent MD melts the miniprotein, and the frames with
CA RMSD above the selection threshold form data/image.dcd. A real application would
use frames from an experimental reference ensemble instead.
"""

import os

import mdtraj as md
from openmm import LangevinIntegrator, Platform, unit
from openmm.app import DCDReporter, ForceField, PDBFile, Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.join(HERE, "data", "trpcage.pdb")

pdb = PDBFile(TOP)
ff = ForceField("amber14-all.xml", "implicit/obc2.xml")
system = ff.createSystem(pdb.topology)
integrator = LangevinIntegrator(500 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
try:
    platform = Platform.getPlatformByName("CUDA")
except Exception:
    platform = Platform.getPlatformByName("CPU")
sim = Simulation(pdb.topology, system, integrator, platform)
sim.context.setPositions(pdb.positions)
sim.reporters.append(DCDReporter("/tmp/trpcage_hot.dcd", 500))
sim.step(500000)

traj = md.load("/tmp/trpcage_hot.dcd", top=TOP)
ref = md.load(TOP)
traj.superpose(ref)
rmsd = md.rmsd(traj, ref, atom_indices=traj.topology.select("name CA")) * 10
unfolded = traj[rmsd >= 4.0]
unfolded[:: max(1, unfolded.n_frames // 300)].save_dcd(os.path.join(HERE, "data", "image.dcd"))
print(f"pool: {min(unfolded.n_frames, 300)} of {traj.n_frames} frames above 4 A")
