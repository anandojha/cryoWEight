"""Seeding MD that generates the initial trajectory. Reads reweight_config.json."""

from openmm.app import PDBFile, Modeller, Simulation, DCDReporter, StateDataReporter
from openmm.unit import kelvin, picoseconds, nanometer, kilojoule_per_mole
from openmm import LangevinIntegrator, Platform, LocalEnergyMinimizer
import build_system
import json

CFG = json.load(open("reweight_config.json"))
# Load the structure from the PDB file
pdb = PDBFile(CFG["input_pdb"])
# Load the force field parameters
forcefield = build_system.build_forcefield(CFG)
explicit = CFG["solvent_model"] == "explicit"
if explicit:
    # Create a Modeller object and add solvent
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        forcefield,
        model=CFG["water_model"],
        padding=float(CFG["solvent_padding_nm"]) * nanometer,
        neutralize=CFG["neutralize"],
    )
    topology = modeller.topology
    positions = modeller.positions
else:
    # For implicit solvent the system is built directly on the raw PDB topology
    topology = pdb.topology
    positions = pdb.positions
# Create the system using the (solvated) topology
system = build_system.build_system(forcefield, topology, CFG)
# Minimize the system
integrator = LangevinIntegrator(
    CFG["temperature_K"] * kelvin,
    CFG["friction_per_ps"] / picoseconds,
    CFG["timestep_ps"] * picoseconds,
)
try:
    platform = Platform.getPlatformByName(CFG["platform"])
except Exception:
    platform = Platform.getPlatformByName("CPU")
simulation = Simulation(topology, system, integrator, platform)
simulation.context.setPositions(positions)
# Perform energy minimization
LocalEnergyMinimizer.minimize(
    simulation.context,
    tolerance=CFG.get("minimize_tolerance_kj_nm", 10.0) * kilojoule_per_mole / nanometer,
    maxIterations=CFG.get("minimize_max_iterations", 10000),
)
minimized_positions = simulation.context.getState(getPositions=True).getPositions()
if explicit:
    # Convert periodic box vectors to plain floats
    box_vectors = system.getDefaultPeriodicBoxVectors()
    plain_box_vectors = [[float(v.x), float(v.y), float(v.z)] for v in box_vectors]
    topology.setPeriodicBoxVectors(plain_box_vectors)
# Save the minimized structure
with open(CFG["out_pdb"], 'w') as output:
    PDBFile.writeFile(topology, minimized_positions, output)
# Run molecular dynamics simulations
simulation.context.setPositions(minimized_positions)
# Add reporters for logging and trajectory output
simulation.reporters.append(StateDataReporter(
    CFG["out_log"], CFG["log_report_interval"],
    step=True, potentialEnergy=True, kineticEnergy=True, temperature=True, speed=True))
simulation.reporters.append(DCDReporter(CFG["out_dcd"], CFG["dcd_report_interval"]))
# Run MD simulation
simulation.step(CFG["n_steps"])
# Save the final state to a restart file
simulation.saveState(CFG["out_state"])
