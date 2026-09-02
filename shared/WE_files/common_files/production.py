"""WESTPA per segment MD propagator. Reads seg_config.json."""

import json
from openmm.app import PDBFile, Simulation, ForceField, StateDataReporter, DCDReporter
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picoseconds
from build_system import build_system

cfg = json.load(open("seg_config.json"))

pdb = PDBFile(cfg["input_pdb"])
forcefield = ForceField(cfg["ff_main"], cfg["ff_solvent"])
system = build_system(forcefield, pdb.topology, cfg)
integrator = LangevinIntegrator(
    cfg["temperature_K"] * kelvin,
    cfg["friction_per_ps"] / picoseconds,
    cfg["timestep_ps"] * picoseconds,
)
integrator.setRandomNumberSeed(RAND)
try:
    platform = Platform.getPlatformByName(cfg["platform"])
except Exception:
    platform = Platform.getPlatformByName("CPU")
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.loadState(cfg["restart_state"])
simulation.reporters.append(
    StateDataReporter(
        cfg["out_log"],
        cfg["log_report_interval"],
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        speed=True,
    )
)
simulation.reporters.append(DCDReporter(cfg["out_dcd"], cfg["dcd_report_interval"]))
simulation.step(cfg["n_steps_per_segment"])
simulation.saveState(cfg["out_state"])
