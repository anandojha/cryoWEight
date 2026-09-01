"""Light seeding MD: short implicit-solvent run from bstate.pdb -> seed.dcd.

Dependency-light: numpy + mdtraj + openmm only. Produces the initial conformational
ensemble that the run0 reweight step bins and reweights. The implicit OBC2 system
matches common_files/build_system.py / seg_config.json exactly (amber14-all.xml +
implicit/obc2.xml, CutoffNonPeriodic 2 nm, HBonds, dielectrics 1.0/78.5).

Kept fast on CPU: a few thousand steps with frequent DCD frames. This is only a
seed; the WE run itself does the heavy sampling.
"""
import os
import json
import argparse

import numpy as np
import mdtraj as md
from openmm.app import PDBFile, ForceField, Simulation, CutoffNonPeriodic, HBonds, DCDReporter
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picoseconds, nanometer

_CONSTRAINTS = {"HBonds": HBonds, "None": None, None: None}


def main():
    ap = argparse.ArgumentParser(description="Short implicit MD seed from a PDB.")
    ap.add_argument("--pdb", required=True, help="Starting structure (bstate.pdb).")
    ap.add_argument("--seg-config", required=True, help="seg_config.json (FF/solvent).")
    ap.add_argument("--out", required=True, help="Output seed DCD path.")
    ap.add_argument("--n-steps", type=int, default=5000, help="Total MD steps.")
    ap.add_argument("--report-interval", type=int, default=250,
                    help="Steps between saved frames.")
    ap.add_argument("--platform", default="CPU", help="OpenMM platform (CPU|CUDA).")
    ap.add_argument("--seed", type=int, default=0, help="Langevin RNG seed.")
    args = ap.parse_args()

    cfg = json.load(open(args.seg_config))
    pdb = PDBFile(args.pdb)
    ff = ForceField(cfg["ff_main"], cfg["ff_solvent"])
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=float(cfg["nonbonded_cutoff_nm"]) * nanometer,
        constraints=_CONSTRAINTS[cfg.get("constraints", "HBonds")],
        soluteDielectric=float(cfg["solute_dielectric"]),
        solventDielectric=float(cfg["solvent_dielectric"]),
    )
    integrator = LangevinIntegrator(
        cfg["temperature_K"] * kelvin,
        cfg["friction_per_ps"] / picoseconds,
        cfg["timestep_ps"] * picoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    sim = Simulation(pdb.topology, system, integrator,
                     Platform.getPlatformByName(args.platform))
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(maxIterations=100)
    sim.context.setVelocitiesToTemperature(cfg["temperature_K"] * kelvin, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sim.reporters.append(DCDReporter(args.out, args.report_interval))
    sim.step(args.n_steps)
    # Flush + report frame count.
    del sim
    n = md.load(args.out, top=args.pdb).n_frames
    print(f"[seed] wrote {n} frames ({args.n_steps} steps) -> {args.out}")


if __name__ == "__main__":
    main()
