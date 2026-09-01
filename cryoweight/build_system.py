"""Build the OpenMM System for a topology, in explicit or implicit solvent.

cfg["solvent_model"] chooses between the two. Explicit solvent uses PME in a periodic
box, implicit solvent uses a plain cutoff with the generalized Born dielectrics.
"""

from openmm.app import ForceField, PME, CutoffNonPeriodic, HBonds
from openmm.unit import nanometer

_CONSTRAINTS = {"HBonds": HBonds, None: None, "None": None}


def build_forcefield(cfg):
    return ForceField(cfg["ff_main"], cfg["ff_solvent"])


def build_system(forcefield, topology, cfg):
    """Return an OpenMM System for the topology, in the solvent model named in the configuration."""
    cutoff = float(cfg["nonbonded_cutoff_nm"]) * nanometer
    constraints = _CONSTRAINTS[cfg.get("constraints", "HBonds")]
    model = cfg["solvent_model"]
    if model == "explicit":
        return forcefield.createSystem(
            topology,
            nonbondedMethod=PME,
            nonbondedCutoff=cutoff,
            constraints=constraints,
        )
    if model == "implicit":
        return forcefield.createSystem(
            topology,
            nonbondedMethod=CutoffNonPeriodic,
            constraints=constraints,
            nonbondedCutoff=cutoff,
            soluteDielectric=float(cfg["solute_dielectric"]),
            solventDielectric=float(cfg["solvent_dielectric"]),
        )
    raise ValueError(f"unknown solvent_model {model!r} (expected explicit|implicit)")
