"""OpenMM System factory; explicit or implicit by cfg["solvent_model"]."""
from openmm.app import ForceField, PME, CutoffNonPeriodic, HBonds
from openmm.unit import nanometer

_CONSTRAINTS = {"HBonds": HBonds, None: None, "None": None}


def build_forcefield(cfg):
    return ForceField(cfg["ff_main"], cfg["ff_solvent"])


def build_system(forcefield, topology, cfg):
    """Return an OpenMM System for `topology` per the solvent model in cfg."""
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
