"""Behavioral equivalence of the refactored build_system and cv_families against
what the original per system production.py and cv.py produced.

For build_system, each System is built the original inline way and via the factory
              assert identical OpenMM XML serialization.
and the XML must match. For the CV, both routes must give
              identical arrays.

Run as python tests/test_phase2_equivalence.py with openmm and mdtraj installed.
"""

import os, sys
import numpy as np
import mdtraj
from openmm.app import PDBFile, ForceField, PME, CutoffNonPeriodic, HBonds
from openmm.unit import nanometer
from openmm import XmlSerializer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "cryoweight"))
import build_system as bs
import cv_families as cvf

ADK_COMMON = os.path.join(ROOT, "examples", "adk", "overrides", "WE_files", "common_files")
NTL9_COMMON = os.path.join(ROOT, "examples", "ntl9", "overrides", "WE_files", "common_files")
CHIG_BSTATE = os.path.join(
    ROOT, "systems", "chignolin", "overrides", "WE_files", "common_files", "bstate.pdb"
)
CHIG_TRAJ = os.path.join(ROOT, "systems", "chignolin", "init_MD", "chignolin.dcd")
CHIG_TOP = os.path.join(ROOT, "systems", "chignolin", "init_MD", "chignolin.pdb")

SYSTEMS = {
    "adk": dict(
        pdb=os.path.join(ADK_COMMON, "bstate.pdb"),
        solvent_model="explicit",
        ff_main="amber14-all.xml",
        ff_solvent="amber14/tip3p.xml",
        nonbonded_cutoff_nm=1,
    ),
    "chignolin": dict(
        pdb=CHIG_BSTATE,
        solvent_model="explicit",
        ff_main="amber14-all.xml",
        ff_solvent="amber14/tip3p.xml",
        nonbonded_cutoff_nm=1,
    ),
    "ntl9": dict(
        pdb=os.path.join(NTL9_COMMON, "bstate.pdb"),
        solvent_model="implicit",
        ff_main="amber14-all.xml",
        ff_solvent="implicit/obc2.xml",
        nonbonded_cutoff_nm=2,
        solute_dielectric=1.0,
        solvent_dielectric=78.5,
    ),
}


def original_system(name, cfg, top):
    ff = ForceField(cfg["ff_main"], cfg["ff_solvent"])
    if name in ("adk", "chignolin"):  # explicit solvent, verbatim from the original production.py
        return ff.createSystem(
            top, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
        )
    return ff.createSystem(
        top,
        nonbondedMethod=CutoffNonPeriodic,
        constraints=HBonds,
        nonbondedCutoff=2 * nanometer,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )


def test_build_system():
    print("== build_system equivalence ==")
    for name, cfg in SYSTEMS.items():
        pdb = PDBFile(cfg["pdb"])
        cfg = dict(cfg, constraints="HBonds")
        ff = bs.build_forcefield(cfg)
        new = bs.build_system(ff, pdb.topology, cfg)
        old = original_system(name, cfg, pdb.topology)
        xn, xo = XmlSerializer.serialize(new), XmlSerializer.serialize(old)
        ok = xn == xo
        print(
            f"   {name:10} {'OK' if ok else 'FAIL'}  ({pdb.topology.getNumAtoms()} atoms, system XML {len(xn)} chars)"
        )
        assert ok, f"{name}: build_system XML differs from original"


# original inline CV logic, verbatim from the per system cv.py
def orig_rmsd_rg(parent, seg, reference, sel="name CA"):
    ai = reference.topology.select(sel)
    rmsd_parent = mdtraj.rmsd(parent, reference, atom_indices=ai)
    rmsd_traj = mdtraj.rmsd(seg, reference, atom_indices=ai)
    r = np.asarray(np.append(rmsd_parent, rmsd_traj)) * 10
    rgp = mdtraj.compute_rg(parent.atom_slice(ai))
    rgt = mdtraj.compute_rg(seg.atom_slice(ai))
    rg = np.asarray(np.append(rgp, rgt)) * 10
    return np.column_stack((r, rg))


def orig_adk(parent, seg, reference, g):
    sel = lambda spec: reference.topology.select(f"name CA and resid {spec}")
    nmp, cnmp, clid, lid, cla = (
        sel(g["nmp"]),
        sel(g["core_nmp"]),
        sel(g["core_lid"]),
        sel(g["lid"]),
        sel(g["core_lid_angle"]),
    )
    comb = parent.join(seg)
    tn = cvf._calculate_angles(comb, clid, cnmp, nmp)
    tl = cvf._calculate_angles(comb, cla, clid, lid)
    return np.column_stack((tn, tl))


def test_cv():
    print("== cv_families equivalence ==")
    # rmsd_rg via chignolin trajectory
    traj = mdtraj.load(CHIG_TRAJ, top=CHIG_TOP)
    ref = traj[0]
    parent, seg = traj[0:3], traj[3:6]
    cfg = dict(cv_family="rmsd_rg", cv_atom_selection="name CA")
    new = cvf.compute(parent, seg, ref, cfg)
    old = orig_rmsd_rg(parent, seg, ref)
    print(f"   rmsd_rg/chignolin {'OK' if np.array_equal(new, old) else 'FAIL'}  shape={new.shape}")
    assert np.array_equal(new, old)
    # rmsd_rg via ntl9 (reference = folded.pdb, distinct from topology)
    nt = mdtraj.load(os.path.join(NTL9_COMMON, "bstate.pdb"))
    nref = mdtraj.load(os.path.join(NTL9_COMMON, "folded.pdb"))
    cfg = dict(cv_family="rmsd_rg", cv_atom_selection="name CA")
    new = cvf.compute(nt, nt, nref, cfg)
    old = orig_rmsd_rg(nt, nt, nref)
    print(f"   rmsd_rg/ntl9      {'OK' if np.array_equal(new, old) else 'FAIL'}  shape={new.shape}")
    assert np.array_equal(new, old)
    # adk_angles
    ad = mdtraj.load(os.path.join(ADK_COMMON, "bstate.pdb"))
    groups = dict(
        nmp="35 to 55",
        core_nmp="90 to 100",
        core_lid="115 to 125",
        lid="125 to 153",
        core_lid_angle="179 to 185",
    )
    cfg = dict(cv_family="adk_angles", resid_groups=groups)
    new = cvf.compute(ad, ad, ad, cfg)
    old = orig_adk(ad, ad, ad, groups)
    print(
        f"   adk_angles/adk    {'OK' if np.array_equal(new, old) else 'FAIL'}  shape={new.shape}  sample={new[0]}"
    )
    assert np.array_equal(new, old)


if __name__ == "__main__":
    test_build_system()
    test_cv()
    print("\nALL PHASE-2 EQUIVALENCE TESTS PASSED")
