"""End to end test of the shared config driven cryoWEight on the three real systems,
adk, chignolin and ntl9, with three checks per system. First, assemble with validate
reproduces the original run tree when the original corpus is present, faithful files
byte identical and refactored files behaviorally validated. Second, the shared OpenMM
System builder produces XML identical to the original createSystem call. Third, the
shared CV reproduces the original inline CV exactly on the real 64 frame cryo-EM image
sample of each system. Run as python tests/test_three_systems.py
"""

import os, sys, json, subprocess, tempfile
import numpy as np, mdtraj as md
from openmm.app import PDBFile, ForceField, PME, CutoffNonPeriodic, HBonds
from openmm.unit import nanometer
from openmm import XmlSerializer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMP = os.path.join(HERE, "three_test")
ADK_COMMON = os.path.join(ROOT, "examples", "adk", "overrides", "WE_files", "common_files")
NTL9_COMMON = os.path.join(ROOT, "examples", "ntl9", "overrides", "WE_files", "common_files")
sys.path.insert(0, os.path.join(ROOT, "shared", "WE_files", "common_files"))
sys.path.insert(0, ROOT)
from cryoweight import configio
import build_system as bs, cv_families as cvf

# solvated/protein structure for build_system (boxed for explicit PME)
STRUCT = {
    "adk": os.path.join(ADK_COMMON, "bstate.pdb"),
    "chignolin": os.path.join(
        ROOT, "systems", "chignolin", "overrides", "WE_files", "common_files", "bstate.pdb"
    ),
    "ntl9": os.path.join(NTL9_COMMON, "bstate.pdb"),
}


def original_system(name, ff, top):
    if name in ("adk", "chignolin"):
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


def orig_cv(traj, ref, cvcfg):
    if cvcfg["cv_family"] == "rmsd_rg":
        ai = ref.topology.select(cvcfg.get("cv_atom_selection", "name CA"))
        r = np.asarray(md.rmsd(traj, ref, atom_indices=ai)) * 10
        rg = np.asarray(md.compute_rg(traj.atom_slice(ai))) * 10
        return np.column_stack((r, rg))
    g = cvcfg["resid_groups"]
    sel = lambda s: ref.topology.select(f"name CA and resid {s}")
    tn = cvf._calculate_angles(traj, sel(g["core_lid"]), sel(g["core_nmp"]), sel(g["nmp"]))
    tl = cvf._calculate_angles(traj, sel(g["core_lid_angle"]), sel(g["core_lid"]), sel(g["lid"]))
    return np.column_stack((tn, tl))


def main():

    print(f"{'system':10} {'validate':>10} {'build_system':>14} {'cv_of (real data)':>22}")
    print("-" * 62)
    ok = True
    for s in ["adk", "chignolin", "ntl9"]:
        cfg = configio.read_xml(os.path.join(ROOT, "systems", f"{s}.xml"))
        rc = cfg["reweight_config"]

        # (1) validate against the original run tree (corpus), if available
        corpus = os.path.join(ROOT, "..", "corpus", s)
        if os.path.isdir(corpus):
            v = subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, "assemble.py"),
                    "--system",
                    s,
                    "--validate",
                    corpus,
                ],
                capture_output=True,
                text=True,
            )
            v_ok = v.returncode == 0
            v_lbl = "OK" if v_ok else "FAIL"
        else:
            v_ok = True
            v_lbl = "skip"  # corpus not shipped with the repo

        # (2) build_system XML equivalence
        pdb = PDBFile(STRUCT[s])
        new = bs.build_system(bs.build_forcefield(rc), pdb.topology, rc)
        old = original_system(s, ForceField(rc["ff_main"], rc["ff_solvent"]), pdb.topology)
        b_ok = XmlSerializer.serialize(new) == XmlSerializer.serialize(old)

        # (3) cv_of on a real per system image sample
        traj = md.load(
            os.path.join(SAMP, f"{s}_img64.dcd"), top=os.path.join(SAMP, f"{s}_imgtop.pdb")
        )
        ref = md.load(os.path.join(SAMP, f"{s}_imgtop.pdb"))
        new_cv = cvf.cv_of(traj, ref, rc["cv_cfg"])
        old_cv = orig_cv(traj, ref, rc["cv_cfg"])
        c_ok = np.array_equal(new_cv, old_cv)

        ok &= v_ok and b_ok and c_ok
        print(
            f"{s:10} {v_lbl:>10} {('OK '+str(new.getNumParticles())+'p') if b_ok else 'FAIL':>14} "
            f"{('OK '+rc['cv_cfg']['cv_family']) if c_ok else 'FAIL':>22}"
        )
    print("-" * 62)
    print("ALL THREE SYSTEMS PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
