#!/usr/bin/env python3
"""
Create systems/<name>.xml for a new system (see --help for inputs).

    python new_system.py --name <name> --input-pdb start.pdb --topology-we top.pdb \
        --reference-pdb folded.pdb --solvent implicit --cv rmsd_rg --cv-range 0 8 6 12 \
        --n-pixel 128 --pixel-size 0.3 --snr 1.0 --sigma 1.5 --n-iterations 30 --tau-ps 20
"""

from __future__ import annotations
import argparse, copy, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cryoweight import configio

BASE_FOR = {"explicit": "chignolin", "implicit": "ntl9"}
FF_DEFAULT = {
    "explicit": ("amber14-all.xml", "amber14/tip3p.xml"),
    "implicit": ("amber14-all.xml", "implicit/obc2.xml"),
}


def load_base(name):
    return configio.read_xml(os.path.join(HERE, "systems", f"{name}.xml"))


def _set(d, key, val):
    if val is not None:
        d[key] = val


def scaffold(a) -> dict:
    cfg = copy.deepcopy(load_base(BASE_FOR[a.solvent]))
    rc, sc = cfg["reweight_config"], cfg["seg_config"]
    ff_main, ff_solvent = (a.ff_main, a.ff_solvent)
    if ff_main is None:
        ff_main, ff_solvent = FF_DEFAULT[a.solvent]

    # Solvent model and force field, read by build_system at every stage.
    for blk in (rc, sc):
        blk["solvent_model"] = a.solvent
        blk["ff_main"], blk["ff_solvent"] = ff_main, ff_solvent

    # Collective variable family.
    for blk in (rc, sc):
        blk["cv_family"] = a.cv
        _set(blk, "cv_atom_selection", a.cv_atom_selection)
    rc["cv_cfg"] = (
        {"cv_family": "adk_angles", "resid_groups": a.resid_groups}
        if a.cv == "adk_angles"
        else {"cv_family": "rmsd_rg", "cv_atom_selection": a.cv_atom_selection}
    )
    if a.cv == "adk_angles":
        rc["resid_groups"] = sc["resid_groups"] = a.resid_groups

    # Input structures and topologies.
    ref = os.path.basename(a.reference_pdb)
    we_top = os.path.basename(a.topology_we)
    for blk in (rc, sc):
        blk["cv_reference_pdb"] = ref
    rc["input_pdb"] = os.path.basename(a.input_pdb)
    rc["topology_we"] = cfg["topology_explicit"] = we_top
    if a.topology_stripped:
        rc["topology_file"] = rc["topology_analysis"] = cfg["topology_stripped"] = os.path.basename(
            a.topology_stripped
        )
    for stem in ("out_dcd", "out_log", "out_pdb", "out_state", "init_md_dcd"):
        ext = rc[stem].rsplit(".", 1)[-1]
        rc[stem] = f"{a.name}.{ext}"

    # Extent of the collective variable space, used for binning and plotting.
    if a.cv_range:
        xmin, xmax, ymin, ymax = a.cv_range
        rc.update(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            x_range=[xmin, xmax],
            y_range=[ymin, ymax],
            bin_x_min=xmin,
            bin_x_max=xmax,
            bin_y_min=ymin,
            bin_y_max=ymax,
        )
        cfg["base_boundary_0"] = str([round(xmin + i * (xmax - xmin) / 11, 4) for i in range(12)])
        cfg["base_boundary_1"] = str([round(ymin + i * (ymax - ymin) / 11, 4) for i in range(12)])
        cfg["mab_at"] = str([round((xmin + xmax) / 2, 2), round((ymin + ymax) / 2, 2)])

    # Synthetic cryo-EM imaging parameters.
    _set(rc, "n_pixel", a.n_pixel)
    _set(rc, "pixel_size", a.pixel_size)
    _set(rc, "snr", a.snr)
    _set(rc, "sigma", a.sigma)
    if a.snr is not None:
        rc["snr_int"] = int(a.snr) if float(a.snr).is_integer() else a.snr

    # Weighted ensemble schedule.
    if a.n_iterations is not None:
        rc["n_iterations"] = cfg["n_iterations"] = a.n_iterations
    if a.tau_ps is not None:
        cfg["resampling_time_ps"] = a.tau_ps
        sc["n_steps_per_segment"] = int(round(a.tau_ps / float(sc["timestep_ps"])))

    cfg["ssh_host"] = a.ssh_host or cfg.get("ssh_host", "rusty")
    return cfg


def validate_inputs(a):
    errs = []
    for f in (a.input_pdb, a.topology_we, a.reference_pdb):
        if not os.path.isfile(f):
            errs.append(f"input file not found: {f}")
    if a.cv == "adk_angles" and not a.resid_groups:
        errs.append("cv adk_angles requires --resid-groups")
    # reference and topology atom order, the correctness precondition of rmsd_rg
    if a.cv == "rmsd_rg" and os.path.isfile(a.reference_pdb) and os.path.isfile(a.topology_we):
        try:
            import mdtraj

            r = mdtraj.load(a.reference_pdb)
            t = mdtraj.load(a.topology_we)
            rs = [(at.name, at.residue.name) for at in r.topology.atoms if at.name == "CA"]
            ts = [(at.name, at.residue.name) for at in t.topology.atoms if at.name == "CA"]
            if rs and ts and rs != ts:
                errs.append(
                    "reference and topology CA atoms differ in ordering/identity; "
                    "mdtraj.rmsd would be invalid (provide a reference with matching topology)"
                )
        except Exception as e:
            print(f"[warn] could not check reference/topology atom order: {e}")
    if errs:
        raise SystemExit("Input validation failed:\n  - " + "\n  - ".join(errs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scaffold a new cryoWEight system from inputs")
    p.add_argument("--name", required=True)
    p.add_argument("--input-pdb", required=True)
    p.add_argument("--topology-we", required=True)
    p.add_argument("--reference-pdb", required=True)
    p.add_argument("--topology-stripped")
    p.add_argument("--solvent", choices=["explicit", "implicit"], required=True)
    p.add_argument("--cv", choices=["rmsd_rg", "adk_angles"], default="rmsd_rg")
    p.add_argument("--cv-atom-selection", default="name CA")
    p.add_argument(
        "--resid-groups",
        type=lambda s: dict(kv.split("=") for kv in s.split(",")) if s else None,
        help="adk_angles only: nmp=35 to 55,core_nmp=...,core_lid=...,lid=...,core_lid_angle=...",
    )
    p.add_argument("--cv-range", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    p.add_argument("--n-pixel", type=int)
    p.add_argument("--pixel-size", type=float)
    p.add_argument("--snr", type=float)
    p.add_argument("--sigma", type=float)
    p.add_argument("--n-iterations", type=int)
    p.add_argument("--tau-ps", type=float)
    p.add_argument("--ff-main")
    p.add_argument("--ff-solvent")
    p.add_argument("--ssh-host")
    a = p.parse_args()
    validate_inputs(a)
    cfg = scaffold(a)
    out = os.path.join(HERE, "systems", f"{a.name}.xml")
    configio.write_xml(out, cfg)
    print(
        f"wrote {out}\nnext: python assemble.py --system {a.name} --dest run_{a.name}   (then cryoWEight.py --system {a.name} --local init)"
    )
