"""Collective-variable families selected by cfg["cv_family"]:
  rmsd_rg     [CA-RMSD-to-reference, radius of gyration] (A)
  adk_angles  [theta_NMP, theta_LID] (deg)
"""
import numpy as np
import mdtraj


def _select(reference, cfg):
    return reference.topology.select(cfg.get("cv_atom_selection", "name CA"))


def rmsd_rg_traj(traj, reference, cfg):
    """[CA-RMSD-to-reference, Rg], both in Angstrom (nm*10), one row per frame of traj."""
    atom_indices = _select(reference, cfg)
    rmsd = np.asarray(mdtraj.rmsd(traj, reference, atom_indices=atom_indices)) * 10
    rg = np.asarray(mdtraj.compute_rg(traj.atom_slice(atom_indices))) * 10
    return np.column_stack((rmsd, rg))


def rmsd_rg(parent, seg, reference, cfg):
    """Segment CV: parent frames then seg frames."""
    return rmsd_rg_traj(parent.join(seg), reference, cfg)


def _calculate_angles(traj, indices1, indices2, indices3):
    """Angle (deg) at centroid2 between centroid1-centroid2-centroid3, per frame."""
    center1 = traj.xyz[:, indices1, :].mean(axis=1)
    center2 = traj.xyz[:, indices2, :].mean(axis=1)
    center3 = traj.xyz[:, indices3, :].mean(axis=1)
    vec1 = center1 - center2
    vec2 = center3 - center2
    dot_product = np.sum(vec1 * vec2, axis=1)
    norm1 = np.linalg.norm(vec1, axis=1)
    norm2 = np.linalg.norm(vec2, axis=1)
    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def adk_angles_traj(traj, reference, cfg):
    """[theta_NMP, theta_LID] (deg) over CA residue-group centroids, one row per frame."""
    g = cfg["resid_groups"]
    sel = lambda spec: reference.topology.select(f"name CA and resid {spec}")
    nmp_ca = sel(g["nmp"])
    core_nmp_ca = sel(g["core_nmp"])
    core_lid_ca = sel(g["core_lid"])
    lid_ca = sel(g["lid"])
    core_lid_angle_ca = sel(g["core_lid_angle"])
    theta_nmp = _calculate_angles(traj, core_lid_ca, core_nmp_ca, nmp_ca)
    theta_lid = _calculate_angles(traj, core_lid_angle_ca, core_lid_ca, lid_ca)
    return np.column_stack((theta_nmp, theta_lid))


def adk_angles(parent, seg, reference, cfg):
    """Segment CV: angles over parent.join(seg)."""
    return adk_angles_traj(parent.join(seg), reference, cfg)


# trajectory-level CV (one row per frame): reweight, merge, get_distribution, plots
TRAJ_FAMILIES = {"rmsd_rg": rmsd_rg_traj, "adk_angles": adk_angles_traj}
# segment-level CV (parent then seg): the per-segment cv.py
FAMILIES = {"rmsd_rg": rmsd_rg, "adk_angles": adk_angles}


def cv_of(traj, reference, cfg):
    """Trajectory-level CV, (n_frames, 2) for cfg["cv_family"]. Entry point for all
    analysis code."""
    fam = cfg["cv_family"]
    if fam not in TRAJ_FAMILIES:
        raise ValueError(f"unknown cv_family {fam!r} (expected {list(TRAJ_FAMILIES)})")
    return TRAJ_FAMILIES[fam](traj, reference, cfg)


def compute(parent, seg, reference, cfg):
    """Segment-level CV (parent frames then seg frames), used by the runtime cv.py."""
    fam = cfg["cv_family"]
    if fam not in FAMILIES:
        raise ValueError(f"unknown cv_family {fam!r} (expected {list(FAMILIES)})")
    return FAMILIES[fam](parent, seg, reference, cfg)
