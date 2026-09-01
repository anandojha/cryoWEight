"""Collective variable families selected by cfg["cv_family"].
rmsd_rg gives CA RMSD to the reference and radius of gyration in Angstrom.
adk_angles  [theta_NMP, theta_LID] (deg)
"""

import numpy as np
import mdtraj


def _select(reference, cfg):
    return reference.topology.select(cfg.get("cv_atom_selection", "name CA"))


def rmsd_rg_traj(traj, reference, cfg):
    """CA RMSD to the reference and radius of gyration in Angstrom, one row per frame."""
    atom_indices = _select(reference, cfg)
    rmsd = np.asarray(mdtraj.rmsd(traj, reference, atom_indices=atom_indices)) * 10
    rg = np.asarray(mdtraj.compute_rg(traj.atom_slice(atom_indices))) * 10
    return np.column_stack((rmsd, rg))


def rmsd_rg(parent, seg, reference, cfg):
    """Collective variables for one segment, over the parent frame followed by the segment frames."""
    # Computed per trajectory and stacked rather than on parent.join(seg), because
    # mdtraj refuses to join when only one side carries a unitcell, which is the case
    # for a state saved from an implicit solvent system. The values are unchanged.
    return np.vstack((rmsd_rg_traj(parent, reference, cfg), rmsd_rg_traj(seg, reference, cfg)))


def _calculate_angles(traj, indices1, indices2, indices3):
    """Return the angle in degrees at the second centroid, one value per frame."""
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
    """The NMP and LID opening angles in degrees, one row per frame."""
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
    """Domain angles for one segment, over the parent frame followed by the segment frames."""
    # Computed per trajectory and stacked rather than on parent.join(seg), because
    # mdtraj refuses to join when only one side carries a unitcell, which is the case
    # for a state saved from an implicit solvent system. The values are unchanged.
    return np.vstack(
        (adk_angles_traj(parent, reference, cfg), adk_angles_traj(seg, reference, cfg))
    )


# trajectory CV with one row per frame, used by reweight, merge, get_distribution and plots
TRAJ_FAMILIES = {"rmsd_rg": rmsd_rg_traj, "adk_angles": adk_angles_traj}
# segment CV over the parent frame then the segment frames, used by cv.py
FAMILIES = {"rmsd_rg": rmsd_rg, "adk_angles": adk_angles}


def cv_of(traj, reference, cfg):
    """Return the collective variables of every frame as an array of shape (n_frames, 2).

    cfg["cv_family"] selects which pair is computed. Every analysis stage goes through
    here rather than computing its own, so the definition cannot drift between them.
    """
    fam = cfg["cv_family"]
    if fam not in TRAJ_FAMILIES:
        raise ValueError(f"unknown cv_family {fam!r} (expected {list(TRAJ_FAMILIES)})")
    return TRAJ_FAMILIES[fam](traj, reference, cfg)


def compute(parent, seg, reference, cfg):
    """Collective variables for one segment, called by the cv.py staged into each run directory."""
    fam = cfg["cv_family"]
    if fam not in FAMILIES:
        raise ValueError(f"unknown cv_family {fam!r} (expected {list(FAMILIES)})")
    return FAMILIES[fam](parent, seg, reference, cfg)
