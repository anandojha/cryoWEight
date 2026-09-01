"""Light CV-reweight + WESTPA basis-state builder for the chignolin demo.

Dependency-light: numpy, scipy, mdtraj, openmm ONLY. None of the heavy ML /
analysis stacks are imported (see README for the explicit exclusion list).

This is the CV-signal analogue of the cryoWEight reweighting step. Instead of
reweighting an ensemble to a target *cryo-EM image* distribution, it reweights to
a target *CV* distribution ([CA-RMSD-to-folded, Rg], both in Angstrom) taken from
the unfolded basin of data/image.dcd. The EM weight update is the same one used in
production (demo/cv_reweight.py), just driven by a CV-space likelihood.

Pipeline (one reweight step), following the cryoWEight protocol:
  1. Load the ensemble trajectory (seeding init/seed.dcd for run0, or the merged
     WE trajectory merged_WE/traj_all.dcd for run>=1) + topology with mdtraj.
  2. Compute per-frame CV = [CA-RMSD-to-folded * 10, Rg * 10] (Angstrom) inline.
  3. Build the target CV cloud from image.dcd: frames with CA-RMSD in
     [cv_target_lo, cv_target_hi] (default [4, 6] A, the unfolded basin).
  4. Cluster the ensemble in CV space (bin width Delta): keep one representative
     frame per occupied bin (the frame closest to the bin centroid). This is the
     paper's clustering step -- the WE is seeded from distinct per-bin structures,
     not multinomial copies of one frame.
  5. cv_reweight(...) the M representatives -> EM-optimized weights (numpy EM).
  6. Seed WESTPA with ALL M representatives, each carrying its reweighted weight
     (not uniform): build the implicit OBC2 OpenMM system, set the frame's
     positions, minimizeEnergy, saveState -> bstates/{i:04d}/bstate.xml; write its
     pcoord -> bstates/{i:04d}/pcoord.init; append "{i:04d}  {weight}  {i:04d}" to
     bstates/bstates.txt.
  7. Recompute the MAB bottleneck [RMSD, Rg] (compute_bottleneck: the FES local max
     subject to xi >= mu + n*sigma) and write it to bottleneck.txt so run.sh can
     patch the NEXT WE run's west.cfg `at:` for the "adaptive binning" step.

The output layout matches exactly what get_pcoord.sh and init.sh
(w_init --bstate-file bstates/bstates.txt) consume.
"""
import os
import sys
import json
import shutil
import argparse

import numpy as np
import mdtraj as md
from scipy.ndimage import maximum_filter, generate_binary_structure
from openmm.app import PDBFile, ForceField, Simulation, CutoffNonPeriodic, HBonds
from openmm import LangevinIntegrator, Platform
from openmm.unit import kelvin, picoseconds, nanometer

# Pure-numpy CV reweighter (same EM as production; lives one dir up in demo/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_reweight import cv_reweight

_CONSTRAINTS = {"HBonds": HBonds, "None": None, None: None}


# --------------------------------------------------------------------------- #
# CV computation (inline, mdtraj only)                                         #
# --------------------------------------------------------------------------- #
def compute_cv(traj, reference, atom_selection="name CA"):
    """[CA-RMSD-to-reference, Rg], both in Angstrom (nm*10), one row per frame.

    Identical math to common_files/cv_families.rmsd_rg_traj so the analysis CV and
    the per-segment pcoord agree.
    """
    ca = reference.topology.select(atom_selection)
    rmsd = np.asarray(md.rmsd(traj, reference, atom_indices=ca)) * 10.0
    rg = np.asarray(md.compute_rg(traj.atom_slice(ca))) * 10.0
    return np.column_stack((rmsd, rg))


def build_target_cv(image_dcd, topology, reference, lo, hi, atom_selection="name CA"):
    """Target CV cloud = frames of image.dcd with CA-RMSD in [lo, hi] (the unfolded
    basin). Falls back to the full image.dcd CV cloud if the band is empty."""
    img = md.load(image_dcd, top=topology)
    cv = compute_cv(img, reference, atom_selection)
    sel = (cv[:, 0] >= lo) & (cv[:, 0] <= hi)
    if np.any(sel):
        print(f"[reweight] target: {int(sel.sum())} frames of "
              f"{os.path.basename(image_dcd)} with CA-RMSD in [{lo}, {hi}] A")
        return cv[sel]
    print(f"[reweight] target band [{lo}, {hi}] A empty; using all "
          f"{cv.shape[0]} image.dcd frames")
    return cv


def cluster_representatives(cv, bin_width):
    """Paper clustering step (eqs 14-17): partition the ensemble into a regular CV
    grid of width `bin_width` and keep one representative frame per occupied bin --
    the frame closest to the bin centroid. Returns the representative frame indices
    (sorted). The WE is then seeded from these distinct per-bin structures."""
    bins = np.floor(cv / bin_width).astype(int)
    groups = {}
    for a in range(len(cv)):
        groups.setdefault((bins[a, 0], bins[a, 1]), []).append(a)
    reps = []
    for (b1, b2), members in groups.items():
        centroid = (np.array([b1, b2]) + 0.5) * bin_width
        members = np.asarray(members)
        reps.append(int(members[np.argmin(np.linalg.norm(cv[members] - centroid, axis=1))]))
    return np.array(sorted(reps))


# --------------------------------------------------------------------------- #
# Bottleneck-coordinate recomputation (MAB `at:` per cryoWEight iteration)      #
# --------------------------------------------------------------------------- #
def compute_bottleneck(cv, nbins=(100, 100), sigma_mults=(3, 2, 1),
                       default=(5.0, 6.0)):
    """[RMSD, Rg] bottleneck coordinate for the MAB `at:` point, from the prior
    ensemble's 2D CV distribution.

    Faithful small-numpy/scipy reimplementation of chignolin_cv get_bottleneck
    (systems/chignolin_cv/overrides/scripts/reweight.py:1861) restricted to the
    positive-sigma (distal/open) branch the cryoWEight protocol patches into `at:`.

    Mirrors these get_bottleneck lines:
      * H,xedges,yedges = np.histogram2d(rmsd, rg, bins=nbins, density=True)  (l.1918)
      * F = -np.log(H + 1e-12); F -= F.min()                                  (l.1919-1920)
      * xg = bin centers along RMSD; Fg = F.T; mask empty bins                (l.1921-1925)
      * mu, sigma = rmsd.mean(), rmsd.std()                                   (l.1931)
      * tau = mu + m*sigma; cols = where(xg >= tau) for the m>=0 open side    (l.1936-1937)
      * select the FES extremum in that masked region                        (l.1958-1963)

    Two faithful deviations, both kept light:
      * the progress-coordinate threshold steps n = 3 -> 2 -> 1 (distal, medial,
        proximal), falling back to a smaller n only when the more distal band has no
        populated bins -- exactly the "stepping n until the region contains populated
        bins" rule the task specifies (get_bottleneck loops all sigma_mults and emits
        a point per populated level; here we want the single most-distal populated one);
      * the metastable basin is found as a LOCAL MAXIMUM of F (a sparsely populated
        valley wall, high free energy) using scipy.ndimage maximum_filter +
        generate_binary_structure, instead of get_bottleneck's masked argmin. The
        threshold xi >= mu + n*sigma already restricts to the distal tail; the local
        max picks the sparsest (highest-F) populated cell there.

    Parameters
    ----------
    cv : (N, 2) array   per-frame [CA-RMSD, Rg] (Angstrom).
    nbins : (int, int)  RMSD, Rg histogram bins.
    sigma_mults : iterable of n   progress-coordinate thresholds n in xi >= mu+n*sigma,
                                  tried most-distal first (default 3, 2, 1).
    default : (float, float)   returned if no thresholded band is ever populated.

    Returns
    -------
    (rmsd, rg) : tuple of float   the bottleneck coordinate for `at:`.
    """
    cv = np.asarray(cv, float)
    rmsd, rg = cv[:, 0], cv[:, 1]
    # FES F = -ln P on a 2D [RMSD, Rg] grid, shifted to min 0 (get_bottleneck l.1918-1920).
    H, xedges, yedges = np.histogram2d(rmsd, rg, bins=nbins, density=True)
    F = -np.log(H + 1e-12)
    F -= F.min()
    xg = 0.5 * (xedges[:-1] + xedges[1:])          # RMSD bin centers (progress coord)
    yg = 0.5 * (yedges[:-1] + yedges[1:])          # Rg bin centers
    populated = H > 0                               # mask of populated bins (l.1924)
    # Local maxima of F (sparsely populated metastable cells) via ndimage; a cell is a
    # local max if it equals the max over its 8-neighborhood and is itself populated.
    footprint = generate_binary_structure(2, 2)     # 3x3 incl diagonals
    local_max = (F == maximum_filter(F, footprint=footprint)) & populated
    # mu, sigma of the prior CV distribution along the progress coordinate (l.1931).
    mu, sigma = rmsd.mean(), rmsd.std()
    # Step n = 3 -> 2 -> 1: keep the most-distal threshold xi >= mu + n*sigma that
    # still has a populated local-max cell (l.1936-1937 open side, falling back).
    for n in sigma_mults:
        tau = mu + n * sigma
        cols = np.where(xg >= tau)[0]               # distal RMSD columns
        if cols.size == 0:
            continue
        region = np.zeros_like(F, bool)
        region[cols, :] = True
        cand = local_max & region                   # populated FES maxima in the band
        if not cand.any():
            # no isolated local max in the band; fall back to the highest-F populated
            # cell there (the get_bottleneck masked-extremum behavior).
            cand = populated & region
            if not cand.any():
                continue
            Fmask = np.where(cand, F, -np.inf)
            i, j = np.unravel_index(np.argmax(Fmask), F.shape)
            return float(xg[i]), float(yg[j])
        Fmask = np.where(cand, F, -np.inf)
        i, j = np.unravel_index(np.argmax(Fmask), F.shape)
        return float(xg[i]), float(yg[j])
    # Nothing populated in any distal band: fall back to the supplied default.
    return float(default[0]), float(default[1])


# --------------------------------------------------------------------------- #
# Basis-state builder (implicit OBC2 OpenMM system, minimize, saveState)       #
# --------------------------------------------------------------------------- #
def _build_simulation(topology_pdb, seg_cfg, platform_name):
    """Implicit OBC2 OpenMM Simulation matching common_files/build_system.py:
    amber14-all.xml + implicit/obc2.xml, CutoffNonPeriodic (2 nm), HBonds,
    dielectrics 1.0 / 78.5."""
    pdb = PDBFile(topology_pdb)
    ff = ForceField(seg_cfg["ff_main"], seg_cfg["ff_solvent"])
    constraints = _CONSTRAINTS[seg_cfg.get("constraints", "HBonds")]
    cutoff = float(seg_cfg["nonbonded_cutoff_nm"]) * nanometer
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=cutoff,
        constraints=constraints,
        soluteDielectric=float(seg_cfg["solute_dielectric"]),
        solventDielectric=float(seg_cfg["solvent_dielectric"]),
    )
    integrator = LangevinIntegrator(
        seg_cfg["temperature_K"] * kelvin,
        seg_cfg["friction_per_ps"] / picoseconds,
        seg_cfg["timestep_ps"] * picoseconds,
    )
    platform = Platform.getPlatformByName(platform_name)
    sim = Simulation(pdb.topology, system, integrator, platform)
    return pdb, sim


def build_bstates(frames_xyz, weights, pcoords, topology_pdb, seg_cfg,
                  bstates_dir, pcoord_len, platform_name, minimize_iters=100):
    """Build one WESTPA basis state per (frame, weight, pcoord) triple.

    frames_xyz : (K, n_atoms, 3) positions in nanometer (mdtraj convention).
    weights    : (K,) basis-state probabilities (must sum to 1).
    pcoords    : (K, 2) [RMSD, Rg] in Angstrom for each frame.

    Writes, for each i:
      bstates/{i:04d}/bstate.xml   (OpenMM serialized State after minimization)
      bstates/{i:04d}/pcoord.init  (pcoord repeated to pcoord_len rows)
    and appends "{i:04d}  {weight}  {i:04d}" to bstates/bstates.txt.
    """
    if os.path.isdir(bstates_dir):
        shutil.rmtree(bstates_dir)
    os.makedirs(bstates_dir)

    # One Simulation object reused across frames (topology is fixed).
    pdb, sim = _build_simulation(topology_pdb, seg_cfg, platform_name)

    bstates_lines = []
    K = len(weights)
    for i in range(K):
        folder = os.path.join(bstates_dir, f"{i:04d}")
        os.makedirs(folder, exist_ok=True)
        # Positions: mdtraj xyz is in nm; OpenMM Simulation expects nm-valued
        # Quantity. Setting a bare array assumes nm, which is what we want.
        sim.context.setPositions(frames_xyz[i] * nanometer)
        sim.minimizeEnergy(maxIterations=minimize_iters)
        sim.saveState(os.path.join(folder, "bstate.xml"))
        # pcoord.init: a basis state has ONE pcoord point (RMSD, Rg) -> shape (2,).
        # (pcoord_len rows is only for per-segment trajectories, not basis states.)
        pc = np.asarray(pcoords[i], float).reshape(1, -1)
        np.savetxt(os.path.join(folder, "pcoord.init"), pc)
        bstates_lines.append(f"{i:04d}  {weights[i]:.32e}  {i:04d}\n")
        print(f"[reweight] bstate {i:04d}: w={weights[i]:.3e} "
              f"RMSD={pcoords[i][0]:.2f} Rg={pcoords[i][1]:.2f}")

    with open(os.path.join(bstates_dir, "bstates.txt"), "w") as f:
        f.writelines(bstates_lines)
    print(f"[reweight] wrote {K} basis states + bstates.txt to {bstates_dir}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Light CV-reweight + WESTPA bstate builder (chignolin demo).")
    ap.add_argument("--ensemble", required=True,
                    help="Ensemble trajectory to reweight (seed.dcd for run0, "
                         "merged_WE/traj_all.dcd for run>=1).")
    ap.add_argument("--topology", required=True, help="Topology PDB (138-atom).")
    ap.add_argument("--folded", required=True, help="Folded reference PDB (RMSD target).")
    ap.add_argument("--image", required=True, help="image.dcd (CV target source).")
    ap.add_argument("--seg-config", required=True, help="seg_config.json (FF / solvent).")
    ap.add_argument("--out", required=True, help="Output bstates/ directory.")
    ap.add_argument("--k", type=int, default=32,
                    help="Cap on the number of basis states (occupied bins kept).")
    ap.add_argument("--bin-width", type=float, default=0.5,
                    help="CV clustering bin width (Angstrom) for representative selection.")
    ap.add_argument("--sigma", type=float, default=0.6,
                    help="CV-space bandwidth (Angstrom).")
    ap.add_argument("--cv-target-lo", type=float, default=4.0,
                    help="Lower CA-RMSD (A) of the unfolded target basin.")
    ap.add_argument("--cv-target-hi", type=float, default=6.0,
                    help="Upper CA-RMSD (A) of the unfolded target basin.")
    ap.add_argument("--pcoord-len", type=int, default=3,
                    help="Rows per pcoord.init (must match west.cfg pcoord_len).")
    ap.add_argument("--bottleneck-out", default=None,
                    help="Write the recomputed MAB bottleneck [RMSD, Rg] (one line "
                         "'RMSD Rg') here for run.sh to patch the NEXT run's west.cfg "
                         "`at:`. Default: <out>/../bottleneck.txt.")
    ap.add_argument("--platform", default="CPU", help="OpenMM platform (CPU|CUDA).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for resampling.")
    ap.add_argument("--minimize-iters", type=int, default=100,
                    help="Max minimization iterations per basis state.")
    args = ap.parse_args()

    seg_cfg = json.load(open(args.seg_config))
    sel = seg_cfg.get("cv_atom_selection", "name CA")

    # 1-2. Ensemble + CV.
    reference = md.load(args.folded)
    ens = md.load(args.ensemble, top=args.topology)
    if ens.n_frames == 0:
        raise RuntimeError(f"ensemble {args.ensemble} has 0 frames")
    cv_struct = compute_cv(ens, reference, sel)
    print(f"[reweight] ensemble: {ens.n_frames} frames from "
          f"{os.path.basename(args.ensemble)}")

    # 3. Target CV cloud (unfolded basin of image.dcd).
    cv_target = build_target_cv(args.image, args.topology, reference,
                                args.cv_target_lo, args.cv_target_hi, sel)

    # 4. Cluster the ensemble in CV space -> one representative per occupied bin.
    rep_idx = cluster_representatives(cv_struct, args.bin_width)
    print(f"[reweight] {len(rep_idx)} occupied CV bins (bin width {args.bin_width} A) "
          f"from {ens.n_frames} frames")
    cv_rep = cv_struct[rep_idx]

    # 5. CV reweight the representatives -> EM-optimized weights (numpy EM).
    w = cv_reweight(cv_rep, cv_target, args.sigma)
    # Cap at --k highest-weight representatives if more bins than the cap are occupied.
    order = np.argsort(w)[::-1][:args.k]
    rep_idx, cv_rep, w = rep_idx[order], cv_rep[order], w[order]
    # Floor the weights so every distinct clustered structure seeds a (non-degenerate)
    # walker for bin coverage; the EM weights still dominate, and WESTPA's per-bin
    # target count samples each occupied bin regardless of its small weight.
    floor = 1e-3 * w.max() if w.max() > 0 else 1.0
    weights = np.maximum(w, floor)
    weights /= weights.sum()
    print(f"[reweight] weights: min={weights.min():.2e} max={weights.max():.2e} "
          f"ESS={1.0 / np.sum(weights ** 2):.1f}  ({len(weights)} basis states)")

    # 6. Seed WESTPA with the reweighted representatives (implicit OBC2 bstates),
    #    each carrying its reweighted weight (NOT uniform).
    build_bstates(ens.xyz[rep_idx], weights, cv_rep, args.topology, seg_cfg,
                  args.out, args.pcoord_len, args.platform, args.minimize_iters)

    # 7. Recompute the MAB bottleneck `at:` coordinate from this iteration's full CV
    #    distribution (the "WE simulations with adaptive binning" hand-off). run.sh
    #    reads bottleneck.txt and seds it into the NEXT run's west.cfg `at:` line.
    bx, by = compute_bottleneck(cv_struct)
    out_bn = args.bottleneck_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "bottleneck.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_bn)), exist_ok=True)
    with open(out_bn, "w") as f:
        f.write(f"{bx:.4f} {by:.4f}\n")
    print(f"[reweight] bottleneck (MAB at:) RMSD={bx:.2f} Rg={by:.2f} -> {out_bn}")


if __name__ == "__main__":
    main()
