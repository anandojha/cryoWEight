"""Likelihood families selected by cfg["likelihood"].
  cryoem  squared L2 distance between synthetic cryo-EM particles and projected structures
  cv      squared distance in collective variable space to a target distribution

Every family returns (diff, scale). diff[j, i] is the squared distance from structure j to
observation i, and the log likelihood the EM step consumes is -diff / (2 * scale**2). The
families differ only in what an observation is and how the distance to it is measured, so
everything downstream of this module is shared between them.
"""

import os

import numpy as np
import mdtraj as md
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Imported relatively when this file is used as part of the installed package, and
# by bare name when assemble has copied it into a run directory, where a compute
# node executes it as a loose script with no package around it.
try:
    from . import cv_families
except ImportError:
    import cv_families
try:
    from .cryoER_core import (
        align_traj,
        approx_lmbd,
        calc_image_struc_distance,
        make_synthetic_images,
    )
except ImportError:
    from cryoER_core import (
        align_traj,
        approx_lmbd,
        calc_image_struc_distance,
        make_synthetic_images,
    )


def _plot_image_grid(images, path):
    """Grid of the first sixteen particles, on a shared intensity scale."""
    # The 1st and 99th percentiles set the scale so that a few extreme pixels cannot
    # wash out the rest of the grid.
    vmin, vmax = np.percentile(images, [1, 99])
    fig = plt.figure(figsize=(8, 8), dpi=500)
    for i in range(min(16, images.shape[0])):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(images[i], cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=500, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def cryoem(ctx):
    """Squared image distance from each structure to each synthetic cryo-EM particle."""
    cfg, out = ctx["cfg"], ctx["output_directory"]
    rot_mats_image, ctfs, images = make_synthetic_images(
        top_image=ctx["reference_top"],
        traj_image=ctx["reference_traj"],
        outdir=out,
        n_pixel=cfg["n_pixel"],
        pixel_size=cfg["pixel_size"],
        sigma=cfg["sigma"],
        snr=cfg["snr"],
        n_image_per_struc=cfg["n_image_per_struc"],
        add_ctf=cfg["add_ctf"],
        defocus_min=cfg["defocus_min"],
        defocus_max=cfg["defocus_max"],
        device=ctx["device"],
        batch_size=cfg["batch_size"],
    )
    _plot_image_grid(images, os.path.join(out, "imgs_grid.png"))
    # Each particle is a projection of a randomly oriented copy of the reference, so the
    # structures must be brought into the same frame before any pixel comparison means
    # anything. align_traj writes the rotation matrices that calc_image_struc_distance reads.
    rot_mats_align = align_traj(
        top_image=ctx["reference_top"],
        traj_image=ctx["reference_traj"],
        top_struc=ctx["simulation_top"],
        traj_struc=ctx["simulation_traj"],
        outdir=os.path.join(out, ""),
        device=ctx["device"],
    )
    print("Shape of the rotation matrix: ", rot_mats_align.shape)
    # lambda is the pixel noise standard deviation implied by the requested SNR, and it sets
    # the width of the Gaussian likelihood. It is estimated from the particles themselves
    # rather than assumed, because the noise depends on the projected mass of the system.
    scale = approx_lmbd(
        top_struc=ctx["simulation_top"],
        traj_struc=ctx["simulation_traj"],
        n_pixel=cfg["n_pixel"],
        pixel_size=cfg["pixel_size"],
        sigma=cfg["sigma"],
        signal_to_noise_ratio=cfg["snr"],
        add_ctf=cfg.get("add_ctf", True),
        defocus_min=cfg["defocus_min"],
        defocus_max=cfg["defocus_max"],
        n_image_per_struc=cfg.get("n_image_per_struc", 1),
        n_batch=cfg.get("n_batch", 10),
        device=ctx["device"],
    )
    print("Approximated lambda at SNR %.1e: %.3e" % (cfg["snr"], scale))
    calc_image_struc_distance(
        images=images,
        ctfs=ctfs,
        rot_mats_image=rot_mats_image,
        top_struc=ctx["simulation_top"],
        traj_struc=ctx["simulation_traj"],
        rotmat_struc_imgstruc="%s/rot_mats_struc_image.npy" % out,
        outdir=out,
        n_pixel=cfg["n_pixel"],
        pixel_size=cfg["pixel_size"],
        sigma=cfg["sigma"],
        snr=cfg["snr"],
        add_ctf=cfg.get("add_ctf", True),
        defocus_min=cfg["defocus_min"],
        defocus_max=cfg["defocus_max"],
        batch_size=cfg["batch_size"],
    )
    # calc_image_struc_distance writes the matrix to disk rather than returning it.
    diff = np.load(
        out
        + "/diff_npix%d_ps%.2f_s%.1f_snr%.1E.npy"
        % (cfg["n_pixel"], cfg["pixel_size"], cfg["sigma"], cfg["snr"])
    )
    return diff, float(scale)


def squared_distance(a, b):
    """Squared Euclidean distance from every row of a to every row of b."""
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def cv(ctx):
    """Squared collective variable distance from each structure to each target frame."""
    cfg = ctx["cfg"]
    scale = float(cfg.get("cv_sigma", 0.6))
    struc_ref = md.load(ctx["simulation_top"])
    struc_traj = md.load(ctx["simulation_traj"], top=ctx["simulation_top"])
    cv_struct = np.asarray(cv_families.cv_of(struc_traj, struc_ref, cfg), dtype=float)
    cv_target = _cv_target(ctx, struc_ref)
    print(f"CV of structures shape: {cv_struct.shape}; CV of target shape: {cv_target.shape}")
    print(f"CV space bandwidth sigma_cv = {scale:.3f} A")
    return squared_distance(cv_struct, cv_target), scale


def _cv_target(ctx, struc_ref):
    """CV of the target basin, taken from the converged trajectory by an RMSD window."""
    cfg = ctx["cfg"]
    target_dcd = os.path.join(ctx["data_dir"], cfg.get("traj_file", "image.dcd"))
    lo = float(cfg.get("cv_target_rmsd_lo", 4.0))
    hi = float(cfg.get("cv_target_rmsd_hi", 6.0))
    if os.path.exists(target_dcd):
        target_traj = md.load(target_dcd, top=ctx["reference_top"])
        pool = np.asarray(cv_families.cv_of(target_traj, struc_ref, cfg), dtype=float)
        sel = (pool[:, 0] >= lo) & (pool[:, 0] <= hi)
        if np.any(sel):
            print(
                f"Target: {int(sel.sum())} frames of {os.path.basename(target_dcd)} "
                f"with CA-RMSD in [{lo}, {hi}] A"
            )
            return pool[sel]
    # The window is empty for a system whose target basin is defined elsewhere, in which
    # case the selection already written into reference.dcd is the target.
    target_traj = md.load(ctx["reference_traj"], top=ctx["reference_top"])
    print("Target: falling back to the selection in reference.dcd")
    return np.asarray(cv_families.cv_of(target_traj, struc_ref, cfg), dtype=float)


FAMILIES = {"cryoem": cryoem, "cv": cv}
# Axis labels for the diagnostic figures, which differ only in what an observation is.
LABELS = {"cryoem": ("Image index", "L2 distance"), "cv": ("Target index", "CV squared distance")}


def distance_of(ctx):
    """Squared distance matrix and noise scale for the likelihood family named in the configuration."""
    fam = ctx["cfg"].get("likelihood", "cryoem")
    if fam not in FAMILIES:
        raise ValueError(f"unknown likelihood {fam!r} (expected {list(FAMILIES)})")
    return FAMILIES[fam](ctx)


def labels_of(cfg):
    """Observation and distance axis labels for cfg["likelihood"]."""
    return LABELS[cfg.get("likelihood", "cryoem")]
