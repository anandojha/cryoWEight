"""Plot the 2D free energy surface over the CV."""

import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

# Imported relatively when this file is used as part of the installed package, and
# by bare name when assemble has copied it into a run directory, where a compute
# node executes it as a loose script with no package around it.
try:
    from . import cv_families
except ImportError:
    import cv_families

# Loaded on demand rather than at import, so that this module can be imported from
# anywhere and a test can supply its own configuration.
CFG = {}


def load_config(path="reweight_config.json"):
    """Read the per run configuration into the module global the stages below use."""
    global CFG
    with open(path) as handle:
        CFG = json.load(handle)
    return CFG


def plot_free_energy(trajectory_file, topology_file, kB, T, nbins, output_file):
    """Plot the weighted free energy surface of a trajectory over the two CVs."""
    # Load trajectory and reference (reference supplies the topology/atom selection
    # for cv_of and is the superposition target for the rmsd_rg family).
    traj = md.load(trajectory_file, top=topology_file)
    reference = md.load(topology_file)
    traj.superpose(reference)
    # Collective variables via the shared CV registry (cfg.cv_family picks
    # rmsd_rg gives CA RMSD and Rg in Angstrom, adk_angles gives θNMP and θLID in degrees).
    cv = cv_families.cv_of(traj, reference, CFG)
    cv0 = cv[:, 0]
    cv1 = cv[:, 1]
    x_range = (CFG["xmin"], CFG["xmax"])
    y_range = (CFG["ymin"], CFG["ymax"])
    hist, xedges, yedges = np.histogram2d(
        cv0, cv1, bins=nbins, range=[x_range, y_range], density=True
    )
    # Free energy
    F = -kB * T * np.log(hist + 1e-12)
    F -= F.min()
    F_smooth = gaussian_filter(F, sigma=(3, 1))
    empty_mask = hist < 1e-8
    cutoff = np.percentile(F_smooth, 98)
    tail_mask = F_smooth > cutoff
    full_mask = empty_mask | tail_mask
    F_masked = np.ma.array(F_smooth, mask=full_mask)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis
    cmap.set_bad("white")
    cmap.set_under("white")
    mesh = ax.pcolormesh(
        xedges, yedges, F_masked.T, cmap=cmap, shading="auto", vmin=0, vmax=cutoff * 0.5
    )
    cbar = fig.colorbar(mesh, ax=ax, extend="neither")
    cbar.set_label("Free Energy (kJ/mol)", fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    levels = np.linspace(0, cutoff, 10)
    ax.contour(xedges[:-1], yedges[:-1], F_smooth.T, levels=levels, colors="black", linewidths=0.5)
    ax.set_xlabel(CFG["xlabel"], fontsize=16)
    ax.set_ylabel(CFG["ylabel"], fontsize=16)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    if CFG.get("xticks") is not None:
        ax.set_xticks(np.arange(*CFG["xticks"]))
    if CFG.get("yticks") is not None:
        ax.set_yticks(np.arange(*CFG["yticks"]))
    ax.tick_params(axis="both", which="major", labelsize=12, direction="out", length=4, width=1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    # plt.show()
    plt.close()


def main():
    """Run this stage in the current directory."""

    # The merged trajectory carries the atoms the WE segments propagated, so it is read
    # with topology_we. topology_file is the stripped analysis topology and would not match.
    plot_free_energy(
        trajectory_file=CFG["trajectory_file"],
        topology_file=CFG.get("topology_we", CFG["topology_file"]),
        kB=CFG["kB"],
        T=CFG["T"],
        nbins=tuple(CFG["nbins"]),
        output_file=CFG["output_file"],
    )


if __name__ == "__main__":
    load_config()
    main()
