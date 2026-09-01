"""Three target runs side by side: weighted CA-RMSD ridgeline per run.

  HDF5_USE_FILE_LOCKING=FALSE python plot_three.py     # works mid-run

Reads run_4_6 / run_6_8 / run_8_10 and their targets, writes
plot.png.
"""
import os
import sys
import glob
import json
import numpy as np
import h5py
import mdtraj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

_bw = os.environ.get("PLOT_BW")          # KDE bandwidth override; default Scott's rule
BW = float(_bw) if _bw else None
OUTNAME = os.environ.get("PLOT_OUT", "plot.png")
DPI = int(os.environ.get("PLOT_DPI", "1000"))

HERE = os.path.dirname(os.path.abspath(__file__))
CF = os.path.join(HERE, "westpa_files", "common_files")
DATA = os.path.join(HERE, "westpa_files", "data")
sys.path.insert(0, CF)
import cv_families

cfg = json.load(open(os.path.join(CF, "seg_config.json")))
ref = mdtraj.load(os.path.join(CF, "folded.pdb"))
ca = ref.topology.select("name CA")
top = os.path.join(DATA, "topology.pdb")

RUNS = [                          # (work_dir, target_dcd, lo, hi, letter, color)
    ("run_4_6", "image_4_6.dcd", 4, 6, "a", "#6E9BC8"),
    ("run_5_7", "image_5_7.dcd", 5, 7, "b", "#E6A47C"),
]
xr = np.linspace(-0.5, 10.2, 240)    # start below 0 so the folded left tail isn't clipped
off = 1.2


def wdensity(vals, wt, xr):
    vals = np.asarray(vals, float)
    wt = np.asarray(wt, float)
    if len(vals) < 2 or vals.std() < 1e-6:
        m = np.average(vals, weights=wt) if wt.sum() else float(vals.mean())
        return np.exp(-0.5 * ((xr - m) / 0.15) ** 2)
    try:
        return gaussian_kde(vals, weights=wt, bw_method=BW)(xr)
    except Exception:
        return np.exp(-0.5 * ((xr - np.average(vals, weights=wt)) / 0.3) ** 2)


def run_dists(work):
    dists = []
    seed = os.path.join(work, "init", "seed.dcd")
    if os.path.exists(seed):
        r = mdtraj.rmsd(mdtraj.load(seed, top=top), ref, atom_indices=ca) * 10.0
        dists.append((0, r, np.ones(len(r))))
    for h5 in sorted(glob.glob(os.path.join(work, "run*", "west.h5")),
                     key=lambda p: int(os.path.basename(os.path.dirname(p))[3:])):
        n = int(os.path.basename(os.path.dirname(h5))[3:])
        if n > 12:           # cut to iteration 12 for both panels
            continue
        pc = w = None
        with h5py.File(h5, "r") as f:
            for k in sorted(f["iterations"].keys(), reverse=True):
                g = f["iterations"][k]
                cand = np.asarray(g["pcoord"][:, -1, 0], float)   # RMSD column
                ww = np.asarray(g["seg_index"][:]["weight"], float)
                if np.any(cand != 0) and ww.sum() > 0:
                    pc, w = cand, ww
                    break
        if pc is not None:
            dists.append((n, pc, w))
    return dists


all_dists = [run_dists(r[0]) for r in RUNS]
maxrows = max(len(d) for d in all_dists)

fig, axes = plt.subplots(1, len(RUNS), figsize=(5 * len(RUNS), 6.25), sharey=True)
for ax, dists, (work, image, lo, hi, letter, color) in zip(axes, all_dists, RUNS):
    for i, (n, vals, wt) in enumerate(dists):
        y = wdensity(vals, wt, xr)
        y = y / y.max() * 0.9 if y.max() > 0 else y
        base = i * off
        z = len(dists) - i
        ax.fill_between(xr, base, base + y, color=color, alpha=0.85, lw=0, zorder=z)
        ax.plot(xr, base + y, color="0.5", lw=0.8, zorder=z)
    cvimg = cv_families.cv_of(mdtraj.load(os.path.join(DATA, image), top=top), ref, cfg)
    sel = (cvimg[:, 0] >= lo) & (cvimg[:, 0] <= hi)
    tgt = cvimg[sel].mean(0) if sel.any() else cvimg.mean(0)
    bandtop = (maxrows - 1) * off + 1.25         # grey band rises above the ridges
    tr = cvimg[sel, 0] if sel.any() else cvimg[:, 0]   # band spans the target's full RMSD min..max
    ax.add_patch(Rectangle((tr.min(), 0), tr.max() - tr.min(), bandtop,
                           facecolor="0.88", edgecolor="none", zorder=0))
    ax.plot([tgt[0], tgt[0]], [0, bandtop], ls="--", color="k", lw=0.9, zorder=50, clip_on=False)
    ax.set_xlim(-0.35, 9.5)
    ax.set_xlabel(r"RMSD ($\mathrm{\AA}$)", fontsize=13, labelpad=1)
    ax.tick_params(labelsize=11)
    # "Initial / distribution" inline by each panel's bottom (folded) ridge.
    iv = dists[0][1]                       # iteration-0 (seed) RMSD values
    xt = float(iv.mean() + 2 * iv.std() + 0.3)   # just right of this panel's own seed ridge
    ax.text(xt, 0.50, "Initial\ndistribution", fontsize=8.0, ha="left", va="center",
            multialignment="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.065, 0.97, "%s)" % letter, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="bottom", ha="center", clip_on=False)   # aligned over the y-axis label
    # "Target distribution" as one line over the grey target band.
    ax.text((lo + hi) / 2.0, bandtop + 0.05, r"Target distribution [%d,%d] $\mathrm{\AA}$" % (lo, hi),
            ha="center", va="bottom", fontsize=8.0)

ymax = (maxrows - 1) * off + 1.65    # clears the "Target distribution" label above the band
axes[0].set_ylim(0, ymax)
for ax in axes:                            # iteration markers on every panel
    ax.set_yticks([i * off for i in range(maxrows)])
    ax.set_yticklabels([""] + [str(i) for i in range(1, maxrows)])
    ax.tick_params(labelleft=True)
    ax.set_ylabel("Iterations", fontsize=13, labelpad=1)
fig.tight_layout()
fig.savefig(OUTNAME, dpi=DPI)
print("wrote plot.png")
