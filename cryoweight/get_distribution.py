"""Select target frames for image generation and resample them by Boltzmann weight."""

from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm
import mdtraj as md
import numpy as np
import warnings
import json
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

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


def select_distribution(
    traj_file,
    topology_file,
    out_sel_dcd,
    out_boltz_dcd,
    kB,
    T,
    N_draw,
    nbins,
    sigma,
    x_range,
    y_range,
    hist_eps,
    select_mode,
    cv_cfg,
    x_lower=None,
    x_upper=None,
    x_thresh=None,
    y_lower=None,
    y_upper=None,
):
    """Select frames by select_mode with the CV from cv_families, then draw N_draw by Boltzmann weight."""
    beta = 1.0 / (kB * T)
    # Load trajectory. The reference structure is the topology itself (matches
    # the originals, md.load(topology_file) for rmsd_rg and the trajectory topology for adk).
    traj = md.load(traj_file, top=topology_file)
    reference = md.load(topology_file)
    if cv_cfg.get("cv_family") == "rmsd_rg":
        # rmsd_rg originals superpose traj onto the reference before CV.
        traj.superpose(reference)
        ca_idx = traj.topology.select(cv_cfg.get("cv_atom_selection", "name CA"))
        if ca_idx.size == 0:
            raise ValueError("No CA atoms found in topology.")
    # Route CV math through the shared registry. cv0 and cv1 replace the inline
    # (rmsd, rg) or (theta_nmp, theta_lid) columns of the originals.
    cv = cv_families.cv_of(traj, reference, cv_cfg)
    x = cv[:, 0]
    y = cv[:, 1]

    if select_mode == "angle_window":
        # For adk the window applies to both collective variables, theta_NMP and theta_LID.
        mask = (x >= x_lower) & (x <= x_upper) & (y >= y_lower) & (y <= y_upper)
        sel_traj = traj[mask]
        sel_traj.save_dcd(out_sel_dcd)
        print(
            f"Saved {sel_traj.n_frames} frames with θNMP ∈ [{x_lower}°, {x_upper}°] and θLID ∈ [{y_lower}°, {y_upper}°] to '{out_sel_dcd}'"
        )
        x_sel = x[mask]
        y_sel = y[mask]
        H, x_edges, y_edges = np.histogram2d(
            x_sel, y_sel, bins=nbins, range=[x_range, y_range], density=True
        )
        F = -kB * T * np.log(H + hist_eps)
        F -= F.min()
        F_smooth = gaussian_filter(F, sigma=sigma)
        ix = np.clip(np.digitize(x_sel, x_edges) - 1, 0, nbins[0] - 1)
        iy = np.clip(np.digitize(y_sel, y_edges) - 1, 0, nbins[1] - 1)
        F_i = F_smooth[ix, iy]
        w = np.exp(-beta * F_i)
        w /= w.sum()
        actual_N_draw = min(N_draw, len(sel_traj))
        if actual_N_draw < N_draw:
            print(
                f"[WARNING] Requested {N_draw} frames but only {len(sel_traj)} available. Using {actual_N_draw} frames."
            )
        chosen = np.random.choice(len(sel_traj), size=actual_N_draw, replace=False, p=w)
        boltz = sel_traj.slice(chosen)
        boltz.save_dcd(out_boltz_dcd)
        print(f"Saved {len(chosen)} Boltzmann-sampled frames to '{out_boltz_dcd}'")
        return

    if select_mode == "thresh":
        # For chignolin every frame with cv0 above the threshold is kept. The draw is clamped to what passes, as in
        # the angle_window and window branches below.
        mask_open = x >= x_thresh
        open_traj = traj[mask_open]
        open_traj.save_dcd(out_sel_dcd)
        print(f"Saved {open_traj.n_frames} frames with RMSD ≥ {x_thresh} Å to '{out_sel_dcd}'")
        r_open = x[mask_open]
        g_open = y[mask_open]
        H, r_edges, g_edges = np.histogram2d(
            r_open, g_open, bins=nbins, range=[x_range, y_range], density=True
        )
        F = -kB * T * np.log(H + hist_eps)
        F -= F.min()
        F_smooth = gaussian_filter(F, sigma=sigma)
        ir = np.clip(np.digitize(r_open, r_edges) - 1, 0, nbins[0] - 1)
        ig = np.clip(np.digitize(g_open, g_edges) - 1, 0, nbins[1] - 1)
        F_i = F_smooth[ir, ig]
        w = np.exp(-beta * F_i)
        w /= w.sum()
        actual_N_draw = min(N_draw, len(open_traj))
        if actual_N_draw < N_draw:
            print(
                f"[WARNING] Requested {N_draw} frames but only {len(open_traj)} available. Using {actual_N_draw} frames."
            )
        chosen = np.random.choice(len(open_traj), size=actual_N_draw, replace=False, p=w)
        boltz = open_traj.slice(chosen)
        boltz.save_dcd(out_boltz_dcd)
        print(f"Saved {len(chosen)} Boltzmann-sampled frames to '{out_boltz_dcd}'")
        return

    if select_mode in ("window", "ge", "le"):
        # The ntl9 dispatcher selects on cv0 alone, with guards, kept verbatim.
        if select_mode == "window":
            if x_lower is None and x_upper is None:
                raise ValueError("mode='window' requires rmsd_lower and/or rmsd_upper")
            mask = np.ones_like(x, dtype=bool)
            if x_lower is not None:
                mask &= x >= x_lower
            if x_upper is not None:
                mask &= x <= x_upper
            tag = f"[window] RMSD ∈ [{'-∞' if x_lower is None else x_lower}, {'+∞' if x_upper is None else x_upper}] Å"
        elif select_mode == "ge":
            if x_thresh is None:
                raise ValueError("mode='ge' requires rmsd_thresh")
            mask = x >= x_thresh
            tag = f"[ge] RMSD ≥ {x_thresh} Å"
        elif select_mode == "le":
            if x_thresh is None:
                raise ValueError("mode='le' requires rmsd_thresh")
            mask = x <= x_thresh
            tag = f"[le] RMSD ≤ {x_thresh} Å"
        if not np.any(mask):
            print(f"[WARN] No frames match selection {tag}; nothing written.")
            return
        sel_traj = traj[mask]
        if out_sel_dcd:
            sel_traj.save_dcd(out_sel_dcd)
            print(f"[select] {sel_traj.n_frames} frames saved → '{out_sel_dcd}'")
        r_sel = x[mask]
        g_sel = y[mask]
        H, r_edges, g_edges = np.histogram2d(
            r_sel, g_sel, bins=nbins, range=[x_range, y_range], density=True
        )
        F = -kB * T * np.log(H + hist_eps)
        F -= F.min()
        F_smooth = gaussian_filter(F, sigma=sigma)
        ir = np.clip(np.digitize(r_sel, r_edges) - 1, 0, nbins[0] - 1)
        ig = np.clip(np.digitize(g_sel, g_edges) - 1, 0, nbins[1] - 1)
        w = np.exp(-beta * F_smooth[ir, ig])
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            print("[WARN] Invalid weights; skipping Boltzmann resample.")
            return
        w /= w_sum
        n_draw = min(int(N_draw), sel_traj.n_frames)
        if n_draw > 0 and out_boltz_dcd:
            chosen = np.random.choice(sel_traj.n_frames, size=n_draw, replace=False, p=w)
            sel_traj.slice(chosen).save_dcd(out_boltz_dcd)
            print(f"[boltz] {n_draw} frames saved → '{out_boltz_dcd}'")
        else:
            print("[INFO] No Boltzmann sample written (n_draw=0 or no output path).")
        return

    raise ValueError("select_mode must be 'angle_window', 'thresh', 'window', 'ge', or 'le'")


def main():
    """Run this stage in the current directory."""

    # Resample frames from the converged trajectory (orchestration driven by CFG).
    select_distribution(
        traj_file=CFG["traj_file"],
        topology_file=CFG["topology_file"],
        out_sel_dcd=CFG["out_sel_dcd"],
        out_boltz_dcd=CFG["out_boltz_dcd"],
        kB=CFG["kB"],
        T=CFG["T"],
        N_draw=CFG["N_draw"],
        nbins=tuple(CFG["nbins"]),
        sigma=tuple(CFG.get("fes_sigma", (3, 1))),
        x_range=tuple(CFG["x_range"]),
        y_range=tuple(CFG["y_range"]),
        hist_eps=CFG["hist_eps"],
        select_mode=CFG["select_mode"],
        cv_cfg=CFG["cv_cfg"],
        x_lower=CFG.get("x_lower"),
        x_upper=CFG.get("x_upper"),
        x_thresh=CFG.get("x_thresh"),
        y_lower=CFG.get("y_lower"),
        y_upper=CFG.get("y_upper"),
    )


if __name__ == "__main__":
    load_config()
    main()
