"""Merge WE segment trajectories and filter to a band of the CV."""

import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import mdtraj as md
import numpy as np
import argparse
import tarfile
import shutil
import json
import os

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


def extract_iteration(iter_num, traj_dir):
    itr_str = f"{iter_num:06d}"
    tar_path = os.path.join(traj_dir, f"{itr_str}.tar")
    # Handle tar extraction
    if os.path.isfile(tar_path):
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=traj_dir)
        # Primary extracted folder
        base = os.path.join(traj_dir, itr_str)
        # Case A is the direct layout
        if os.path.isdir(base):
            return base, True
        # Case B is nested under traj_segs inside the extracted directory
        nested = os.path.join(traj_dir, itr_str, "traj_segs", itr_str)
        if os.path.isdir(nested):
            return nested, True
        # Case C is an extraction that created the traj_segs root
        nested2 = os.path.join(traj_dir, "traj_segs", itr_str)
        if os.path.isdir(nested2):
            return nested2, True
        raise FileNotFoundError(f"After extracting {tar_path}, no valid iteration folder found.")
    # Handle existing folder
    folder = os.path.join(traj_dir, itr_str)
    if os.path.isdir(folder):
        return folder, False
    raise FileNotFoundError(f"Missing iteration data: {tar_path} or {folder}")


def load_iteration_trajs(iter_folder, topology):
    # Identify seg dirs root
    seg_root = (
        os.path.join(iter_folder, "traj_segs")
        if os.path.isdir(os.path.join(iter_folder, "traj_segs"))
        else iter_folder
    )
    seg_dirs = sorted(d for d in os.listdir(seg_root) if os.path.isdir(os.path.join(seg_root, d)))
    trajs = []
    for seg in seg_dirs:
        seg_dcd = os.path.join(seg_root, seg, "seg.dcd")
        if os.path.isfile(seg_dcd):
            trajs.append(md.load(seg_dcd, top=topology))
    return trajs


def main():
    # The reference topology filename and default iteration count come from CFG.
    default_topology = os.path.join("common_files", CFG.get("cv_reference_pdb", "bstate.pdb"))
    default_end_iter = int(CFG["n_iterations"])
    parser = argparse.ArgumentParser(description="Merge and analyze WE segment trajectories.")
    parser.add_argument("--traj_dir", default="traj_segs")
    parser.add_argument("--topology", default=default_topology)
    parser.add_argument("--start_iter", type=int, default=1)
    parser.add_argument("--end_iter", type=int, default=default_end_iter)
    parser.add_argument("--output", default="traj")
    parser.add_argument("--output_dir", default="merged_WE")
    # band width multiplier, with --std_left and --std_right as aliases
    parser.add_argument(
        "--std_mult",
        "--std_left",
        "--std_right",
        dest="std_mult",
        type=float,
        default=1.0,
        help="Number of std devs (band side set by cfg) from the mean to retain",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    topo = md.load(args.topology)
    all_iters = []
    total_segs = 0
    n_iters = args.end_iter - args.start_iter + 1
    # Iteration loop with progress
    for idx, i in enumerate(tqdm(range(args.start_iter, args.end_iter + 1), desc="Iterations"), 1):
        folder, extracted = extract_iteration(i, args.traj_dir)
        trajs = load_iteration_trajs(folder, topo)
        seg_count = len(trajs)
        total_segs += seg_count
        print(
            f"Iteration {i}: loaded {seg_count} segment trajectories."
            f" Cumulative segments: {total_segs}."
            f" ({idx}/{n_iters} = {idx/n_iters*100:.1f}% complete)"
        )
        if extracted:
            root = (
                os.path.dirname(folder)
                if "traj_segs" in folder and not folder.endswith(f"{i:06d}")
                else folder
            )
            shutil.rmtree(root)
            segs_dir = os.path.join(args.traj_dir, "traj_segs")
            if os.path.isdir(segs_dir):
                shutil.rmtree(segs_dir)
        all_iters.extend(trajs)
    if not all_iters:
        raise RuntimeError("No segment trajectories found across iterations.")
    # Merging
    print(f"\nMerging {len(all_iters)} segment trajectories into a single trajectory...")
    combined = md.join(all_iters)
    print(f"Merged trajectory contains {combined.n_frames} frames.")
    out_prefix = os.path.splitext(args.output)[0]
    all_path = os.path.join(output_dir, f"{out_prefix}_all.dcd")
    combined.save_dcd(all_path)
    print(f"Saved full trajectory to {all_path}")
    # Asymmetric Gaussian filtering on CV column 0 (theta_NMP for adk, RMSD for
    # chignolin/ntl9). The reference is the loaded topology and cv_of dispatches on
    # cfg["cv_family"] and reproduces the original inline CV math exactly.
    ref = md.load(args.topology)
    cv = cv_families.cv_of(combined, ref, CFG)
    coord = cv[:, 0]
    mu = np.mean(coord)
    sigma = np.std(coord)
    print(f"CV mean (mu): {mu:.4f}, std (sigma): {sigma:.4f}")
    i = args.std_mult
    # For band side "+" everything above the mean is kept plus a [mu - i*sigma, mu]
    # band below (adk, chignolin). "-"/left MIRRORS, so everything below the mean
    # plus a [mu, mu+i*sigma] band above (ntl9).
    sign = CFG.get("sigma_sign", "+")
    band_side = CFG.get("band_side", "right" if sign == "+" else "left")
    if band_side == "right":
        left_mask = (coord <= mu) & (coord >= mu - i * sigma)
        right_mask = coord > mu
    else:
        left_mask = coord < mu
        right_mask = (coord >= mu) & (coord <= mu + i * sigma)
    final_mask = left_mask | right_mask
    filtered = combined[final_mask]
    print(f"Filtered trajectory contains {filtered.n_frames} frames")
    out_name = os.path.join(output_dir, f"{out_prefix}.dcd")
    filtered.save_dcd(out_name)
    print(f"Saved filtered DCD to {out_name}")


if __name__ == "__main__":
    load_config()
    main()
