"""Filter and renumber WE basis states by a band on the progress coordinate."""

import numpy as np
import argparse
import shutil
import json
import os

# Loaded on demand rather than at import, so that this module can be imported from
# anywhere and a test can supply its own configuration.
CFG = {}


def load_config(path="reweight_config.json"):
    """Read the per run configuration into the module global the stages below use."""
    global CFG
    with open(path) as handle:
        CFG = json.load(handle)
    return CFG


def restore_from_backup_if_exists(target_dir):
    backup_dir = f"{target_dir}_all"
    if os.path.isdir(target_dir) and os.path.isdir(backup_dir):
        print(f"Found both {target_dir} and {backup_dir}. Replacing {target_dir} with backup...")
        shutil.rmtree(target_dir)
        shutil.copytree(backup_dir, target_dir)
        print(f"Restored {target_dir} from {backup_dir}")
    elif os.path.isdir(target_dir) and not os.path.isdir(backup_dir):
        print(f"Creating backup: copying {target_dir} → {backup_dir}")
        shutil.copytree(target_dir, backup_dir)
        print(f"Backup created at {backup_dir}")
    elif not os.path.isdir(target_dir) and os.path.isdir(backup_dir):
        print(f"{target_dir} not found. Reconstructing it from {backup_dir}...")
        shutil.copytree(backup_dir, target_dir)
        print(f"Reconstructed {target_dir} from {backup_dir}")
    else:
        raise FileNotFoundError(f"Neither {target_dir} nor {backup_dir} found. Aborting.")


def filter_and_rename_bstates(target_dir):
    pcoord_path = os.path.join(target_dir, "pcoord.init")
    pcoord = np.loadtxt(pcoord_path)
    rmsd = pcoord[:, 0]
    mu, sigma = np.mean(rmsd), np.std(rmsd)
    print(f"RMSD μ = {mu:.4f}, σ = {sigma:.4f}")
    # Asymmetric filtering with the band side driven by cfg. Side "+" keeps everything
    # above the mean plus a [mu - sigma, mu] band below, as adk and chignolin do. Side "-"
    # mirrors that, everything below plus a [mu, mu + sigma] band above, as ntl9 does.
    sign = CFG.get("sigma_sign", "+")
    band_side = CFG.get("band_side", "right" if sign == "+" else "left")
    if band_side == "right":
        left_mask = (rmsd <= mu) & (rmsd >= mu - sigma)
        right_mask = rmsd > mu
    else:
        left_mask = rmsd < mu
        right_mask = (rmsd >= mu) & (rmsd <= mu + sigma)
    final_mask = left_mask | right_mask
    filtered_indices = np.where(final_mask)[0]
    print(f"Filtered {len(filtered_indices)} of {len(rmsd)} frames")
    pcoord_filtered = pcoord[filtered_indices]
    # Filter bstates
    with open(os.path.join(target_dir, "bstates.txt")) as f:
        bstates = f.readlines()
    bstates_filtered = [bstates[i] for i in filtered_indices]
    # Normalize probabilities
    rows = [line.strip().split() for line in bstates_filtered]
    probs = np.array([float(row[1]) for row in rows])
    norm_probs = probs / np.sum(probs)
    for i, row in enumerate(rows):
        row[1] = f"{norm_probs[i]:.32e}"
    bstates_normalized = [" ".join(row) + "\n" for row in rows]
    # Backup original folders
    folder_list = sorted([f for f in os.listdir(target_dir) if f.isdigit() and len(f) == 4])
    for folder in folder_list:
        src = os.path.join(target_dir, folder)
        dst = os.path.join(target_dir, f"{folder}_")
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"Backed up {folder} → {folder}_")
    # Rename sequentially
    backup_folders = sorted(
        [f for f in os.listdir(target_dir) if f.endswith("_") and f[:-1].isdigit()]
    )
    existing = set(f[:-1] for f in backup_folders)
    filtered_with_folders = [
        (i, idx) for i, idx in enumerate(filtered_indices) if f"{idx:04d}" in existing
    ]
    new_pcoord = []
    new_bstates = []
    new_bstates_norm = []
    for new_idx, (row_num, old_idx) in enumerate(filtered_with_folders):
        old_folder = os.path.join(target_dir, f"{old_idx:04d}_")
        new_folder = os.path.join(target_dir, f"{new_idx:04d}")
        shutil.move(old_folder, new_folder)
        print(f"Renamed {old_idx:04d}_ → {new_idx:04d}")
        new_pcoord.append(pcoord_filtered[row_num])
        new_bstates.append(bstates_filtered[row_num])
        new_bstates_norm.append(bstates_normalized[row_num])
    np.savetxt(os.path.join(target_dir, "pcoord_temp.init"), np.array(new_pcoord))
    with open(os.path.join(target_dir, "bstates_temp.txt"), "w") as f:
        f.writelines(new_bstates)
    with open(os.path.join(target_dir, "bstates_normalized.txt"), "w") as f:
        f.writelines(new_bstates_norm)
    # Sequential indices in column 1 and 3
    norm_path = os.path.join(target_dir, "bstates_normalized.txt")
    with open(norm_path) as f:
        lines = f.readlines()
    width = max(4, len(str(len(lines) - 1)))
    updated = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        new_id = f"{i:0{width}d}"
        updated.append(f"{new_id} {parts[1]} {new_id}\n")
    with open(norm_path, "w") as f:
        f.writelines(updated)
    # Delete unused backups
    leftovers = [f for f in os.listdir(target_dir) if f.endswith("_") and f[:-1].isdigit()]
    for f in leftovers:
        shutil.rmtree(os.path.join(target_dir, f))
        print(f"Deleted leftover {f}")
    # Cleanup and renaming
    os.remove(os.path.join(target_dir, "pcoord.init"))
    os.remove(os.path.join(target_dir, "bstates.txt"))
    os.remove(os.path.join(target_dir, "bstates_temp.txt"))
    shutil.move(
        os.path.join(target_dir, "pcoord_temp.init"), os.path.join(target_dir, "pcoord.init")
    )
    shutil.move(
        os.path.join(target_dir, "bstates_normalized.txt"), os.path.join(target_dir, "bstates.txt")
    )
    print("All files renamed and cleaned up successfully.")


def main():
    parser = argparse.ArgumentParser(description="Filter and rename bstates by RMSD.")
    parser.add_argument(
        "--dir",
        type=str,
        default="bstates",
        help="Directory containing bstates.txt and pcoord.init (default: bstates)",
    )
    args = parser.parse_args()
    restore_from_backup_if_exists(args.dir)
    filter_and_rename_bstates(args.dir)


if __name__ == "__main__":
    load_config()
    main()
