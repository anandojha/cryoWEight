"""Light merge of WESTPA segment trajectories into one ensemble DCD.

Dependency-light: numpy + mdtraj only. Walks
traj_segs/{iter:06d}/{seg:06d}/seg.dcd across the run's iterations and md.join()s
them into merged_WE/traj_all.dcd, which reweight.py then consumes as the ensemble
for the next iteration.
"""
import os
import glob
import argparse

import mdtraj as md


def main():
    ap = argparse.ArgumentParser(description="Merge WE seg.dcd files into one DCD.")
    ap.add_argument("--traj-dir", default="traj_segs",
                    help="WESTPA traj_segs directory.")
    ap.add_argument("--topology", required=True, help="Topology PDB for seg.dcd.")
    ap.add_argument("--out-dir", default="merged_WE", help="Output directory.")
    ap.add_argument("--out", default="traj_all.dcd", help="Merged DCD filename.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    top = md.load(args.topology)

    # seg.dcd lives at traj_segs/{iter:06d}/{seg:06d}/seg.dcd
    seg_paths = sorted(glob.glob(os.path.join(args.traj_dir, "*", "*", "seg.dcd")))
    if not seg_paths:
        raise RuntimeError(f"no seg.dcd found under {args.traj_dir}")

    trajs = []
    for p in seg_paths:
        try:
            trajs.append(md.load(p, top=top))
        except Exception as e:  # skip a truncated/empty segment rather than abort
            print(f"[merge] skipping {p}: {e}")
    if not trajs:
        raise RuntimeError("no loadable segment trajectories")

    combined = md.join(trajs)
    out_path = os.path.join(args.out_dir, args.out)
    combined.save_dcd(out_path)
    print(f"[merge] merged {len(trajs)} segments / {combined.n_frames} frames "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
