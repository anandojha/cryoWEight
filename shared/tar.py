"""Archive selected WE iterations; config from reweight_config.json."""

import subprocess
import shutil
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    "reweight_config.json",
    os.path.join(_HERE, "reweight_config.json"),
    os.path.join(_HERE, "scripts", "reweight_config.json"),
):
    if os.path.isfile(_cand):
        CFG = json.load(open(_cand))
        break
else:
    raise FileNotFoundError("reweight_config.json not found next to tar.py or in scripts/")


def tar_iterations(pwd_path, iterations, delete_after=False):
    """Tar the named iteration directories under traj_segs and verify each archive."""
    print(f"\n  Working directory: {pwd_path}")
    print(f"  Iterations to tar: {iterations}")
    print(f"  Delete after: {delete_after}")

    traj_dir = os.path.join(pwd_path, "traj_segs")
    if not os.path.isdir(traj_dir):
        print(f"  ERROR: traj_segs/ not found at {traj_dir}")
        return

    tarred = []

    for it in iterations:
        folder = f"{int(it):06d}"
        tar_file = os.path.join(pwd_path, "traj_segs", f"{folder}.tar")
        folder_path = os.path.join(pwd_path, "traj_segs", folder)

        print(f"\n  --- Iteration {folder} ---")

        # Skip if tar already exists
        if os.path.exists(tar_file):
            tar_size = os.path.getsize(tar_file) / (1024 * 1024)
            print(f"  SKIP: {folder}.tar already exists ({tar_size:.1f} MB)")
            continue
        if not os.path.isdir(folder_path):
            print(f"  SKIP: folder {folder}/ does not exist")
            continue

        # Count files in folder
        file_count = sum(len(files) for _, _, files in os.walk(folder_path))
        du_result = subprocess.run(["du", "-sm", folder_path], capture_output=True, text=True)
        folder_size = du_result.stdout.split()[0] if du_result.returncode == 0 else "?"
        print(f"  Found {file_count} files (~{folder_size} MB) in {folder}/")

        # Tar from pwd_path so structure is traj_segs/XXXXXX/...
        print(f"  Tarring {folder}...")
        result = subprocess.run(
            ["tar", "-cvf", f"traj_segs/{folder}.tar", f"traj_segs/{folder}"],
            cwd=pwd_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR tarring {folder}: {result.stderr}")
            continue

        tar_size = os.path.getsize(tar_file) / (1024 * 1024)
        print(f"  Created {folder}.tar ({tar_size:.1f} MB)")

        # Verify
        print(f"  Verifying {folder}.tar...")
        verify = subprocess.run(
            ["tar", "-tf", f"traj_segs/{folder}.tar"], cwd=pwd_path, capture_output=True, text=True
        )
        first_line = verify.stdout.split("\n")[0]
        num_entries = len(verify.stdout.strip().split("\n"))
        if first_line.startswith(f"traj_segs/{folder}"):
            print(f"  OK: {folder}.tar verified - {num_entries} entries, starts with {first_line}")
            tarred.append(folder_path)
        else:
            print(f"  WARNING: {folder}.tar structure unexpected: {first_line}")

    print(f"\n  Summary: {len(tarred)}/{len(iterations)} successfully tarred")

    if delete_after and tarred:
        print(f"\n  Deleting {len(tarred)} verified folders...")
        for path in tarred:
            shutil.rmtree(path)
            print(f"  DELETED: {path}")
        print(f"  Cleanup complete!")
    elif delete_after and not tarred:
        print(f"\n  Nothing to delete.")


base = CFG["tar_base"]
for i in range(1, CFG["tar_n_runs"] + 1):
    print(f"\n{'='*60}")
    print(f"Processing run{i}")
    print(f"{'='*60}")
    tar_iterations(f"{base}/run{i}", CFG["tar_iterations"], delete_after=True)
