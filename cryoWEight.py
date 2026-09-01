#!/usr/bin/env python3
"""
WESTPA and cryo-EM iterative reweighting driver.

    python cryoWEight.py --system <name> init [--sigma-strategy permissive]
    python cryoWEight.py --system <name> iterate --range 2 4
    python cryoWEight.py --system <name>                # init then iterate
"""

from glob import glob
import mdtraj as md
import subprocess
import argparse
import shutil
import io
import time
import re
import os

import assemble

ROOT = os.getcwd()
DATA_DIR = "data"
SCRIPTS = "scripts"
WE_FILES = "WE_files"

# Resolved from the command line in __main__. They are declared here so that the module
# can be imported and its stages driven directly, rather than only through the entry point.
CONFIG_SYSTEM = None
CFG = {}
SIGMA_SIGN = "+"
SIGMA_STRATEGY = "permissive"
TOPOLOGY_EXPLICIT = ""
TOPOLOGY_STRIPPED = ""
SSH_HOST = ""
LOCAL = False
N_WORKERS = 4


def _resolve_sigma_sign(reweight_dir, cfg):
    """Which side of the prior the target lies on along the leading CV.

    The target region comes from the selection keys of the configuration, and the
    prior mean comes from the binned seeding ensemble that run0 wrote. The weight
    shift of the reweighting is not used, because a prior with no overlap with the
    target ranks its own structures in an arbitrary direction.
    """
    weights, bins = [], []
    with open(os.path.join(reweight_dir, "output", "bin_crds_wght.txt")) as handle:
        for line in handle:
            m = re.search(r"Bin (\d+),.*Weight: ([0-9.eE+-]+)", line)
            if m:
                bins.append(int(m.group(1)))
                weights.append(float(m.group(2)))
    total = sum(weights)
    mean_bin = sum(b * w for b, w in zip(bins, weights)) / total
    prior = float(cfg["bin_x_min"]) + (mean_bin + 0.5) * float(cfg["bin_width"])

    mode = cfg.get("select_mode", "thresh")
    if mode in ("thresh", "ge", "le"):
        target = float(cfg["x_thresh"])
    else:
        target = (float(cfg["x_lower"]) + float(cfg["x_upper"])) / 2.0
    sign = "+" if target >= prior else "-"
    print(
        f"sigma_sign auto: prior mean cv {prior:.2f}, target region at {target:.2f}, "
        f"resolved {sign!r}"
    )
    return sign


def _sigma_priority(sigma_strategy, sigma_sign):
    """Build the σ-coordinate priority list for the configured sign (+ or -).

    Mirrors the original per system literal lists, e.g. adk and chignolin permissive
    -> ['+3σ', '+2σ', '+1σ'] and ntl9 permissive -> ['-3σ', '-2σ', '-1σ'].
    """
    if sigma_strategy == "permissive":
        levels = [3, 2, 1]
    elif sigma_strategy == "moderate":
        levels = [2, 1]
    elif sigma_strategy == "strict":
        levels = [1]
    else:
        raise ValueError(f"Invalid SIGMA_STRATEGY: {sigma_strategy}")
    return [f"{sigma_sign}{n}σ" for n in levels]


def _select_bottleneck_coords(bottleneck_file, sigma_strategy, sigma_sign):
    """Coordinates of the best available sigma level in a bottleneck_coordinates.txt.

    A strategy names a preference order rather than one level, because a run does not
    always reach the outer ones. permissive tries 3, 2 then 1 sigma and takes whichever is
    present, strict accepts only 1.
    """
    with open(bottleneck_file) as handle:
        lines = handle.readlines()
    for sigma in _sigma_priority(sigma_strategy, sigma_sign):
        for line in lines:
            if sigma in line:
                x, y = line.split()[-2:]
                x, y = f"{float(x):.2f}", f"{float(y):.2f}"
                print(f"Using {sigma} coords: [{x}, {y}]")
                return x, y
    have = [line.split(chr(9))[1] for line in lines if chr(9) in line]
    raise RuntimeError(
        f"No {', '.join(_sigma_priority(sigma_strategy, sigma_sign))} line in {bottleneck_file} "
        f"using strategy '{sigma_strategy}'. "
        f"It holds {have}. The ensemble has no bottleneck on the {sigma_sign} side, which "
        f"means the weighted ensemble run did not spread far enough for this target."
    )


def _write_west_cfg_coords(cfg_path, x, y):
    """Replace the anchor coordinate line of a west.cfg, keeping its indentation."""
    lines = open(cfg_path).read().splitlines()
    for i, line in enumerate(lines):
        if "at:" in line and "[" in line:
            indent = re.match(r"^\s*", line).group(0)
            lines[i] = f"{indent}at: [{x}, {y}]"
            break
    else:
        raise RuntimeError(f"no 'at: [...]' line to patch in {cfg_path}")
    open(cfg_path, "w").write("\n".join(lines) + "\n")


def _stage_we_files(run_dir):
    """Copy the assembled WE_files and scripts tree for this system into run_dir.

    Replaces the original hardwired copytree of WE_files. assemble lays down
    shared/ plus rendered templates/ plus per system overrides, and the WE_files/
    members (and simtime.py) into run_dir exactly as the original loop placed them.
    """
    assemble.assemble(CONFIG_SYSTEM, run_dir)
    # The original driver copied these specific members of WE_files into the run dir.
    src_we = os.path.join(run_dir, WE_FILES)
    for item in [
        "common_files",
        "init.sh",
        "west.cfg",
        "env.sh",
        "run.sh",
        "submit_WE.sh",
        "westpa_scripts",
        "simtime.py",
    ]:
        src = os.path.join(src_we, item)
        dst = os.path.join(run_dir, item)
        if not os.path.exists(src):
            # simtime.py may be assembled at WE_files/simtime.py or at the top level.
            alt = os.path.join(run_dir, item)
            if os.path.exists(alt):
                continue
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy(src, dst)


STAGE_MODULES = (
    "cv_families.py",
    "likelihoods.py",
    "cryoER_core.py",
    "build_system.py",
    "reweight_config.json",
)


def _stage_script(script, dst):
    """Copy a stage script and the modules it imports into dst, returning what was added.

    Each stage runs as its own process with dst as the working directory, so its imports
    and its config have to sit beside it. Only files this call creates are returned, so a
    caller cleaning up never removes something an earlier stage put there.
    """
    added = []
    for s in (script,) + STAGE_MODULES:
        src, dest = os.path.join(SCRIPTS, s), os.path.join(dst, s)
        if os.path.exists(src) and not os.path.exists(dest):
            shutil.copy(src, dest)
            added.append(dest)
    return added


def _stage_reweight_scripts(dst):
    """Copy the reweight script and everything it imports into a reweight run directory.

    The script runs as its own process in that directory, so its imports and its config
    have to sit beside it rather than on any path.
    """
    for s in (
        "reweight.py",
        "cryoER_core.py",
        "cv_families.py",
        "likelihoods.py",
        "build_system.py",
        "reweight_config.json",
    ):
        src = os.path.join(SCRIPTS, s)
        if os.path.exists(src):
            shutil.copy(src, dst)


def _resolve_run_placeholder(run_dir, run_name):
    """Point the configuration for this run at its own WE directory.

    The system YAML writes paths such as ../runx/merged_WE so that one config serves every
    iteration. The placeholder lives in reweight_config.json, not in the script, so this
    rewrites the config the staged script is about to read.
    """
    path = os.path.join(run_dir, "reweight_config.json")
    text = io.open(path, encoding="utf-8").read()
    if "runx" not in text:
        raise RuntimeError(f"no runx placeholder found in {path}")
    io.open(path, "w", encoding="utf-8").write(text.replace("runx", run_name))


def _run_westpa(run_dir, tag):
    """Initialize (w_init via init.sh) and run one WESTPA weighted ensemble run.
    --local runs w_run directly here, otherwise sbatch on SSH_HOST and poll squeue."""
    subprocess.run(["bash", "init.sh"], cwd=run_dir, check=True)
    if LOCAL:
        subprocess.run(
            [
                "bash",
                "-c",
                f"source env.sh && w_run --n-workers {N_WORKERS} --work-manager processes",
            ],
            cwd=run_dir,
            check=True,
        )
        print(f"{tag}: local w_run finished")
        return
    submit = subprocess.run(
        ["ssh", SSH_HOST, f"cd {os.path.abspath(run_dir)} && sbatch submit_WE.sh"],
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = re.search(r"Submitted batch job (\d+)", submit.stdout).group(1)
    print(f"{tag}: submitted job {job_id}")
    misses = 0
    while True:
        s = subprocess.run(
            ["ssh", SSH_HOST, "squeue", "-j", job_id], capture_output=True, text=True
        )
        if s.returncode != 0:
            misses += 1
            if misses > 5:
                raise RuntimeError(f"squeue failed {misses} times in a row: {s.stderr.strip()}")
            time.sleep(60)
            continue
        misses = 0
        if job_id not in s.stdout:
            print(f"{tag}: job {job_id} finished")
            break
        time.sleep(60)


def run_init_reweight_simulation():
    # A) Cryo-EM target selection (skipped for CV signal systems that build the
    #    target directly from data/image.dcd in reweight.py).
    if CFG.get("run_get_distribution", True):
        staged = _stage_script("get_distribution.py", DATA_DIR)
        try:
            subprocess.run(["python", "get_distribution.py"], cwd=DATA_DIR, check=True)
        finally:
            for path in staged:
                os.remove(path)
    # B) Prepare and execute reweight_run0/
    R0_DIR = "reweight_run0"
    if os.path.isdir(R0_DIR):
        shutil.rmtree(R0_DIR)
    os.makedirs(R0_DIR)
    shutil.copytree(DATA_DIR, os.path.join(R0_DIR, DATA_DIR))
    _stage_reweight_scripts(R0_DIR)
    subprocess.run(["python", "reweight.py", "--run0"], cwd=R0_DIR, check=True)
    shutil.rmtree(os.path.join(R0_DIR, DATA_DIR))
    if os.path.isdir(os.path.join(R0_DIR, "__pycache__")):
        shutil.rmtree(os.path.join(R0_DIR, "__pycache__"))
    # C) Set up run1 for WESTPA
    RUN1 = "run1"
    if os.path.isdir(RUN1):
        shutil.rmtree(RUN1)
    os.makedirs(RUN1)
    shutil.copytree(os.path.join(R0_DIR, "bstates"), os.path.join(RUN1, "bstates"))
    # Copy the assembled WE_files and scripts tree for this system (system agnostic).
    _stage_we_files(RUN1)
    global SIGMA_SIGN
    if SIGMA_SIGN == "auto":
        import json

        rc = json.load(open(os.path.join(SCRIPTS, "reweight_config.json")))
        SIGMA_SIGN = _resolve_sigma_sign(R0_DIR, rc)
        # The staged runtime scripts read the sign from their config, so the resolved
        # value is written back into the assembled tree.
        rc_path = os.path.join(SCRIPTS, "reweight_config.json")
        rc["sigma_sign"] = SIGMA_SIGN
        with open(rc_path, "w") as handle:
            json.dump(rc, handle, indent=2)
    x, y = _select_bottleneck_coords(
        os.path.join(R0_DIR, "output", "bottleneck_coordinates.txt"), SIGMA_STRATEGY, SIGMA_SIGN
    )
    _write_west_cfg_coords(os.path.join(RUN1, "west.cfg"), x, y)
    # D) Initialize + run the WESTPA weighted ensemble run (local or cluster)
    _run_westpa(RUN1, "run1")
    # E) Merge trajectories in run1/
    staged = _stage_script("merge.py", RUN1)
    try:
        subprocess.run(["python", "merge.py"], cwd=RUN1, check=True)
    finally:
        for path in staged:
            os.remove(path)
    # F) Plot free energy in merged_WE
    MERGED = os.path.join(RUN1, "merged_WE")
    for topo in (TOPOLOGY_EXPLICIT, TOPOLOGY_STRIPPED):
        shutil.copy(os.path.join(DATA_DIR, topo), os.path.join(MERGED, topo))
    staged = _stage_script("plot_free_energy.py", MERGED)
    try:
        subprocess.run(["python", "plot_free_energy.py"], cwd=MERGED, check=True)
    finally:
        for path in staged:
            os.remove(path)
    # G) Prepare and execute reweight_run1/
    R1_DIR = "reweight_run1"
    if os.path.isdir(R1_DIR):
        shutil.rmtree(R1_DIR)
    os.makedirs(R1_DIR)
    shutil.copytree(DATA_DIR, os.path.join(R1_DIR, DATA_DIR))
    _stage_reweight_scripts(R1_DIR)
    _resolve_run_placeholder(R1_DIR, "run1")
    subprocess.run(["python", "reweight.py"], cwd=R1_DIR, check=True)
    shutil.rmtree(os.path.join(R1_DIR, DATA_DIR))
    staged = _stage_script("sample_bstates.py", R1_DIR)
    try:
        subprocess.run(["python", "sample_bstates.py"], cwd=R1_DIR, check=True)
    finally:
        for path in staged:
            os.remove(path)
    if os.path.isdir(os.path.join(R1_DIR, "__pycache__")):
        shutil.rmtree(os.path.join(R1_DIR, "__pycache__"))
    print("Initial reweighting complete.")


def run_iterative_reweight_simulation(N):
    global SIGMA_SIGN
    if SIGMA_SIGN == "auto":
        # init resolved the sign and wrote it into the assembled tree, and iterate is
        # its own process, so the resolved value is read back from there.
        import json

        rc = json.load(open(os.path.join(SCRIPTS, "reweight_config.json")))
        if rc.get("sigma_sign") in ("+", "-"):
            SIGMA_SIGN = rc["sigma_sign"]
            print(f"sigma_sign auto: using {SIGMA_SIGN!r} resolved at init")
        else:
            raise SystemExit("sigma_sign auto is resolved by init, run init before iterate")
    run_dir = f"run{N}"
    prev_re = f"reweight_run{N-1}"
    re_dir = f"reweight_run{N}"
    merged_dir = os.path.join(run_dir, "merged_WE")
    # A) Set up WestPA runN
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    # Copy bstates from the previous reweight
    shutil.copytree(os.path.join(prev_re, "bstates"), os.path.join(run_dir, "bstates"))
    # Copy the assembled WE_files and scripts tree for this system (system agnostic).
    _stage_we_files(run_dir)
    x, y = _select_bottleneck_coords(
        os.path.join(prev_re, "output", "bottleneck_coordinates.txt"), SIGMA_STRATEGY, SIGMA_SIGN
    )
    _write_west_cfg_coords(os.path.join(run_dir, "west.cfg"), x, y)
    # B) Initialize + run the WESTPA weighted ensemble run (local or cluster)
    _run_westpa(run_dir, f"run{N}")
    # C) Merge segment trajectories for the current WE run
    staged = _stage_script("merge.py", run_dir)
    try:
        subprocess.run(["python", "merge.py"], cwd=run_dir, check=True)
    finally:
        for path in staged:
            os.remove(path)
    prev_run = f"run{N-1}"
    for topo in (TOPOLOGY_EXPLICIT, TOPOLOGY_STRIPPED):
        shutil.copy(os.path.join(prev_run, "merged_WE", topo), os.path.join(merged_dir, topo))
    # D) Plot free energy
    staged = _stage_script("plot_free_energy.py", merged_dir)
    try:
        subprocess.run(["python", "plot_free_energy.py"], cwd=merged_dir, check=True)
    finally:
        for path in staged:
            os.remove(path)
    # E) Reweight → reweight_runN
    if os.path.isdir(re_dir):
        shutil.rmtree(re_dir)
    os.makedirs(re_dir)
    shutil.copytree(DATA_DIR, os.path.join(re_dir, DATA_DIR))
    _stage_reweight_scripts(re_dir)
    _resolve_run_placeholder(re_dir, f"run{N}")
    subprocess.run(["python", "reweight.py"], cwd=re_dir, check=True)
    shutil.rmtree(os.path.join(re_dir, DATA_DIR))
    staged = _stage_script("sample_bstates.py", re_dir)
    try:
        subprocess.run(["python", "sample_bstates.py"], cwd=re_dir, check=True)
    finally:
        for path in staged:
            os.remove(path)
    if os.path.isdir(os.path.join(re_dir, "__pycache__")):
        shutil.rmtree(os.path.join(re_dir, "__pycache__"))
    print(f"run{N}: reweight complete")


def _binned_distribution(reweight_dir):
    """The reweighted probability per occupied CV bin, from mergd_bins_wght_rescld.txt.

    The bin grid is fixed by the configuration, so bin indices are comparable across
    iterations and the distributions of consecutive iterations share a common support.
    """
    import ast

    path = os.path.join(reweight_dir, "output", "mergd_bins_wght_rescld.txt")
    dist = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = ast.literal_eval(re.sub(r"np\.\w+\(([^()]*)\)", r"\1", line))
            key = tuple(row["Bin"])
            dist[key] = dist.get(key, 0.0) + sum(row["Weights"])
    total = sum(dist.values())
    return {b: w / total for b, w in dist.items()}


def _kl_divergence(curr_dir, prev_dir, eps=1e-12):
    """Kullback Leibler divergence D(P_curr || P_prev) over the union of occupied bins."""
    import math

    p, q = _binned_distribution(curr_dir), _binned_distribution(prev_dir)
    support = set(p) | set(q)
    zp = sum(p.get(b, 0.0) + eps for b in support)
    zq = sum(q.get(b, 0.0) + eps for b in support)
    kl = 0.0
    for b in support:
        pb = (p.get(b, 0.0) + eps) / zp
        qb = (q.get(b, 0.0) + eps) / zq
        kl += pb * math.log(pb / qb)
    return kl


def _iterate_until_converged():
    """Run iterations until the KL divergence between consecutive reweighted
    distributions falls below the configured threshold, the criterion of the paper."""
    try:
        threshold = float(CFG["kl_threshold"])
        max_iterations = int(CFG["max_iterations"])
    except KeyError as missing:
        raise SystemExit(
            f"--until-converged needs a convergence block in the system config.xml "
            f"(kl_threshold, max_iterations); {missing} is not set."
        )
    for n in range(2, max_iterations + 1):
        run_iterative_reweight_simulation(n)
        kl = _kl_divergence(f"reweight_run{n}", f"reweight_run{n - 1}")
        print(f"iteration {n}: KL vs iteration {n - 1} = {kl:.4f} (threshold {threshold})")
        if kl < threshold:
            print(f"converged at iteration {n}")
            return
    print(f"not converged within max_iterations {max_iterations}")


def _expand_runs(args) -> list[int]:
    """Build a sorted, unique list of run indices from --range/--runs."""
    runs = set()
    if getattr(args, "range", None):
        a, b = args.range
        if a > b:
            a, b = b, a
        runs.update(range(a, b + 1))
    if getattr(args, "runs", None):
        # Accept "2-4,5-7 9" or "2 3 4"
        tokens = args.runs.replace(",", " ").split()
        for tok in tokens:
            if "-" in tok:
                a, b = map(int, tok.split("-", 1))
                if a > b:
                    a, b = b, a
                runs.update(range(a, b + 1))
            else:
                runs.add(int(tok))
    if not runs:
        raise SystemExit("No runs specified. Use --range START END and/or --runs.")
    return sorted(runs)


def main(argv=None):
    """Parse the command line, resolve the settings for this system, and run the modes requested."""
    global CONFIG_SYSTEM, CFG, SIGMA_SIGN, TOPOLOGY_EXPLICIT, TOPOLOGY_STRIPPED
    global SSH_HOST, LOCAL, N_WORKERS, SIGMA_STRATEGY
    parser = argparse.ArgumentParser(description="cryoWEight: init or iterative WE + reweighting")
    parser.add_argument(
        "--system",
        default=os.environ.get("CRYOWEIGHT_SYSTEM"),
        help="System name under systems/, or a path such as examples/chignolin.",
    )
    parser.add_argument(
        "--sigma-strategy",
        choices=["permissive", "moderate", "strict"],
        default=None,
        help="SIGMA_STRATEGY (default: permissive). Used in auto mode only.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Iterative range in auto mode (default: 2 4).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run WESTPA locally (w_run) instead of sbatch on ssh_host. Put before the subcommand.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4, help="w_run workers for --local (default 4)."
    )
    sub = parser.add_subparsers(dest="cmd")
    # init
    p_init = sub.add_parser("init", help="Run initial reweighting and set up run1")
    p_init.add_argument(
        "--sigma-strategy",
        choices=["permissive", "moderate", "strict"],
        help="SIGMA_STRATEGY for bottleneck coords",
    )
    p_iter = sub.add_parser("iterate", help="Run iterative reweighting for selected runs")
    p_iter.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive range, e.g. --range 2 4",
    )
    p_iter.add_argument("--runs", type=str, help="Comma/space list and/or ranges, e.g. '2-4,5-7 9'")
    p_iter.add_argument(
        "--until-converged",
        action="store_true",
        help="Iterate until the KL divergence drops below convergence.kl_threshold",
    )
    p_iter.add_argument(
        "--sigma-strategy",
        choices=["permissive", "moderate", "strict"],
        help="SIGMA_STRATEGY for bottleneck coords",
    )
    args = parser.parse_args(argv)

    # Resolve the system and load its config (same flatten/load logic as the assembler).
    if not args.system:
        raise SystemExit("No system specified. Use --system <name> or set $CRYOWEIGHT_SYSTEM.")
    CONFIG_SYSTEM = args.system
    CFG = assemble.load_cfg(CONFIG_SYSTEM)
    # Once per system literals, now read from the config
    SIGMA_SIGN = CFG["sigma_sign"]
    TOPOLOGY_EXPLICIT = CFG["topology_explicit"]
    TOPOLOGY_STRIPPED = CFG["topology_stripped"]
    SSH_HOST = CFG["ssh_host"]
    LOCAL = bool(args.local)
    N_WORKERS = args.n_workers

    # SIGMA_STRATEGY resolved once, command line over environment over fallback.
    SIGMA_STRATEGY = (
        getattr(args, "sigma_strategy", None)
        or os.environ.get("CRYOWEIGHT_SIGMA_STRATEGY")
        or "permissive"
    )

    if args.cmd == "init":
        run_init_reweight_simulation()
    elif args.cmd == "iterate":
        if getattr(args, "until_converged", False):
            _iterate_until_converged()
            return
        for N in _expand_runs(args):
            run_iterative_reweight_simulation(N)
    else:
        # Auto mode with no subcommand runs init then iterate 2 to 4, or the given --range
        a, b = args.range if args.range else (2, 4)
        print(
            f"Auto mode: init + iterate {a}–{b}, system={CONFIG_SYSTEM}, sigma-strategy={SIGMA_STRATEGY}"
        )
        run_init_reweight_simulation()
        for N in range(a, b + 1):
            run_iterative_reweight_simulation(N)


if __name__ == "__main__":
    main()


"""
# Initial run (creates run1 and reweight_run1)
python cryoWEight.py --system adk init --sigma-strategy permissive/moderate/strict
# Iterative runs (example: runs 2 to 4)
python cryoWEight.py --system ntl9 iterate --range 2 4 --sigma-strategy permissive/moderate/strict
# Iterative runs (example: runs 2–4, 6, and 8)
python cryoWEight.py --system chignolin iterate --runs "2-4,6 8" --sigma-strategy permissive/moderate/strict
"""
