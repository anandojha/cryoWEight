"""End to end run of the reweighting pipeline on model data.

    python tests/test_end_to_end.py

Roughly twenty seconds, against the three second unit suite in test_cryoweight.py, because
it builds an OpenMM system for every basis state. It is worth that, since the runx placeholder
bug survived because nothing ever ran the generalised pipeline from one end to the other,
and this is what would have caught it.

The data is generated, not production. tests/testdata/make_model_traj.py runs a short
implicit solvent trajectory from the chignolin structure committed in systems/, at raised
temperature so the frames spread rather than sitting on top of each other. A synthetic
CA only chain was tried first and rejected, because the pipeline builds an OpenMM system from each
basis state, and no force field has a template for a residue that is one carbon.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTDATA = os.path.join(HERE, "testdata")
SCRIPTS = os.path.join(ROOT, "cryoweight")
sys.path.insert(0, ROOT)
from cryoweight import configio

# The model trajectory spans RMSD 0.75 to 7.53 A and Rg 5.01 to 7.98 A, so every
# coordinate range, bin edge and axis limit has to move off the chignolin numbers.
OVERRIDES = {
    "topology_analysis": "model_chignolin.pdb",
    "topology_we": "model_chignolin.pdb",
    "topology_file": "model_chignolin.pdb",
    "init_md_dir": "../init_MD",
    "init_md_dcd": "model_chignolin.dcd",
    "init_md_max_frames": 40,
    "traj_file": "model_target.dcd",
    "out_boltz_dcd": "image_sel.dcd",
    "likelihood": "cv",
    "cv_sigma": 1.0,
    "cv_target_rmsd_lo": 5.5,
    "cv_target_rmsd_hi": 9.0,
    "bin_x_min": 0,
    "bin_x_max": 8,
    "bin_y_min": 4,
    "bin_y_max": 9,
    "bin_width": 1.0,
    "xmin": 0,
    "xmax": 8,
    "ymin": 4,
    "ymax": 9,
    "x_range": [0.0, 8.0],
    "y_range": [4.0, 9.0],
    "heatmap_xticks": [0, 8, 2],
    "heatmap_yticks": [4, 9, 1],
    "cluster_axis_limit": 9,
    "fe_nbins": [20, 20],
    "nbins": [20, 20],
    "bottleneck_nbins": [40, 40],
    "step": 2,
    "run_plot_free_energy": False,
    "em_iterations": 200,
}


def build_run(dest):
    """Lay out a reweight_run0 directory the way cryoWEight.py stages one."""
    init_md = os.path.join(dest, "init_MD")
    run0 = os.path.join(dest, "reweight_run0")
    data = os.path.join(run0, "data")
    for d in (init_md, data):
        os.makedirs(d)
    for name, target in (
        ("model_chignolin.pdb", init_md),
        ("model_chignolin.dcd", init_md),
        ("model_chignolin.pdb", data),
        ("model_target.dcd", data),
    ):
        shutil.copy(os.path.join(TESTDATA, name), os.path.join(target, name))
    # plot_overlap draws the selected target beside the seeding ensemble.
    shutil.copy(os.path.join(TESTDATA, "model_target.dcd"), os.path.join(data, "image_sel.dcd"))
    for name in (
        "reweight.py",
        "cryoER_core.py",
        "cv_families.py",
        "likelihoods.py",
        "build_system.py",
    ):
        src = os.path.join(SCRIPTS, name)
        if os.path.exists(src):
            shutil.copy(src, run0)
    cfg = dict(
        configio.read_xml(os.path.join(ROOT, "systems", "chignolin_cv.xml"))["reweight_config"]
    )
    cfg.update(OVERRIDES)
    with open(os.path.join(run0, "reweight_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    return run0


def test_the_seeding_iteration_runs_from_one_end_to_the_other():
    """Every stage, from the seeding trajectory through to the basis states for WESTPA.

    main() is called in process rather than through a subprocess so that a coverage run
    can see which lines of the pipeline it reaches. The subprocess form is what the driver
    actually does, and test_cryoweight.py asserts the driver still invokes it that way.
    """
    sys.path.insert(0, SCRIPTS)
    import reweight

    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        run0 = build_run(tmp)
        os.chdir(run0)
        reweight.load_config()
        reweight.main(run0=True)
        os.chdir(cwd)

        weights = np.loadtxt(os.path.join(run0, "output", "rescaled_weights_all.txt"))
        assert weights.ndim == 1 and len(weights) > 1
        assert abs(weights.sum() - 1.0) < 1e-6
        assert bool(np.all(weights >= 0.0))

        # bstates.txt is what WESTPA reads to start a run, one line per basis state.
        with open(os.path.join(run0, "bstates", "bstates.txt")) as fh:
            rows = [l for l in fh if l.strip() and not l.startswith("#")]
        assert len(rows) == len(weights)

        # The sigma levels are what cryoWEight.py patches into west.cfg.
        with open(os.path.join(run0, "output", "bottleneck_coordinates.txt")) as fh:
            bottleneck = fh.read()
        assert "Maximum Sampling" in bottleneck
        assert "σ" in bottleneck

        for produced in (
            "output/pcoord_plot.png",
            "bstates/pcoord.init",
            "output/selected_frames.dcd",
        ):
            path = os.path.join(run0, produced)
            assert os.path.exists(path) and os.path.getsize(path) > 0, produced
    finally:
        os.chdir(cwd)
        reweight.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def _main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
