"""Unit tests for the shared cryoWEight modules.

Run standalone as python tests/test_cryoweight.py
or under pytest as pytest tests/test_cryoweight.py

Every test lives here, from the unit level checks through the end to end pipeline run
and the equivalence of the shared code against the original per system scripts.
"""

import ast
import io
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Imported through the package rather than by bare name, so that a module and its
# cryoweight.<name> spelling stay one object with one set of globals.
from cryoweight import cryoER_core as core
from cryoweight import cv_families as cvf
from cryoweight import likelihoods as lk

# cryoER_core, weights and the expectation maximization loop


def test_normalize_weights_sums_to_one():
    w = core.normalize_weights(torch.tensor([-3.0, 0.5, 2.0, -7.25]))
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert bool((w > 0).all())


def test_normalize_weights_is_shift_invariant():
    """Log weights are defined up to an additive constant, which normalisation removes."""
    log_w = torch.tensor([-3.0, 0.5, 2.0])
    a = core.normalize_weights(log_w)
    b = core.normalize_weights(log_w + 11.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_em_collapses_onto_the_only_structure_that_explains_the_data():
    """Structure 1 is a far better match for every observation, so it takes all the weight."""
    n_obs = 40
    log_Pij = torch.full((n_obs, 3), -50.0)
    log_Pij[:, 1] = 0.0
    log_w, losses = core.expectation_maximization_weights(log_Pij, num_iterations=200)
    w = torch.exp(log_w)
    assert float(w[1]) > 0.999
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_em_leaves_a_symmetric_problem_uniform():
    """With every structure equally likely there is no information to break the tie."""
    log_Pij = torch.zeros((25, 4))
    log_w, _ = core.expectation_maximization_weights(log_Pij, num_iterations=50)
    w = torch.exp(log_w)
    assert torch.allclose(w, torch.full((4,), 0.25), atol=1e-6)


def test_em_loss_is_recorded_once_per_iteration_and_does_not_increase():
    torch.manual_seed(0)
    log_Pij = torch.log(torch.rand(30, 5) + 1e-3)
    _, losses = core.expectation_maximization_weights(log_Pij, num_iterations=60)
    assert len(losses) == 60
    assert float(losses[-1]) <= float(losses[0]) + 1e-6


def test_em_uniform_initial_log_weights_match_the_default():
    """The default init assigns 1/M rather than log(1/M), which is uniform either way."""
    log_Pij = torch.log(torch.rand(20, 4) + 1e-3)
    a, _ = core.expectation_maximization_weights(log_Pij, num_iterations=30)
    explicit = torch.full((4,), -math.log(4.0))
    b, _ = core.expectation_maximization_weights(log_Pij, explicit, num_iterations=30)
    assert torch.allclose(torch.exp(a), torch.exp(b), atol=1e-6)


def test_evaluate_nll_matches_a_hand_computation():
    log_w = torch.tensor([0.0, 0.0])
    log_Pij = torch.log(torch.tensor([[0.8, 0.2], [0.4, 0.6]]))
    # Equal weights of 0.5, so each image contributes log(0.5 * (p0 + p1)) = log(0.5).
    expected = -math.log(0.5)
    assert abs(float(core.evaluate_nll(log_w, log_Pij)) - expected) < 1e-6


# cryoER_core, the other solvers
#
# Three routines solve the same problem over the weight simplex. The pipeline uses EM,
# and the source itself notes that the multiplicative gradient computes the same thing,
# so the strongest available check is that they agree on a problem with a known answer.


def _collapse_problem(n_obs=40, n_struct=3, winner=1):
    """A likelihood where one structure explains every observation far better than the rest."""
    log_Pij = torch.full((n_obs, n_struct), -50.0)
    log_Pij[:, winner] = 0.0
    return log_Pij


def test_naive_log_likelihood_is_the_gaussian_kernel_written_out():
    """log P = -d / (2 sigma^2), which is the definition the whole method rests on."""
    d = torch.tensor([[0.0, 1.0], [4.0, 9.0]])
    sigma = 0.5
    assert torch.allclose(core.eval_naive_log_Pij(d, sigma), -d / (2 * sigma**2))


def test_naive_log_likelihood_falls_away_with_distance():
    d = torch.tensor([[0.0, 1.0, 4.0]])
    out = core.eval_naive_log_Pij(d, 1.0)
    assert bool((torch.diff(out) < 0).all())


def test_frank_wolfe_gap_is_zero_when_every_direction_is_equally_good():
    """The gap bounds how far the current weights are from optimal, so a flat gradient is zero."""
    grad = torch.ones(5)
    assert abs(float(core.fw_gap(torch.full((5,), 0.2), grad))) < 1e-9


def test_frank_wolfe_gap_is_positive_when_a_better_direction_exists():
    grad = torch.tensor([1.0, 1.0, 3.0])
    assert float(core.fw_gap(torch.full((3,), 1 / 3), grad)) > 0.0


def test_multiplicative_gradient_agrees_with_expectation_maximization():
    """Two solvers, one problem. The source says they compute the same thing, so check it."""
    log_Pij = _collapse_problem()
    em, _ = core.expectation_maximization_weights(log_Pij, num_iterations=300)
    mg, _ = core.multiplicative_gradient(
        log_Pij, tol=1e-8, max_iterations=300, stats_frequency=10_000
    )
    assert torch.allclose(torch.exp(em), torch.exp(mg), atol=1e-4)


def test_multiplicative_gradient_returns_a_normalised_weight_vector():
    log_weights, _ = core.multiplicative_gradient(
        _collapse_problem(), tol=1e-8, max_iterations=200, stats_frequency=10_000
    )
    w = torch.exp(log_weights)
    assert abs(float(w.sum()) - 1.0) < 1e-5
    assert float(w[1]) > 0.99


def test_clustering_em_reduces_to_plain_em_when_every_cluster_holds_one_structure():
    """cluster_sizes weights each structure by how many frames it stands for."""
    log_Pij = _collapse_problem()
    plain, _ = core.expectation_maximization_weights(log_Pij, num_iterations=200)
    clustered, _ = core.expectation_maximization_weights_from_clustering(
        log_Pij, cluster_sizes=torch.ones(log_Pij.shape[1]), num_iterations=200
    )
    assert torch.allclose(torch.exp(plain), torch.exp(clustered), atol=1e-4)


def test_cluster_size_cancels_when_the_data_says_nothing():
    """The size is multiplied in for the fit and divided back out, so a flat likelihood
    returns equal per structure weights however uneven the clusters are."""
    for sizes in ([1.0, 9.0], [1.0, 1.0], [2.0, 3.0]):
        log_w, _ = core.expectation_maximization_weights_from_clustering(
            torch.zeros((30, 2)), cluster_sizes=torch.tensor(sizes), num_iterations=300
        )
        w = torch.exp(log_w)
        assert torch.allclose(w, torch.full((2,), 0.5), atol=1e-5), sizes


def test_cluster_size_does_not_rescue_a_structure_the_data_rejects():
    """Standing for nine frames is no help if the particles do not look like it."""
    log_Pij = torch.full((30, 2), -50.0)
    log_Pij[:, 0] = 0.0
    log_w, _ = core.expectation_maximization_weights_from_clustering(
        log_Pij, cluster_sizes=torch.tensor([1.0, 9.0]), num_iterations=300
    )
    w = torch.exp(log_w)
    assert float(w[0]) > 0.999


def test_gradient_descent_finds_the_same_answer_as_expectation_maximization():
    """The third solver. Adam on the log weights, and it should reach the same optimum."""
    log_Pij = _collapse_problem()
    em, _ = core.expectation_maximization_weights(log_Pij, num_iterations=300)
    gd, losses = core.gradient_descent_weights(
        log_Pij, log_weights_init=torch.zeros(log_Pij.shape[1]), num_iterations=400
    )
    assert int(torch.argmax(torch.exp(gd))) == int(torch.argmax(torch.exp(em)))
    assert float(losses[-1]) < float(losses[0])


def test_gradient_descent_accepts_a_regulariser():
    """The hook exists so a prior can be added to the fit, so it has to be applied."""
    log_Pij = _collapse_problem()
    _, plain = core.gradient_descent_weights(
        log_Pij, log_weights_init=torch.zeros(log_Pij.shape[1]), num_iterations=60
    )
    _, penalised = core.gradient_descent_weights(
        log_Pij,
        log_weights_init=torch.zeros(log_Pij.shape[1]),
        num_iterations=60,
        regularization_fxn=lambda w: 10.0 * (w**2).sum(),
    )
    assert float(penalised[-1]) > float(plain[-1])


def test_signal_standard_deviation_is_the_root_mean_square_inside_the_mask():
    """It measures signal over the central disc only, so the edges must not contribute."""
    img = torch.zeros((2, 32, 32))
    mask = core.circular_mask(32, 0.4 * 32)
    img[:, mask] = 3.0
    out = core.signal_std_torch_batch(img)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.full((2,), 3.0), atol=1e-5)


def test_signal_standard_deviation_ignores_what_lies_outside_the_mask():
    inside = torch.zeros((1, 32, 32))
    mask = core.circular_mask(32, 0.4 * 32)
    inside[:, mask] = 2.0
    noisy_edges = inside.clone()
    noisy_edges[:, ~mask] = 99.0
    assert torch.allclose(
        core.signal_std_torch_batch(inside), core.signal_std_torch_batch(noisy_edges), atol=1e-6
    )


# cryoER_core, image geometry


def test_gen_grid_is_centred_and_evenly_spaced():
    g = core.gen_grid(8, 0.5)
    assert g.shape == (8,)
    assert abs(float(g.sum())) < 1e-6
    spacing = torch.diff(g)
    assert torch.allclose(spacing, torch.full((7,), 0.5), atol=1e-6)


def test_circular_mask_area_follows_pi_r_squared():
    """The radius must be supplied in pixels, as both call sites in this module do."""
    n = 128
    r = 0.4 * n
    frac = float(core.circular_mask(n, r).float().mean())
    assert abs(frac - math.pi * 0.4**2) < 0.01


def test_circular_mask_default_radius_selects_nothing():
    """The default of 0.4 is a fraction, but the body compares it against pixel units.

    Both callers scale by n_pixel first so nothing is broken today, but calling this with
    the default returns an empty mask, which would make the SNR normalisation divide by
    the standard deviation of no pixels.
    """
    assert int(core.circular_mask(128).sum()) == 0


def test_quaternion_sampler_always_returns_the_number_asked_for():
    """It is rejection sampling, and it used to draw one fixed batch and slice it.

    The accepted shell is about 31 percent of the cube it draws from, so a single batch of
    five times the request falls short by chance. At twelve structures that happened about
    once in forty calls, which crashed image generation with a broadcast error against the
    coordinates; at one structure it happened about one time in six.
    """
    for n in (1, 2, 3, 5, 12, 40):
        for _ in range(25):
            assert core.gen_quat_torch(n, device="cpu").shape == (n, 4), n


def test_sampled_quaternions_are_unit_vectors():
    """They index rotations, so a non unit quaternion would scale the structure as well."""
    q = core.gen_quat_torch(64, device="cpu")
    assert torch.allclose(
        torch.linalg.vector_norm(q, dim=1), torch.ones(64, dtype=q.dtype), atol=1e-6
    )


def test_quaternion_to_matrix_returns_rotations():
    torch.manual_seed(0)
    q = core.gen_quat_torch(16, device="cpu")
    m = core.quaternion_to_matrix(q)
    assert m.shape == (16, 3, 3)
    # gen_quat_torch works in float64 while the image pipeline runs in float32, so the
    # comparison tensors have to follow the matrix rather than the torch default.
    eye = torch.eye(3, dtype=m.dtype).expand(16, 3, 3)
    assert torch.allclose(m @ m.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(m), torch.ones(16, dtype=m.dtype), atol=1e-5)


# cv_families


def _traj_of(points):
    """A one frame stand in whose xyz is the given (n_atoms, 3) array, in nm."""

    class _T:
        xyz = np.asarray(points, dtype=np.float32)[None, :, :]

    return _T()


def test_calculate_angles_on_a_right_angle():
    # Centroid 2 sits at the origin, centroid 1 on x and centroid 3 on y.
    t = _traj_of([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ang = cvf._calculate_angles(t, [0], [1], [2])
    assert abs(float(ang[0]) - 90.0) < 1e-3


def test_calculate_angles_on_collinear_points():
    t = _traj_of([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert abs(float(cvf._calculate_angles(t, [0], [1], [2])[0]) - 180.0) < 1e-3
    t = _traj_of([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert abs(float(cvf._calculate_angles(t, [0], [1], [2])[0])) < 1e-3


def test_calculate_angles_is_independent_of_arm_length():
    """The angle depends on direction only, so scaling one arm must not change it."""
    near = _traj_of([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    far = _traj_of([[9.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a = float(cvf._calculate_angles(near, [0], [1], [2])[0])
    b = float(cvf._calculate_angles(far, [0], [1], [2])[0])
    assert abs(a - b) < 1e-4


def test_cv_of_rejects_an_unknown_family():
    try:
        cvf.cv_of(None, None, {"cv_family": "not_a_family"})
    except ValueError as exc:
        assert "not_a_family" in str(exc)
    else:
        raise AssertionError("cv_of accepted an unknown family")


def test_compute_rejects_an_unknown_family():
    try:
        cvf.compute(None, None, None, {"cv_family": "not_a_family"})
    except ValueError as exc:
        assert "not_a_family" in str(exc)
    else:
        raise AssertionError("compute accepted an unknown family")


# likelihoods


def test_squared_distance_matches_an_explicit_computation():
    a = np.array([[0.0, 0.0], [3.0, 4.0]])
    b = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    expected = np.array([[0.0, 1.0, 4.0], [25.0, 20.0, 13.0]])
    assert np.allclose(lk.squared_distance(a, b), expected)


def test_squared_distance_shape_is_structures_by_observations():
    a, b = np.zeros((7, 2)), np.zeros((11, 2))
    assert lk.squared_distance(a, b).shape == (7, 11)


def test_squared_distance_of_a_set_with_itself_has_a_zero_diagonal():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 2))
    d = lk.squared_distance(a, a)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)


def test_distance_of_rejects_an_unknown_likelihood():
    try:
        lk.distance_of({"cfg": {"likelihood": "not_a_family"}})
    except ValueError as exc:
        assert "not_a_family" in str(exc)
    else:
        raise AssertionError("distance_of accepted an unknown likelihood")


def test_likelihood_defaults_to_cryoem():
    assert lk.labels_of({}) == lk.LABELS["cryoem"]
    assert lk.FAMILIES["cryoem"] is lk.cryoem


def test_every_family_has_labels():
    assert set(lk.FAMILIES) == set(lk.LABELS)


# the toy system in tests/testdata
#
# Twelve CA atoms on a circular arc that opens from a closed ring to a nearly straight
# chain over 40 frames, so both RMSD to the ring and radius of gyration rise with frame
# index. Rebuild with python tests/testdata/make_testdata.py

import mdtraj as md

TESTDATA = os.path.join(HERE, "testdata")
TOY_TOP = os.path.join(TESTDATA, "toy.pdb")
TOY_ENSEMBLE = os.path.join(TESTDATA, "toy_ensemble.dcd")
TOY_CV_CFG = {"cv_family": "rmsd_rg", "cv_atom_selection": "name CA"}


def _toy():
    reference = md.load(TOY_TOP)
    return reference, md.load(TOY_ENSEMBLE, top=TOY_TOP)


def _toy_ctx(sigma=1.0, lo=9.0, hi=12.0):
    """A reweighting context whose target is the open end of the toy ensemble."""
    cfg = dict(
        TOY_CV_CFG,
        likelihood="cv",
        cv_sigma=sigma,
        traj_file="toy_ensemble.dcd",
        cv_target_rmsd_lo=lo,
        cv_target_rmsd_hi=hi,
    )
    return {
        "cfg": cfg,
        "data_dir": TESTDATA,
        "output_directory": TESTDATA,
        "reference_top": TOY_TOP,
        "reference_traj": TOY_ENSEMBLE,
        "simulation_top": TOY_TOP,
        "simulation_traj": TOY_ENSEMBLE,
    }


def test_toy_fixture_has_the_shape_the_tests_assume():
    reference, ensemble = _toy()
    assert ensemble.n_frames == 40 and ensemble.n_atoms == 12
    cv = cvf.cv_of(ensemble, reference, TOY_CV_CFG)
    assert bool(np.all(np.diff(cv[:, 0]) > 0)), "RMSD must rise with frame index"
    assert bool(np.all(np.diff(cv[:, 1]) > 0)), "Rg must rise with frame index"


def test_rmsd_of_the_reference_against_itself_is_zero():
    reference, _ = _toy()
    cv = cvf.cv_of(reference, reference, TOY_CV_CFG)
    assert abs(float(cv[0, 0])) < 1e-4


def test_cv_of_returns_one_row_per_frame_in_angstrom():
    reference, ensemble = _toy()
    cv = cvf.cv_of(ensemble, reference, TOY_CV_CFG)
    assert cv.shape == (40, 2)
    # A twelve residue chain is a few Angstrom across, not a few nanometre.
    assert 1.0 < float(cv[:, 1].min()) < 100.0


def test_segment_cv_concatenates_parent_then_segment():
    reference, ensemble = _toy()
    parent, seg = ensemble[:5], ensemble[5:12]
    cv = cvf.compute(parent, seg, reference, TOY_CV_CFG)
    assert cv.shape == (12, 2)
    assert np.allclose(cv[:5], cvf.cv_of(parent, reference, TOY_CV_CFG))


def test_cv_likelihood_returns_a_structures_by_targets_matrix():
    diff, scale = lk.cv(_toy_ctx())
    assert diff.shape[0] == 40
    assert diff.shape[1] > 0
    assert scale == 1.0
    assert bool(np.all(diff >= 0.0))


def test_cv_likelihood_puts_the_open_structures_closest_to_the_target():
    """The target is drawn from the open end, so the open structures must match it best."""
    diff, _ = lk.cv(_toy_ctx())
    mean_distance = diff.mean(axis=1)
    assert int(np.argmin(mean_distance)) > 30
    assert mean_distance[0] > mean_distance[-1]


def test_distance_of_dispatches_to_the_same_result_as_calling_the_family():
    a, sa = lk.distance_of(_toy_ctx())
    b, sb = lk.cv(_toy_ctx())
    assert np.allclose(a, b) and sa == sb


def test_reweighting_moves_the_weight_towards_the_target():
    """The end to end claim of the method, checked on a system with a known answer.

    Starting from uniform weights over the whole arc, reweighting against a target taken
    from the open end must move the weight to the open end. The weighted mean frame index
    is the summary statistic, and it has to rise well past the uniform value of 19.5.
    """
    diff, scale = lk.distance_of(_toy_ctx())
    log_lik = torch.from_numpy((-diff / (2 * scale**2)).astype(np.float32))
    log_weights, _ = core.expectation_maximization_weights(log_lik.T, num_iterations=500)
    weights = torch.exp(log_weights).numpy()
    assert abs(weights.sum() - 1.0) < 1e-5
    mean_index = float((weights * np.arange(40)).sum())
    assert mean_index > 30.0, f"weight stayed at frame {mean_index:.1f}"


def test_a_wider_bandwidth_reweights_less_aggressively():
    """sigma_cv is the width of the CV space kernel, so a larger one is a weaker pull."""

    def mean_index(sigma):
        diff, scale = lk.distance_of(_toy_ctx(sigma=sigma))
        log_lik = torch.from_numpy((-diff / (2 * scale**2)).astype(np.float32))
        lw, _ = core.expectation_maximization_weights(log_lik.T, num_iterations=500)
        return float((torch.exp(lw).numpy() * np.arange(40)).sum())

    assert mean_index(0.5) >= mean_index(4.0)


# reweight.py, reachable now that its pipeline sits inside main()
#
# Importing this module used to run the whole pipeline and read reweight_config.json from
# the working directory. It no longer does either, so its stages can be called directly.

from cryoweight import reweight


def test_importing_reweight_does_not_run_the_pipeline():
    """The guard against the regression that made every function below untestable."""
    assert reweight.CFG == {} or isinstance(reweight.CFG, dict)
    assert callable(reweight.main)
    assert callable(reweight.load_config)


def test_load_config_populates_the_module_global():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "reweight_config.json")
        with open(path, "w") as fh:
            fh.write('{"strip_selection": "not water", "answer": 42}')
        try:
            cfg = reweight.load_config(path)
            assert cfg["answer"] == 42
            assert reweight.CFG["answer"] == 42
        finally:
            reweight.CFG = {}


def test_uniform_weights_are_one_over_the_frame_count():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "frame_count.txt"), "w") as fh:
            fh.write("8\n")
        reweight.generate_weights_from_file(tmp, tmp, "frame_count.txt", "w.txt")
        w = np.loadtxt(os.path.join(tmp, "w.txt"))
        assert w.shape == (8,)
        assert np.allclose(w, 1.0 / 8.0)
        assert abs(w.sum() - 1.0) < 1e-9


def test_uniform_weights_returns_none_rather_than_raising_on_a_missing_count():
    """The function swallows its errors and returns None, so a caller sees no exception."""
    with tempfile.TemporaryDirectory() as tmp:
        assert reweight.generate_weights_from_file(tmp, tmp, "absent.txt", "w.txt") is None


def test_uniform_weights_rejects_a_frame_count_of_zero():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "frame_count.txt"), "w") as fh:
            fh.write("0\n")
        assert reweight.generate_weights_from_file(tmp, tmp, "frame_count.txt", "w.txt") is None


def test_save_every_nth_frame_keeps_every_nth():
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(TOY_TOP, os.path.join(tmp, "toy.pdb"))
        shutil.copy(TOY_ENSEMBLE, os.path.join(tmp, "toy.dcd"))
        reweight.save_every_nth_frame(tmp, tmp, "toy.dcd", "toy.pdb", "sub.dcd", 4)
        sub = md.load(os.path.join(tmp, "sub.dcd"), top=os.path.join(tmp, "toy.pdb"))
        assert sub.n_frames == 10  # 40 frames, every 4th
        full = md.load(os.path.join(tmp, "toy.dcd"), top=os.path.join(tmp, "toy.pdb"))
        assert np.allclose(sub.xyz[1], full.xyz[4], atol=1e-5)


def test_get_init_traj_honours_the_configured_strip_selection_and_frame_cap():
    """Both the atom selection and the frame cap come from config, so both are checked."""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(TOY_TOP, os.path.join(tmp, "toy.pdb"))
        shutil.copy(TOY_ENSEMBLE, os.path.join(tmp, "toy.dcd"))
        reweight.CFG = {"strip_selection": "not water"}
        try:
            reweight.get_init_traj(tmp, tmp, "toy.dcd", "toy.pdb", "stripped.dcd", 7)
            out = md.load(os.path.join(tmp, "stripped.dcd"), top=os.path.join(tmp, "toy.pdb"))
            assert out.n_frames == 7
            assert out.n_atoms == 12  # the toy system has no water to strip
        finally:
            reweight.CFG = {}


def test_main_takes_a_run0_flag_defaulting_to_the_normal_iteration():
    """The seeding iteration is a flag on main, not a second copy of the whole script."""
    import inspect

    sig = inspect.signature(reweight.main)
    assert list(sig.parameters) == ["run0"]
    assert sig.parameters["run0"].default is False


def test_the_seeding_iteration_reads_the_init_md_config_keys():
    """run0 draws from the seeding MD; every later iteration draws from the previous WE run."""
    import inspect

    src = inspect.getsource(reweight.main)
    assert 'CFG["init_md_dir"] if run0 else CFG["merged_we_dir"]' in src
    assert 'CFG["init_md_dcd"] if run0 else CFG["merged_we_dcd"]' in src
    assert 'CFG["init_md_max_frames"] if run0 else CFG["merged_max_frames"]' in src


def test_only_one_reweight_script_is_staged_and_invoked():
    """Guards the merge, since a second script reappearing means the fork has grown back."""
    import inspect

    src = inspect.getsource(cryoWEight)
    assert "reweight_run0.py" not in src
    assert '"reweight.py", "--run0"' in src
    assert not os.path.exists(os.path.join(ROOT, "cryoweight", "reweight_run0.py"))


def test_copy_file_creates_the_destination_directory():
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "a.txt")
        with open(src, "w") as fh:
            fh.write("payload")
        dst = os.path.join(tmp, "nested", "deeper", "b.txt")
        reweight.copy_file(src, dst)
        assert open(dst).read() == "payload"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_copy_file_names_a_missing_source():
    tmp = tempfile.mkdtemp()
    try:
        reweight.copy_file(os.path.join(tmp, "absent.txt"), os.path.join(tmp, "b.txt"))
    except FileNotFoundError as exc:
        assert "absent.txt" in str(exc)
    else:
        raise AssertionError("a missing source was copied anyway")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_copy_file_refuses_a_directory():
    tmp = tempfile.mkdtemp()
    try:
        reweight.copy_file(tmp, os.path.join(tmp, "b.txt"))
    except IsADirectoryError as exc:
        assert tmp in str(exc)
    else:
        raise AssertionError("a directory was copied as a file")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_bstate_stages_return_quietly_when_their_inputs_are_absent():
    """Worth pinning because it is the shape of the runx bug, a stage that does nothing.

    Neither writes anything and neither raises, so a missing frame count silently produces
    an empty bstates directory rather than stopping the run.
    """
    for stage in (reweight.process_all_frames, reweight.process_cv_files):
        tmp = tempfile.mkdtemp()
        try:
            bstates = os.path.join(tmp, "bstates")
            os.makedirs(bstates)
            assert (
                stage(
                    data_dir=tmp,
                    output_directory=tmp,
                    bstates_dir=bstates,
                    framecount_file="absent.txt",
                    topology_file="absent.pdb",
                )
                is None
            )
            assert os.listdir(bstates) == []
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _bottleneck(tmp, neg_fallback):
    """Run get_bottleneck over the model target and return the labels it emitted."""
    shutil.copy(MODEL_TOP, tmp)
    shutil.copy(MODEL_TARGET, tmp)
    reweight.CFG = {
        "cv_family": "rmsd_rg",
        "cv_atom_selection": "name CA",
        "bottleneck_xlabel": "RMSD",
        "bottleneck_ylabel": "Rg",
        "bottleneck_contourf_extend": "neither",
        "xmin": 5,
        "xmax": 9,
        "ymin": 4,
        "ymax": 9,
        "bottleneck_neg_fallback": neg_fallback,
    }
    reweight.get_bottleneck(
        data_dir=tmp,
        dcd_file="model_target.dcd",
        topo_file="model_chignolin.pdb",
        output_dir=tmp,
        nbins=(20, 20),
        sigma_mults=[-3, -2, -1, 1, 2, 3],
        bottleneck_file="bottleneck_coordinates.txt",
    )
    with open(os.path.join(tmp, "bottleneck_coordinates.txt")) as fh:
        return [line.split("\t")[1] for line in fh if line.strip()]


def test_a_sigma_level_with_no_frames_is_skipped_by_default():
    """The model target spans one Angstrom, so minus two and three sigma fall off the grid."""
    tmp = tempfile.mkdtemp()
    try:
        labels = _bottleneck(tmp, neg_fallback=False)
        assert labels[0] == "Maximum Sampling"
        assert not any(l.startswith("-3") or l.startswith("-2") for l in labels)
    finally:
        reweight.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_negative_fallback_emits_one_level_when_none_was_reached():
    """ntl9 pulls towards lower RMSD and needs a negative bottleneck even off the grid.

    The fallback snaps to the grid column nearest mu minus sigma, and neg_done means it
    fires once however many negative levels came up empty.
    """
    tmp = tempfile.mkdtemp()
    try:
        labels = _bottleneck(tmp, neg_fallback=True)
        negatives = [l for l in labels if l.startswith("-1")]
        assert len(negatives) >= 1
        assert not any(l.startswith("-3") or l.startswith("-2") for l in labels)
    finally:
        reweight.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_bottleneck_file_always_starts_from_the_free_energy_minimum():
    """The driver reads the sigma rows, but the minimum anchors what they are relative to."""
    tmp = tempfile.mkdtemp()
    try:
        labels = _bottleneck(tmp, neg_fallback=False)
        assert labels[0] == "Maximum Sampling"
        assert os.path.getsize(os.path.join(tmp, "bottleneck_coordinates.txt")) > 0
    finally:
        reweight.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_check_device_reports_a_torch_device():
    device = reweight.check_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "cpu")


# the cryo-EM likelihood
#
# Synthetic particles are generated from the model trajectory, then every structure is
# compared against every particle. At 32 pixels the whole path costs about a second on
# CPU, so the headline capability of the package is reachable from the fast suite.

MODEL_TOP = os.path.join(TESTDATA, "model_chignolin.pdb")
MODEL_TARGET = os.path.join(TESTDATA, "model_target.dcd")


def _cryoem_ctx(tmp, snr=1.0, n_pixel=32):
    shutil.copy(MODEL_TOP, tmp)
    shutil.copy(MODEL_TARGET, tmp)
    cfg = {
        "n_pixel": n_pixel,
        "pixel_size": 1.0,
        "sigma": 1.5,
        "snr": snr,
        "n_image_per_struc": 1,
        "add_ctf": True,
        "likelihood": "cryoem",
        "defocus_min": 0.027,
        "defocus_max": 0.09,
        "batch_size": 4,
    }
    return {
        "cfg": cfg,
        "data_dir": tmp,
        "output_directory": tmp,
        "reference_top": os.path.join(tmp, "model_chignolin.pdb"),
        "reference_traj": os.path.join(tmp, "model_target.dcd"),
        "simulation_top": os.path.join(tmp, "model_chignolin.pdb"),
        "simulation_traj": os.path.join(tmp, "model_target.dcd"),
        "device": torch.device("cpu"),
    }


def test_cryoem_returns_a_structures_by_images_matrix():
    tmp = tempfile.mkdtemp()
    try:
        diff, scale = lk.cryoem(_cryoem_ctx(tmp))
        n = md.load(MODEL_TARGET, top=MODEL_TOP).n_frames
        assert diff.shape == (n, n)
        assert bool(np.all(np.isfinite(diff))) and bool(np.all(diff >= 0.0))
        assert scale > 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cryoem_scores_a_structure_best_against_its_own_particle():
    """The correctness claim of the likelihood, on data where the answer is known.

    Every particle is a projection of one structure in the same set, so the distance from
    a structure to its own particle must be smaller on average than to the others. It is
    an average rather than a per row minimum because at this image size and SNR the
    conformations are genuinely similar.
    """
    tmp = tempfile.mkdtemp()
    try:
        diff, _ = lk.cryoem(_cryoem_ctx(tmp))
        own = np.diag(diff)
        other = diff[~np.eye(diff.shape[0], dtype=bool)]
        assert own.mean() < other.mean()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cryoem_writes_the_distance_matrix_it_reads_back():
    """calc_image_struc_distance returns through the filesystem, not through a value."""
    tmp = tempfile.mkdtemp()
    try:
        ctx = _cryoem_ctx(tmp)
        lk.cryoem(ctx)
        cfg = ctx["cfg"]
        written = os.path.join(
            tmp,
            "diff_npix%d_ps%.2f_s%.1f_snr%.1E.npy"
            % (cfg["n_pixel"], cfg["pixel_size"], cfg["sigma"], cfg["snr"]),
        )
        assert os.path.exists(written)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cryoem_noise_scale_grows_as_the_signal_to_noise_falls():
    """lambda is the pixel noise width implied by the SNR, so a noisier image widens it."""
    quiet, noisy = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        _, high_snr = lk.cryoem(_cryoem_ctx(quiet, snr=1.0))
        _, low_snr = lk.cryoem(_cryoem_ctx(noisy, snr=0.01))
        assert low_snr > high_snr
    finally:
        shutil.rmtree(quiet, ignore_errors=True)
        shutil.rmtree(noisy, ignore_errors=True)


def test_distance_of_dispatches_to_the_cryoem_family():
    tmp = tempfile.mkdtemp()
    try:
        ctx = _cryoem_ctx(tmp)
        diff, scale = lk.distance_of(ctx)
        assert diff.shape[0] == diff.shape[1] and scale > 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# the run placeholder in cryoWEight.py
#
# The system YAML writes ../runx/merged_WE so that one config serves every iteration, and
# the driver rewrites runx to the current run name before the reweight script reads it.

sys.path.insert(0, ROOT)
import cryoWEight


def _config_dir(text):
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "reweight_config.json"), "w") as fh:
        fh.write(text)
    return tmp


def test_run_placeholder_is_resolved_in_the_config_not_the_script():
    """The placeholder lives in reweight_config.json, which is where it must be replaced.

    The driver used to substitute it in reweight.py, which has never contained the string,
    so the config reached the pipeline with merged_we_dir still pointing at ../runx.
    """
    tmp = _config_dir('{"merged_we_dir": "../runx/merged_WE"}')
    try:
        cryoWEight._resolve_run_placeholder(tmp, "run3")
        with open(os.path.join(tmp, "reweight_config.json")) as fh:
            written = fh.read()
        assert "runx" not in written
        assert "../run3/merged_WE" in written
    finally:
        shutil.rmtree(tmp)


def test_run_placeholder_raises_when_there_is_nothing_to_replace():
    """Silently doing nothing is what let the original bug survive, so this must be loud."""
    tmp = _config_dir('{"merged_we_dir": "../already_run1/merged_WE"}')
    try:
        cryoWEight._resolve_run_placeholder(tmp, "run3")
    except RuntimeError as exc:
        assert "runx" in str(exc)
    else:
        raise AssertionError("a missing placeholder passed unnoticed")
    finally:
        shutil.rmtree(tmp)


def test_every_system_config_carries_the_placeholder():
    """Every system relies on the substitution, so every one must contain the placeholder."""
    import glob

    configs = glob.glob(os.path.join(ROOT, "systems", "*.xml")) + glob.glob(
        os.path.join(ROOT, "examples", "*", "config.xml")
    )
    assert configs
    for path in configs:
        with open(path) as fh:
            assert "runx" in fh.read(), f"{os.path.basename(path)} has no runx placeholder"


# the stage scripts the driver invokes
#
# Each runs as its own process in a run directory. They used to read reweight_config.json
# at import, so nothing here was reachable; the config now loads on demand instead.

from cryoweight import get_distribution
from cryoweight import plot_free_energy
from cryoweight import sample_bstates


def test_selection_keeps_only_frames_inside_the_window():
    """The contract of select_distribution is that the saved frames satisfy both CV bounds."""
    tmp = tempfile.mkdtemp()
    try:
        top = os.path.join(tmp, "model_chignolin.pdb")
        traj = os.path.join(tmp, "model_target.dcd")
        shutil.copy(MODEL_TOP, top)
        shutil.copy(MODEL_TARGET, traj)
        sel = os.path.join(tmp, "sel.dcd")
        cv_cfg = {"cv_family": "rmsd_rg", "cv_atom_selection": "name CA"}
        lo, hi, ylo, yhi = 3.0, 8.0, 5.0, 8.0
        np.random.seed(0)
        get_distribution.select_distribution(
            traj_file=traj,
            topology_file=top,
            out_sel_dcd=sel,
            out_boltz_dcd=os.path.join(tmp, "boltz.dcd"),
            kB=0.008314,
            T=300.0,
            N_draw=3,
            nbins=(10, 10),
            sigma=(1, 1),
            x_range=[0.0, 10.0],
            y_range=[4.0, 9.0],
            hist_eps=1e-12,
            select_mode="angle_window",
            cv_cfg=cv_cfg,
            x_lower=lo,
            x_upper=hi,
            y_lower=ylo,
            y_upper=yhi,
        )
        kept = md.load(sel, top=top)
        reference = md.load(top)
        kept.superpose(reference)
        cv = cvf.cv_of(kept, reference, cv_cfg)
        assert kept.n_frames > 0
        assert bool(np.all((cv[:, 0] >= lo) & (cv[:, 0] <= hi)))
        assert bool(np.all((cv[:, 1] >= ylo) & (cv[:, 1] <= yhi)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_selection_never_draws_more_frames_than_it_selected():
    """N_draw is a request, and the resample is without replacement."""
    tmp = tempfile.mkdtemp()
    try:
        top = os.path.join(tmp, "model_chignolin.pdb")
        traj = os.path.join(tmp, "model_target.dcd")
        shutil.copy(MODEL_TOP, top)
        shutil.copy(MODEL_TARGET, traj)
        boltz = os.path.join(tmp, "boltz.dcd")
        np.random.seed(0)
        get_distribution.select_distribution(
            traj_file=traj,
            topology_file=top,
            out_sel_dcd=os.path.join(tmp, "sel.dcd"),
            out_boltz_dcd=boltz,
            kB=0.008314,
            T=300.0,
            N_draw=10_000,
            nbins=(10, 10),
            sigma=(1, 1),
            x_range=[0.0, 10.0],
            y_range=[4.0, 9.0],
            hist_eps=1e-12,
            select_mode="angle_window",
            cv_cfg={"cv_family": "rmsd_rg", "cv_atom_selection": "name CA"},
            x_lower=0.0,
            x_upper=99.0,
            y_lower=0.0,
            y_upper=99.0,
        )
        drawn = md.load(boltz, top=top)
        available = md.load(traj, top=top).n_frames
        assert drawn.n_frames == available
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _selection(tmp, mode, **kwargs):
    """Run one selection mode against the model trajectory and return the kept frames."""
    top = os.path.join(tmp, "model_chignolin.pdb")
    traj = os.path.join(tmp, "model_target.dcd")
    if not os.path.exists(top):
        shutil.copy(MODEL_TOP, top)
        shutil.copy(MODEL_TARGET, traj)
    sel = os.path.join(tmp, f"sel_{mode}.dcd")
    np.random.seed(0)
    get_distribution.select_distribution(
        traj_file=traj,
        topology_file=top,
        out_sel_dcd=sel,
        out_boltz_dcd=os.path.join(tmp, f"boltz_{mode}.dcd"),
        kB=0.008314,
        T=300.0,
        N_draw=kwargs.pop("N_draw", 3),
        nbins=(10, 10),
        sigma=(1, 1),
        x_range=[0.0, 10.0],
        y_range=[4.0, 9.0],
        hist_eps=1e-12,
        select_mode=mode,
        cv_cfg={"cv_family": "rmsd_rg", "cv_atom_selection": "name CA"},
        **kwargs,
    )
    if not os.path.exists(sel):
        return None
    kept = md.load(sel, top=top)
    reference = md.load(top)
    kept.superpose(reference)
    return cvf.cv_of(kept, reference, {"cv_family": "rmsd_rg", "cv_atom_selection": "name CA"})


def test_threshold_mode_keeps_everything_above_the_cut():
    """chignolin selects its unfolded target by a single RMSD threshold."""
    tmp = tempfile.mkdtemp()
    try:
        cv = _selection(tmp, "thresh", x_thresh=5.5)
        assert cv is not None and len(cv) > 0
        assert bool(np.all(cv[:, 0] >= 5.5))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_window_mode_keeps_only_what_lies_between_both_bounds():
    tmp = tempfile.mkdtemp()
    try:
        cv = _selection(tmp, "window", x_lower=5.5, x_upper=7.0)
        assert cv is not None and len(cv) > 0
        assert bool(np.all((cv[:, 0] >= 5.5) & (cv[:, 0] <= 7.0)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_greater_equal_and_less_equal_modes_are_complementary():
    """ntl9 pulls the other way, so it selects with a one sided cut in either direction."""
    tmp = tempfile.mkdtemp()
    try:
        # Every frame of the model target lies above 6 A, so the cut has to sit inside
        # the range for both sides to be non empty.
        above = _selection(tmp, "ge", x_thresh=6.5)
        below = _selection(tmp, "le", x_thresh=6.5)
        assert above is not None and below is not None
        assert bool(np.all(above[:, 0] >= 6.5))
        assert bool(np.all(below[:, 0] <= 6.5))
        total = md.load(MODEL_TARGET, top=MODEL_TOP).n_frames
        assert len(above) + len(below) == total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_window_mode_requires_at_least_one_bound():
    tmp = tempfile.mkdtemp()
    try:
        _selection(tmp, "window")
    except ValueError as exc:
        assert "window" in str(exc)
    else:
        raise AssertionError("a window with no bounds was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_one_sided_modes_require_their_threshold():
    for mode in ("ge", "le"):
        tmp = tempfile.mkdtemp()
        try:
            _selection(tmp, mode)
        except ValueError as exc:
            assert mode in str(exc)
        else:
            raise AssertionError(f"mode {mode} was accepted with no threshold")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_a_selection_that_matches_nothing_writes_nothing():
    """Better an empty run directory than a silently truncated one."""
    tmp = tempfile.mkdtemp()
    try:
        assert _selection(tmp, "ge", x_thresh=1e6) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# the adk angle CV family

ADK_TOP = os.path.join(HERE, "three_test", "adk_imgtop.pdb")
ADK_TRAJ = os.path.join(HERE, "three_test", "adk_img64.dcd")


def _adk_cfg():
    return dict(assemble.load_raw("adk")["reweight_config"])


def test_adk_angles_are_returned_in_degrees_for_every_frame():
    """The second CV family, hinge angles rather than RMSD and radius of gyration."""
    cfg = _adk_cfg()
    reference = md.load(ADK_TOP)
    traj = md.load(ADK_TRAJ, top=ADK_TOP)
    cv = cvf.cv_of(traj, reference, cfg)
    assert cv.shape == (traj.n_frames, 2)
    assert bool(np.all((cv >= 0.0) & (cv <= 180.0)))


def test_adk_angles_place_the_committed_sample_in_the_closed_basin():
    """The 64 frame sample in three_test is the closed starting state, not the open target.

    theta_NMP near 45 degrees is the closed basin the prior occupies; the cryo-EM target
    for this system sits near 75. Anchoring both angles here catches a change in the
    residue groups or in the angle convention, either of which would move the whole CV.
    """
    cfg = _adk_cfg()
    cv = cvf.cv_of(md.load(ADK_TRAJ, top=ADK_TOP), md.load(ADK_TOP), cfg)
    assert 40.0 < float(cv[:, 0].mean()) < 50.0
    assert 100.0 < float(cv[:, 1].mean()) < 120.0


def test_segment_level_adk_angles_concatenate_parent_then_segment():
    cfg = _adk_cfg()
    reference = md.load(ADK_TOP)
    traj = md.load(ADK_TRAJ, top=ADK_TOP)
    parent, seg = traj[:4], traj[4:10]
    cv = cvf.compute(parent, seg, reference, cfg)
    assert cv.shape == (10, 2)
    assert np.allclose(cv[:4], cvf.cv_of(parent, reference, cfg))


def _bstates_dir(rmsd_values):
    """A minimal bstates directory with one pcoord row, one bstates line and one directory."""
    tmp = tempfile.mkdtemp()
    pcoord = np.column_stack([np.asarray(rmsd_values, dtype=float), np.full(len(rmsd_values), 5.0)])
    np.savetxt(os.path.join(tmp, "pcoord.init"), pcoord)
    with open(os.path.join(tmp, "bstates.txt"), "w") as fh:
        for i in range(len(rmsd_values)):
            fh.write(f"{i:04d} {1.0 / len(rmsd_values):.32e} {i:04d}\n")
    for i in range(len(rmsd_values)):
        os.makedirs(os.path.join(tmp, f"{i:04d}"))
    return tmp


def test_bstate_filter_keeps_above_the_mean_plus_one_sigma_band_below():
    """The right sided rule keeps everything above the mean plus a one sigma band under it."""
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 20.0]
    tmp = _bstates_dir(values)
    sample_bstates.CFG = {"sigma_sign": "+"}
    try:
        sample_bstates.filter_and_rename_bstates(tmp)
        kept = np.loadtxt(os.path.join(tmp, "pcoord.init"))[:, 0]
        mu, sd = np.mean(values), np.std(values)
        expected = sorted(v for v in values if v > mu or (mu - sd <= v <= mu))
        assert sorted(kept.tolist()) == expected
    finally:
        sample_bstates.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_bstate_filter_mirrors_for_a_left_sided_target():
    """ntl9 pulls the other way, so the band sits above the mean instead of below."""
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 20.0]
    tmp = _bstates_dir(values)
    sample_bstates.CFG = {"sigma_sign": "-"}
    try:
        sample_bstates.filter_and_rename_bstates(tmp)
        kept = np.loadtxt(os.path.join(tmp, "pcoord.init"))[:, 0]
        mu, sd = np.mean(values), np.std(values)
        expected = sorted(v for v in values if v < mu or (mu <= v <= mu + sd))
        assert sorted(kept.tolist()) == expected
    finally:
        sample_bstates.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_bstate_filter_renormalises_the_weights_and_renumbers():
    """WESTPA needs the surviving basis states to be a contiguous, normalised set."""
    tmp = _bstates_dir([0.0, 1.0, 2.0, 3.0, 4.0, 20.0])
    sample_bstates.CFG = {"sigma_sign": "+"}
    try:
        sample_bstates.filter_and_rename_bstates(tmp)
        rows = [l.split() for l in open(os.path.join(tmp, "bstates.txt")) if l.strip()]
        assert abs(sum(float(r[1]) for r in rows) - 1.0) < 1e-9
        assert [r[0] for r in rows] == [f"{i:04d}" for i in range(len(rows))]
        assert sorted(f for f in os.listdir(tmp) if f.isdigit()) == [
            f"{i:04d}" for i in range(len(rows))
        ]
    finally:
        sample_bstates.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_free_energy_surface_is_written():
    tmp = tempfile.mkdtemp()
    try:
        top = os.path.join(tmp, "model_chignolin.pdb")
        traj = os.path.join(tmp, "model_target.dcd")
        shutil.copy(MODEL_TOP, top)
        shutil.copy(MODEL_TARGET, traj)
        out = os.path.join(tmp, "fes.png")
        plot_free_energy.CFG = {
            "cv_family": "rmsd_rg",
            "cv_atom_selection": "name CA",
            "xlabel": "RMSD",
            "ylabel": "Rg",
            "xmin": 0,
            "xmax": 10,
            "ymin": 4,
            "ymax": 9,
            "xticks": None,
            "yticks": None,
        }
        plot_free_energy.plot_free_energy(
            trajectory_file=traj,
            topology_file=top,
            kB=0.008314,
            T=300.0,
            nbins=(10, 10),
            output_file=out,
        )
        assert os.path.exists(out) and os.path.getsize(out) > 0
    finally:
        plot_free_energy.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


# the coordinate handoff from the driver to WESTPA
#
# Each reweighting iteration ends by writing bottleneck_coordinates.txt, and the driver
# picks one sigma level from it and writes it into the west.cfg of the next run. That handoff
# is the only place the two halves of the method meet.

BOTTLENECK = """1\tMaximum Sampling\t7.4407\t7.9384
2\t-2σ (τ=1.57)\t0.8379\t5.0471
3\t-1σ (τ=3.15)\t2.1923\t5.7143
4\t+1σ (τ=6.32)\t7.4407\t7.9384
"""


def _bottleneck_file(text=BOTTLENECK):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "bottleneck_coordinates.txt")
    with open(path, "w") as fh:
        fh.write(text)
    return tmp, path


def test_sigma_priority_orders_outermost_first():
    """A strategy is a preference order, and the outer levels are tried before the inner."""
    assert cryoWEight._sigma_priority("permissive", "+") == ["+3σ", "+2σ", "+1σ"]
    assert cryoWEight._sigma_priority("moderate", "+") == ["+2σ", "+1σ"]
    assert cryoWEight._sigma_priority("strict", "+") == ["+1σ"]


def test_sigma_priority_mirrors_for_a_left_sided_target():
    """ntl9 pulls towards lower RMSD, so its levels are negative."""
    assert cryoWEight._sigma_priority("permissive", "-") == ["-3σ", "-2σ", "-1σ"]


def test_sigma_priority_rejects_an_unknown_strategy():
    try:
        cryoWEight._sigma_priority("aggressive", "+")
    except ValueError as exc:
        assert "aggressive" in str(exc)
    else:
        raise AssertionError("an unknown strategy was accepted")


def test_bottleneck_selection_falls_back_to_the_level_that_exists():
    """permissive asks for 3 sigma first, but this run only reached 1, so it takes that."""
    tmp, path = _bottleneck_file()
    try:
        assert cryoWEight._select_bottleneck_coords(path, "permissive", "+") == ("7.44", "7.94")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bottleneck_selection_reads_the_sign_it_was_given():
    tmp, path = _bottleneck_file()
    try:
        assert cryoWEight._select_bottleneck_coords(path, "moderate", "-") == ("0.84", "5.05")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bottleneck_selection_raises_when_no_level_is_present():
    """Silently picking nothing would start a WE run at whatever west.cfg already said."""
    tmp, path = _bottleneck_file("1\tMaximum Sampling\t1.0\t2.0\n")
    try:
        cryoWEight._select_bottleneck_coords(path, "strict", "+")
    except RuntimeError as exc:
        assert "strict" in str(exc)
    else:
        raise AssertionError("a missing sigma level passed unnoticed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_west_cfg_patch_keeps_the_indentation_of_the_line_it_replaces():
    """west.cfg is YAML, so losing the indentation would change which block `at` sits in."""
    tmp = tempfile.mkdtemp()
    try:
        cfg = os.path.join(tmp, "west.cfg")
        with open(cfg, "w") as fh:
            fh.write("west:\n  system:\n    bins:\n        at: [1.0, 2.0]\n  other: keep\n")
        cryoWEight._write_west_cfg_coords(cfg, "7.44", "7.94")
        written = open(cfg).read().splitlines()
        assert "        at: [7.44, 7.94]" in written
        assert "  other: keep" in written
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_west_cfg_patch_raises_when_there_is_no_line_to_patch():
    tmp = tempfile.mkdtemp()
    try:
        cfg = os.path.join(tmp, "west.cfg")
        with open(cfg, "w") as fh:
            fh.write("west:\n  system:\n    nothing: here\n")
        cryoWEight._write_west_cfg_coords(cfg, "1.00", "2.00")
    except RuntimeError as exc:
        assert "at:" in str(exc)
    else:
        raise AssertionError("a west.cfg with no at line was patched anyway")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_staging_copies_the_script_and_everything_it_imports():
    """A reweight run is its own process in its own directory, so its imports travel with it.

    The driver reads scripts/ relative to the working directory, which is the run root, so
    the test stands one up rather than pointing at the repository.
    """
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        scripts = os.path.join(tmp, "scripts")
        os.makedirs(scripts)
        expected = (
            "reweight.py",
            "cryoER_core.py",
            "cv_families.py",
            "likelihoods.py",
            "build_system.py",
            "reweight_config.json",
        )
        for name in expected:
            with open(os.path.join(scripts, name), "w") as fh:
                fh.write("")
        with open(os.path.join(scripts, "unrelated.py"), "w") as fh:
            fh.write("")
        dest = os.path.join(tmp, "reweight_run0")
        os.makedirs(dest)
        os.chdir(tmp)
        cryoWEight._stage_reweight_scripts(dest)
        for name in expected:
            assert os.path.exists(os.path.join(dest, name)), name
        assert not os.path.exists(os.path.join(dest, "unrelated.py"))
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


# the driver orchestration
#
# run_init_reweight_simulation lays out reweight_run0, run1 and reweight_run1, staging a
# tree for each and handing coordinates between them. Only the subprocess calls are faked
# here; assemble, the staging and the west.cfg patching all run for real, because those
# are where the bugs have been.

import types


def _fake_subprocess(recorded, bottleneck=BOTTLENECK):
    """Stand in for subprocess.run that records the command and leaves what the next stage reads."""

    def run(cmd, cwd=None, check=False, capture_output=False, text=False, **kwargs):
        recorded.append((list(cmd), cwd))
        if cmd[:2] == ["python", "reweight.py"]:
            os.makedirs(os.path.join(cwd, "bstates"), exist_ok=True)
            out = os.path.join(cwd, "output")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "bottleneck_coordinates.txt"), "w") as fh:
                fh.write(bottleneck)
        if cmd[:2] == ["python", "merge.py"]:
            os.makedirs(os.path.join(cwd, "merged_WE"), exist_ok=True)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _run_root(system="chignolin_cv"):
    """A run directory as the user would have it before starting, assembled and with data."""
    tmp = tempfile.mkdtemp()
    assemble.assemble(system, tmp, stage_data=False)
    cfg = assemble.load_cfg(system)
    data = os.path.join(tmp, "data")
    os.makedirs(data, exist_ok=True)
    for name in (cfg["topology_explicit"], cfg["topology_stripped"]):
        with open(os.path.join(data, name), "w") as fh:
            fh.write("")
    return tmp, cfg


def _drive(system, fn, *args, bottleneck=BOTTLENECK):
    """Run one orchestration function against a fresh run root, faking only subprocess."""
    tmp, cfg = _run_root(system)
    recorded = []
    saved = (
        cryoWEight.CFG,
        cryoWEight.CONFIG_SYSTEM,
        cryoWEight.SIGMA_SIGN,
        cryoWEight.SIGMA_STRATEGY,
        cryoWEight.TOPOLOGY_EXPLICIT,
        cryoWEight.TOPOLOGY_STRIPPED,
        cryoWEight.LOCAL,
        cryoWEight.N_WORKERS,
        cryoWEight.subprocess,
    )
    cwd = os.getcwd()
    try:
        cryoWEight.CFG = cfg
        cryoWEight.CONFIG_SYSTEM = system
        cryoWEight.SIGMA_SIGN = cfg["sigma_sign"]
        cryoWEight.SIGMA_STRATEGY = "permissive"
        cryoWEight.TOPOLOGY_EXPLICIT = cfg["topology_explicit"]
        cryoWEight.TOPOLOGY_STRIPPED = cfg["topology_stripped"]
        cryoWEight.LOCAL = True
        cryoWEight.N_WORKERS = 1
        cryoWEight.subprocess = types.SimpleNamespace(run=_fake_subprocess(recorded, bottleneck))
        os.chdir(tmp)
        fn(*args)
        return tmp, recorded
    finally:
        os.chdir(cwd)
        (
            cryoWEight.CFG,
            cryoWEight.CONFIG_SYSTEM,
            cryoWEight.SIGMA_SIGN,
            cryoWEight.SIGMA_STRATEGY,
            cryoWEight.TOPOLOGY_EXPLICIT,
            cryoWEight.TOPOLOGY_STRIPPED,
            cryoWEight.LOCAL,
            cryoWEight.N_WORKERS,
            cryoWEight.subprocess,
        ) = saved


def test_init_builds_the_three_directories_an_iteration_needs():
    tmp = None
    try:
        tmp, _ = _drive("chignolin_cv", cryoWEight.run_init_reweight_simulation)
        for produced in ("reweight_run0", "run1", "reweight_run1"):
            assert os.path.isdir(os.path.join(tmp, produced)), produced
        assert os.path.isdir(os.path.join(tmp, "run1", "bstates"))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def test_init_runs_the_seeding_iteration_before_the_first_weighted_ensemble():
    """Order matters, because run1 starts from basis states that reweight_run0 writes first."""
    tmp = None
    try:
        tmp, recorded = _drive("chignolin_cv", cryoWEight.run_init_reweight_simulation)
        commands = [(" ".join(c), w) for c, w in recorded]
        seeding = next(i for i, (c, w) in enumerate(commands) if "--run0" in c)
        westpa = next(i for i, (c, w) in enumerate(commands) if c.startswith("bash init.sh"))
        assert seeding < westpa
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def test_init_writes_the_bottleneck_coordinates_into_west_cfg():
    """The only place the reweighting half hands a number to the sampling half."""
    tmp = None
    try:
        tmp, _ = _drive("chignolin_cv", cryoWEight.run_init_reweight_simulation)
        west = open(os.path.join(tmp, "run1", "west.cfg")).read()
        assert "at: [7.44, 7.94]" in west
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def test_init_resolves_the_run_placeholder_for_the_second_reweighting():
    """reweight_run1 reads the trajectories of its own run, so runx must already be run1."""
    tmp = None
    try:
        tmp, _ = _drive("chignolin_cv", cryoWEight.run_init_reweight_simulation)
        cfg = open(os.path.join(tmp, "reweight_run1", "reweight_config.json")).read()
        assert "runx" not in cfg
        assert "run1" in cfg
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def test_init_leaves_no_data_copy_or_bytecode_behind():
    """Each reweight directory gets its own copy of data/, which is deleted once used."""
    tmp = None
    try:
        tmp, _ = _drive("chignolin_cv", cryoWEight.run_init_reweight_simulation)
        for d in ("reweight_run0", "reweight_run1"):
            assert not os.path.exists(os.path.join(tmp, d, "data"))
            assert not os.path.exists(os.path.join(tmp, d, "__pycache__"))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def test_init_stops_when_no_sigma_level_was_reached():
    """Rather than start a weighted ensemble run at whatever west.cfg happened to say."""
    tmp = None
    try:
        tmp, _ = _drive(
            "chignolin_cv",
            cryoWEight.run_init_reweight_simulation,
            bottleneck="1\\tMaximum Sampling\\t1.0\\t2.0\\n",
        )
    except RuntimeError as exc:
        assert "permissive" in str(exc)
    else:
        raise AssertionError("a run with no usable sigma level proceeded anyway")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def test_an_iteration_carries_the_previous_basis_states_forward():
    """runN starts from the basis states reweight_run(N-1) produced."""
    tmp = None
    try:
        tmp, cfg = _run_root("chignolin_cv")
        prev = os.path.join(tmp, "reweight_run1")
        os.makedirs(os.path.join(prev, "bstates"))
        with open(os.path.join(prev, "bstates", "marker.txt"), "w") as fh:
            fh.write("carried")
        out = os.path.join(prev, "output")
        os.makedirs(out)
        with open(os.path.join(out, "bottleneck_coordinates.txt"), "w") as fh:
            fh.write(BOTTLENECK)
        os.makedirs(os.path.join(tmp, "run1", "merged_WE"), exist_ok=True)
        for name in (cfg["topology_explicit"], cfg["topology_stripped"]):
            with open(os.path.join(tmp, "run1", "merged_WE", name), "w") as fh:
                fh.write("")
        recorded = []
        saved = (
            cryoWEight.CFG,
            cryoWEight.CONFIG_SYSTEM,
            cryoWEight.SIGMA_SIGN,
            cryoWEight.SIGMA_STRATEGY,
            cryoWEight.TOPOLOGY_EXPLICIT,
            cryoWEight.TOPOLOGY_STRIPPED,
            cryoWEight.LOCAL,
            cryoWEight.N_WORKERS,
            cryoWEight.subprocess,
        )
        cwd = os.getcwd()
        try:
            cryoWEight.CFG = cfg
            cryoWEight.CONFIG_SYSTEM = "chignolin_cv"
            cryoWEight.SIGMA_SIGN = cfg["sigma_sign"]
            cryoWEight.SIGMA_STRATEGY = "permissive"
            cryoWEight.TOPOLOGY_EXPLICIT = cfg["topology_explicit"]
            cryoWEight.TOPOLOGY_STRIPPED = cfg["topology_stripped"]
            cryoWEight.LOCAL, cryoWEight.N_WORKERS = True, 1
            cryoWEight.subprocess = types.SimpleNamespace(run=_fake_subprocess(recorded))
            os.chdir(tmp)
            cryoWEight.run_iterative_reweight_simulation(2)
        finally:
            os.chdir(cwd)
            (
                cryoWEight.CFG,
                cryoWEight.CONFIG_SYSTEM,
                cryoWEight.SIGMA_SIGN,
                cryoWEight.SIGMA_STRATEGY,
                cryoWEight.TOPOLOGY_EXPLICIT,
                cryoWEight.TOPOLOGY_STRIPPED,
                cryoWEight.LOCAL,
                cryoWEight.N_WORKERS,
                cryoWEight.subprocess,
            ) = saved
        assert os.path.exists(os.path.join(tmp, "run2", "bstates", "marker.txt"))
        assert os.path.isdir(os.path.join(tmp, "reweight_run2"))
        cfg_text = open(os.path.join(tmp, "reweight_run2", "reweight_config.json")).read()
        assert "runx" not in cfg_text and "run2" in cfg_text
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# the command line


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_a_range_expands_to_every_run_between_its_ends():
    assert cryoWEight._expand_runs(_Args(range=(2, 5), runs=None)) == [2, 3, 4, 5]


def test_a_reversed_range_is_read_the_way_it_was_meant():
    assert cryoWEight._expand_runs(_Args(range=(5, 2), runs=None)) == [2, 3, 4, 5]


def test_a_run_list_accepts_commas_spaces_and_ranges_together():
    assert cryoWEight._expand_runs(_Args(range=None, runs="2-4,6 8")) == [2, 3, 4, 6, 8]


def test_run_indices_are_deduplicated_and_sorted():
    assert cryoWEight._expand_runs(_Args(range=(3, 5), runs="9 4")) == [3, 4, 5, 9]


def test_asking_for_no_runs_at_all_stops_rather_than_doing_nothing():
    try:
        cryoWEight._expand_runs(_Args(range=None, runs=None))
    except SystemExit as exc:
        assert "runs" in str(exc)
    else:
        raise AssertionError("an empty run list was accepted")


def _cli(argv):
    """Run the entry point with the two orchestration stages replaced by recorders."""
    calls = []
    saved = (cryoWEight.run_init_reweight_simulation, cryoWEight.run_iterative_reweight_simulation)
    try:
        cryoWEight.run_init_reweight_simulation = lambda: calls.append("init")
        cryoWEight.run_iterative_reweight_simulation = lambda n: calls.append(n)
        cryoWEight.main(argv)
        return calls
    finally:
        cryoWEight.run_init_reweight_simulation, cryoWEight.run_iterative_reweight_simulation = (
            saved
        )


def test_init_runs_the_seeding_stage_only():
    assert _cli(["--system", "chignolin_cv", "init"]) == ["init"]


def test_iterate_runs_the_requested_runs_in_order():
    assert _cli(["--system", "chignolin_cv", "iterate", "--range", "2", "4"]) == [2, 3, 4]


def test_iterate_accepts_a_list_as_well_as_a_range():
    assert _cli(["--system", "chignolin_cv", "iterate", "--runs", "2-3,7"]) == [2, 3, 7]


def test_no_subcommand_seeds_and_then_iterates():
    """The bare form is the whole campaign, which is why it does both."""
    assert _cli(["--system", "chignolin_cv"]) == ["init", 2, 3, 4]


def test_no_subcommand_honours_an_explicit_range():
    assert _cli(["--system", "chignolin_cv", "--range", "5", "6"]) == ["init", 5, 6]


def test_the_command_line_resolves_the_system_settings():
    """Every stage reads these globals, so the entry point has to set them from the config."""
    _cli(["--system", "chignolin_cv", "init"])
    assert cryoWEight.CONFIG_SYSTEM == "chignolin_cv"
    assert cryoWEight.SIGMA_SIGN == assemble.load_cfg("chignolin_cv")["sigma_sign"]
    assert cryoWEight.SIGMA_STRATEGY == "permissive"


def test_a_strategy_given_on_the_command_line_beats_the_default():
    _cli(["--system", "chignolin_cv", "init", "--sigma-strategy", "strict"])
    assert cryoWEight.SIGMA_STRATEGY == "strict"


def test_running_without_a_system_stops_instead_of_guessing_one():
    saved = os.environ.pop("CRYOWEIGHT_SYSTEM", None)
    try:
        cryoWEight.main([])
    except SystemExit as exc:
        assert "system" in str(exc).lower()
    else:
        raise AssertionError("the driver ran with no system selected")
    finally:
        if saved is not None:
            os.environ["CRYOWEIGHT_SYSTEM"] = saved


def test_a_cluster_run_submits_and_then_waits_for_the_job():
    """Off the local path the driver submits with sbatch and polls squeue until it clears."""
    tmp = tempfile.mkdtemp()
    recorded = []

    def fake_run(cmd, cwd=None, check=False, capture_output=False, text=False, **kw):
        recorded.append(list(cmd))
        out = "Submitted batch job 4242\\n" if "sbatch" in " ".join(cmd) else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    saved = (cryoWEight.subprocess, cryoWEight.LOCAL, cryoWEight.SSH_HOST)
    try:
        cryoWEight.subprocess = types.SimpleNamespace(run=fake_run)
        cryoWEight.LOCAL = False
        cryoWEight.SSH_HOST = "cluster"
        cryoWEight._run_westpa(tmp, "run1")
        joined = [" ".join(c) for c in recorded]
        assert any("init.sh" in c for c in joined)
        assert any("sbatch submit_WE.sh" in c for c in joined)
        assert any("squeue -j 4242" in c for c in joined)
    finally:
        cryoWEight.subprocess, cryoWEight.LOCAL, cryoWEight.SSH_HOST = saved
        shutil.rmtree(tmp, ignore_errors=True)


# assemble.py
#
# One YAML per system builds the whole run tree. Every system therefore has to survive
# being assembled, which is the cheapest possible check that a config is well formed.

import glob

import assemble

SYSTEMS = sorted(
    os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "systems", "*.xml"))
)


def test_there_are_systems_to_test():
    assert SYSTEMS, "no system config files found"


def test_flatten_lifts_nested_blocks_into_one_namespace():
    """Templates reference a leaf by its own name, so nesting is a grouping device only."""
    out = assemble.flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
    assert out == {"a": 1, "c": 2, "e": 3}


def test_flatten_turns_a_null_into_an_empty_string():
    """A YAML null would otherwise render as the text None inside a template."""
    assert assemble.flatten({"a": None}) == {"a": ""}


def test_flatten_refuses_a_duplicated_leaf_key():
    """One namespace means two blocks cannot both define the same leaf and be silent."""
    try:
        assemble.flatten({"one": {"shared": 1}, "two": {"shared": 2}})
    except ValueError as exc:
        assert "shared" in str(exc)
    else:
        raise AssertionError("a duplicate leaf key was accepted")


def test_every_system_carries_the_keys_the_driver_reads():
    """load_cfg is the namespace for the driver, and cryoWEight.py reads these four from it."""
    for system in SYSTEMS:
        cfg = assemble.load_cfg(system)
        for key in ("sigma_sign", "topology_explicit", "topology_stripped", "ssh_host"):
            assert key in cfg, f"{system} has no {key}"


def test_load_cfg_excludes_the_blocks_that_become_json():
    """seg_config and reweight_config are written as files, not flattened into templates."""
    raw = assemble.load_raw(SYSTEMS[0])
    cfg = assemble.load_cfg(SYSTEMS[0])
    assert "reweight_config" in raw
    assert "reweight_config" not in cfg and "seg_config" not in cfg
    assert "n_pixel" not in cfg


def test_every_system_assembles_a_run_tree():
    """The whole design rests on this, one config per system and no per system Python."""
    for system in SYSTEMS:
        tmp = tempfile.mkdtemp()
        try:
            written = assemble.assemble(system, tmp, stage_data=False)
            assert written, f"{system} produced nothing"
            produced = os.path.join(tmp, "scripts", "reweight_config.json")
            assert os.path.exists(produced), f"{system} wrote no reweight_config.json"
            cfg = json.load(open(produced))
            assert "topology_analysis" in cfg
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def test_a_template_token_is_replaced_by_its_config_value():
    """Templates carry {{token}} holes that the system config fills."""
    out = assemble.render("temperature is {{temperature_K}} K", {"temperature_K": 300}, "x")
    assert out == "temperature is 300 K"


def test_an_unresolved_template_token_stops_the_build():
    """A hole nobody filled would otherwise reach a compute node as literal braces."""
    try:
        assemble.render("{{no_such_key}}", {"other": 1}, "some/file")
    except KeyError as exc:
        assert "no_such_key" in str(exc) and "some/file" in str(exc)
    else:
        raise AssertionError("an unresolved token was rendered as is")


def test_validate_reports_a_clean_tree_as_matching():
    """validate assembles again and compares byte for byte, so a tree matches itself."""
    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble("chignolin_cv", tmp, stage_data=False)
        assert assemble.validate("chignolin_cv", tmp) == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validate_reports_a_changed_file_as_a_mismatch():
    """The point of it is catching a run tree that has drifted from what the config builds."""
    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble("chignolin_cv", tmp, stage_data=False)
        # scripts/reweight.py is in PHASE2_REFACTORED, the set validate deliberately
        # exempts because those files were rewritten on purpose. Pick one it compares.
        with open(os.path.join(tmp, "WE_files", "env.sh"), "a") as fh:
            fh.write("\n# drifted\n")
        assert assemble.validate("chignolin_cv", tmp) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validate_reports_a_missing_file_as_a_mismatch():
    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble("chignolin_cv", tmp, stage_data=False)
        os.remove(os.path.join(tmp, "WE_files", "env.sh"))
        assert assemble.validate("chignolin_cv", tmp) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_assemble_leaves_no_bytecode_in_the_run_tree():
    """__pycache__ in a staged tree would travel to every compute node for no reason."""
    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble("chignolin_cv", tmp, stage_data=False)
        for dirpath, dirnames, filenames in os.walk(tmp):
            assert "__pycache__" not in dirnames
            assert not any(f.endswith(".pyc") for f in filenames)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_run_placeholder_survives_assemblation():
    """It is resolved by the driver at run time, so it must still be there afterwards."""
    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble(SYSTEMS[0], tmp, stage_data=False)
        text = open(os.path.join(tmp, "scripts", "reweight_config.json")).read()
        assert "runx" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# merge.py
#
# After a WESTPA iteration finishes, its segments are merged into one trajectory. The
# iteration arrives either as a directory or as a tar, and the tar can unpack in three
# different shapes depending on how it was made, so each is checked here.

import tarfile

from cryoweight import merge


def _iteration_tree(root, iter_num=1, n_segs=3, nested=False, with_dcd=True):
    """A WESTPA style iteration tree of numbered segment directories each holding seg.dcd."""
    itr = f"{iter_num:06d}"
    base = os.path.join(root, "traj_segs", itr) if nested else os.path.join(root, itr)
    frames = md.load(MODEL_TARGET, top=MODEL_TOP)
    for s in range(n_segs):
        seg = os.path.join(base, f"{s:06d}")
        os.makedirs(seg)
        if with_dcd:
            frames[s : s + 1].save_dcd(os.path.join(seg, "seg.dcd"))
    return base


def test_iteration_is_found_as_a_plain_directory():
    tmp = tempfile.mkdtemp()
    try:
        _iteration_tree(tmp)
        folder, from_tar = merge.extract_iteration(1, tmp)
        assert folder == os.path.join(tmp, "000001")
        assert from_tar is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_iteration_is_extracted_from_a_tar():
    """Finished iterations are tarred to save inodes, so the merge has to unpack them."""
    tmp = tempfile.mkdtemp()
    try:
        _iteration_tree(tmp)
        with tarfile.open(os.path.join(tmp, "000001.tar"), "w") as tar:
            tar.add(os.path.join(tmp, "000001"), arcname="000001")
        shutil.rmtree(os.path.join(tmp, "000001"))
        folder, from_tar = merge.extract_iteration(1, tmp)
        assert folder == os.path.join(tmp, "000001")
        assert from_tar is True
        assert os.path.isdir(folder)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_iteration_is_found_when_the_tar_unpacks_under_traj_segs():
    """Some tars carry the traj_segs prefix, which puts the iteration one level deeper."""
    tmp = tempfile.mkdtemp()
    try:
        _iteration_tree(tmp, nested=True)
        with tarfile.open(os.path.join(tmp, "000001.tar"), "w") as tar:
            tar.add(os.path.join(tmp, "traj_segs"), arcname="traj_segs")
        shutil.rmtree(os.path.join(tmp, "traj_segs"))
        folder, from_tar = merge.extract_iteration(1, tmp)
        assert folder == os.path.join(tmp, "traj_segs", "000001")
        assert from_tar is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_iteration_names_both_places_it_looked():
    tmp = tempfile.mkdtemp()
    try:
        merge.extract_iteration(7, tmp)
    except FileNotFoundError as exc:
        assert "000007" in str(exc)
    else:
        raise AssertionError("a missing iteration was reported as found")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_segments_load_in_segment_order():
    """Segment order is the walker order, so it has to be sorted rather than listdir order."""
    tmp = tempfile.mkdtemp()
    try:
        base = _iteration_tree(tmp, n_segs=4)
        trajs = merge.load_iteration_trajs(base, MODEL_TOP)
        assert len(trajs) == 4
        assert all(t.n_frames == 1 for t in trajs)
        original = md.load(MODEL_TARGET, top=MODEL_TOP)
        for i, t in enumerate(trajs):
            assert np.allclose(t.xyz[0], original.xyz[i], atol=1e-5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_segments_are_found_under_a_traj_segs_subdirectory():
    tmp = tempfile.mkdtemp()
    try:
        base = os.path.join(tmp, "000001")
        os.makedirs(base)
        inner = os.path.join(base, "traj_segs")
        frames = md.load(MODEL_TARGET, top=MODEL_TOP)
        for s in range(2):
            seg = os.path.join(inner, f"{s:06d}")
            os.makedirs(seg)
            frames[s : s + 1].save_dcd(os.path.join(seg, "seg.dcd"))
        assert len(merge.load_iteration_trajs(base, MODEL_TOP)) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_segment_without_a_trajectory_is_skipped_rather_than_failing():
    """A crashed walker leaves its directory behind, and the merge must survive that."""
    tmp = tempfile.mkdtemp()
    try:
        base = _iteration_tree(tmp, n_segs=2)
        os.makedirs(os.path.join(base, "000009"))
        assert len(merge.load_iteration_trajs(base, MODEL_TOP)) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_merging_joins_every_segment_and_then_filters_by_the_cv_band():
    """The whole merge stage unpacks each iteration, joins the walkers and keeps the band."""
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    saved_cfg, saved_argv = merge.CFG, sys.argv
    try:
        traj_dir = os.path.join(tmp, "traj_segs")
        os.makedirs(traj_dir)
        for iteration in (1, 2):
            _iteration_tree(traj_dir, iter_num=iteration, n_segs=3)
        merge.CFG = {
            "cv_family": "rmsd_rg",
            "cv_atom_selection": "name CA",
            "cv_reference_pdb": "model_chignolin.pdb",
            "n_iterations": 2,
            "sigma_sign": "+",
        }
        os.chdir(tmp)
        sys.argv = [
            "merge.py",
            "--traj_dir",
            "traj_segs",
            "--topology",
            MODEL_TOP,
            "--start_iter",
            "1",
            "--end_iter",
            "2",
            "--output_dir",
            "merged_WE",
        ]
        merge.main()
        full = md.load(os.path.join(tmp, "merged_WE", "traj_all.dcd"), top=MODEL_TOP)
        band = md.load(os.path.join(tmp, "merged_WE", "traj.dcd"), top=MODEL_TOP)
        assert full.n_frames == 6  # two iterations of three walkers
        assert 0 < band.n_frames <= full.n_frames
    finally:
        os.chdir(cwd)
        merge.CFG, sys.argv = saved_cfg, saved_argv
        shutil.rmtree(tmp, ignore_errors=True)


def test_merging_stops_when_no_segments_were_found():
    """An empty merge would otherwise write a zero frame trajectory and carry on."""
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    saved_cfg, saved_argv = merge.CFG, sys.argv
    try:
        traj_dir = os.path.join(tmp, "traj_segs")
        _iteration_tree(traj_dir, iter_num=1, n_segs=2, with_dcd=False)
        merge.CFG = {"cv_family": "rmsd_rg", "n_iterations": 1}
        os.chdir(tmp)
        sys.argv = [
            "merge.py",
            "--traj_dir",
            "traj_segs",
            "--topology",
            MODEL_TOP,
            "--end_iter",
            "1",
            "--output_dir",
            "merged_WE",
        ]
        merge.main()
    except RuntimeError as exc:
        assert "No segment trajectories" in str(exc)
    else:
        raise AssertionError("an empty merge was allowed through")
    finally:
        os.chdir(cwd)
        merge.CFG, sys.argv = saved_cfg, saved_argv
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_bstates_backup_is_created_on_the_first_pass():
    """Filtering is destructive, so the unfiltered set is kept before anything is removed."""
    tmp = _bstates_dir([1.0, 2.0])
    try:
        sample_bstates.restore_from_backup_if_exists(tmp)
        assert os.path.isdir(f"{tmp}_all")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(f"{tmp}_all", ignore_errors=True)


def test_the_bstates_backup_is_restored_on_a_second_pass():
    """Rerunning has to start from the unfiltered set, not from the already filtered one."""
    tmp = _bstates_dir([1.0, 2.0])
    try:
        sample_bstates.restore_from_backup_if_exists(tmp)
        with open(os.path.join(tmp, "pcoord.init"), "w") as fh:
            fh.write("0 0\n")  # stand in for a filtered result
        sample_bstates.restore_from_backup_if_exists(tmp)
        assert np.loadtxt(os.path.join(tmp, "pcoord.init")).shape == (2, 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(f"{tmp}_all", ignore_errors=True)


def test_the_bstates_directory_is_rebuilt_when_only_the_backup_survives():
    tmp = _bstates_dir([1.0, 2.0])
    try:
        shutil.copytree(tmp, f"{tmp}_all")
        shutil.rmtree(tmp)
        sample_bstates.restore_from_backup_if_exists(tmp)
        assert os.path.isdir(tmp)
        assert np.loadtxt(os.path.join(tmp, "pcoord.init")).shape == (2, 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(f"{tmp}_all", ignore_errors=True)


def test_the_bstates_step_stops_when_neither_directory_exists():
    missing = os.path.join(tempfile.mkdtemp(), "absent")
    try:
        sample_bstates.restore_from_backup_if_exists(missing)
    except FileNotFoundError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("a missing bstates directory was accepted")


# build_system and the last branches

from cryoweight import build_system as bs
from openmm.app import PDBFile


def test_an_implicit_solvent_system_is_built_without_a_periodic_box():
    cfg = dict(assemble.load_raw("chignolin_cv")["reweight_config"])
    pdb = PDBFile(
        os.path.join(
            ROOT, "systems", "chignolin_cv", "overrides", "WE_files", "common_files", "folded.pdb"
        )
    )
    system = bs.build_system(bs.build_forcefield(cfg), pdb.topology, cfg)
    assert system.getNumParticles() == pdb.topology.getNumAtoms()
    assert system.usesPeriodicBoundaryConditions() is False


def test_an_explicit_solvent_system_is_built_with_periodic_boundaries():
    """The other half of the factory, PME and a box rather than a cutoff in vacuum."""
    cfg = dict(assemble.load_raw("chignolin")["reweight_config"])
    cfg["solvent_model"] = "explicit"
    cfg["ff_solvent"] = "amber14/tip3pfb.xml"
    pdb = PDBFile(os.path.join(ROOT, "systems", "chignolin", "data", "topology_explicit.pdb"))
    system = bs.build_system(bs.build_forcefield(cfg), pdb.topology, cfg)
    assert system.usesPeriodicBoundaryConditions() is True


def test_an_unknown_solvent_model_is_refused():
    """A typo here would otherwise fall through and build nothing."""
    cfg = dict(assemble.load_raw("chignolin_cv")["reweight_config"])
    cfg["solvent_model"] = "vacuum"
    pdb = PDBFile(
        os.path.join(
            ROOT, "systems", "chignolin_cv", "overrides", "WE_files", "common_files", "folded.pdb"
        )
    )
    try:
        bs.build_system(bs.build_forcefield(cfg), pdb.topology, cfg)
    except ValueError as exc:
        assert "vacuum" in str(exc)
    else:
        raise AssertionError("an unknown solvent model built a system anyway")


def test_the_cv_target_falls_back_when_the_rmsd_window_is_empty():
    """A system whose target basin is defined elsewhere leaves the window empty on purpose."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.copy(MODEL_TOP, tmp)
        shutil.copy(MODEL_TARGET, tmp)
        cfg = {
            "cv_family": "rmsd_rg",
            "cv_atom_selection": "name CA",
            "likelihood": "cv",
            "cv_sigma": 1.0,
            "traj_file": "model_target.dcd",
            "cv_target_rmsd_lo": 900.0,
            "cv_target_rmsd_hi": 999.0,
        }
        ctx = {
            "cfg": cfg,
            "data_dir": tmp,
            "output_directory": tmp,
            "reference_top": os.path.join(tmp, "model_chignolin.pdb"),
            "reference_traj": os.path.join(tmp, "model_target.dcd"),
            "simulation_top": os.path.join(tmp, "model_chignolin.pdb"),
            "simulation_traj": os.path.join(tmp, "model_target.dcd"),
        }
        diff, scale = lk.cv(ctx)
        total = md.load(MODEL_TARGET, top=MODEL_TOP).n_frames
        assert diff.shape == (total, total)  # the whole reference stands in as target
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_free_energy_axes_take_their_ticks_from_the_config():
    """xticks and yticks are optional, and the tick path is only taken when they are set."""
    tmp = tempfile.mkdtemp()
    try:
        top = os.path.join(tmp, "model_chignolin.pdb")
        traj = os.path.join(tmp, "model_target.dcd")
        shutil.copy(MODEL_TOP, top)
        shutil.copy(MODEL_TARGET, traj)
        out = os.path.join(tmp, "fes_ticked.png")
        plot_free_energy.CFG = {
            "cv_family": "rmsd_rg",
            "cv_atom_selection": "name CA",
            "xlabel": "RMSD",
            "ylabel": "Rg",
            "xmin": 0,
            "xmax": 10,
            "ymin": 4,
            "ymax": 9,
            "xticks": [0, 11, 2],
            "yticks": [4, 10, 1],
        }
        plot_free_energy.plot_free_energy(
            trajectory_file=traj,
            topology_file=top,
            kB=0.008314,
            T=300.0,
            nbins=(10, 10),
            output_file=out,
        )
        assert os.path.exists(out) and os.path.getsize(out) > 0
    finally:
        plot_free_energy.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_stage_script_loads_its_config_from_a_file():
    """The lazy config is the reason these modules are importable, so it is worth pinning."""
    for module in (plot_free_energy, get_distribution, sample_bstates, merge):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "reweight_config.json")
            with open(path, "w") as fh:
                fh.write('{"marker": 7}')
            assert module.load_config(path)["marker"] == 7
            assert module.CFG["marker"] == 7
        finally:
            module.CFG = {}
            shutil.rmtree(tmp, ignore_errors=True)


# regressions from the first end to end run of all three systems


def test_a_staged_script_is_given_the_modules_it_imports():
    """Each stage runs as a loose process, so a lone script cannot import cv_families."""
    import cryoWEight

    src = tempfile.mkdtemp()
    dst = tempfile.mkdtemp()
    try:
        for name in ("merge.py", "cv_families.py", "reweight_config.json"):
            with open(os.path.join(src, name), "w") as fh:
                fh.write("")
        old, cryoWEight.SCRIPTS = cryoWEight.SCRIPTS, src
        try:
            added = cryoWEight._stage_script("merge.py", dst)
        finally:
            cryoWEight.SCRIPTS = old
        staged = sorted(os.path.basename(p) for p in added)
        assert staged == ["cv_families.py", "merge.py", "reweight_config.json"], staged
        # A second call adds nothing, so cleanup never removes a file an earlier stage put there.
        old, cryoWEight.SCRIPTS = cryoWEight.SCRIPTS, src
        try:
            assert cryoWEight._stage_script("merge.py", dst) == []
        finally:
            cryoWEight.SCRIPTS = old
    finally:
        shutil.rmtree(src, ignore_errors=True)
        shutil.rmtree(dst, ignore_errors=True)


def test_pcoord_len_follows_the_segment_settings():
    """west.cfg is rendered with the computed row count, so any evenly dividing
    interval is valid and an uneven one is refused with the reason."""
    import assemble

    assert (
        assemble._pcoord_len(
            {"seg_config": {"n_steps_per_segment": 500, "dcd_report_interval": 250}}
        )
        == 3
    )
    assert (
        assemble._pcoord_len(
            {"seg_config": {"n_steps_per_segment": 5000, "dcd_report_interval": 5000}}
        )
        == 2
    )
    assert (
        assemble._pcoord_len(
            {"seg_config": {"n_steps_per_segment": 1000, "dcd_report_interval": 100}}
        )
        == 11
    )
    # An uneven interval truncates the unsaved tail, as the original per system runs did.
    assert (
        assemble._pcoord_len(
            {"seg_config": {"n_steps_per_segment": 2500, "dcd_report_interval": 1000}}
        )
        == 3
    )
    try:
        assemble._pcoord_len(
            {"seg_config": {"n_steps_per_segment": 100, "dcd_report_interval": 500}}
        )
    except SystemExit as exc:
        assert "at least one frame" in str(exc), exc
    else:
        raise AssertionError("a segment saving no frames passed unnoticed")


def test_the_kl_divergence_separates_identical_from_disjoint_distributions():
    """The convergence criterion is zero for a repeated distribution and large when
    the reweighted ensemble has moved to different bins."""
    import cryoWEight

    rows_a = [
        "{'Bin': (1, 1), 'Indices': [0], 'Weights': [0.5]}",
        "{'Bin': (2, 2), 'Indices': [1], 'Weights': [0.5]}",
    ]
    rows_c = ["{'Bin': (9, 9), 'Indices': [0], 'Weights': [1.0]}"]
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        for name, rows in (
            ("reweight_runA", rows_a),
            ("reweight_runB", rows_a),
            ("reweight_runC", rows_c),
        ):
            os.makedirs(os.path.join(tmp, name, "output"))
            with open(os.path.join(tmp, name, "output", "mergd_bins_wght_rescld.txt"), "w") as fh:
                fh.write("\n".join(rows))
        os.chdir(tmp)
        assert cryoWEight._kl_divergence("reweight_runA", "reweight_runB") < 1e-9
        assert cryoWEight._kl_divergence("reweight_runC", "reweight_runA") > 1.0
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_we_constants_render_from_the_config():
    """walkers per bin, wallclock, segments per state and the computed pcoord_len all
    come from the system config rather than being baked into the templates."""
    import assemble

    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble("examples/chignolin", tmp, stage_data=False)
        west = open(os.path.join(tmp, "WE_files", "west.cfg")).read()
        assert "pcoord_len: 3" in west
        assert "bin_target_counts: 4" in west
        assert "max_run_wallclock:    167:00:00" in west
        init = open(os.path.join(tmp, "WE_files", "init.sh")).read()
        assert "--segs-per-state 4" in init
        for dp, _, fs in os.walk(tmp):
            for f in fs:
                if f.endswith(".sh"):
                    path = os.path.join(dp, f)
                    assert os.access(path, os.X_OK), f"{path} is not executable"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_segment_cv_survives_a_parent_without_a_unitcell():
    """A state saved from implicit solvent carries no box, and join refuses to mix them."""
    import mdtraj as md

    ref = md.load(TOY_TOP)
    seg = md.load(TOY_ENSEMBLE, top=TOY_TOP)[:3]
    parent = md.load(TOY_TOP)
    parent.unitcell_vectors = None
    seg.unitcell_vectors = np.tile(np.eye(3) * 5.0, (seg.n_frames, 1, 1))
    cfg = {"cv_family": "rmsd_rg", "cv_atom_selection": "all"}
    cv = cvf.rmsd_rg(parent, seg, ref, cfg)
    assert cv.shape == (parent.n_frames + seg.n_frames, 2), cv.shape
    # The rows are the parent followed by the segment, computed independently.
    assert np.allclose(cv[:1], cvf.rmsd_rg_traj(parent, ref, cfg))
    assert np.allclose(cv[1:], cvf.rmsd_rg_traj(seg, ref, cfg))


def test_free_energy_plot_reads_the_topology_the_segments_propagated():
    """The merged trajectory holds WE segment atoms, not the stripped analysis subset."""
    from cryoweight import plot_free_energy as pfe

    seen = {}
    real_fn, real_cfg = pfe.plot_free_energy, pfe.CFG
    pfe.plot_free_energy = lambda **kw: seen.update(kw)
    pfe.CFG = {
        "trajectory_file": "traj.dcd",
        "topology_we": "topology_explicit.pdb",
        "topology_file": "topology_implicit.pdb",
        "kB": 0.008314,
        "T": 300,
        "nbins": [10, 10],
        "output_file": "fes.png",
    }
    try:
        pfe.main()
    finally:
        pfe.plot_free_energy, pfe.CFG = real_fn, real_cfg
    assert seen["topology_file"] == "topology_explicit.pdb", seen["topology_file"]


def test_a_rendered_shell_script_is_left_executable():
    """WESTPA runs post_iter.sh itself, and a 0644 file is skipped with only a warning,
    which silently disables the archiving that keeps traj_segs from growing without bound."""
    import assemble

    tmp = tempfile.mkdtemp()
    try:
        sh = os.path.join(tmp, "post_iter.sh")
        assemble._write(sh, "#!/bin/bash\necho hi\n")
        assert os.access(sh, os.X_OK), "a rendered shell script has to be executable"
        cfg = os.path.join(tmp, "seg_config.json")
        assemble._write(cfg, "{}")
        assert not os.access(cfg, os.X_OK), "a rendered config should not be executable"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_subiteration_count_is_set_once_at_the_top_of_the_yaml():
    """assemble injects the top level n_iterations into reweight_config.json and
    refuses a config that carries a different value inside the block."""
    import copy
    import json
    import assemble

    real = assemble.load_raw
    base = real("examples/chignolin")

    injected = copy.deepcopy(base)
    injected["n_iterations"] = "7"
    injected["reweight_config"].pop("n_iterations", None)
    assemble.load_raw = lambda system: injected
    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble("examples/chignolin", tmp, stage_data=False)
        rc = json.load(open(os.path.join(tmp, "scripts", "reweight_config.json")))
        assert rc["n_iterations"] == 7, rc["n_iterations"]
    finally:
        assemble.load_raw = real
        shutil.rmtree(tmp, ignore_errors=True)

    conflicted = copy.deepcopy(base)
    conflicted["n_iterations"] = "7"
    conflicted["reweight_config"]["n_iterations"] = 3
    assemble.load_raw = lambda system: conflicted
    tmp = tempfile.mkdtemp()
    try:
        assemble.assemble("examples/chignolin", tmp, stage_data=False)
    except SystemExit as exc:
        assert "Set it once" in str(exc), exc
    else:
        raise AssertionError("a conflicting duplicate passed unnoticed")
    finally:
        assemble.load_raw = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_xml_round_trips_every_value_type():
    """write_xml and read_xml invert each other for every type a config carries,
    including whitespace only strings, which pretty printing would destroy."""
    from cryoweight import configio

    cfg = {
        "name": "x",
        "pad": "    ",
        "trailing": "[2.5, 4.5]   ",
        "count": 4,
        "rate": 0.05,
        "flag": True,
        "off": False,
        "nothing": None,
        "grid": [100, 100],
        "mixed": [1, 2.5, "a"],
        "block": {"inner": {"deep": 7}, "text": "line one\nline two"},
    }
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "config.xml")
        configio.write_xml(path, cfg)
        assert configio.read_xml(path) == cfg
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_auto_sigma_sign_puts_the_bottleneck_on_the_target_side():
    """The sign compares where the configured target region lies against the mean of
    the binned seeding ensemble, so a prior with no target overlap still resolves."""
    import cryoWEight

    def write_run(tmp, bins_weights):
        out = os.path.join(tmp, "output")
        os.makedirs(out)
        with open(os.path.join(out, "bin_crds_wght.txt"), "w") as fh:
            for i, (b, w) in enumerate(bins_weights):
                fh.write(f"Frame {i}: RMSD Bin {b}, Rg Bin 5, Weight: {w:.6e}\n")

    cfg = {"bin_x_min": 0, "bin_width": 1.0, "select_mode": "thresh", "x_thresh": 7.25}
    tmp = tempfile.mkdtemp()
    try:
        write_run(tmp, [(1, 0.5), (3, 0.5)])
        assert cryoWEight._resolve_sigma_sign(tmp, cfg) == "+"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cfg = {
        "bin_x_min": 0,
        "bin_width": 1.0,
        "select_mode": "window",
        "x_lower": 2.75,
        "x_upper": 3.5,
    }
    tmp = tempfile.mkdtemp()
    try:
        write_run(tmp, [(12, 0.5), (14, 0.5)])
        assert cryoWEight._resolve_sigma_sign(tmp, cfg) == "-"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_iteration_stages_report_done_only_on_real_artifacts():
    """Resume trusts artifact content, not file existence, so a half written run is
    redone rather than skipped."""
    import h5py

    import cryoWEight

    tmp = tempfile.mkdtemp()
    try:
        run = os.path.join(tmp, "run2")
        os.makedirs(os.path.join(run, "merged_WE"))
        assert not cryoWEight._westpa_done(run, 2)
        with h5py.File(os.path.join(run, "west.h5"), "w") as fh:
            fh.attrs["west_current_iteration"] = 2
        assert not cryoWEight._westpa_done(run, 2), "an unfinished run passed as done"
        with h5py.File(os.path.join(run, "west.h5"), "w") as fh:
            fh.attrs["west_current_iteration"] = 3
        assert cryoWEight._westpa_done(run, 2)

        assert not cryoWEight._merge_done(run)
        for f in ("traj.dcd", "traj_all.dcd"):
            open(os.path.join(run, "merged_WE", f), "w").close()
        assert cryoWEight._merge_done(run)

        re_dir = os.path.join(tmp, "reweight_run2")
        os.makedirs(os.path.join(re_dir, "bstates"))
        os.makedirs(os.path.join(re_dir, "output"))
        assert not cryoWEight._reweight_done(re_dir)
        open(os.path.join(re_dir, "bstates", "bstates.txt"), "w").close()
        open(os.path.join(re_dir, "output", "bottleneck_coordinates.txt"), "w").close()
        assert cryoWEight._reweight_done(re_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# the equivalence and pipeline suites, formerly separate files

if __name__ == "__main__":
    print("loading the MD stack, the tests follow", flush=True)

import json as _json
import subprocess as _subprocess

import mdtraj
import mdtraj as md
from openmm import XmlSerializer
from openmm.app import CutoffNonPeriodic, ForceField, HBonds, PDBFile, PME
from openmm.unit import nanometer

from cryoweight import build_system as bs
from cryoweight import configio

E2E_PKG = os.path.join(ROOT, "cryoweight")
TS_SAMP = os.path.join(HERE, "three_test")
ADK_COMMON = os.path.join(ROOT, "examples", "adk", "overrides", "WE_files", "common_files")
NTL9_COMMON = os.path.join(ROOT, "examples", "ntl9", "overrides", "WE_files", "common_files")
CHIG_BSTATE = os.path.join(
    ROOT, "systems", "chignolin", "overrides", "WE_files", "common_files", "bstate.pdb"
)
CHIG_TRAJ = os.path.join(ROOT, "systems", "chignolin", "init_MD", "chignolin.dcd")
CHIG_TOP = os.path.join(ROOT, "systems", "chignolin", "init_MD", "chignolin.pdb")


# The model trajectory spans RMSD 0.75 to 7.53 A and Rg 5.01 to 7.98 A, so every
# coordinate range, bin edge and axis limit has to move off the chignolin numbers.
E2E_OVERRIDES = {
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
    "em_iterations": 200,
}


def _e2e_build_run(dest):
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
    # The Boltzmann selected target set the reweight stage reads.
    shutil.copy(os.path.join(TESTDATA, "model_target.dcd"), os.path.join(data, "image_sel.dcd"))
    for name in (
        "reweight.py",
        "cryoER_core.py",
        "cv_families.py",
        "likelihoods.py",
        "build_system.py",
    ):
        src = os.path.join(E2E_PKG, name)
        if os.path.exists(src):
            shutil.copy(src, run0)
    cfg = dict(
        configio.read_xml(os.path.join(ROOT, "systems", "chignolin_cv.xml"))["reweight_config"]
    )
    cfg.update(E2E_OVERRIDES)
    with open(os.path.join(run0, "reweight_config.json"), "w") as fh:
        _json.dump(cfg, fh, indent=2)
    return run0


def test_the_seeding_iteration_runs_from_one_end_to_the_other():
    """Every stage, from the seeding trajectory through to the basis states for WESTPA.

    main() is called in process rather than through a subprocess so that a coverage run
    can see which lines of the pipeline it reaches. The subprocess form is what the driver
    actually does, and test_cryoweight.py asserts the driver still invokes it that way.
    """
    sys.path.insert(0, E2E_PKG)
    import reweight

    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        run0 = _e2e_build_run(tmp)
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
            "bstates/pcoord.init",
            "output/selected_frames.dcd",
        ):
            path = os.path.join(run0, produced)
            assert os.path.exists(path) and os.path.getsize(path) > 0, produced
    finally:
        os.chdir(cwd)
        reweight.CFG = {}
        shutil.rmtree(tmp, ignore_errors=True)


# solvated/protein structure for build_system (boxed for explicit PME)
TS_STRUCT = {
    "adk": os.path.join(ADK_COMMON, "bstate.pdb"),
    "chignolin": os.path.join(
        ROOT, "systems", "chignolin", "overrides", "WE_files", "common_files", "bstate.pdb"
    ),
    "ntl9": os.path.join(NTL9_COMMON, "bstate.pdb"),
}


def _ts_original_system(name, ff, top):
    if name in ("adk", "chignolin"):
        return ff.createSystem(
            top, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
        )
    return ff.createSystem(
        top,
        nonbondedMethod=CutoffNonPeriodic,
        constraints=HBonds,
        nonbondedCutoff=2 * nanometer,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )


def _ts_orig_cv(traj, ref, cvcfg):
    if cvcfg["cv_family"] == "rmsd_rg":
        ai = ref.topology.select(cvcfg.get("cv_atom_selection", "name CA"))
        r = np.asarray(md.rmsd(traj, ref, atom_indices=ai)) * 10
        rg = np.asarray(md.compute_rg(traj.atom_slice(ai))) * 10
        return np.column_stack((r, rg))
    g = cvcfg["resid_groups"]
    sel = lambda s: ref.topology.select(f"name CA and resid {s}")
    tn = cvf._calculate_angles(traj, sel(g["core_lid"]), sel(g["core_nmp"]), sel(g["nmp"]))
    tl = cvf._calculate_angles(traj, sel(g["core_lid_angle"]), sel(g["core_lid"]), sel(g["lid"]))
    return np.column_stack((tn, tl))


def test_the_three_real_systems_reproduce_their_originals():
    """The shared builder and CV reproduce each original system exactly on real data."""

    ok = True
    for s in ["adk", "chignolin", "ntl9"]:
        cfg = configio.read_xml(os.path.join(ROOT, "systems", f"{s}.xml"))
        rc = cfg["reweight_config"]

        # (1) validate against the original run tree (corpus), if available
        corpus = os.path.join(ROOT, "..", "corpus", s)
        if os.path.isdir(corpus):
            v = _subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, "assemble.py"),
                    "--system",
                    s,
                    "--validate",
                    corpus,
                ],
                capture_output=True,
                text=True,
            )
            v_ok = v.returncode == 0
            v_lbl = "OK" if v_ok else "FAIL"
        else:
            v_ok = True
            v_lbl = "skip"  # corpus not shipped with the repo

        # (2) build_system XML equivalence
        pdb = PDBFile(TS_STRUCT[s])
        new = bs.build_system(bs.build_forcefield(rc), pdb.topology, rc)
        old = _ts_original_system(s, ForceField(rc["ff_main"], rc["ff_solvent"]), pdb.topology)
        b_ok = XmlSerializer.serialize(new) == XmlSerializer.serialize(old)

        # (3) cv_of on a real per system image sample
        traj = md.load(
            os.path.join(TS_SAMP, f"{s}_img64.dcd"), top=os.path.join(TS_SAMP, f"{s}_imgtop.pdb")
        )
        ref = md.load(os.path.join(TS_SAMP, f"{s}_imgtop.pdb"))
        new_cv = cvf.cv_of(traj, ref, rc["cv_cfg"])
        old_cv = _ts_orig_cv(traj, ref, rc["cv_cfg"])
        c_ok = np.array_equal(new_cv, old_cv)

        ok &= v_ok and b_ok and c_ok
        print(
            f"{s:10} {v_lbl:>10} {('OK '+str(new.getNumParticles())+'p') if b_ok else 'FAIL':>14} "
            f"{('OK '+rc['cv_cfg']['cv_family']) if c_ok else 'FAIL':>22}"
        )
    assert ok, "a real system diverged from its original"


P2_SYSTEMS = {
    "adk": dict(
        pdb=os.path.join(ADK_COMMON, "bstate.pdb"),
        solvent_model="explicit",
        ff_main="amber14-all.xml",
        ff_solvent="amber14/tip3p.xml",
        nonbonded_cutoff_nm=1,
    ),
    "chignolin": dict(
        pdb=CHIG_BSTATE,
        solvent_model="explicit",
        ff_main="amber14-all.xml",
        ff_solvent="amber14/tip3p.xml",
        nonbonded_cutoff_nm=1,
    ),
    "ntl9": dict(
        pdb=os.path.join(NTL9_COMMON, "bstate.pdb"),
        solvent_model="implicit",
        ff_main="amber14-all.xml",
        ff_solvent="implicit/obc2.xml",
        nonbonded_cutoff_nm=2,
        solute_dielectric=1.0,
        solvent_dielectric=78.5,
    ),
}


def _p2_original_system(name, cfg, top):
    ff = ForceField(cfg["ff_main"], cfg["ff_solvent"])
    if name in ("adk", "chignolin"):  # explicit solvent, verbatim from the original production.py
        return ff.createSystem(
            top, nonbondedMethod=PME, nonbondedCutoff=1 * nanometer, constraints=HBonds
        )
    return ff.createSystem(
        top,
        nonbondedMethod=CutoffNonPeriodic,
        constraints=HBonds,
        nonbondedCutoff=2 * nanometer,
        soluteDielectric=1.0,
        solventDielectric=78.5,
    )


def test_the_shared_system_factory_matches_the_original_inline_builds():
    """Each System built by the factory serializes to the same XML as the original call."""
    for name, cfg in P2_SYSTEMS.items():
        pdb = PDBFile(cfg["pdb"])
        cfg = dict(cfg, constraints="HBonds")
        ff = bs.build_forcefield(cfg)
        new = bs.build_system(ff, pdb.topology, cfg)
        old = _p2_original_system(name, cfg, pdb.topology)
        xn, xo = XmlSerializer.serialize(new), XmlSerializer.serialize(old)
        ok = xn == xo
        print(
            f"   {name:10} {'OK' if ok else 'FAIL'}  ({pdb.topology.getNumAtoms()} atoms, system XML {len(xn)} chars)"
        )
        assert ok, f"{name}: build_system XML differs from original"


# original inline CV logic, verbatim from the per system cv.py
def _p2_orig_rmsd_rg(parent, seg, reference, sel="name CA"):
    ai = reference.topology.select(sel)
    rmsd_parent = mdtraj.rmsd(parent, reference, atom_indices=ai)
    rmsd_traj = mdtraj.rmsd(seg, reference, atom_indices=ai)
    r = np.asarray(np.append(rmsd_parent, rmsd_traj)) * 10
    rgp = mdtraj.compute_rg(parent.atom_slice(ai))
    rgt = mdtraj.compute_rg(seg.atom_slice(ai))
    rg = np.asarray(np.append(rgp, rgt)) * 10
    return np.column_stack((r, rg))


def _p2_orig_adk(parent, seg, reference, g):
    sel = lambda spec: reference.topology.select(f"name CA and resid {spec}")
    nmp, cnmp, clid, lid, cla = (
        sel(g["nmp"]),
        sel(g["core_nmp"]),
        sel(g["core_lid"]),
        sel(g["lid"]),
        sel(g["core_lid_angle"]),
    )
    comb = parent.join(seg)
    tn = cvf._calculate_angles(comb, clid, cnmp, nmp)
    tl = cvf._calculate_angles(comb, cla, clid, lid)
    return np.column_stack((tn, tl))


def test_the_shared_cv_families_match_the_original_inline_cv():
    """Both CV families reproduce the original inline math exactly on real structures."""
    # rmsd_rg via chignolin trajectory
    traj = mdtraj.load(CHIG_TRAJ, top=CHIG_TOP)
    ref = traj[0]
    parent, seg = traj[0:3], traj[3:6]
    cfg = dict(cv_family="rmsd_rg", cv_atom_selection="name CA")
    new = cvf.compute(parent, seg, ref, cfg)
    old = _p2_orig_rmsd_rg(parent, seg, ref)
    print(f"   rmsd_rg/chignolin {'OK' if np.array_equal(new, old) else 'FAIL'}  shape={new.shape}")
    assert np.array_equal(new, old)
    # rmsd_rg via ntl9 (reference = folded.pdb, distinct from topology)
    nt = mdtraj.load(os.path.join(NTL9_COMMON, "bstate.pdb"))
    nref = mdtraj.load(os.path.join(NTL9_COMMON, "folded.pdb"))
    cfg = dict(cv_family="rmsd_rg", cv_atom_selection="name CA")
    new = cvf.compute(nt, nt, nref, cfg)
    old = _p2_orig_rmsd_rg(nt, nt, nref)
    print(f"   rmsd_rg/ntl9      {'OK' if np.array_equal(new, old) else 'FAIL'}  shape={new.shape}")
    assert np.array_equal(new, old)
    # adk_angles
    ad = mdtraj.load(os.path.join(ADK_COMMON, "bstate.pdb"))
    groups = dict(
        nmp="35 to 55",
        core_nmp="90 to 100",
        core_lid="115 to 125",
        lid="125 to 153",
        core_lid_angle="179 to 185",
    )
    cfg = dict(cv_family="adk_angles", resid_groups=groups)
    new = cvf.compute(ad, ad, ad, cfg)
    old = _p2_orig_adk(ad, ad, ad, groups)
    print(
        f"   adk_angles/adk    {'OK' if np.array_equal(new, old) else 'FAIL'}  shape={new.shape}  sample={new[0]}"
    )
    assert np.array_equal(new, old)


def _main():
    import contextlib
    import io as _io

    verbose = "-v" in sys.argv
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    total = len(tests)
    failed = []
    live = sys.stdout.isatty() and not verbose

    def draw(done, name):
        filled = 24 * done // total
        pct = 100 * done // total
        text = name[:38]
        sys.stdout.write(f"\r[{'█' * filled}{'░' * (24 - filled)}] {pct:3d}% {done:3d}/{total}  {text:<38}")
        sys.stdout.flush()

    for i, (name, fn) in enumerate(tests, 1):
        if live:
            draw(i - 1, name)
        buffer = _io.StringIO()
        try:
            if verbose:
                fn()
            else:
                # The pipeline stages narrate as they run, and a test run should show
                # progress only. A failing test prints what it captured.
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    fn()
            if not live:
                print(f"[{i:3d}/{total}] pass  {name}", flush=True)
        except Exception as exc:
            failed.append((name, exc))
            if live:
                sys.stdout.write("\r" + " " * 90 + "\r")
            print(f"[{i:3d}/{total}] FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
            tail = buffer.getvalue().strip().splitlines()[-25:]
            for line in tail:
                print(f"        {line}")
    if live:
        draw(total, "")
        sys.stdout.write("\n")
    print(f"\n{total - len(failed)} passed, {len(failed)} failed, {total} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
