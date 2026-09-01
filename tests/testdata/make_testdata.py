"""Build the tiny synthetic system the unit tests run against.

    python tests/testdata/make_testdata.py

Twelve CA atoms interpolated from a compact turn to an extended chain, which gives a
monotonic spread in both RMSD to the compact reference and radius of gyration. That is
enough signal for the CV, likelihood and reweighting layers to be checked against an
answer known in advance, and small enough that a full pass costs seconds rather than
hours. The files are regenerated rather than edited, so the seed below fixes them.
"""
import os

import numpy as np
import mdtraj as md
from mdtraj.core import element

HERE = os.path.dirname(os.path.abspath(__file__))
N_RESIDUES = 12
N_FRAMES = 40
N_TARGET = 12
SEED = 20260825
NOISE_NM = 0.004


def build_topology(n_residues):
    """A single chain of n_residues alanines represented by their CA atom alone."""
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_residues):
        residue = top.add_residue("ALA", chain, resSeq=i + 1)
        top.add_atom("CA", element.carbon, residue)
    return top


def arc_coordinates(n, total_angle, spacing=0.38):
    """A chain of n atoms on a circular arc of fixed contour length, bent by total_angle.

    The chain opens from a closed ring at 2*pi to a straight rod as the angle goes to
    zero, and because the contour length is fixed the radius of gyration rises with every
    step of that opening. Interpolating between a fixed compact structure and a fixed
    extended one was tried first and rejected: it shrinks the structure across the chain
    faster than it lengthens it, so the radius of gyration dips before it rises.
    """
    length = spacing * (n - 1)
    if total_angle < 1e-6:
        return np.column_stack([np.linspace(0.0, length, n), np.zeros(n), np.zeros(n)])
    radius = length / total_angle
    angles = np.linspace(0.0, total_angle, n)
    return np.column_stack([radius * np.sin(angles),
                            radius * (1.0 - np.cos(angles)),
                            np.zeros(n)])


def main():
    rng = np.random.default_rng(SEED)
    top = build_topology(N_RESIDUES)
    # Frame k unbends a little further than frame k-1, so the CV of frame k is ordered by
    # k and every test can predict which frames a target at either end favours.
    # The sweep stops short of a straight rod because the radius of gyration stops
    # changing there, and the last frames would then be ordered by noise alone.
    angles = np.linspace(2.0 * np.pi, 0.6, N_FRAMES)
    xyz = np.stack([arc_coordinates(N_RESIDUES, a) for a in angles])
    xyz += rng.normal(scale=NOISE_NM, size=xyz.shape)
    ensemble = md.Trajectory(xyz.astype(np.float32), top)
    # The reference is the closed ring, which is what RMSD is measured against.
    reference = arc_coordinates(N_RESIDUES, 2.0 * np.pi)
    md.Trajectory(reference.astype(np.float32)[None, :, :], top).save_pdb(
        os.path.join(HERE, "toy.pdb"))
    ensemble.save_dcd(os.path.join(HERE, "toy_ensemble.dcd"))
    # The target is the extended tail of the same ensemble, so reweighting towards it has
    # a known right answer: weight should move to the last frames.
    ensemble[-N_TARGET:].save_dcd(os.path.join(HERE, "toy_target.dcd"))
    for name in ("toy.pdb", "toy_ensemble.dcd", "toy_target.dcd"):
        path = os.path.join(HERE, name)
        print(f"  {name:20s} {os.path.getsize(path):>8d} bytes")


if __name__ == "__main__":
    main()
