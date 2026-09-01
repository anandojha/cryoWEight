# Ensemble reweighting using cryo-EM particles
import warnings

warnings.filterwarnings("ignore")
# Imported relatively when this file is used as part of the installed package, and
# by bare name when assemble has copied it into a run directory, where a compute
# node executes it as a loose script with no package around it.
try:
    from .cryoER_core import expectation_maximization_weights, evaluate_nll, normalize_weights
except ImportError:
    from cryoER_core import expectation_maximization_weights, evaluate_nll, normalize_weights
try:
    from . import likelihoods
except ImportError:
    import likelihoods
from openmm.app import (
    PDBFile,
    Modeller,
    Simulation,
    DCDReporter,
    PDBReporter,
    StateDataReporter,
    ForceField,
    PME,
    HBonds,
    CutoffNonPeriodic,
)
from openmm import LangevinIntegrator, Platform, LocalEnergyMinimizer, XmlSerializer
from openmm.unit import kelvin, picoseconds, nanometer, kilojoule_per_mole
from scipy.ndimage import minimum_filter, generate_binary_structure
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KernelDensity
from matplotlib.colors import BoundaryNorm
from MDAnalysis.analysis import align
from matplotlib.patches import Patch
from scipy.special import logsumexp
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import MDAnalysis as mda
from tqdm import tqdm
import pandas as pd
import mdtraj as md
import numpy as np
import subprocess
import shutil
import pickle
import heapq
import torch
import time
import glob
import sys
import csv
import os
import json
import io

try:
    from . import cv_families
except ImportError:
    import cv_families
try:
    from . import build_system
except ImportError:
    import build_system

# The per system configuration (topology and reference names, CV family, image settings,
# axis ranges, solvent model, orchestration paths ...). Assembled from the
# reweight_config block of the system config into the script working directory.
# The configuration is a module global because every stage below reads it, and it is
# loaded on demand rather than at import so that a test can supply its own.
CFG = {}


def load_config(path="reweight_config.json"):
    """Read the per run configuration into the module global the pipeline stages use."""
    global CFG
    with io.open(path, encoding="utf-8") as handle:
        CFG = json.load(handle)
    return CFG


def setup_directories(data_dir, output_directory):
    """Create the data and output directories if they do not exist."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    else:
        print(f"Output directory already exists: {output_directory}")


def get_init_traj(
    data_dir, output_directory, dcd_file, topology_file, dcd_stripped_file, max_frames
):
    """Load the seeding trajectory, strip solvent and ions, and save the stripped copy."""
    traj_path = os.path.join(data_dir, dcd_file)
    top_path = os.path.join(data_dir, topology_file)
    traj_output_path = os.path.join(output_directory, dcd_stripped_file)
    traj = md.load(traj_path, top=top_path)
    total_frames = traj.n_frames
    print(f"Original trajectory frames: {total_frames}")
    # Strip the trajectory
    solute_atoms = traj.topology.select(CFG["strip_selection"])
    stripped_traj = traj.atom_slice(solute_atoms)
    # Truncate if requested
    if max_frames is not None:
        truncated_traj = stripped_traj[: min(max_frames, stripped_traj.n_frames)]
        print(f"Truncated trajectory to first {min(max_frames, stripped_traj.n_frames)} frames")
    else:
        truncated_traj = stripped_traj
    truncated_traj.save_dcd(traj_output_path)


def plot_free_energy(
    data_dir, output_directory, trajectory_file, topology_file, kB, T, output_file, nbins
):
    """Plot the free energy surface of a trajectory over the collective variables."""
    traj_path = os.path.join(data_dir, trajectory_file)
    top_path = os.path.join(data_dir, topology_file)
    fig_output_path = os.path.join(output_directory, output_file)
    traj = md.load(traj_path, top=top_path)
    ref = md.load(top_path)
    cv = cv_families.cv_of(traj, ref, CFG)
    rmsd = cv[:, 0]
    rg = cv[:, 1]
    hist, xedges, yedges = np.histogram2d(rmsd, rg, bins=nbins, density=True)
    # Compute the free energy surface F = -kB T ln P
    F = -kB * T * np.log(hist + 1e-12)
    F -= F.min()  # shift minimum to zero
    # Map each frame to its free energy
    xidx = np.clip(np.digitize(rmsd, xedges) - 1, 0, nbins[0] - 1)
    yidx = np.clip(np.digitize(rg, yedges) - 1, 0, nbins[1] - 1)
    fe_point = F[xidx, yidx]
    # Scatter plot of F(rmsd, Rg)
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(rmsd, rg, c=fe_point, cmap="jet", s=20, alpha=0.8, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Free Energy (kJ/mol)", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax.set_xlabel(CFG["cv_xlabel"], fontsize=20)
    ax.set_ylabel(CFG["cv_ylabel"], fontsize=20)
    ax.set_xlim(CFG["xmin"], CFG["xmax"])
    ax.set_ylim(CFG["ymin"], CFG["ymax"])
    ax.tick_params(labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_output_path, dpi=500)
    plt.close(fig)


def plot_overlap(
    data_dir,
    output_directory,
    topology,
    initial_md,
    selected_traj,
    output_file,
    bw,
    n_levels,
    nx,
    ny,
    xmin,
    xmax,
    ymin,
    ymax,
):
    """Overlay density contours of the seeding ensemble and the selected target ensemble."""
    top_path = os.path.join(data_dir, topology)
    init_path = os.path.join(output_directory, initial_md)
    sel_path = os.path.join(data_dir, selected_traj)
    out_path = os.path.join(output_directory, output_file)
    ref = md.load(top_path)
    init_tr = md.load(init_path, top=top_path)
    sel_tr = md.load(sel_path, top=top_path)
    cv_init = cv_families.cv_of(init_tr, ref, CFG)
    cv_sel = cv_families.cv_of(sel_tr, ref, CFG)
    rmsd_init, rg_init = cv_init[:, 0], cv_init[:, 1]
    rmsd_sel, rg_sel = cv_sel[:, 0], cv_sel[:, 1]
    # Fit 2D Gaussian KDEs
    data_init = np.vstack([rmsd_init, rg_init]).T
    data_sel = np.vstack([rmsd_sel, rg_sel]).T
    kde_init = KernelDensity(kernel="tophat", bandwidth=bw).fit(data_init)
    kde_sel = KernelDensity(kernel="tophat", bandwidth=bw).fit(data_sel)
    # Prepare a fine grid
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xgrid, ygrid)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    # Evaluate densities
    Z_init = np.exp(kde_init.score_samples(pts)).reshape(ny, nx)
    Z_sel = np.exp(kde_sel.score_samples(pts)).reshape(ny, nx)
    # Choose levels *without* skipping
    levels_init = np.linspace(Z_init.min(), Z_init.max(), n_levels)[1:]
    levels_sel = np.linspace(Z_sel.min(), Z_sel.max(), n_levels)[1:]
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")
    # dashed contours only
    ax.contour(
        X,
        Y,
        Z_init,
        levels=levels_init,
        colors="darkgreen",
        linestyles="--",
        linewidths=1.0,
        zorder=1,
    )
    ax.contour(
        X,
        Y,
        Z_sel,
        levels=levels_sel,
        colors="darkorange",
        linestyles="--",
        linewidths=1.0,
        zorder=2,
    )
    # legend
    legend_elems = [
        Patch(edgecolor="darkgreen", facecolor="white", linestyle="--", label="MD Distribution"),
        Patch(
            edgecolor="darkorange", facecolor="white", linestyle="--", label="Selected Distribution"
        ),
    ]
    ax.legend(handles=legend_elems, fontsize=12, loc="upper right")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(CFG["cv_xlabel"], fontsize=16)
    ax.set_ylabel(CFG["cv_ylabel"], fontsize=16)
    ax.tick_params(labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=500)
    plt.close(fig)


def initialize(
    data_dir,
    output_directory,
    traj_file,
    top_file,
    ref_file,
    bin_width,
    output_dcd,
    index_file,
    frame_count_file,
):
    """Select one representative frame per occupied collective variable bin."""
    traj_path = os.path.join(output_directory, traj_file)
    top_path = os.path.join(data_dir, top_file)
    ref_path = os.path.join(data_dir, ref_file)
    output_dcd_path = os.path.join(output_directory, output_dcd)
    index_file_path = os.path.join(output_directory, index_file)
    frame_count_file_path = os.path.join(output_directory, frame_count_file)
    traj = md.load(traj_path, top=top_path)
    ref = md.load(ref_path, top=top_path)  # Load reference structure
    num_frames = traj.n_frames  # Get the number of frames
    print(f"Total number of frames at the beginning: {num_frames}")
    cv = cv_families.cv_of(traj, ref, CFG)
    rmsd = cv[:, 0]
    rg = cv[:, 1]
    rmsd_min, rmsd_max = np.min(rmsd), np.max(rmsd)
    rg_min, rg_max = np.min(rg), np.max(rg)
    rmsd_bins = np.arange(rmsd_min, rmsd_max + bin_width, bin_width)
    rg_bins = np.arange(rg_min, rg_max + bin_width, bin_width)
    bin_indices = np.digitize(rmsd, rmsd_bins), np.digitize(rg, rg_bins)
    # Dictionary to store frame indices per bin
    bin_dict = {}
    for i, (r_bin, g_bin) in enumerate(zip(*bin_indices)):
        bin_key = (r_bin, g_bin)
        if bin_key not in bin_dict:
            bin_dict[bin_key] = []
        bin_dict[bin_key].append(i)
    # Select one representative frame per bin
    selected_indices = []
    for frames in bin_dict.values():
        selected_indices.extend(frames[:1])
    selected_traj = traj[selected_indices]
    selected_traj.save_dcd(output_dcd_path)
    np.savetxt(index_file_path, selected_indices, fmt="%d")
    with open(frame_count_file_path, "w") as f:
        f.write(str(len(selected_indices)))
    plt.figure(figsize=(8, 6))
    plt.scatter(rmsd, rg, color="black", s=30, alpha=0.8, label="All Frames")
    plt.scatter(
        rmsd[selected_indices],
        rg[selected_indices],
        color="red",
        s=30,
        alpha=0.8,
        label="Selected Frames",
        linewidth=1.5,
    )
    # Labels and aesthetics
    plt.xlabel(CFG["cv_xlabel"], fontsize=14)
    plt.ylabel(CFG["cv_ylabel"], fontsize=14)
    plt.legend(frameon=False, fontsize=12)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(os.path.join(output_directory, "initialized_cvs.png"), dpi=500)
    # plt.show()
    num_selected = len(selected_indices)
    print(f"Total number of selected frames at the end: {num_selected}")
    print(f"Saved {num_selected} frames to {output_dcd_path}")
    print(f"Saved selected indices to {index_file_path}")
    print(f"Saved frame count to {frame_count_file_path}")


def save_every_nth_frame(data_dir, output_directory, traj_file, top_file, output_dcd, step):
    """Write every nth frame of a trajectory to a new DCD file."""
    traj_path = os.path.join(data_dir, traj_file)
    top_path = os.path.join(data_dir, top_file)
    output_dcd_path = os.path.join(output_directory, output_dcd)
    trajectory = md.load(traj_path, top=top_path)
    print(f"Total number of frames in the trajectory: {trajectory.n_frames}")
    trajectory[::step].save(output_dcd_path)
    saved_trajectory = md.load(output_dcd_path, top=top_path)
    print(f"Total number of frames in {output_dcd}: {saved_trajectory.n_frames}")


def generate_weights_from_file(data_dir, output_directory, frame_count_file, output_file):
    """Write a uniform weight for every frame of the trajectory."""
    frame_count_path = os.path.join(output_directory, frame_count_file)
    output_path = os.path.join(output_directory, output_file)
    try:
        with open(frame_count_path, "r") as file:
            number = file.read().strip()
            num_frames = float(number) if "." in number else int(number)
    except Exception as e:
        print(f"Error reading frame count file: {e}")
        return None
    # Ensure valid frame count
    if num_frames <= 0:
        print("Invalid number of frames. Must be greater than zero.")
        return None
    weight = 1 / num_frames
    try:
        with open(output_path, "w") as file:
            for _ in range(num_frames):
                file.write(f"{weight:.6f}\n")
        print(f"Generated {num_frames} weights and saved to {output_path}")
    except Exception as e:
        print(f"Error writing weights file: {e}")
        return None


def process_weights_from_sim(
    data_dir,
    output_directory,
    traj_initial_weights,
    simulation_top,
    md_simulation_traj,
    bin_width,
    bin_coordinates_weights,
    occupied_bins_summary,
    final_bins_file,
    merged_details_file,
    merged_details_pickle,
    simulation_traj,
    initial_weights,
    plot_filename,
):
    """Bin the weighted trajectory in the collective variable space and merge each bin onto one representative frame."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    traj_initial_weights = os.path.join(output_directory, traj_initial_weights)
    simulation_top = os.path.join(data_dir, simulation_top)
    md_simulation_traj = os.path.join(output_directory, md_simulation_traj)
    final_bins_file = os.path.join(output_directory, final_bins_file)
    bin_coordinates_weights = os.path.join(output_directory, bin_coordinates_weights)
    occupied_bins_summary = os.path.join(output_directory, occupied_bins_summary)
    simulation_traj = os.path.join(output_directory, simulation_traj)
    plot_filename = os.path.join(output_directory, plot_filename)
    merged_details_file = os.path.join(output_directory, merged_details_file)
    merged_details_pickle = os.path.join(output_directory, merged_details_pickle)
    initial_weights = os.path.join(output_directory, initial_weights)
    init_weights = np.loadtxt(traj_initial_weights)  # Array of weights, one per frame
    simulation_reference = md.load(simulation_top)
    traj = md.load(md_simulation_traj, top=simulation_top)
    cv = cv_families.cv_of(traj, simulation_reference, CFG)
    rmsd = list(cv[:, 0])
    rg = list(cv[:, 1])
    print(f"Number of frames in the initial simulation trajectory: {len(rmsd)}")
    # Binning RMSD and Rg values
    print(f"Initial weights shape: {init_weights.shape}")
    x_bin_edges = np.arange(CFG["bin_x_min"], CFG["bin_x_max"] + bin_width, bin_width)
    y_bin_edges = np.arange(CFG["bin_y_min"], CFG["bin_y_max"] + bin_width, bin_width)
    num_x_bins = len(x_bin_edges) - 1
    num_y_bins = len(y_bin_edges) - 1
    total_bins = num_x_bins * num_y_bins
    print(
        f"Total number of bins: {total_bins} ({num_x_bins} x {num_y_bins}) with bin width {bin_width}"
    )
    # Assign each CV value to a bin (cfg.bin_digitize_right / bin_clip reproduce the
    # adk right=True+clip path while keeping rmsd_rg systems at right=False, no clip)
    _right = CFG.get("bin_digitize_right", False)
    rmsd_bin_indices = np.digitize(rmsd, x_bin_edges, right=_right) - 1  # bin indices along x
    rg_bin_indices = np.digitize(rg, y_bin_edges, right=_right) - 1  # bin indices along y
    if CFG.get("bin_clip", False):
        rmsd_bin_indices = np.clip(rmsd_bin_indices, 0, num_x_bins - 1).astype(int)
        rg_bin_indices = np.clip(rg_bin_indices, 0, num_y_bins - 1).astype(int)
    # Dictionary to track bin occupancy
    bin_occupancy = {}
    # Track bin occupancy for all frames (needed for subsequent steps)
    for i in range(len(rmsd)):
        rmsd_bin = rmsd_bin_indices[i]
        rg_bin = rg_bin_indices[i]
        weight = init_weights[i]
        bin_coord = (rmsd_bin, rg_bin)
        if bin_coord not in bin_occupancy:
            bin_occupancy[bin_coord] = []
        bin_occupancy[bin_coord].append((i, weight))
    with open(bin_coordinates_weights, "w") as f:
        for i in range(len(rmsd)):
            rmsd_bin = rmsd_bin_indices[i]
            rg_bin = rg_bin_indices[i]
            weight = init_weights[i]
            f.write(f"Frame {i}: RMSD Bin {rmsd_bin}, Rg Bin {rg_bin}, Weight: {weight:.10e}\n")
    with open(occupied_bins_summary, "w") as f:
        for bin_coord, indices in bin_occupancy.items():
            f.write(f"Bin {bin_coord}: {len(indices)} structures\n")
    print(f"Data saved to {bin_coordinates_weights} and {occupied_bins_summary}")
    occupancy_grid = np.zeros((num_x_bins, num_y_bins))
    # Populate the occupancy grid with the count of points in each bin
    for bin_coord, indices in bin_occupancy.items():
        rmsd_bin, rg_bin = bin_coord
        occupancy_grid[rmsd_bin, rg_bin] = len(indices)
    cmap = plt.cm.plasma
    cmap.set_under("white")
    plt.figure(figsize=(8, 6), facecolor="white")
    im = plt.imshow(
        occupancy_grid.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[CFG["bin_x_min"], CFG["bin_x_max"], CFG["bin_y_min"], CFG["bin_y_max"]],
        vmin=1,
        vmax=occupancy_grid.max(),
    )
    colorbar = plt.colorbar(im, label="Number of Frames per Bin")
    colorbar.set_ticks(range(1, int(occupancy_grid.max()) + 1))
    colorbar.ax.yaxis.label.set_size(16)
    colorbar.ax.tick_params(labelsize=10)
    # Labels and formatting
    plt.xlabel(CFG["cv_xlabel"], fontsize=16)
    plt.ylabel(CFG["cv_ylabel"], fontsize=16)
    plt.xticks(np.arange(*CFG["heatmap_xticks"]), fontsize=16)
    plt.yticks(np.arange(*CFG["heatmap_yticks"]), fontsize=16)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.savefig(plot_filename, dpi=500, bbox_inches="tight")
    # plt.show()
    # Merging frames in the same bin
    bin_mapping = defaultdict(list)
    # Populate the bin mapping with frame indices and weights
    for i in range(len(rmsd_bin_indices)):
        bin_coord = (rmsd_bin_indices[i], rg_bin_indices[i])
        bin_mapping[bin_coord].append((i, init_weights[i]))
    final_bins = []  # Stores the selected representative frame and its merged weight
    merged_details = []  # Stores all original indices in the bin and the total merged weight
    # Process each occupied bin
    for bin_coord, indices_weights in bin_mapping.items():
        if len(indices_weights) == 1:
            index, weight = indices_weights[0]
            final_bins.append([index, weight])
            merged_details.append([bin_coord, [index], [weight]])
        else:
            total_weight = sum(weight for _, weight in indices_weights)
            selected_index = np.random.choice([idx for idx, _ in indices_weights])
            final_bins.append([selected_index, total_weight])
            merged_details.append([bin_coord, [idx for idx, _ in indices_weights], [total_weight]])
    np.savetxt(final_bins_file, final_bins, fmt="%d %.10e", header="FrameIndex MergedWeight")
    with open(merged_details_pickle, "wb") as f:
        pickle.dump(merged_details, f)
    print(f"Data saved to {merged_details_pickle}")
    with open(merged_details_pickle, "rb") as f:
        loaded_merged_details = pickle.load(f)
    print(f"Loaded data from {merged_details_pickle}")
    # Verify that the loaded data matches the original
    if merged_details == loaded_merged_details:
        print("The loaded data is identical to the original.")
    else:
        print("The loaded data is different! Check saving/loading issues.")
    with open(merged_details_file, "w") as f:
        f.write("BinCoord AllIndices TotalWeight\n")
        for entry in merged_details:
            bin_coord, indices, total_weight = entry
            indices_str = ",".join(map(str, indices))
            total_weight_str = ",".join(map(lambda x: f"{x:.10e}", total_weight))
            f.write(f"{bin_coord} {indices_str} {total_weight_str}\n")
    print(f"Data saved to {final_bins_file} and {merged_details_file}")
    # Extracting and saving selected frames and weights
    selected_indices = [entry[0] for entry in final_bins]
    traj = md.load(md_simulation_traj, top=simulation_top)
    selected_traj = traj.slice(selected_indices)
    selected_traj.save_dcd(simulation_traj)
    # Reload the saved DCD trajectory for verification
    loaded_traj = md.load(simulation_traj, top=simulation_top)
    # Verify the number of frames in the extracted trajectory
    print(f"Length of selected_indices: {len(selected_indices)}")
    print(f"Number of frames in loaded DCD: {loaded_traj.n_frames}")
    merged_weights = [entry[1] for entry in final_bins]
    # Verify the lengths
    print(f"Length of selected_indices: {len(selected_indices)}")
    print(f"Length of selected_weights: {len(merged_weights)}")
    with open(initial_weights, "w") as f:
        for weight in merged_weights:
            f.write(f"{weight:.32e}\n")
    print(f"Merged initial weights saved to {initial_weights}")
    print(f"Selected frames for reweighting saved to {simulation_traj}")


# Beginning of reweighting functions


def check_cuda():
    """Report whether CUDA is available and which device is in use."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_status = "CUDA is available" if torch.cuda.is_available() else "CUDA is not available"
    print("The device in use is", device)
    print(cuda_status)


def count_conformations(
    data_dir, output_directory, reference_traj, reference_top, simulation_traj, simulation_top
):
    """Print the number of frames in the reference and simulation trajectories."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    reference_traj = os.path.join(output_directory, reference_traj)
    reference_top = os.path.join(data_dir, reference_top)
    simulation_traj = os.path.join(output_directory, simulation_traj)
    simulation_top = os.path.join(data_dir, simulation_top)
    u_image = mda.Universe(reference_top, reference_traj)
    n_image_frames = len(u_image.trajectory)
    u_struc = mda.Universe(simulation_top, simulation_traj)
    n_struc_frames = len(u_struc.trajectory)
    print(f"Number of conformations in reference.dcd: {n_image_frames}")
    print(f"Number of conformations in simulaton.dcd: {n_struc_frames}")
    print(n_image_frames, n_struc_frames)


def check_device():
    """Return the device torch will run on, cuda when a GPU is available and cpu otherwise."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def mdau_to_pos_arr(u):
    """Return the CA coordinates of a Universe as a tensor of shape (n_frames, n_atoms, 3)."""
    protein_CA = u.select_atoms("protein and name CA")  # Select alpha carbons (CA)
    pos = torch.zeros((len(u.trajectory), len(protein_CA), 3), dtype=float)  # Initialize tensor
    for i, ts in enumerate(u.trajectory):
        pos[i] = torch.from_numpy(protein_CA.positions)  # Fill tensor with positions
    pos -= pos.mean(1).unsqueeze(1)  # Center each frame by subtracting the mean position
    return pos


def get_clusters(
    data_dir, output_directory, reference_top, reference_traj, simulation_top, simulation_traj
):
    """Report the structure and image counts and plot a sample of the cluster centres."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    reference_top = os.path.join(data_dir, reference_top)
    reference_traj = os.path.join(output_directory, reference_traj)
    simulation_top = os.path.join(data_dir, simulation_top)
    simulation_traj = os.path.join(output_directory, simulation_traj)
    uImage = mda.Universe(reference_top, reference_traj)
    uStruc = mda.Universe(simulation_top, simulation_traj)
    posStruc = mdau_to_pos_arr(uStruc)
    posImage = mdau_to_pos_arr(uImage)
    # Both position tensors must already be centred, because the image comparison places
    # every structure at the origin and would otherwise measure a translation.
    assert torch.allclose(
        posStruc.mean(1), torch.zeros_like(posStruc.mean(1))
    ), "posStruc not centered"
    assert torch.allclose(
        posImage.mean(1), torch.zeros_like(posImage.mean(1))
    ), "posImage not centered"
    nStruc = posStruc.shape[0]
    nImage = posImage.shape[0]
    print(f"Number of reference structures: {nStruc}")
    print(f"Number of images (1 synthetic image per structure): {nImage}")
    nCluster = len(mda.Universe(simulation_top, simulation_traj).trajectory)  # Number of clusters
    print("Number of cluster centers: ", nCluster)
    fig = plt.figure(figsize=(15, 10), dpi=500)
    num_clusters_to_plot = 16
    selected_indices = np.linspace(
        0, nCluster - 1, num_clusters_to_plot, dtype=int
    )  # Select 16 structures for plotting
    for i, idx in enumerate(selected_indices):
        ax = fig.add_subplot(
            4, 4, i + 1, aspect="equal", projection="3d", proj_type="ortho"
        )  # 3D subplot for each structure
        nAtom = posStruc[idx].shape[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, nAtom))  # Assign unique color for each atom
        for j in range(nAtom):
            ax.scatter(
                posStruc[idx, j, 0],
                posStruc[idx, j, 1],
                posStruc[idx, j, 2],
                s=10,
                c=colors[j],
                marker="o",
            )
            if j > 0:
                ax.plot(
                    [posStruc[idx, j - 1, 0], posStruc[idx, j, 0]],
                    [posStruc[idx, j - 1, 1], posStruc[idx, j, 1]],
                    [posStruc[idx, j - 1, 2], posStruc[idx, j, 2]],
                    color=colors[j],
                    linewidth=2,
                )  # Backbone connection
            # Connect atoms within 6 Å to visualize possible interactions
            if j < nAtom - 1:
                for k in range(j + 1, nAtom):
                    dist = torch.norm(posStruc[idx, j] - posStruc[idx, k])
                    if dist < 6.0:
                        ax.plot(
                            [posStruc[idx, j, 0], posStruc[idx, k, 0]],
                            [posStruc[idx, j, 1], posStruc[idx, k, 1]],
                            [posStruc[idx, j, 2], posStruc[idx, k, 2]],
                            color=plt.cm.Greys(dist / 12.0),
                            linewidth=0.5,
                        )  # Scale connection color by distance
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(-CFG["cluster_axis_limit"], CFG["cluster_axis_limit"])
        ax.set_ylim(-CFG["cluster_axis_limit"], CFG["cluster_axis_limit"])
        ax.set_zlim(-CFG["cluster_axis_limit"], CFG["cluster_axis_limit"])
        ax.set_title(f"Cluster {idx}")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.09)
    plot_filename = os.path.join(output_directory, "clust_struct.png")
    plt.savefig(plot_filename, dpi=500, bbox_inches="tight")
    # plt.show()


def _plot_distance_matrix(diff, obs_label, dist_label, path):
    """Heatmap of the structure by observation squared distance matrix."""
    n_struct, n_obs = diff.shape
    fig = plt.figure(figsize=(12, 8), dpi=500)
    ax = fig.add_subplot(111)
    cax = ax.imshow(
        diff,
        cmap="afmhot",
        origin="lower",
        aspect="auto",
        interpolation="none",
        extent=[0.5, n_obs + 0.5, 0.5, n_struct + 0.5],
        vmin=0,
        vmax=np.percentile(diff, 99),
    )
    ax.set_xticks(np.linspace(0, n_obs, min(10, max(n_obs, 2)), dtype=int))
    ax.set_yticks(np.linspace(0, n_struct, min(10, max(n_struct, 2)), dtype=int))
    ax.set_xlabel(obs_label, fontsize=16)
    ax.set_ylabel("Structure index", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    cbar = fig.colorbar(cax)
    cbar.set_label(dist_label, fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def _plot_per_structure_histogram(values, x_label, path):
    """One density curve per structure, coloured by structure index."""
    # The upper edge follows the 99th percentile so the same code frames both the raw
    # distances and the negative log likelihood, whose ranges differ by orders of magnitude.
    bins = np.linspace(0.0, float(np.percentile(values, 99)) + 1e-6, 101)
    n_struct = values.shape[0]
    fig = plt.figure(figsize=(12, 8), dpi=500)
    ax = fig.add_subplot(111)
    colors = plt.cm.plasma(np.linspace(0, 1, n_struct))
    for i in range(n_struct):
        hist, edges = np.histogram(values[i], bins=bins, density=True)
        ax.plot(edges[:-1], hist, color=colors[i], alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=max(n_struct - 1, 1)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Structure index", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def reweight(
    ctx,
    img_struc_dist_file,
    hist_img_struct_dist_file,
    hist_neg_lkhood_file,
    initial_weights,
    rescaled_weights_file,
):
    """Reweight the ensemble by expectation maximization against the configured likelihood."""
    out = ctx["output_directory"]
    obs_label, dist_label = likelihoods.labels_of(ctx["cfg"])
    diff, scale = likelihoods.distance_of(ctx)
    _plot_distance_matrix(diff, obs_label, dist_label, os.path.join(out, img_struc_dist_file))
    _plot_per_structure_histogram(diff, dist_label, os.path.join(out, hist_img_struct_dist_file))
    # A Gaussian likelihood of width scale turns a squared distance into a log probability,
    # so the negative log likelihood is the distance over twice the squared scale.
    nll = diff / (2 * scale**2)
    _plot_per_structure_histogram(
        nll, "Negative log likelihood", os.path.join(out, hist_neg_lkhood_file)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    log_lik_mat = torch.from_numpy((-nll).astype(np.float32)).to(device)
    print("The shape of log likelihood matrix: ", log_lik_mat.shape)
    init_weights = np.loadtxt(os.path.join(out, initial_weights))
    print("Length of the weights list: ", len(init_weights))
    init_weights /= np.sum(init_weights)
    log_weights = torch.log(torch.from_numpy(init_weights.astype(np.float32))).to(device)
    n_iter = int(ctx["cfg"].get("em_iterations", 5000))
    log_weights, loss = expectation_maximization_weights(log_lik_mat.T, log_weights, n_iter)
    rescaled_weight = torch.exp(log_weights).cpu().numpy()
    rescaled_weight /= np.sum(rescaled_weight)
    print("Final rescaled weights shape:", rescaled_weight.shape)
    print("Sum of rescaled_weights:", np.sum(rescaled_weight))
    np.savetxt(os.path.join(out, rescaled_weights_file), rescaled_weight, fmt="%.6e")


# End of reweighting functions


def rescale_weights(
    data_dir,
    output_directory,
    rescaled_weights_file,
    merged_details_pickle,
    rescaled_merged_details_file,
    sorted_indices_weights_file,
    rescaled_weights_all,
):
    """Redistribute the optimized bin weights back onto the frames merged into each bin."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    rescaled_weights_file = os.path.join(output_directory, rescaled_weights_file)
    merged_details_pickle = os.path.join(output_directory, merged_details_pickle)
    rescaled_merged_details_file = os.path.join(output_directory, rescaled_merged_details_file)
    sorted_indices_weights_file = os.path.join(output_directory, sorted_indices_weights_file)
    rescaled_weights_all = os.path.join(output_directory, rescaled_weights_all)
    # Loading and Updating Rescaled Weights
    rescaled_weights = []
    with open(rescaled_weights_file, "r") as f:
        rescaled_weights = [
            max(float(line.strip()), 1e-32) for line in f
        ]  # Ensure no weight is below 1e-32
    # Verify the length and sum of rescaled weights
    print(f"Length of rescaled_weights: {len(rescaled_weights)}")
    sum_of_weights = sum(rescaled_weights)
    print(f"Sum of rescaled_weights: {sum_of_weights}")
    with open(merged_details_pickle, "rb") as f:
        merged_details = pickle.load(f)
    print(f"Loaded data from {merged_details_pickle}")
    # Check for length mismatches
    print(f"Length of rescaled_weights: {len(rescaled_weights)}")
    print(f"Length of merged_details: {len(merged_details)}")
    # Ensure correct length before updating merged_details
    if len(rescaled_weights) == len(merged_details):
        for i in range(len(merged_details)):
            merged_details[i][2] = [
                rescaled_weights[i]
            ]  # Replace old weight with new rescaled weight
        print("Updated merged_details with rescaled weights:")
        for entry in merged_details[:5]:  # Print only first 5 entries for verification
            print(entry)
    else:
        print(
            "Error: Length mismatch detected. Ensure rescaled weights has the correct number of entries."
        )
    # Reversing the merging and normalizing final weights
    reversed_details = []
    for bin_coord, indices, weight in merged_details:  # Unpack bin, indices, and weight
        if len(indices) == 1:
            # If there is only one index, keep the weight as it is
            reversed_details.append({"Bin": bin_coord, "Indices": indices, "Weights": weight})
        else:
            # If multiple indices exist, split the weight equally among them
            split_weight = [weight[0] / len(indices)] * len(indices)
            reversed_details.append({"Bin": bin_coord, "Indices": indices, "Weights": split_weight})
    with open(rescaled_merged_details_file, "w") as f:
        # Parsed back by the convergence stop with ast.literal_eval, so entries are
        # cast to plain Python types before writing.
        for entry in reversed_details:
            plain = {
                "Bin": tuple(int(b) for b in entry["Bin"]),
                "Indices": [int(i) for i in entry["Indices"]],
                "Weights": [float(w) for w in entry["Weights"]],
            }
            f.write(f"{plain}\n")
    # Flatten the reversed details into separate index and weight pairs
    flattened_details = []
    for entry in reversed_details:
        indices = entry["Indices"]
        weights = entry["Weights"]
        for i in range(len(indices)):
            flattened_details.append((indices[i], weights[i]))
    # Sort the flattened details by index in ascending order
    sorted_flattened_details = sorted(flattened_details, key=lambda x: x[0])
    with open(sorted_indices_weights_file, "w") as f:
        for entry in sorted_flattened_details[:]:
            f.write(f"{entry}\n")
    sorted_indices = [entry[0] for entry in sorted_flattened_details]
    sorted_weights = [entry[1] for entry in sorted_flattened_details]
    # Normalize the weights to ensure they sum to 1
    total_weight = sum(sorted_weights)
    normalized_weights = [w / total_weight for w in sorted_weights]
    with open(rescaled_weights_all, "w") as f:
        for weight in normalized_weights:
            f.write(f"{weight:.32e}\n")
    # Verify the sum of the final weights
    print(f"\nSum of weights: {sum(normalized_weights):.32e}")
    print(f"Data saved to  {rescaled_merged_details_file} and {sorted_indices_weights_file}")
    print(f"Final rescaled weights saved to {rescaled_weights_all}")


def get_rescaled_plots(
    data_dir,
    output_directory,
    init_weights,
    rescaled_weights,
    simulation_top,
    simulation_traj,
    n_bins,
):
    """Plot the initial and the rescaled weights over the collective variable space."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    simulation_traj = os.path.join(output_directory, simulation_traj)
    simulation_top = os.path.join(data_dir, simulation_top)
    init_weights = os.path.join(output_directory, init_weights)
    rescaled_weights = os.path.join(output_directory, rescaled_weights)
    init_weights = np.loadtxt(init_weights)
    print("Length of the initial weights list: ", len(init_weights))
    rescaled_weights = np.loadtxt(rescaled_weights)
    print("Length of the rescaled weights list: ", len(rescaled_weights))
    ref = md.load(simulation_top)  # Load the simulation topology
    traj = md.load(simulation_traj, top=simulation_top)  # Load the simulation trajectory
    cv = cv_families.cv_of(traj, ref, CFG)
    rmsd_values = cv[:, 0]
    rg_values = cv[:, 1]
    # Ensure cv1 and cv2 have the same length
    cv1 = rmsd_values
    cv2 = rg_values
    assert len(cv1) == len(
        cv2
    ), f"cv1 and cv2 must have the same shape. Got {len(cv1)} and {len(cv2)}"
    print(f"Length of RMSD: {len(cv1)}, Length of Rg: {len(cv2)}")
    n_bins = n_bins
    xedges = np.linspace(np.min(cv1) - 0.1, np.max(cv1) + 0.1, n_bins)
    yedges = np.linspace(np.min(cv2) - 0.1, np.max(cv2) + 0.1, n_bins)
    H1, _, _ = np.histogram2d(
        cv1, cv2, bins=(xedges, yedges), density=True, weights=rescaled_weights
    )
    H2, _, _ = np.histogram2d(cv1, cv2, bins=(xedges, yedges), density=True, weights=init_weights)
    # Determine a common vmax for both plots
    common_vmax = max(np.max(H1), np.max(H2))
    rmsd_values = cv1
    rg_values = cv2
    weights = init_weights
    assert len(rmsd_values) == len(rg_values) == len(weights), "Arrays must have the same length"
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
    constant_size = 1000
    sc = ax.scatter(
        rmsd_values,
        rg_values,
        s=constant_size,
        c=weights,
        cmap="plasma",
        alpha=1,
        edgecolors="black",
        vmin=0,
        vmax=1,
    )
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Weights", fontsize=16)
    cbar.mappable.set_clim(0, 1)
    ax.set_xlabel(CFG["cv_xlabel"], fontsize=16)
    ax.set_ylabel(CFG["cv_ylabel"], fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "init_weights.png"), dpi=500)
    # plt.show()
    rmsd_values = cv1
    rg_values = cv2
    weights = rescaled_weights
    assert len(rmsd_values) == len(rg_values) == len(weights), "Arrays must have the same length"
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
    constant_size = 1000
    sc = ax.scatter(
        rmsd_values,
        rg_values,
        s=constant_size,
        c=weights,
        cmap="plasma",
        alpha=1,
        edgecolors="black",
        vmin=0,
        vmax=1,
    )
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Weights", fontsize=16)
    cbar.mappable.set_clim(0, 1)
    ax.set_xlabel(CFG["cv_xlabel"], fontsize=16)
    ax.set_ylabel(CFG["cv_ylabel"], fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "rescaled_weights.png"), dpi=500)
    # plt.show()


# Functions to create starting directories for simulations


def extract_selected_frames(
    data_dir, output_directory, input_dcd, input_pdb, index_file, output_dcd
):
    """Write the frames named in an index file to a new DCD file."""
    # Construct full file paths
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    traj_path = os.path.join(data_dir, input_dcd)
    top_path = os.path.join(data_dir, input_pdb)
    index_path = os.path.join(output_directory, index_file)
    output_dcd_path = os.path.join(output_directory, output_dcd)
    traj = md.load(traj_path, top=top_path)
    print(f"Total number of frames in the original trajectory: {traj.n_frames}")
    with open(index_path, "r") as f:
        selected_indices = [int(line.strip()) for line in f.readlines()]
    # Ensure indices are within valid range
    selected_indices = [i for i in selected_indices if i < traj.n_frames]
    extracted_traj = traj[selected_indices]
    extracted_traj.save_dcd(output_dcd_path)
    print(f"Total number of selected frames saved in {output_dcd}: {len(selected_indices)}")


def extract_frames(data_dir, output_directory, bstates_dir, input_dcd, input_pdb):
    """Write every frame of a trajectory as a separate DCD file in the basis state directory."""
    # Construct full file paths
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    traj_path = os.path.join(output_directory, input_dcd)
    top_path = os.path.join(data_dir, input_pdb)
    traj = md.load(traj_path, top=top_path)
    num_frames = traj.n_frames
    print(f"Total number of frames: {num_frames}")
    # Ensure the bstates directory exists (remove and recreate if it exists)
    if os.path.exists(bstates_dir):
        shutil.rmtree(bstates_dir)
    os.makedirs(bstates_dir)
    for i in range(num_frames):
        frame = traj[i : i + 1]  # Extract single frame
        output_file = os.path.join(bstates_dir, f"{i}.dcd")
        frame.save_dcd(output_file)
    print(f"All {num_frames} frames have been saved to {bstates_dir}.")


def save_xml(frame_index, topology_file):
    """Minimize one basis state frame, run a short simulation, and save the OpenMM state."""
    dcd_file = f"{frame_index}.dcd"
    traj = md.load(dcd_file, top=topology_file)
    initial_positions = traj.openmm_positions(0)
    pdb = PDBFile(topology_file)
    forcefield = build_system.build_forcefield(CFG)
    system = build_system.build_system(forcefield, pdb.topology, CFG)
    integrator = LangevinIntegrator(
        CFG["T"] * kelvin,
        CFG["integrator_friction_per_ps"] / picoseconds,
        CFG["timestep_ps"] * picoseconds,
    )
    try:
        platform = Platform.getPlatformByName(CFG.get("platform", "CUDA"))
    except Exception:
        # A CPU only OpenMM build does not register the CUDA platform at all.
        platform = Platform.getPlatformByName("CPU")
    try:
        simulation = Simulation(pdb.topology, system, integrator, platform)
    except Exception:
        # The CUDA plugin registers even where no device is visible, so a missing GPU
        # surfaces only when the context is built. The integrator is bound to the failed
        # context and has to be rebuilt for the fallback.
        print(f"{platform.getName()} platform unavailable. Using CPU platform.")
        integrator = LangevinIntegrator(
            CFG["T"] * kelvin,
            CFG["integrator_friction_per_ps"] / picoseconds,
            CFG["timestep_ps"] * picoseconds,
        )
        simulation = Simulation(pdb.topology, system, integrator, Platform.getPlatformByName("CPU"))
    simulation.context.setPositions(initial_positions)
    LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=CFG.get("minimize_tolerance_kj_nm", 10.0) * kilojoule_per_mole / nanometer,
        maxIterations=CFG.get("bstate_minimize_max_iterations", 100),
    )
    minimized_positions = simulation.context.getState(getPositions=True).getPositions()
    with open(f"{frame_index}_minimized.pdb", "w") as output:
        PDBFile.writeFile(pdb.topology, minimized_positions, output)
    simulation.reporters.append(
        StateDataReporter(
            f"{frame_index}_simulation.log",
            CFG.get("bstate_report_interval", 100),
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True,
        )
    )
    simulation.reporters.append(
        DCDReporter(f"{frame_index}_traj.dcd", CFG.get("bstate_report_interval", 100))
    )
    simulation.step(CFG.get("bstate_md_steps", 100))
    state = simulation.context.getState(getPositions=True, getVelocities=True, getParameters=True)
    with open(f"{frame_index}.xml", "w") as xml_file:
        xml_file.write(XmlSerializer.serialize(state))
    print(f"Frame {frame_index}.xml saved.")


def process_all_frames(data_dir, output_directory, bstates_dir, framecount_file, topology_file):
    """Minimize every basis state frame and write its OpenMM state file."""
    data_dir = os.path.abspath(data_dir)
    bstates_dir = os.path.abspath(bstates_dir)
    output_directory = os.path.abspath(output_directory)
    # Construct absolute paths for framecount_file and topology_file
    framecount_file_path = os.path.join(output_directory, framecount_file)
    topology_file_path = os.path.join(data_dir, topology_file)
    # Ensure frame count file and topology file exist
    if not os.path.exists(framecount_file_path):
        print(f"ERROR: Frame count file not found: {framecount_file_path}")
        return
    if not os.path.exists(topology_file_path):
        print(f"ERROR: Topology file not found: {topology_file_path}")
        return
    # Change directory to bstates_dir
    current_dir = os.getcwd()
    os.chdir(bstates_dir)
    with open(framecount_file_path, "r") as file:
        num_frames = int(file.read().strip())
    print(f"Processing {num_frames} frames in {bstates_dir} using topology {topology_file_path}...")
    # Process all frames
    for i in range(num_frames):
        save_xml(frame_index=i, topology_file=topology_file_path)
    os.chdir(current_dir)
    print("Processing complete. Returned to the original directory.")


def save_cv(frame_index, topology_file):
    """Compute the collective variables of one frame and write them to its init file."""
    reference = md.load(topology_file)
    traj = md.load(f"{frame_index}.xml", top=topology_file)
    cv = cv_families.cv_of(traj, reference, CFG)
    np.savetxt(f"{frame_index}.init", cv)
    print(f"Frame {frame_index} CV saved as {frame_index}.init")


def process_cv_files(data_dir, output_directory, bstates_dir, framecount_file, topology_file):
    """Compute the collective variables of every basis state frame."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    bstates_dir = os.path.abspath(bstates_dir)
    # Construct absolute paths for framecount_file and topology_file
    framecount_file_path = os.path.join(output_directory, framecount_file)
    topology_file_path = os.path.join(data_dir, topology_file)
    # Ensure frame count file and topology file exist
    if not os.path.exists(framecount_file_path):
        print(f"ERROR: Frame count file not found: {framecount_file_path}")
        return
    if not os.path.exists(topology_file_path):
        print(f"ERROR: Topology file not found: {topology_file_path}")
        return
    # Change to the output directory (bstates)
    current_dir = os.getcwd()
    os.chdir(bstates_dir)
    with open(framecount_file_path, "r") as file:
        num_frames = int(file.read().strip())
    print(f"Processing {num_frames} frames in {bstates_dir} using topology {topology_file_path}...")
    # Process all frames
    for i in range(num_frames):
        save_cv(frame_index=i, topology_file=topology_file_path)
    os.chdir(current_dir)
    print("All .init files have been saved.")


def combine_init_files(bstates_dir, combined_file):
    """Concatenate the numbered init files into a single progress coordinate file."""
    print(f"Looking for numbered .init files in: {bstates_dir}")
    # List all files ending with .init and ensure they have a numeric prefix
    init_files = [
        f for f in os.listdir(bstates_dir) if f.endswith(".init") and f.split(".")[0].isdigit()
    ]
    # Sort numerically to process in order
    init_files = sorted(init_files, key=lambda x: int(x.split(".")[0]))
    if not init_files:
        print("No numbered .init files found to combine.")
        return
    all_data = []  # Initialize an empty list to store all data
    for file_name in init_files:
        file_path = os.path.join(bstates_dir, file_name)
        try:
            data = np.loadtxt(file_path)  # Load data
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Skipping {file_name} due to error: {e}")
    if all_data:
        try:
            final_data = np.vstack(all_data)  # Combine all the data into a single array
            combined_file_path = os.path.join(bstates_dir, combined_file)

            # Delete the existing combined file if it exists
            if os.path.exists(combined_file_path):
                os.remove(combined_file_path)
                print(f"Deleted existing file: {combined_file_path}")
            np.savetxt(combined_file_path, final_data)
            print(f"Combined file created: {combined_file_path}")
        except ValueError:
            print("Error: .init files have inconsistent shapes, unable to combine.")
    else:
        print("No valid .init files to combine.")


def plot_pcoord_and_save(bstates_dir, output_directory, pcoord_file, output_file):
    """Plot the progress coordinates of the basis states."""
    # Construct absolute paths for pcoord file and output plot
    pcoord_path = os.path.join(bstates_dir, pcoord_file)
    output_directory = os.path.abspath(output_directory)
    output_path = os.path.join(output_directory, output_file)
    # Ensure pcoord file exists
    if not os.path.exists(pcoord_path):
        print(f"ERROR: pcoord file not found: {pcoord_path}")
        return
    pcoord_data = np.loadtxt(pcoord_path)
    plt.figure(figsize=(10, 6))
    plt.scatter(pcoord_data[:, 0], pcoord_data[:, 1], s=200, alpha=0.7)
    plt.xlabel(CFG["cv_xlabel"], fontsize=14)
    plt.ylabel(CFG["cv_ylabel"], fontsize=14)
    plt.xticks(
        np.arange(int(min(pcoord_data[:, 0])), int(max(pcoord_data[:, 0])) + 1, 1), fontsize=12
    )
    plt.yticks(
        np.arange(int(min(pcoord_data[:, 1])), int(max(pcoord_data[:, 1])) + 1, 1), fontsize=12
    )
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.savefig(output_path, bbox_inches="tight", dpi=500)
    plt.close()
    print(f"Plot saved in bstates_dir: {output_path}")


def organize_files_to_dir(data_dir, output_directory, bstates_dir, framecount_file):
    """Move each basis state into its own numbered directory."""
    # Construct absolute paths
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    bstates_dir = os.path.abspath(bstates_dir)
    framecount_file_path = os.path.join(output_directory, framecount_file)
    # Ensure framecount file exists
    if not os.path.exists(framecount_file_path):
        print(f"ERROR: Frame count file not found: {framecount_file_path}")
        return
    with open(framecount_file_path, "r") as file:
        num_frames = int(file.read().strip())
    # Force directory names to four digits, 0000 to 9999
    # num_digits = 4
    # Move files into numbered folders
    for i in range(num_frames):
        folder_name = os.path.join(bstates_dir, f"{i:04}")
        os.makedirs(folder_name, exist_ok=True)
        xml_file = os.path.join(bstates_dir, f"{i}.xml")
        init_file = os.path.join(bstates_dir, f"{i}.init")
        if os.path.exists(xml_file):
            shutil.move(xml_file, os.path.join(folder_name, "bstate.xml"))
        if os.path.exists(init_file):
            shutil.move(init_file, os.path.join(folder_name, "pcoord.init"))
    print("All files have been organized and renamed.")


def move_intermediate_files(bstates_dir, intermediate_dir):
    """Move the intermediate files into a separate directory."""
    bstates_dir = os.path.abspath(bstates_dir)
    intermediate_dir = os.path.join(bstates_dir, intermediate_dir)
    # Remove existing intermediate directory if present, then recreate it
    if os.path.exists(intermediate_dir):
        shutil.rmtree(intermediate_dir)
    os.makedirs(intermediate_dir)
    patterns = ["*_minimized.pdb", "*_simulation.log", "*.dcd", "*.png"]
    # Move files matching the patterns
    for pattern in patterns:
        files_to_move = glob.glob(os.path.join(bstates_dir, pattern))  # Use absolute path for glob
        for file_path in files_to_move:
            shutil.move(file_path, intermediate_dir)
    print("All specified files have been moved to the intermediate folder.")


def generate_bstates_file(
    data_dir, output_directory, bstates_dir, framecount_file, weights_file, output_file
):
    """Write the bstates.txt file mapping each basis state to its weight."""
    data_dir = os.path.abspath(data_dir)
    output_directory = os.path.abspath(output_directory)
    bstates_dir = os.path.abspath(bstates_dir)
    framecount_file_path = os.path.join(output_directory, framecount_file)
    weights_file_path = os.path.join(output_directory, weights_file)
    output_file_path = os.path.join(bstates_dir, output_file)
    # Ensure framecount and weights files exist
    if not os.path.exists(framecount_file_path):
        print(f"ERROR: Frame count file not found: {framecount_file_path}")
        return
    if not os.path.exists(weights_file_path):
        print(f"ERROR: Weights file not found: {weights_file_path}")
        return
    with open(framecount_file_path, "r") as file:
        num_frames = int(file.read().strip())
    with open(weights_file_path, "r") as f:
        weights = [line.strip() for line in f]  # Read as strings to retain formatting
    if len(weights) != num_frames:
        print(
            f"ERROR: Number of frames ({num_frames}) does not match the number of weights ({len(weights)})."
        )
        return
    with open(output_file_path, "w") as f:
        for i in range(num_frames):
            index = f"{i:04}"  # Four digit format, 0000 to 9999
            weight = weights[i]
            f.write(f"{index} {weight} {index}\n")
    print(f"{output_file} has been created successfully in {bstates_dir}.")


def get_bottleneck(
    data_dir, dcd_file, topo_file, output_dir, fig_bottleneck, nbins, sigma_mults, bottleneck_file
):
    """Locate the bottleneck coordinates on the free energy surface and write them out."""
    # Paths
    traj_path = os.path.join(data_dir, dcd_file)
    top_path = os.path.join(data_dir, topo_file)
    fig_save_path = os.path.join(output_dir, fig_bottleneck)
    file_save_path = os.path.join(output_dir, bottleneck_file)
    traj = md.load(traj_path, top=top_path)
    ref = md.load(top_path)
    cv = cv_families.cv_of(traj, ref, CFG)
    rmsd = cv[:, 0]
    rg = cv[:, 1]
    H, xedges, yedges = np.histogram2d(rmsd, rg, bins=nbins, density=True)
    F = -np.log(H + 1e-12)
    F -= F.min()
    xg = 0.5 * (xedges[:-1] + xedges[1:])
    yg = 0.5 * (yedges[:-1] + yedges[1:])
    Fg = F.T
    mask = H.T == 0
    Fg_masked = np.ma.array(Fg, mask=mask)
    # Global minimum
    i0, j0 = np.unravel_index(np.argmin(Fg_masked), Fg_masked.shape)
    folded_xy = (xg[j0], yg[i0])
    # ±σ minima. cfg.bottleneck_neg_fallback selects the ntl9 behavior (always emit a
    # single −1σ bottleneck when no requested negative σ level has any frames).
    mu, sigma = rmsd.mean(), rmsd.std()
    points = [("Maximum Sampling", folded_xy)]
    neg_fallback = CFG.get("bottleneck_neg_fallback", False)
    neg_done = False
    for m in sigma_mults:
        tau = mu + m * sigma
        cols = np.where(xg >= tau if m >= 0 else xg <= tau)[0]
        if neg_fallback:
            if cols.size == 0 and m < 0:
                if not neg_done:
                    tau_neg = mu - sigma
                    idx = int(np.argmin(np.abs(xg - tau_neg)))
                    cols = np.array([idx])
                    label = f"-1σ (τ={tau_neg:.2f})"
                    neg_done = True
                else:
                    continue
            elif cols.size == 0:
                continue
            else:
                label = f"{m:+.0f}σ (τ={tau:.2f})"
            region = np.zeros_like(Fg_masked.mask, bool)
            region[:, cols] = True
            Freg = np.ma.masked_where(~region, Fg_masked)
            i1, j1 = np.unravel_index(np.argmin(Freg), Freg.shape)
            points.append((label, (xg[j1], yg[i1])))
        else:
            if cols.size:
                region = np.zeros_like(Fg_masked.mask, bool)
                region[:, cols] = True
                Freg = np.ma.masked_where(~region, Fg_masked)
                i1, j1 = np.unravel_index(np.argmin(Freg), Freg.shape)
                points.append((f"{'+' if m>0 else ''}{m}σ (τ={tau:.2f})", (xg[j1], yg[i1])))
    os.makedirs(output_dir, exist_ok=True)
    with open(file_save_path, "w", encoding="utf-8") as f:
        for idx, (label, (xr, yr)) in enumerate(points, start=1):
            f.write(f"{idx}\t{label}\t{xr:.4f}\t{yr:.4f}\n")
    print(f"Wrote bottleneck data → {file_save_path}")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    X, Y = np.meshgrid(xg, yg)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    levels = np.linspace(Fg_masked.min(), Fg_masked.max(), 60)
    cs = ax.contourf(
        X,
        Y,
        Fg_masked,
        levels=levels,
        cmap=cmap,
        **(
            {"extend": CFG["bottleneck_contourf_extend"]}
            if CFG.get("bottleneck_contourf_extend")
            else {}
        ),
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for idx, (label, (xr, yr)) in enumerate(points, start=1):
        ax.scatter(xr, yr, c="k", s=10)
        ax.text(xr + 0.05, yr + 0.05, str(idx), fontsize=8, color="black")
    legend_labels = [f"{i}: {lbl}" for i, (lbl, _) in enumerate(points, start=1)]
    ax.legend(legend_labels, frameon=False, fontsize=8, loc="upper left")
    if CFG.get("bottleneck_xlabel", ""):
        ax.set_xlabel(CFG["bottleneck_xlabel"], fontsize=12)
    if CFG.get("bottleneck_ylabel", ""):
        ax.set_ylabel(CFG["bottleneck_ylabel"], fontsize=12)
    ax.set_xlim(CFG["xmin"], CFG["xmax"])
    ax.set_ylim(CFG["ymin"], CFG["ymax"])
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(cs, cax=cax, extend="neither")
    cbar.set_label("Free Energy", fontsize=12)
    for spine in ("top", "right", "left"):
        cax.spines[spine].set_visible(False)
    cbar.ax.yaxis.set_ticks_position("right")
    fig.savefig(fig_save_path, dpi=300)
    plt.close(fig)


def copy_file(src, dst):
    """Copy one file, raising when the source is absent or unreadable."""
    # Ensure source exists and is a file
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file not found: {src}")
    if not os.path.isfile(src):
        raise IsADirectoryError(f"Source is not a file: {src}")
    dest_dir = os.path.dirname(dst)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    # Copy the file, preserving metadata
    shutil.copy2(src, dst)


def main(run0=False):
    """Run every stage of the reweighting in order, in the current directory."""

    # Iteration zero has no preceding WE run, so its structures come from the seeding MD.
    source_dir = CFG["init_md_dir"] if run0 else CFG["merged_we_dir"]
    source_dcd = CFG["init_md_dcd"] if run0 else CFG["merged_we_dcd"]
    source_max = CFG["init_md_max_frames"] if run0 else CFG["merged_max_frames"]

    setup_directories(data_dir="data", output_directory="output")

    # Copy necessary files
    if not run0:
        copy_file(
            src="data/" + CFG["topology_we"], dst=CFG["merged_we_dir"] + "/" + CFG["topology_we"]
        )

    get_init_traj(
        data_dir=source_dir,
        output_directory="output",
        dcd_file=source_dcd,
        topology_file=CFG["topology_we"],
        dcd_stripped_file="init_md.dcd",
        max_frames=source_max,
    )

    # (cfg.run_plot_free_energy gates the call, ntl9 defines the fn but skips it)
    if not run0 and CFG.get("run_plot_free_energy", True):
        plot_free_energy(
            data_dir="data",
            output_directory="output",
            trajectory_file="image.dcd",
            topology_file=CFG["topology_analysis"],
            kB=CFG["kB"],
            T=CFG["T"],
            output_file="free_energy_image.png",
            nbins=tuple(CFG["fe_nbins"]),
        )

    # Overlay the dashed KDE contours of the seeding and selected distributions
    selected = CFG.get("out_boltz_dcd", "image_sel.dcd")
    if not os.path.exists(os.path.join("data", selected)):
        # When run_get_distribution is false no Boltzmann selection was made, so the raw
        # target pool stands in for it.
        print(f"{selected} not present, using {CFG['traj_file']} as the selected target")
        selected = CFG["traj_file"]
    plot_overlap(
        data_dir="data",
        output_directory="output",
        topology=CFG["topology_analysis"],
        initial_md="init_md.dcd",
        selected_traj=selected,
        output_file="overlap_sim_image.png",
        bw=0.75,
        n_levels=25,
        nx=400,
        ny=400,
        xmin=CFG["xmin"],
        xmax=CFG["xmax"],
        ymin=CFG["ymin"],
        ymax=CFG["ymax"],
    )

    initialize(
        data_dir="data",
        output_directory="output",
        traj_file="init_md.dcd",
        top_file=CFG["topology_analysis"],
        ref_file=CFG["topology_analysis"],
        bin_width=CFG["bin_width"],
        output_dcd="simulation.dcd",
        index_file="simulation_indices.txt",
        frame_count_file="frame_count.txt",
    )

    # reference.dcd is the structure set the synthetic particles are generated from, so
    # it is thinned to keep the image count and the distance matrix tractable.
    save_every_nth_frame(
        data_dir="data",
        output_directory="output",
        traj_file=selected,
        top_file=CFG["topology_analysis"],
        output_dcd="reference.dcd",
        step=CFG["step"],
    )

    generate_weights_from_file(
        data_dir="data",
        output_directory="output",
        frame_count_file="frame_count.txt",
        output_file="simulation_weights.txt",
    )

    # Process weights by binning CVs, merges frames within bins, and extracts representative structures
    process_weights_from_sim(
        data_dir="data",
        output_directory="output",
        traj_initial_weights="simulation_weights.txt",
        simulation_top=CFG["topology_analysis"],
        md_simulation_traj="simulation.dcd",
        bin_width=CFG["bin_width"],
        bin_coordinates_weights="bin_crds_wght.txt",
        occupied_bins_summary="occ_bins_summ.txt",
        final_bins_file="bin_wght.txt",
        merged_details_file="mergd_bins_wght.txt",
        merged_details_pickle="merged_details.pkl",
        simulation_traj="selected_frames.dcd",
        initial_weights="selected_weights.txt",
        plot_filename="bin_dist.png",
    )

    check_cuda()

    # Count and print the number of conformations in `reference.dcd` and `selected_frames.dcd`
    count_conformations(
        data_dir="data",
        output_directory="output",
        reference_traj="reference.dcd",
        reference_top=CFG["topology_analysis"],
        simulation_traj="selected_frames.dcd",
        simulation_top=CFG["topology_analysis"],
    )

    # Identify and visualize structural clusters from `selected_frames.dcd` and `reference.dcd`
    get_clusters(
        data_dir="data",
        output_directory="output",
        reference_top=CFG["topology_analysis"],
        reference_traj="reference.dcd",
        simulation_top=CFG["topology_analysis"],
        simulation_traj="selected_frames.dcd",
    )

    # Reweight the ensemble against the likelihood configured for this system
    DATA_ABS = os.path.abspath("data")
    OUT_ABS = os.path.abspath("output")
    CTX = {
        "cfg": CFG,
        "data_dir": DATA_ABS,
        "output_directory": OUT_ABS,
        "reference_top": os.path.join(DATA_ABS, CFG["topology_analysis"]),
        "reference_traj": os.path.join(OUT_ABS, "reference.dcd"),
        "simulation_top": os.path.join(DATA_ABS, CFG["topology_analysis"]),
        "simulation_traj": os.path.join(OUT_ABS, "selected_frames.dcd"),
        "device": check_device(),
    }
    reweight(
        CTX,
        img_struc_dist_file="img_struct_dist_mtx.png",
        hist_img_struct_dist_file="hist_img_struct_dist.png",
        hist_neg_lkhood_file="hist_neg_lkhood.png",
        initial_weights="selected_weights.txt",
        rescaled_weights_file="rescaled_weights.txt",
    )

    # Rescale the computed weights and update the weight distribution
    rescale_weights(
        data_dir="data",
        output_directory="output",
        rescaled_weights_file="rescaled_weights.txt",
        merged_details_pickle="merged_details.pkl",
        rescaled_merged_details_file="mergd_bins_wght_rescld.txt",
        sorted_indices_weights_file="srtd_ind_wghts.txt",
        rescaled_weights_all="rescaled_weights_all.txt",
    )

    get_rescaled_plots(
        data_dir="data",
        output_directory="output",
        init_weights="selected_weights.txt",
        rescaled_weights="rescaled_weights.txt",
        simulation_top=CFG["topology_analysis"],
        simulation_traj="selected_frames.dcd",
        n_bins=20,
    )

    extract_selected_frames(
        data_dir=source_dir,
        output_directory="output",
        input_dcd=source_dcd,
        input_pdb=CFG["topology_we"],
        index_file="simulation_indices.txt",
        output_dcd="selected_trajectory.dcd",
    )

    # Split `selected_trajectory.dcd` into individual frames and stores them in `bstates` directory
    extract_frames(
        data_dir="data",
        output_directory="output",
        bstates_dir="bstates",
        input_dcd="selected_trajectory.dcd",
        input_pdb=CFG["topology_we"],
    )

    # Process all extracted frames by performing energy minimization and saving simulation states
    process_all_frames(
        data_dir="data",
        output_directory="output",
        bstates_dir="bstates",
        framecount_file="frame_count.txt",
        topology_file=CFG["topology_we"],
    )

    process_cv_files(
        data_dir="data",
        output_directory="output",
        bstates_dir="bstates",
        framecount_file="frame_count.txt",
        topology_file=CFG["topology_we"],
    )

    # Combine all `.init` files into a single `pcoord.init` file
    combine_init_files(bstates_dir="bstates", combined_file="pcoord.init")

    plot_pcoord_and_save(
        bstates_dir="bstates",
        output_directory="output",
        pcoord_file="pcoord.init",
        output_file="pcoord_plot.png",
    )

    # Organize basis state files into numbered directories for weighted ensemble simulations
    organize_files_to_dir(
        data_dir="data",
        output_directory="output",
        bstates_dir="bstates",
        framecount_file="frame_count.txt",
    )

    # Move intermediate files (trajectories, logs, and minimization files) to `intermediates` directory
    move_intermediate_files(bstates_dir="bstates", intermediate_dir="intermediates")

    # Generate `bstates.txt`, mapping basis states to their rescaled weights for weighted ensemble initialization
    generate_bstates_file(
        data_dir="data",
        output_directory="output",
        bstates_dir="bstates",
        framecount_file="frame_count.txt",
        weights_file="rescaled_weights_all.txt",
        output_file="bstates.txt",
    )

    # Locate the bottleneck
    get_bottleneck(
        data_dir=source_dir,
        dcd_file=source_dcd,
        topo_file=CFG["topology_we"],
        output_dir="output",
        fig_bottleneck="bottleneck.png",
        nbins=tuple(CFG["bottleneck_nbins"]),
        sigma_mults=CFG["sigma_mults"],
        bottleneck_file="bottleneck_coordinates.txt",
    )


if __name__ == "__main__":
    load_config()
    main(run0="--run0" in sys.argv)
