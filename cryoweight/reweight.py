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
from collections import defaultdict
import MDAnalysis as mda
import mdtraj as md
import numpy as np
import shutil
import pickle
import torch
import glob
import sys
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
    """Report the structure and image counts and check both position sets are centred."""
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


def reweight(
    ctx,
    initial_weights,
    rescaled_weights_file,
):
    """Reweight the ensemble by expectation maximization against the configured likelihood."""
    out = ctx["output_directory"]
    diff, scale = likelihoods.distance_of(ctx)
    # A Gaussian likelihood of width scale turns a squared distance into a log probability,
    # so the negative log likelihood is the distance over twice the squared scale.
    nll = diff / (2 * scale**2)
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
    data_dir, dcd_file, topo_file, output_dir, nbins, sigma_mults, bottleneck_file
):
    """Locate the bottleneck coordinates on the free energy surface and write them out."""
    # Paths
    traj_path = os.path.join(data_dir, dcd_file)
    top_path = os.path.join(data_dir, topo_file)
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

    # The Boltzmann selected target set, or the raw target pool when no selection was made
    selected = CFG.get("out_boltz_dcd", "image_sel.dcd")
    if not os.path.exists(os.path.join("data", selected)):
        print(f"{selected} not present, using {CFG['traj_file']} as the selected target")
        selected = CFG["traj_file"]

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

    # Check the structure and image sets from `selected_frames.dcd` and `reference.dcd`
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
        nbins=tuple(CFG["bottleneck_nbins"]),
        sigma_mults=CFG["sigma_mults"],
        bottleneck_file="bottleneck_coordinates.txt",
    )


if __name__ == "__main__":
    load_config()
    main(run0="--run0" in sys.argv)
