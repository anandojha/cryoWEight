"""Image forward model and weight optimization, adapted from the cryoER package.

The forward model, the structure to image distance, and the noise scale estimate follow the
original implementation. The Markov chain Monte Carlo sampling of the weight posterior is
replaced here by expectation maximization of the image likelihood.

Tang, W. S.; Silva-Sanchez, D.; Giraldo-Barreto, J.; Carpenter, B.; Hanson, S. M.;
Barnett, A. H.; Thiede, E. H.; Cossio, P. Ensemble reweighting using cryo-EM particle
images. J. Phys. Chem. B 2023, 127, 5410-5421.
"""

from typing import Callable, Optional
import torch.optim as optim
import MDAnalysis as mda
from tqdm import tqdm
import numpy as np
import torch
import math
import os

##########################################################################################


def normalize_weights(
    log_weights: torch.Tensor,
) -> torch.Tensor:
    """ """
    weighted_alphas = torch.exp(log_weights)
    weighted_alphas = weighted_alphas / torch.sum(weighted_alphas)
    return weighted_alphas


def evaluate_nll(
    log_weights: torch.Tensor,
    log_Pij: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the negative log likelihood of the data given the weights."""
    # Normalize the log weights
    weighted_alphas = normalize_weights(log_weights)
    log_weighted_alphas = torch.log(weighted_alphas)

    # Evaluate the log likelihood
    likelihood_per_image = torch.logsumexp(log_Pij + log_weighted_alphas, axis=1)
    neg_total_ll = -1 * torch.mean(likelihood_per_image)
    return neg_total_ll


def eval_naive_log_Pij(sq_dist_matrix: torch.Tensor, noise_stdev: float) -> torch.Tensor:
    """Evaluate the likelihood of generating image i from cluster j."""
    return sq_dist_matrix / (-2 * noise_stdev**2)


# From Robert Gower, slightly modified
def fw_gap(weights, grad):
    """The Frank-Wolfe gap, an upper bound on the optimality gap."""
    # For a convex f, would be
    # -(np.min(grad) - np.inner(grad, param))
    # In our case, grad is the negative of the gradient, so
    # -(np.min(-grad) - np.inner(-grad, param))
    # simplifies to
    # np.max(grad) - np.inner(grad, param)
    # BUT, for our problem, np.inner(grad, param) is always 1
    return torch.max(grad) - 1


def grad_log_prob(
    weights: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the gradient of the log likelihood of the data given the weights."""
    num_images, num_structures = log_likelihood.shape

    log_weights = torch.log(weights)
    log_density_at_weights = torch.logsumexp(log_likelihood + log_weights, axis=1)

    aux = log_likelihood - log_density_at_weights.reshape(num_images, 1)
    grad = (1 / num_images) * (torch.exp(torch.logsumexp(aux, axis=0)))
    return grad


def multiplicative_gradient(
    log_likelihood,
    tol: Optional[float] = 10**-4,
    max_iterations: Optional[int] = 20000,
    stats_frequency: Optional[int] = 100,
) -> float:
    """Update the weights by expectation maximization, iterating to a Frank-Wolfe gap tolerance."""
    num_images, num_structures = log_likelihood.shape
    weights = (1 / num_structures) * torch.ones(num_structures)
    stats_tracking = {}
    stats_tracking["losses"] = []
    stats_tracking["entropies"] = []
    stats_tracking["idx"] = []
    for k in range(max_iterations):
        grad = grad_log_prob(weights, log_likelihood)
        weights = weights * grad
        # Check stopping criterion
        gap = fw_gap(weights, grad)
        if k % stats_frequency == 0:
            log_weights = torch.log(weights)
            loss = -torch.mean(torch.logsumexp(log_likelihood + log_weights, axis=1))
            entropy = -torch.sum(weights * log_weights)
            stats_tracking["losses"].append(loss)
            stats_tracking["entropies"].append(entropy)
            stats_tracking["idx"].append(k)
            print(f"#iterations: {k}")
            print(f"loss: {loss}")
            print(f"frank-wolfe gap: {gap}")
            print(f"entropy: {entropy}")
            print("\n")
        if gap < tol:
            print("exiting!")
            print(f"#iterations at exit: {k}")
            break

    log_weights = torch.log(weights)
    log_weights = torch.log(normalize_weights(log_weights))
    return log_weights, stats_tracking


# This computes the same quantity as multiplicative_gradient, written differently and with different stats tracked
def expectation_maximization_weights(
    log_Pij: torch.Tensor,
    log_weights_init: Optional[torch.Tensor] = None,
    num_iterations: Optional[int] = 1000,
):
    """Update the weights by expectation maximization for a fixed number of iterations."""
    num_images, num_structures = log_Pij.shape
    if log_weights_init is None:
        log_weights = (1 / num_structures) * torch.ones(num_structures)
    else:
        log_weights = torch.clone(log_weights_init)
    losses = []
    for k in range(num_iterations):
        log_weighted_likelihoods = log_Pij + log_weights
        log_likelihood_per_image = torch.logsumexp(log_weighted_likelihoods, axis=1)
        log_posteriors = log_weighted_likelihoods - log_likelihood_per_image.reshape(
            log_likelihood_per_image.shape[0], 1
        )
        log_weights = torch.logsumexp(log_posteriors - np.log(num_images), axis=0)
        loss = -torch.mean(torch.logsumexp(log_Pij + log_weights, axis=1))
        losses.append(loss.item())
    log_weights = torch.log(normalize_weights(log_weights))
    return log_weights, torch.tensor(losses)


def expectation_maximization_weights_from_clustering(
    log_Pij: torch.Tensor,
    log_weights_init: Optional[torch.Tensor] = None,
    cluster_sizes: Optional[torch.Tensor] = None,
    num_iterations: Optional[int] = 1000,
):
    """Update the weights by expectation maximization, with each structure scaled by its cluster size."""
    num_images, num_structures = log_Pij.shape
    if log_weights_init is None:
        log_weights = (1 / num_structures) * torch.ones(log_Pij.shape[1])
    else:
        log_weights = torch.clone(log_weights_init)
    if cluster_sizes is None:
        cluster_sizes = torch.ones_like(log_weights)
    log_cluster_sizes = torch.log(cluster_sizes)
    log_scaled_weights = torch.log(normalize_weights(log_weights + log_cluster_sizes))
    losses = []
    for k in range(num_iterations):
        log_weighted_likelihoods = log_Pij + log_scaled_weights
        log_likelihood_per_image = torch.logsumexp(log_weighted_likelihoods, axis=1)
        log_posteriors = log_weighted_likelihoods - log_likelihood_per_image.reshape(
            log_likelihood_per_image.shape[0], 1
        )
        log_scaled_weights = torch.logsumexp(log_posteriors - np.log(num_images), axis=0)

        loss = -torch.mean(torch.logsumexp(log_Pij + log_scaled_weights, axis=1))
        losses.append(loss.item())
    # Get back weights without cluster sizes
    log_weights = torch.log(normalize_weights(log_scaled_weights - log_cluster_sizes))
    return log_weights, torch.tensor(losses)


def gradient_descent_weights(
    log_Pij: torch.Tensor,
    loss_fxn: Optional[Callable] = None,
    log_weights_init: Optional[torch.Tensor] = None,
    cluster_sizes: Optional[torch.Tensor] = None,
    regularization_fxn: Optional[Callable] = None,
    num_iterations: Optional[int] = 1000,
):
    if log_weights_init is None:
        log_weights = torch.randn_like(log_Pij[0]) * 0.01
    else:
        log_weights = torch.clone(log_weights_init)
    log_weights.requires_grad = True
    if loss_fxn is None:
        loss_fxn = lambda x: evaluate_nll(x, log_Pij)
    if regularization_fxn is None:
        regularization_fxn = lambda x: 0
    if cluster_sizes is None:
        cluster_sizes = torch.ones_like(log_weights)
    log_cluster_sizes = torch.log(cluster_sizes)
    # Optimization Loop
    optimizer = optim.Adam([log_weights], lr=0.1)
    losses = []
    for i in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        log_scaled_weights = log_weights + log_cluster_sizes
        loss = loss_fxn(log_scaled_weights)
        total_loss = loss + regularization_fxn(log_weights)
        total_loss.backward()
        losses.append(total_loss.item())
        optimizer.step()
    return log_weights, torch.tensor(losses)


##########################################################################################


def gen_grid(n_pixel, pixel_size):
    """Generate a 1D grid of pixel center positions for a square image."""
    grid_min = -pixel_size * (n_pixel - 1) * 0.5
    grid_max = -grid_min  # pixel_size*(n_pixel-1)*0.5
    grid = torch.linspace(grid_min, grid_max, n_pixel)
    return grid


def gen_quat_torch(num_quaternions, device="cuda"):
    """Sample quaternions from a spherically uniform random distribution of directions."""
    # Rejection sampling. Draw from the cube and keep the shell 0.2 < |q| < 1, which is
    # about 31% of it. Drawing a fixed multiple and slicing can therefore return fewer
    # than asked for, by chance, and the caller then fails to broadcast rotations against
    # structures. Draw until enough have been accepted instead. This does not change the
    # distribution, only how many candidates it takes to fill the request.
    over_produce = 5
    accepted = []
    have = 0
    while have < num_quaternions:
        quat = (
            torch.rand((num_quaternions * over_produce, 4), dtype=torch.float64, device=device)
            * 2.0
            - 1.0
        )
        norm = torch.linalg.vector_norm(quat, ord=2, dim=1)
        quat /= norm.unsqueeze(1)
        good_ones = torch.bitwise_and(torch.gt(norm, 0.2), torch.lt(norm, 1.0))
        accepted.append(quat[good_ones])
        have += int(good_ones.sum())
    return torch.cat(accepted)[:num_quaternions]


def quaternion_to_matrix(quaternions):
    """Convert rotations given as quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    rot_mat = o.reshape(quaternions.shape[:-1] + (3, 3))
    return rot_mat


def calc_ctf_torch_batch(freq2_2d, amp, gamma, b_factor):
    """Generate a batch of random Contrast Transfer Functions (CTFs)."""
    # env = torch.exp(- b_factor.view(-1,1,1) * freq2_2d.unsqueeze(0) * 0.5)
    # ctf = amp.view(-1,1,1) * torch.cos(gamma.view(-1,1,1) * freq2_2d * 0.5) - torch.sqrt(1 - amp.view(-1,1,1) **2) * torch.sin(gamma.view(-1,1,1)  * freq2_2d * 0.5) + torch.zeros_like(freq2_2d) * 1j
    env = torch.exp(-b_factor * freq2_2d.unsqueeze(0) * 0.5)
    ctf = (
        amp * torch.cos(gamma.view(-1, 1, 1) * freq2_2d * 0.5)
        - np.sqrt(1 - amp**2) * torch.sin(gamma.view(-1, 1, 1) * freq2_2d * 0.5)
        + torch.zeros_like(freq2_2d) * 1j
    )
    ctf *= env
    return ctf


def gen_img_torch_batch(coord, grid, sigma, norm, ctfs=None):
    """Generate synthetic cryo-EM images from atomic coordinates using a Gaussian imaging model."""
    gauss_x = -0.5 * ((grid[:, :, None] - coord[:, :, 0]) / sigma) ** 2
    gauss_y = (
        -0.5 * ((grid[:, :, None] - coord[:, :, 1]) / sigma) ** 2
    )  # Pixels are square, grid is same for x and y directions
    gauss = torch.exp(gauss_x.unsqueeze(1) + gauss_y)
    image = gauss.sum(3) * norm
    image = image.permute(2, 0, 1)
    if ctfs is not None:
        ft_image = torch.fft.fft2(image, dim=(1, 2), norm="ortho")
        image_ctf = torch.real(torch.fft.ifft2(ctfs * ft_image, dim=(1, 2), norm="ortho"))
        return image_ctf
    else:
        return image


def circular_mask(n_pixel, radius=0.4):
    """Create a circular mask centered in the image for SNR calculation."""
    grid = torch.linspace(-0.5 * (n_pixel - 1), 0.5 * (n_pixel - 1), n_pixel)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing="ij")
    r_2d = grid_x**2 + grid_y**2
    mask = r_2d < radius**2
    return mask


def add_noise_torch_batch(img, snr, device="cuda"):
    """Add colorless Gaussian pixel noise to a batch of images."""
    n_pixel = img.shape[1]
    radius = n_pixel * 0.4
    mask = circular_mask(n_pixel, radius)
    image_noise = torch.empty_like(img, device=device)
    for i, image in enumerate(img):
        image_masked = image[mask]
        signal_std = image_masked.pow(2).mean().sqrt()
        noise_std = signal_std / np.sqrt(snr)
        noise = torch.distributions.normal.Normal(0, noise_std).sample(image.shape)
        image_noise[i] = image + noise
    return image_noise


def generate_images(
    coord,
    n_pixel=128,
    pixel_size=0.3,
    sigma=1.0,
    snr=1.0,
    rotation=True,
    add_ctf=False,
    defocus_min=0.027,
    defocus_max=0.090,
    batch_size=8,
    device="cuda",
):
    """Generate synthetic cryo-EM images from atomic coordinates, with optional random rotations and CTF application."""
    if type(coord) == np.ndarray:
        coord = torch.from_numpy(coord).type(torch.float64)
    coord = coord.to(device)
    n_struc = coord.shape[0]
    n_atoms = coord.shape[1]
    norm = 0.5 / (np.pi * sigma**2 * n_atoms)
    N_images = n_struc
    n_batch = int(N_images / batch_size)
    if n_batch * batch_size < N_images:
        n_batch += 1
    if rotation:
        quats = gen_quat_torch(N_images, device)
        rot_mats = quaternion_to_matrix(quats).type(torch.float64)
        rot_mats = rot_mats.to(device)
        coord_rot = coord.matmul(rot_mats)
    else:
        rot_mats = torch.eye(3).unsqueeze(0).repeat(N_images, 1, 1).type(torch.float64)
        coord_rot = coord
    grid = gen_grid(n_pixel, pixel_size).reshape(-1, 1)
    grid = grid.to(device)
    ctfs_cpu = torch.empty((N_images, n_pixel, n_pixel), dtype=torch.complex64, device="cpu")
    images_cpu = torch.empty((N_images, n_pixel, n_pixel), dtype=torch.float64, device="cpu")
    if add_ctf:
        amp = 0.1  # Amplitude constrast ratio
        b_factor = 1.0  # B factor
        defocus = (
            torch.rand(N_images, dtype=torch.float64, device=device) * (defocus_max - defocus_min)
            + defocus_min
        )  # Defocus
        elecwavel = 0.019866  # Electron wavelength in Angstrom
        gamma = defocus * (
            np.pi * 2.0 * 10000 * elecwavel
        )  # Gamma coefficient in SI equation 4 that include the defocus
        freq_pix_1d = torch.fft.fftfreq(n_pixel, d=pixel_size, dtype=torch.float64, device=device)
        freq_x, freq_y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing="ij")
        freq2_2d = freq_x**2 + freq_y**2  # Square of modulus of spatial frequency
    for i in tqdm(range(n_batch), desc="Generating images for batch"):
        start = i * batch_size
        end = (i + 1) * batch_size
        coords_batch = coord_rot[start:end]
        coords_batch = coords_batch.to(device)
        if add_ctf:
            ctf_batch = calc_ctf_torch_batch(freq2_2d, amp, gamma[start:end], b_factor)
            ctfs_cpu[start:end] = ctf_batch
            image_batch = gen_img_torch_batch(coords_batch, grid, sigma, norm, ctf_batch.to(device))
        else:
            image_batch = gen_img_torch_batch(coords_batch, grid, sigma, norm)
        if not np.isinf(snr):
            image_batch = add_noise_torch_batch(image_batch, snr, device)
        images_cpu[start:end] = image_batch.cpu()
    if device == "cuda":
        rot_mats = rot_mats.cpu()
    return rot_mats, ctfs_cpu, images_cpu


def calc_struc_image_diff(
    coord,
    n_pixel=128,
    pixel_size=0.3,
    sigma=1.0,
    images=None,
    ctfs=None,
    batch_size=8,
    device="cuda",
    return_template=False,
):
    """Calculate the difference between a structure and a set of synthetic cryo-EM images."""
    if type(coord) == np.ndarray:
        coord = torch.from_numpy(coord).type(torch.float64)
    coord = coord.to(device)
    n_atoms = coord.shape[1]
    norm = 0.5 / (np.pi * sigma**2 * n_atoms)
    N_images = coord.shape[0]
    n_batch = int(N_images / batch_size)
    if n_batch * batch_size < N_images:
        n_batch += 1
    grid = gen_grid(n_pixel, pixel_size).reshape(-1, 1)
    grid = grid.to(device)
    if return_template:
        image_template = torch.empty(N_images, n_pixel, n_pixel, dtype=torch.float64, device=device)
    diff = torch.empty(N_images, dtype=torch.float64, device="cpu")
    for i in range(n_batch):
        start = i * batch_size
        end = (i + 1) * batch_size
        if ctfs is not None:
            ctf_batch = ctfs[start:end]
            ctf_batch = ctf_batch.to(device)
            image_batch = gen_img_torch_batch(coord[start:end], grid, sigma, norm, ctf_batch)
        else:
            image_batch = gen_img_torch_batch(coord[start:end], grid, sigma, norm)
        image_batch = image_batch - image_batch.mean(dim=(1, 2)).view(-1, 1, 1)
        diff[start:end] = torch.sum((image_batch - images[start:end].to(device)) ** 2, dim=(1, 2))
        if return_template:
            image_template[start:end] = image_batch.cpu()
    if return_template:
        return diff, image_template
    else:
        return diff


def mdau_to_pos_arr(u, frame_cluster=None):
    protein_CA = u.select_atoms("protein and name CA")
    if frame_cluster is None:
        n_frame = len(u.trajectory)
    else:
        n_frame = len(frame_cluster)
    pos = torch.zeros((n_frame, len(protein_CA), 3), dtype=float)
    if frame_cluster is None:
        for i, ts in enumerate(u.trajectory):
            pos[i] = torch.from_numpy(protein_CA.positions)
    else:
        for i, ts in enumerate(u.trajectory[frame_cluster]):
            pos[i] = torch.from_numpy(protein_CA.positions)
    pos -= pos.mean(1).unsqueeze(1)
    return pos


def signal_std_torch_batch(img):
    n_pixels = img.shape[1]
    radius = n_pixels * 0.4
    mask = circular_mask(n_pixels, radius)
    image_masked = img[:, mask]
    signal_std = image_masked.pow(2).mean(1).sqrt()
    return signal_std


def align_traj(
    top_image="image.gro",
    traj_image="image.xtc",
    top_struc="struc.gro",
    traj_struc="struc.xtc",
    outdir="./output/",
    device="cpu",
):

    uImage = mda.Universe(top_image, traj_image)
    uStruc = mda.Universe(top_struc, traj_struc)
    posImage = mdau_to_pos_arr(uImage)
    posStruc = mdau_to_pos_arr(uStruc)
    nImage = posImage.shape[0]
    nStruc = posStruc.shape[0]
    output_directory = outdir
    device = torch.device(device)
    rot_mats = torch.empty((nStruc, nImage, 3, 3), dtype=torch.float64, device="cpu")
    posImage = posImage.to(device)
    posStruc = posStruc.to(device)
    n_batch = 1
    batch_size = nStruc // n_batch
    print("Calculating rotation matrices...")
    for i_batch in tqdm(range(n_batch)):
        batch_start = i_batch * batch_size
        batch_end = (i_batch + 1) * batch_size
        Hs = torch.einsum("nji,mjk->nmik", posImage, posStruc[batch_start:batch_end])
        u, s, vh = torch.linalg.svd(Hs.flatten(0, 1))
        v = vh.transpose(1, 2)
        R = torch.matmul(v, u.transpose(1, 2))
        rot_mats[batch_start:batch_end, :, :, :] = torch.reshape(R.cpu(), Hs.shape).transpose(0, 1)
    rot_mats = rot_mats.cpu().numpy()
    np.save(output_directory + "rot_mats_struc_image.npy", rot_mats)
    print("Done!")
    return rot_mats


def make_synthetic_images(
    top_image="image.gro",
    traj_image="image.xtc",
    n_pixel=128,
    pixel_size=0.2,
    sigma=1.5,
    snr=1e-2,
    n_image_per_struc=1,
    add_ctf=False,
    defocus_min=0.027,
    defocus_max=0.090,
    device="cpu",
    batch_size=16,
    outdir=None,
):

    file_prefix = "npix%d_ps%.2f_s%.1f_snr%.1E" % (n_pixel, pixel_size, sigma, snr)
    print("Reading trajectory...")
    # Reading image trajectory
    print("Reading image trajectory from %s and %s..." % (top_image, traj_image))
    uImg = mda.Universe(top_image, traj_image)
    coord_img = mdau_to_pos_arr(uImg)
    if n_image_per_struc > 1:
        coord_img = coord_img.repeat(n_image_per_struc, 1, 1)
    coord_img = coord_img.to(device)
    rot_mats, ctfs, images = generate_images(
        coord_img,
        n_pixel=n_pixel,
        pixel_size=pixel_size,
        sigma=sigma,
        snr=snr,
        add_ctf=add_ctf,
        defocus_min=defocus_min,
        defocus_max=defocus_max,
        batch_size=batch_size,
        device=device,
    )
    if outdir is not None:
        print("Saving images to %s..." % outdir)
        np.save("%s/rot_mats_%s.npy" % (outdir, file_prefix), rot_mats.cpu().numpy())
        np.save("%s/ctf_%s.npy" % (outdir, file_prefix), ctfs.cpu().numpy())
        np.save("%s/images_%s.npy" % (outdir, file_prefix), images.cpu().numpy())
        print("Done!")
    return rot_mats, ctfs, images


def calc_image_struc_distance(
    images=None,
    ctfs=None,
    rot_mats_image=None,
    top_struc="struc.gro",
    traj_struc="struc.xtc",
    rotmat_struc_imgstruc=None,
    outdir="./output/",
    n_pixel=128,
    pixel_size=0.2,
    sigma=1.5,
    snr=1e-2,
    add_ctf=False,
    defocus_min=0.027,
    defocus_max=0.090,
    device="cpu",
    batch_size=16,
):

    file_prefix = "npix%d_ps%.2f_s%.1f_snr%.1E" % (n_pixel, pixel_size, sigma, snr)
    print("Reading trajectory from %s and %s..." % (top_struc, traj_struc))
    uStr = mda.Universe(top_struc, traj_struc)
    coord_str = mdau_to_pos_arr(uStr)
    n_struc = coord_str.shape[0]
    if rotmat_struc_imgstruc is not None:
        print("Reading struc-images alignment matrices from %s..." % rotmat_struc_imgstruc)
        rot_mats_align = torch.from_numpy(np.load(rotmat_struc_imgstruc)).to(device)
    else:
        print("No struc-imgstruc alignment matrices specified, assume only using poses")
    if images is None:
        print("Loading images from %s/images_%s.npy..." % (outdir, file_prefix))
        images = np.load("%s/images_%s.npy" % (outdir, file_prefix))
    if ctfs is None:
        print("Loading CTFs from %s/ctf_%s.npy..." % (outdir, file_prefix))
        ctfs = np.load("%s/ctf_%s.npy" % (outdir, file_prefix))
    if rot_mats_image is None:
        print("Loading poses from %s/rot_mats_%s.npy..." % (outdir, file_prefix))
        rot_mats_image = np.load("%s/rot_mats_%s.npy" % (outdir, file_prefix))
    if not torch.is_tensor(images):
        images = torch.from_numpy(images)
    if not torch.is_tensor(ctfs):
        ctfs = torch.from_numpy(ctfs)
    if not torch.is_tensor(rot_mats_image):
        rot_mats_image = torch.from_numpy(rot_mats_image)
    images = images.to(device)
    ctfs = ctfs.to(device)
    rot_mats_image = rot_mats_image.to(device)
    n_image = images.shape[0]
    diff = np.zeros((n_struc, n_image), dtype=float)
    for i in tqdm(range(n_struc), desc="Computing image-structure distance for structure"):
        if rotmat_struc_imgstruc is not None:
            aligned_coord = (
                coord_str[i].unsqueeze(0).matmul(rot_mats_align[i]).matmul(rot_mats_image)
            )
        else:
            aligned_coord = coord_str[i].unsqueeze(0).matmul(rot_mats_image)
        diff[i] = calc_struc_image_diff(
            aligned_coord,
            n_pixel=n_pixel,
            pixel_size=pixel_size,
            sigma=sigma,
            images=images,
            ctfs=ctfs,
            batch_size=batch_size,
            device=device,
        )
    print("Saving...")
    np.save("%s/diff_%s.npy" % (outdir, file_prefix), diff)
    print("Done!")
    return diff


def approx_lmbd(
    top_struc,
    traj_struc,
    n_pixel,
    pixel_size,
    sigma,
    signal_to_noise_ratio,
    add_ctf,
    defocus_min,
    defocus_max,
    n_image_per_struc,
    n_batch,
    device,
):

    uStruc = mda.Universe(top_struc, traj_struc)
    posStruc = mdau_to_pos_arr(uStruc)
    posStruc = posStruc.repeat(n_image_per_struc, 1, 1)
    _, _, images = generate_images(
        posStruc,
        n_pixel=n_pixel,
        pixel_size=pixel_size,
        sigma=sigma,
        snr=np.inf,
        rotation=True,
        add_ctf=add_ctf,
        defocus_min=defocus_min,
        defocus_max=defocus_max,
        batch_size=n_batch,
        device=device,
    )
    signal_std = signal_std_torch_batch(images)
    snr = signal_to_noise_ratio
    noise_std = signal_std / np.sqrt(snr)
    lmbd = noise_std.mean().numpy()
    return lmbd
