from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mdtraj')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='mdtraj')
traj = md.load('adk.dcd', top='adk.pdb')
traj = traj[:10000]
topology = traj.topology
print(f"Loaded trajectory with {traj.n_frames} frames")
nmp_ca = topology.select('name CA and resid 35 to 55')
core_nmp_ca = topology.select('name CA and resid 90 to 100')
core_lid_ca = topology.select('name CA and resid 115 to 125')
lid_ca = topology.select('name CA and resid 125 to 153')
core_lid_angle_ca = topology.select('name CA and resid 179 to 185')
print("Calculating CVs...")
center_nmp = traj.xyz[:, nmp_ca, :].mean(axis=1)
center_core_nmp = traj.xyz[:, core_nmp_ca, :].mean(axis=1)
center_core_lid = traj.xyz[:, core_lid_ca, :].mean(axis=1)
vec1_nmp = center_core_lid - center_core_nmp
vec2_nmp = center_nmp - center_core_nmp
dot_nmp = np.sum(vec1_nmp * vec2_nmp, axis=1)
norm1_nmp = np.linalg.norm(vec1_nmp, axis=1)
norm2_nmp = np.linalg.norm(vec2_nmp, axis=1)
cos_nmp = np.clip(dot_nmp / (norm1_nmp * norm2_nmp), -1.0, 1.0)
theta_nmp = np.degrees(np.arccos(cos_nmp))
center_lid = traj.xyz[:, lid_ca, :].mean(axis=1)
center_core_lid_angle = traj.xyz[:, core_lid_angle_ca, :].mean(axis=1)
vec1_lid = center_core_lid_angle - center_core_lid
vec2_lid = center_lid - center_core_lid
dot_lid = np.sum(vec1_lid * vec2_lid, axis=1)
norm1_lid = np.linalg.norm(vec1_lid, axis=1)
norm2_lid = np.linalg.norm(vec2_lid, axis=1)
cos_lid = np.clip(dot_lid / (norm1_lid * norm2_lid), -1.0, 1.0)
theta_lid = np.degrees(np.arccos(cos_lid))
print(f"θNMP range: {theta_nmp.min():.1f}° - {theta_nmp.max():.1f}°")
print(f"θLID range: {theta_lid.min():.1f}° - {theta_lid.max():.1f}°")
cv_data = np.column_stack((theta_nmp, theta_lid))
np.savetxt("adk_cvs.dat", cv_data, fmt="%.2f", header="θNMP θLID", comments="# ")
print("CVs saved to adk_cvs.dat")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_nmp, theta_lid, '-', linewidth=0.8, alpha=0.7, color='darkblue')
# Mark start and end points
ax.scatter(theta_nmp[0], theta_lid[0], c='green', s=100, marker='o', 
           label='Start', zorder=5, edgecolors='black', linewidth=1.5)
ax.scatter(theta_nmp[-1], theta_lid[-1], c='red', s=100, marker='s', 
           label='End', zorder=5, edgecolors='black', linewidth=1.5)

ax.set_xlabel('NMP-CORE angle (θNMP)', fontsize=14)
ax.set_ylabel('LID-CORE angle (θLID)', fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best')

margin = 2  # degrees
ax.set_xlim([theta_nmp.min()-margin, theta_nmp.max()+margin])
ax.set_ylim([theta_lid.min()-margin, theta_lid.max()+margin])

plt.tight_layout()
plt.savefig("adk_cv_trajectory.png", dpi=300)
print("Trajectory plot saved to adk_cv_trajectory.png")
plt.show()
print("\nCalculating free energy landscape...")
bins = 50  
H, xedges, yedges = np.histogram2d(theta_nmp, theta_lid, bins=bins)
P = H / H.sum()
# Calculate free energy: F = -kT ln(P)
kT = 0.596  # kcal/mol at 300K
F = np.zeros_like(P)
F[P > 0] = -kT * np.log(P[P > 0])
F = F - F.min()
F_smooth = gaussian_filter(F, sigma=1)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
fig, ax = plt.subplots(figsize=(6, 4))
levels = np.linspace(0, min(F_smooth.max(), 5), 20)  
contour = ax.contourf(X, Y, F_smooth.T, levels=levels, cmap='jet')
ax.contour(X, Y, F_smooth.T, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Free energy (kcal/mol)', fontsize=12)
ax.set_xlabel('NMP-CORE angle (θNMP)', fontsize=12)
ax.set_ylabel('LID-CORE angle (θLID)', fontsize=12)
plt.tight_layout()
plt.savefig("adk_free_energy.png", dpi=300)
print("Free energy plot saved to adk_free_energy.png")
plt.show()
print("\n=== Summary Statistics ===")
print(f"Total frames: {traj.n_frames}")
print(f"Simulation time: {traj.time[-1] - traj.time[0]:.2f} ps")
print(f"\nCV Statistics:")
print(f"θNMP: mean = {theta_nmp.mean():.1f}°, std = {theta_nmp.std():.1f}°")
print(f"θLID: mean = {theta_lid.mean():.1f}°, std = {theta_lid.std():.1f}°")
# Identify most visited state
hist_max_idx = np.unravel_index(H.argmax(), H.shape)
most_visited_nmp = (xedges[hist_max_idx[0]] + xedges[hist_max_idx[0]+1]) / 2
most_visited_lid = (yedges[hist_max_idx[1]] + yedges[hist_max_idx[1]+1]) / 2
print(f"\nMost visited state: θNMP = {most_visited_nmp:.1f}°, θLID = {most_visited_lid:.1f}°")
