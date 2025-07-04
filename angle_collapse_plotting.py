#%%
import numpy as np
import matplotlib.pyplot as plt

# Define cotangent
def cot(a):
    return 1/np.tan(a)

# Define RHS of the system
def theta_phi_dot(theta, phi):
    alpha_over_H=1
    theta_dot = alpha_over_H*(np.sin(theta)**2) * (
        2*cot(phi)
        - (cot(phi)**2)*cot(theta)
        - (cot(theta)**3)
        - cot(theta)*(cot(theta) + cot(phi))**2
    )

    phi_dot =alpha_over_H* (np.sin(phi)**2) * (
        2*cot(theta)
        - (cot(theta)**2)*cot(phi)
        - (cot(phi)**3)
        - cot(phi)*(cot(theta) + cot(phi))**2
    )

    return theta_dot, phi_dot

# Set up grid
phi = np.arange(0.01, np.pi - 0.01, 0.01) # 0.05 for vector fields, 0.01 for heat map
theta = np.arange(0.01, np.pi - 0.01, 0.01)
THETA, PHI = np.meshgrid(theta, phi)

# Compute vector field
THETA_DOT, PHI_DOT = theta_phi_dot(THETA, PHI)

# Mask out region where theta + phi >= pi
mask = (THETA + PHI) > np.pi-0.01
THETA_DOT = np.where(mask, np.nan, THETA_DOT)
PHI_DOT = np.where(mask, np.nan, PHI_DOT)
THETA_DOT_sqrd=THETA_DOT**2
PHI_DOT_sqrd=PHI_DOT**2
MAG_VECTORS=np.sqrt(THETA_DOT_sqrd+PHI_DOT_sqrd)
THETA_DOT_norm=THETA_DOT/MAG_VECTORS
PHI_DOT_norm=PHI_DOT/MAG_VECTORS


# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(THETA, PHI, THETA_DOT, PHI_DOT, color='black', angles='xy', scale_units='xy', scale=10, width=0.001)

ax.set_xlim(0, np.pi)
ax.set_ylim(0, np.pi)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax.set_title('Phase Plane Vector Field (valid region: θ + φ < π)')

# Add triangle boundary line
ax.plot([0, np.pi], [np.pi, 0], 'k--', lw=1)

ax.set_aspect('equal')
plt.show()
# %%

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.quiver(THETA, PHI, THETA_DOT_norm, PHI_DOT_norm, color='black', angles='xy', scale_units='xy', scale=25, width=0.002)

ax.set_xlim(0, np.pi)
ax.set_ylim(0, np.pi)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax.set_title('Phase Plane Normalised Vector Field (valid region: θ + φ < π)')

# Add triangle boundary line
ax.plot([0, np.pi], [np.pi, 0], 'k--', lw=1)

ax.set_aspect('equal')
plt.show()
# %%
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

ticks = [ 0, np.pi / 4, np.pi / 2, 3*np.pi / 4, np.pi]
tick_labels = [r"$0$",r"$\pi/4$", r"$\pi/2$",r"$3\pi/4$", r"$\pi$"]



save_dir="/Users/luke_dev/Documents/Phase_plane_triangle_analysis"
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16, 8))
axis_font_size=15
ax1.quiver(THETA, PHI, THETA_DOT, PHI_DOT, color='black', angles='xy', scale_units='xy', scale=10, width=0.002)

ax1.set_xlim(0, np.pi)
ax1.set_ylim(0, np.pi)
ax1.set_xlabel(r'$\theta$',fontsize=axis_font_size)
ax1.set_ylabel(r'$\phi$',fontsize=axis_font_size, rotation=0)
ax1.set_title('Vector Field')
ax1.set_yticks(ticks)
ax1.set_xticks(ticks)
ax1.set_yticklabels(tick_labels)
ax1.set_xticklabels(tick_labels)
# Add triangle boundary line
ax1.plot([0, np.pi], [np.pi, 0], 'k--', lw=1)

ax1.set_aspect('equal')


ax2.quiver(THETA, PHI, THETA_DOT_norm, PHI_DOT_norm, color='black', angles='xy', scale_units='xy',scale=25, width=0.002)

ax2.set_xlim(0, np.pi)
ax2.set_ylim(0, np.pi)
ax2.set_xlabel(r'$\theta$',fontsize=axis_font_size)
ax2.set_ylabel(r'$\phi$',fontsize=axis_font_size,rotation=0)
ax2.set_title('Normalised Vector Field')
ax2.set_yticks(ticks)
ax2.set_xticks(ticks)
ax2.set_yticklabels(tick_labels)
ax2.set_xticklabels(tick_labels)
# Add triangle boundary line
ax2.plot([0, np.pi], [np.pi, 0], 'k--', lw=1)

ax2.set_aspect('equal')
fname = f"{save_dir}/phase_plane_norm_and_not_norm.png"
plt.savefig(fname, dpi=1200)
plt.show()

# %%
# Plot heat map
import matplotlib.colors as mcolors

vmin = np.nanpercentile(MAG_VECTORS[MAG_VECTORS>0], 0.1)
vmax = np.nanpercentile(MAG_VECTORS[MAG_VECTORS>0], 99.9)

fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.pcolormesh(THETA, PHI, MAG_VECTORS,
                         shading='auto',
                         cmap='hot',
                         norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
fig.colorbar(heatmap, ax=ax, label='Vector Magnitude (log scale)')

ax.plot([0, np.pi], [np.pi, 0], 'k--', lw=1)
ax.set_xlim(0, np.pi)
ax.set_ylim(0, np.pi)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\phi$')
ax.set_title('Phase Plane Magnitude Heat Map (valid region: θ + φ < π)')
ax.set_aspect('equal')

plt.show()
# %%
