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
# %% Plotting energy landscape with stream lines 
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
plt.figure(figsize=(8, 5))
save_dir="/Users/luke_dev/Documents/Phase_plane_triangle_analysis"
# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

# --- Left plot: Dumbbell Model ---
# Grid in Q1, Q2
Q1 = np.linspace(-2, 2, 300)
Q2 = np.linspace(-2, 2, 300)
Q1g, Q2g = np.meshgrid(Q1, Q2)
levels=50
# Potential energy U = 1/2 (Q1^2 + Q2^2)
U1 = 0.5 * (Q1g**2 + Q2g**2)

# Filled contours
levels1 = levels
cs1 = ax1.contourf(Q1g, Q2g, U1, levels=levels1, cmap='viridis')
# Dotted contour lines
ax1.contour(Q1g, Q2g, U1, levels=levels1, colors='k', linestyles=':')
ax1.set_aspect('equal')
# ax1.set_title(r'$U=\tfrac12\,(Q_1^2 + Q_2^2)$')
ax1.set_xlabel('$Q_1$')
ax1.set_ylabel('$Q_2$')
plt.colorbar(cs1, ax=ax1, shrink=0.8)

# --- Right plot: Area Model ---
# Grid in ell, theta
ell = np.linspace(0, 10, 300)
θ = np.linspace(0, np.pi, 300)
ellg, θg = np.meshgrid(ell, θ)

# Potential energy U = 1/2 (ell sin θ)^2
U2 = 0.5 * (ellg * np.sin(θg))**2 

# Filled contours
levels2 = levels
cs2 = ax2.contourf(ellg, θg, U2, levels=levels2, cmap='viridis')
# Dotted contour lines
ax2.contour(ellg, θg, U2, levels=levels2, colors='k', linestyles='-')
# ax2.set_title(r'$U=\tfrac12\,(\ell\sin\theta)^2$')
ax2.set_xlabel('$\ell$')
ax2.set_ylabel(r'$\theta$')
plt.colorbar(cs2, ax=ax2, shrink=0.8)

# Tight layout
plt.tight_layout()

plt.show()

# dumbbell field 
# Define grid
l1 = np.linspace(0, 5, 400)
theta = np.linspace(0.01, np.pi - 0.01, 400)
L1, THETA = np.meshgrid(l1, theta)

# Define energy function
U = 0.5 * (L1 * np.sin(THETA))**2

# Compute gradients (negative for descent)
dU_dl1 = L1 * np.sin(THETA)**2
dU_dtheta = 0.5 * L1**2 * np.sin(2 * THETA)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Contour plot
contours = ax.contourf(L1, THETA, U, levels=30, cmap='viridis')
cbar = plt.colorbar(contours, ax=ax)
cbar.set_label( 'U', fontsize=12,rotation=0)

# Add streamlines
ax.streamplot(l1, theta, -dU_dl1, -dU_dtheta, color='k', density=2, arrowsize=1)

# # Plot zero-energy lines
# ax.axvline(0, color='red', linestyle='--', linewidth=2, label=r'$l_1 = 0$')
# ax.axhline(0, color='red', linestyle='--', linewidth=2, label=r'$\theta = 0$')
# ax.axhline(np.pi, color='red', linestyle='--', linewidth=2, label=r'$\theta = \pi$')

# Labels
ax.set_xlabel(r'$L$', fontsize=14)
ax.set_ylabel(r'$\theta$', fontsize=14)
ax.set_title('Area-Based Model: $U = \\frac{1}{2}(L \\sin\\theta)^2$\n', fontsize=14)

# # Add annotations
# ax.text(0.2, np.pi / 2, 'Zero-energy line:\n$l_1 = 0$', color='red', fontsize=12)
# ax.text(1.1, 0.1, r'$\theta = 0$', color='red', fontsize=12)
# ax.text(1.1, np.pi - 0.1, r'$\theta = \pi$', color='red', fontsize=12)

plt.tight_layout()
fname = f"{save_dir}/area_energy_landscape.png"
plt.savefig(fname, dpi=1200)
plt.show()
# %%
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
plt.figure(figsize=(8, 5))


# Define grid
q1 = np.linspace(-3, 3, 400)
q2 = np.linspace(-3, 3, 400)
Q1, Q2 = np.meshgrid(q1, q2)

# Energy function: U = 0.5 * (Q1^2 + Q2^2)
U = 0.5 * (Q1**2 + Q2**2)

# Gradient (negative for descent)
dU_dq1 = Q1
dU_dq2 = Q2

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Contour plot
contours = ax.contourf(Q1, Q2, U, levels=30, cmap='viridis')
cbar = plt.colorbar(contours, ax=ax)
cbar.set_label('U', fontsize=12, rotation=0)

# Add streamlines
ax.streamplot(q1, q2, -dU_dq1, -dU_dq2, color='k', density=2, arrowsize=1)

# Labels
ax.set_xlabel(r'$Q_1$', fontsize=14)
ax.set_ylabel(r'$Q_2$', fontsize=14)
ax.set_title('Dumbbell Model: $U = \\frac{1}{2}(Q_1^2 + Q_2^2)$', fontsize=14)

plt.tight_layout()
fname = f"{save_dir}/db_energy_landscape.png"
plt.savefig(fname, dpi=1200)
plt.show()
# %%
# Define grid
l1 = np.linspace(0.01, 5, 50)
l2 = np.linspace(0.01, 5, 50)
L1, L2 = np.meshgrid(l1, l2)

# Energy function: U = 0.5 * l1 * l2
U = 0.5 * L1 * L2

# Compute negative gradient
dU_dl1 = 0.5 * L2
dU_dl2 = 0.5 * L1

# Plotting
fig, ax = plt.subplots(figsize=(6, 5))

# Contour plot
levels = 30
cs = ax.contourf(L1, L2, U, levels=levels, cmap='viridis')
ax.contour(L1, L2, U, levels=levels, colors='k', linestyles=':')

# Add quiver plot
ax.streamplot(L1, L2, -dU_dl1, -dU_dl2, color='k', density=2, arrowsize=1)

# Labels
ax.set_xlabel(r'$l_1$', fontsize=14)
ax.set_ylabel(r'$l_2$', fontsize=14)
ax.set_title(r'Rectangular Model: $U = \frac{1}{2} l_1 l_2$', fontsize=14)
plt.colorbar(cs, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()
# %%
