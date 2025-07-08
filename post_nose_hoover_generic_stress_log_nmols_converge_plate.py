# %% new section where stress and log files are used for intial run 
import os
import re
import glob
import pandas as pd
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
from plotting_module import *
import seaborn as sns 
# === Setup

path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/langevin_runs/n_mol_converge/DB_nmols_converge_test_run_n_mols_range_63_5696_tstep_1e-05__mass_1_stiff_0.25_2.0_1_strain_100_T_1_R_0.1_R_n_1"
#path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/n_mols_converge/"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/n_mol_converge/plate_nmols_converge_test_run_n_mols_range_63_5696_tstep_1e-05__mass_1_stiff_0.25_2.0_1_BK_1000_T_1_R_0.5_R_n_1"
os.chdir(path_2_files)
mol_density=13500/(300**3)
box_size_bar=np.array([50,75,100,125,150,175,200,225]).astype('int')


vol=box_size_bar**3
n_mols=np.ceil((vol*mol_density)).astype('int')


os.chdir(path_2_files)
K = 1.0
mass=1
n_shear_points=1
log_name_list = glob.glob("log*_K_"+str(K))

erate=np.array([0.1])

spring_relaxation_time=np.sqrt(mass/K)
Wi=erate*spring_relaxation_time
reals=3
#%%
def read_lammps_log_incomplete(filename):
    """
    Reads LAMMPS log file safely, extracting thermo data blocks, ignoring incomplete rows.
    
    Returns:
        thermo_data (list of pd.DataFrame): List of DataFrames for each thermo block
    """

    thermo_data = []
    current_headers = None
    current_block = []

    with open(filename, 'r', errors='ignore') as f:
        for line in f:
            stripped = line.strip()

            # Detect start of thermo block
            if stripped.startswith("Step"):
                # If there was previous block, save it
                if current_headers and current_block:
                    df = pd.DataFrame(current_block, columns=current_headers)
                    thermo_data.append(df)

                # Start new block
                current_headers = stripped.split()
                current_block = []
                continue

            # If we are inside thermo block
            if current_headers:
                if stripped == "" or not stripped[0].isdigit():
                    # Likely end of block
                    continue

                parts = stripped.split()

                # Check if row is complete
                if len(parts) == len(current_headers):
                    try:
                        current_block.append([float(x) for x in parts])
                    except ValueError:
                        pass  # Skip malformed line
                else:
                    pass  # Skip incomplete line

        # Save last block
        if current_headers and current_block:
            df = pd.DataFrame(current_block, columns=current_headers)
            thermo_data.append(df)

    return thermo_data
#%%
# Example usage:
# filename = ""
# thermo_data = read_lammps_log_incomplete(filename)

# Show summary
# for i, df in enumerate(thermo_data):
#     print(f"Thermo block {i+1}: {df.shape[0]} rows")
#     display(df.head())


def read_lammps_log_if_complete(filename):
    """
    Reads a LAMMPS log file by filename and returns a list of thermo DataFrames
    if the file is complete (i.e., ends with 'Total wall time: ...').

    Parameters:
        filename (str): Name of the log file (in current working directory or script's path).

    Returns:
        list of pd.DataFrame or None: Thermo sections if valid, else None.
    """
    try:
        with open(filename, 'r', errors='ignore') as f:
            lines = f.readlines()

        # Check for completion
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if not non_empty_lines or not non_empty_lines[-1].startswith("Total wall time"):
            return None

        # Extract thermo sections
        thermo_sections = []
        header_indices = [i for i, l in enumerate(lines) if l.strip().startswith("Step")]

        for idx in header_indices:
            headers = lines[idx].split()
            section = []
            for line in lines[idx + 1:]:
                if line.strip().startswith("Step"): break
                parts = line.strip().split()
                if len(parts) == len(headers):
                    section.append(parts)
                else:
                    break
            if section:
                df = pd.DataFrame(section, columns=headers).apply(pd.to_numeric)
                thermo_sections.append(df)

        return thermo_sections if thermo_sections else None

    except Exception as e:
        print(f"❌ Error reading '{filename}': {e}")
        return None

data=read_lammps_log_incomplete(log_name_list[0])
eq_columns=list(data[0].columns)
eq_outs=50

real_target = 3
box_size_count = np.zeros(box_size_bar.size, dtype=int)

# Preallocate data arrays
eq_log_data_array = np.zeros((real_target, box_size_bar.size, eq_outs, 7))

for file in log_name_list:

    data = read_lammps_log_incomplete(file)

    if data is None:
        continue

    # Extract shear rate from filename
    file_meta_data = file.split("_")
    print(file_meta_data)
    box_size_file = round(float(file_meta_data[14]), 7)
    box_size_index = int(np.where(box_size_file == box_size_bar)[0])

    # Check if real_target already reached
    if box_size_count[box_size_index] >= real_target:
        continue

    # Assign realisation index (zero-based)
    real_index = box_size_count[box_size_index]

    # Extract thermo outputs as numpy arrays
    eq_log_data_array_raw = data[0].to_numpy()
   

    print(eq_log_data_array_raw.shape)
  

    # Store data
    eq_log_data_array[real_index, box_size_index] = eq_log_data_array_raw[:eq_outs]
   

    # Increment count
    box_size_count[box_size_index] += 1

print(box_size_count)

print(eq_log_data_array.shape)
#%% stress data 


def read_stress_tensor_file(filename='stress_tensor_avg.dat', volume=vol, return_data=True):
    """
    Analyze LAMMPS time-averaged global stress (unnormalized by volume).
    
    Parameters:
    -----------
    filename : str
        File containing time and raw summed stresses.
    volume : float
        Simulation box volume to normalize (mandatory).
    show_plots : bool
        Whether to plot.
    return_data : bool
        Whether to return arrays.

    Returns:
    --------
    Dictionary of time series.
    """
    if volume is None:
        raise ValueError("You must specify the box volume to normalize stress!")

    data = np.loadtxt(filename, comments='#')
    time, sxx_sum, syy_sum, szz_sum, sxy_sum, sxz_sum, syz_sum = data.T

    # Normalize stress components
    sxx = sxx_sum / volume
    syy = syy_sum / volume
    szz = szz_sum / volume
    sxy = sxy_sum / volume
    sxz = sxz_sum / volume
    syz = syz_sum / volume

    N1 = sxx - syy
    N2 = syy - szz

    if return_data:
        return {
            'time': time,
            '$\sigma_{xx}$': sxx, '$\sigma_{yy}$': syy, '$\sigma_{zz}$': szz,
            '$\sigma_{xy}$': sxy, '$\sigma_{xz}$': sxz, '$\sigma_{yz}$': syz,
            '$N_{1}$': N1, '$N_{2}$': N2
        }
    
stress_name_list=glob.glob("eq_stress*K_"+str(K)+"*.dat")
print(stress_name_list)
data_dict = read_stress_tensor_file(filename=stress_name_list[0], volume=vol[0], return_data=True)
stress_columns = list(data_dict.keys())
output_cutoff=eq_outs
real_target = 3
n_mol_count = np.zeros(n_mols.size, dtype=int)
stress_array = np.zeros((real_target, n_mols.size, output_cutoff, 9))


#%%
for file in stress_name_list:
    file_meta_data = file.split("_")
    print(file_meta_data)
    # #DB
    # box_side=int(file_meta_data[10])
    # box_index=np.where(box_side==box_size_bar)[0][0]

    # # plate 
    box_side=int(file_meta_data[18])
    box_index=np.where(box_side==box_size_bar)[0][0]
    print(box_index)

    data_dict = read_stress_tensor_file(filename=file, volume=vol[box_index], return_data=True)
    
    if data_dict is None:
        continue

    if data_dict["time"].size <output_cutoff:
        continue

    # DB
    # real_index = int(file_meta_data[9])   # zero-based indexing
    # # plate 
    real_index = int(file_meta_data[17])   # zero-based indexing
    print(real_index)

    if real_index >= real_target:
        continue  # skip if real_index exceeds target
    
    n_mol_count[box_index] += 1
    # print(timestep[timestep_index])
    print(f"Realisation: {real_index}")

    # # Fill stress array
    for column, key in enumerate(stress_columns):
         raw_stress_array = data_dict[key][:output_cutoff]
         stress_array[real_index, box_index, :, column] = raw_stress_array

# Compute mean
mean_stress_array = np.mean(stress_array, axis=0)
print(n_mol_count)

print("Mean stress array shape:", mean_stress_array.shape)

#%% plate orientations 
spring_name_list=glob.glob("eq*hookean*K_"+str(K)+".dump")
def read_lammps_posvel_dump_to_numpy(filename):
    timesteps_data = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break  # End of file

            if "ITEM: TIMESTEP" in line:
                timestep = int(f.readline().strip())
                f.readline()  # ITEM: NUMBER OF ATOMS
                num_atoms = int(f.readline().strip())
                f.readline()  # ITEM: BOX BOUNDS
                for _ in range(3):
                    f.readline()  # Skip box bounds
                f.readline()  # ITEM: ATOMS

                atoms_data = []
                for _ in range(num_atoms):
                    parts = f.readline().split()
                    atom_data = [float(x) for x in parts]  # id, type, xu, yu, zu, vx, vy, vz
                    atoms_data.append(atom_data)

                atoms_array = np.array(atoms_data, dtype=np.float64)
                timesteps_data.append(atoms_array)

    result_array = np.array(timesteps_data)  # Shape: (timesteps, atoms, 8)
    return result_array


def convert_cart_2_spherical_y_inc_plate_from_dump(dump_array, n_mols, output_cutoff,box_index):
    position_array = dump_array[:output_cutoff, :, 2:5]
    vel_array=dump_array[:output_cutoff, :, 5:]
    print("position_array shape:", position_array.shape)

    # Reshape into (timesteps, plates, 3 vertices, 3 coords)
    position_plates_array = np.reshape(position_array, (output_cutoff, n_mols[box_index], 3, 3))

    # Define edge vectors from vertex 0
    ell_1 = position_plates_array[:, :, 1] - position_plates_array[:, :, 0]
    ell_2 = position_plates_array[:, :, 2] - position_plates_array[:, :, 0]

    print("ell_1 shape", ell_1.shape)
    print("ell_2 shape", ell_2.shape)

    # Compute area vectors (normal to plate)
    area_vector = np.cross(ell_1, ell_2, axis=-1)
    print("area_vector shape", area_vector.shape)

    # Flip vectors to point in +Y hemisphere
    area_vector[area_vector[:, :, 1] < 0] *= -1

    x = area_vector[:, :, 0]
    y = area_vector[:, :, 1]
    z = area_vector[:, :, 2]

    spherical_coords_array = np.zeros((output_cutoff, n_mols[box_index], 3))

    # r: radial magnitude
    spherical_coords_array[:, :, 0] = np.sqrt(x**2 + y**2 + z**2)

    # theta: azimuthal angle in XZ plane (from +X to +Z)
    spherical_coords_array[:, :, 1] = np.arctan2(z, x)

    # phi: polar angle from +Y axis down
    spherical_coords_array[:, :, 2] = np.arccos(y / spherical_coords_array[:, :, 0])

    return spherical_coords_array,position_array,vel_array


dump_data = read_lammps_posvel_dump_to_numpy(spring_name_list[0])

# creating list of arrays to contain the dumps 
box_sizes_list_array=[]
spherical_box_sizes_array=[]
for i in range(box_size_bar.size):
    array=np.zeros((real_target,output_cutoff,n_mols[i],3))
    box_sizes_list_array.append(array)
    spherical_box_sizes_array.append(array)

#%%

# creating dict to store the list in 
#spring_data_dict={'box_sizes':box_sizes_list_array}
spherical_coords_data_dict={'box_sizes':spherical_box_sizes_array}


for file in spring_name_list:
    file_meta_data = file.split("_")
    print(file_meta_data)
    
    # plate
    real_index = int(file_meta_data[14])   # zero-based indexing
    print(real_index)
    box_side=int(file_meta_data[15])
    print(box_side)
    box_index=np.where(box_side==box_size_bar)[0][0]
    print(box_index)

    dump_data = read_lammps_posvel_dump_to_numpy(file)


    spherical_coords_array,position_array,vel_array=convert_cart_2_spherical_y_inc_plate_from_dump(dump_data,n_mols,output_cutoff,box_index)
   
        # #spring_data_dict["box_sizes"][box_index][real_index,i]=dump_data_np_array
    spherical_coords_data_dict["box_sizes"][box_index][real_index]=spherical_coords_array


#%% plotting distributions 


def plot_spherical_kde_nmols(spherical_coords_data_dict, spherical_box_sizes_array, n_mols, cutoff=400, save=False, save_dir="plots", use_latex=True):
    """
    KDE plots of spherical coordinate data (rho, theta, phi) aggregated over all box sizes.
    Each subplot contains all box sizes plotted together with legends.
    """

    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    pi_theta_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    pi_theta_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    pi_phi_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
    pi_phi_labels = [r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"]

    if save:
        os.makedirs(save_dir, exist_ok=True)

    # Create figure with 3 subplots (rho, theta, phi)
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    smooth=1
    # Iterate over each box and plot its KDE on the respective axis
    for i in range(len(spherical_box_sizes_array)):
        box_data = spherical_coords_data_dict["box_sizes"][i][:, cutoff:, :, :]
        label = f"$N_{{\mathrm{{mols}}}}=${n_mols[i]}"

        # Flatten the data
        rho = np.ravel(box_data[:, :, :, 0])
        theta = np.ravel(np.concatenate([
            box_data[:, :, :, 1] - 2 * np.pi,
            box_data[:, :, :, 1],
            box_data[:, :, :, 1] + 2 * np.pi
        ]))
        phi = np.ravel(np.concatenate([
            box_data[:, :, :, 2],
            np.pi - box_data[:, :, :, 2]
        ]))

        # --- RHO ---
        sns.kdeplot(rho, ax=axes[0], linewidth=2, label=label,bw_adjust=smooth)

        # --- THETA ---
        sns.kdeplot(theta, ax=axes[1], linewidth=2, label=label,bw_adjust=smooth)

        # --- PHI ---
        sns.kdeplot(phi, ax=axes[2], linewidth=2, label=label,bw_adjust=smooth)

    # Customize each axis
    #axes[0].set_title(r"$\rho$ Distribution")
    axes[0].set_xlabel(r"$\rho$")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    #axes[1].set_title(r"$\Theta$ Distribution")
    axes[1].set_xlabel(r"$\Theta$")
    axes[1].set_ylabel("Density")
    axes[1].set_xticks(pi_theta_ticks)
    axes[1].set_xticklabels(pi_theta_labels)
    axes[1].set_xlim(-np.pi, np.pi)
    axes[1].legend(frameon=False)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    #axes[2].set_title(r"$\phi$ Distribution")
    axes[2].set_xlabel(r"$\phi$")
    axes[2].set_ylabel("Density")
    axes[2].set_xticks(pi_phi_ticks)
    axes[2].set_xticklabels(pi_phi_labels)
    axes[2].set_xlim(0, np.pi / 2)
    axes[2].legend(frameon=False)
    axes[2].grid(True, linestyle='--', alpha=0.7)

    fig.subplots_adjust(hspace=0.4, top=0.93)
    fig.suptitle("Spherical Coordinate Distributions Across Box Sizes", fontsize=16)

    if save:
        fig.savefig(f"{save_dir}/combined_spherical_distributions.png", dpi=300)

    plt.show()
    plt.close('all')
       

plot_spherical_kde_nmols(spherical_coords_data_dict, spherical_box_sizes_array, n_mols, cutoff=0, save=True, save_dir="plots_K_"+f"{K}")


#%%

# now realisation average 

#mean_shear_log_data_array=np.mean(shear_log_data_array,axis=0)
mean_eq_log_data_array=np.mean(eq_log_data_array,axis=0)

#print(mean_shear_log_data_array.shape)
print(mean_eq_log_data_array.shape)
            

def plot_time_series_n_mol_converge(data, n_mols, column_names, use_latex=True, save=False, save_dir="plots"):
    """
    Plots time series data for each column, showing all timesteps on the same graph.
    Adds mean, std deviation (over last 60%), and gradient stats to legend and stores them in an array.

    Returns:
        stats_array (ndarray): shape (n_cols, n_timestep, 4) → [mean, std, mean_grad, std_grad] for each timestep and column
    """

    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    n_timestep, n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")

    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    stats_array = np.zeros((n_cols, n_timestep, 4))

    for col in range(n_cols):
        plt.figure(figsize=(10, 5))

        for i in range(n_timestep):
            y = data[i, :, col]
            number_of_steps=np.linspace(0,y.shape[0]*800000,y.shape[0])
            

            # Last 60% of the signal
            last_60_percent = y[int(0.2 * len(y)):]

            # Compute mean and std
            mean = np.mean(last_60_percent)
            std = np.std(last_60_percent)

            # Compute gradient
            gradients = np.gradient(last_60_percent)

            mean_grad = np.mean(gradients)
            std_grad = np.std(gradients)

            # Store stats
            stats_array[col, i, 0] = mean
            stats_array[col, i, 1] = std
            stats_array[col, i, 2] = mean_grad
            stats_array[col, i, 3] = std_grad

            # Plot
            plt.plot(number_of_steps,y, label=rf"$N_{{mol}}= {n_mols[i]}$", linewidth=1.5)

       # plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("$\Delta t$")
        plt.ylabel(rf"\textbf{{{column_names[col]}}}")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        
        save_string=column_names[col].replace(' ', '_')
        save_string=save_string.replace('$', '')
        save_string=save_string.replace('\\', '')

        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}.png"
            plt.savefig(fname, dpi=300)

        plt.show()

    return stats_array



def plot_stats_vs_n_mols(stats_array, n_mols, column_names, use_latex=True, gradient_threshold=1e-2, save=False, save_dir="plots"):
    """
    Plots stress mean and gradient mean vs timestep with std as error bars using twin y-axes.
    Highlights convergence points with high-contrast markers and saves plots if requested.
    """

    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    n_cols = stats_array.shape[0]

    for col in range(n_cols):
        means = stats_array[col, :, 0]
        stds = stats_array[col, :, 1]

        grad_means = stats_array[col, :, 2]
        grad_stds = stats_array[col, :, 3]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Plot stress mean ± std
        ax1.errorbar(n_mols, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, color='tab:blue')
        ax1.set_xlabel(r"mol count")
        ax1.set_ylabel(r"Steady State Mean", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot gradient mean ± std on twin axis
        ax2 = ax1.twinx()
        ax2.errorbar(n_mols, grad_means, yerr=grad_stds, fmt='s--', capsize=4, linewidth=2, color='black', markersize=5)
        ax2.set_ylabel(r"Gradient Mean", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Highlight converged points (updated color for clarity)
        converged = np.abs(grad_means) < gradient_threshold
        ax2.plot(np.array(n_mols)[converged], grad_means[converged], 'o', markersize=10,
                 markerfacecolor='tab:red', markeredgecolor='none', markeredgewidth=0, label='Converged (|grad| < tol)')

        # Clean manual legend
        handles = [
            plt.Line2D([], [], color='tab:blue', marker='o', linestyle='-', linewidth=2, label="Steady State Mean ± Std"),
            plt.Line2D([], [], color='black', marker='s', linestyle='--', linewidth=2, label="Gradient Mean ± Std"),
            plt.Line2D([], [], color='tab:red', marker='o', markeredgecolor='none', linestyle='None', markersize=10, label="Converged ($|\\mathrm{grad}| < \\mathrm{tol}$)")
        ]

        ax1.legend(handles=handles, loc='upper right', fontsize=11, frameon=False, bbox_to_anchor=(1, 1))

        # Title and layout
        plt.title(rf"\textbf{{{column_names[col]}}} ")
        fig.tight_layout()

        # Save if requested
        save_string=column_names[col].replace(' ', '_')
        save_string=save_string.replace('$', '')
        save_string=save_string.replace('\\', '')

        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}_stats.png"
            plt.savefig(fname, dpi=300)

        plt.show()


# plot_time_series(mean_shear_log_data_array, erate,shear_columns)

plot_time_series_n_mol_converge(mean_eq_log_data_array, n_mols,eq_columns)


#%%
xlabel=r"$N_{mol}$"
stats_array=plot_time_series_n_mol_converge(mean_stress_array,n_mols,stress_columns,save=True, save_dir="plots_K_"+str(K))
plot_stats_vs_indepvar_log_file(stats_array, n_mols,xlabel, stress_columns, use_latex=True, gradient_threshold=1e-2,save=True, save_dir="plots_K_"+str(K))
# changing eq colums for plotting 
eq_columns=['Step',
 '$E_{K}$',
 '$E_{P}$',
 'Press',
 '$T$',
 '$E_{t}$',
 'Econserve',
 'c_VACF[4]']
stats_array=plot_time_series_n_mol_converge(mean_eq_log_data_array, n_mols,eq_columns,save=True, save_dir="plots_K_"+str(K))
#plot_stats_vs_n_mols(stats_array, n_mols,eq_columns,save=True, save_dir="plots_K_"+str(K))


plot_stats_vs_indepvar_log_file(stats_array, n_mols,xlabel,eq_columns, use_latex=True, gradient_threshold=1e-2,save=True, save_dir="plots_K_"+str(K))
# %%
labels_stress = np.array(
    [
        "\sigma_{xx}$",
        "\sigma_{yy}$",
        "\sigma_{zz}$",
        "\sigma_{xz}$",
        "\sigma_{xy}$",
        "\sigma_{yz}$",
    ])
truncate=700
time_mean_stress=np.mean(mean_stress_array[:,truncate:,:],axis=1)
time_std_stress=np.std(mean_stress_array[:,truncate:,:],axis=1)
# %%
import matplotlib.pyplot as plt
for i in range(1,4):
    plt.errorbar(Wi,time_mean_stress[:,i],yerr=time_std_stress[:,i], label=stress_columns[i])

plt.xlabel("$Wi$")
plt.legend()
plt.show()

for i in range(4,7):
    plt.plot(Wi,time_mean_stress[:,i], label=stress_columns[i])

plt.legend()
plt.xlabel("$Wi$")
plt.show()

for i in range(7,9):
    plt.plot(Wi,time_mean_stress[:,i], label=stress_columns[i])

plt.legend()
plt.xlabel("$Wi$")
plt.show()
# %%
def plot_stress_components(
    Wi,
    time_mean_stress,
    time_std_stress,
    stress_columns,
    i_range=(1, 4),
    fit=False,
    fit_index=None
):
    """
    Plots stress components with error bars, with optional quadratic fit.

    Parameters:
        Wi (array): Weissenberg numbers (x-axis)
        time_mean_stress (2D array): shape (N, M), stress means
        time_std_stress (2D array): shape (N, M), stress stds
        stress_columns (list): list of column names for the stress components
        i_range (tuple): (start, end) indices of columns to plot [start, end)
        fit (bool): whether to show a quadratic fit
        fit_index (int or None): which column index to fit, must be in i_range if fit is True
    """

    # LaTeX-style plot settings
    plt.rcParams.update({
        "text.usetex": "True",
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    for i in range(*i_range):
        ax.errorbar(
            Wi,
            time_mean_stress[:, i],
            yerr=time_std_stress[:, i],
            label=rf"{stress_columns[i]}",
            capsize=3,
            marker='o',
            linestyle="none",
            linewidth=1.5
        )

        # Optional: add ax^2 fit
        if fit and i == fit_index:
            x = np.array(Wi)
            y = time_mean_stress[:, i]
            a, _, _, _ = np.linalg.lstsq(x[:, np.newaxis]**2, y, rcond=None)
            fit_y = a[0] * x**2

            ax.plot(
                x,
                fit_y,
                '--',
                color='black',
                linewidth=2,
                label=rf"Fit: {stress_columns[i]} $= {a[0]:.3g} \cdot Wi^2$"
            )

    ax.set_xlabel(r"Wi")
    #ax.set_ylabel(r"Stress")
    ax.legend()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


plot_stress_components(Wi, time_mean_stress, time_std_stress, stress_columns, i_range=(7,9), fit=True, fit_index=7)

plot_stress_components(Wi, time_mean_stress, time_std_stress, stress_columns, i_range=(1,4), fit=False, fit_index=7)

plot_stress_components(Wi, time_mean_stress, time_std_stress, stress_columns, i_range=(4,7), fit=False, fit_index=7)
# %%
