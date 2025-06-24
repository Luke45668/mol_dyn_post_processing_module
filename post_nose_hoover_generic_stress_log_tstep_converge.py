# %% new section where stress and log files are used for intial run 
import os
import re
import glob
import pandas as pd
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
# === Setup
path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_mass_10_stiff_0.005_1_1_sllod_100_strain_T_0.01_R_1_R_n_1_N_864/logs_and_stress/"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_tstep_0.0005_mass_10_stiff_0.005_1_1_sllod_25_strain_T_0.01_R_1_R_n_1_N_500/logs_and_stress/"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/langevin_runs/tstep_converge/DB_tstep_converge_test_run_tsteprange_0.0001_1e-06_mass_1_stiff_0.25_2.0_1_sllod_strain_100_T_1_R_0.1_R_n_1_nmols_1688_L_150"

vol=150**3
eq_outs=801
os.chdir(path_2_files)
K = 0.5
mass=1
n_shear_points=30
log_name_list = glob.glob("log*_K_"+str(K))
erate=np.array([0.1])
timestep=np.round(np.logspace(-4,-6,8),7)
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

data=read_lammps_log_if_complete(log_name_list[0])
eq_columns=list(data[0].columns)


real_target = 3
tstep_count = np.zeros(timestep.size, dtype=int)
eq_outs=1000
# Preallocate data arrays
eq_log_data_array = np.zeros((real_target, timestep.size, eq_outs, 8))


for file in log_name_list:

    data = read_lammps_log_if_complete(file)

    if data is None:
        continue

    # Extract shear rate from filename
    file_meta_data = file.split("_")
    print(file_meta_data)
    tstep_file = round(float(file_meta_data[15]), 7)
    tstep_index = int(np.where(timestep == tstep_file)[0])

    # Check if real_target already reached
    if tstep_count[tstep_index] >= real_target:
        continue

    # Assign realisation index (zero-based)
    real_index = tstep_count[tstep_index]

    # Extract thermo outputs as numpy arrays
    eq_log_data_array_raw = data[0].to_numpy()
   
    print(eq_log_data_array_raw.shape)
   

    # Store data
    eq_log_data_array[real_index, tstep_index] = eq_log_data_array_raw[:1000]
   

    # Increment count
    tstep_count[tstep_index] += 1

print(tstep_count)

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
vel_pos_dump_name_list=glob.glob("*_hookean_flat_elastic_mass_*K_"+str(K)+"*.dump")
data_dict = read_stress_tensor_file(filename=stress_name_list[0], volume=vol, return_data=True)
stress_columns = list(data_dict.keys())
output_cutoff=1000
real_target = 3
tstep_count = np.zeros(timestep.size, dtype=int)
stress_array = np.zeros((real_target, timestep.size, output_cutoff, 9))

#%%
for file in stress_name_list:
    data_dict = read_stress_tensor_file(filename=file, volume=vol, return_data=True)
    
    if data_dict is None:
        continue

    if data_dict["time"].size <output_cutoff:
        continue

    # Extract metadata
    file_meta_data = file.split("_")
    print(file_meta_data)
    #DB
    # tstep_file = round(float(file_meta_data[12]), 9)
    # timestep_index = int(np.where(timestep == tstep_file)[0])
    # real_index = int(file_meta_data[9]) - 1  # zero-based indexing
    #plate
    tstep_file = round(float(file_meta_data[12]), 7)
    timestep_index = int(np.where(timestep == tstep_file)[0])
    real_index = int(file_meta_data[9])  

    if real_index >= real_target:
        continue  # skip if real_index exceeds target

    tstep_count[timestep_index] += 1
    print(timestep[timestep_index])
    print(f"Realisation: {real_index}")

    # Fill stress array
    for column, key in enumerate(stress_columns):
        raw_stress_array = data_dict[key][:output_cutoff+1]
        stress_array[real_index, timestep_index, :, column] = raw_stress_array

# Compute mean
mean_stress_array = np.mean(stress_array, axis=0)

print("Mean stress array shape:", mean_stress_array.shape)
print(tstep_count)

#%% processing dump files
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

def convert_cart_2_spherical_z_inc_plate_from_dump(vel_pos_array,n_mols,output_cutoff):      
        position_array=vel_pos_array[:,:,:3]
        print(position_array.shape)
        #reshape into number of plates

        position_plates_array=np.reshape(position_array,(output_cutoff,n_mols,3,3))

        ell_1=position_plates_array[:,:,1]-position_plates_array[:,:,0]
        ell_2=position_plates_array[:,:,2]-position_plates_array[:,:,0]

        print("ell_1 shape",ell_1.shape)
        print("ell_2 shape",ell_2.shape)

        area_vector=np.cross(ell_1,ell_2,axis=-1)
        print(area_vector.shape)



        area_vector[area_vector[:, :, 2] < 0] *= -1

        x = area_vector[:, :, 0]
        y = area_vector[:, :, 1]
        z = area_vector[:, :, 2]

        spherical_coords_array = np.zeros(
            ( output_cutoff,n_mols, 3)
        )

        # radial coord
        spherical_coords_array[:, :, 0] = np.sqrt((x**2) + (y**2) + (z**2))

        #  theta coord
        spherical_coords_array[:, :, 1] = np.sign(y) * np.arccos(
            x / (np.sqrt((x**2) + (y**2)))
        )

        # spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
        # spherical_coords_array[:,:,:,1]=np.arctan(y/x)

        # phi coord
        # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
        spherical_coords_array[:, :, 2] = np.arccos(
            z / np.sqrt((x**2) + (y**2) + (z**2))
        )

        return spherical_coords_array

n_mols=1688
tstep_count = np.zeros(timestep.size, dtype=int)
vel_pos_array = np.zeros((real_target, timestep.size, output_cutoff, n_mols*3,6 ))
area_vector_array = np.zeros((real_target, timestep.size, output_cutoff, n_mols,3 ))

for file in vel_pos_dump_name_list:
    data=read_lammps_posvel_dump_to_numpy(file)
    if data is None:
        continue
    print(data.shape[0])
    if data.shape[0] <output_cutoff:
        continue
    
    # Extract metadata
    file_meta_data = file.split("_")
    print(file_meta_data)

   
    #plate
    tstep_file = round(float(file_meta_data[17]), 9)
    timestep_index = int(np.where(timestep == tstep_file)[0])
    print(timestep_index)
    real_index = int(file_meta_data[14])  
    print(real_index)
    

    # if real_index >= real_target:
    #     continue  # skip if real_index exceeds target

    tstep_count[timestep_index] += 1
    print(tstep_count)

    vel_pos_array[real_index,timestep_index]=data[:output_cutoff,:,2:]
    area_vector_array[real_index,timestep_index]=convert_cart_2_spherical_z_inc_plate_from_dump(vel_pos_array[real_index,timestep_index],n_mols,output_cutoff)

print(tstep_count)

#%%
def plot_spherical_kde_plate_from_numpy( area_vector_array, timestep,cutoff, save=False, save_dir="plots", use_latex=True):
    """
    KDE plots of spherical coordinate data (rho, theta, phi) for each box size.
    Each box plotted as subplots. Fully compatible version for interactive environments.
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
    pi_theta_labels =[r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    pi_phi_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
    pi_phi_labels = [r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"]

    if save:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(timestep.size):
        box_label = f"{timestep[i]}"

        # Prepare data
        rho = np.ravel( area_vector_array[:,i,cutoff:, :, 0])
        theta = area_vector_array[:,i,cutoff:,:, 1]
        theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
        phi = area_vector_array[:,i, cutoff:, :,2]
        phi = np.ravel(np.array([phi, np.pi - phi]))

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # --- RHO ---
        sns.kdeplot(rho, color="tab:blue", linewidth=2, ax=axes[0])
        axes[0].set_xlabel(r"$\rho$")
        axes[0].set_ylabel("Density")
        axes[0].set_title(r"$\rho$ Distribution")
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # --- THETA ---
        sns.kdeplot(theta, color="tab:green", linewidth=2, ax=axes[1])
        axes[1].set_xlabel(r"$\Theta$")
        axes[1].set_ylabel("Density")
        axes[1].set_xticks(pi_theta_ticks)
        axes[1].set_xticklabels(pi_theta_labels)
        axes[1].set_xlim(-np.pi, np.pi)
        #axes[1].legend([box_label], loc="upper right", frameon=False)
        axes[1].set_title(r"$\Theta$ Distribution")
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # --- PHI ---
        sns.kdeplot(phi, color="tab:red", linewidth=2, ax=axes[2])
        axes[2].set_xlabel(r"$\phi$")
        axes[2].set_ylabel("Density")
        axes[2].set_xticks(pi_phi_ticks)
        axes[2].set_xticklabels(pi_phi_labels)
        axes[2].set_xlim(0, np.pi / 2)
        #axes[2].legend([box_label], loc="upper right", frameon=False)
        axes[2].set_title(r"$\phi$ Distribution")
        axes[2].grid(True, linestyle='--', alpha=0.7)

        # Adjust layout safely
        fig.subplots_adjust(hspace=0.4, top=0.9)
        fig.suptitle(f"Timestep {box_label} - Spherical Distributions", fontsize=16)

        # Save if requested
        if save:
            fig.savefig(f"{save_dir}/{box_label}_spherical_distributions.png", dpi=300)

        # Show
        plt.show()

        # Close
        plt.close('all')

plot_spherical_kde_plate_from_numpy( area_vector_array, timestep,600, save=False, save_dir="plots", use_latex=True)

#%%
def plot_spherical_kde_plate_selected(area_vector_array, timestep, selected_timesteps, save=False, save_dir="plots", use_latex=True):
    """
    KDE plots of spherical coordinate data (rho, theta, phi) comparing selected timesteps on the same plot.
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
    pi_theta_labels =[r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    pi_phi_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
    pi_phi_labels = [r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"]

    if save:
        os.makedirs(save_dir, exist_ok=True)

    # Create subplots
    

    colors = sns.color_palette("Set1", len(selected_timesteps))

    
    for i in range(timestep.size):
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        for j in range(len(selected_timesteps)):
        # Prepare data
            rho = np.ravel(area_vector_array[:,i, selected_timesteps[j], :, 0])
            theta = area_vector_array[:, i, selected_timesteps[j], :, 1]
            theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
            phi = area_vector_array[:, i, selected_timesteps[j], :, 2]
            phi = np.ravel(np.array([phi, np.pi - phi]))

            # --- RHO ---
            sns.kdeplot(rho, color=colors[j], linewidth=2, ax=axes[0], label=f"output {selected_timesteps[j]}")

            # --- THETA ---
            sns.kdeplot(theta, color=colors[j], linewidth=2, ax=axes[1], label=f"output {selected_timesteps[j]}")

            # --- PHI ---
            sns.kdeplot(phi, color=colors[j], linewidth=2, ax=axes[2], label=f"output {selected_timesteps[j]}")

        # Rho plot
        axes[0].set_xlabel(r"$\rho$")
        axes[0].set_ylabel("Density")
        axes[0].set_title(r"$\rho$ Distribution")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Theta plot
        axes[1].set_xlabel(r"$\Theta$")
        axes[1].set_ylabel("Density")
        axes[1].set_xticks(pi_theta_ticks)
        axes[1].set_xticklabels(pi_theta_labels)
        axes[1].set_xlim(-np.pi, np.pi)
        axes[1].set_title(r"$\Theta$ Distribution")
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Phi plot
        axes[2].set_xlabel(r"$\phi$")
        axes[2].set_ylabel("Density")
        axes[2].set_xticks(pi_phi_ticks)
        axes[2].set_xticklabels(pi_phi_labels)
        axes[2].set_xlim(0, np.pi / 2)
        axes[2].set_title(r"$\phi$ Distribution")
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.7)

        # Adjust and save/show
        fig.subplots_adjust(hspace=0.4, top=0.9)
        fig.subplots_adjust(hspace=0.4, top=0.9)
        fig.suptitle(f"Timestep {timestep[i]} - Spherical Distributions", fontsize=16)

        if save:
            fig.savefig(f"{save_dir}/selected_timesteps_spherical_distributions.png", dpi=300)

        plt.show()
        plt.close('all')

plot_spherical_kde_plate_selected(area_vector_array, timestep, selected_timesteps=[100,200, 500,600,700])

#%%

# now realisation average 

#mean_shear_log_data_array=np.mean(shear_log_data_array,axis=0)
mean_eq_log_data_array=np.mean(eq_log_data_array,axis=0)

# #print(mean_shear_log_data_array.shape)
# print(mean_eq_log_data_array.shape)
            

def plot_time_series_tstep_converge(data, timestep, column_names, use_latex=True, save=False, save_dir="plots"):
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
            number_of_steps=np.linspace(0,(1e-5*1e8),y.shape[0])
            

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
            plt.plot(number_of_steps,y, label=rf"Timestep ${timestep[i]:.7f}$", linewidth=1.5)

        #plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("$t/\\tau$")
        plt.ylabel(rf"\textbf{{{column_names[col]}}}")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        save_string = column_names[col].replace(' ', '_').replace('$', '').replace('\\', '')
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}.png"
            plt.savefig(fname, dpi=300)

        plt.show()

    return stats_array

def plot_stats_vs_timestep(stats_array, timestep, column_names, use_latex=True, gradient_threshold=1e-2, save=False, save_dir="plots"):
    """
    Plots Steady State Mean and gradient mean vs timestep with std as error bars using twin y-axes.
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

        # Plot Steady State Mean ± std
        ax1.errorbar(timestep, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, color='tab:blue', zorder=2)
        ax1.set_xlabel(r"Timestep")
        ax1.set_ylabel(r"Steady State Mean", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xscale('log')
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot gradient mean ± std on twin axis
        ax2 = ax1.twinx()
        ax2.errorbar(timestep, grad_means, yerr=grad_stds, fmt='s--', capsize=4, linewidth=2, color='black', markersize=5, zorder=2)
        ax2.set_ylabel(r"Gradient Mean", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Highlight converged points (high contrast color + edge)
        converged = np.abs(grad_means) < gradient_threshold
        ax2.plot(np.array(timestep)[converged], grad_means[converged], 'o', markersize=10,
                 markerfacecolor='red', markeredgecolor='none', markeredgewidth=1.5,
                 label='Converged (|grad| < tol)')

        # Clean manual legend
        handles = [
            plt.Line2D([], [], color='tab:blue', marker='o', linestyle='-', linewidth=2, label="Steady State Mean ± Std"),
            plt.Line2D([], [], color='black', marker='s', linestyle='--', linewidth=2, label="Gradient Mean ± Std"),
            plt.Line2D([], [], color='red', marker='o', markeredgecolor='none', linestyle='None', markersize=10, label="Converged ($|\\mathrm{grad}| < \\mathrm{tol}$)")
        ]

        # Create legend and ensure it's in front
        ax2.legend(
            handles=handles,
            loc='upper right',
            fontsize=11,
            bbox_to_anchor=(1,1),
            facecolor='white',
            edgecolor='black'
        )
        

        # Title and layout
        plt.title(rf"\textbf{{{column_names[col]}}}")
        fig.tight_layout()

        # Clean filename
        save_string = column_names[col].replace(' ', '_').replace('$', '').replace('\\', '')

        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}_stats.png"
            plt.savefig(fname, dpi=300)

        plt.show()

#%%

# plot_time_series(mean_shear_log_data_array, erate,shear_columns)



stats_array=plot_time_series_tstep_converge(mean_stress_array,timestep,stress_columns, use_latex=True, save=True, save_dir="plots_K_"+str(K))
plot_stats_vs_timestep(stats_array, timestep, stress_columns,save=True, save_dir="plots_K_"+str(K))

eq_columns=['Step',
 '$E_{K}$',
 '$E_{P}$',
 'Press',
 '$T$',
 '$E_{t}$',
 'Econserve',
 'c_VACF[4]']

stats_array=plot_time_series_tstep_converge(mean_eq_log_data_array, timestep,eq_columns,save=True, save_dir="plots_K_"+str(K))
plot_stats_vs_timestep(stats_array, timestep,eq_columns,save=True, save_dir="plots_K_"+str(K))



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
