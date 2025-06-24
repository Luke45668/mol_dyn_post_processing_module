# %% new section where stress and log files are used for intial run 
import os
import re
import glob
import pandas as pd
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
# === Setup

path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/langevin_runs/n_mol_converge/DB_nmols_converge_test_run_n_mols_range_63_5696_tstep_1e-05__mass_1_stiff_0.25_2.0_1_strain_100_T_1_R_0.1_R_n_1"
#path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/n_mols_converge/"
os.chdir(path_2_files)
mol_density=13500/(300**3)
box_size_bar=np.array([50,75,100,125,150,175,200,225]).astype('int')

eq_outs=1001
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

data=read_lammps_log_if_complete(log_name_list[0])
eq_columns=list(data[0].columns)


real_target = 3
box_size_count = np.zeros(box_size_bar.size, dtype=int)

# Preallocate data arrays
eq_log_data_array = np.zeros((real_target, box_size_bar.size, eq_outs, 8))

for file in log_name_list:

    data = read_lammps_log_if_complete(file)

    if data is None:
        continue

    # Extract shear rate from filename
    file_meta_data = file.split("_")
    print(file_meta_data)
    box_size_file = round(float(file_meta_data[13]), 7)
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
    eq_log_data_array[real_index, box_size_index] = eq_log_data_array_raw
   

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
output_cutoff=1000
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
    box_side=int(file_meta_data[10])
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
    real_index = int(file_meta_data[9])   # zero-based indexing
    print(real_index)

    if real_index >= real_target:
        continue  # skip if real_index exceeds target
    
    n_mol_count[box_index] += 1
    # print(timestep[timestep_index])
    print(f"Realisation: {real_index}")

    # # Fill stress array
    for column, key in enumerate(stress_columns):
         raw_stress_array = data_dict[key][:output_cutoff+1]
         stress_array[real_index, box_index, :, column] = raw_stress_array

# Compute mean
mean_stress_array = np.mean(stress_array, axis=0)
print(n_mol_count)

print("Mean stress array shape:", mean_stress_array.shape)

#%% spring orientation data
spring_name_list=glob.glob("*tensor*K_"+str(K)+".dump")
def read_lammps_dump_tensor(filename):
    """
    Reads LAMMPS dump style file with tensor entries (can handle incomplete files).
    
    Returns:
        dump_data (list of dict): Each dict has timestep, number of entries, box bounds, DataFrame of entries
    """

    dump_data = []
    current_data = None

    with open(filename, "r", errors="ignore") as f:
        for line in f:
            stripped = line.strip()

            # TIMESTEP
            if stripped.startswith("ITEM: TIMESTEP"):
                if current_data is not None:
                    dump_data.append(current_data)
                current_data = {"timestep": None, "n_entries": None, "box_bounds": [], "columns": None, "data": []}
                current_data["timestep"] = int(next(f).strip())
                continue

            # NUMBER OF ENTRIES
            if stripped.startswith("ITEM: NUMBER OF ENTRIES"):
                current_data["n_entries"] = int(next(f).strip())
                continue

            # BOX BOUNDS
            if stripped.startswith("ITEM: BOX BOUNDS"):
                for _ in range(3):
                    current_data["box_bounds"].append(next(f).strip())
                continue

            # ENTRIES HEADER
            if stripped.startswith("ITEM: ENTRIES"):
                current_data["columns"] = stripped.replace("ITEM: ENTRIES", "").split()
                continue

            # DATA rows
            if current_data and current_data["columns"]:
                parts = stripped.split()

                # Skip incomplete rows
                if len(parts) != len(current_data["columns"]):
                    continue

                try:
                    current_data["data"].append([float(x) for x in parts])
                except ValueError:
                    continue  # Skip malformed rows

    # Save last block
    if current_data is not None:
        dump_data.append(current_data)

    # Convert data to pandas DataFrame for each block
    for block in dump_data:
        block["data"] = pd.DataFrame(block["data"], columns=block["columns"])

    return dump_data
def convert_cart_2_spherical_z_inc_DB_from_dict(spring_vector_ray,n_mols,i
   
):
        
        spring_vector_ray[spring_vector_ray[ :, 2] < 0] *= -1

        x = spring_vector_ray[ :, 0]
        y = spring_vector_ray[ :, 1]
        z = spring_vector_ray[ :, 2]

        spherical_coords_array = np.zeros(
            ( n_mols[i], 3)
        )

        # radial coord
        spherical_coords_array[ :, 0] = np.sqrt((x**2) + (y**2) + (z**2))

        #  theta coord
        spherical_coords_array[ :, 1] = np.sign(y) * np.arccos(
            x / (np.sqrt((x**2) + (y**2)))
        )

        # spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
        # spherical_coords_array[:,:,:,1]=np.arctan(y/x)

        # phi coord
        # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
        spherical_coords_array[ :, 2] = np.arccos(
            z / np.sqrt((x**2) + (y**2) + (z**2))
        )

        return spherical_coords_array


dump_data = read_lammps_dump_tensor(spring_name_list[0])

# creating list of arrays to contain the dumps 
box_sizes_list_array=[]
spherical_box_sizes_array=[]
for i in range(box_size_bar.size):
    array=np.zeros((real_target,1000,n_mols[i],3))
    box_sizes_list_array.append(array)
    spherical_box_sizes_array.append(array)

#%%

# creating dict to store the list in 
#spring_data_dict={'box_sizes':box_sizes_list_array}
spherical_coords_data_dict={'box_sizes':spherical_box_sizes_array}


for file in spring_name_list:
    file_meta_data = file.split("_")
    print(file_meta_data)
    
    # DB
    real_index = int(file_meta_data[7])   # zero-based indexing
    print(real_index)
    box_side=int(file_meta_data[8])
    print(box_side)
    box_index=np.where(box_side==box_size_bar)[0][0]
    print(box_index)

    dump_data = read_lammps_dump_tensor(file)

    for i in range(1000):

        dump_data_np_array=dump_data[i]['data'].to_numpy()
        spherical_np_array=convert_cart_2_spherical_z_inc_DB_from_dict(dump_data_np_array,n_mols,box_index)
        #spring_data_dict["box_sizes"][box_index][real_index,i]=dump_data_np_array
        spherical_coords_data_dict["box_sizes"][box_index][real_index,i]=spherical_np_array

#%% plotting distributions 
import seaborn as sns 
cutoff=400
pi_theta_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
pi_theta_tick_labels = ["-π", "-π/2", "0", "π/2", "π"]
pi_phi_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
pi_phi_tick_labels = ["0", "π/8", "π/4", "3π/8", "π/2"]


for i in range(len(spherical_box_sizes_array)):

    rho= np.ravel(spherical_coords_data_dict["box_sizes"][i][:,cutoff:,:,0])
    theta= spherical_coords_data_dict["box_sizes"][i][:,cutoff:,:,1]
    
    theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
    phi= spherical_coords_data_dict["box_sizes"][i][:,cutoff:,:,2]
    phi=np.ravel( np.array([phi, np.pi - phi]))


    # sns.kdeplot(rho)
    # plt.xlabel("$\\rho$")
    # plt.ylabel("Density")
    # plt.show()


    sns.kdeplot(theta)
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
    plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)
    plt.ylabel("Density")
    plt.xlim(-np.pi, np.pi)
    plt.show()

    # sns.kdeplot(phi)
    # plt.xlabel("$\phi$")
    # plt.xticks(pi_phi_ticks, pi_phi_tick_labels)
    # plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)
    # plt.ylabel("Density")
    # plt.xlim(0, np.pi / 2)
    # plt.show()
    
#%%
import seaborn as sns 
def plot_spherical_kde_nmols(spherical_coords_data_dict, spherical_box_sizes_array, n_mols, cutoff=400, save=False, save_dir="plots", use_latex=True):
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

    for i in range(len(spherical_box_sizes_array)):
        box_label = f"{n_mols[i]}"

        # Prepare data
        rho = np.ravel(spherical_coords_data_dict["box_sizes"][i][:, cutoff:, :, 0])
        theta = spherical_coords_data_dict["box_sizes"][i][:, cutoff:, :, 1]
        theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
        phi = spherical_coords_data_dict["box_sizes"][i][:, cutoff:, :, 2]
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
        fig.suptitle(f"Box {box_label} - Spherical Distributions", fontsize=16)

        # Save if requested
        if save:
            fig.savefig(f"{save_dir}/{box_label}_spherical_distributions.png", dpi=300)

        # Show
        plt.show()

        # Close
        plt.close('all')
       

plot_spherical_kde_nmols(spherical_coords_data_dict, spherical_box_sizes_array, n_mols, cutoff=400, save=False, save_dir="spherical_plots")


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

stats_array=plot_time_series_n_mol_converge(mean_stress_array,n_mols,stress_columns,save=True, save_dir="plots_K_"+str(K))
plot_stats_vs_n_mols(stats_array, n_mols, stress_columns,save=True, save_dir="plots_K_"+str(K))
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
plot_stats_vs_n_mols(stats_array, n_mols,eq_columns,save=True, save_dir="plots_K_"+str(K))

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
