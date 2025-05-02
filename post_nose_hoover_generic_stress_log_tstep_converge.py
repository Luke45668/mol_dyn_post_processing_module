# %% new section where stress and log files are used for intial run 
import os
import re
import glob
import pandas as pd
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
# === Setup
path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_mass_10_stiff_0.005_1_1_sllod_100_strain_T_0.01_R_1_R_n_1_N_864/logs_and_stress/"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_tstep_0.0005_mass_10_stiff_0.005_1_1_sllod_25_strain_T_0.01_R_1_R_n_1_N_500/logs_and_stress/"
vol=100**3
eq_outs=1001
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/tstep_converge"
vol=300**3
eq_outs=801
timestep=np.round(np.logspace(-2.3010299956639813,-6.301029995663981,8),9)

os.chdir(path_2_files)
K = 0.1
mass=10
n_shear_points=30
log_name_list = glob.glob("log*_K_"+str(K))

erate=np.logspace(-2.5, -1,n_shear_points)
erate=np.round(erate,7)
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
shear_columns=list(data[1].columns)

real_target = 3
erate_count = np.zeros(erate.size, dtype=int)

# Preallocate data arrays
eq_log_data_array = np.zeros((real_target, erate.size, eq_outs, 7))
shear_log_data_array = np.zeros((real_target, erate.size, 1000, 11))

for file in log_name_list:

    data = read_lammps_log_if_complete(file)

    if data is None:
        continue

    # Extract shear rate from filename
    file_meta_data = file.split("_")
    print(file_meta_data)
    erate_file = round(float(file_meta_data[21]), 7)
    erate_index = int(np.where(erate == erate_file)[0])

    # Check if real_target already reached
    if erate_count[erate_index] >= real_target:
        continue

    # Assign realisation index (zero-based)
    real_index = erate_count[erate_index]

    # Extract thermo outputs as numpy arrays
    eq_log_data_array_raw = data[0].to_numpy()
    shear_log_data_array_raw = data[1].to_numpy()

    print(eq_log_data_array_raw.shape)
    print(shear_log_data_array_raw.shape)

    # Store data
    eq_log_data_array[real_index, erate_index] = eq_log_data_array_raw
    shear_log_data_array[real_index, erate_index] = shear_log_data_array_raw[:1000]

    # Increment count
    erate_count[erate_index] += 1

print(erate_count)
print(shear_log_data_array.shape)
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
    sxx = -sxx_sum / volume
    syy = -syy_sum / volume
    szz = -szz_sum / volume
    sxy = -sxy_sum / volume
    sxz = -sxz_sum / volume
    syz = -syz_sum / volume

    N1 = sxx - szz
    N2 = szz - syy

    if return_data:
        return {
            'time': time,
            '$\sigma_{xx}$': sxx, '$\sigma_{yy}$': syy, '$\sigma_{zz}$': szz,
            '$\sigma_{xy}$': sxy, '$\sigma_{xz}$': sxz, '$\sigma_{yz}$': syz,
            '$N_{1}$': N1, '$N_{2}$': N2
        }
    
stress_name_list=glob.glob("eq_stress*K_"+str(K)+"*.dat")
data_dict = read_stress_tensor_file(filename=stress_name_list[0], volume=vol, return_data=True)
stress_columns = list(data_dict.keys())
output_cutoff=60
real_target = 3
tstep_count = np.zeros(erate.size, dtype=int)
stress_array = np.zeros((real_target, timestep.size, 61, 9))

#%%
for file in stress_name_list:
    data_dict = read_stress_tensor_file(filename=file, volume=vol, return_data=True)
    
    if data_dict is None:
        continue

    if data_dict["time"].size <62:
        continue

    # Extract metadata
    file_meta_data = file.split("_")
    print(file_meta_data)

    tstep_file = round(float(file_meta_data[12]), 9)
    timestep_index = int(np.where(timestep == tstep_file)[0])

    real_index = int(file_meta_data[9]) - 1  # zero-based indexing

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


#%%

# now realisation average 

#mean_shear_log_data_array=np.mean(shear_log_data_array,axis=0)
# mean_eq_log_data_array=np.mean(eq_log_data_array,axis=0)

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

            # Last 60% of the signal
            last_60_percent = y[int(0.4 * len(y)):]

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
            plt.plot(y, label=rf"Timestep ${timestep[i]:.7f}$", linewidth=1.5)

        plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("Output Step")
        plt.ylabel(rf"\textbf{{{column_names[col]}}}")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{column_names[col].replace(' ', '_')}.png"
            plt.savefig(fname, dpi=300)

        plt.show()

    return stats_array



def plot_stats_vs_timestep(stats_array, timestep, column_names, use_latex=True, gradient_threshold=1e-7, save=False, save_dir="plots"):
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
        ax1.errorbar(timestep, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, color='tab:blue')
        ax1.set_xlabel(r"Timestep")
        ax1.set_ylabel(r"Stress Mean", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xscale('log')
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot gradient mean ± std on twin axis
        ax2 = ax1.twinx()
        ax2.errorbar(timestep, grad_means, yerr=grad_stds, fmt='s--', capsize=4, linewidth=2, color='black', markersize=5)
        ax2.set_ylabel(r"Gradient Mean", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Highlight converged points (high contrast color + edge)
        converged = np.abs(grad_means) < gradient_threshold
        ax2.plot(np.array(timestep)[converged], grad_means[converged], 'o', markersize=12,
                 markerfacecolor='gold', markeredgecolor='black', markeredgewidth=1.5, label='Converged (|grad| < tol)')

        # Clean manual legend
        handles = [
            plt.Line2D([], [], color='tab:blue', marker='o', linestyle='-', linewidth=2, label="Stress Mean ± Std"),
            plt.Line2D([], [], color='black', marker='s', linestyle='--', linewidth=2, label="Gradient Mean ± Std"),
            plt.Line2D([], [], color='gold', marker='o', markeredgecolor='black', linestyle='None', markersize=10, label="Converged ($|\\mathrm{grad}| < \\mathrm{tol}$)")
        ]

        ax1.legend(handles=handles, loc='best', fontsize=11, frameon=False,bbox_to_anchor=(1,1))

        # Title and layout
        plt.title(rf"\textbf{{{column_names[col]}}} - Stress and Gradient vs Timestep")
        fig.tight_layout()

        # Save if requested
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{column_names[col].replace(' ', '_')}_stats.png"
            fig.savefig(fname, dpi=300)

        plt.show()
# plot_time_series(mean_shear_log_data_array, erate,shear_columns)

# plot_time_series(mean_eq_log_data_array,erate,eq_columns)



#%%

stats_array=plot_time_series_tstep_converge(mean_stress_array,timestep,stress_columns)
plot_stats_vs_timestep(stats_array, timestep, stress_columns)

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
