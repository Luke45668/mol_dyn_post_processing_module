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




path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_prod_run_n_mols_1688_tstep_3e-07__mass_10_stiff_1_3_sllod_strain_50_T_1_R_0.1_R_n_1_L_150/"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_prod_run_n_mols_1688_tstep_3e-07__mass_10_stiff_1_3_sllod_strain_50_T_0.01_R_0.1_R_n_1_L_150/"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_prod_run_n_mols_1688_tstep_3e-07__mass_10_stiff_1_3_sllod_strain_50_T_0.1_R_0.1_R_n_1_L_150/"
#path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_prod_run_n_mols_1688_tstep_3e-05__mass_10_stiff_0.1_0.5_sllod_strain_100_T_0.01_R_0.5_R_n_1_L_150"
#path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_prod_run_n_mols_1688_tstep_3e-06__mass_10_stiff_1.0_0.5_sllod_strain_200_T_0.001_R_0.1_R_n_1_L_150"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/langevin_runs/DB_shear_prod_run_n_mols_1688_tstep_3e-05__mass_1_stiff_0.25_2.0_sllod_strain_50_T_1_R_0.1_R_n_1_L_150"
timestep=3e-5
vol=150**3
n_mols=1688
n_shear_points=10
erate=np.logspace(-2.5, -1,n_shear_points)
erate=np.linspace(0.1, 0.3,n_shear_points)
erate=np.linspace(0.1, 1,n_shear_points)
erate=np.linspace(0.01, 0.6,n_shear_points)
os.chdir(path_2_files)
K =0.5
mass=1
total_strain=50
damp=0.1

log_name_list = glob.glob("log*_K_"+str(K))
stress_name_list=glob.glob("stress*_K_"+str(K)+"*dat")
spring_name_list=glob.glob("*tensor*_K_"+str(K)+"*dump")
pos_vel_dump_name_list=glob.glob("*_hookean_dumb_bell_*_K_"+str(K)+"*dump")

erate=np.round(erate,7)
zeta=mass/damp
spring_relaxation_time=zeta/(4*K)
Wi=erate*spring_relaxation_time
reals=5
real_target=5
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

data=read_lammps_log_if_complete(log_name_list[15])
eq_columns=list(data[0].columns)
shear_columns=list(data[1].columns)

real_target = 4
erate_count = np.zeros(erate.size, dtype=int)
eq_outs=201
shear_outs=1000
erate_file_name_index=23 #21
eq_cols_count=8
shear_cols_count=12
# Preallocate data arrays
eq_log_data_array = np.zeros((real_target, erate.size, eq_outs, eq_cols_count))
shear_log_data_array = np.zeros((real_target, erate.size, shear_outs, shear_cols_count))

for file in log_name_list:

    
    # Check if real_target already reached
    

    data = read_lammps_log_incomplete(file)

    if data is None:
        continue

    # Extract shear rate from filename
    file_meta_data = file.split("_")
    print(file_meta_data)
    erate_file = round(float(file_meta_data[erate_file_name_index]), 7)
    erate_index = int(np.where(erate == erate_file)[0])

    if erate_count[erate_index] >= real_target:
        continue

    # Assign realisation index (zero-based)

    real_index = erate_count[erate_index]

    if data[0].to_numpy().shape[0] < eq_outs:
        continue

    # Extract thermo outputs as numpy arrays
    eq_log_data_array_raw = data[0].to_numpy()
    

    if data[1].to_numpy().shape[0] < shear_outs:
        continue 
    else:
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



#%% plotting log file data 

# now realisation average 
output_cutoff=1000
mean_shear_log_data_array=np.mean(shear_log_data_array,axis=0)
mean_eq_log_data_array=np.mean(eq_log_data_array,axis=0)

#print(mean_shear_log_data_array.shape)
print(mean_eq_log_data_array.shape)
            

def plot_time_series_eq_converge(data, erate, column_names,output_cutoff, use_latex=True, save=True, save_dir="plots"):
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

    n_erate, n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")

    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    stats_array = np.zeros((n_cols, n_erate, 4))

    for col in range(n_cols):
        plt.figure(figsize=(10, 5))

        for i in range(n_erate):
            y = data[i, :, col]
            number_of_steps=np.linspace(0,(1e-5*1e8)*(output_cutoff/1000),y.shape[0])
            

            # Last 60% of the signal
            last_60_percent = y[int(0.4 * len(y)):]

            # Compute mean and std
            mean = np.mean(last_60_percent)
            std = np.std(last_60_percent)

            # Compute gradient
            gradients = np.gradient(last_60_percent)

            mean_grad = np.mean(gradients)
            print("mean_gradient",mean_grad)
            std_grad = np.std(gradients)

            # Store stats
            stats_array[col, i, 0] = mean
            stats_array[col, i, 1] = std
            stats_array[col, i, 2] = mean_grad
            stats_array[col, i, 3] = std_grad

            # Plot
            plt.plot(number_of_steps,y, label=rf"erate ${erate[i]:.7f}$", linewidth=1.5)

        plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("$t/\\tau$")
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

def plot_time_series_shear_converge(data, erate, column_names,output_cutoff,total_strain, use_latex=True, save=True, save_dir="plots"):
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

    n_erate, n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")

    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    stats_array = np.zeros((n_cols, n_erate, 4))

    for col in range(n_cols):
        plt.figure(figsize=(10, 5))

        for i in range(n_erate):
            y = data[i, :, col]
            number_of_steps=np.linspace(0,total_strain,y.shape[0])
            

            # Last 60% of the signal
            last_60_percent = y[int(0.6 * len(y)):]

            # Compute mean and std
            mean = np.mean(last_60_percent)
            std = np.std(last_60_percent)

            # Compute gradient
            gradients = np.gradient(last_60_percent)

            mean_grad = np.mean(gradients)
            print("mean_gradient",mean_grad)
            std_grad = np.std(gradients)

            # Store stats
            stats_array[col, i, 0] = mean
            stats_array[col, i, 1] = std
            stats_array[col, i, 2] = mean_grad
            stats_array[col, i, 3] = std_grad

            # Plot
            plt.plot(number_of_steps,y, label=rf"erate ${erate[i]:.7f}$", linewidth=1.5)

        plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("$\gamma$")
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

def plot_stats_vs_timestep_log_file(stats_array, timestep, column_names, use_latex=True, gradient_threshold=1e-4, save=True, save_dir="plots"):
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

        # # Plot stress mean ± std
        # ax1.errorbar(timestep, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, color='tab:blue')
        # ax1.set_xlabel(r"Timestep")
        # ax1.set_ylabel(r"Stress Mean", color='tab:blue')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')
        # ax1.set_xscale('log')
        # ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot gradient mean ± std on twin axis
        # ax2 = ax1.twinx()
        ax1.errorbar(timestep, grad_means, yerr=grad_stds, fmt='s--', capsize=4, linewidth=2, color='black', markersize=5)
        ax1.set_ylabel(r"Gradient Mean", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Highlight converged points (high contrast color + edge)
        converged = np.abs(grad_means) < gradient_threshold
        ax1.plot(np.array(timestep)[converged], grad_means[converged], 'o', markersize=12,
                 markerfacecolor='gold', markeredgecolor='black', markeredgewidth=1.5, label='Converged (|grad| < tol)')

        # Clean manual legend
        handles = [
            #plt.Line2D([], [], color='tab:blue', marker='o', linestyle='-', linewidth=2, label="Stress Mean ± Std"),
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

stats_array_eq=plot_time_series_eq_converge(mean_eq_log_data_array, erate, eq_columns,output_cutoff, use_latex=True, save=True, save_dir="plots")

stats_array_shear=plot_time_series_shear_converge(mean_shear_log_data_array, erate, shear_columns,output_cutoff,total_strain, use_latex=True, save=True, save_dir="plots")



plot_stats_vs_timestep_log_file(stats_array_eq,erate,eq_columns,use_latex=True, gradient_threshold=1e-2, save=True, save_dir="plots" )

plot_stats_vs_timestep_log_file(stats_array_shear,erate,shear_columns,use_latex=True, gradient_threshold=1e-2, save=True, save_dir="plots" )






#%% energy drift plot 










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

    N1 = sxx-syy
    N2 = syy - szz

    if return_data:
        return {
            'time': time,
            '$\sigma_{xx}$': sxx, '$\sigma_{yy}$': syy, '$\sigma_{zz}$': szz,
            '$\sigma_{xy}$': sxy, '$\sigma_{xz}$': sxz, '$\sigma_{yz}$': syz,
            '$N_{1}$': N1, '$N_{2}$': N2
        }

stress_name_list=glob.glob("stress*K_"+str(K)+"*.dat")
data_dict = read_stress_tensor_file(filename=stress_name_list[0], volume=vol, return_data=True)
stress_columns = list(data_dict.keys())
output_cutoff=999

erate_count = np.zeros(erate.size, dtype=int)
stress_array = np.zeros((real_target, erate.size, output_cutoff, 9))
erate_file_name_index=26 #18
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
    erate_file = round(float(file_meta_data[erate_file_name_index]), 7)
    erate_index = int(np.where(erate == erate_file)[0])
    if erate_count[erate_index] >= real_target:
        continue
    else:

        real_index=erate_count[erate_index]

        erate_count[erate_index] += 1
        print(erate[erate_index])
        print(f"Realisation: {real_index}")

    # Fill stress array
    for column, key in enumerate(stress_columns):
        raw_stress_array = data_dict[key][:output_cutoff]
        stress_array[real_index, erate_index, :, column] = raw_stress_array

# Compute mean
mean_stress_array = np.mean(stress_array, axis=0)

print("Mean stress array shape:", mean_stress_array.shape)
print(erate_count)


stats_array_shear=plot_time_series_shear_converge(mean_stress_array, erate,stress_columns,output_cutoff,total_strain, use_latex=True, save=True, save_dir="plots")

#%%
plot_stats_vs_timestep_log_file(stats_array_shear,erate,stress_columns,use_latex=True, gradient_threshold=1e-3, save=True, save_dir="plots" )






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
truncate=800
time_mean_stress=np.mean(mean_stress_array[:,truncate:,:],axis=1)
time_std_stress=np.std(mean_stress_array[:,truncate:,:],axis=1)


# %%
def plot_stress_components(
    Wi,
    time_mean_stress,
    time_std_stress,
    stress_columns,
    i_range=(1, 4),
    fit_type=None,                # 'linear', 'quadratic', or None
    fit_index=None,              # column index to fit
    fit_points=None,             # list or array of indices for fitting, e.g., [0, 1, 2]
    save=False,
    save_path="plots/stress_components.png"
):
    """
    Plots stress components with error bars, with optional linear (through origin) or quadratic fit.

    Parameters:
        Wi (array): Weissenberg numbers (x-axis)
        time_mean_stress (2D array): shape (N, M), stress means
        time_std_stress (2D array): shape (N, M), stress stds
        stress_columns (list): list of column names for the stress components
        i_range (tuple): (start, end) indices of columns to plot [start, end)
        fit_type (str or None): 'linear', 'quadratic', or None
        fit_index (int or None): which column index to fit, must be in i_range if fitting is desired
        fit_points (list or None): list of indices to use for fitting (subset of data points)
        save (bool): whether to save the plot
        save_path (str): filepath to save the figure
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os

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

        if fit_type and i == fit_index:
            x = np.array(Wi)
            y = time_mean_stress[:, i]

            if fit_points is not None:
                x = x[fit_points]
                y = y[fit_points]

            if fit_type == 'linear':
                # Least squares fit through origin: y = m*x
                m = np.dot(x, y) / np.dot(x, x)
                fit_y = m * np.array(Wi)
                label = rf"Linear fit: {stress_columns[i]} $= {m:.3g} \cdot Wi$"

            elif fit_type == 'quadratic':
                A = np.vstack([x**2, np.ones_like(x)]).T
                a, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                fit_y = a * np.array(Wi)**2
                label = rf"Quadratic fit: {stress_columns[i]} $= {a:.3g} \cdot Wi^2$"

            else:
                raise ValueError("Invalid fit_type. Use 'linear', 'quadratic', or None.")

            ax.plot(
                Wi,
                fit_y,
                '--',
                color='black',
                linewidth=2,
                label=label
            )

    ax.set_xlabel(r"Wi")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close()

quad_fit=[0,1,2,3,4,5,6]
plot_stress_components(Wi, time_mean_stress, time_std_stress, stress_columns, i_range=(7,9), fit_type="quadratic", fit_index=7,fit_points=quad_fit,save=True,
    save_path="plots/stress_components_"+str(7)+"_"+str(9)+".png")

plot_stress_components(Wi, time_mean_stress, time_std_stress, stress_columns, i_range=(1,4),  fit_type=None, fit_index=1,fit_points=None,save=True,
     save_path="plots/stress_components_"+str(1)+"_"+str(4)+".png")


linear_fit=[0,1,2,3,4,5,6]
plot_stress_components(Wi, time_mean_stress, time_std_stress, stress_columns, i_range=(4,7),  fit_type="linear", fit_index=4,fit_points=linear_fit, save=True,
    save_path="plots/stress_components_"+str(4)+"_"+str(7)+".png")
# %% now looking at the orientation distributions only for dumbbells

spring_name_list=glob.glob("DBshear*tensor*_K_"+str(K)+"*dump")

def read_lammps_dump_tensor(filename):
    """
    Reads LAMMPS dump style file with tensor entries (can handle incomplete files).

    Returns:
        dump_data (list of dict): Each dict has timestep, number of entries, box bounds, column names, and NumPy array of entries
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
                current_data = {
                    "timestep": None,
                    "n_entries": None,
                    "box_bounds": [],
                    "columns": None,
                    "data": []
                }
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
                if len(parts) != len(current_data["columns"]):
                    continue
                try:
                    current_data["data"].append([float(x) for x in parts])
                except ValueError:
                    continue

    # Save last block
    if current_data is not None:
        dump_data.append(current_data)

    # Convert data to NumPy arrays for each block
    for block in dump_data:
        block["data"] = np.array(block["data"], dtype=np.float64)

    return dump_data

def convert_cart_2_spherical_z_inc_DB_from_dict(spring_vector_ray,n_mols
   
):
        
        spring_vector_ray[spring_vector_ray[ :, 2] < 0] *= -1

        x = spring_vector_ray[ :, 0]
        y = spring_vector_ray[ :, 1]
        z = spring_vector_ray[ :, 2]

        spherical_coords_array = np.zeros(
            ( n_mols, 3)
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

def convert_cart_2_spherical_y_inc_DB_from_dict(spring_vector_ray, n_mols):
    # Ensure vectors point into the positive Y hemisphere
    spring_vector_ray[spring_vector_ray[:, 1] < 0] *= -1

    x = spring_vector_ray[:, 0]
    y = spring_vector_ray[:, 1]
    z = spring_vector_ray[:, 2]

    spherical_coords_array = np.zeros((n_mols, 3))

    # r: radial distance
    spherical_coords_array[:, 0] = np.sqrt(x**2 + y**2 + z**2)

    # theta: azimuthal angle in XZ plane from +X toward +Z
    spherical_coords_array[:, 1] = np.arctan2(z, x)

    # phi: polar angle from Y-axis down
    spherical_coords_array[:, 2] = np.arccos(y / spherical_coords_array[:, 0])

    return spherical_coords_array

erate_count = np.zeros(erate.size, dtype=int)
# dump_data = read_lammps_dump_tensor(spring_name_list[0])


#%%

# creating dict to store the list in 
#spring_data_dict={'box_sizes':box_sizes_list_array}
erate_file_name_index=23
tensor_col_count=3
output_cutoff=999
spherical_coords_array=np.zeros((real_target,erate.size,output_cutoff,n_mols,tensor_col_count))
erate_count = np.zeros(erate.size, dtype=int)

for file in spring_name_list:
    file_meta_data = file.split("_")
    print(file_meta_data)

    dump_data = read_lammps_dump_tensor(file)
    
    if dump_data is None:
        continue
    print("n_outs",len(dump_data))
    if len(dump_data) <output_cutoff:
        continue

    # Extract metadata
    
    
    file_meta_data = file.split("_")
    print(file_meta_data)
    erate_file = round(float(file_meta_data[erate_file_name_index]), 7)
    erate_index = int(np.where(erate == erate_file)[0])
    print(erate_index)
   
    if erate_count[erate_index] >= real_target:
        continue

    
    else:

        real_index=erate_count[erate_index]

        erate_count[erate_index] += 1
        print(erate[erate_index])
        print(f"Realisation: {real_index}")

    

        for i in range(output_cutoff):

            dump_data_np_array=dump_data[i]['data']
            spherical_np_array=convert_cart_2_spherical_y_inc_DB_from_dict(dump_data_np_array,n_mols)
            spherical_coords_array[real_index,erate_index,i]=spherical_np_array
            
print(erate_count)

#%%
def plot_spherical_kde_plate_from_numpy_DB(
    spherical_coords_array,
    erate,
    cutoff,
    selected_erate_indices,
    save=False,
    save_dir="plots",
    use_latex=True
):
    """
    Overlayed KDE plots of spherical coordinates (rho, theta, phi) for selected erate indices.
    Adds mean(rho) to legend only in the rho subplot.

    Parameters:
        spherical_coords_array: np.ndarray with shape [samples, erates, time, particles, coords(3)]
        erate: np.ndarray of strain rates
        cutoff: int, time index cutoff
        selected_erate_indices: list of indices into erate array
        save: bool, whether to save the figures
        save_dir: str, directory to save plots
        use_latex: bool, whether to use LaTeX rendering
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

    if save:
        os.makedirs(save_dir, exist_ok=True)

    pi_theta_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    pi_theta_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    pi_phi_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
    pi_phi_labels = [r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"]

    # Prepare figures
    fig_rho, ax_rho = plt.subplots(figsize=(8, 4))
    fig_theta, ax_theta = plt.subplots(figsize=(8, 4))
    fig_phi, ax_phi = plt.subplots(figsize=(8, 4))

    for j in range(len(selected_erate_indices)):
        i = int(selected_erate_indices[j])
        
        # Extract data
        rho = np.ravel(spherical_coords_array[:, i, cutoff:, :, 0])
        theta = spherical_coords_array[:, i, cutoff:, :, 1]
        theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
        phi = spherical_coords_array[:, i, cutoff:, :, 2]
        phi = np.ravel(np.array([phi, np.pi - phi]))

        rho_mean = np.mean(rho)
        label_rho = rf"$\dot{{\gamma}} = {erate[i]:.1e}$, $\langle \rho \rangle = {rho_mean:.2f}$"
        label = rf"$\dot{{\gamma}} = {erate[i]:.1e}$"

        # Plot KDEs
        sns.kdeplot(rho, label=label_rho, ax=ax_rho, linewidth=2)
        sns.kdeplot(theta, label=label, ax=ax_theta, linewidth=2)
        sns.kdeplot(phi, label=label, ax=ax_phi, linewidth=2)

    # Format RHO plot
    ax_rho.set_xlabel(r"$\rho$")
    ax_rho.set_ylabel("Density")
    ax_rho.set_title(r"$\rho$ Distribution")
    ax_rho.grid(True, linestyle='--', alpha=0.7)
    ax_rho.legend()

    # Format THETA plot
    ax_theta.set_xlabel(r"$\Theta$")
    ax_theta.set_ylabel("Density")
    ax_theta.set_title(r"$\Theta$ Distribution")
    ax_theta.set_xticks(pi_theta_ticks)
    ax_theta.set_xticklabels(pi_theta_labels)
    ax_theta.set_xlim(-np.pi, np.pi)
    ax_theta.grid(True, linestyle='--', alpha=0.7)
    ax_theta.legend()

    # Format PHI plot
    ax_phi.set_xlabel(r"$\phi$")
    ax_phi.set_ylabel("Density")
    ax_phi.set_title(r"$\phi$ Distribution")
    ax_phi.set_xticks(pi_phi_ticks)
    ax_phi.set_xticklabels(pi_phi_labels)
    ax_phi.set_xlim(0, np.pi / 2)
    ax_phi.grid(True, linestyle='--', alpha=0.7)
    ax_phi.legend()

    # Save if requested
    if save:
        fig_rho.savefig(f"{save_dir}/rho_kde.png", dpi=300)
        fig_theta.savefig(f"{save_dir}/theta_kde.png", dpi=300)
        fig_phi.savefig(f"{save_dir}/phi_kde.png", dpi=300)

    plt.show()
    plt.close('all')

plot_spherical_kde_plate_from_numpy_DB( spherical_coords_array, erate, 600, save=True, selected_erate_indices=[0,1,3,5,6])


#%% now looking at dump files of velocity and position 
#pos_vel_dump_name_list
output_cutoff=1000
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

pos_vel_dump_array=np.zeros((real_target,erate.size,output_cutoff,n_mols*2,6))
erate_count = np.zeros(erate.size, dtype=int)
erate_file_name_index=23
for file in pos_vel_dump_name_list:
    file_meta_data = file.split("_")
    print(file_meta_data)

    dump_data = read_lammps_posvel_dump_to_numpy(file)
    
    if dump_data is None:
        continue
    print("n_outs",dump_data.shape[0])
    if dump_data.shape[0] <output_cutoff:
        continue

    # Extract metadata
    
    
    file_meta_data = file.split("_")
    print(file_meta_data)
    erate_file = round(float(file_meta_data[erate_file_name_index]), 7)
    erate_index = int(np.where(erate == erate_file)[0])
    print(erate_index)
   
    if erate_count[erate_index] >= real_target:
        continue

    
    else:

        real_index=erate_count[erate_index]

        erate_count[erate_index] += 1
        print(erate[erate_index])
        print(f"Realisation: {real_index}")
        pos_vel_dump_array[real_index,erate_index]=dump_data[:,:,2:]

print(erate_count)


#%% plotting vx against z to check velocity profiles 

# will just truncate and take time average 
#x y z vx vy vz
erate_skip_array=[0,1,2]
for j in range(len(erate_skip_array)):
    i=erate_skip_array[j]
    v_x=np.ravel(pos_vel_dump_array[:,i,:,:,3])
    r_y=np.ravel(pos_vel_dump_array[:,i,:,:,1])

    m, _ = np.polyfit(r_y, v_x, 1)
    fit_line = m * r_z

    plt.figure(figsize=(6, 4))
    plt.scatter(r_y, v_x, alpha=0.6, label="Data")
    plt.plot(r_y, fit_line, color='red', label=fr"Fit: $v_x = {m:.5f} \cdot r_z$")
    
    plt.xlabel(r"$r_z$")
    plt.ylabel(r"$v_x$")
    plt.title(fr"$\dot{{\gamma}} = {erate[i]:.1e}$")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




# %%
def plot_vx_vs_rz_with_fit(
    pos_vel_dump_array,
    erate,
    selected_erate_indices,
    save=False,
    save_dir="plots",
    use_latex=True
):
    """
    Plots v_x vs r_ywith linear best-fit line (v_x = m * r_z) for selected erate indices.

    Parameters:
        pos_vel_dump_array: np.ndarray, shape [samples, erates, time, particles, values]
        erate: np.ndarray of strain rates
        selected_erate_indices: list of erate indices to include
        save: bool, whether to save the plots
        save_dir: str, path to directory for saving
        use_latex: bool, toggle LaTeX rendering
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

    if save:
        os.makedirs(save_dir, exist_ok=True)

    for idx in selected_erate_indices:
        # Flatten v_x and r_z
        v_x = np.ravel(pos_vel_dump_array[:, idx, :, :, 3])
        r_y= np.ravel(pos_vel_dump_array[:, idx, :, :, 1])

        # Linear fit: v_x = m * r_z
        m, _ = np.polyfit(r_y, v_x, 1)
        fit_line = m * r_y

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(r_y, v_x, alpha=0.5, label="Data")
        ax.plot(r_y, fit_line, color='red', label=fr"Fit: $v_x = {m:.5f} \cdot r_y$")
        
        ax.set_xlabel(r"$r_y$")
        ax.set_ylabel(r"$v_x$")
        ax.set_title(fr"$\dot{{\gamma}} = {erate[idx]:.1e}$")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        if save:
            fig.savefig(f"{save_dir}/vx_vs_rz_fit_erate_{idx}.png", dpi=300)

        plt.show()
        plt.close(fig)

plot_vx_vs_rz_with_fit(pos_vel_dump_array, erate,save=True, selected_erate_indices=[0, 1, 2, 9])
# %%
# now need to look at rotation velocity of dumbells , could look at the rate of rotation of the rho vector , from the spherical coordinates
# perhaps there is a way to compute eigen values, to look at the rotation 