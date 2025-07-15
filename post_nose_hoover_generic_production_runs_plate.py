# %% new section where stress and log files are used for intial run 
import os
import re
import glob
import pandas as pd
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from plotting_module import *
# === Setup




path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_prod_run_n_mols_1688_tstep_3e-05__mass_1_stiff_0.25_2.0_sllod_strain_50_T_1_R_0.1_R_n_0.5_L_150"
timestep=3e-5
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_prod_run_n_mols_1688_tstep_1.25e-06__mass_1_Bend_250_stiff_0.25_2.0_sllod_strain_25_T_1_R_0.1_R_n_2.598_L_150"
# timestep=1.25e-6
#path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_prod_run_n_mols_1688_tstep_1.25e-06__mass_1_Bend_500_stiff_0.25_2.0_sllod_strain_25_T_1_R_0.1_R_n_2.598_L_150"

#path_2_files='/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_prod_run_n_mols_1688_tstep_1e-05__mass_1_Bend_50_stiff_0.25_2.0_sllod_strain_25_T_1_R_0.1_R_n_1_L_150'
#path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_phantnothermo_prod_run_n_mols_1688_tstep_1e-05__mass_1_Bend_50_stiff_0.25_2.0_sllod_strain_25_T_1_R_0.1_R_n_1_L_150"
#path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_phantnothermo_prod_run_n_mols_1688_tstep_1e-05__mass_1_Bend_500_stiff_0.25_2.0_sllod_strain_25_T_1_R_0.1_R_n_1_L_150"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_prod_run_n_mols_1688_tstep_1e-05__mass_1_Bend_10000_stiff_0.25_2.0_sllod_strain_25_T_1_R_0.1_R_n_1_L_150/"
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/plate_runs/langevin_runs/plate_shear_prod_run_n_mols_1688_tstep_1e-05__mass_1_Bend_1000_stiff_0.25_2.0_sllod_strain_25_T_1_R_0.1_R_n_1_L_150/"
timestep=1e-5

vol=150**3
n_mols=1688
n_shear_points=10
erate=np.linspace(0.01, 0.4,n_shear_points)
#erate=np.linspace(0.01, 0.6,n_shear_points)
os.chdir(path_2_files)
K = 1.0
mass=1
total_strain=25

log_name_list = glob.glob("log*_K_"+str(K))

spring_name_list=glob.glob("*tensor*_K_"+str(K)+"*dump")
pos_vel_dump_name_list=glob.glob("plateshearnvt*_hookean_flat_elastic_*_K_"+str(K)+"*dump")

erate=np.round(erate,7)
spring_relaxation_time=np.sqrt(mass/K)
Wi=erate*spring_relaxation_time
reals=5
real_target=4
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
shear_columns=list(data[1].columns)

real_target = 5
erate_count = np.zeros(erate.size, dtype=int)
eq_outs=201
shear_outs=700 
erate_file_name_index=22
eq_cols_count=9
shear_cols_count=12
# Preallocate data arrays
eq_log_data_array = np.zeros((real_target, erate.size, eq_outs, eq_cols_count))
shear_log_data_array = np.zeros((real_target, erate.size, shear_outs, shear_cols_count))

for file in log_name_list:

    
    # Check if real_target already reached
    

    data = read_lammps_log_incomplete(file)

    if data is None:
        continue
    if len(data)<2:
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
    shear_log_data_array[real_index, erate_index] = shear_log_data_array_raw[:shear_outs]

    # Increment count
    erate_count[erate_index] += 1

print(erate_count)
print(shear_log_data_array.shape)
print(eq_log_data_array.shape)


#%% now selecting only the data that has the right number of  realisations 

success_index_list=[]
for i in range(len(erate_count)):
    if erate_count[i]==real_target:
       success_index_list.append(i)


    
     




#%% plotting log file data 

# now realisation average 
output_cutoff=1000
mean_shear_log_data_array=np.mean(shear_log_data_array,axis=0)
mean_eq_log_data_array=np.mean(eq_log_data_array,axis=0)

#print(mean_shear_log_data_array.shape)
print(mean_eq_log_data_array.shape)
            

def plot_time_series_eq_converge(data, erate, column_names,output_cutoff,success_index_list, use_latex=True, save=False, save_dir="plots"):
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

    n_erate,n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")

    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    stats_array = np.zeros((n_cols, n_erate, 4))

    for col in range(n_cols):
        plt.figure(figsize=(10, 5))

        for j in range(len(success_index_list)):
            i=success_index_list[j]

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
            std_grad = np.std(gradients)

            # Store stats
            stats_array[col, i, 0] = mean
            stats_array[col, i, 1] = std
            stats_array[col, i, 2] = mean_grad
            stats_array[col, i, 3] = std_grad

            # Plot
            plt.plot(number_of_steps,y, label=rf"erate ${erate[i]:.2f}$", linewidth=1.5)

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

def plot_time_series_shear_converge(data, erate, column_names,output_cutoff,total_strain,success_index_list,shear_outs, use_latex=True, save=False, save_dir="plots"):
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

    n_erate,n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")

    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    stats_array = np.zeros((n_cols, n_erate, 4))

    for col in range(n_cols):
        plt.figure(figsize=(10, 5))

        for j in range(len(success_index_list)):
            i=success_index_list[j]
            y = data[i, :, col]
            number_of_steps=np.linspace(0,(shear_outs/1000)*total_strain,y.shape[0])
            

            # Last 60% of the signal
            last_60_percent = y[int(0.6 * len(y)):]

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
            plt.plot(number_of_steps,y, label=rf"$\dot{{\gamma}}={erate[i]:.2f}$", linewidth=1.5)

       # plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("$\gamma$")
        plt.ylabel(rf"\textbf{{{column_names[col]}}}",rotation=0, labelpad=10)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.tight_layout()
        save_string = column_names[col].replace(' ', '_').replace('$', '').replace('\\', '')
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}.png"
            plt.savefig(fname, dpi=300)

        plt.show()

    return stats_array

def plot_stats_vs_erate_log_file(
    stats_array,
    erate,
    column_names,
    success_index_list,
    use_latex=True,
    gradient_threshold=1e-4,
    save=False,
    save_dir="plots"
):
    """
    Plots gradient mean ± std vs shear rate (erate) from stats_array, for successful indices only.
    Highlights convergence points (|grad| < threshold) and optionally saves figures.

    Parameters:
        stats_array: shape (n_columns, n_cases, 4), where:
                     [:, :, 2] = grad_mean, [:, :, 3] = grad_std
        erate: array of shear rates
        column_names: list of names corresponding to stress components (one per column)
        success_index_list: list of indices to include from the data (subset of cases)
        use_latex: bool, whether to render using LaTeX
        gradient_threshold: float, convergence threshold for gradient
        save: bool, whether to save the plots
        save_dir: str, folder to save plots
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

    # Subset erate and stats array
    erate_sub = np.array(erate)[success_index_list]
    stats_sub = stats_array[:, success_index_list, :]  # shape: [n_cols, len(success), 4]

    n_cols = stats_sub.shape[0]

    for col in range(n_cols):
        fig, ax = plt.subplots(figsize=(8, 5))

        grad_means = stats_sub[col, :, 2]
        grad_stds = stats_sub[col, :, 3]

        # Plot gradient mean ± std
        ax.errorbar(
            erate_sub,
            grad_means,
            yerr=grad_stds,
            fmt='s--',
            capsize=4,
            linewidth=2,
            color='black',
            markersize=5
        )

        # Highlight convergence points
        converged = np.abs(grad_means) < gradient_threshold
        ax.plot(
            erate_sub[converged],
            grad_means[converged],
            'o',
            markersize=12,
            markerfacecolor='gold',
            markeredgecolor='black',
            markeredgewidth=1.5,
            label='Converged (|grad| < tol)'
        )

        # Labels and styling
        ax.set_xlabel(r"$\dot{\gamma}$")
        ax.set_ylabel(r"Gradient Mean", color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        handles = [
            plt.Line2D([], [], color='black', marker='s', linestyle='--', linewidth=2, label="Gradient Mean ± Std"),
            plt.Line2D([], [], color='gold', marker='o', markeredgecolor='black', linestyle='None', markersize=10, label="Converged ($|\\mathrm{grad}| < \\mathrm{tol}$)")
        ]
        ax.legend(handles=handles, loc='best', fontsize=11, frameon=False, bbox_to_anchor=(1, 1))

        plt.title(rf"\textbf{{{column_names[col]}}} - Gradient vs $\dot{{\gamma}}$")
        fig.tight_layout()

        # Save if needed
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{column_names[col].replace(' ', '_')}_gradient_vs_erate.png"
            fig.savefig(fname, dpi=300)
            print(f"Saved: {fname}")

        plt.show()
        plt.close()
# plot_time_series(mean_shear_log_data_array, erate,shear_columns)

# plot_time_series(mean_eq_log_data_array,erate,eq_columns)

def plot_time_series_shear_comparison(
    data_list,                   # list of arrays (n_erate, n_steps, n_cols)
    erate,                       # single array of erate values, same for all datasets
    column_names,                # list of column names
    total_strain,
    shear_outs,
    dataset_labels=None,         # optional list of labels for the datasets
    use_latex=True,
    save=False,
    save_dir="plots"
):
    """
    Plots time series data for each column.
    For multiple datasets (e.g. 3), makes subplots in the same figure per column for easy comparison.
    All erate indices are plotted; no filtering by success indices.
    """

    # Plot styling
    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    n_datasets = len(data_list)
    n_cols = data_list[0].shape[2]

    if dataset_labels is None:
        dataset_labels = [f"Dataset {i+1}" for i in range(n_datasets)]

    # Check consistency
    for data in data_list:
        if data.shape[2] != n_cols:
            raise ValueError("All datasets must have the same number of columns")

    # For each column, make one figure with subplots
    for col in range(n_cols):
        fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), sharey=True)

        # Handle case of only 1 dataset
        if n_datasets == 1:
            axes = [axes]

        for d in range(n_datasets):
            data = data_list[d]
            n_erate = data.shape[0]
            ax = axes[d]

            for i in range(n_erate):
                y = data[i, :, col]
                number_of_steps = np.linspace(0, (shear_outs / 1000) * total_strain, y.shape[0])
                ax.plot(number_of_steps, y, label=rf"$\dot\gamma={erate[i]:.2f}$", linewidth=1.5)

            ax.set_title(rf"\textbf{{{dataset_labels[d]}}}")
            ax.set_xlabel(r"$\gamma$")
            # if d == 0:
            #     ax.set_ylabel(rf"\textbf{{{column_names[col]}}}",rotation=0)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
         
         
        axes[n_datasets-1].legend(loc='best',bbox_to_anchor=(1,1))
        fig.suptitle(rf"\textbf{{{column_names[col]}}}", fontsize=20)
       # plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tight_layout()

        save_string = column_names[col].replace(' ', '_').replace('$', '').replace('\\', '')
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}_time_series_comparison.png"
            plt.savefig(fname, dpi=300)

        plt.show()


#%%
shear_columns =['Step',
 '$E_{K}$',
 '$E_{P,s}$',
 '$E_{P,a}$',
 '$E_{P}$',
 'Press',
 'c_myTemp',
 '$T$',
 'c_bias_2',
 'Econserve',
 'Ecouple',
 '$E_{t}$',]
xlabel=r"$\dot{\gamma}$"

#stats_array_eq=plot_time_series_eq_converge(mean_eq_log_data_array, erate, eq_columns,output_cutoff,success_index_list, use_latex=True, save=True, save_dir="plots")


stats_array_shear=plot_time_series_shear_converge(mean_shear_log_data_array, erate, shear_columns,output_cutoff,total_strain,success_index_list,shear_outs, use_latex=True, save=True, save_dir="plots_K_"+f"{K}")



plot_stats_vs_indepvar_log_file(stats_array_shear,erate,xlabel,shear_columns,use_latex=True, gradient_threshold=1e-2, save=True, save_dir="plots_K_"+f"{K}" )


#%% save log data array 

File_name=f"plate_mean_shear_log_data_K_{K}"
np.save(File_name,mean_shear_log_data_array)

#%% energy drift 

# need to load in both stiffnesses from the log file data, can save the mean shear log data files
K_list=[0.5,1.0]
mean_shear_log_data_array_all_K=np.zeros((len(K_list),erate.size,700,12))
for i in range(len(K_list)):
    mean_shear_log_data_array_all_K[i]=np.load(f"plate_mean_shear_log_data_K_{K_list[i]}.npy")


def plot_energy_drift(data, erate,E_t_col,total_strain,timestep,n_mols,K_list, use_latex=True, save=True, save_dir="plots"):
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
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    n_K,n_erate, n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")
    gradient_threshold=4e-13
    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    

    
    plt.figure(figsize=(10, 5))
    for j in range(len(K_list)):
        drift_array = np.zeros((n_erate, 1))
        for i in range(n_erate):
            y = data[j, i, :, E_t_col]
            number_of_steps=total_strain/(erate[i]*timestep)
            print(number_of_steps)
                

            drift_per_particle=(y[-1]-y[0])/(y[0]) *1/(number_of_steps*n_mols*2)

                # Store stats
            drift_array[i, 0] = drift_per_particle
        

                # Plot
        plt.scatter(erate,drift_array[:, 0], label=rf"$K={K_list[j]}$", linewidth=1.5, marker="x")
    plt.fill_between(erate, -gradient_threshold, gradient_threshold, color='red', alpha=0.1, label='Tolerance Band')
    #plt.title(rf"\textbf{{{column_names[col]}}}")
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel(r"$\Delta E$", rotation=0, labelpad=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

   
    save_string=f"E_t_shear_drift_all_K"

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{save_dir}/{save_string}.png"
        plt.savefig(fname, dpi=300)

    plt.show()

    return drift_array

E_t_col=8
plot_energy_drift(mean_shear_log_data_array_all_K, erate,E_t_col,total_strain,timestep,n_mols,K_list,use_latex=True, save=True, save_dir="plots")


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


#stress_name_list=sorted(glob.glob("stress_tensor_avg_*K_"+str(K)+"*.dat"))
stress_name_list=sorted(glob.glob("stress_tensor_allavg*K_"+str(K)+"*.dat"))
small_stress_name_list=sorted(glob.glob("stress_tensor_smallavg*K_"+str(K)+"*.dat"))
phantom_stress_name_list=sorted(glob.glob("stress_tensor_phantavg*K_"+str(K)+"*.dat"))


# stress_name_list=sorted(glob.glob("stress_tensor_avgang*K_"+str(K)+"*.dat"))
# phantom_stress_name_list=sorted(glob.glob("phantomstress_tensor_avgang*K_"+str(K)+"*.dat"))

data_dict = read_stress_tensor_file(filename=stress_name_list[5], volume=vol, return_data=True)
stress_columns = list(data_dict.keys())

shear_outs=700 
erate_count = np.zeros(erate.size, dtype=int)
stress_array = np.zeros((real_target, erate.size, shear_outs, 9))
phantom_stress_array=np.zeros((real_target, erate.size, shear_outs, 9))
small_stress_array=np.zeros((real_target, erate.size, shear_outs, 9))
erate_file_name_index=20
#%%
for file,file1,file2 in zip(stress_name_list,phantom_stress_name_list,small_stress_name_list):
    data_dict = read_stress_tensor_file(filename=file, volume=vol, return_data=True)
    phantom_data_dict=read_stress_tensor_file(filename=file1, volume=vol, return_data=True)
    small_data_dict=read_stress_tensor_file(filename=file2, volume=vol, return_data=True)
    if data_dict is None:
        continue

    if data_dict["time"].size <shear_outs:
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
        raw_phantom_stress_array=phantom_data_dict[key][:output_cutoff]
        raw_small_stress_array=small_data_dict[key][:output_cutoff]
        
        stress_array[real_index, erate_index, :, column] = raw_stress_array[:shear_outs] #+ raw_phantom_stress_array[:shear_outs]
        small_stress_array[real_index, erate_index, :, column] = raw_small_stress_array[:shear_outs] #+ raw_phantom_stress_array[:shear_outs]
        phantom_stress_array[real_index, erate_index, :, column] = raw_phantom_stress_array[:shear_outs] #+ raw_phantom_stress_array[:shear_outs]
       
        
# Compute mean
mean_stress_array = np.mean(stress_array, axis=0)
mean_small_stress_array = np.mean(small_stress_array, axis=0)
mean_phantom_stress_array = np.mean(phantom_stress_array, axis=0)


print("Mean stress array shape:", mean_stress_array.shape)
print(erate_count)


#stats_array_stress=plot_time_series_shear_converge(mean_stress_array, erate,stress_columns,output_cutoff,total_strain,success_index_list,shear_outs, use_latex=True, save=True, save_dir="plots")
stats_array_stress=plot_time_series_shear_converge(mean_stress_array, erate,stress_columns,output_cutoff,total_strain,success_index_list,shear_outs, use_latex=True, save=True, save_dir="plots_K_"+f"{K}")
#stats_array_small_stress=plot_time_series_shear_converge(mean_small_stress_array, erate,stress_columns,output_cutoff,total_strain,success_index_list,shear_outs, use_latex=True, save=False, save_dir="plots_ang")
#stats_array_phantom_stress=plot_time_series_shear_converge(mean_phantom_stress_array, erate,stress_columns,output_cutoff,total_strain,success_index_list,shear_outs, use_latex=True, save=False, save_dir="plots_ang")
#%% comparison of time series 

stress_data_list=[mean_stress_array,mean_small_stress_array,mean_phantom_stress_array]
dataset_labels=["All particles", "Stokes beads","Phantom beads"]
plot_time_series_shear_comparison(
    stress_data_list,                   # list of arrays (n_erate, n_steps, n_cols)
    erate,                       # single array of erate values, same for all datasets
    stress_columns,                # list of column names
    total_strain,   
    shear_outs,
    dataset_labels,         # optional list of labels for the datasets
    use_latex=True,
    save=True,
    save_dir="plots_K_"+f"{K}"
)


#%%
xlabel=r"$\dot{\gamma}$"
plot_stats_vs_indepvar_log_file(stats_array_stress,erate,xlabel,stress_columns,use_latex=True, gradient_threshold=1e-2, save=True, save_dir="plots_K_"+f"{K}" )
#plot_stats_vs_indepvar_log_file(stats_array_small_stress,erate,xlabel,stress_columns,use_latex=True, gradient_threshold=1e-2, save=True, save_dir="plots_K_"+f"{K}" )
#plot_stats_vs_indepvar_log_file(stats_array_phantom_stress,erate,xlabel,stress_columns,use_latex=True, gradient_threshold=1e-2, save=True, save_dir="plots_K_"+f"{K}" )



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
truncate=400 # needs to be at 10 strain 
time_mean_stress=np.mean(mean_stress_array[:,truncate:,:],axis=1)
time_std_stress=np.std(mean_stress_array[:,truncate:,:],axis=1)
file_name_mean=f"time_mean_stress_K_{K}_trunc_{truncate}"
file_name_std=f"time_std_stress_K_{K}_trunc_{truncate}"

np.save(file_name_mean,time_mean_stress)
np.save(file_name_std,time_std_stress)

#%% loading in both stiffnesses
K_list=[0.5,1.0]
trunc_list=[400,400]
time_mean_stress_data_array_all_K=np.zeros((len(K_list),erate.size,len(stress_columns)))
time_std_stress_data_array_all_K=np.zeros((len(K_list),erate.size,len(stress_columns)))
for i in range(len(K_list)):
    time_mean_stress_data_array_all_K[i]=np.load(f"time_mean_stress_K_{K_list[i]}_trunc_{trunc_list[i]}.npy")
    time_std_stress_data_array_all_K[i]=np.load(f"time_std_stress_K_{K_list[i]}_trunc_{trunc_list[i]}.npy")

# %%
def plot_stress_components(
    erate,
    K_list,
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

    linestyle=['--','-']

    for j in range(len(K_list)):
        time_mean_stress= time_mean_stress_data_array_all_K[j]
        time_std_stress=time_std_stress_data_array_all_K[j]
        for i in range(*i_range):
            ax.errorbar(
                erate,
                time_mean_stress[:, i],
                yerr=time_std_stress[:, i],
                label=rf"{stress_columns[i]}, K={K_list[j]}",
                capsize=3,
                marker='o',
                linestyle="none",
                linewidth=1.5
            )
            ax.grid(True, linestyle='--', alpha=0.3)

            if fit_type and i == fit_index:
                x = np.array(erate)
                y = time_mean_stress[:, i]

                if fit_points is not None:
                    x = x[fit_points]
                    y = y[fit_points]

                if fit_type == 'linear':
                    # Least squares fit through origin: y = m*x
                    m = np.dot(x, y) / np.dot(x, x)
                    fit_y = m * np.array(erate)
                    label = rf"Linear fit: {stress_columns[i]} $= {m:.3g} \dot{{\gamma}}, K={K_list[j]}$"

                elif fit_type == 'quadratic':
                    A = np.vstack([x**2, np.ones_like(x)]).T
                    a, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                    fit_y = a * np.array(erate)**2
                    label = rf"Quadratic fit: {stress_columns[i]} $= {a:.3g}\dot{{\gamma}}^{2}, K={K_list[j]}$"

                else:
                    raise ValueError("Invalid fit_type. Use 'linear', 'quadratic', or None.")

                ax.plot(
                    erate,
                    fit_y,
                    linestyle[j],
                    color='black',
                    linewidth=1,
                    label=label
                )

    ax.set_xlabel(rf"$\dot{{\gamma}}$")
    ax.legend(frameon=False)
    ax.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close()

quad_fit=[0,1,2,3,4]
plot_stress_components(erate ,K_list,time_mean_stress_data_array_all_K,time_std_stress_data_array_all_K, stress_columns, i_range=(7,9), fit_type="quadratic", fit_index=7,fit_points=quad_fit,save=True,
    save_path="plots/stress_components_"+str(7)+"_"+str(9)+".png")

plot_stress_components(erate,K_list,time_mean_stress_data_array_all_K,time_std_stress_data_array_all_K, stress_columns, i_range=(1,4),  fit_type=None, fit_index=1,fit_points=None,save=True,
     save_path="plots/stress_components_"+str(1)+"_"+str(4)+".png")


linear_fit=[0,1,2,3,4]
plot_stress_components(erate,K_list,time_mean_stress_data_array_all_K,time_std_stress_data_array_all_K, stress_columns, i_range=(4,7),  fit_type="linear", fit_index=4,fit_points=linear_fit, save=True,
    save_path="plots/stress_components_"+str(4)+"_"+str(7)+".png")
# %% now looking at the orientation distributions only plates

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

def convert_cart_2_spherical_y_inc_plate_from_dump(vel_pos_array, n_mols, output_cutoff):
    position_array = vel_pos_array[:, :, :3]
    print("position_array shape:", position_array.shape)

    # Reshape into (timesteps, plates, 3 vertices, 3 coords)
    position_plates_array = np.reshape(position_array, (output_cutoff, n_mols, 6, 3))

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

    spherical_coords_array = np.zeros((output_cutoff, n_mols, 3))

    # r: radial magnitude
    spherical_coords_array[:, :, 0] = np.sqrt(x**2 + y**2 + z**2)

    # theta: azimuthal angle in XZ plane (from +X to +Z)
    spherical_coords_array[:, :, 1] = np.arctan2(z, x)

    # phi: polar angle from +Y axis down
    spherical_coords_array[:, :, 2] = np.arccos(y / spherical_coords_array[:, :, 0])

    return spherical_coords_array

n_mols=1688
erate_count = np.zeros(erate.size, dtype=int)
real_target=1
vel_pos_array = np.zeros((real_target, erate.size, shear_outs, n_mols*6,6 ))
area_vector_array = np.zeros((real_target, erate.size, shear_outs, n_mols,3 ))
vel_pos_dump_name_list=glob.glob("plateshearnvt*_hookean_flat_elastic_*K_"+str(K)+"*.dump")
erate_file_name_index=15

for file in vel_pos_dump_name_list:

    # Extract metadata
    file_meta_data = file.split("_")
    print(file_meta_data)

   
    #plate
    erate_file = round(float(file_meta_data[erate_file_name_index]), 7)
    erate_index = int(np.where(erate == erate_file)[0])
    print(erate_index)
    
    # if real_index >= real_target:
    #     continue  # skip if real_index exceeds target

    if erate_count[erate_index] >= real_target:
        continue

    data=read_lammps_posvel_dump_to_numpy(file)

    if data is None:
        continue
    print(data.shape[0])
    if data.shape[0] <shear_outs:
        continue

    else:

        real_index=erate_count[erate_index]

        erate_count[erate_index] += 1
        print(erate[erate_index])
        print(f"Realisation: {real_index}")

    print(erate_count)

    #idx = list(range(2, 5)) + list(range(8,11))
    vel_pos_array[real_index,erate_index]=data[:shear_outs,:,5:]
    area_vector_array[real_index,erate_index]=convert_cart_2_spherical_y_inc_plate_from_dump(vel_pos_array[real_index,erate_index],n_mols,shear_outs)

print(erate_count)

#%%
def plot_spherical_kde_plate_from_numpy_DB(
    spherical_coords_array,
    erate,
    K,
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
    ax_rho.set_title(rf"$\rho$ Distribution, K={K}")
    ax_rho.grid(True, linestyle='--', alpha=0.7)
    ax_rho.legend()

    # Format THETA plot
    ax_theta.set_xlabel(r"$\Theta$")
    ax_theta.set_ylabel("Density")
    ax_theta.set_title(rf"$\Theta$ Distribution, K={K}")
    ax_theta.set_xticks(pi_theta_ticks)
    ax_theta.set_xticklabels(pi_theta_labels)
    ax_theta.set_xlim(-np.pi, np.pi)
    ax_theta.grid(True, linestyle='--', alpha=0.7)
    ax_theta.legend()

    # Format PHI plot
    ax_phi.set_xlabel(r"$\phi$")
    ax_phi.set_ylabel("Density")
    ax_phi.set_title(rf"$\phi$ Distribution, K={K}")
    ax_phi.set_xticks(pi_phi_ticks)
    ax_phi.set_xticklabels(pi_phi_labels)
    ax_phi.set_xlim(0, np.pi / 2)
    ax_phi.grid(True, linestyle='--', alpha=0.7)
    ax_phi.legend()

    # Save if requested
    if save:
        fig_rho.savefig(f"{save_dir}/rho_kdeK_{K}.png", dpi=300)
        fig_theta.savefig(f"{save_dir}/theta_kde_{K}.png", dpi=300)
        fig_phi.savefig(f"{save_dir}/phi_kde_{K}.png", dpi=300)

    plt.show()
    plt.close('all')

plot_spherical_kde_plate_from_numpy_DB( area_vector_array, erate,K, 400, save=True, selected_erate_indices=[0,1,2,3,4])


#%% plotting theta phi scatter

def plot_theta_vs_phi_scatter(
    spherical_coords_array,
    erate,
    cutoff,
    selected_erate_indices,
    K,
    save=False,
    save_dir="plots",
    use_latex=True
):
    """
    Scatter plots of theta vs phi for selected erate indices.

    Parameters:
        spherical_coords_array: np.ndarray [samples, erates, time, particles, coords(3)]
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

    for j in selected_erate_indices:
        i = int(j)

        # Extract data
        theta = spherical_coords_array[:, i, cutoff:, :, 1]
        theta = np.ravel(theta)
        phi = spherical_coords_array[:, i, cutoff:, :, 2]
        phi = np.ravel(phi)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(theta, phi, alpha=0.3, s=0.05)

        ax.set_xlabel(r"$\Theta$")
        ax.set_ylabel(r"$\phi$")
        ax.set_title(rf"$\dot{{\gamma}} = {erate[i]:.1e}, K={K}$")

        ax.set_xticks(pi_theta_ticks)
        ax.set_xticklabels(pi_theta_labels)
        ax.set_xlim(-np.pi, np.pi)

        ax.set_yticks(pi_phi_ticks)
        ax.set_yticklabels(pi_phi_labels)
        ax.set_ylim(0, np.pi / 2)

        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        if save:
            fname = f"{save_dir}/theta_vs_phi_{K}_erate_{i}.png"
            fig.savefig(fname, dpi=300)
            print(f"Saved: {fname}")

        plt.show()
        plt.close()

plot_theta_vs_phi_scatter(
    area_vector_array,
    erate,
    400,
    [0,1,2,3,4],
    K,
    save=True,
    save_dir="plots/scatter"
)

#%% now looking at dump files of velocity and position 
#pos_vel_dump_name_list
output_cutoff=700
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

pos_vel_dump_array=np.zeros((real_target,erate.size,output_cutoff,n_mols*6,6))
erate_count = np.zeros(erate.size, dtype=int)
erate_file_name_index=15
real_target=1
for file in pos_vel_dump_name_list:
    file_meta_data = file.split("_")
    print(file_meta_data)

    

    # Extract metadata
    
    file_meta_data = file.split("_")
    print(file_meta_data)
    erate_file = round(float(file_meta_data[erate_file_name_index]), 7)
    erate_index = int(np.where(erate == erate_file)[0])
    print(erate_index)
   
    if erate_count[erate_index] >= real_target:
        continue


    dump_data = read_lammps_posvel_dump_to_numpy(file)
    
    if dump_data is None:
        continue
    print("n_outs",dump_data.shape[0])
    if dump_data.shape[0] <output_cutoff:
        continue

    
    else:

        real_index=erate_count[erate_index]

        erate_count[erate_index] += 1
        print(erate[erate_index])
        print(f"Realisation: {real_index}")
        idx = list(range(2, 5)) + list(range(8,11))
        pos_vel_dump_array[real_index,erate_index]=dump_data[:output_cutoff,:,idx]

print(erate_count)


#%% plotting vx against z to check velocity profiles 

# will just truncate and take time average 
#x y z vx vy vz
erate_skip_array=[0,10,20,29]
for j in range(len(erate_skip_array)):
    i=erate_skip_array[j]
    v_x=np.ravel(pos_vel_dump_array[:,i,:,:,3])
    r_y=np.ravel(pos_vel_dump_array[:,i,:,:,2])

    m, _ = np.polyfit(r_y, v_x, 1)
    fit_line = m * r_y

    plt.figure(figsize=(6, 4))
    plt.scatter(r_y, v_x, alpha=0.6, label="Data")
    plt.plot(r_y, fit_line, color='red', label=fr"Fit: $v_x = {m:.5f} \cdot r_y$")
    
    plt.xlabel(r"$r_y$")
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
    Plots v_x vs r_y with linear best-fit line (v_x = m * r_y) for selected erate indices.

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
        # Flatten v_x and r_y
        v_x = np.ravel(pos_vel_dump_array[:, idx, :, :, 3])
        r_y = np.ravel(pos_vel_dump_array[:, idx, :, :, 1])

        # Linear fit: v_x = m * r_y
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

plot_vx_vs_rz_with_fit(pos_vel_dump_array, erate,save=True, selected_erate_indices=[0,5,9])
# %%
# now need to look at rotation velocity of dumbells , could look at the rate of rotation of the rho vector , from the spherical coordinates
# perhaps there is a way to compute eigen values, to look at the rotation 