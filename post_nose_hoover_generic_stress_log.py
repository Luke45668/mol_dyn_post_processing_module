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
path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_tstep_0.0005_mass_10_stiff_0.005_1_1_sllod_25_strain_T_0.01_R_1_R_n_1_N_864/logs_and_stress/"
vol=120**3
eq_outs=801

os.chdir(path_2_files)
K = 0.1
mass=10
n_shear_points=30
log_name_list = glob.glob("log*_K_"+str(K))

erate=np.logspace(-2.5, -1,n_shear_points)
erate=np.round(erate,7)
spring_relaxation_time=np.sqrt(mass/K)
Wi=erate*spring_relaxation_time
reals=5

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


erate_count=np.zeros(erate.size).astype('int')
eq_log_data_array=np.zeros((reals,erate.size,eq_outs,7))
shear_log_data_array=np.zeros((reals,erate.size,1000,11))
for file in log_name_list:

    data=read_lammps_log_if_complete(file)

    if data==None:
        continue
        
    else:
        file_meta_data=file.split("_")
        print(file_meta_data)
        erate_file=file_meta_data[21]
        erate_file=round(float(erate_file), 7)
        erate_index=int(np.where(erate==erate_file)[0])
        erate_count[erate_index]+=1
        print(erate[erate_index])  
        real_index=int(file_meta_data[12])
        print(real_index)
        eq_log_data_array_raw=data[0].to_numpy()
        print(eq_log_data_array_raw.shape)
        shear_log_data_array_raw=data[1].to_numpy() 
        print(shear_log_data_array_raw.shape)
        eq_log_data_array[real_index,erate_index]=eq_log_data_array_raw
        shear_log_data_array[real_index,erate_index]=shear_log_data_array_raw[:1000]


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
            '$\sigma_{xx}': sxx, '$\sigma_{yy}$': syy, '$\sigma_{zz}$': szz,
            '$\sigma_{xy}$': sxy, '$\sigma_{xz}$': sxz, '$\sigma_{yz}$': syz,
            '$N_{1}$': N1, '$N_{2}$': N2
        }
    
stress_name_list=glob.glob("stress*K_"+str(K)+".dat")
data_dict=read_stress_tensor_file(filename=stress_name_list[0], volume=vol, return_data=True)
stress_columns=list(data_dict.keys())
stress_array=np.zeros((reals,erate.size,999,9))
for file in stress_name_list:
    data_dict=read_stress_tensor_file(filename=file, volume=vol, return_data=True)
    if data==None:
        continue
        
    else:
        file_meta_data=file.split("_")
        print(file_meta_data)
        erate_file=file_meta_data[18]
        erate_file=round(float(erate_file), 7)
        erate_index=int(np.where(erate==erate_file)[0])
        erate_count[erate_index]+=1
        print(erate[erate_index])  
        real_index=int(file_meta_data[9])
        print(real_index)
        column=0
        for keys in data_dict:
            
            raw_stress_array=data_dict[keys]
            stress_array[real_index,erate_index,:,column]=raw_stress_array[:999]
            column+=1


        # raw_stress_array=array = np.column_stack([data_dict[key] for key in data])
        # print(raw_stress_array.shape)

        
mean_stress_array=np.mean(stress_array,axis=0)     



#%%

# now realisation average 

mean_shear_log_data_array=np.mean(shear_log_data_array,axis=0)
mean_eq_log_data_array=np.mean(eq_log_data_array,axis=0)

print(mean_shear_log_data_array.shape)
print(mean_eq_log_data_array.shape)
            

def plot_time_series(data, erate, column_names):
    """
    data: shape (30, 1000, 11)
    erate: length-30 array of shear rates
    column_names: list of 11 log column names
    """
    n_shear, n_steps, n_cols = data.shape

    for i in range(n_shear):
        fig, axs = plt.subplots(n_cols, 1, figsize=(10, 2.5 * n_cols), sharex=True)
        fig.suptitle(f"Shear rate = {erate[i]:.7f}", fontsize=14)

        for j in range(n_cols):
            axs[j].plot(data[i, :, j])
            axs[j].set_ylabel(column_names[j])
            axs[j].grid(True)

        axs[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

plot_time_series(mean_shear_log_data_array, erate,shear_columns)

plot_time_series(mean_eq_log_data_array,erate,eq_columns)


#%%

plot_time_series(mean_stress_array,erate,stress_columns)


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
        "text.usetex": "False",
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
