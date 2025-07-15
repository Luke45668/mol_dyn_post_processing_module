#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/plate_test_phantom_thermo"

os.chdir(Path_2_log)





# %% inspecting log files 


log_file_name="log.plateshearnvt_no988576_hookean_flat_elastic_mass_1_R_n_1_R_0.1_934_4_150_0.1_1e-5_29700_29747_2000000000_0_gdot_0.1_BK_1000_K_0.5"

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

thermo_data=read_lammps_log_incomplete(log_file_name)
columns=thermo_data[0].columns

# plotting shear run total energy 
col='TotEng'
plt.figure()
plt.plot(thermo_data[1]['Step'], thermo_data[1][col], label=col)
plt.xlabel('Step')
plt.ylabel(col)
plt.title(f'{col} vs Step')
plt.legend()
plt.tight_layout()
plt.show()

#%% all thermo 

for col in columns:

    plt.figure()
    plt.plot(thermo_data[0]['Step'], thermo_data[0][col], label=col)
    plt.xlabel('Step')
    plt.ylabel(col)
    plt.title(f'{col} vs Step')
    plt.legend()
    plt.tight_layout()
    plt.show()
    if col=='c_msd[4]':
       gradient= np.mean(np.gradient(thermo_data[0][col][20:50]))
       diffusivity= (1/6)*gradient
       print(diffusivity)

#%%
# # shearing data 
columns=thermo_data[1].columns
for col in columns:
    plt.figure()
    plt.plot(thermo_data[1]['Step'], thermo_data[1][col], label=col)
    plt.xlabel('Step')
    plt.ylabel(col)
    plt.title(f'{col} vs Step')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# %%
def analyze_raw_stress_data(filename='stress_tensor_avg.dat', volume=None, show_plots=True, return_data=False):
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

    if show_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(time, sxx, label=r'$\sigma_{xx}$')
        plt.plot(time, syy, label=r'$\sigma_{yy}$')
        plt.plot(time, szz, label=r'$\sigma_{zz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normalized Stress')
        plt.legend()
        plt.grid(True)
        #plt.yscale('log')
        plt.title('Normal Stress Components')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time, sxy, label=r'$\sigma_{xy}$')
        plt.plot(time, sxz, label=r'$\sigma_{xz}$')
        plt.plot(time, syz, label=r'$\sigma_{yz}$')
        plt.xlabel('Time (timesteps)')
        print("mean_shear stress",np.mean(sxy[-500:]))
        plt.ylabel('Normalized Shear Stress')
        plt.legend()
        
        plt.grid(True)
        plt.title('Shear Stress Components')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time, N1, label=r'$N_1 = \sigma_{xx} - \sigma_{zz}$')
        
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Differences')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
       
        plt.plot(time, N2, label=r'$N_2 = \sigma_{zz} - \sigma_{yy}$')
        
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Differences')
        plt.tight_layout()
        plt.show()

        

    if return_data:
        return {
            'time': time,
            'sxx': sxx, 'syy': syy, 'szz': szz,
            'sxy': sxy, 'sxz': sxz, 'syz': syz,
            'N1': N1, 'N2': N2
        }
    
def analyze_raw_stress_data_after_n_steps(truncate,filename='stress_tensor_avg.dat', volume=None, show_plots=True, return_data=False):
    """
    Analyze LAMMPS time-averaged global stress (unnormalized by volume).
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

    # Mask for time > 500
    mask = time >= truncate
    time_plot = time[mask]
    sxx_plot = sxx[mask]
    syy_plot = syy[mask]
    szz_plot = szz[mask]
    sxy_plot = sxy[mask]
    sxz_plot = sxz[mask]
    syz_plot = syz[mask]
    N1_plot = N1[mask]
    N2_plot = N2[mask]

    if show_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, sxx_plot, label=r'$\sigma_{xx}$')
        plt.plot(time_plot, syy_plot, label=r'$\sigma_{yy}$')
        plt.plot(time_plot, szz_plot, label=r'$\sigma_{zz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normalized Stress')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Components (after '+str(mask)+' steps)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, sxy_plot, label=r'$\sigma_{xy}$')
        plt.plot(time_plot, sxz_plot, label=r'$\sigma_{xz}$')
        plt.plot(time_plot, syz_plot, label=r'$\sigma_{yz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normalized Shear Stress')
        plt.legend()
        plt.grid(True)
        plt.title('Shear Stress Components (after 500 steps)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, N1_plot, label=r'$N_1 = \sigma_{xx} - \sigma_{yy}$')
        plt.plot(time_plot, N2_plot, label=r'$N_2 = \sigma_{yy} - \sigma_{zz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Differences (after 500 steps)')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, N1_plot/N2_plot, label=r'$N_1/N_2$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Ratio (after 500 steps)')
        plt.tight_layout()
        plt.show()

    if return_data:
        return {
            'time': time,
            'sxx': sxx, 'syy': syy, 'szz': szz,
            'sxy': sxy, 'sxz': sxz, 'syz': syz,
            'N1': N1, 'N2': N2
        }
    
def analyze_raw_stress_data_after_n_steps(
    truncate,
    filename='stress_tensor_avg.dat',
    volume=None,
    show_plots=True,
    return_data=False,
    ylim=None
):
    """
    Analyze LAMMPS time-averaged global stress (unnormalized by volume).

    Parameters:
        truncate (float): Time threshold for analysis.
        filename (str): File with stress tensor data.
        volume (float): Simulation box volume to normalize stress.
        show_plots (bool): Whether to display plots.
        return_data (bool): Whether to return data.
        xlim (tuple): Tuple of (xmin, xmax) for x-axis in all plots.
    """
    if volume is None:
        raise ValueError("You must specify the box volume to normalize stress!")

    data = np.loadtxt(filename, comments='#')
    time, sxx_sum, syy_sum, szz_sum, sxy_sum, sxz_sum, syz_sum = data.T

    sxx = sxx_sum / volume
    syy = syy_sum / volume
    szz = szz_sum / volume
    sxy = sxy_sum / volume
    sxz = sxz_sum / volume
    syz = syz_sum / volume

    N1 = sxx - syy
    N2 = syy - szz

    mask = time >= truncate
    time_plot = time[mask]
    sxx_plot = sxx[mask]
    syy_plot = syy[mask]
    szz_plot = szz[mask]
    sxy_plot = sxy[mask]
    sxz_plot = sxz[mask]
    syz_plot = syz[mask]
    N1_plot = N1[mask]
    N2_plot = N2[mask]

    if show_plots:
        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, sxx_plot, label=r'$\sigma_{xx}$')
        plt.plot(time_plot, syy_plot, label=r'$\sigma_{yy}$')
        plt.plot(time_plot, szz_plot, label=r'$\sigma_{zz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normalized Stress')
        plt.legend()
        plt.grid(True)
        plt.title(f'Normal Stress Components (after {truncate} steps)')
        if ylim: plt.ylim(ylim)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, sxy_plot, label=r'$\sigma_{xy}$')
        plt.plot(time_plot, sxz_plot, label=r'$\sigma_{xz}$')
        plt.plot(time_plot, syz_plot, label=r'$\sigma_{yz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normalized Shear Stress')
        plt.legend()
        plt.grid(True)
        plt.title(f'Shear Stress Components (after {truncate} steps)')
        if ylim: plt.ylim(ylim)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, N1_plot, label=r'$N_1 = \sigma_{xx} - \sigma_{yy}$')
        plt.plot(time_plot, N2_plot, label=r'$N_2 = \sigma_{yy} - \sigma_{zz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title(f'Normal Stress Differences (after {truncate} steps)')
        if ylim: plt.ylim(ylim)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time_plot, N1_plot / N2_plot, label=r'$N_1/N_2$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normal Stress Ratio')
        plt.legend()
        plt.grid(True)
        plt.title(f'Normal Stress Ratio (after {truncate} steps)')
        if ylim: plt.ylim(ylim)
        plt.tight_layout()
        plt.show()

    if return_data:
        return {
            'time': time,
            'sxx': sxx, 'syy': syy, 'szz': szz,
            'sxy': sxy, 'sxz': sxz, 'syz': syz,
            'N1': N1, 'N2': N2
        }






file_name="stress_tensor_allBOND_mass_1_R_n_1_R_0.1_934_4_150_0.1_1e-5_29700_29747_2000000000_0_gdot_0.1_BK_1000_K_0.5.dat"

data= analyze_raw_stress_data_after_n_steps(0,filename=file_name, volume=150**3, show_plots=True, return_data=True)
file_name="stress_tensor_allANG_mass_1_R_n_1_R_0.1_934_4_150_0.1_1e-5_29700_29747_2000000000_0_gdot_0.1_BK_1000_K_0.5.dat"

data_ang= analyze_raw_stress_data_after_n_steps(0,filename=file_name, volume=150**3, show_plots=True, return_data=True)
file_name="stress_tensor_allKE_mass_1_R_n_1_R_0.1_934_4_150_0.1_1e-5_29700_29747_2000000000_0_gdot_0.1_BK_1000_K_0.5.dat"

data_KE= analyze_raw_stress_data_after_n_steps(0,filename=file_name, volume=150**3, show_plots=True, return_data=True)

def plot_combined_stress(data_bond,data_ang,data_ke,component_string):
    plt.plot(data_bond[component_string],label="spring")
    plt.plot(data_ang[component_string],label="angle")
    plt.plot(data_bond[component_string]+data_ang[component_string],label="combined")
    plt.title(component_string)
    plt.legend()
    plt.show()

def plot_combined_stress_ratio(data_bond,data_ang,data_ke,component_string):
    plt.plot(data_bond[component_string]/data_ang[component_string],label="spring/angle")
    #plt.plot(data_ang[component_string],label="angle")
    #plt.plot(data_ke[component_string],label="ke")
    plt.title(component_string)
    plt.legend()
    plt.show()

plot_combined_stress(data,data_ang,data_KE,"sxx")
plot_combined_stress(data,data_ang,data_KE,"syy")
plot_combined_stress(data,data_ang,data_KE,"szz")
plot_combined_stress(data,data_ang,data_KE,"N1")
plot_combined_stress(data,data_ang,data_KE,"N2")
plot_combined_stress_ratio(data,data_ang,data_KE,"N2")
# %%
