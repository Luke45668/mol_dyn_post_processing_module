#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import glob
import pandas as pd
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from plotting_module import *
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/plate_test_rotation_rate"

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


#%% 

# %%

file_name="angular_data_1.0.txt"
def parse_angmom_inertia_with_omega(filepath):
    """
    Parses a LAMMPS angular data file and returns arrays for angular momentum, inertia, and angular velocity.

    Parameters:
    filepath (str): Path to the angular data file.

    Returns:
    tuple: (angular_momentum, inertia_tensor, angular_velocity, timesteps, molecule_ids)
    """
    angmom_list = []
    inertia_list = []
    omega_list = []
    timestep_list = []
    molid_list = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if lines[i].strip() and not lines[i].startswith("#"):
            parts = lines[i].strip().split()
            if len(parts) == 2:
                timestep = int(parts[0])
                num_mols = int(parts[1])
                i += 1
                for _ in range(num_mols):
                    data_line = lines[i].strip().split()
                    mol_id = int(data_line[0])
                    angmom = np.array(list(map(float, data_line[1:4])))
                    I_tensor = np.array([
                        [float(data_line[4]), float(data_line[7]), float(data_line[8])],
                        [float(data_line[7]), float(data_line[5]), float(data_line[9])],
                        [float(data_line[8]), float(data_line[9]), float(data_line[6])]
                    ])
                    try:
                        omega = np.linalg.solve(I_tensor, angmom)
                    except np.linalg.LinAlgError:
                        omega = np.full(3, np.nan)
                    angmom_list.append(angmom)
                    inertia_list.append(I_tensor)
                    omega_list.append(omega)
                    timestep_list.append(timestep)
                    molid_list.append(mol_id)
                    i += 1
            else:
                i += 1
        else:
            i += 1

    return (
        np.array(angmom_list),
        np.array(inertia_list),
        np.array(omega_list),
        np.array(timestep_list),
        np.array(molid_list)
    )
# %%

file_name="angular_data_1.0.txt"
angmom, inertia, omega, times, molids = parse_angmom_inertia_with_omega(file_name)


n_outs=int(omega.shape[0]/1688)
omega_for_each_timestep=np.reshape(omega,(n_outs,1688,3))

# for i in range(n_outs):
#     sns.kdeplot(omega_for_each_timestep[i,:,0], label="mean:"+str(np.mean(omega_for_each_timestep[i,:,0])))
# plt.legend()
# plt.show()
# for i in range(n_outs):
#     sns.kdeplot(omega_for_each_timestep[i,:,1],label="mean:"+str(np.mean(omega_for_each_timestep[i,:,0])))
# plt.legend()
# plt.show()
K_1=[]
for i in range(n_outs):
   # sns.kdeplot(omega_for_each_timestep[i,:,2],label="mean:"+str(np.mean(omega_for_each_timestep[i,:,2])))
    K_1.append(np.mean(omega_for_each_timestep[i,:,2]))
plt.legend()
plt.show()


# %%

file_name="angular_data_0.5.txt"
angmom, inertia, omega, times, molids = parse_angmom_inertia_with_omega(file_name)


n_outs=int(omega.shape[0]/1688)
omega_for_each_timestep=np.reshape(omega,(n_outs,1688,3))

# for i in range(n_outs):
#     sns.kdeplot(omega_for_each_timestep[i,:,0], label="mean:"+str(np.mean(omega_for_each_timestep[i,:,0])))
# plt.legend()
# plt.show()
# for i in range(n_outs):
#     sns.kdeplot(omega_for_each_timestep[i,:,1],label="mean:"+str(np.mean(omega_for_each_timestep[i,:,0])))
# plt.legend()
# plt.show()
K_05=[]
for i in range(n_outs):
   # sns.kdeplot(omega_for_each_timestep[i,:,2],label="mean:"+str(np.mean(omega_for_each_timestep[i,:,2])))
    K_05.append(np.mean(omega_for_each_timestep[i,:,2]))
plt.legend()
plt.show()


# %%
plt.plot(K_05,label="0.5")
plt.plot(K_1,label="1.0")
plt.legend()
plt.show()

# %%
