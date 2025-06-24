#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/sliplink_validations"
deltat=1e-6
os.chdir(Path_2_log)
def compute_forces(pos_vels, K):
    r0 = pos_vels[0, 0, :3]
    r1 = pos_vels[0, 1, :3]
    r2 = pos_vels[0, 2, :3]

    l1 = r1 - r0
    l2 = r2 - r0

    l1_sq = np.dot(l1, l1)
    l2_sq = np.dot(l2, l2)
    dot12 = np.dot(l1, l2)

    F1 = -K * l2_sq * (l1 - (dot12 / l2_sq) * l2)
    F2 = -K * l1_sq * (l2 - (dot12 / l1_sq) * l1)
    F0 = -(F1 + F2)

    forces=np.array([F0, F1, F2])

    return forces
def compute_new_position(pos_vels,forces):


    # positions from previous step
        R_t=pos_vels[:,:,0:3]

        # velocities from previous_step

        v_t= pos_vels[:,:,3:6]

        R_t_plus_dt= R_t+ v_t*deltat + (forces/2*mass)*(deltat**2)
        print("timestep"+str(i)+", new positions", R_t_plus_dt)

        return R_t_plus_dt,v_t 
def compute_new_vel(v_t, forces, forces_t_plus_dt, mass, deltat):
       v_t_plus_dt = v_t + 0.5 * (forces + forces_t_plus_dt) * (deltat / mass)
       print("timestep"+str(i)+", new vel", v_t_plus_dt)
       return v_t_plus_dt
#%% import dump file 
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

from sigfig import round 

# timestep 0 test 
K=1
file="sliplink_0.3_0.1_1.dump"
pos_vel_array=read_lammps_posvel_dump_to_numpy(file)[:,:,5:]
pos_vel_array=np.reshape(pos_vel_array,(1,1, 3, 6))
forces_from_lammps=np.round(read_lammps_posvel_dump_to_numpy(file)[:,:,2:5],3)

print(forces_from_lammps)
forces_from_positions=np.round(compute_forces(pos_vel_array[0], K),3)
print(forces_from_positions)
forces_from_positions==forces_from_lammps

force_free_check=np.sum(forces_from_lammps,axis=1)
print(force_free_check)


#%% import log data 
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
log_file='log.lammps_runthermo_0.1_250'
thermo_data=read_lammps_log_incomplete(log_file)
columns=thermo_data[0].columns
for col in columns:

    plt.figure()
    plt.plot(thermo_data[0]['Step'][:], thermo_data[0][col][:], label=col)
    if col == "TotEng":
        Energy_drift= (thermo_data[0][col][thermo_data[0][col].size-1]- thermo_data[0][col][0])/(thermo_data[0][col].size*1000*250)
        print(Energy_drift)
    plt.xlabel('Step')
    plt.ylabel(col)
    plt.title(f'{col} vs Step')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

#%% analytically compute trajectories from initial conditions 
K=1



# create arrays to store computations 


R_t_plus_dt_list=[]
v_t_plus_dt_list=[]
v_t_list=[]

for i in range(0,50):#pos_vel_array.shape[0]
#for i in range(287122,287124):
    pos_vels=pos_vel_array[i]
    #print(pos_vels)
    forces= compute_forces(pos_vels,K)
    print("forces",forces)


    # compute t+dt position 
    mass=1
   

    R_t_plus_dt ,v_t=compute_new_position(pos_vels,forces)

    R_t_plus_dt_list.append(R_t_plus_dt)
    v_t_list.append(v_t)
        

    # Reconstruct pos_vels format at t+dt with velocities set to zero or old v_t

    #pos_vels_t_plus_dt = np.zeros_like(pos_vels)
    #pos_vels_t_plus_dt[:, :, 0:3] = R_t_plus_dt
    #pos_vels_t_plus_dt[:, :, 3:6] = v_t  # or use np.zeros if you want to exclude velocity influence
    # compute t+dt force 
    forces_t_plus_dt = compute_forces(R_t_plus_dt, K)
    print("forces_t_plus_dt",forces_t_plus_dt)

   
     


    # compute new velocity t +dt 

   

    v_t_plus_dt=compute_new_vel(v_t, forces, forces_t_plus_dt, mass, deltat)
    v_t_plus_dt_list.append(v_t_plus_dt)






# %% comparing predicted to actual positions 
timestep_rt_check_list=[]
timestep_vt_check_list=[]
for i in range(1,50):

    pos_vels=pos_vel_array[i]
    actual_pos=pos_vels[:,:,0:3]
    actual_vel=pos_vels[:,:,3:6]

    predicted_pos= R_t_plus_dt_list[i-1]
    predicted_vel= v_t_plus_dt_list[i-1]

    Bool_array_pos=np.isclose(actual_pos,predicted_pos,rtol=1e-8)
    Bool_array_vel=np.isclose(actual_vel,predicted_vel,rtol=1e-8)


    if np.all(Bool_array_pos==1):
       timestep_rt_check_list.append(1)

    if np.all(Bool_array_vel==1):
       timestep_vt_check_list.append(1)


print("Successful position predictions",len(timestep_rt_check_list))
print("Successful velocity predictions", len(timestep_vt_check_list))

print("Max velocity diff at step", i, ":", np.max(np.abs(actual_vel - predicted_vel)))
print("Max position diff at step", i, ":", np.max(np.abs(actual_pos - predicted_pos)))

 
   




# %%
