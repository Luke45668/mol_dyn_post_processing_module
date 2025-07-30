#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
n_mol=1
file="sliplink_run0nothermo_0.1_1.dump"
n_mol=100
file="sliplink_run0nothermo_0.1_100.dump"
pos_vel_array=read_lammps_posvel_dump_to_numpy(file)[:,:,5:]
pos_vel_array=np.reshape(pos_vel_array,(1,n_mol, 3, 6))
forces_from_lammps=np.round(read_lammps_posvel_dump_to_numpy(file)[:,:,2:5],3)
#%%
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
log_file='log.lammps_runthermo_0.1_1000'
thermo_data=read_lammps_log_incomplete(log_file)
columns=thermo_data[0].columns
labels=["Step","$E_{K}$","$E_{P}$","$T$","$E_{t}$"]
n_outs=len(thermo_data[0]['Step'][:])
n_mol=250
save_dir=Path_2_log
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
plt.figure(figsize=(8, 5))
for col, label in zip(columns, labels):
    plt.figure()
    plt.plot(thermo_data[0]['Step'], thermo_data[0][col], label=label)
    
    cutoff = int(n_outs / 2)
    last_50_percent_grad = np.abs(np.mean(np.gradient(thermo_data[0][col][cutoff:])))
    print(last_50_percent_grad)
    print()
    
    
    
    # Create a custom (invisible) legend entry for the gradient
    grad_label = rf"Grad$={last_50_percent_grad:.3e}$"
    grad_proxy = Line2D([0], [0], color='none', label=grad_label)

    # Add both the actual line and the custom gradient label to the legend
    plot_line = Line2D([0], [0], color='blue', label=label)  # Or get actual plot line color


    if col == "TotEng":
        Energy_drift = (thermo_data[0][col][n_outs-1] - thermo_data[0][col][0]) / (thermo_data[0][col].size * 1000 * n_mol)
        print(Energy_drift)
        drift_label= Line2D([0], [0], color='none', label=rf"Drift=${Energy_drift:.3e}$")
        
        plt.legend(handles=[plot_line, grad_proxy,drift_label])
        

    else:
        plt.legend(handles=[plot_line, grad_proxy])

    plt.xlabel('$N_{t}$')
    #plt.ylabel(label, rotation=0)
    plt.title(f'{label}')
    plt.tight_layout()
    fname = f"{save_dir}/{col}.png"
    plt.savefig(fname, dpi=1200)
    plt.show()
    
#%% eq stress data

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
        plt.xlabel('$N_{t}$')
        #plt.ylabel('Normalized Stress')
        plt.legend()
        plt.grid(True)
        #plt.yscale('log')
        plt.title('Normal Stress Components')
        plt.tight_layout()
        plt.savefig("normal_stress.png", dpi=1200)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time, sxy, label=r'$\sigma_{xy}$')
        plt.plot(time, sxz, label=r'$\sigma_{xz}$')
        plt.plot(time, syz, label=r'$\sigma_{yz}$')
        plt.xlabel('$N_{t}$')
        print("mean_shear stress",np.mean(sxy[-500:]))
        #plt.ylabel('Normalized Shear Stress')
        plt.legend()
        
        plt.grid(True)
        plt.title('Shear Stress Components')
        plt.tight_layout()
        plt.savefig("shear_stress.png", dpi=1200)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time, N1, label=r'$N_1 = \sigma_{xx} - \sigma_{zz}$')
        
        plt.xlabel('$N_{t}$')
        #plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Differences')
        plt.tight_layout()
        plt.savefig("N1.png", dpi=1200)
        plt.show()

        plt.figure(figsize=(8, 5))
       
        plt.plot(time, N2, label=r'$N_2 = \sigma_{zz} - \sigma_{yy}$')
        
        plt.xlabel('$N_{t}$')
        #plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Differences')
        plt.tight_layout()
        plt.savefig("N2.png", dpi=1200)
        plt.show()

        

    if return_data:
        return {
            'time': time,
            'sxx': sxx, 'syy': syy, 'szz': szz,
            'sxy': sxy, 'sxz': sxz, 'syz': syz,
            'N1': N1, 'N2': N2
        }
analyze_raw_stress_data(filename='eq_stress_tensor_avg_runthermo_0.1_1000.dat', volume=100**3, show_plots=True, return_data=False)

#%% analytically compute trajectories from initial conditions 
K=1



# create arrays to store computations 
file="sliplink_runthermo_0.1_250.dump"
pos_vel_array=read_lammps_posvel_dump_to_numpy(file)[:,:,5:]
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
