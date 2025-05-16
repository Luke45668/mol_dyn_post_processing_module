#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/tchain_5_tdam_100_rsl_5_strain_mass_1/"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_0.5_erate_0.05_1_strain_20/"
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/plate_tests"
#Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/plate_test_weak_hydro"
#Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/chain_tests"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_25_T_1_sllod_wi/"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_500_sllod_wi"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_mass_10_stiff_0.005_1_1_sllod_100_strain_T_0.01_R_1_R_n_1_N_864/logs_and_stress"
os.chdir(Path_2_log)


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

    N1 = sxx - szz
    N2 = szz - syy

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
        print("mean_shear stress",np.mean(sxz[-50:]))
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
    sxx = -sxx_sum / volume
    syy = -syy_sum / volume
    szz = -szz_sum / volume
    sxy = -sxy_sum / volume
    sxz = -sxz_sum / volume
    syz = -syz_sum / volume

    N1 = sxx - szz
    N2 = szz - syy

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
        plt.plot(time_plot, N1_plot, label=r'$N_1 = \sigma_{xx} - \sigma_{zz}$')
        plt.plot(time_plot, N2_plot, label=r'$N_2 = \sigma_{zz} - \sigma_{yy}$')
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
#%%

def read_lammps_log(filename='log.lammps'):
    """
    Read a LAMMPS log file and return a pandas DataFrame of the thermo output.

    Parameters:
    -----------
    filename : str
        Path to the LAMMPS log file (default: 'log.lammps').

    Returns:
    --------
    df : pandas.DataFrame
        Thermo data as a DataFrame (columns = thermo keywords).
    """

    with open(filename, 'r') as file:
        lines = file.readlines()

    thermo_data = []
    thermo_headers = []
    reading_thermo = False

    for line in lines:
        # Check if the line is a header (starts with Step)
        if re.match(r'\s*Step\s+', line):
            thermo_headers = line.strip().split()
            reading_thermo = True
            continue

        # Check if reading thermo data
        if reading_thermo:
            # If line is empty or non-numeric, stop reading
            if not line.strip() or not re.match(r'^[\s\d\.\-Ee]+$', line):
                reading_thermo = False
                continue
            # Otherwise, parse data line
            data_line = [float(x) for x in line.strip().split()]
            thermo_data.append(data_line)

    if not thermo_data:
        raise RuntimeError("No thermo data found in the log file.")

    df = pd.DataFrame(thermo_data, columns=thermo_headers)
    return df

#

df = read_lammps_log(filename='log.DB_minimal_shear_T_0.01_K_100_mass_0.1_R_n_0.01_R_0.0005_erate_0.0003_tstep_6.00000250e-04')

# See the available thermo columns
print(df.columns)
# Plot temperature vs time
df=df.tail(300)
plt.plot(df['Step'], df['c_spring_pe'])
plt.xlabel('Time (Step)')
plt.ylabel('c_spring_pe')
plt.grid(True)
plt.title('Pe vs Time')
plt.show()
# %%
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
file_name="eq_plateeqnvt_no988576_hookean_flat_elastic_mass_10_R_n_1_R_0.5_927734_4_150_0.035_3e-5_29700_29747_297470420_0_gdot_0.06723357536499335_BK_500_K_0.1.dump"
file_name="eq_plateeqnvt_no988576_hookean_flat_elastic_mass_10_R_n_1_R_0.5_927734_4_150_1_3e-5_29700_29747_297470420_0_gdot_0.06723357536499335_BK_500_K_0.1.dump"
#file_name="eq_plateeqnvt_no988576_hookean_flat_elastic_mass_10_R_n_1_R_0.5_927734_4_150_5_3e-5_29700_29747_297470420_0_gdot_0.06723357536499335_BK_500_K_0.1.dump"
#file_name="eq_plateeqnvt_no988576_hookean_flat_elastic_mass_10_R_n_1_R_0.5_927734_4_150_10_3e-5_29700_29747_297470420_0_gdot_0.06723357536499335_BK_500_K_0.1.dump"
vel_pos_array=read_lammps_posvel_dump_to_numpy(file_name)
output_cutoff=vel_pos_array.shape[0]
n_mols=1688


def convert_cart_2_spherical_z_inc_plate_from_dump_single_file(vel_pos_array,n_mols,output_cutoff):      
        
        position_array=vel_pos_array[:,:,2:5]
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
area_vector_array=convert_cart_2_spherical_z_inc_plate_from_dump_single_file(vel_pos_array,n_mols,output_cutoff)

def plot_spherical_kde_plate_selected_single_file(area_vector_array, selected_timesteps, save=False, save_dir="plots", use_latex=True):
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
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    colors = sns.color_palette("Set1", len(selected_timesteps))

    
    
    for j in range(len(selected_timesteps)):
    # Prepare data
        rho = np.ravel(area_vector_array[ selected_timesteps[j], :, 0])
        theta = area_vector_array[ selected_timesteps[j], :, 1]
        theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
        phi = area_vector_array[ selected_timesteps[j], :, 2]
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
    fig.suptitle("Spherical Distributions for Selected Timesteps", fontsize=16)

    if save:
        fig.savefig(f"{save_dir}/selected_timesteps_spherical_distributions.png", dpi=300)

    plt.show()
    plt.close('all')

plot_spherical_kde_plate_selected_single_file(area_vector_array, selected_timesteps=[25,50,75,100,130])

# %%
file_name="stress_tensor_avg_DBshearnvtall1_no988576_db_hooke_tensorlgeq_T_0.01_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dat"
file_name="stress_tensor_avg_DBshearnvtall1_no988576_db_hooke_tensorlgeq_T_0.1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dat"
#file_name="stress_tensor_avg_DBshearnvtall1_no988576_db_hooke_tensorlgeq_T_1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dat"
file_name="stress_tensor_avg_mass_10_R_n_1_R_0.1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dat"
#file_name="stress_tensor_avgang_mass_10_R_n_1_R_0.1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dat"
#file_name="phantomstress_tensor_avgang_mass_10_R_n_1_R_0.1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dat"
data=analyze_raw_stress_data(filename=file_name, volume=150**3, show_plots=True, return_data=True)

# %%
file_name="DBshearnvtall1_no988576_db_hooke_tensor_T_0.01_m_10_R_0.1_Rn_1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dump"
file_name="DBshearnvtall1_no988576_db_hooke_tensor_T_0.1_m_10_R_0.1_Rn_1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_BK_50_K_1.dump"
n_mols=1688



erate=1
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

        # #  theta coord
        # spherical_coords_array[ :, 1] = np.sign(y) * np.arccos(
        #     x / (np.sqrt((x**2) + (y**2)))
        # )

        spherical_coords_array[ :, 1]=np.arctan2(y, x)

        # spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
        # spherical_coords_array[:,:,:,1]=np.arctan(y/x)

        # phi coord
        # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
        spherical_coords_array[ :, 2] = np.arccos(
            z / np.sqrt((x**2) + (y**2) + (z**2))
        )

        return spherical_coords_array

dump_data=read_lammps_dump_tensor(file_name)
spherical_coords_array=np.zeros((len(dump_data),n_mols,3))
for i in range(len(dump_data)):
    spherical_coords_array[i]=convert_cart_2_spherical_z_inc_DB_from_dict(dump_data[i]['data'],n_mols)

print(len(dump_data))


#%%

def plot_spherical_kde_plate_from_numpy_DB_single(
    spherical_coords_array,
    start,
    finish,
    erate,
    save=False,
    save_dir="plots",
    use_latex=True,
    
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
    long_phi_ticks=[0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2,5 * np.pi / 8,3 * np.pi / 4,7 * np.pi / 8,np.pi]
    long_pi_phi_labels = [r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$",r"$5\pi/8$",r"$3\pi/4$",r"$7\pi/8$",r"$\pi$"]

    # Prepare figures
    fig_rho, ax_rho = plt.subplots(figsize=(8, 4))
    fig_theta, ax_theta = plt.subplots(figsize=(8, 4))
    fig_phi, ax_phi = plt.subplots(figsize=(8, 4))
    fig_phi_theta, ax_phi_theta = plt.subplots(figsize=(8, 4))
    fig_theta_rho,ax_theta_rho=plt.subplots(figsize=(8, 4))
    #for j in range(len(selected_erate_indices)):
    i = 0
    
    # Extract data
    rho = np.ravel(spherical_coords_array[ start:finish, :, 0])
    theta = spherical_coords_array[start:finish, :, 1]
    theta_scatter=np.ravel(theta)
    theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
    phi = spherical_coords_array[start:finish, :, 2]
   # phi = np.ravel(np.array([phi, np.pi - phi]))
    phi = np.ravel(phi)

    rho_mean = np.mean(rho)
    label_rho = rf"$\dot{{\gamma}} = {erate:.1e}$, $\langle \rho \rangle = {rho_mean:.2f}$"
    label = rf"$\dot{{\gamma}} = {erate:.1e}$"

    # Plot KDEs
    sns.kdeplot(rho, label=label_rho, ax=ax_rho, linewidth=2,bw_adjust=0.5)
    sns.kdeplot(theta, label=label, ax=ax_theta, linewidth=2,bw_adjust=0.5)
    sns.kdeplot(phi, label=label, ax=ax_phi, linewidth=2,bw_adjust=0.5)
    print("mean phi ", np.mean(phi))
    

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
    #ax_phi.set_xlim(0, np.pi / 2)
    ax_phi.set_xlim(0, np.pi)
    ax_phi.grid(True, linestyle='--', alpha=0.7)
    ax_phi.legend()

    ax_phi_theta.scatter(theta_scatter,phi,s=0.005)
    ax_phi_theta.set_ylabel(r"$\phi$")
    ax_phi_theta.set_xlabel(r"$\Theta$")
    ax_phi_theta.set_title(r"$\phi,\theta$ Scatter")
    ax_phi_theta.set_yticks(long_phi_ticks)
    ax_phi_theta.set_yticklabels(long_pi_phi_labels)
    ax_phi_theta.set_xticks(pi_theta_ticks)
    ax_phi_theta.set_xticklabels(pi_theta_labels)
    ax_phi_theta.set_xlim(-np.pi, np.pi)
    #ax_phi.set_xlim(0, np.pi / 2)
    ax_phi_theta.set_ylim(0, np.pi)
    ax_phi_theta.grid(True, linestyle='--', alpha=0.7)

    ax_theta_rho.scatter(theta_scatter,rho,s=0.0005)
    ax_theta_rho.set_xticks(pi_theta_ticks)
    ax_theta_rho.set_xticklabels(pi_theta_labels)
    ax_theta_rho.set_xlim(-np.pi, np.pi)
    ax_theta_rho.set_ylabel(r"$\rho$")
    ax_theta_rho.set_xlabel(r"$\Theta$")

   


    # Save if requested
    if save:
        fig_rho.savefig(f"{save_dir}/rho_kde.png", dpi=300)
        fig_theta.savefig(f"{save_dir}/theta_kde.png", dpi=300)
        fig_phi.savefig(f"{save_dir}/phi_kde.png", dpi=300)

    plt.show()
    plt.close('all')



plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,0,50,erate, save=False)

# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,950,1000,erate, save=False)

# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,1000,1100,erate, save=False)

# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,1200,1300,erate, save=False)

plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,50,100,erate, save=False)

#plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,100,200,erate, save=False)
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,100,200,erate, save=False)
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,200,300,erate, save=False)##
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,300,400,erate, save=False)##

plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,1000,1300,erate, save=False)##
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,1300,1600,erate, save=False)


# %% inspecting log files 

log_file_name="log.DBshearnvtall1_no988576_hookean_dumb_bell_T_0.01_m_10_R_0.1_Rn_1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_K_1"
log_file_name="log.DBshearnvtall1_no988576_hookean_dumb_bell_T_0.1_m_10_R_0.1_Rn_1_934_4_150_1_3e-7_29700_29747_2000000000_0_gdot_1_K_1"
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
for col in columns:
    plt.figure()
    plt.plot(thermo_data[0]['Step'], thermo_data[0][col], label=col)
    plt.xlabel('Step')
    plt.ylabel(col)
    plt.title(f'{col} vs Step')
    plt.legend()
    plt.tight_layout()
    plt.show()

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
