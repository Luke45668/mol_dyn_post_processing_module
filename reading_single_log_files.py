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
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/sliplink_tests/"
#Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/sliplink_eq_tests/"
#Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/sliplink_exten_tests/"
#Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/edge_bonds_tests"
#Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/sliplink_angle_constrain_tests/"
#Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/plate_test_phantom_thermo"
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


file_name="flow_flat_elastic_0.3_0.1_K_1.dump"
file_name="flow_flat_elastic_0.1_0.1_K_1.dump"
#file_name="sliplink_0.3_0.1_K_1.dump"
K=1
# file_name="flow_flat_elastic_0.3_0.1_K_0.5.dump"
file_name="flow_flat_elastic_0.1_0.1_K_0.5.dump"
K=0.5
#file_name="flow_flat_elastic_0.3_0.1_K_0.25.dump"
# K=0.25
#file_name="flow_flat_elastic_0.3_0.1_K_2.dump"
#file_name="flow_flat_elastic_0.3_0.1_K_5.dump"
#file_name="sliplink_0.3_0.1_K_1.dump"
# file_name="sliplink_0.3_0.1_K_0.5.dump"

vel_pos_array=read_lammps_posvel_dump_to_numpy(file_name)
output_cutoff=vel_pos_array.shape[0]
n_mols=1688
n_mols=250


def convert_cart_2_spherical_z_inc_plate_from_dump_single_file(vel_pos_array,n_mols,output_cutoff):      
        
        position_array=vel_pos_array[:,:,5:8]
        print(position_array.shape)
        #reshape into number of plates

        position_plates_array=np.reshape(position_array,(output_cutoff,n_mols,6,3))

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

def convert_cart_2_spherical_y_up_plate_from_dump_single_file(vel_pos_array, n_mols, output_cutoff):
    # Extract position array
    position_array = vel_pos_array[:, :, 5:8]
    print("position_array shape:", position_array.shape)

    # Reshape to separate plates
    position_plates_array = np.reshape(position_array, (output_cutoff, n_mols, 6, 3))

    # Define edge vectors from atom 0
    ell_1 = position_plates_array[:, :, 1] - position_plates_array[:, :, 0]
    ell_2 = position_plates_array[:, :, 2] - position_plates_array[:, :, 0]

    print("ell_1 shape", ell_1.shape)
    print("ell_2 shape", ell_2.shape)

    # Compute area vectors (normal to plate)
    area_vector = np.cross(ell_1, ell_2, axis=-1)
    print("area_vector shape", area_vector.shape)

    # Ensure area vectors point into the positive Y hemisphere
    area_vector[area_vector[:, :, 1] < 0] *= -1

    x = area_vector[:, :, 0]
    y = area_vector[:, :, 1]
    z = area_vector[:, :, 2]

    spherical_coords_array = np.zeros((output_cutoff, n_mols, 3))

    # r: radial magnitude
    spherical_coords_array[:, :, 0] = np.sqrt(x**2 + y**2 + z**2)

    # theta: azimuthal angle in XZ plane (from +X toward +Z)
    spherical_coords_array[:, :, 1] = np.arctan2(z, x)

    # phi: polar angle from +Y down
    spherical_coords_array[:, :, 2] = np.arccos(y / spherical_coords_array[:, :, 0])

    return spherical_coords_array
#area_vector_array=convert_cart_2_spherical_y_up_plate_from_dump_single_file(vel_pos_array,n_mols,output_cutoff)

def convert_cart_2_spherical_y_up_plate_from_dump_single_file_3_stokes_only(K,vel_pos_array, n_mols, output_cutoff):
    # Extract position array
    position_array = vel_pos_array[:, :, 5:8]
    force_array=vel_pos_array[:, :, 2:5]
    print("position_array shape:", position_array.shape)

    # Reshape to separate plates
    position_plates_array = np.reshape(position_array, (output_cutoff, n_mols, 3, 3))

    # Define edge vectors from atom 0
    ell_1 = position_plates_array[:, :, 1] - position_plates_array[:, :, 0]
    ell_2 = position_plates_array[:, :, 2] - position_plates_array[:, :, 0]

    ell2minusell1=ell_2-ell_1

    #magnitudes 
    mag_ell_1_sqrd=np.sqrt(np.sum(ell_1**2,axis=2))
    mag_ell_2_sqrd=np.sqrt(np.sum(ell_2**2,axis=2))
    mag_ell2minusell1=np.sqrt(np.sum(ell2minusell1**2,axis=2))


    print("ell_1 shape", ell_1.shape)
    print("ell_2 shape", ell_2.shape)

    # Compute area vectors (normal to plate)
    area_vector = np.cross(ell_1, ell_2, axis=-1)
    print("area_vector shape", area_vector.shape)
    area_vector_magnitude=np.sqrt(np.sum(area_vector**2, axis=2))

    sin_theta=area_vector_magnitude/(mag_ell_1_sqrd*mag_ell_2_sqrd)

    theta_from_cross=np.arcsin(sin_theta)

    # Compute dot products and magnitudes
    dot12 = np.sum(ell_1 * ell_2, axis=2)
    l1_sq = np.sum(ell_1**2, axis=2)
    l2_sq = np.sum(ell_2**2, axis=2)

    # Compute projections
    proj_1_on_2 = (dot12[:, :, np.newaxis] / l2_sq[:, :, np.newaxis]) * ell_2
    proj_2_on_1 = (dot12[:, :, np.newaxis] / l1_sq[:, :, np.newaxis]) * ell_1


    # Compute perpendicular components
    perp_1 = ell_1 - proj_1_on_2
    perp_2 = ell_2 - proj_2_on_1

    # Compute forces, this needs K variable 
    F1 = -K * l2_sq[:, :, np.newaxis] * perp_1
    F2 = -K * l1_sq[:, :, np.newaxis] * perp_2
    F0 = -(F1 + F2)

    # Magnitudes
    F1_mag = np.linalg.norm(F1, axis=2)
    F2_mag = np.linalg.norm(F2, axis=2)
    F0_mag = np.linalg.norm(F0, axis=2)


    # Ensure area vectors point into the positive Y hemisphere
    area_vector[area_vector[:, :, 1] < 0] *= -1

    x = area_vector[:, :, 0]
    y = area_vector[:, :, 1]
    z = area_vector[:, :, 2]

    spherical_coords_array = np.zeros((output_cutoff, n_mols, 3))


    # r: radial magnitude
    spherical_coords_array[:, :, 0] = np.sqrt(x**2 + y**2 + z**2)

    # theta: azimuthal angle in XZ plane (from +X toward +Z)
    spherical_coords_array[:, :, 1] = np.arctan2(z, x)

    # phi: polar angle from +Y down
    spherical_coords_array[:, :, 2] = np.arccos(y / spherical_coords_array[:, :, 0])




    return spherical_coords_array, ell_1,ell_2,theta_from_cross,mag_ell_1_sqrd,mag_ell_2_sqrd,sin_theta,F0_mag,F1_mag,F2_mag, mag_ell2minusell1, force_array,F0,F1,F2
    

area_vector_array, ell_1,ell_2,theta_from_cross,mag_ell_1_sqrd,mag_ell_2_sqrd,sin_theta,F0_mag,F1_mag,F2_mag,mag_ell2minusell1,force_array,F0,F1,F2=convert_cart_2_spherical_y_up_plate_from_dump_single_file_3_stokes_only(K,vel_pos_array,n_mols,output_cutoff)


def convert_forces_to_spherical_coords(force_array,n_mols):

    force_array=np.reshape(force_array,(force_array.shape[0],n_mols,3,3))

    spherical_force_coords_array = np.zeros((force_array.shape[0], n_mols,3, 3))

    x = force_array[:, :,:, 0]
    y = force_array[:, :,:, 1]
    z = force_array[:, :,:, 2]

     # r: radial magnitude
    spherical_force_coords_array[:,:, :, 0] = np.sqrt(x**2 + y**2 + z**2)

    # theta: azimuthal angle in XZ plane (from +X toward +Z)
    spherical_force_coords_array[:,:, :, 1] = np.arctan2(z, x)

    # phi: polar angle from +Y down
    spherical_force_coords_array[:,:, :, 2] = np.arccos(y / spherical_force_coords_array[:, :,:, 0])

    return spherical_force_coords_array



spherical_force_coords_array = convert_forces_to_spherical_coords(force_array,n_mols)



def plot_spherical_kde_forces_plate_selected_single_file(spherical_force_coords_array, selected_timesteps, save=False, save_dir="plots", use_latex=True):
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
        rho = np.ravel(spherical_force_coords_array[ selected_timesteps[j],:, :, 0])
        theta = spherical_force_coords_array[ selected_timesteps[j],:, :, 1]
        theta = np.ravel(np.array([theta - 2 * np.pi, theta, theta + 2 * np.pi]))
        phi = spherical_force_coords_array[ selected_timesteps[j],:, :, 2]
        phi = np.ravel(np.array([phi, np.pi - phi]))

        # --- RHO ---
        sns.kdeplot(rho, color=colors[j], linewidth=2, ax=axes[0], label=f"output {selected_timesteps[j]}")
        print("min rho value", np.min(rho))
        print("max rho value", np.max(rho))

        # --- THETA ---
        sns.kdeplot(theta, color=colors[j], linewidth=2, ax=axes[1], label=f"output {selected_timesteps[j]}")

        # --- PHI ---
        sns.kdeplot(phi, color=colors[j], linewidth=2, ax=axes[2], label=f"output {selected_timesteps[j]}")

        # Rho plot
    axes[0].set_xlabel(r"$f_{\rho}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\rho$ Distribution")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Theta plot
    axes[1].set_xlabel(r"$f_{\Theta}$")
    axes[1].set_ylabel("Density")
    axes[1].set_xticks(pi_theta_ticks)
    axes[1].set_xticklabels(pi_theta_labels)
    axes[1].set_xlim(-np.pi, np.pi)
    axes[1].set_title(r"$\Theta$ Distribution")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Phi plot
    axes[2].set_xlabel(r"$f_{\phi}$")
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
        print("min rho value", np.min(rho))
        print("max rho value", np.max(rho))

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



def extract_triangle_sides_angles(vel_pos_array, n_mols, output_cutoff):
    # Extract position array
    position_array = vel_pos_array[:, :, 5:8]
    position_plates_array = np.reshape(position_array, (output_cutoff, n_mols, 3, 3))

    # Extract triangle vertices
    A = position_plates_array[:, :, 0]
    B = position_plates_array[:, :, 1]
    C = position_plates_array[:, :, 2]

    # Side vectors
    AB = B - A
    BC = C - B
    CA = A - C

    

    # Side lengths
    a = np.linalg.norm(BC, axis=2)  # opposite to A
    b = np.linalg.norm(CA, axis=2)  # opposite to B
    c = np.linalg.norm(AB, axis=2)  # opposite to C

    # Use cosine rule to find angles
    # To avoid divide-by-zero issues, clip dot product terms
    angle_A = np.arccos(np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0))
    angle_B = np.arccos(np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0))
    angle_C = np.arccos(np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0))

    # Return side lengths and angles (in radians)
    return a, b, c, angle_A, angle_B, angle_C

a, b, c, angle_A, angle_B, angle_C= extract_triangle_sides_angles(vel_pos_array, n_mols, output_cutoff)
#%%

#
plot_spherical_kde_plate_selected_single_file(area_vector_array,selected_timesteps=[0,500,1000,1300,1666])

#plot_spherical_kde_forces_plate_selected_single_file(spherical_force_coords_array, selected_timesteps=[0,500,1000,1300,1666], save=False, save_dir="plots", use_latex=True)


#%% plotting internal angle distributions 
# need to check the distributions have the same number of data points in each , so they can be compared
pi_theta_ticks = [ 0, np.pi / 4, np.pi / 2,3* np.pi / 4, np.pi]
pi_theta_labels =[r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
gdot=0.3
file_name=f"flow_flat_elastic_{gdot}_0.1_K_1.dump"
vel_pos_array=read_lammps_posvel_dump_to_numpy(file_name)[:1500]
output_cutoff=vel_pos_array.shape[0]
print(output_cutoff)
K=1
a_1, b_1, c_1, angle_A_1, angle_B_1, angle_C_1= extract_triangle_sides_angles(vel_pos_array, n_mols, output_cutoff)


file_name=f"flow_flat_elastic_{gdot}_0.1_K_0.5.dump"
K=0.5 
vel_pos_array=read_lammps_posvel_dump_to_numpy(file_name)[:1500]
output_cutoff=vel_pos_array.shape[0]
print(output_cutoff)
K=1
a_05, b_05, c_05, angle_A_05, angle_B_05, angle_C_05= extract_triangle_sides_angles(vel_pos_array, n_mols, output_cutoff)
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
A_05=np.array([np.ravel(angle_A_05)-np.pi,np.ravel(angle_A_05),np.ravel(angle_A_05)+np.pi])
A_1=np.array([np.ravel(angle_A_1)-np.pi,np.ravel(angle_A_1),np.ravel(angle_A_1)+np.pi])
sns.kdeplot(np.ravel(A_1),ax=axes[0],label="$K=1$")
sns.kdeplot(np.ravel(A_05),ax=axes[0],label="$K=0.5$")
axes[0].set_xlim(0,np.pi)
axes[0].set_xlabel("$\\angle A$")
axes[0].legend()
axes[0].set_xticks(pi_theta_ticks)
axes[0].set_xticklabels(pi_theta_labels)


B_05=np.array([np.ravel(angle_B_05)-np.pi,np.ravel(angle_B_05),np.ravel(angle_B_05)+np.pi])
B_1=np.array([np.ravel(angle_B_1)-np.pi,np.ravel(angle_B_1),np.ravel(angle_B_1)+np.pi])
sns.kdeplot(np.ravel(B_1),ax=axes[1],label="$K=1$")
sns.kdeplot(np.ravel(B_05),ax=axes[1],label="$K=0.5$")
axes[1].set_xlim(0,np.pi)
axes[1].set_xlabel("$\\angle B$")
axes[1].legend()
axes[1].set_xticks(pi_theta_ticks)
axes[1].set_xticklabels(pi_theta_labels)


C_05=np.array([np.ravel(angle_C_05)-np.pi,np.ravel(angle_C_05),np.ravel(angle_C_05)+np.pi])
C_1=np.array([np.ravel(angle_C_1)-np.pi,np.ravel(angle_C_1),np.ravel(angle_C_1)+np.pi])
sns.kdeplot(np.ravel(C_1),ax=axes[2],label="$K=1$")
sns.kdeplot(np.ravel(C_05),ax=axes[2],label="$K=0.5$")
axes[2].set_xlim(0,np.pi)
axes[2].set_xlabel("$\\angle C$")
axes[2].legend()
axes[2].set_xticks(pi_theta_ticks)
axes[2].set_xticklabels(pi_theta_labels)
fig.suptitle(f"$\dot{{\gamma}}={gdot}$", fontsize=16)
fig.subplots_adjust(hspace=0.4, top=0.95)
plt.savefig(f"internal_angle_plots_gdot_{gdot}.png",dpi=1200)
plt.show()

#%% plotting ell vectors squared vs sin theta ^2
#80,87
E=0.5*K* (mag_ell_1_sqrd**2)* (mag_ell_2_sqrd**2) *(sin_theta**2)
mag_ells=(mag_ell_1_sqrd**2)* (mag_ell_2_sqrd**2)
sin2theta=sin_theta**2 
theta=np.arcsin(sin_theta)
area=0.5*mag_ell_1_sqrd*mag_ell_2_sqrd*sin_theta
Instability_index=((mag_ell_1_sqrd+mag_ell_2_sqrd+mag_ell2minusell1)**2)/area
#Instability_index=mag_ells/sin2theta
colinearity_metric=(mag_ell_1_sqrd*mag_ell_2_sqrd)/np.sqrt(area)
all_sides=np.array([a,b,c])
aspect_ratio=np.max(all_sides,axis=0)/np.min(all_sides,axis=0)


# K=5
# start=137
# end=140
# K=2
# start=149
# end=152
K=1 # 0.3 gdot 
start=176
end=179
K=1 # 0.1 gdot 
start=189
end=192

# #K=0.5 gdot 0.3
# start=190
# end=193

K=0.5 #gdot 0.1
start=145
end=148


# #K=0.25
# start=19
# end=22
# start=7
# end=10

E_ang_A=0.5*K* (c**2)* (b**2) *(np.sin(angle_A)**2)
E_ang_B=0.5*K* (c**2)* (a**2) *(np.sin(angle_B)**2)
E_ang_C=0.5*K* (a**2)* (b**2) *(np.sin(angle_C)**2)
# I_ang_A=(c**2)* (b**2)/np.sin(angle_A)**2
# I_ang_B=(c**2)* (a**2)/(np.sin(angle_B)**2)
# I_ang_C=(a**2)* (b**2) /(np.sin(angle_C)**2)
I=((a+b+c)**2)/(0.5*a*b*np.sin(angle_C))

fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
for i in range(start,end):
    timesteps=np.linspace(0,a[:,i].size*3000,a[:,i].size)


# Plot side lengths

    axs[0].plot(timesteps, a[:, i], label=f"side a, Particle {i+1}")
    axs[0].plot(timesteps, b[:, i], label=f"side b, Particle {i+1}")
    axs[0].plot(timesteps, c[:, i], label=f"side c, Particle {i+1}")
    axs[0].set_ylabel("Side Lengths")
    axs[0].legend(bbox_to_anchor=(1,1))

    # Plot instability index and angle energy

    axs[1].plot(timesteps, I[:, i], label=f"$I$, Particle {i+1}")
    axs[1].plot(timesteps, E_ang_A[:, i], label=f"$E_{{angA}}$, Particle {i+1}")
    
    axs[1].set_ylabel("Instability Index / Energy")
    axs[1].set_yscale("log")
    

    # Plot angles
   
    axs[2].plot(timesteps, angle_A[:, i], label=f"angle A, Particle {i+1}")
    axs[2].plot(timesteps, angle_B[:, i], label=f"angle B, Particle {i+1}")
    axs[2].plot(timesteps, angle_C[:, i], label=f"angle C, Particle {i+1}")
    axs[2].set_xlabel("Timestep")
    axs[2].set_ylabel("Angles (rad)")
    axs[2].legend(bbox_to_anchor=(1,1))
axs[1].axhline(10**3, linestyle='dashed', color='gray', label="$I_{bound}$")
axs[1].legend(bbox_to_anchor=(1,1))
plt.tight_layout()
base_name = os.path.splitext(file_name)[0]
save_name = f"Instability_data_{base_name}_highlight.png"

# Save the plot
plt.savefig(save_name, dpi=300)
plt.show()

#%% spring force 

force_array=np.reshape(force_array,(force_array.shape[0],250,3,3))


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Select the particle index and timesteps
particle_index = 176
selected_timesteps = [0,500,1000,1250,1500,1741]  # K=1 shear run 
particle_index=190
selected_timesteps = [0,500,1000,1250,1500,1666]  # K=0.5 shear run 

#selected_timesteps = [0,100,200,300,400,500]  # K=1 eq run 
#particle_index = 82
#selected_timesteps = [0,100,200,241]
# Reshape positions (assuming vel_pos_array already loaded)
positions = np.reshape(vel_pos_array[:, :, 5:8], (vel_pos_array.shape[0], n_mols, 3, 3))
position_0 = positions[:, particle_index, 0]
position_1 = positions[:, particle_index, 1]
position_2 = positions[:, particle_index, 2]

forces_on_0 = F0[:, particle_index, :]
forces_on_1 = F1[:, particle_index, :]
forces_on_2 = F2[:, particle_index, :]

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

def check_in_plane(f0, f1, f2, pos0, pos1, pos2, tol=1e-6):
    ell1 = pos1 - pos0
    ell2 = pos2 - pos0
    normal = np.cross(ell1, ell2)
    norm_mag = np.linalg.norm(normal)
    if norm_mag == 0:
        return False, 0, 0, 0, normal

    normal /= norm_mag
    dp0 = np.abs(np.dot(f0, normal))
    dp1 = np.abs(np.dot(f1, normal))
    dp2 = np.abs(np.dot(f2, normal))

    in_plane = dp0 < tol and dp1 < tol and dp2 < tol
    return in_plane, dp0, dp1, dp2, normal

def compute_force_dot_products_with_cosines(f0, f1, f2):
    def cosine_angle(fa, fb):
        dot = np.dot(fa, fb)
        norm_product = np.linalg.norm(fa) * np.linalg.norm(fb)
        if norm_product == 0:
            return dot, np.nan
        return dot, dot / norm_product

    dot01, cos01 = cosine_angle(f0, f1)
    dot02, cos02 = cosine_angle(f0, f2)
    dot12, cos12 = cosine_angle(f1, f2)
    return (dot01, cos01), (dot02, cos02), (dot12, cos12)

def plot_with_dot_products(t, pos0, pos1, pos2, f0, f1, f2, particle_index):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    triangle_vertices = [pos0, pos1, pos2]
    forces = [f0, f1, f2]
    positions = [pos0, pos1, pos2]
    colors = ['red', 'green', 'blue']

    # Plot forces with magnitudes in legend
    for i, (label, position, force, color) in enumerate(zip(['0', '1', '2'], positions, forces, colors)):
        magnitude = np.linalg.norm(force)
        ax.quiver(*position, *force, length=2, color=color, normalize=True,
                  label=f"Spring Force {label} , |F|={magnitude}")
        ax.scatter(*position, color=color, edgecolor='k', s=30)

    triangle = Poly3DCollection([triangle_vertices], color='gray', alpha=0.2)
    ax.add_collection3d(triangle)

    in_plane, dp0, dp1, dp2, _ = check_in_plane(f0, f1, f2, pos0, pos1, pos2)
    (dot01, cos01), (dot02, cos02), (dot12, cos12) = compute_force_dot_products_with_cosines(f0, f1, f2)

    ax.set_title(
        f"Particle {particle_index}, Timestep {t} | In-plane: {in_plane}\n"
        f"Dot(f0,f1): {dot01:.2e}, cosθ: {cos01:.2f} | "
        f"Dot(f0,f2): {dot02:.2e}, cosθ: {cos02:.2f} | "
        f"Dot(f1,f2): {dot12:.2e}, cosθ: {cos12:.2f}"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

for t in selected_timesteps:
   plot_with_dot_products(
        t,
        position_0[t], position_1[t], position_2[t],
        forces_on_0[t], forces_on_1[t], forces_on_2[t],
        particle_index
    )

#%% checking deviation from ell vectors 
# neeed to fix this 

def compute_mean_perp_distance_from_ell_vectors(vel_pos_array, n_mols, output_cutoff):
    # Extract position array
    position_array = vel_pos_array[:, :, 5:8]
    print("position_array shape:", position_array.shape)

    # Reshape to separate plates
    position_plates_array = np.reshape(position_array, (output_cutoff, n_mols, 6, 3))

    # Define edge vectors from atom 0
    ell_1 = position_plates_array[:, :, 1] - position_plates_array[:, :, 0]
    ell_2 = position_plates_array[:, :, 2] - position_plates_array[:, :, 0]
    ell_3=  position_plates_array[:, :, 2]-position_plates_array[:, :, 1]

    phantom_1_2_stokes_1_vector=position_plates_array[:, :, 3]-position_plates_array[:, :, 0]
    phantom_2_2_stokes_2_vector=position_plates_array[:, :, 4]-position_plates_array[:, :, 1]
    phantom_3_2_stokes_3_vector=position_plates_array[:, :, 5]-position_plates_array[:, :, 2]

    dot_ell_1_phantom_1_vector=np.sum(ell_1*phantom_1_2_stokes_1_vector,axis=2)**2
    dot_ell_3_phantom_2_vector=np.sum(ell_3*phantom_2_2_stokes_2_vector,axis=2)**2
    dot_ell_2_phantom_3_vector=np.sum(-ell_2*phantom_3_2_stokes_3_vector,axis=2)**2

    mag_phantom_1_2_stokes_1_vector=np.sum(phantom_1_2_stokes_1_vector**2,axis=2)
    mag_phantom_2_2_stokes_2_vector=np.sum(phantom_2_2_stokes_2_vector**2,axis=2)
    mag_phantom_3_2_stokes_3_vector=np.sum(phantom_3_2_stokes_3_vector**2,axis=2)
    print(mag_phantom_1_2_stokes_1_vector-dot_ell_1_phantom_1_vector)

    distance_phantom_1_2_ell_1=np.sqrt(mag_phantom_1_2_stokes_1_vector-dot_ell_1_phantom_1_vector)
    distance_phantom_2_2_ell_3=np.sqrt(mag_phantom_2_2_stokes_2_vector-dot_ell_3_phantom_2_vector)
    distance_phantom_3_2_ell_2=np.sqrt(mag_phantom_3_2_stokes_3_vector- dot_ell_2_phantom_3_vector)

    return distance_phantom_1_2_ell_1, distance_phantom_2_2_ell_3, distance_phantom_3_2_ell_2
    
distance_phantom_1_2_ell_1, distance_phantom_2_2_ell_3, distance_phantom_3_2_ell_2= compute_mean_perp_distance_from_ell_vectors(vel_pos_array, n_mols, output_cutoff)



# %%


file_name="stress_tensor_allavg_0.3_0.1_K_2.dat"

data= analyze_raw_stress_data_after_n_steps(500,filename=file_name, volume=100**3, show_plots=True, return_data=True,ylim=(-0.001,0.001))
# plt.plot(data["N2"][500:50000])
# plt.show()
# plt.plot(data["N1"][500:50000])
# plt.show()
plt.plot(np.abs(data["N1"][360000:]/data["N2"][360000:]))
plt.show()

# print(np.mean(np.abs(data["N1"][18000:]/data["N2"][18000:])))
# %%
file_name="DBshearlang_no988576_db_hooke_tensor_T_1_m_1_R_0.1_Rn_1_9354_4_150_0.1_3e-5_29700_29747_2000000000_0_gdot_0.1_BK_50_K_0.5.dump"
file_name="DBshearlang_no988576_db_hooke_tensor_T_1_m_1_R_0.1_Rn_1_934_4_150_0.1_3e-5_29700_29747_2000000000_0_gdot_0.5_BK_50_K_0.5.dump"
#file_name="DBshearlang_no988576_db_hooke_tensor_T_1_m_100_R_0.1_Rn_1_934_4_150_1_3e-5_29700_29747_2000000000_0_gdot_0.01_BK_50_K_0.2.dump"


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
        
        #spring_vector_ray[spring_vector_ray[ :, 2] < 0] *= -1

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

def convert_cart_2_spherical_y_inc_xz_theta(spring_vector_ray, n_mols):
    x = spring_vector_ray[:, 0]
    y = spring_vector_ray[:, 1]
    z = spring_vector_ray[:, 2]

    spherical_coords_array = np.zeros((n_mols, 3))

    # r: radial distance
    spherical_coords_array[:, 0] = np.sqrt(x**2 + y**2 + z**2)

    # theta: angle in xz-plane from x-axis (azimuthal)
    spherical_coords_array[:, 1] = np.arctan2(z, x)

    # phi: inclination from y-axis
    spherical_coords_array[:, 2] = np.arccos(y / spherical_coords_array[:, 0])

    return spherical_coords_array

dump_data=read_lammps_dump_tensor(file_name)
spherical_coords_array=np.zeros((len(dump_data),n_mols,3))
for i in range(len(dump_data)):
    spherical_coords_array[i]=convert_cart_2_spherical_y_inc_xz_theta(dump_data[i]['data'],n_mols)

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




plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,0,20,erate, save=False)
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,20,40,erate, save=False)
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,60,80,erate, save=False)
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,80,100,erate, save=False)
# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,0,100,erate, save=False)
plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,100,200,erate, save=False)
# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,200,300,erate, save=False)
# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,400,500,erate, save=False)
# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,500,600,erate, save=False)

# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,0,500,erate, save=False)
# # plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,150,200,erate, save=False)

#plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,500,1000,erate, save=False)##


# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,1000,1200,erate, save=False)


# plot_spherical_kde_plate_from_numpy_DB_single( spherical_coords_array,1200,1400,erate, save=False)


# %% inspecting log files 


log_file_name="log.lammps_0.3_0.1_K_5"
# log_file_name="log.lammps_0.3_0.1_K_2"
# log_file_name="log.lammps_0.3_0.1_K_1"
# log_file_name="log.lammps_0.3_0.1_K_0.5"
# log_file_name="log.lammps_0.3_0.1_K_0.25"
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
