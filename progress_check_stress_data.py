"""
This script allows me to look at the current stress signals of a simulation run by looking at tensor dump files.
"""

#%%
from file_manipulations_module import *
from lammps_file_readers_module import *
import numpy as np

# import matplotlib.pyplot as plt
from plotting_module import *
from calculations_module import *
from statistical_tests import *
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
import sigfig
import os
import glob 
#%% check file sizes, disgard ones that are too short 
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_100_sllod_Wi"
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_500_sllod_wi"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_25_T_1_sllod_wi/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_0.1_T_0.1_stiff_0.0125_strain_25/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_50_sllod_Wi_R_1_N_500/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_50_sllod_Wi_R_1.5_N_500/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_50_sllod_Wi_R_2_N_500/"
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_50_R_1_N_4000/"
# filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_50_R_1_N_4000_no_ramp/"
# filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_50_sllod_Wi_R_1_N_500_no_ramp/"
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_100_T_2_N_500/"
# filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_100_T_2_N_108/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_100_T_2_N_864/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0_1_strain_100_R_2.59_N_500_lang_eq/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0_1_strain_100_R_2_N_500_lang_eq/"
# filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0_1_strain_100_R_2_N_500/"
# filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0_1_strain_100_R_1.5_N_500/"
#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0_1_strain_100_R_2_N_500_nh_eq_tchain_10/"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.0001_0.01_strain_10_N_500"
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_mass_500_stiff_0.00025_0.0025_1_sllod_50_strain_T_0.001_R_2.59_N_500/"


#filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0_1_strain_100_R_1.5_N_500_nh_eq_tchain_10/"
erate=np.array([0, 0.05, 0.24, 0.43, 0.62, 0.81, 1.0  ])
erate=np.array([0.0001    , 0.00021544, 0.00046416, 0.001     , 0.00215443,
       0.00464159, 0.01      , 0.02154435, 0.04641589, 0.1      ])
erate=np.logspace(-4, -2,20)
os.chdir(filepath)
Wi=np.array([ 0, 0.14142136, 0.67882251, 1.21622366, 1.75362482, 2.29102597, 2.82842712 ])
K=0.0005#0.00025 #,0.0025,0.0005
number_of_particles_per_dump=1000
n_plates=500
box_size=100
strain=50
Path_2_dump=filepath
dump_name_list=glob.glob("*tensor*_K_"+str(K)+".dump")
print(len(dump_name_list))
min_outs=1000
dump_start_line="ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]"

labels_stress = np.array(
    [
        "$\sigma_{xx}$",
        "$\sigma_{yy}$",
        "$\sigma_{zz}$",
        "$\sigma_{xz}$",
        "$\sigma_{xy}$",
        "$\sigma_{yz}$",
    ]
)
failed_files=[]
passed_file_names=[]
passed_file_dumps=[]
real_target=1
# def check_file_sizes(erate,list,real_target,min_outs,n_plates,number_of_particles_per_dump):
#     count=np.zeros((erate.size)).astype("int")
#     count_failed=np.zeros((erate.size)).astype("int")
#     count_reals=np.zeros((erate.size)).astype("int")
#     dump_out_array=np.zeros((erate.size,real_target,min_outs,n_plates,6))
#     for file in list:
#         try:
#             split_name=file.split('_')
#             #erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
#             erate_file=np.around(float(split_name[15]),5)
#             #print(erate_file)
#             #erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
#             erate_ind=int(np.where(erate==erate_file)[0][0])
#             # realisation_ind=int(split_name[6])
#             # spring_stiff=int(split_name[17])

#             dump_data=dump2numpy_tensor_1tstep(dump_start_line,
#                       Path_2_dump,file,
#                       number_of_particles_per_dump,n_plates)

#             file_size_rows=dump_data.shape[0]

            
#             if count[erate_ind]==real_target:

#                     continue

#             elif file_size_rows<min_outs:
                    
#                     failed_files.append(file)
#                     count_failed[erate_ind]+=1
#                     continue

#             else:
#                     passed_file_names.append(file)
#                     #passed_file_dumps.append(dump_data)
#                     count[erate_ind]+=1
#                     realisation_ind=count_reals[erate_ind]
#                     dump_out_array[erate_ind,realisation_ind]=dump_data[:min_outs]
#                     count_reals[erate_ind] += 1
#                     print(count_reals)

                
#         except:
#             failed_files.append(file)
#             count_failed[erate_ind]+=1

#             continue


#     return count,count_failed, dump_out_array

def check_file_sizes(erate, file_list, real_target, min_outs, n_plates, number_of_particles_per_dump):
    # Create a mapping from erate value to index for fast lookup
    erate_map = {round(val, 5): idx for idx, val in enumerate(erate)}
    
    count = np.zeros(len(erate), dtype=int)
    count_failed = np.zeros(len(erate), dtype=int)
    count_reals = np.zeros(len(erate), dtype=int)
    dump_out_array = np.zeros((len(erate), real_target, min_outs, n_plates, 6))

    for file in file_list:
        try:
            split_name = file.split('_')
            erate_val = round(float(split_name[15]), 5)
            erate_ind = erate_map.get(erate_val)

            if erate_ind is None:
                continue  # Skip if erate not found

            if count[erate_ind] >= real_target:
                continue

            dump_data = dump2numpy_tensor_1tstep(
                dump_start_line, Path_2_dump, file,
                number_of_particles_per_dump, n_plates
            )

            if dump_data.shape[0] < min_outs:
                failed_files.append(file)
                count_failed[erate_ind] += 1
                continue

            passed_file_names.append(file)
            realisation_ind = count_reals[erate_ind]
            dump_out_array[erate_ind, realisation_ind] = dump_data[:min_outs]
            count[erate_ind] += 1
            count_reals[erate_ind] += 1
            print(count_reals)

        except Exception:
            failed_files.append(file)
            if 'erate_ind' in locals():
                count_failed[erate_ind] += 1
            continue

    return count, count_failed, dump_out_array



                    

# find min max file size 
count,count_failed, dump_out_array=  check_file_sizes(erate,dump_name_list,real_target,min_outs,n_plates,number_of_particles_per_dump)

print(count)
print(count_failed)






#%%
#compute stress tensor 

def  compute_spring_pos_tensor(erate,dump_out_array,n_plates):
    spring_force_positon_array=np.zeros((erate.size,real_target,dump_out_array.shape[2],n_plates,6))
    spring_force_positon_array[:,:,:,:,0]=-dump_out_array[:,:,:,:,0]*dump_out_array[:,:,:,:,3]#xx
    spring_force_positon_array[:,:,:,:,1]=-dump_out_array[:,:,:,:,1]*dump_out_array[:,:,:,:,4]#yy
    spring_force_positon_array[:,:,:,:,2]=-dump_out_array[:,:,:,:,2]*dump_out_array[:,:,:,:,5]#zz
    spring_force_positon_array[:,:,:,:,3]=-dump_out_array[:,:,:,:,0]*dump_out_array[:,:,:,:,5]#xz
    spring_force_positon_array[:,:,:,:,4]=-dump_out_array[:,:,:,:,0]*dump_out_array[:,:,:,:,4]#xy
    spring_force_positon_array[:,:,:,:,5]=-dump_out_array[:,:,:,:,1]*dump_out_array[:,:,:,:,5]#yz    
    return spring_force_positon_array

spring_pos_tensor=compute_spring_pos_tensor(erate,dump_out_array,n_plates)        

# # take volume average 
stress_tensor=np.sum(spring_pos_tensor,axis=3)/(box_size**3)
print(stress_tensor.shape)
# dimensions = erate,realisation,time,stress_component

# # take realisation average 
stress_tensor_mean=np.mean(stress_tensor,axis=1)

print(stress_tensor_mean.shape)


n_blocks = 50
# only do this after realisation mean and stress tensor compute 
def block_average_array(n_blocks,stress_tensor_mean):
    time_len = stress_tensor_mean.shape[1]
    block_size = time_len // n_blocks  
    if block_size == 0:
       raise ValueError("Not enough time steps for block averaging.")
    trim_len = block_size * n_blocks
    print(trim_len)
    stress_trimmed = stress_tensor_mean[:, :trim_len, :]  # shape: (erate, time, component)

    # Reshape and compute block-wise time series
    stress_blocks = stress_trimmed.reshape((stress_trimmed.shape[0], n_blocks, block_size, stress_trimmed.shape[2]))
    stress_time_series = stress_blocks.mean(axis=2)  # shape: (erate, n_blocks, component)
    stress_time_series_std=stress_blocks.std(axis=2)
    print("Block-averaged stress time series shape:", stress_time_series.shape)

    return stress_time_series,stress_time_series_std
    
stress_time_series,stress_time_series_std=block_average_array(n_blocks,stress_tensor_mean)

#%%
strainplot=np.linspace(0,(min_outs/1000)*strain ,min_outs)
strainplot=np.linspace(0,(min_outs/1000)*strain ,n_blocks)

for k in range(7):
    for l in range(6):
        #plt.plot(strainplot,stress_tensor_mean[k,:,l], label=labels_stress[l])
        plt.plot(strainplot,stress_time_series[k,:,l], label=labels_stress[l])
    plt.xlabel("$\gamma$")
    plt.ylabel("$\sigma_{\\alpha \\beta} $")
    #plt.ylim(-0.05,0.6)
    plt.legend()
    plt.title(f"$K={K}, \\dot{{\\gamma}}={erate[k]}$")
    plt.savefig(filepath+"/plots/K_"+str(K)+"_gdot_"+str(erate[k])+"_stress_vs_strain.pdf",dpi=1200,bbox_inches='tight')
    plt.show()



# #%% rough test of convergence
# strainplot=np.linspace(0,(min_outs/1000)*strain ,min_outs)
# for k in range(erate.size):
#     for l in range(6):
#         plt.plot(strainplot,stress_tensor_mean_box_100[k,:,l]-stress_tensor_mean_box_200[k,:,l], label="$\Delta$"+labels_stress[l])
#         #plt.plot(strainplot,stress_tensor_mean_box_200[k,:,l], label=labels_stress[l]+",N=4000")
#     plt.xlabel("$\gamma$")
#     plt.ylabel("$\sigma_{\\alpha \\beta} $")
#     #plt.ylim(-0.05,0.6)
#     plt.legend(bbox_to_anchor=(1,1))
#     plt.title(f"$K={K}, \\dot{{\\gamma}}={erate[k]}$")
#     plt.show()


#%%

#cutoff=300
# SS_mean_stress_tensor=np.mean(stress_tensor_mean[:,cutoff:,:],axis=1)
# SS_mean_stress_tensor_std=np.std(stress_tensor_mean[:,cutoff:,:],axis=1)
cutoff_block=40
cutoff=int(np.round(min_outs*(cutoff_block/n_blocks)))
SS_mean_stress_tensor=np.mean(stress_time_series[:,cutoff_block:,:],axis=1)
SS_mean_stress_tensor_std=np.abs(np.std(stress_time_series[:,cutoff_block:,:],axis=1))#/np.sqrt(n_blocks-cutoff_block)
end_cut=7
for l in range(6):
    plt.errorbar(Wi[:end_cut],SS_mean_stress_tensor[:end_cut,l],yerr=SS_mean_stress_tensor_std[:end_cut,l],linestyle="dashed", label=labels_stress[l])
    plt.xlabel("$Wi$")
    plt.ylabel("$\\bar{\sigma}_{\\alpha \\beta} $")
    plt.title("$K="f"{K}$")
    plt.legend()
plt.savefig(filepath+"/plots/K_"+str(K)+"_stress_vs_erate.pdf",dpi=1200,bbox_inches='tight')
plt.show()

#n1
n_1=SS_mean_stress_tensor[:,0]-SS_mean_stress_tensor[:,2]
n_2=SS_mean_stress_tensor[:,2]-SS_mean_stress_tensor[:,1]
popt, cov_matrix_n1 = curve_fit(
        quadfunc, erate[:end_cut], n_1[:end_cut])
difference = np.sqrt(
        np.sum(
            (n_1[:end_cut] - (popt[0] * (erate[:end_cut]) ** 2))
            ** 2
        )
        / (erate.size))

# make smooth with many points 
erate_plot=np.linspace(0,erate[-1],40)

#erate_plot=np.linspace(0,0.81,40)
plt.plot(
        erate_plot,
        popt[0] * (erate_plot) ** 2, label=f"$bx^{2}=${popt[0]:.8f},$\\varepsilon=${(difference):.8f}")
        # f"{timestep_skip_array[l]}, mean bond ext = {np.mean(np.ravel(spring_extension_array[i,:,m:, :]) - 0.05):.3f}"
plt.errorbar(erate,SS_mean_stress_tensor[:,0]-SS_mean_stress_tensor[:,2],yerr=np.sqrt(np.mean(SS_mean_stress_tensor_std[:,0]**2 +SS_mean_stress_tensor_std[:,1]**2)),linestyle="dashed")
plt.ylabel("$N_{1}$")
plt.xlabel("$\dot{\gamma}$")
plt.legend()
plt.title("$K="f"{K}$")
plt.savefig(filepath+"/plots/K_"+str(K)+"_N1_vs_erate.pdf",dpi=1200,bbox_inches='tight')
plt.show()


#n2 
n_2=SS_mean_stress_tensor[:,2]-SS_mean_stress_tensor[:,1]

plt.errorbar(erate,SS_mean_stress_tensor[:,2]-SS_mean_stress_tensor[:,1],yerr=np.sqrt(np.mean(SS_mean_stress_tensor_std[:,2]**2 +SS_mean_stress_tensor_std[:,1]**2)),linestyle="dashed")
plt.ylabel("$N_{2}$")
plt.xlabel("$\dot{\gamma}$")
plt.title("$K="f"{K}$")
plt.savefig(filepath+"/plots/K_"+str(K)+"_N2_vs_erate.pdf",dpi=1200,bbox_inches='tight')
plt.show()

# shear visc with rms mean
shear_stress=np.sqrt(np.mean(stress_tensor_mean[:,cutoff:,3]**2,axis=1))

plt.errorbar(erate,SS_mean_stress_tensor[:,3],yerr=SS_mean_stress_tensor_std[:,3] ,linestyle="dashed")
plt.xlabel("$\dot{\gamma}$")
plt.ylabel(labels_stress[3])
plt.savefig(filepath+"/plots/K_"+str(K)+"_sigxz_vs_erate.pdf",dpi=1200,bbox_inches='tight')
plt.show()


plt.plot(erate,n_1/SS_mean_stress_tensor[:,3],linestyle="dashed")
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$\\frac{N_{1}}{\sigma_{xz}}$",rotation=0)
plt.savefig(filepath+"/plots/K_"+str(K)+"_N1_sigxz_vs_erate.pdf",dpi=1200,bbox_inches='tight')
plt.show()

# taking RMS mean of shear stress


#%% plot orientations 
plt.rcParams.update({"font.size": 10})
pi_theta_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
pi_theta_tick_labels = ["-π", "-π/2", "0", "π/2", "π"]
pi_phi_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
pi_phi_tick_labels = ["0", "π/8", "π/4", "3π/8", "π/2"]

timestep_skip_array=[]
timestep_skip_array=[0,150,300,cutoff]

#timestep_skip_array=[0,100,200,300,400,499]
#timestep_skip_array=[0,25,50,99]
#timestep_skip_array=[799]
# timestep_skip_array=[0,100,200,300,400,500]


def convert_cart_2_spherical_z_inc_DB(
    dump_out_array, n_plates, cutoff
):
    spherical_coords_tuple = ()
    

    area_vector_ray = dump_out_array
    
    area_vector_ray[area_vector_ray[:, :,:, :, 5] < 0] *= -1

    x = area_vector_ray[ :, :,:, :, 3]
    y = area_vector_ray[:, :,:, :, 4]
    z = area_vector_ray[ :, :,:, :, 5]
    spring_extension_array=np.sqrt(x**2 + y**2 +z**2)        

    spherical_coords_array = np.zeros(
        ( erate.size,real_target,area_vector_ray.shape[2] , n_plates, 3)
    )

    # radial coord
    spherical_coords_array[ :, :,:, :, 0] = np.sqrt((x**2) + (y**2) + (z**2))

    #  theta coord
    spherical_coords_array[ :, :,:, :, 1] = np.sign(y) * np.arccos(
        x / (np.sqrt((x**2) + (y**2)))
    )

    # spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
    # spherical_coords_array[:,:,:,1]=np.arctan(y/x)

    # phi coord
    # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
    spherical_coords_array[ :, :,:, :, 2] = np.arccos(
        z / np.sqrt((x**2) + (y**2) + (z**2))
    )

    spherical_coords_tuple = spherical_coords_tuple + (spherical_coords_array,)

    return spherical_coords_array,spring_extension_array

adjust_factor = 1

spherical_coords,spring_extension_array = convert_cart_2_spherical_z_inc_DB(
    dump_out_array, n_plates, cutoff
)
#%%
for l in range(len(timestep_skip_array)):
    for i in range(erate.size):
     

        data = spherical_coords[ i, :,:, :, 2]

        periodic_data = np.array([data, np.pi - data])

        
        m = timestep_skip_array[l]
        sns.kdeplot(
            data=np.ravel(periodic_data[ :, :,m, :]),label = f"{timestep_skip_array[l]}, $\\dot{{\\gamma}}={str(erate[i])}$",
            bw_adjust=adjust_factor
        )
    
        plt.xlabel("$\phi$")
        
        plt.xticks(pi_phi_ticks, pi_phi_tick_labels)

        plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

        plt.ylabel("Density")
        plt.xlim(0, np.pi / 2)

        # plt.xlim(0,np.pi)
        plt.tight_layout()
        plt.savefig(filepath+"/plots/K_"+str(K)+"_phi_dist.pdf",dpi=1200,bbox_inches='tight')
    plt.show()



# %% different style plot of theta

for l in range(len(timestep_skip_array)):
    for i in range(erate.size):
        data = spherical_coords[ i, :,:, :, 1]

        periodic_data = np.array([data - 2 * np.pi, data, data + 2 * np.pi])
        
        m = timestep_skip_array[l]
        sns.kdeplot(
            data=np.ravel(periodic_data[ :, :,m, :]),label = f"{timestep_skip_array[l]}, $\\dot{{\\gamma}}={str(erate[i])}$",
            bw_adjust=adjust_factor
        )

        plt.xlabel("$\Theta$")

        
        plt.xticks(pi_theta_ticks, pi_theta_tick_labels)

        plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

        plt.ylabel("Density")
        plt.xlim(-np.pi, np.pi)

        # plt.xlim(0,np.pi)
        plt.tight_layout()
        plt.savefig(filepath+"/plots/K_"+str(K)+"_theta_dist.pdf",dpi=1200,bbox_inches='tight')

    plt.show()
# %%
mean_extension=[]
for i in range(erate.size):
    for l in range(len(timestep_skip_array)):
        m = timestep_skip_array[l]
        sns.kdeplot(
            data=np.ravel(spring_extension_array[i,:,m, :])-0.05,label=f"$\dot{{\gamma}}={erate[i]}$, mean bond ext = {np.mean(np.ravel(spring_extension_array[i,:,m:, :]) - 1):.3f}",
            bw_adjust=adjust_factor
        )
    mean_extension.append(np.mean(np.ravel(spring_extension_array[i,:,m:, :]) - 1))
    
    
    plt.xlabel("$\Delta x$")

    plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

    plt.ylabel("Density")
    # plt.xlim(0, np.pi / 2)

    # plt.xlim(0,np.pi)
    plt.tight_layout()
    # plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
plt.savefig(filepath+"/plots/K_"+str(K)+"_ext_dist.pdf",dpi=1200,bbox_inches='tight')
plt.show()

plt.plot(erate,mean_extension)
 
plt.ylabel("$\\bar{\Delta x}$")
plt.xlabel("$\dot{\gamma}$")
plt.show()
# %% plot theta against phi


for i in range(erate.size):
    theta = np.ravel(spherical_coords[ i, :,:, :, 1])
    phi=np.ravel(spherical_coords[ i, :,:, :, 2])
    # plt.hist2d(phi, theta, bins=50, cmap='viridis')
    # plt.colorbar(label="Counts")
    plt.scatter(theta,phi,s=0.000005)
    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)
    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
    plt.xlabel("$\Theta$")
    plt.ylabel("$\phi$")
    plt.title(f"$K={K}, \\dot{{\\gamma}}={erate[i]}$")
    plt.savefig(filepath+"/plots/K_"+str(K)+"_gdot_"+str(erate[k])+"_phi_vs_theta.pdf",dpi=1200,bbox_inches='tight')
    plt.show()


# %%
