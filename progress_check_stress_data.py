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
filepath="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_500_sllod_wi"
erate=np.array([0.05, 0.24, 0.43, 0.62, 0.81, 1.0  ])
os.chdir(filepath)
K=0.0625
number_of_particles_per_dump=1000
n_plates=500
Path_2_dump=filepath
dump_name_list=glob.glob("*_K_"+str(K)+".dump")
print(len(dump_name_list))
min_outs=800
dump_start_line="ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]"

failed_files=[]
passed_file_names=[]
passed_file_dumps=[]
real_target=5
def check_file_sizes(erate,list,real_target,min_outs,n_plates,number_of_particles_per_dump):
    count=np.zeros((erate.size)).astype("int")
    count_failed=np.zeros((erate.size)).astype("int")
    count_reals=np.zeros((erate.size)).astype("int")
    dump_out_array=np.zeros((erate.size,real_target,min_outs,n_plates,6))
    for file in list:
        try:
            split_name=file.split('_')
            #erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
            erate_file=np.around(float(split_name[15]),5)
            #print(erate_file)
            #erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
            erate_ind=int(np.where(erate==erate_file)[0][0])
            # realisation_ind=int(split_name[6])
            # spring_stiff=int(split_name[17])

            dump_data=dump2numpy_tensor_1tstep(dump_start_line,
                      Path_2_dump,file,
                      number_of_particles_per_dump,n_plates)

            file_size_rows=dump_data.shape[0]

            
            if count[erate_ind]==real_target:

                    continue

            elif file_size_rows<min_outs:
                    
                    failed_files.append(file)
                    count_failed[erate_ind]+=1
                    continue

            else:
                    passed_file_names.append(file)
                    #passed_file_dumps.append(dump_data)
                    count[erate_ind]+=1
                    realisation_ind=count_reals[erate_ind]
                    dump_out_array[erate_ind,realisation_ind]=dump_data[:min_outs]
                    count_reals[erate_ind] += 1
                    print(count_reals)

                
        except:
            failed_files.append(file)
            count_failed[erate_ind]+=1

            continue


    return count,count_failed, dump_out_array



                    

# find min max file size 
count,count_failed, dump_out_array=  check_file_sizes(erate,dump_name_list,real_target,min_outs,n_plates,number_of_particles_per_dump)

print(count)
print(count_failed)

#%% now sort each passed file into a stress array


# def sort_into_array(passed_file_names,passed_file_dumps,min_outs,erate,n_plates):
#     dump_out_array=np.zeros((erate.size,real_target,min_outs,n_plates,6))
#     count_reals=np.zeros((erate.size)).astype("int")
#     for i in range(len(passed_file_names)):
          
#         split_name=passed_file_names[i].split('_')
#         #erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
#         erate_file=np.around(float(split_name[15]),5)
#         #print(erate_file)
#         #erate_ind=int(np.where(erate==float(split_name[15]))[0][0])
#         erate_ind=int(np.where(erate==erate_file)[0][0])
#         realisation_ind=count_reals[erate_ind]

#         dump_data=passed_file_dumps[i][:min_outs]

#         dump_out_array[erate_ind,realisation_ind]=dump_data

#         count_reals[erate_ind]+=1

#     return dump_out_array,count_reals

# chat gpt optimized
# def sort_into_array(passed_file_names, passed_file_dumps, min_outs, erate, n_plates):
#     # create index dict instead of mannually looking up with np.where
#     erate_lookup = {round(float(val), 5): idx for idx, val in enumerate(erate)}
#     dump_out_array = np.zeros((erate.size, real_target, min_outs, n_plates, 6))
#     count_reals = np.zeros(erate.size, dtype=int)

#     for fname, fdata in zip(passed_file_names, passed_file_dumps):
#         split_name = fname.split('_')
#         erate_file = round(float(split_name[15]), 5)

#         # Fast lookup instead of np.where
#         erate_ind = erate_lookup[erate_file]
#         realisation_ind = count_reals[erate_ind]

#         dump_out_array[erate_ind, realisation_ind] = fdata[:min_outs]
#         count_reals[erate_ind] += 1
#         print(count_reals)

#     return dump_out_array, count_reals

# dump_out_array,count_reals=sort_into_array(passed_file_names,passed_file_dumps,min_outs,erate,n_plates)




#%%
#compute stress tensor 
def  compute_stress_tensor(erate,dump_out_array,n_plates):
    spring_force_positon_array=np.zeros((erate.size,real_target,dump_out_array.shape[2],n_plates,6))
    spring_force_positon_array[:,:,:,:,0]=-dump_out_array[:,:,:,:,0]*dump_out_array[:,:,:,:,3]#xx
    spring_force_positon_array[:,:,:,:,1]=-dump_out_array[:,:,:,:,1]*dump_out_array[:,:,:,:,4]#yy
    spring_force_positon_array[:,:,:,:,2]=-dump_out_array[:,:,:,:,2]*dump_out_array[:,:,:,:,5]#zz
    spring_force_positon_array[:,:,:,:,3]=-dump_out_array[:,:,:,:,0]*dump_out_array[:,:,:,:,5]#xz
    spring_force_positon_array[:,:,:,:,4]=-dump_out_array[:,:,:,:,0]*dump_out_array[:,:,:,:,4]#xy
    spring_force_positon_array[:,:,:,:,5]=-dump_out_array[:,:,:,:,1]*dump_out_array[:,:,:,:,5]#yz    
    return spring_force_positon_array

stress_tensor=compute_stress_tensor(erate,dump_out_array,n_plates)        

# # take realisation average 
stress_tensor_mean=np.mean(stress_tensor,axis=1)
print(stress_tensor_mean.shape)

# # take dumbbell average 

stress_tensor_mean=np.mean(stress_tensor_mean,axis=2)
print(stress_tensor_mean.shape)
strainplot=np.linspace(0,(min_outs/1000)*500 ,min_outs)
for k in range(6):
    for l in range(6):
        plt.plot(strainplot,stress_tensor_mean[k,:,l])
    plt.ylim(-0.05,0.6)
    plt.show()

Wi=np.array([0.4,  1.12, 1.84, 2.56, 3.28, 4.,  ])
cutoff=500
SS_mean_stress_tensor=np.mean(stress_tensor_mean[:,cutoff:,:],axis=1)

SS_mean_stress_tensor_std=np.std(stress_tensor_mean[:,cutoff:,:],axis=1)
for l in range(6):
    plt.errorbar(Wi,SS_mean_stress_tensor[:,l],yerr=SS_mean_stress_tensor_std[:,l],linestyle="dashed")
plt.show()

#n1
n_1=SS_mean_stress_tensor[:,0]-SS_mean_stress_tensor[:,1]
n_2=SS_mean_stress_tensor[:,2]-SS_mean_stress_tensor[:,1]
popt, cov_matrix_n1 = curve_fit(
        quadfunc, erate, n_1)
plt.plot(
        erate,
        popt[0] * (erate) ** 2)
plt.errorbar(erate,SS_mean_stress_tensor[:,0]-SS_mean_stress_tensor[:,1],yerr=np.sqrt(np.mean(SS_mean_stress_tensor_std[:,0]**2 +SS_mean_stress_tensor_std[:,1]**2)),linestyle="dashed")
plt.show()


#n2 
n_2=SS_mean_stress_tensor[:,2]-SS_mean_stress_tensor[:,1]
popt, cov_matrix_n1 = curve_fit(
        quadfunc, erate, n_2)
plt.errorbar(erate,SS_mean_stress_tensor[:,2]-SS_mean_stress_tensor[:,1],yerr=np.sqrt(np.mean(SS_mean_stress_tensor_std[:,2]**2 +SS_mean_stress_tensor_std[:,1]**2)),linestyle="dashed")
plt.show()

# shear visc
shear_stress=np.sqrt(np.mean(stress_tensor_mean[:,cutoff:,3]**2,axis=1))

plt.errorbar(erate,shear_stress,yerr=SS_mean_stress_tensor_std[:,3] ,linestyle="dashed")
plt.show()

# taking RMS mean of shear stress


#%% plot orientations 
timestep_skip_array=[0,100,200,300,400,500,600,700,799]
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

adjust_factor = 0.25

spherical_coords,spring_extension_array = convert_cart_2_spherical_z_inc_DB(
    dump_out_array, n_plates, cutoff
)
#%%
for i in range(erate.size):
     

    data = spherical_coords[ i, :,:, :, 2]

    periodic_data = np.array([data, np.pi - data])

    for l in range(len(timestep_skip_array)):
        m = timestep_skip_array[l]
        sns.kdeplot(
            data=np.ravel(periodic_data[ :, :,m, :]),label=timestep_skip_array[l],
            bw_adjust=adjust_factor
        )
    
    plt.xlabel("$\phi$")

    plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

    plt.ylabel("Density")
    plt.xlim(0, np.pi / 2)

    # plt.xlim(0,np.pi)
    plt.tight_layout()
    # plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()



# %% different style plot of theta


for i in range(erate.size):
    data = spherical_coords[ i, :,:, :, 1]

    periodic_data = np.array([data - 2 * np.pi, data, data + 2 * np.pi])
    for l in range(len(timestep_skip_array)):
        m = timestep_skip_array[l]
        sns.kdeplot(
            data=np.ravel(periodic_data[ :, :,m, :]),label=timestep_skip_array[l],
            bw_adjust=adjust_factor
        )

    plt.xlabel("$\Theta$")

    plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

    plt.ylabel("Density")
    plt.xlim(-np.pi, np.pi)

    # plt.xlim(0,np.pi)
    plt.tight_layout()
    # plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()
# %%
for i in range(erate.size):
    for l in range(len(timestep_skip_array)):
        m = timestep_skip_array[l]
        sns.kdeplot(
            data=np.ravel(spring_extension_array[i,:,m, :])-0.05,label=f"{timestep_skip_array[l]}, mean bond ext = {np.mean(np.ravel(spring_extension_array[i,:,m:, :]) - 0.05):.3f}",
            bw_adjust=adjust_factor
        )
    
    plt.xlabel("$\Delta x$")

    plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

    plt.ylabel("Density")
    # plt.xlim(0, np.pi / 2)

    # plt.xlim(0,np.pi)
    plt.tight_layout()
    # plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()
# %%
