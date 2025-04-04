#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt

Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/tchain_5_tdam_100_rsl_5_strain_mass_1/"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_0.5_erate_0.05_1_strain_20/"
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_1_erate_0.05_1_strain_20_sllod"
os.chdir(Path_2_log)


# %%# mass 1 run 
realisation_name="log.DBshearnvt_no485932_hookean_dumb_belllgeq_130353_16_100_0.035_0.005071624521210362_19999_1999_199999900_4.436450521111511e-06_gdot_0.1111111111111111_K_15"
thermo_vars="         KinEng      c_spring_pe       PotEng         Press         c_myTemp       c_bias_2        c_bias         TotEng       Econserve       Ecouple    "
log_file=log2numpy_reader(realisation_name, Path_2_log, thermo_vars)
log_data=log_file[:,7]
plt.plot(log_data)
print("mean",np.mean(log_data))
print("gradient", np.mean(np.gradient(log_data)))
print("std_dev",np.std(log_data))
log_data=log_file[:,6]
plt.show()
print("mean",np.mean(log_data))
print("gradient", np.mean(np.gradient(log_data)))
print("std_dev",np.std(log_data))
log_file=log2numpy_reader(realisation_name, Path_2_log, thermo_vars)
log_data=log_file[:,7]
plt.plot(log_data)
plt.show()
print("mean",np.mean(log_data))
print("gradient", np.mean(np.gradient(log_data)))
print("std_dev",np.std(log_data))
#%%
number_of_particles_per_dump=1000
dump_start_line="ITEM: ATOMS id type x y z vx vy vz"
dump_realisation_name="DBshearnvtmulti_no270878_hookean_dumb_belllgeq_233845_6_100_0.035_0.005071624521210362_1999999_1999999_1999999000_2.4342664039020638e-06_gdot_0.81_K_120.dump"
dump_realisation_name="hookean_dumb_bell.dump"
dump_realisation_name="DBshearnvtmulti_no883355_hookean_dumb_belllgeq_224910_1_100_0.035_0.005071624521210362_1999999_1999999_1999999000_3.943511574321343e-05_gdot_0.05_K_120.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)
dump_data=np.reshape(dump_data,(1000,number_of_particles_per_dump,8))

# %%
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
skip_array=[0,10,20,30,40,50,60,70,80,90,99]
skip_array=[0,200,400,600,800,999]
#skip_array=[0,37,67,70,100,140,170,189,194]
#skip_array=[0,10,30,44]
for i in range(len(skip_array)):
    j=skip_array[i]
    plt.scatter(z_positon[j],x_vel[j])
    fit=np.polyfit(z_positon[j],x_vel[j],1, full=True)
    print("gradient=",fit[0])
    print(fit)
    plt.show()




#%% tensor dump 
from collections import Counter
import seaborn as sns
number_of_particles_per_dump=1000
n_plates=500
Path_2_dump=Path_2_log
dump_start_line="ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]"
dump_realisation_name="DBshearnvtmulti_no270878_db_hooke_tensorlgeq_159333_4_100_0.035_0.005071624521210362_1999999_1999999_1999999000_1.9717557871606717e-06_gdot_1.0_BK_500_K_120.dump"
#dump_realisation_name="DBshearnvtmulti_no270878_db_hooke_tensorlgeq_233845_6_100_0.035_0.005071624521210362_1999999_1999999_1999999000_2.4342664039020638e-06_gdot_0.81_BK_500_K_120.dump"
#dump_realisation_name="DBshearnvtmulti_no270878_db_hooke_tensorlgeq_273035_0_100_0.035_0.005071624521210362_1999999_1999999_1999999000_3.1802512696139854e-06_gdot_0.6200000000000001_BK_500_K_120.dump"
dump_realisation_name="db_hooke_tensorlgeq.dump"

def dump2numpy_tensor_1tstep(dump_start_line,
                      Path_2_dump,dump_realisation_name,
                      number_of_particles_per_dump):





        os.chdir(Path_2_dump) #+simulation_file+"/" +filename



        with open(dump_realisation_name, 'r') as file:


            lines = file.readlines()

            counter = Counter(lines)

            #print(counter.most_common(3))
            n_outs=int(counter["ITEM: TIMESTEP\n"])
            dump_outarray=np.zeros((n_outs,n_plates,6))
            #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])

            skip_array=np.arange(1,len(lines),n_plates+9)
            for i in range(n_outs):
                k=skip_array[i]
                # timestep_list=[]
                start=k-1
                end=start+n_plates+9
                timestep_list=lines[start:end]
                data_list=timestep_list[9:]
                #print(data_list[0])
                #print(len(data_list))
                data=np.zeros((n_plates,6))
                for j in range(len(data_list)):
                    data[j,:]=data_list[j].split(" ")[0:6]

                dump_outarray[i,:,:]=data


        return dump_outarray

dump_outarray=dump2numpy_tensor_1tstep(dump_start_line,
                      Path_2_dump,dump_realisation_name,
                      number_of_particles_per_dump)
print(dump_outarray.shape)

#%%
cutoff=0
j_=1
timestep_skip_array=[56,59,90,99,140,160]
timestep_skip_array=[0,20,40,70,74,150,70]
timestep_skip_array=[0,6,15,18,42,57]
timestep_skip_array=[0,37,67,70,100,140,170,189,201,413]
#timestep_skip_array=[0,120,260,300,370]
#timestep_skip_array=[0,300,370]
# timestep_skip_array=[0,100,200,320,342]
def convert_cart_2_spherical_z_inc_DB(
    dump_outarray, n_plates, cutoff
):
    spherical_coords_tuple = ()
    

    area_vector_ray = dump_outarray
    area_vector_ray[area_vector_ray[ :, :, 2] < 0] *= -1

    x = area_vector_ray[ cutoff:, :, 3]
    y = area_vector_ray[ cutoff:, :, 4]
    z = area_vector_ray[ cutoff:, :, 5]

    spherical_coords_array = np.zeros(
        ( area_vector_ray.shape[0] , n_plates, 3)
    )

    # radial coord
    spherical_coords_array[ :, :, 0] = np.sqrt((x**2) + (y**2) + (z**2))

    #  theta coord
    spherical_coords_array[ :, :, 1] = np.sign(y) * np.arccos(
        x / (np.sqrt((x**2) + (y**2)))
    )

    # spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
    # spherical_coords_array[:,:,:,1]=np.arctan(y/x)

    # phi coord
    # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
    spherical_coords_array[ :, :, 2] = np.arccos(
        z / np.sqrt((x**2) + (y**2) + (z**2))
    )

    spherical_coords_tuple = spherical_coords_tuple + (spherical_coords_array,)

    return spherical_coords_array

adjust_factor = 0.1

spherical_coords = convert_cart_2_spherical_z_inc_DB(
    dump_outarray, n_plates, cutoff
)



data = spherical_coords[ :, :, 2]

periodic_data = np.array([data, np.pi - data])

for l in range(len(timestep_skip_array)):
    m = timestep_skip_array[l]
    sns.kdeplot(
        data=np.ravel(periodic_data[ :, 400:, :]),label=timestep_skip_array[l],
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



data = spherical_coords[ :, :, 1]

periodic_data = np.array([data - 2 * np.pi, data, data + 2 * np.pi])
for l in range(len(timestep_skip_array)):
    m = timestep_skip_array[l]
    sns.kdeplot(
        data=np.ravel(periodic_data[ :,400:, :]),label=timestep_skip_array[l],
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
