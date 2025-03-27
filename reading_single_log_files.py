#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
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

number_of_particles_per_dump=500
dump_start_line="ITEM: ATOMS id type x y z vx vy vz"
dump_realisation_name="DBshearnvt_no485932_hookean_dumb_belllgeq_130353_16_100_0.035_0.005071624521210362_19999_1999_199999900_4.436450521111511e-06_gdot_0.1111111111111111_K_15.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)
dump_data=np.reshape(dump_data,(10000,500,8))

# %%
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
skip_array=[0,10,20,30,40,50,60,70,80,90,99]
for i in range(len(skip_array)):
    j=skip_array[i]
    plt.scatter(z_positon[j],x_vel[j])
    fit=np.polyfit(z_positon[j],x_vel[j],1, full=True)
    print("gradient=",fit[0])
    print(fit)
    plt.show()
# %%
# mass 0.5 
number_of_particles_per_dump=500
dump_realisation_name="DBshearnvt_no48532_hookean_dumb_belllgeq_130353_16_100_0.035_0.005071624521210362_19999_1999_199999900_4.436450521111511e-06_gdot_0.1111111111111111_K_15.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)



dump_data=np.reshape(dump_data,(10000,500,8))
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
for i in range(len(skip_array)):
    j=skip_array[i]
    plt.scatter(z_positon[j],x_vel[j])
    fit=np.polyfit(z_positon[j],x_vel[j],1, full=True)
    print("gradient=",fit[0])
    print(fit)
    plt.show()

#%% mass 0.05 
number_of_particles_per_dump=500
dump_realisation_name="DBshearnvt_no160279_hookean_dumb_belllgeq_492413_5_100_0.035_0.005071624521210362_1999999_1999999_1999999000_4.436450521111511e-06_gdot_0.1111111111111111_K_15.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)



dump_data=np.reshape(dump_data,(1000,500,8))
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,100,200,300,400,500,600,700,800,900,999]
for i in range(len(skip_array)):
    j=skip_array[i]
    plt.scatter(z_positon[j],x_vel[j])
    fit=np.polyfit(z_positon[j],x_vel[j],1, full=True)
    print("gradient=",fit[0])
    print(fit)
    plt.show()


# %% mass 0.001 
number_of_particles_per_dump=1000
dump_realisation_name="DBshearnvt_no360533_hookean_dumb_bell_61878_0_100_0.035_0.005071624521210362_1999999_1999999_1999999000_4.436450521111511e-06_gdot_0.1111111111111111_K_60.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)



dump_data=np.reshape(dump_data,(1000,1000,8))
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,100,200,300,400,500,600,700,800,900,999]
for i in range(len(skip_array)):
    j=skip_array[i]
    plt.scatter(z_positon[j],x_vel[j])
    fit=np.polyfit(z_positon[j],x_vel[j],1, full=True)
    print("gradient=",fit[0])
    print(fit)
    plt.show()

# %%mass 2 

number_of_particles_per_dump=500
dump_realisation_name="DBshearnvt_no48532_hookean_dumb_belllgeq_130353_16_100_0.035_0.005071624521210362_19999_1999_199999900_4.436450521111511e-06_gdot_0.1111111111111111_K_15.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)



dump_data=np.reshape(dump_data,(10000,500,8))
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
for i in range(len(skip_array)):
    j=skip_array[i]
    plt.scatter(z_positon[j],x_vel[j])
    fit=np.polyfit(z_positon[j],x_vel[j],1, full=True)
    print("gradient=",fit[0])
    print(fit)
    plt.show()

# %%
