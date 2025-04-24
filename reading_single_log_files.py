#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt

Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/tchain_5_tdam_100_rsl_5_strain_mass_1/"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_0.5_erate_0.05_1_strain_20/"
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_25_T_1_sllod_wi/"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_500_sllod_wi"
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


#%% dumbbell VACF test
realisation_name="log.DB_minimal_shear_diff"
file_name="comv_T_100_K_60_mass_1_lang.dat"
file_name="comv_T_50_K_60_mass_1_lang.dat"
file_name="comv_T_10_K_60_mass_1_lang.dat"
#file_name="comv_T_25_K_60_mass_1_lang.dat"
file_name="comv_T_100_K_30_mass_30_lang.dat"
file_name="comv_T_10_K_30_mass_100_lang.dat"
file_name="comv_T_1_K_2.5_mass_40_lang.dat"
file_name="comv_T_0.1_K_1.25_mass_40_lang.dat"
file_name="comv_T_0.05_K_1.25_mass_40_lang.dat"
#file_name="comv_T_0.1_K_0.625_mass_80_lang.dat"
file_name="comv_T_0.01_K_1.25_mass_40_lang.dat"
file_name="comv_T_0.05_K_0.125_mass_4_lang.dat"
file_name="comv_T_0.05_K_0.0625_mass_4_lang.dat"
file_name="comv_T_0.05_K_0.03125_mass_4_lang.dat"
file_name="comv_T_0.05_K_0.625_mass_40_lang.dat"
file_name="comv_T_0.5_K_0.5_mass_4.dat"
file_name="comv_T_1_K_0.5_mass_4.dat"
file_name="comv_T_1_K_0.5_mass_4.dat"
file_name="comv_T_1_K_0.5_mass_R_0.025_R_n_2_N_500.dat"
#file_name="comv_lang.dat"

number_of_mol_per_dump=1000
lines_per_dump=number_of_mol_per_dump
n_cols=4
first_skip=3
Path_2_file=Path_2_log
def generic_vector_file_reader_mols(Path_2_file,file_name,n_cols, number_of_mol_per_dump,first_skip):
    os.chdir(Path_2_file)  # +simulation_file + "/" + filename

    with open(file_name, "r") as file:
        # Skip the first 3 lines and process the rest
        lines = file.readlines()[first_skip:]  # Skip first n lines

        # Precompile regex pattern for efficiency
        pattern = r"\d+\s" + str(number_of_mol_per_dump) + r"\n"
        
        # Find matches for lines that match the pattern (just numbers of outputs)
        matches = [line for line in lines if re.match(pattern, line)]
        
        # find number of dump outputs 
        n_outs = len(matches)  # Number of outputs

        # Efficiently remove lines that match any output line from 'lines'
        lines = [line for line in lines if not re.match(pattern, line)]

        # Convert the remaining lines into an array of floats
        # Assume that each line contains space-separated floats (adjust parsing logic if different)
        float_lines = []

        for line in lines:
            # Convert each line to a list of floats
            try:
                float_line = list(map(float, line.split()))
                float_lines.append(float_line)
            except ValueError as e:
                print(f"Skipping line due to error: {line}")
                continue  # Skip lines that do not contain valid float values

        # Convert list of lists into a NumPy array
        float_array = np.array(float_lines)    
        float_array=np.reshape(float_array,(n_outs, number_of_mol_per_dump, n_cols))
    return float_array



COM_array=generic_vector_file_reader_mols(Path_2_file,file_name,n_cols, number_of_mol_per_dump,first_skip)


#%%
# def compute_autocorrelation(array):
#     t_0=array[0,:,1:]
   
#     acf=np.tensordot(t_0,array[:,:,1:],axes=(2,2))

#     return acf

t_0=COM_array[0,:,1:] 
acf=np.zeros((COM_array.shape[0],number_of_mol_per_dump))
for i in range(0,COM_array.shape[0]):  
    #acf[i,:]=np.tensordot(t_0,COM_array[i,:,1:],axes=(1,1))
    acf[i,:]=np.einsum('ab,ab->a', t_0, COM_array[i,:,1:])
acf_mean=np.mean(acf,axis=1)
total_time=acf_mean.shape[0]*10000*0.2*0.005071624521210362
time_stamps=np.arange(0,total_time,total_time/acf_mean.shape[0])
#plt.plot(time_stamps,acf_mean)
plt.plot(acf_mean)
plt.xlabel("time")
plt.ylabel("COM_VACF")

plt.show()



#%% generic log file reader 
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_50_sllod_Wi_R_1_N_500/"
log_file_name="log.DB_minimal_shear_diff_T_1_K_0.5_mass_4_R_0.025_R_n_1_N_4000"
# complete file 
log_file_name="log.DBshearnvt_no896421_hookean_dumb_belllgeq_838885_9_100_0.035_0.005071624521210362_1999999_1999999_1999999000_7.950628174034963e-06_gdot_0.6200000000000001_K_0.5"

# incomplete file
log_file_name="log.DBshearnvt_no896421_hookean_dumb_belllgeq_583029_1_100_0.035_0.005071624521210362_1999999_1999999_1999999000_6.085666009755159e-06_gdot_0.81_K_0.25"
filepath=Path_2_log

thermo_vars="   Step         KinEng      c_spring_pe       PotEng         Press         c_myTemp       c_bias_2        c_bias         TotEng       Econserve       Ecouple      \n"
end_string="Loop time of * on * procs for * steps with * atoms"
#need to make sure thermovars has an extra space with \n at the end 
os.chdir(filepath)

def generic_log_file_reader():
    
    with open(log_file_name, "r") as file:
        raw_file= file.readlines()
        # find start of log data 
        try:
            if thermo_vars in raw_file:

                start_index=raw_file.index(thermo_vars) 

                print("found log start line")
               
            
        except ValueError as e:
            print("could not find thermo vars")
           

        # file_data=raw_file[start_index+1:]

        # # find end of log data 

        # if end_string in file_data:
        #    end_index=file_data.index(end_string)
        #    raw_log_data=file_data[:end_index]

        # else: 
        #    print("file incomplete")

        # # now remove any incomplete rows from end of file



        # # now convert whole file into string 


        

    #return file_data
    
generic_log_file_reader()

    

#%%plotting Wi against parameters
scale=1
mass=4#/scale
K=mass/16

T=2/scale
erate=np.linspace(0.05,1,6)
spring_timescale=np.sqrt(mass/K)
print("spring timescale",spring_timescale)
print("minimum_timestep",spring_timescale/50)
mean_bond_extension=np.sqrt(3*T/K)
print("mean bond extension",mean_bond_extension)
Wi=spring_timescale*erate

plt.plot(erate,Wi)
plt.xlabel("Shear rate")
plt.ylabel("Wi")
plt.show()
print("Wi",Wi)
print("mass",mass)
print("spring",K)
print("temp",T)


density=500/(100**3)
side_length=60
n_dumbells=density * (side_length**3)
print(n_dumbells)

#generic_vector_file_reader(Path_2_file,file_name,lines_per_dump,count_string)


#%%
number_of_particles_per_dump=1000
dump_start_line="ITEM: ATOMS id type x y z vx vy vz"
dump_realisation_name="DBshearnvtmulti_no270878_hookean_dumb_belllgeq_233845_6_100_0.035_0.005071624521210362_1999999_1999999_1999999000_2.4342664039020638e-06_gdot_0.81_K_120.dump"
dump_realisation_name="hookean_dumb_bell.dump"
dump_realisation_name="DBshearnvt_no608178_hookean_dumb_belllgeq_793810_3_100_0.035_0.005071624521210362_1999999_1999999_1999999000_4.929389467901679e-05_gdot_0.05_K_0.5.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)
dump_data=np.reshape(dump_data,(158,number_of_particles_per_dump,8))

# %%
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
skip_array=[0,10,20,30,40,50,60,70,80,90,99]
skip_array=[0,7]
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

n_plates=int(number_of_particles_per_dump/2)
Path_2_dump=Path_2_log 

dump_start_line="ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]"
dump_realisation_name="DBshearnvtmulti_no270878_db_hooke_tensorlgeq_159333_4_100_0.035_0.005071624521210362_1999999_1999999_1999999000_1.9717557871606717e-06_gdot_1.0_BK_500_K_120.dump"
#dump_realisation_name="DBshearnvtmulti_no270878_db_hooke_tensorlgeq_233845_6_100_0.035_0.005071624521210362_1999999_1999999_1999999000_2.4342664039020638e-06_gdot_0.81_BK_500_K_120.dump"
#dump_realisation_name="DBshearnvtmulti_no270878_db_hooke_tensorlgeq_273035_0_100_0.035_0.005071624521210362_1999999_1999999_1999999000_3.1802512696139854e-06_gdot_0.6200000000000001_BK_500_K_120.dump"
dump_realisation_name="db_hooke_tensorlgeq_T_0.05_K_1.25_mass_40.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff_T_100_K_60_mass_1.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff_T_50_K_60_mass_1.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff_T_25_K_60_mass_1.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff_T_10_K_30_mass_100.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.1_K_0.625_mass_80.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.1_K_1.25_mass_40.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_T_0.01_K_1.25_mass_40.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_T_0.05_K_0.125_mass_4.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_T_0.05_K_0.0625_mass_4.dump"
# dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_951926_0_100_0.035_0.005071624521210362_1999999_1999999_1999999000_0.0009858778935803358_gdot_0.05_BK_500_K_0.0625.dump"
# dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_139570_5_100_0.035_0.005071624521210362_1999999_1999999_1999999000_0.00020539122782923662_gdot_0.24_BK_500_K_0.0625.dump"
# dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_558811_9_100_0.035_0.005071624521210362_1999999_1999999_1999999000_0.00011463696436980648_gdot_0.43_BK_500_K_0.0625.dump"#
# dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_852425_4_100_0.035_0.005071624521210362_1999999_1999999_1999999000_7.950628174034963e-05_gdot_0.6200000000000001_BK_500_K_0.0625.dump"
# dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_796179_0_100_0.035_0.005071624521210362_1999999_1999999_1999999000_6.085666009755159e-05_gdot_0.81_BK_500_K_0.0625.dump"
# dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_442036_2_100_0.035_0.005071624521210362_1999999_1999999_1999999000_4.929389467901679e-05_gdot_1.0_BK_500_K_0.0625.dump"
# #dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_956444_0_100_0.035_0.005071624521210362_1999999_1999999_1999999000_0.0009858778935803358_gdot_0.05_BK_500_K_0.125.dump"
# #dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_921246_5_100_0.035_0.005071624521210362_1999999_1999999_1999999000_0.00020539122782923662_gdot_0.24_BK_500_K_0.125.dump"
# #dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_911653_3_100_0.035_0.005071624521210362_1999999_1999999_1999999000_0.00011463696436980648_gdot_0.43_BK_500_K_0.125.dump"
# #dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_886410_5_100_0.035_0.005071624521210362_1999999_1999999_1999999000_7.950628174034963e-05_gdot_0.6200000000000001_BK_500_K_0.125.dump"
# #dump_realisation_name="DBshearnvt_no920433_db_hooke_tensorlgeq_117582_5_100_0.035_0.005071624521210362_1999999_1999999_1999999000_4.929389467901679e-05_gdot_1.0_BK_500_K_0.125.dump"
dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.05_K_0.03125_mass_1.dump"
#dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.05_K_0.125_mass_4.dump"
# dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.05_K_0.625_mass_40.dump"
# dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.05_K_0.625_mass_4_N_4000.dump"
#
#  dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.05_K_0.625_mass_4_lattice.dump"
dump_realisation_name="db_hooke_tensorlgeq_diff_T_0.5_K_0.5_mass_4.dump"
dump_realisation_name="db_hooke_tensorlgeq_diff_T_1_K_0.5_mass_1.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_T_1_K_0.5_mass_4_R_0.025_R_n_1.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_T_1_K_0.5_mass_4_R_0.025_R_n_1_erate_0.81.dump"
dump_realisation_name="db_hooke_tensorlgeq_diff_T_1_K_0.25_mass_4_R_0.025_R_n_1_N_4000.dump"
dump_realisation_name="db_hooke_tensorlgeq_diff_T_2_K_1_mass_4_R_0.025_R_n_1_N_500.dump"
dump_realisation_name="db_hooke_tensorlgeq_diff_T_1_K_0.5_mass_4_R_0.025_R_n_2_N_500.dump"
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
spring_force_positon_array=np.zeros((dump_outarray.shape[0],n_plates,6))
spring_force_positon_array[:,:,0]=-dump_outarray[:,:,0]*dump_outarray[:,:,3]#xx
spring_force_positon_array[:,:,1]=-dump_outarray[:,:,1]*dump_outarray[:,:,4]#yy
spring_force_positon_array[:,:,2]=-dump_outarray[:,:,2]*dump_outarray[:,:,5]#zz
spring_force_positon_array[:,:,3]=-dump_outarray[:,:,0]*dump_outarray[:,:,5]#xz
spring_force_positon_array[:,:,4]=-dump_outarray[:,:,0]*dump_outarray[:,:,4]#xy
spring_force_positon_array[:,:,5]=-dump_outarray[:,:,1]*dump_outarray[:,:,5]#yz
spring_force_positon_array=spring_force_positon_array

#%%
cutoff=0
j_=1
timestep_skip_array=[56,59,90,99,140,160]
timestep_skip_array=[0,20,40,70,74,150,70]
timestep_skip_array=[0,50,100,300,480]
# timestep_skip_array=[0,37,67,70,100,140,170,189,201,413]
# timestep_skip_array=[0,20,50,98,300,500]
timestep_skip_array=[1000,3000,4000,6000,8000,10000,11000,12000]
#timestep_skip_array=[1000,3000,5000,10000,15000,30000,40000,45000,60000,80000,100000]

# timestep_skip_array=[0,12,15,17,21,24,25,32,36,50,70,90]
#timestep_skip_array=[2000,5000,7000,9000,12000]
#timestep_skip_array=[200,400,600,800,1000]
#timestep_skip_array=[0,289,300,350,400,450,500]
timestep_skip_array=[0,1000,5000,10000,15000,20000,25000]
def convert_cart_2_spherical_z_inc_DB(
    dump_outarray, n_plates, cutoff
):
    spherical_coords_tuple = ()
    

    area_vector_ray = dump_outarray
    
    area_vector_ray[area_vector_ray[ :, :, 5] < 0] *= -1

    x = area_vector_ray[ cutoff:, :, 3]
    y = area_vector_ray[ cutoff:, :, 4]
    z = area_vector_ray[ cutoff:, :, 5]
    spring_extension_array=np.sqrt(x**2 + y**2 +z**2)        

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

    return spherical_coords_array,spring_extension_array

adjust_factor = 0.25

spherical_coords,spring_extension_array = convert_cart_2_spherical_z_inc_DB(
    dump_outarray, n_plates, cutoff
)



data = spherical_coords[ :, :, 2]

periodic_data = np.array([data, np.pi - data])

for l in range(len(timestep_skip_array)):
    m = timestep_skip_array[l]
    sns.kdeplot(
        data=np.ravel(periodic_data[ :, m, :]),label=timestep_skip_array[l],
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
        data=np.ravel(periodic_data[ :,m, :]),label=timestep_skip_array[l],
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
for l in range(len(timestep_skip_array)):
    m = timestep_skip_array[l]
    sns.kdeplot(
        data=np.ravel(spring_extension_array[m, :])-0,label=f"{timestep_skip_array[l]}, mean bond ext = {np.mean(np.ravel(spring_extension_array[m, :]) - 0.05):.3f}",
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
for l in range(0,6):

   
    mean_stress=np.mean(spring_force_positon_array[:],axis=1)
    end_grad=np.mean(np.gradient(mean_stress[:,l]))
    print("SS_mean",np.mean(mean_stress[-1000:,l]))

    plt.plot(mean_stress[:,l],label=f"end grad = {end_grad:.5f}")
#plt.ylim(-0.1,2)
plt.legend()
plt.show()


# %%

theta = np.ravel(spherical_coords[ :, :, 1])
phi=np.ravel(spherical_coords[ :, :, 2])
combine=np.array([theta,phi])
plt.imshow(combine,cmap='viridis')
plt.colorbar()
plt.ylabel("$\Theta$")
plt.xlabel("$\phi$")
plt.title(f"$K={K}, \\dot{{\\gamma}}={erate[i]}$")

plt.show()


# plt.hist2d(phi, theta, bins=50, cmap='viridis')
# plt.colorbar(label="Counts")
# plt.ylabel(r"$\Theta$")
# plt.xlabel(r"$\phi$")
# plt.title(f"$K={K}, \\dot{{\\gamma}}={erate[i]}$")
# plt.show()

# %%
