#%%
from lammps_file_readers_module import *
import os 
import numpy as np
import matplotlib.pyplot as plt

Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/tchain_5_tdam_100_rsl_5_strain_mass_1/"
Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_0.5_erate_0.05_1_strain_20/"
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/dumbell_test"
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/plate_tests"
Path_2_log="/Users/luke_dev/Documents/simulation_test_folder/chain_tests"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_25_T_1_sllod_wi/"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/mass_4_erate_0.05_1_strain_500_sllod_wi"
#Path_2_log="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/db_runs/DB_shear_run_mass_10_stiff_0.005_1_1_sllod_100_strain_T_0.01_R_1_R_n_1_N_864/logs_and_stress"
os.chdir(Path_2_log)


# %%# equilibration runs
realisation_name="log.DB_minimal_shear_diff_T_1_K_0.5_mass_4_R_0.025_R_n_1.5_N_500"
realisation_name="log.DB_minimal_shear_diff_T_1_K_0.5_mass_4_R_0.025_R_n_2.59_N_500"
realisation_name="log.DB_minimal_shear_diff_T_1_K_0.5_mass_4_R_0.025_R_n_2_N_500"

thermo_vars="         KinEng      c_spring_pe       PotEng         Press         c_myTemp      c_VACF[4]        TotEng       Econserve       Ecouple    "
log_file=log2numpy_reader(realisation_name, Path_2_log, thermo_vars)
log_data=log_file[:,1]
plt.plot(log_data)
print("mean",np.mean(log_data))
print("gradient", np.mean(np.gradient(log_data)))
print("std_dev",np.std(log_data))
log_data=log_file[:,2]
plt.show()
print("mean",np.mean(log_data))
print("gradient", np.mean(np.gradient(log_data)))
print("std_dev",np.std(log_data))
log_file=log2numpy_reader(realisation_name, Path_2_log, thermo_vars)
log_data=log_file[:,5]
plt.plot(log_data)
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
file_name="comv_T_1_K_0.5_mass_R_0.025_R_n_2.59_N_500.dat"
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


mass=10#scale
K=0.005#/scale
T=0.01


erate=np.logspace(-4,-2,10)
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
dump_realisation_name="shear_hookean_dumb_bell_T_0.1_K_0.5_mass_4_R_0.025_R_n_1_erate_0.0001.dump"
dump_realisation_name="shear_hookean_dumb_bell_T_1_K_0.5_mass_10000_R_0.0025_R_n_2.59_erate_0.0001.dump"
#dump_realisation_name="shear_hookean_dumb_bell_T_0.1_K_0.5_mass_4_R_0.025_R_n_1_erate_0.01.dump"
dump_data = dump2numpy_f(
    dump_start_line, Path_2_log, dump_realisation_name, number_of_particles_per_dump)
dump_data=np.reshape(dump_data,(139,number_of_particles_per_dump,8))

# %%
z_positon=dump_data[:,:,4].astype("float")
x_vel=dump_data[:,:,5].astype("float")
skip_array=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,9999]
skip_array=[0,10,20,30,40,50,60,70,80,90,99]
skip_array=[0,7,50,100]
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

Path_2_dump=Path_2_log 

dump_start_line="ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]"


number_of_particles_per_dump=1000
n_plates=int(number_of_particles_per_dump/2)



dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_5000_R_0.0025_R_n_0.1_erate_0.0001.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_5000_R_0.0025_R_n_0.1_erate_0.00208.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_5000_R_0.0025_R_n_0.1_erate_0.00406.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_5000_R_0.0025_R_n_0.1_erate_0.00604.dump"


dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_100_R_0.0025_R_n_0.1_erate_0.0001.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_100_R_0.0025_R_n_0.1_erate_0.0002.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_100_R_0.0025_R_n_0.1_erate_0.0003.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_100_R_0.0025_R_n_0.1_erate_0.00208.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_100_R_0.0025_R_n_0.1_erate_0.00406.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_0.0025_mass_100_R_0.0025_R_n_0.1_erate_0.00604.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.025_R_n_0.1_erate_0.0003.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.025_R_n_0.1_erate_0.0002.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.025_R_n_0.1_erate_0.0001.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.0005_R_n_0.01_erate_0.0001.dump"
# dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.0005_R_n_0.01_erate_0.0002.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.0005_R_n_0.01_erate_0.0003.dump"

#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.00005_R_n_0.001_erate_0.0001.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.00005_R_n_0.001_erate_0.0002.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_1_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"

#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_2_mass_1_R_0.00005_R_n_0.001_erate_0.0001.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.005_K_2_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.01_K_10_mass_1_R_0.00005_R_n_0.001_erate_0.0001.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.01_K_10_mass_1_R_0.00005_R_n_0.001_erate_0.0002.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_5.00000250e-03_T_0.01_K_10_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"

#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_2.00000250e-03_T_0.01_K_20_mass_1_R_0.00005_R_n_0.001_erate_0.0001.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_2.00000250e-03_T_0.01_K_20_mass_1_R_0.00005_R_n_0.001_erate_0.0002.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_2.00000250e-03_T_0.01_K_20_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.01_K_200_mass_1_R_0.00005_R_n_0.001_erate_0.0001.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.01_K_200_mass_1_R_0.00005_R_n_0.001_erate_0.0002.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.01_K_200_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_100_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_100_mass_1_R_0.00005_R_n_0.001_erate_0.0001.dump"
dump_realisation_name="eq_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_10_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"


dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_10_mass_1_R_0.00005_R_n_0.001_erate_0.0001.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_10_mass_1_R_0.00005_R_n_0.001_erate_0.0002.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_10_mass_1_R_0.00005_R_n_0.001_erate_0.0003.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_10_mass_1_R_0.0005_R_n_0.001_erate_0.0001.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_10_mass_1_R_0.0005_R_n_0.001_erate_0.0002.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.1_K_10_mass_1_R_0.0005_R_n_0.001_erate_0.0003.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_1_K_10_mass_1_R_0.0005_R_n_0.001_erate_0.0001.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_1_K_10_mass_1_R_0.0005_R_n_0.001_erate_0.0002.dump"
#dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_1_K_10_mass_1_R_0.0005_R_n_0.001_erate_0.0003.dump"

dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_1_K_20_mass_1_R_0.0005_R_n_0.001_erate_0.0003.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.01_K_100_mass_1_R_0.0005_R_n_0.01_erate_0.0003.dump"
dump_realisation_name="eq_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.01_K_100_mass_1_R_0.0005_R_n_0.01_erate_0.0003.dump"
dump_realisation_name="shear_db_hooke_tensorlgeq_tstep_1.00000250e-03_T_0.01_K_100_mass_1_R_0.0005_R_n_0.01_erate_0.0003.dump"


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

box_size=100
meancut=40
for l in range(0,6):

    stress=np.sum(spring_force_positon_array,axis=1)/(box_size**3)
    mean_stress=np.mean(spring_force_positon_array[:],axis=1)
    end_grad=np.mean(np.gradient(stress[:,l]))
    print("SS_mean",np.mean(stress[-meancut:,l]))
   

    plt.plot(stress[10:,l],label=f"end grad = {end_grad:.5f}")
#plt.ylim(-0.1,2)
plt.legend()
plt.show()
print("N_1",np.mean(stress[-meancut:,0])-np.mean(stress[-meancut:,2]))
print("N_2",np.mean(stress[-meancut:,2])-np.mean(stress[-meancut:,1]))

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
timestep_skip_array=[50,80,150,190]

timestep_skip_array=[0,100,200,400,499]
#timestep_skip_array=[0,5,10,50]
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

adjust_factor = 0.025

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
        data=np.ravel(spring_extension_array[m, :])-0.001,label=f"{timestep_skip_array[l]}, mean bond ext = {np.mean(np.ravel(spring_extension_array[m, :]) - 0.05):.3f}",
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



# %%
for l in range(len(timestep_skip_array)):
    m = timestep_skip_array[l]
    theta = np.ravel(spherical_coords[ :l, :, 1])
    phi=np.ravel(spherical_coords[ :l, :, 2])
    combine=np.array([theta,phi])
    plt.scatter(theta,phi,s=0.5)
    plt.xlabel("$\Theta$")
    plt.ylabel("$\phi$")
    #plt.title(f"$K={K}, \\dot{{\\gamma}}={erate[i]}$")

plt.show()


# plt.hist2d(phi, theta, bins=50, cmap='viridis')
# plt.colorbar(label="Counts")
# plt.ylabel(r"$\Theta$")
# plt.xlabel(r"$\phi$")
# plt.title(f"$K={K}, \\dot{{\\gamma}}={erate[i]}$")
# plt.show()

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
    sxx = -sxx_sum / volume
    syy = -syy_sum / volume
    szz = -szz_sum / volume
    sxy = -sxy_sum / volume
    sxz = -sxz_sum / volume
    syz = -syz_sum / volume

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
        plt.title('Normal Stress Components')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time, sxy, label=r'$\sigma_{xy}$')
        plt.plot(time, sxz, label=r'$\sigma_{xz}$')
        plt.plot(time, syz, label=r'$\sigma_{yz}$')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normalized Shear Stress')
        plt.legend()
        plt.grid(True)
        plt.title('Shear Stress Components')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(time, N1, label=r'$N_1 = \sigma_{xx} - \sigma_{zz}$')
        plt.plot(time, N2, label=r'$N_2 = \sigma_{zz} - \sigma_{yy}$')
       
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Normal Stress Differences')
        plt.legend()
        plt.grid(True)
        plt.title('Normal Stress Differences')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        #plt.plot(time, N1/N2, label=r'$N1/N2$')
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

file_name="stress_tensor_avg_1.00000250e-05_T_0.01_K_10_mass_10_R_0.5_R_n_1_erate_0.3.dat"
file_name="stress_tensor_avg_1.00000250e-06_T_0.01_K_10_mass_10_R_0.5_R_n_1_erate_0.3.dat"
data_03=analyze_raw_stress_data(filename=file_name, volume=100**3, show_plots=True, return_data=True)





# %%
file_name="stress_tensor_avg_0.0005_T_0.01_K_0.5_mass_10_R_0.5_R_n_1_erate_0.0006723357536499335.dat"
data_03=analyze_raw_stress_data(filename=file_name, volume=100**3, show_plots=True, return_data=True)


# %%
file_name="stress_tensor_avg_5e-05_T_0.01_K_0.1_mass_10_R_0.5_R_n_1_erate_0.0006723357536499335.dat"
file_name="stress_tensor_avg_5e-05_T_0.01_K_0.1_mass_10_R_0.5_R_n_1_erate_0.0006723357536499335.dat"
file_name="stress_tensor_avg_5e-05_T_0.01_K_0.1_mass_10_R_0.5_R_n_1_erate_0.006723357536499335.dat"
file_name="stress_tensor_avg_5e-05_T_0.01_K_0.1_mass_10_R_0.5_R_n_1_erate_0.0006723357536499335.dat"
analyze_raw_stress_data(filename=file_name, volume=100**3, show_plots=True, return_data=True)


# %%
file_name="stress_tensor_avg_DBshearnvt_no988576_hookean_flatchain_elastic_10_R_n_1_R_0.5_927734_4_100_1_5e-05_29700_29747_297470420_0_gdot_0.006723357536499335_BK_50_K_0.1.dat"
analyze_raw_stress_data(filename=file_name, volume=100**3, show_plots=True, return_data=True)

# %%
