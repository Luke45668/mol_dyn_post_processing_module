from file_manipulations_module import *
import numpy as np 
# import matplotlib.pyplot as plt
from plotting_module import *
from calculations_module import *

#%% constants 
K=np.array([ 60]) # internal spring stiffness
tchain=["60_30"] # string with range of thermostat variables 
n_plates=100
strain_total=250
j_=6 # number of realisations per data point in independent variable 
eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
# thermo variables for log file 
thermo_vars="         KinEng      c_spring_pe       PotEng         Press         c_myTemp        c_bias         TotEng    "


erate=np.array([1.34      , 1.34555556, 1.35111111, 1.35666667, 1.36222222,
       1.36777778, 1.37333333, 1.37888889, 1.38444444, 1.39,1.395     , 1.41222222, 1.42944444, 1.44666667, 1.46388889,
       1.48111111, 1.49833333, 1.51555556, 1.53277778, 1.55,1.6       , 1.62222222, 1.64444444])

path_2_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_250_6_reals_erate_over_1.34_comparison/"


#%% Loading in tuples 
e_end=[] # list to show where the end of the data points is for each loaded data set 
os.chdir(path_2_files)

# creating tuples to load in data 
spring_force_positon_tensor_batch_tuple=()
log_file_batch_tuple=()
log_file_real_batch_tuple=()
area_vector_spherical_batch_tuple=()
pos_batch_tuple=()
vel_batch_tuple=()


# loading in tuples 
for i in range(len(tchain)):

    label='tchain_'+str(tchain[i])+'_K_'+str(K[i])+'_'

   
    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    
    print(len( spring_force_positon_tensor_batch_tuple[i]))
    e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))

    pos_batch_tuple=pos_batch_tuple+(batch_load_tuples(label,"p_positions_tuple.pickle"),)

    vel_batch_tuple=vel_batch_tuple+(batch_load_tuples(label,"p_velocities_tuple.pickle"),)


    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    
    log_file_real_batch_tuple=log_file_real_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_real_tuple.pickle"),)
    # print(len(log_file_batch_tuple[i]))
    area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_tuple.pickle"),)
    
   
   # e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))
# %% inspecting thermo data  realisation by realisation 

n_outputs_per_log_file=1002
indep_var_1=K
indep_var_2=erate
indep_var_2_size=e_end
E_p_column_index=2
E_p_low_lim=0
E_p_up_lim=3
E_p_lim_switch=1
E_k_column_index=1
E_k_low_lim=0
E_k_up_lim=0
E_k_lim_switch=0
T_column_index=6
T_low_lim=0
T_up_lim=0
T_lim_switch=0
E_t_column_index=7
E_t_low_lim=0
E_t_up_lim=0
E_t_lim_switch=0
fig_width=30
fig_height=10
realisation_count=j_
# intialise i and j before to make title string work.
j=0
i=0

leg_x=1.1
leg_y=1
fontsize_plot=25

thermo_variables_plot_against_strain_show_all_reals_gpt(
    strain_total, n_outputs_per_log_file, indep_var_1,indep_var_2, indep_var_2_size,
    E_p_column_index, E_p_low_lim, E_p_up_lim, E_p_lim_switch,
    E_k_column_index, E_k_low_lim, E_k_up_lim, E_k_lim_switch,
    T_column_index, T_low_lim, T_up_lim, T_lim_switch,
    E_t_column_index, E_t_low_lim, E_t_up_lim, E_t_lim_switch,
    realisation_count, log_file_real_batch_tuple,fig_width, fig_height,leg_x,leg_y,fontsize_plot,tchain
)



#%% time series with all realisations 
fig_width=50
fig_height=10
n_outputs_per_stress_file=1000
stress_vars = {
        "\sigma_{xx}": (0),
        "\sigma_{yy}": (1),
        "\sigma_{zz}":  (2),
        "\sigma_{xz}": (3),
        "\sigma_{xy}": (4),
        "\sigma_{yz}": (5)
    }

stress_vars = {
        "\sigma_{xx}": (0),
        "\sigma_{yy}": (1),
        "\sigma_{zz}":  (2)
    }
ss_cut=0.6
stress_vars = {
        "\sigma_{xz}": (3),
        "\sigma_{xy}": (4),
        "\sigma_{yz}": (5) }



stress_tensor_strain_time_series( n_outputs_per_stress_file,
                                     strain_total,
                                     fontsize_plot,
                                     indep_var_1,
                                     indep_var_2,
                                     indep_var_2_size,
                                     fig_width, fig_height,
                                     spring_force_positon_tensor_batch_tuple,
                                     leg_x,leg_y, stress_vars,ss_cut,tchain)



# %% stress tensor avergaging 

trunc2=1
trunc1=0.6# or 0.4 

labels_stress=np.array(["\sigma_{xx}$",
               "\sigma_{yy}$",
               "\sigma_{zz}$",
               "\sigma_{xz}$",
               "\sigma_{xy}$",
               "\sigma_{yz}$"])



stress_tensor_tuple=()
stress_tensor_std_tuple=()

for j in range(K.size):
    stress_tensor=np.zeros((e_end[j],6))
    stress_tensor_std=np.zeros((e_end[j],6))   
    stress_tensor,stress_tensor_std=stress_tensor_averaging_batch(e_end[j],labels_stress,
                            trunc1,
                            trunc2,
                           spring_force_positon_tensor_batch_tuple[j],j_)
    
    stress_tensor_tuple=stress_tensor_tuple+(stress_tensor,)
    stress_tensor_std_tuple=stress_tensor_std_tuple+(stress_tensor_std,)



# %%
