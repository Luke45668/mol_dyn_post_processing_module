# %%
from file_manipulations_module import *
import numpy as np

# import matplotlib.pyplot as plt
from plotting_module import *
from calculations_module import *
from statistical_tests import *
import seaborn as sns
import pandas as pd
from numpy import linalg

# %% constants
damp = np.array([0.035, 0.035, 0.035, 0.035])
K = np.array([60])  # internal spring stiffness
tchain = ["60"]  # string with range of thermostat variables
n_plates = 99
strain_total = 500
j_ = 25  # number of realisations per data point in independent variable
eq_spring_length = 3 * np.sqrt(3) / 2
mass_pol = 5
n_chains = 33
n_plates_per_chain = 3
# thermo variables for log file
thermo_vars = "         KinEng      c_spring_pe       PotEng         Press         c_myTemp        c_bias         TotEng    "
number_of_chains = 33
number_of_particles_per_chain = 7
linestyle_tuple = [
    ("dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dotted", (0, (1, 1))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]

marker = ["x", "+", "^", "1", "X", "d", "*", "P", "v"]


erate = np.array(
    [
        0.6,
        0.70555556,
        0.81111111,
        0.91666667,
        1.02222222,
        1.12777778,
        1.23333333,
        1.33888889,
        1.44444444,
        1.55,
    ]
)


strain_total = 500


path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_250_6_reals_erate_over_1.34_comparison/"
path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/chain_runs/strain_500_tchain_60_14_reals_med_erate/"
path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/chain_runs/shear_chain_strain_500_tdamp_250_25_reals_tchain_15_0.8_1.55_shear_rates/"
# %% Loading in tuples
e_end = []  # list to show where the end of the data points is for each loaded data set
os.chdir(path_2_files)

# creating tuples to load in data
spring_force_positon_tensor_batch_tuple = ()
log_file_batch_tuple = ()
log_file_real_batch_tuple = ()
area_vector_spherical_batch_tuple = ()
pos_batch_tuple = ()
vel_batch_tuple = ()


# loading in tuples
for i in range(len(tchain)):
    # label = "tchain_" + str(tchain[i]) + "_K_" + str(K[i]) + "_"
    label = "damp_" + str(damp[i]) + "_K_" + str(K[i]) + "_"

    spring_force_positon_tensor_batch_tuple = (
        spring_force_positon_tensor_batch_tuple
        + (batch_load_tuples(label, "spring_force_positon_tensor_tuple.pickle"),)
    )

    print(len(spring_force_positon_tensor_batch_tuple[i]))
    e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))

    pos_batch_tuple = pos_batch_tuple + (
        batch_load_tuples(label, "p_positions_tuple.pickle"),
    )

    vel_batch_tuple = vel_batch_tuple + (
        batch_load_tuples(label, "p_velocities_tuple.pickle"),
    )

    log_file_batch_tuple = log_file_batch_tuple + (
        batch_load_tuples(label, "log_file_tuple.pickle"),
    )

    log_file_real_batch_tuple = log_file_real_batch_tuple + (
        batch_load_tuples(label, "log_file_real_tuple.pickle"),
    )
    print(len(log_file_batch_tuple[i]))
    area_vector_spherical_batch_tuple = area_vector_spherical_batch_tuple + (
        batch_load_tuples(label, "area_vector_tuple.pickle"),
    )


# e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))
# %% inspecting thermo data  realisation by realisation

n_outputs_per_log_file = 1002
indep_var_1 = K
indep_var_2 = erate
indep_var_2_size = e_end
E_p_column_index = 2
E_p_low_lim = 0
E_p_up_lim = 6
E_p_lim_switch = 1
E_k_column_index = 1
E_k_low_lim = 0
E_k_up_lim = 0
E_k_lim_switch = 0
T_column_index = 6
T_low_lim = 0
T_up_lim = 0
T_lim_switch = 0
E_t_column_index = 7
E_t_low_lim = 0
E_t_up_lim = 0
E_t_lim_switch = 0
fig_width = 60
fig_height = 25
realisation_count = j_
# intialise i and j before to make title string work.
j = 0
i = 0

leg_x = 1.1
leg_y = 1
fontsize_plot = 25

thermo_variables_plot_against_strain_show_all_reals_gpt(
    strain_total,
    n_outputs_per_log_file,
    indep_var_1,
    indep_var_2,
    indep_var_2_size,
    E_p_column_index,
    E_p_low_lim,
    E_p_up_lim,
    E_p_lim_switch,
    E_k_column_index,
    E_k_low_lim,
    E_k_up_lim,
    E_k_lim_switch,
    T_column_index,
    T_low_lim,
    T_up_lim,
    T_lim_switch,
    E_t_column_index,
    E_t_low_lim,
    E_t_up_lim,
    E_t_lim_switch,
    realisation_count,
    log_file_real_batch_tuple,
    fig_width,
    fig_height,
    leg_x,
    leg_y,
    fontsize_plot,
    tchain,
)


# %% time series with all realisations
fontsize_plot = 30
fig_width = 50
fig_height = 20
n_outputs_per_stress_file = 1000
stress_vars = {
    "\sigma_{xx}": (0),
    "\sigma_{yy}": (1),
    "\sigma_{zz}": (2),
    "\sigma_{xz}": (3),
    "\sigma_{xy}": (4),
    "\sigma_{yz}": (5),
}

stress_vars = {"\sigma_{xx}": (0), "\sigma_{yy}": (1), "\sigma_{zz}": (2)}
ss_cut = 0.6
# stress_vars = {"\sigma_{xz}": (3), "\sigma_{xy}": (4), "\sigma_{yz}": (5)}

SS_grad_array = stress_tensor_strain_time_series(
    n_outputs_per_stress_file,
    strain_total,
    fontsize_plot,
    indep_var_1,
    indep_var_2,
    indep_var_2_size,
    fig_width,
    fig_height,
    spring_force_positon_tensor_batch_tuple,
    leg_x,
    leg_y,
    stress_vars,
    ss_cut,
    tchain,
    realisation_count,
)

# %% plot SS gradient array

plot_steady_state_gradient(
    indep_var_1, indep_var_2_size, indep_var_2, SS_grad_array, j_
)

# %% stress tensor avergaging

trunc2 = 1
trunc1 = 0.4  # or 0.4

labels_stress = np.array(
    [
        "\sigma_{xx}$",
        "\sigma_{yy}$",
        "\sigma_{zz}$",
        "\sigma_{xz}$",
        "\sigma_{xy}$",
        "\sigma_{yz}$",
    ]
)


stress_tensor_tuple, stress_tensor_std_tuple = stress_tensor_tuple_store(
    indep_var_1,
    indep_var_2_size,
    labels_stress,
    trunc1,
    trunc2,
    spring_force_positon_tensor_batch_tuple,
    j_,
)

# %% stress tensor mean plots
plt.rcParams.update({"font.size": 10})


# normal stresses
entry_1 = 0
entry_2 = 3
plot_mean_stress_tensor(
    indep_var_1,
    indep_var_2,
    indep_var_2_size,
    entry_1,
    entry_2,
    stress_tensor_tuple,
    stress_tensor_std_tuple,
    labels_stress,
    j_,
    marker,
    linestyle_tuple,
)

# shear stresses
entry_1 = 3
entry_2 = 6
plot_mean_stress_tensor(
    indep_var_1,
    indep_var_2,
    indep_var_2_size,
    entry_1,
    entry_2,
    stress_tensor_tuple,
    stress_tensor_std_tuple,
    labels_stress,
    j_,
    marker,
    linestyle_tuple,
)


# N1
elem_index_1 = 0
elem_index_2 = 2
y_label = "N_{1}"
x_label = "\dot{\gamma}"
first_point_index = 0
indep_var_1_label = "K"


plot_normal_stress_diff(
    indep_var_1,
    indep_var_2,
    indep_var_2_size,
    stress_tensor_tuple,
    stress_tensor_std_tuple,
    elem_index_1,
    elem_index_2,
    j_,
    first_point_index,
    y_label,
    x_label,
    marker,
    indep_var_1_label,
)

# N2
elem_index_1 = 2
elem_index_2 = 1
y_label = "N_{2}"
plot_normal_stress_diff(
    indep_var_1,
    indep_var_2,
    indep_var_2_size,
    stress_tensor_tuple,
    stress_tensor_std_tuple,
    elem_index_1,
    elem_index_2,
    j_,
    first_point_index,
    y_label,
    x_label,
    marker,
    indep_var_1_label,
)


# %% angle plots constants
plt.rcParams.update({"font.size": 10})
pi_theta_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
pi_theta_tick_labels = ["-π", "-π/2", "0", "π/2", "π"]
pi_phi_ticks = [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2]
pi_phi_tick_labels = ["0", "π/8", "π/4", "3π/8", "π/2"]


spherical_coords_tuple = ()
sample_cut = 0
cutoff = 0
sample_size = 500

# high shear rate
skip_array = [1, 5, 7, 9]
# low shear rate
skip_array = [0, 10, 20, 30, 36]

# %% time series plot of  phi
cutoff = 0
timestep_skip_array = [0, 5, 10, 100, 200, 500, 900]
steady_state_index = 800
adjust_factor = 1
# for j in range(1,K.size):
for j in range(0, 1):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
    )
    p_value_array = np.zeros((len(skip_array), j_, len(timestep_skip_array)))
    KS_stat_array = np.zeros((len(skip_array), j_, len(timestep_skip_array)))
    for i in range(len(skip_array)):
        k = skip_array[i]
        data = spherical_coords_tuple[i][:, :, :, 2]
        periodic_data = np.array([data, np.pi - data])

        for n in range(j_):
            steady_state_dist = np.ravel(periodic_data[:, n, steady_state_index:, :])

            for l in range(len(timestep_skip_array)):
                m = timestep_skip_array[l]
                # timstep_dist=np.ravel(periodic_data[:,n,m,:])
                timstep_dist = np.ravel(periodic_data[:, :, m, :])
                KS_stat, p_value = generic_stat_kolmogorov_2samp(
                    steady_state_dist, timstep_dist
                )
                p_value_array[i, n, l] = p_value
                KS_stat_array[i, n, l] = KS_stat
            #     sns.kdeplot(
            #         data=np.ravel(periodic_data[:, :, m, :]),
            #         label="$N_{t}=" + str(timestep_skip_array[l]) + "$",
            #         linestyle=linestyle_tuple[j][1],
            #         bw_adjust=adjust_factor,
            #     )
            #     # plt.title("$\dot{\gamma}="+str(erate[skip_array[i]])+"$, real="+str(n))
            #     plt.title("$\dot{\gamma}=" + str(erate[skip_array[i]]) + "$")

            # plt.xlabel("$\Phi$")
            # plt.xticks(pi_phi_ticks, pi_phi_tick_labels)

            # # plt.yticks(phi_y_ticks)
            # plt.ylabel("Density")
            # plt.legend(bbox_to_anchor=(1, 0.5), frameon=False)
            # plt.xlim(0, np.pi / 2)
            # # plt.xlim(0,np.pi)
            # plt.tight_layout()
            # # plt.savefig(path_2_log_files+"/plots/phi_dist_.pdf",dpi=1200,bbox_inches='tight')
            # plt.show()
# %%plotting KS statistics
for j in range(len(skip_array)):
    k = skip_array[j]

    for i in range(j_):
        plt.plot(
            timestep_skip_array,
            KS_stat_array[j, i],
            label="real=" + str(i),
            marker=marker[i],
        )
    plt.ylabel("KS difference ")
    plt.xlabel("output count")
    plt.title("$\dot{\gamma}=" + str(erate[k]) + "$")
    plt.legend()
    plt.show()

    for i in range(j_):
        plt.plot(
            timestep_skip_array,
            p_value_array[j, i],
            label="real=" + str(i),
            marker=marker[i],
        )

    plt.ylabel("P value")
    plt.xlabel("output count")
    plt.title("$\dot{\gamma}=" + str(erate[k]) + "$")

    plt.legend()
    plt.show()
# %% different style plot of phi
n_chains = 33
n_plates_per_chain = 3
skip_array = [1, 2, 4, 7]
cutoff = 0
timestep_skip_array = [0, 5, 10, 100, 200, 500, 900]
adjust_factor = 0.5
for j in range(0, 2):
    reshaped_area_vector_array = area_vector_spherical_batch_tuple
    spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
        j_,
        j,
        skip_array,
        area_vector_spherical_batch_tuple,
        n_plates,
        cutoff,
        n_chains,
        n_plates_per_chain,
    )

    for i in range(len(skip_array)):
        k = skip_array[i]

        data = spherical_coords_tuple[i][:, :, :, 2]

        periodic_data = np.array([data, np.pi - data])

        for l in range(len(timestep_skip_array)):
            m = timestep_skip_array[l]
            sns.kdeplot(
                data=np.ravel(periodic_data[:, :, m, :]),
                label="$N_{t}=" + str(timestep_skip_array[l]) + "$",
                linestyle=linestyle_tuple[j][1],
                bw_adjust=adjust_factor,
            )
            plt.title(
                "$\dot{\gamma}=" + str(erate[skip_array[i]]) + ",K=" + str(K[j]) + "$"
            )

        plt.xlabel("$\phi$")
        plt.xticks(pi_phi_ticks, pi_phi_tick_labels)
        plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

        plt.ylabel("Density")
        plt.xlim(0, np.pi / 2)

        # plt.xlim(0,np.pi)
        plt.tight_layout()
        # plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
        plt.show()

# %% different style plot of theta

# theta
adjust_factor = 0.25
for j in range(0, 2):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
        j_,
        j,
        skip_array,
        area_vector_spherical_batch_tuple,
        n_plates,
        cutoff,
        n_chains,
        n_plates_per_chain,
    )

    for i in range(len(skip_array)):
        k = skip_array[i]

        data = spherical_coords_tuple[i][:, :, :, 1]

        periodic_data = np.array([data - 2 * np.pi, data, data + 2 * np.pi])

        for l in range(len(timestep_skip_array)):
            m = timestep_skip_array[l]
            sns.kdeplot(
                data=np.ravel(periodic_data[:, :, m, :]),
                label="$N_{t}=" + str(timestep_skip_array[l]) + "$",
                linestyle=linestyle_tuple[j][1],
                bw_adjust=adjust_factor,
            )
            plt.title(
                "$\dot{\gamma}=" + str(erate[skip_array[i]]) + ",K=" + str(K[j]) + "$"
            )

        plt.xlabel("$\Theta$")
        plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
        plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

        plt.ylabel("Density")
        plt.xlim(-np.pi, np.pi)

        # plt.xlim(0,np.pi)
        plt.tight_layout()
        # plt.savefig(path_2_log_files+"/plots/theta_dist_.pdf",dpi=1200,bbox_inches='tight')
        plt.show()

# %%pplot of rho
# should make a subplot at each erate for both K
# theta
adjust_factor = 1
for j in range(0, 2):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
    )

    for i in range(len(skip_array)):
        k = skip_array[i]

        data = spherical_coords_tuple[i][:, :, :, 0]

        # periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])

        for l in range(len(timestep_skip_array)):
            m = timestep_skip_array[l]
            sns.kdeplot(
                data=np.ravel(data[:, m, :]),
                label="$N_{t}=" + str(timestep_skip_array[l]) + "$",
                linestyle=linestyle_tuple[l][1],
                bw_adjust=adjust_factor,
            )
            plt.title("$\dot{\gamma}=" + str(erate[skip_array[i]]) + "$")

        plt.xlabel("$\\rho$")

        plt.legend(bbox_to_anchor=(1, 0.55), frameon=False)

        plt.ylabel("Density")

        # plt.xlim(0,np.pi)
        plt.tight_layout()

        plt.show()
# %% theta against phi plot
size_marker = 0.000005
for i in range(len(skip_array)):
    k = skip_array[i]
    # for j in range(0, 2):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
        j_,
        0,
        skip_array,
        area_vector_spherical_batch_tuple,
        n_plates,
        cutoff,
        n_chains,
        n_plates_per_chain,
    )

    plt.subplot(1, 2, 1)
    theta = np.ravel(spherical_coords_tuple[i][:, :, :, :, 1])
    phi = np.ravel(spherical_coords_tuple[i][:, :, :, :, 2])

    plt.scatter(
        theta,
        phi,
        s=size_marker,
        label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[0]) + "$",
    )

    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)
    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")

    plt.legend(bbox_to_anchor=(0.85, 1.2))

    #     spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
    #     j_, 1, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff, n_chains,n_plates_per_chain
    # )

    #     plt.subplot(1, 2, 2)
    #     theta = np.ravel(spherical_coords_tuple[i][:, :, :, 1])
    #     phi = np.ravel(spherical_coords_tuple[i][:, :, :, 2])
    #     plt.scatter(
    #         theta,
    #         phi,
    #         s=size_marker,
    #         label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[1]) + "$",
    #     )
    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)

    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)

    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")
    plt.legend(bbox_to_anchor=(0.85, 1.2))

    plt.show()

# %% theta against phi plot specific particles
size_marker = 0.00005
particle = 2
for i in range(len(skip_array)):
    k = skip_array[i]
    # for j in range(0, 2):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
        j_,
        0,
        skip_array,
        area_vector_spherical_batch_tuple,
        n_plates,
        cutoff,
        n_chains,
        n_plates_per_chain,
    )
    particle = 0
    plt.subplot(1, 3, 1)
    theta = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 1])
    phi = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 2])

    plt.scatter(
        theta,
        phi,
        s=size_marker,
        label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[0]) + "$",
    )

    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)
    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")

    plt.legend(bbox_to_anchor=(0.85, 1.2))

    #     spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
    #     j_, 1, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff, n_chains,n_plates_per_chain
    # )

    plt.subplot(1, 3, 2)
    particle = 1
    theta = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 1])
    phi = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 2])
    plt.scatter(
        theta,
        phi,
        s=size_marker,
        label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[0]) + "$",
    )
    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)

    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)

    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")
    plt.legend(bbox_to_anchor=(0.85, 1.2))

    plt.subplot(1, 3, 3)
    particle = 2
    theta = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 1])
    phi = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 2])
    plt.scatter(
        theta,
        phi,
        s=size_marker,
        label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[0]) + "$",
    )
    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)

    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)

    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")
    plt.legend(bbox_to_anchor=(0.85, 1.2))

    plt.show()

# %% theta against phi plot 1 and 3 together with 2 separate
size_marker = 0.00005
particle = 2
for i in range(len(skip_array)):
    k = skip_array[i]
    # for j in range(0, 2):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
        j_,
        0,
        skip_array,
        area_vector_spherical_batch_tuple,
        n_plates,
        cutoff,
        n_chains,
        n_plates_per_chain,
    )
    particle = 0
    plt.subplot(1, 2, 1)
    theta = np.ravel(
        np.array(
            [
                spherical_coords_tuple[i][:, :, :, 0, 1],
                spherical_coords_tuple[i][:, :, :, 2, 1],
            ]
        )
    )
    phi = np.ravel(
        np.array(
            [
                spherical_coords_tuple[i][:, :, :, 0, 2],
                spherical_coords_tuple[i][:, :, :, 2, 2],
            ]
        )
    )

    plt.scatter(
        theta,
        phi,
        s=size_marker,
        label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[0]) + "$",
    )

    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)
    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")

    plt.legend(bbox_to_anchor=(0.85, 1.2))

    #     spherical_coords_tuple = convert_cart_2_spherical_z_inc_chain(
    #     j_, 1, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff, n_chains,n_plates_per_chain
    # )

    plt.subplot(1, 2, 2)
    particle = 1
    theta = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 1])
    phi = np.ravel(spherical_coords_tuple[i][:, :, :, particle, 2])
    plt.scatter(
        theta,
        phi,
        s=size_marker,
        label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[0]) + "$",
    )
    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)

    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)

    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")
    plt.legend(bbox_to_anchor=(0.85, 1.2))

    plt.show()


# %% computing gyration tensor


SS_output_index = 400

j = 0
Gyration_square_matrix = np.zeros((e_end[0], 3, 3))
eigen_data = []
for i in range(e_end[0]):
    Gyration_square_matrix[i, 0, 0] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        0,
        0,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # xx

    Gyration_square_matrix[i, 1, 1] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        1,
        1,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # yy

    Gyration_square_matrix[i, 2, 2] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        2,
        2,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # zz

    Gyration_square_matrix[i, 0, 2] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        0,
        2,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # xz

    Gyration_square_matrix[i, 2, 0] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        2,
        0,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # xz

    Gyration_square_matrix[i, 0, 1] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        0,
        1,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # xy

    Gyration_square_matrix[i, 1, 0] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        1,
        0,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # xy

    Gyration_square_matrix[i, 2, 1] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        1,
        2,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # yz

    Gyration_square_matrix[i, 1, 2] = compute_gyration_tensor_in_loop(
        pos_batch_tuple,
        2,
        1,
        j,
        i,
        number_of_particles_per_chain,
        mass_pol,
        number_of_chains,
        SS_output_index,
    )  # yz

    eigen_numbers = np.linalg.eig(Gyration_square_matrix[i])
    eigen_data.append(eigen_numbers)


# plot the eigen values for each
colour = ["blue", "orange", "green"]
ordered_eigenvalue_matrix = np.zeros((3, e_end[j]))
ordered_eigenvectors_array = np.zeros((e_end[j], 3, 3))
for i in range(e_end[j]):
    eigen_values = np.flip(np.sort(eigen_data[i][0]))
    # now find each associated eigenvector
    for l in range(3):
        eigvect_index = np.where(eigen_values[l] == eigen_data[i][0])[0][0]
        # check if largest is negative
        abs_vector = np.abs(eigen_data[i][1][eigvect_index])
        max_component = np.max(abs_vector)
        max_component_index = np.where(abs_vector == max_component)[0][0]
        if eigen_data[i][1][eigvect_index][max_component_index] < 0:
            ordered_eigenvectors_array[i, l, :] = -1 * eigen_data[i][1][eigvect_index]
        else:
            ordered_eigenvectors_array[i, l, :] = eigen_data[i][1][eigvect_index]

    ordered_eigenvalue_matrix[0, i] = eigen_values[0]
    ordered_eigenvalue_matrix[1, i] = eigen_values[1]
    ordered_eigenvalue_matrix[2, i] = eigen_values[2]


value_index = ["$\lambda_{1}$", "$\lambda_{2}$", "$\lambda_{3}$"]
# plotting eigenvalues vs shear rate

for l in range(3):
    plt.scatter(
        erate[: e_end[0]],
        ordered_eigenvalue_matrix[l],
        label=value_index[l],
        color=colour[l],
    )
plt.yscale("log")
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$\lambda_{i}$")
plt.legend(bbox_to_anchor=(1, 0.8))
plt.show()
compontent_index = ["$x$", "$y$", "$z$"]

vector_index = [
    "$\mathbf{\lambda}_{1}$",
    "$\mathbf{\lambda}_{2}$",
    "$\mathbf{\lambda}_{3}$",
]
for k in range(3):  # loop through eigenvalues
    for l in range(3):  # loop through components of eigenvectors
        plt.scatter(
            erate[: e_end[0]],
            ordered_eigenvectors_array[:, k, l],
            label=compontent_index[l],
            marker=marker[l],
            color=colour[k],
        )

    plt.title(vector_index[k])
    plt.xlabel("$\dot{\gamma}$")
    plt.legend()
    plt.show()


# %%

alpha = 0
beta = 0
j = 0
i = 0
positions = pos_batch_tuple[j][i][:, :, ...]

particle_COM = np.sum(mass_pol * positions, axis=3) / (
    number_of_particles_per_chain * mass_pol
)
S_alpha_beta = np.zeros((25, 1000, number_of_chains, number_of_particles_per_chain))
for i in range(number_of_chains):
    for j in range(number_of_particles_per_chain):
        # S_alpha_beta=np.mean(np.sum((positions[...,i,j,alpha]-particle_COM[...,i,alpha])*(positions[...,i,j,beta]-particle_COM[...,i,beta])))

        S_alpha_beta[:, :, i, j] = (
            positions[..., i, j, alpha] - particle_COM[..., i, alpha]
        ) * (positions[..., i, j, beta] - particle_COM[..., i, beta])


S_alpha_beta = np.mean(np.sum(S_alpha_beta, axis=3))
#  print(np.sum((positions[...,i,j,alpha]-particle_COM[...,j,alpha])*(positions[...,i,j,beta]-particle_COM[...,j,beta])))


# S_alpha_beta=np.mean(np.sum((alpha_positions-particle_COM_alpha)*(beta_positions-particle_COM_beta)))

# %%
