# %%
from file_manipulations_module import *
import numpy as np

# import matplotlib.pyplot as plt
from plotting_module import *
from calculations_module import *
from statistical_tests import *
import seaborn as sns
import pandas as pd

# %% constants
damp = np.array([0.035, 0.035, 0.035, 0.035])
K = np.array([60, 120])  # internal spring stiffness
tchain = ["60_30", "60_30"]  # string with range of thermostat variables
n_plates = 100
strain_total = 250
j_ = 10  # number of realisations per data point in independent variable
eq_spring_length = 3 * np.sqrt(3) / 2
mass_pol = 5
# thermo variables for log file
thermo_vars = "         KinEng      c_spring_pe       PotEng         Press         c_myTemp        c_bias         TotEng    "

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
        1.34,
        1.34555556,
        1.35111111,
        1.35666667,
        1.36222222,
        1.36777778,
        1.37333333,
        1.37888889,
        1.38444444,
        1.39,
        1.395,
        1.41222222,
        1.42944444,
        1.44666667,
        1.46388889,
        1.48111111,
        1.49833333,
        1.51555556,
        1.53277778,
        1.55,
        1.6,
        1.62222222,
        1.64444444,
    ]
)
# high shear rate run
erate = np.array([1.345, 1.3575, 1.37, 1.3825, 1.395, 1.4, 1.4375, 1.475, 1.5125, 1.55])
erate = np.array(
    [
        0,
        0.00388889,
        0.00777778,
        0.01166667,
        0.01555556,
        0.01944444,
        0.02333333,
        0.02722222,
        0.03111111,
        0.035,
        0.07,
        0.13894737,
        0.20789474,
        0.27684211,
        0.34578947,
        0.41473684,
        0.48368421,
        0.55263158,
        0.62157895,
        0.69052632,
        0.75947368,
        0.82842105,
        0.89736842,
        0.96631579,
        1.03526316,
        1.10421053,
        1.17315789,
        1.24,
        1.24210526,
        1.25444444,
        1.26888889,
        1.28333333,
        1.29777778,
        1.31105263,
        1.31222222,
        1.32666667,
        1.34111111,
        1.35555556,
        1.37,
        1.38,
        1.4,
        1.42222222,
        1.44444444,
        1.46666667,
        1.48888889,
        1.51111111,
        1.53333333,
        1.55555556,
        1.57777778,
        1.6,
    ]
)


strain_total = 250


path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_250_6_reals_erate_over_1.34_comparison/"
path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_250_10_reals_erate_up_to_1.34/"
# path_2_files = "/Users/luke_dev/Documents/MYRIAD_lammps_runs/nvt_runs/shear_runs/strain_500_15_reals_over_erate_1.345/"

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

    # log_file_real_batch_tuple = log_file_real_batch_tuple + (
    #     batch_load_tuples(label, "log_file_real_tuple.pickle"),
    # )
    print(len(log_file_batch_tuple[i]))
    area_vector_spherical_batch_tuple = area_vector_spherical_batch_tuple + (
        batch_load_tuples(label, "area_vector_tuple.pickle"),
    )


# e_end.append(len(spring_force_positon_tensor_batch_tuple[i]))
# %% inspecting thermo data  realisation by realisation
e_end = [37, 37]
n_outputs_per_log_file = 1002
indep_var_1 = K
indep_var_2 = erate
indep_var_2_size = e_end
E_p_column_index = 2
E_p_low_lim = 0
E_p_up_lim = 4
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

stress_vars = {"\sigma_{xx}": (0), "\sigma_{yy}": (1)}
ss_cut = 0.6
# stress_vars = {"\sigma_{xz}": (3), "\sigma_{xy}": (4)}
# "\sigma_{zz}": (2)
# "\sigma_{yz}": (5)
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

for j in range(K.size):
    fraction_steady = (
        np.count_nonzero(np.abs(SS_grad_array[j]) < 0.0075) / SS_grad_array[0].size
    )
    print(fraction_steady * 100)
    for i in range(e_end[j]):
        plt.scatter(
            np.arange(0, j_, 1),
            np.abs(SS_grad_array[j, i, :]),
            label="$\dot{\gamma}=" + str(erate[i]) + "$",
        )

    plt.legend()
    plt.show()


# %% stress tensor avergaging

trunc2 = 1
trunc1 = 0.6  # or 0.4

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


stress_tensor_tuple = ()
stress_tensor_std_tuple = ()

for j in range(K.size):
    stress_tensor = np.zeros((e_end[j], 6))
    stress_tensor_std = np.zeros((e_end[j], 6))
    stress_tensor, stress_tensor_std = stress_tensor_averaging_batch(
        e_end[j],
        labels_stress,
        trunc1,
        trunc2,
        spring_force_positon_tensor_batch_tuple[j],
        j_,
    )

    stress_tensor_tuple = stress_tensor_tuple + (stress_tensor,)
    stress_tensor_std_tuple = stress_tensor_std_tuple + (stress_tensor_std,)


# %% stress tensor mean plots
plt.rcParams.update({"font.size": 10})
#### note need to turn into function

for j in range(K.size):
    for l in range(3):
        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])

        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1], marker=marker[j])
        plt.errorbar(
            erate[: e_end[j]],
            stress_tensor_tuple[j][:, l],
            yerr=stress_tensor_std_tuple[j][:, l] / np.sqrt(j_),
            label="$K=" + str(K[j]) + "," + str(labels_stress[l]),
            linestyle=linestyle_tuple[j][1],
            marker=marker[j],
        )

        # plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")

        plt.xlabel("$\dot{\\gamma}$")
        plt.ylabel("$\sigma_{\\alpha \\alpha}$", rotation=0, labelpad=15)
        # plt.yticks(y_ticks_stress)
        # plt.ylim(0.9,1.3)

    plt.tight_layout()
    # plt.xscale('log')

    plt.legend(frameon=False)
    # plt.savefig(path_2_log_files+"/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()

for j in range(K.size):
    for l in range(3, 6):
        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[0])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])

        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]),ls=linestyle_tuple[j], marker=marker[j])
        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$tdamp="+str(thermal_damp_multiplier[j])+","+str(labels_stress[l]), marker=marker[j])
        # plt.plot(erate[:e_end[j]],stress_tensor_tuple[j][:,l],label="$K="+str(K[j])+","+str(labels_stress[l]),linestyle=linestyle_tuple[j][1], marker=marker[j])
        plt.errorbar(
            erate[: e_end[j]],
            stress_tensor_tuple[j][:, l],
            yerr=stress_tensor_std_tuple[j][:, l] / np.sqrt(j_),
            label="$K=" + str(K[j]) + "," + str(labels_stress[l]),
            linestyle=linestyle_tuple[j][1],
            marker=marker[j],
        )

        # plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")

        plt.xlabel("$\dot{\\gamma}$")
        plt.ylabel("$\sigma_{\\alpha \\alpha}$", rotation=0, labelpad=15)
        # plt.yticks(y_ticks_stress)
        # plt.ylim(0.9,1.3)

    plt.tight_layout()
    # plt.xscale('log')

    plt.legend(frameon=False)
    # plt.savefig(path_2_log_files+"/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
# %%now plot n1 vs erate with y=ax^2
# probably need to turn this into a a function
n_y_ticks = [-10, 0, 20, 40, 60, 80]
cutoff = 0
quadratic_end = 8
# plt.plot(0,0,marker='none',label="fit: $y=ax^{2}$",linestyle='none')
for j in range(K.size):
    # plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")

    n_1, n_1_error = compute_n_stress_diff(
        stress_tensor_tuple[j],
        stress_tensor_std_tuple[j],
        0,
        2,
        j_,
        n_plates,
    )
    plt.errorbar(
        erate[cutoff : e_end[j]],
        n_1[cutoff : e_end[j]],
        yerr=n_1_error[cutoff : e_end[j]],
        ls="none",
        label="$N_{1},K=" + str(K[j]) + "$",
        marker=marker[j],
    )
    # plt.plot(erate[cutoff:e_end[j]], n_1[cutoff:e_end[j]],
    #               ls="none",label="$N_{1},K="+str(K[j])+"$",marker=marker[j] )


plt.legend(fontsize=10, frameon=False)
# plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{1}$", rotation=0)
# plt.yticks(n_y_ticks)
plt.tight_layout()
# plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_ybxa_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()

# %%now plot n2 vs erate with y=ax^2
# probably need to turn this into a a function
n_y_ticks = [-10, 0, 20, 40, 60, 80]
cutoff = 0
quadratic_end = 8
# plt.plot(0,0,marker='none',label="fit: $y=ax^{2}$",linestyle='none')
for j in range(K.size):
    # plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")

    n_2, n_2_error = compute_n_stress_diff(
        stress_tensor_tuple[j],
        stress_tensor_std_tuple[j],
        2,
        1,
        j_,
        n_plates,
    )
    plt.errorbar(
        erate[cutoff : e_end[j]],
        n_2[cutoff : e_end[j]],
        yerr=n_2_error[cutoff : e_end[j]],
        ls="none",
        label="$N_{2},K=" + str(K[j]) + "$",
        marker=marker[j],
    )


plt.legend(fontsize=10, frameon=False)
# plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{2}$", rotation=0)
# plt.yticks(n_y_ticks)
plt.tight_layout()
# plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_ybxa_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
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


adjust_factor = 1
for j in range(0, 2):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
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
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
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
size_marker = 0.0005
for i in range(len(skip_array)):
    k = skip_array[i]
    # for j in range(0, 2):
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, 0, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
    )

    plt.subplot(1, 2, 1)
    theta = np.ravel(spherical_coords_tuple[i][:, :, :, 1])
    phi = np.ravel(spherical_coords_tuple[i][:, :, :, 2])

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

    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, 1, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
    )

    plt.subplot(1, 2, 2)
    theta = np.ravel(spherical_coords_tuple[i][:, :, :, 1])
    phi = np.ravel(spherical_coords_tuple[i][:, :, :, 2])
    plt.scatter(
        theta,
        phi,
        s=size_marker,
        label="$\dot{\gamma}=" + str(erate[k]) + ", K=" + str(K[1]) + "$",
    )
    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)

    plt.xticks(pi_theta_ticks, pi_theta_tick_labels)

    plt.ylim(0, np.pi / 2)
    plt.xlim(-np.pi, np.pi)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\Theta$")
    plt.legend(bbox_to_anchor=(0.85, 1.2))

    plt.show()

# %% violin plot of phi

adjust_factor = 2
erate_1 = 0
erate_2 = 10
plt.rcParams["figure.figsize"] = (25, 6)
for j in range(K.size):
    skip_array = np.arange(erate_1, erate_2, 1)
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
    )
    periodic_data_list = []
    erate_list = []
    for i in range(skip_array.size):
        data = np.ravel(
            spherical_coords_tuple[i][:, :, :, 2]
        )  # Assuming this extracts the spherical data
        periodic_data = np.ravel(
            np.array([data, np.pi - data])
        )  # Handling the periodic nature
        periodic_data_list.append(periodic_data)

    # Convert lists to DataFrames at the end
    periodic_data_df = pd.DataFrame(periodic_data_list)
    periodic_data_df = periodic_data_df.transpose()
    erate_str = np.around(erate[erate_1:erate_2], 3).astype("str")
    periodic_data_df.columns = erate_str
    print(periodic_data_df.isna().sum())
    # erate_df = pd.DataFrame(erate[:e_end[j]])
    # full_df = pd.concat([erate_df, periodic_data_df], axis=0)
    # full_df = full_df.rename(columns={full_df.columns[0]: "erate"})

    # # rename columns 1 to end
    # full_df.columns = full_df.columns[:1].tolist() + [f"part_angle" for i in range(1, len(full_df.columns))]

    # # Combine both DataFrames into a final DataFrame

    sns.violinplot(data=periodic_data_df, inner=None, linewidth=0, scale="width")
    plt.yticks(pi_phi_ticks, pi_phi_tick_labels)
    plt.ylim(0, np.pi / 2)
    plt.ylabel("$\Phi$")
    plt.xlabel("$\dot{\gamma}$")
    plt.show()

# %% violin plot of theta
adjust_factor = 0.005
erate_1 = 0
erate_2 = 10

for j in range(0, K.size):
    skip_array = np.arange(erate_1, erate_2, 1)
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
    )
    periodic_data_list = []
    erate_list = []
    for i in range(skip_array.size):
        data = np.ravel(
            spherical_coords_tuple[i][:, :, :, 1]
        )  # Assuming this extracts the spherical data
        periodic_data = np.ravel(
            np.array([data, np.pi - data])
        )  # Handling the periodic nature
        periodic_data_list.append(periodic_data)

    # Convert lists to DataFrames at the end
    periodic_data_df = pd.DataFrame(periodic_data_list)
    periodic_data_df = periodic_data_df.transpose()

    erate_str = erate[erate_1:erate_2].astype("str")
    periodic_data_df.columns = erate_str
    print(periodic_data_df.isna().sum())
    # erate_df = pd.DataFrame(erate[:e_end[j]])
    # full_df = pd.concat([erate_df, periodic_data_df], axis=0)
    # full_df = full_df.rename(columns={full_df.columns[0]: "erate"})

    # # rename columns 1 to end
    # full_df.columns = full_df.columns[:1].tolist() + [f"part_angle" for i in range(1, len(full_df.columns))]

    # # Combine both DataFrames into a final DataFrame

    sns.violinplot(data=periodic_data_df, inner=None, linewidth=0, scale="width")
    plt.ylim(-np.pi, np.pi)
    plt.yticks(pi_theta_ticks, pi_theta_tick_labels)
    plt.ylabel("$\Theta$")
    plt.xlabel("$\dot{\gamma}$")
    plt.show()
# %% violin plot of rho
adjust_factor = 0.005
erate_1 = 0
erate_2 = 10

for j in range(0, K.size):
    skip_array = np.arange(erate_1, erate_2, 1)
    spherical_coords_tuple = convert_cart_2_spherical_z_inc(
        j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
    )
    periodic_data_list = []
    erate_list = []
    for i in range(skip_array.size):
        data = np.ravel(
            spherical_coords_tuple[i][:, :, :, 0]
        )  # Assuming this extracts the spherical data
        periodic_data = np.ravel(np.array([data]))  # Handling the periodic nature
        periodic_data_list.append(periodic_data)

    # Convert lists to DataFrames at the end
    periodic_data_df = pd.DataFrame(periodic_data_list)
    periodic_data_df = periodic_data_df.transpose()

    erate_str = erate[erate_1:erate_2].astype("str")
    periodic_data_df.columns = erate_str
    print(periodic_data_df.isna().sum())
    # erate_df = pd.DataFrame(erate[:e_end[j]])
    # full_df = pd.concat([erate_df, periodic_data_df], axis=0)
    # full_df = full_df.rename(columns={full_df.columns[0]: "erate"})

    # # rename columns 1 to end
    # full_df.columns = full_df.columns[:1].tolist() + [f"part_angle" for i in range(1, len(full_df.columns))]

    # # Combine both DataFrames into a final DataFrame

    sns.violinplot(data=periodic_data_df, inner=None, linewidth=0, scale="width")
    plt.ylim(0, 12)
    # plt.yticks(pi_theta_ticks, pi_theta_tick_labels)
    plt.ylabel("$\\rho$")
    plt.xlabel("$\dot{\gamma}$")
    plt.show()
# %%
