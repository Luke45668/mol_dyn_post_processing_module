import numpy as np
import matplotlib.pyplot as plt


# generic functions for best fit lines
def quadfunc(x, a):
    return a * (x**2)


def linearfunc(x, a, b):
    return (a * x) + b


def linearthru0(x, a):
    return a * x


def powerlaw(x, a, n):
    return a * (x ** (n))


def plotting_stress_vs_strain(
    spring_force_positon_tensor_tuple,
    i_1,
    i_2,
    j_,
    strain_total,
    cut,
    aftcut,
    stress_component,
    label_stress,
    erate,
):
    mean_grad_l = []
    for i in range(i_1, i_2):
        # for j in range(j_):
        cutoff = int(
            np.round(
                cut
                * spring_force_positon_tensor_tuple[i][0, :, :, stress_component].shape[
                    0
                ]
            )
        )
        aftcutoff = int(
            np.round(
                aftcut
                * spring_force_positon_tensor_tuple[i][0, :, :, stress_component].shape[
                    0
                ]
            )
        )

        strain_plot = np.linspace(
            cut * strain_total,
            aftcut * strain_total,
            spring_force_positon_tensor_tuple[i][
                0, cutoff:aftcutoff, :, stress_component
            ].shape[0],
        )
        cutoff = int(
            np.round(
                cut
                * spring_force_positon_tensor_tuple[i][0, :, :, stress_component].shape[
                    0
                ]
            )
        )
        aftcutoff = int(
            np.round(
                aftcut
                * spring_force_positon_tensor_tuple[i][0, :, :, stress_component].shape[
                    0
                ]
            )
        )
        stress = np.mean(
            spring_force_positon_tensor_tuple[i][:, :, :, stress_component], axis=0
        )
        stress = stress[cutoff:aftcutoff]
        gradient_vec = np.gradient(np.mean(stress, axis=1))
        mean_grad = np.mean(gradient_vec)
        mean_grad_l.append(mean_grad)
        # print(stress.shape)
        # plt.plot(strain_plot,np.mean(stress,axis=1))
        # plt.ylabel(labels_stress[stress_component],rotation=0)
        # plt.xlabel("$\gamma$")
        # plt.plot(strain_plot,gradient_vec, label="$\\frac{dy}{dx}="+str(mean_grad)+"$")

        # plt.legend()
        # plt.show()

    plt.scatter(erate, mean_grad_l, label=label_stress)
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$\\frac{d\\bar{\sigma}_{\\alpha\\beta}}{dt}$", rotation=0, labelpad=20)
    # plt.show()


def plot_stress_tensor(
    t_0,
    t_1,
    stress_tensor,
    stress_tensor_std,
    j_,
    n_plates,
    labels_stress,
    marker,
    cutoff,
    erate,
    e_end,
    ls_pick,
):
    for l in range(t_0, t_1):
        plt.errorbar(
            erate[cutoff:e_end],
            stress_tensor[cutoff:, l],
            yerr=stress_tensor_std[cutoff:, l] / np.sqrt(j_ * n_plates),
            ls=ls_pick,
            label=labels_stress[l],
            marker=marker[l],
        )
        plt.xlabel("$\dot{\gamma}$")
        plt.ylabel("$\sigma_{\\alpha\\beta}$", rotation=0, labelpad=20)
    plt.legend()
    # plt.show()


def thermo_variables_plot_against_strain_show_all_reals(
    strain_total,
    n_ouputs_per_log_file,
    indep_var_1,
    indep_var_2_size,
    E_p_column_index,
    E_p_low_lim,
    E_p_up_lim,
    E_p_lim_swtich,
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
    sup_title_string,
    log_file_real_batch_tuple,
    fig_width,
    fig_height,
):
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)

    strainplot = np.linspace(0, strain_total, n_ouputs_per_log_file)
    for j in range(indep_var_1.size):
        # using size variable to account for varying ranges of erate.
        for i in range(indep_var_2_size[j]):
            for k in range(realisation_count):
                column = E_p_column_index
                plt.subplot(1, 4, 1)
                plt.plot(strainplot, log_file_real_batch_tuple[j][i][k, :, column])
                plt.xlabel("$\gamma$")
                plt.ylabel("$E_{p}$")
                if E_p_lim_swtich == 1:
                    plt.ylim(E_p_low_lim, E_p_up_lim)

                column = E_k_column_index
                plt.subplot(1, 4, 2)
                plt.plot(strainplot, log_file_real_batch_tuple[j][i][k, :, column])
                # plt.yscale('log')
                plt.xlabel("$\gamma$")
                plt.ylabel("$E_{k}$")
                if E_k_lim_switch == 1:
                    plt.ylim(E_k_low_lim, E_k_up_lim)

                column = T_column_index
                plt.subplot(1, 4, 3)
                # remove frist
                plt.plot(
                    strainplot[50:], log_file_real_batch_tuple[j][i][k, 50:, column]
                )
                plt.xlabel("$\gamma$")
                plt.ylabel("$T$")
                if T_lim_switch == 1:
                    plt.ylim(T_low_lim, T_up_lim)

                column = E_t_column_index
                plt.subplot(1, 4, 4)
                plt.plot(
                    strainplot[50:], log_file_real_batch_tuple[j][i][k, 50:, column]
                )
                plt.xlabel("$\gamma$")
                plt.ylabel("$E_{t}$")
                if E_t_lim_switch == 1:
                    plt.ylim(E_t_low_lim, E_t_up_lim)

            plt.suptitle(sup_title_string)
            # should contain indep variables for each plot
            plt.show()


# above function optimised with chat gpt
def thermo_variables_plot_against_strain_show_all_reals_gpt(
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
):
    strainplot = np.linspace(0, strain_total, n_outputs_per_log_file)
    plt.rcParams.update({"font.size": fontsize_plot})
    # Store energy variables in a dictionary for cleaner loop processing
    energy_vars = {
        "E_{p}": (E_p_column_index, E_p_low_lim, E_p_up_lim, E_p_lim_switch, ""),
        "E_{k}": (E_k_column_index, E_k_low_lim, E_k_up_lim, E_k_lim_switch, ""),
        "T": (T_column_index, T_low_lim, T_up_lim, T_lim_switch, "[50:]"),
        "E_{t}": (E_t_column_index, E_t_low_lim, E_t_up_lim, E_t_lim_switch, "[50:]"),
    }

    for j in range(indep_var_1.size):
        for i in range(indep_var_2_size[j]):
            fig, axes = plt.subplots(
                1, len(energy_vars), figsize=(fig_width, fig_height)
            )  # 1 row, 4 columns
            fig.suptitle(
                "$\dot{\gamma}="
                + str(indep_var_2[i])
                + ", K="
                + str(indep_var_1[j])
                + ",N_{c}=$"
                + str(tchain[j])
            )

            for k in range(realisation_count):
                for ax, (label, (col_idx, low_lim, up_lim, lim_switch, trim)) in zip(
                    axes, energy_vars.items()
                ):
                    y_values = log_file_real_batch_tuple[j][i][k, :, col_idx]

                    # Apply trimming (used for T and Et)
                    if trim:
                        y_values = y_values[50:]
                        x_values = strainplot[50:]
                    else:
                        x_values = strainplot

                    ax.plot(x_values, y_values, label=f"Real {k+1}")
                    ax.set_xlabel("$\gamma$")
                    ax.set_ylabel(f"${label}$")

                    # Apply limits if switch is enabled
                    if lim_switch:
                        ax.set_ylim(low_lim, up_lim)

            plt.legend(bbox_to_anchor=(leg_x, leg_y))
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
            plt.show()

def thermo_variables_plot_against_strain_show_mean_reals_gpt(
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
):
    strainplot = np.linspace(0, strain_total, n_outputs_per_log_file)
    plt.rcParams.update({"font.size": fontsize_plot})
    # Store energy variables in a dictionary for cleaner loop processing
    energy_vars = {
        "E_{p}": (E_p_column_index, E_p_low_lim, E_p_up_lim, E_p_lim_switch, ""),
        "E_{k}": (E_k_column_index, E_k_low_lim, E_k_up_lim, E_k_lim_switch, ""),
        "T": (T_column_index, T_low_lim, T_up_lim, T_lim_switch, "[50:]"),
        "E_{t}": (E_t_column_index, E_t_low_lim, E_t_up_lim, E_t_lim_switch, "[50:]"),
    }

    for j in range(indep_var_1.size):
        for i in range(indep_var_2_size[j]):
            fig, axes = plt.subplots(
                1, len(energy_vars), figsize=(fig_width, fig_height)
            )  # 1 row, 4 columns
            fig.suptitle(
                "$\dot{\gamma}="
                + str(indep_var_2[i])
                + ", K="
                + str(indep_var_1[j])
                + ",N_{c}=$"
                + str(tchain[j])
            )

            #for k in range(realisation_count):
            for ax, (label, (col_idx, low_lim, up_lim, lim_switch, trim)) in zip(
                axes, energy_vars.items()
            ):
                y_values = np.mean(log_file_real_batch_tuple[j][i][:, :, col_idx],axis=0)

                # Apply trimming (used for T and Et)
                if trim:
                    y_values = y_values[50:]
                    x_values = strainplot[50:]
                else:
                    x_values = strainplot

                ax.plot(x_values, y_values)
                ax.set_xlabel("$\gamma$")
                ax.set_ylabel(f"${label}$")

                # Apply limits if switch is enabled
                if lim_switch:
                    ax.set_ylim(low_lim, up_lim)

            plt.legend(bbox_to_anchor=(leg_x, leg_y))
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
            plt.show()


# time series plots of stress each realisation
def stress_tensor_strain_time_series(
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
):
    strainplot = np.linspace(0, strain_total, n_outputs_per_stress_file)
    plt.rcParams.update({"font.size": fontsize_plot})

    # Store stress variables in a dictionary for cleaner loop processing
    SS_grad_array = np.zeros((indep_var_1.size, indep_var_2_size[0], realisation_count))
    for j in range(indep_var_1.size):
        for i in range(indep_var_2_size[j]):
            fig, axes = plt.subplots(
                1, len(stress_vars), figsize=(fig_width, fig_height)
            )  # 1 row, n columns
            fig.suptitle(
                "$\dot{\gamma}="
                + str(indep_var_2[i])
                + ", K="
                + str(indep_var_1[j])
                + ",N_{c}=$"
                + str(tchain[j])
            )

            # compute mean stress over n_plates
            mean_stress_tensor = np.mean(
                spring_force_positon_tensor_batch_tuple[j][i], axis=2
            )

            for k in range(realisation_count):
                for ax, (label, (tensor_idx)) in zip(axes, stress_vars.items()):
                    x_values = strainplot
                    y_values = mean_stress_tensor[k, :, tensor_idx]
                    grad_cutoff = int(
                        np.round(ss_cut * mean_stress_tensor[k, :, tensor_idx].size)
                    )
                    SS_grad = np.mean(
                        np.gradient(
                            mean_stress_tensor[k, grad_cutoff:, tensor_idx], axis=0
                        )
                    )
                    SS_grad = np.around(SS_grad, 5)
                    SS_grad_array[j, i, k] = SS_grad

                    ax.plot(x_values, y_values, label=f"Real {k+1}, SS_grad={SS_grad}")
                    ax.set_xlabel("$\gamma$")
                    ax.set_ylabel(f"${label}$")
                    ax.legend(bbox_to_anchor=(leg_x, leg_y))
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
            plt.show()
    return SS_grad_array


def plot_steady_state_gradient(
    indep_var_1, indep_var_2_size, indep_var_2, SS_grad_array, j_
):
    for j in range(indep_var_1.size):
        fraction_steady = (
            np.count_nonzero(np.abs(SS_grad_array[j]) < 0.0075) / SS_grad_array[0].size
        )
        print("% of runs which reach steady state", fraction_steady * 100)
        for i in range(indep_var_2_size[j]):
            plt.scatter(
                np.arange(0, j_, 1),
                np.abs(SS_grad_array[j, i, :]),
                label="$\dot{\gamma}=" + str(indep_var_2[i]) + "$",
            )

        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()


def plot_mean_stress_tensor(
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
):
    for j in range(indep_var_1.size):
        for l in range(entry_1, entry_2):
            plt.errorbar(
                indep_var_2[: indep_var_2_size[j]],
                stress_tensor_tuple[j][:, l],
                yerr=stress_tensor_std_tuple[j][:, l] / np.sqrt(j_),
                label="$K=" + str(indep_var_1[j]) + "," + str(labels_stress[l]),
                linestyle=linestyle_tuple[j][1],
                marker=marker[j],
            )

            # plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")

            plt.xlabel("$\dot{\\gamma}$")
            plt.ylabel("$\sigma_{\\alpha \\alpha}$", rotation=0, labelpad=15)

        plt.tight_layout()

        plt.legend(frameon=False)
        # plt.savefig(path_2_log_files+"/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight')
    plt.show()


def plot_normal_stress_diff(
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
):
    from calculations_module import compute_n_stress_diff

    for j in range(indep_var_1.size):
        # plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")

        n_diff, n_diff_error = compute_n_stress_diff(
            stress_tensor_tuple[j],
            stress_tensor_std_tuple[j],
            elem_index_1,
            elem_index_2,
            j_,
        )
        plt.errorbar(
            indep_var_2[first_point_index : indep_var_2_size[j]],
            n_diff[first_point_index : indep_var_2_size[j]],
            yerr=n_diff_error[first_point_index : indep_var_2_size[j]],
            ls="none",
            label="$"
            + y_label
            + ","
            + indep_var_1_label
            + "="
            + str(indep_var_1[j])
            + "$",
            marker=marker[j],
        )

        plt.legend(fontsize=10, frameon=False)

        plt.xlabel("$" + x_label + "$")
        plt.ylabel("$" + y_label + "$", rotation=0)
        plt.tight_layout()
        # plt.savefig(path_2_log_files+"/plots/N1_vs_gdot_ybxa_plots.pdf",dpi=1200,bbox_inches='tight')
        plt.show()
