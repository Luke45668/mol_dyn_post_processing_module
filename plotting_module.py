import numpy as np
import matplotlib.pyplot as plt
import os 


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
    tchain,n_particles,n_steps
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
                    # check total energy drift

                    if label=="E_{t}":
                        relative_drift=(y_values[0]-y_values[-1])/y_values[0]
                        print("relative drift per atom per step=",relative_drift/(n_particles*n_steps))

                       

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
    tchain,n_particles,n_steps
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
                if label=="E_{t}":
                        relative_drift=(y_values[0]-y_values[-1])/y_values[0]
                        print("relative drift per atom per strain unit=",relative_drift/(n_particles*n_steps))


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
    stress_tensor_all_reals,
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
            mean_stress_tensor = stress_tensor_all_reals[j,i]
            

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
                   # ax.legend(bbox_to_anchor=(leg_x, leg_y))
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


def plot_stats_vs_indepvar_log_file(stats_array, timestep,xlabel, column_names, use_latex=True, gradient_threshold=1e-4, save=True, save_dir="plots"):
    """
    Plots stress mean and gradient mean vs timestep with std as error bars using stacked subplots.
    Highlights convergence points and saves plots if requested.
    """

    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    n_cols = stats_array.shape[0]

    for col in range(n_cols):
        means = stats_array[col, :, 0]
        stds = stats_array[col, :, 1]

        grad_means = stats_array[col, :, 2]
        grad_stds = stats_array[col, :, 3]

        converged = np.abs(grad_means) < gradient_threshold

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[1, 1])

        # --- Top: Steady State Mean ---
        ax1.errorbar(timestep, means, yerr=stds, fmt='o-', capsize=4, linewidth=2, color='tab:blue', label="Steady State Mean ± Std")
        ax1.set_ylabel("Steady State Mean", color='tab:blue')
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.legend(loc='best',bbox_to_anchor=(1,1))

        # --- Bottom: Gradient Mean ---
        ax2.errorbar(timestep, grad_means, yerr=grad_stds, fmt='s--', capsize=4, linewidth=2, color='black', markersize=5, label="Gradient Mean ± Std")
        ax2.fill_between(timestep, -gradient_threshold, gradient_threshold, color='yellow', alpha=0.2, label='Tolerance Band')
        
        ax2.plot(np.array(timestep)[converged], grad_means[converged], 'o', markersize=10,
                 markerfacecolor='gold', markeredgecolor='black', markeredgewidth=1.5,
                 label=r'Converged ($|\mathrm{grad}| < \mathrm{tol}$)')
       
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Gradient Mean", color='black')
        ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.legend(loc='best',bbox_to_anchor=(1,1))

        # --- Title and layout ---
        fig.suptitle(rf"\textbf{{{column_names[col]}}}", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88, hspace=0.35)

        # --- Save ---
        save_string = column_names[col].replace(' ', '_').replace('$', '').replace('\\', '')
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}_stats.png"
            plt.savefig(fname, dpi=300)

        plt.show()


def plot_stress_components(
    erate,
    K_list,
    time_mean_stress_data_array_all_K,
    time_std_stress_data_array_all_K,
    stress_columns,
    i_range=(1, 4),
    fit_type=None,                # 'linear', 'quadratic', or None
    fit_index=None,              # column index to fit
    fit_points=None,             # list or array of indices for fitting, e.g., [0, 1, 2]
    save=False,
    save_path="plots/stress_components.png"
):
    """
    Plots stress components with error bars, with optional linear (through origin) or quadratic fit.

    Parameters:
        Wi (array): Weissenberg numbers (x-axis)
        time_mean_stress (2D array): shape (N, M), stress means
        time_std_stress (2D array): shape (N, M), stress stds
        stress_columns (list): list of column names for the stress components
        i_range (tuple): (start, end) indices of columns to plot [start, end)
        fit_type (str or None): 'linear', 'quadratic', or None
        fit_index (int or None): which column index to fit, must be in i_range if fitting is desired
        fit_points (list or None): list of indices to use for fitting (subset of data points)
        save (bool): whether to save the plot
        save_path (str): filepath to save the figure
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # LaTeX-style plot settings
    plt.rcParams.update({
        "text.usetex": "True",
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    linestyle=['--','-']

    for j in range(len(K_list)):
        time_mean_stress= time_mean_stress_data_array_all_K[j]
        time_std_stress=time_std_stress_data_array_all_K[j]
        for i in range(*i_range):
            ax.errorbar(
                erate,
                time_mean_stress[:, i],
                yerr=time_std_stress[:, i],
                label=rf"{stress_columns[i]}, K={K_list[j]}",
                capsize=3,
                marker='o',
                linestyle="none",
                linewidth=1.5
            )
            ax.grid(True, linestyle='--', alpha=0.3)

            if fit_type and i == fit_index:
                x = np.array(erate)
                y = time_mean_stress[:, i]

                if fit_points is not None:
                    x = x[fit_points]
                    y = y[fit_points]

                if fit_type == 'linear':
                    # Least squares fit through origin: y = m*x
                    m = np.dot(x, y) / np.dot(x, x)
                    fit_y = m * np.array(erate)
                    label = rf"Linear fit: {stress_columns[i]} $= {m:.3g} \dot{{\gamma}}, K={K_list[j]}$"

                elif fit_type == 'quadratic':
                    A = np.vstack([x**2, np.ones_like(x)]).T
                    a, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                    fit_y = a * np.array(erate)**2
                    label = rf"Quadratic fit: {stress_columns[i]} $= {a:.3g}\dot{{\gamma}}^{2}, K={K_list[j]}$"

                else:
                    raise ValueError("Invalid fit_type. Use 'linear', 'quadratic', or None.")

                ax.plot(
                    erate,
                    fit_y,
                    linestyle[j],
                    color='black',
                    linewidth=1,
                    label=label
                )

    ax.set_xlabel(rf"$\dot{{\gamma}}$")
    ax.legend(frameon=False)
    ax.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()
    plt.close()


def plot_time_series_eq_converge(data, erate, column_names,output_cutoff,success_index_list, use_latex=True, save=False, save_dir="plots"):
    """
    Plots time series data for each column, showing all timesteps on the same graph.
    Adds mean, std deviation (over last 60%), and gradient stats to legend and stores them in an array.

    Returns:
        stats_array (ndarray): shape (n_cols, n_timestep, 4) → [mean, std, mean_grad, std_grad] for each timestep and column
    """

    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    n_erate,n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")

    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    stats_array = np.zeros((n_cols, n_erate, 4))

    for col in range(n_cols):
        plt.figure(figsize=(10, 5))

        for j in range(len(success_index_list)):
            i=success_index_list[j]

            y = data[i, :, col]
            number_of_steps=np.linspace(0,(1e-5*1e8)*(output_cutoff/1000),y.shape[0])
            

            # Last 60% of the signal
            last_60_percent = y[int(0.4 * len(y)):]

            # Compute mean and std
            mean = np.mean(last_60_percent)
            std = np.std(last_60_percent)

            # Compute gradient
            gradients = np.gradient(last_60_percent)

            mean_grad = np.mean(gradients)
            std_grad = np.std(gradients)

            # Store stats
            stats_array[col, i, 0] = mean
            stats_array[col, i, 1] = std
            stats_array[col, i, 2] = mean_grad
            stats_array[col, i, 3] = std_grad

            # Plot
            plt.plot(number_of_steps,y, label=rf"erate ${erate[i]:.2f}$", linewidth=1.5)

        plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("$t/\\tau$")
        plt.ylabel(rf"\textbf{{{column_names[col]}}}")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{column_names[col].replace(' ', '_')}.png"
            plt.savefig(fname, dpi=300)

        plt.show()

    return stats_array

def plot_time_series_shear_converge(data, erate, column_names,output_cutoff,total_strain,success_index_list,shear_outs, use_latex=True, save=False, save_dir="plots"):
    """
    Plots time series data for each column, showing all timesteps on the same graph.
    Adds mean, std deviation (over last 60%), and gradient stats to legend and stores them in an array.

    Returns:
        stats_array (ndarray): shape (n_cols, n_timestep, 4) → [mean, std, mean_grad, std_grad] for each timestep and column
    """

    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    n_erate,n_steps, n_cols = data.shape
    cmap = plt.get_cmap("tab10")

    # Prepare stats storage → now 4 columns (mean, std, mean_grad, std_grad)
    stats_array = np.zeros((n_cols, n_erate, 4))

    for col in range(n_cols):
        plt.figure(figsize=(10, 5))

        for j in range(len(success_index_list)):
            i=success_index_list[j]
            y = data[i, :, col]
            number_of_steps=np.linspace(0,(shear_outs/1000)*total_strain,y.shape[0])
            

            # Last 60% of the signal
            last_60_percent = y[int(0.6 * len(y)):]

            # Compute mean and std
            mean = np.mean(last_60_percent)
            std = np.std(last_60_percent)

            # Compute gradient
            gradients = np.gradient(last_60_percent)

            mean_grad = np.mean(gradients)
            std_grad = np.std(gradients)

            # Store stats
            stats_array[col, i, 0] = mean
            stats_array[col, i, 1] = std
            stats_array[col, i, 2] = mean_grad
            stats_array[col, i, 3] = std_grad

            # Plot
            plt.plot(number_of_steps,y, label=rf"erate ${erate[i]:.2f}$", linewidth=1.5)

       # plt.title(rf"\textbf{{{column_names[col]}}}")
        plt.xlabel("$\gamma$")
        plt.ylabel(rf"\textbf{{{column_names[col]}}}",rotation=0, labelpad=10)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        
        save_string = column_names[col].replace(' ', '_').replace('$', '').replace('\\', '')
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}.png"
            plt.savefig(fname, dpi=300)

        plt.show()

    return stats_array

def plot_stats_vs_erate_log_file(
    stats_array,
    erate,
    column_names,
    success_index_list,
    use_latex=True,
    gradient_threshold=1e-4,
    save=False,
    save_dir="plots"
):
    """
    Plots gradient mean ± std vs shear rate (erate) from stats_array, for successful indices only.
    Highlights convergence points (|grad| < threshold) and optionally saves figures.

    Parameters:
        stats_array: shape (n_columns, n_cases, 4), where:
                     [:, :, 2] = grad_mean, [:, :, 3] = grad_std
        erate: array of shear rates
        column_names: list of names corresponding to stress components (one per column)
        success_index_list: list of indices to include from the data (subset of cases)
        use_latex: bool, whether to render using LaTeX
        gradient_threshold: float, convergence threshold for gradient
        save: bool, whether to save the plots
        save_dir: str, folder to save plots
    """

    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    # Subset erate and stats array
    erate_sub = np.array(erate)[success_index_list]
    stats_sub = stats_array[:, success_index_list, :]  # shape: [n_cols, len(success), 4]

    n_cols = stats_sub.shape[0]

    for col in range(n_cols):
        fig, ax = plt.subplots(figsize=(8, 5))

        grad_means = stats_sub[col, :, 2]
        grad_stds = stats_sub[col, :, 3]

        # Plot gradient mean ± std
        ax.errorbar(
            erate_sub,
            grad_means,
            yerr=grad_stds,
            fmt='s--',
            capsize=4,
            linewidth=2,
            color='black',
            markersize=5
        )

        # Highlight convergence points
        converged = np.abs(grad_means) < gradient_threshold
        ax.plot(
            erate_sub[converged],
            grad_means[converged],
            'o',
            markersize=12,
            markerfacecolor='gold',
            markeredgecolor='black',
            markeredgewidth=1.5,
            label='Converged (|grad| < tol)'
        )

        # Labels and styling
        ax.set_xlabel(r"$\dot{\gamma}$")
        ax.set_ylabel(r"Gradient Mean", color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.set_xscale('log')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        handles = [
            plt.Line2D([], [], color='black', marker='s', linestyle='--', linewidth=2, label="Gradient Mean ± Std"),
            plt.Line2D([], [], color='gold', marker='o', markeredgecolor='black', linestyle='None', markersize=10, label="Converged ($|\\mathrm{grad}| < \\mathrm{tol}$)")
        ]
        ax.legend(handles=handles, loc='best', fontsize=11, frameon=False, bbox_to_anchor=(1, 1))

        plt.title(rf"\textbf{{{column_names[col]}}} - Gradient vs $\dot{{\gamma}}$")
        fig.tight_layout()

        # Save if needed
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{column_names[col].replace(' ', '_')}_gradient_vs_erate.png"
            fig.savefig(fname, dpi=300)
            print(f"Saved: {fname}")

        plt.show()
        plt.close()
# plot_time_series(mean_shear_log_data_array, erate,shear_columns)

# plot_time_series(mean_eq_log_data_array,erate,eq_columns)

def plot_time_series_shear_comparison(
    data_list,                   # list of arrays (n_erate, n_steps, n_cols)
    erate,                       # single array of erate values, same for all datasets
    column_names,                # list of column names
    total_strain,
    shear_outs,
    dataset_labels=None,         # optional list of labels for the datasets
    use_latex=True,
    save=False,
    save_dir="plots"
):
    """
    Plots time series data for each column.
    For multiple datasets (e.g. 3), makes subplots in the same figure per column for easy comparison.
    All erate indices are plotted; no filtering by success indices.
    """

    # Plot styling
    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    n_datasets = len(data_list)
    n_cols = data_list[0].shape[2]

    if dataset_labels is None:
        dataset_labels = [f"Dataset {i+1}" for i in range(n_datasets)]

    # Check consistency
    for data in data_list:
        if data.shape[2] != n_cols:
            raise ValueError("All datasets must have the same number of columns")

    # For each column, make one figure with subplots
    for col in range(n_cols):
        fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), sharey=True)

        # Handle case of only 1 dataset
        if n_datasets == 1:
            axes = [axes]

        for d in range(n_datasets):
            data = data_list[d]
            n_erate = data.shape[0]
            ax = axes[d]

            for i in range(n_erate):
                y = data[i, :, col]
                number_of_steps = np.linspace(0, (shear_outs / 1000) * total_strain, y.shape[0])
                ax.plot(number_of_steps, y, label=rf"$\dot\gamma={erate[i]:.2f}$", linewidth=1.5)

            ax.set_title(rf"\textbf{{{dataset_labels[d]}}}")
            ax.set_xlabel(r"$\gamma$")
            # if d == 0:
            #     ax.set_ylabel(rf"\textbf{{{column_names[col]}}}",rotation=0)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
         
         
        axes[n_datasets-1].legend(loc='best',bbox_to_anchor=(1,1))
        fig.suptitle(rf"\textbf{{{column_names[col]}}}", fontsize=20)
       # plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tight_layout()

        save_string = column_names[col].replace(' ', '_').replace('$', '').replace('\\', '')
        if save:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"{save_dir}/{save_string}_time_series_comparison.png"
            plt.savefig(fname, dpi=300)

        plt.show()