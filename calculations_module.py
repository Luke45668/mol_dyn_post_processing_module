import numpy as np


def stress_tensor_averaging(
    e_end, labels_stress, trunc1, trunc2, spring_force_positon_tensor_tuple, j_
):
    stress_tensor = np.zeros((e_end, 6))
    stress_tensor_std = np.zeros((e_end, 6))
    stress_tensor_reals = np.zeros((e_end, j_, 6))
    stress_tensor_std_reals = np.zeros((e_end, j_, 6))
    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff = int(
                    np.round(
                        trunc1
                        * spring_force_positon_tensor_tuple[i][j, :, :, l].shape[0]
                    )
                )
                aftercutoff = int(
                    np.round(
                        trunc2
                        * spring_force_positon_tensor_tuple[i][j, :, :, l].shape[0]
                    )
                )
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0])
                # print(cutoff)
                # print(aftercutoff)
                data = np.ravel(
                    spring_force_positon_tensor_tuple[i][j, cutoff:aftercutoff, :, l]
                )
                stress_tensor_reals[i, j, l] = np.mean(data)
                stress_tensor_std_reals[i, j, l] = np.std(data)
            stress_tensor = np.mean(stress_tensor_reals, axis=1)
            stress_tensor_std = np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor, stress_tensor_std


def stress_tensor_averaging_var_trunc(
    e_end, labels_stress, trunc1, trunc2, spring_force_positon_tensor_tuple, j_
):
    stress_tensor = np.zeros((e_end, 6))
    stress_tensor_std = np.zeros((e_end, 6))
    stress_tensor_reals = np.zeros((e_end, j_, 6))
    stress_tensor_std_reals = np.zeros((e_end, j_, 6))
    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff = int(
                    np.round(
                        trunc1[i]
                        * spring_force_positon_tensor_tuple[i][j, :, :, l].shape[0]
                    )
                )
                aftercutoff = int(
                    np.round(
                        trunc2[i]
                        * spring_force_positon_tensor_tuple[i][j, :, :, l].shape[0]
                    )
                )
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0])
                # print(cutoff)
                # print(aftercutoff)
                data = np.ravel(
                    spring_force_positon_tensor_tuple[i][j, cutoff:aftercutoff, :, l]
                )
                stress_tensor_reals[i, j, l] = np.mean(data)
                stress_tensor_std_reals[i, j, l] = np.std(data)
                stress_tensor = np.mean(stress_tensor_reals, axis=1)
                stress_tensor_std = np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor, stress_tensor_std


def compute_n_stress_diff(
    stress_tensor,
    stress_tensor_std,
    i1,
    i2,
    j_,
    n_plates,
):
    n_diff = stress_tensor[:, i1] - stress_tensor[:, i2]
    n_diff_error = np.sqrt(
        stress_tensor_std[:, i1] ** 2 + stress_tensor_std[:, i2] ** 2
    ) / np.sqrt(j_ * n_plates)

    return n_diff, n_diff_error



