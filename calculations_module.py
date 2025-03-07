import numpy as np


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


def stress_tensor_averaging_batch(
    e_end, labels_stress, trunc1, trunc2, spring_force_positon_tensor_batch_tuple, j_
):
    stress_tensor = np.zeros((e_end, 6))
    stress_tensor_std = np.zeros((e_end, 6))
    stress_tensor_reals = np.zeros((e_end, j_, 6))
    stress_tensor_std_reals = np.zeros((e_end, j_, 6))

    # Pre-calculate the cutoff and aftercutoff for all indices outside of loops to avoid recalculation in each iteration.
    trunc1_cutoff = int(
        np.round(trunc1 * spring_force_positon_tensor_batch_tuple[0][0].shape[0])
    )
    trunc2_aftercutoff = int(
        np.round(trunc2 * spring_force_positon_tensor_batch_tuple[0][0].shape[0])
    )

    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff = trunc1_cutoff
                aftercutoff = trunc2_aftercutoff

                # Use the slicing directly
                data = spring_force_positon_tensor_batch_tuple[i][j][
                    cutoff:aftercutoff, :, l
                ].ravel()

                # Compute mean and std
                stress_tensor_reals[i, j, l] = np.mean(data)
                stress_tensor_std_reals[i, j, l] = np.std(data)

    # Now compute the mean over the second axis (axis=1)
    stress_tensor = np.mean(stress_tensor_reals, axis=1)
    stress_tensor_std = np.mean(stress_tensor_std_reals, axis=1)

    return stress_tensor, stress_tensor_std


# convert cartesian area vector to spherical and reflect all points into upper hemisphere


def convert_cart_2_spherical_z_inc(
    j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
):
    spherical_coords_tuple = ()
    for i in range(len(skip_array)):
        i = skip_array[i]

        area_vector_ray = area_vector_spherical_batch_tuple[j][i]
        area_vector_ray[area_vector_ray[:, :, :, 2] < 0] *= -1

        x = area_vector_ray[:, cutoff:, :, 0]
        y = area_vector_ray[:, cutoff:, :, 1]
        z = area_vector_ray[:, cutoff:, :, 2]

        spherical_coords_array = np.zeros(
            (j_, area_vector_ray.shape[1] - cutoff, n_plates, 3)
        )

        # radial coord
        spherical_coords_array[:, :, :, 0] = np.sqrt((x**2) + (y**2) + (z**2))

        #  theta coord
        spherical_coords_array[:, :, :, 1] = np.sign(y) * np.arccos(
            x / (np.sqrt((x**2) + (y**2)))
        )

        # spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
        # spherical_coords_array[:,:,:,1]=np.arctan(y/x)

        # phi coord
        # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
        spherical_coords_array[:, :, :, 2] = np.arccos(
            z / np.sqrt((x**2) + (y**2) + (z**2))
        )

        spherical_coords_tuple = spherical_coords_tuple + (spherical_coords_array,)

    return spherical_coords_tuple
