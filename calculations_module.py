import numpy as np


def compute_n_stress_diff(stress_tensor, stress_tensor_std, i1, i2, j_):
    n_diff = stress_tensor[:, i1] - stress_tensor[:, i2]
    n_diff_error = np.sqrt(
        stress_tensor_std[:, i1] ** 2 + stress_tensor_std[:, i2] ** 2
    ) / np.sqrt(j_)

    return n_diff, n_diff_error



def compute_stress_tensor(K,erate,n_outputs_per_stress_file,spring_force_positon_tensor_batch_tuple,box_size,j_):

    stress_tensor_real_average=np.zeros((K.size,erate.size,n_outputs_per_stress_file,6))
    stress_tensor_all_reals=np.zeros((K.size,erate.size,j_,n_outputs_per_stress_file,6))
    for j in range(K.size):
        for i in range(erate.size):
            spring_pos_tensor=spring_force_positon_tensor_batch_tuple[j][i]
            # take volume average 
            stress_tensor_all_reals[j,i]=np.sum(spring_pos_tensor,axis=2)/(box_size**3)
            print( stress_tensor_all_reals.shape)
            # take realisation mean 
            stress_tensor_real_average[j,i]=np.mean(stress_tensor_all_reals[j,i],axis=0)
            print( stress_tensor_real_average.shape)

    return  stress_tensor_real_average,  stress_tensor_all_reals


def block_average_stress(n_blocks,stress_tensor_real_average):
    time_len = stress_tensor_real_average.shape[2]
    print(time_len)
    block_size = time_len // n_blocks  
    if block_size == 0:
       raise ValueError("Not enough time steps for block averaging.")
    trim_len = block_size * n_blocks
    print(trim_len)
    stress_trimmed =  stress_tensor_real_average[:,:, :trim_len, :]  # shape: (erate, time, component)

    # Reshape and compute block-wise time series
    stress_blocks = stress_trimmed.reshape((stress_trimmed.shape[0],stress_trimmed.shape[1], n_blocks, block_size, stress_trimmed.shape[3]))
    stress_time_series = stress_blocks.mean(axis=3)  # shape: (erate, n_blocks, component)
    stress_time_series_std=stress_blocks.std(axis=3)
    print("Block-averaged stress time series shape:", stress_time_series.shape)

    return stress_time_series,stress_time_series_std


def stress_tensor_averaging_batch(
    e_end, labels_stress, trunc1, trunc2, spring_force_positon_tensor_batch_tuple, j_,volume
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
                print(data.shape)

                # compute mean by summing over particles then dividing by volume 

                # Compute mean and std
                stress_tensor_reals[i, j, l] = np.sum(data)/(volume*j_)
                stress_tensor_std_reals[i, j, l] = np.std(data)

    # Now compute the mean over the second axis (axis=1)
    stress_tensor = np.mean(stress_tensor_reals, axis=1)
    stress_tensor_std = np.mean(stress_tensor_std_reals, axis=1)

    return stress_tensor, stress_tensor_std

def compute_stress_tensor(K,erate,n_outputs_per_stress_file,spring_force_positon_tensor_batch_tuple,box_size,j_):

    stress_tensor_real_average=np.zeros((K.size,erate.size,n_outputs_per_stress_file,6))
    stress_tensor_all_reals=np.zeros((K.size,erate.size,j_,n_outputs_per_stress_file,6))
    for j in range(K.size):
        for i in range(erate.size):
            spring_pos_tensor=spring_force_positon_tensor_batch_tuple[j][i]
            # take volume average 
            stress_tensor_all_reals[j,i]=np.sum(spring_pos_tensor,axis=2)/(box_size**3)
            print( stress_tensor_all_reals.shape)
            # take realisation mean 
            stress_tensor_real_average[j,i]=np.mean(stress_tensor_all_reals[j,i],axis=0)
            print( stress_tensor_real_average.shape)
    return  stress_tensor_real_average,  stress_tensor_all_reals



def stress_tensor_tuple_store(
    indep_var_1,
    indep_var_2_size,
    labels_stress,
    trunc1,
    trunc2,
    spring_force_positon_tensor_batch_tuple,
    j_,
):
    stress_tensor_tuple = ()
    stress_tensor_std_tuple = ()

    for j in range(indep_var_1.size):
        stress_tensor = np.zeros((indep_var_2_size[j], 6))
        stress_tensor_std = np.zeros((indep_var_2_size[j], 6))
        stress_tensor, stress_tensor_std = stress_tensor_averaging_batch(
            indep_var_2_size[j],
            labels_stress,
            trunc1,
            trunc2,
            spring_force_positon_tensor_batch_tuple[j],
            j_,
        )

        stress_tensor_tuple = stress_tensor_tuple + (stress_tensor,)
        stress_tensor_std_tuple = stress_tensor_std_tuple + (stress_tensor_std,)

    return stress_tensor_tuple, stress_tensor_std_tuple


# convert cartesian area vector to spherical and reflect all points into upper hemisphere


def convert_cart_2_spherical_z_inc(
    j_, j, skip_array, area_vector_spherical_batch_tuple, n_plates, cutoff
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


def convert_cart_2_spherical_z_inc_chain(
    j_,
    j,
    skip_array,
    area_vector_spherical_batch_tuple,
    n_plates,
    cutoff,
    n_chains,
    n_plates_per_chain,
):
    spherical_coords_tuple = ()
    for i in range(len(skip_array)):
        i = skip_array[i]

        area_vector_ray = area_vector_spherical_batch_tuple[j][i]
        area_vector_ray[area_vector_ray[:, :, :, :, 2] < 0] *= -1

        x = area_vector_ray[:, cutoff:, :, :, 0]
        y = area_vector_ray[:, cutoff:, :, :, 1]
        z = area_vector_ray[:, cutoff:, :, :, 2]

        spherical_coords_array = np.zeros(
            (j_, area_vector_ray.shape[1] - cutoff, n_chains, n_plates_per_chain, 3)
        )

        # radial coord
        spherical_coords_array[:, :, :, :, 0] = np.sqrt((x**2) + (y**2) + (z**2))

        #  theta coord
        spherical_coords_array[:, :, :, :, 1] = np.sign(y) * np.arccos(
            x / (np.sqrt((x**2) + (y**2)))
        )

        # spherical_coords_array[:,:,:,1]=np.sign(x)*np.arccos(y/(np.sqrt((x**2)+(y**2))))
        # spherical_coords_array[:,:,:,1]=np.arctan(y/x)

        # phi coord
        # print(spherical_coords_array[spherical_coords_array[:,:,:,0]==0])
        spherical_coords_array[:, :, :, :, 2] = np.arccos(
            z / np.sqrt((x**2) + (y**2) + (z**2))
        )

        spherical_coords_tuple = spherical_coords_tuple + (spherical_coords_array,)

    return spherical_coords_tuple


# computes the alphabeta element of the tensor
def compute_gyration_tensor_in_loop(
    pos_batch_tuple,
    alpha,
    beta,
    j,
    i,
    number_of_particles_per_chain,
    mass_pol,
    number_of_chains,
    SS_output_index,
):
    # positions=pos_batch_tuple[j][i][:,300:800,...]
    positions = pos_batch_tuple[j][i][:, SS_output_index:, ...]

    particle_COM = np.sum(mass_pol * positions, axis=3) / (
        number_of_particles_per_chain * mass_pol
    )
    S_alpha_beta = np.zeros((25, 600, number_of_chains, number_of_particles_per_chain))
    for i in range(number_of_chains):
        for j in range(number_of_particles_per_chain):
            # S_alpha_beta=np.mean(np.sum((positions[...,i,j,alpha]-particle_COM[...,i,alpha])*(positions[...,i,j,beta]-particle_COM[...,i,beta])))

            S_alpha_beta[:, :, i, j] = (
                positions[..., i, j, alpha] - particle_COM[..., i, alpha]
            ) * (positions[..., i, j, beta] - particle_COM[..., i, beta])

    S_alpha_beta = np.mean(np.sum(S_alpha_beta, axis=3))

    return S_alpha_beta


def convert_cart_2_spherical_z_inc_DB(
    j_, j, skip_array, spring_dump_batch_tuple, n_plates, cutoff
):
    spherical_coords_tuple = ()
    for i in range(len(skip_array)):
        i = skip_array[i]

        area_vector_ray = spring_dump_batch_tuple[j][i]
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