"""
This module contains functions which read lammps files and perform manipulations to the data
"""
import os as os
import Counter
import numpy as np


def dump2numpy_tensor_1tstep(
    dump_start_line,
    Path_2_dump,
    dump_realisation_name,
    number_of_particles_per_dump,
    lines_per_dump,
    cols_per_dump,
):
    os.chdir(Path_2_dump)  # +simulation_file+"/" +filename

    with open(dump_realisation_name, "r") as file:
        lines = file.readlines()

        counter = Counter(lines)

        # print(counter.most_common(3))
        n_outs = int(counter["ITEM: TIMESTEP\n"])
        dump_outarray = np.zeros((n_outs, lines_per_dump, cols_per_dump))
        # print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
        skip_spacing = lines_per_dump + 9
        skip_array = np.arange(1, len(lines), skip_spacing)
        for i in range(n_outs):
            k = skip_array[i]
            # timestep_list=[]
            start = k - 1
            end = start + skip_spacing
            timestep_list = lines[start:end]
            data_list = timestep_list[9:]
            # print(data_list[0])
            # print(len(data_list))
            data = np.zeros((lines_per_dump, cols_per_dump))
            for j in range(len(data_list)):
                data[j, :] = data_list[j].split(" ")[0:cols_per_dump]

            dump_outarray[i, :, :] = data

    return dump_outarray


def dump2numpy_box_coords_1tstep(Path_2_dump, dump_realisation_name, lines_per_dump):
    os.chdir(Path_2_dump)  # +simulation_file+"/" +filename

    with open(dump_realisation_name, "r") as file:
        lines = file.readlines()

        counter = Counter(lines)

        # print(counter.most_common(3))
        n_outs = int(counter["ITEM: TIMESTEP\n"])
        # dump_outarray=np.zeros((n_outs,lines_per_dump,cols_per_dump))
        # print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
        skip_spacing = lines_per_dump + 9
        # print(skip_spacing)
        skip_array = np.arange(1, len(lines), skip_spacing)
        data_list = []
        for i in range(n_outs):
            k = skip_array[i]

            # timestep_list=[]
            start = k - 1
            end = start + skip_spacing
            timestep_list = lines[start:end]

            data_list.append(timestep_list[:9])
            # print(data_list)

            # print(data_list[0])
            # print(len(data_list))
            # data=np.zeros((lines_per_dump,cols_per_dump))
            # for j in range(len(data_list)):
            #     data[j,:]=data_list[j].split(" ")[0:cols_per_dump]

            # dump_outarray[i,:,:]=data

    return data_list


def cfg2numpy_coords(
    Path_2_dump, dump_realisation_name, number_of_particles_per_dump, cols_per_dump
):
    os.chdir(Path_2_dump)  # +simulation_file+"/" +filename

    with open(dump_realisation_name, "r") as file:
        lines = file.readlines()
        box_vec_lines = lines[2:11]
        # print(box_vec_lines[0])
        box_vec_array = np.zeros((9))

        for i in range(9):
            box_vec_array[i] = box_vec_lines[i].split(" ")[2]
            # print(coord)

        box_vec_array = np.reshape(box_vec_array, (3, 3))

        lines = lines[15:]  # remove intial file lines

        for i in range(len(lines) - 1):
            # print(lines)
            try:
                lines.remove("C \n")
                lines.remove("5.000000 \n")

            except:
                continue

            # print(lines)
            # print(i)

        dump_outarray = np.zeros((number_of_particles_per_dump, cols_per_dump))
        # print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])

        for i in range(len(lines)):
            dump_outarray[i, :] = lines[i].split(" ")[0:cols_per_dump]
            # print(lines[i].split(" ")[0:cols_per_dump])

    return dump_outarray, box_vec_array


# def cfg2numpy_coords_plate(Path_2_dump,dump_realisation_name,
#                       number_of_particles_per_dump,cols_per_dump):


#         os.chdir(Path_2_dump) #+simulation_file+"/" +filename


#         with open(dump_realisation_name, 'r') as file:


#             lines = file.readlines()
#             box_vec_lines=lines[2:11]
#             #print(box_vec_lines[0])
#             box_vec_array=np.zeros((9))

#             for i in range(9):

#                  box_vec_array[i]=box_vec_lines[i].split(" ")[2]
#                  #print(coord)


#             box_vec_array=np.reshape(box_vec_array,(3,3))

#             lines=lines[15:] #remove intial file lines

#             for i in range(len(lines)-1):
#                 #print(lines)
#                 try:
#                    lines.remove("C \n")
#                    lines.remove("5.000000 \n")

#                 except:
#                     continue

#                 # print(lines)
#                 # print(i)


#             dump_outarray=np.zeros((number_of_particles_per_dump,cols_per_dump))
#             #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])

#             for i in range(len(lines)):
#                 dump_outarray[i,:]=lines[i].split(" ")[0:cols_per_dump]
#                 #print(lines[i].split(" ")[0:cols_per_dump])


#         return dump_outarray,box_vec_array


def cfg2numpy_coords(
    Path_2_dump, dump_realisation_name, number_of_particles_per_dump, cols_per_dump
):
    os.chdir(Path_2_dump)  # +simulation_file+"/" +filename

    with open(dump_realisation_name, "r") as file:
        lines = file.readlines()
        box_vec_lines = lines[2:11]
        # print(box_vec_lines[0])
        box_vec_array = np.zeros((9))

        for i in range(9):
            box_vec_array[i] = box_vec_lines[i].split(" ")[2]

        box_vec_array = np.reshape(box_vec_array, (3, 3))

        lines = lines[15:]  # remove intial file lines

        # print(len(lines))
        # print(lines)

        # list filtering
        lines = list(filter(("C \n").__ne__, lines))
        lines = list(filter(("5.000000 \n").__ne__, lines))
        lines = list(filter(("0.050000 \n").__ne__, lines))

        # for i in range(len(lines)-1):

        #     try:
        #        lines.remove("C \n")
        #        lines.remove("5.000000 \n")

        #        lines.remove('0.050000 \n')

        #     except:
        #         continue

        #     # print(lines)
        #     # print(i)

        # print(lines[596])

        dump_outarray = np.zeros((number_of_particles_per_dump, cols_per_dump))
        # print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
        # print(lines)
        for i in range(len(lines)):
            # print(i)
            # print(lines[i])
            dump_outarray[i, :] = lines[i].split(" ")[0:cols_per_dump]
            # print(lines[i].split(" ")[0:cols_per_dump])

    return dump_outarray, box_vec_array
