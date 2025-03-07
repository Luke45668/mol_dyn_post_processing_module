"""
In this module we have functions that are used to manipulate files from simulations, such as loading in analysted tuples 

"""


import pickle as pck
import os as os


def batch_load_tuples(label, tuple_name):
    with open(label + tuple_name, "rb") as f:
        load_in = pck.load(f)

    return load_in


# this function will check if a folder exists in a certain location,
# if not it will create the folder and enter it 
def folder_check_or_create(filepath, folder):
    os.chdir(filepath)
    # combine file name with wd path
    check_path = filepath + "/" + folder
    print((check_path))
    if os.path.exists(check_path) == 1:
        print("file exists, proceed")
        os.chdir(check_path)
    else:
        print("file does not exist, making new directory")
        os.chdir(filepath)
        os.mkdir(folder)
        os.chdir(filepath + "/" + folder)

# this function will check if a folder exists in a certain location,
# if not it will create the folder without entering
def folder_check_or_create_no_enter(filepath, folder):
    os.chdir(filepath)
    # combine file name with wd path
    check_path = filepath + "/" + folder
    print((check_path))
    if os.path.exists(check_path) == 1:
        print("file exists, proceed")
    # os.chdir(check_path)
    else:
        print("file does not exist, making new directory")
        os.chdir(filepath)
        os.mkdir(folder)
    


# realisation organisation class, this gives the files 3 properties, name,
#  data set and realisation index
class realisation:
    def __init__(self, realisation_full_str, data_set, realisation_index_):
        self.realisation_full_str = realisation_full_str
        self.data_set = data_set
        self.realisation_index_ = realisation_index_

    def __repr__(self):
        return "({},{},{})".format(
            self.realisation_full_str, self.data_set, self.realisation_index_
        )


# this function uses the above class to sort realisations. 
def org_names(
     split_list_for_sorting, unsorted_list, first_sort_index, second_sort_index
     ):
     for i in unsorted_list:
          realisation_index_ = int(i.split("_")[first_sort_index])
          data_set = i.split("_")[second_sort_index]
          split_list_for_sorting.append(realisation(i, data_set, realisation_index_))

     realisation_name_sorted = sorted(
          split_list_for_sorting, key=lambda x: (x.data_set, x.realisation_index_)
     )
     realisation_name_sorted_final = []
     for i in realisation_name_sorted:
          realisation_name_sorted_final.append(i.realisation_full_str)

     return realisation_name_sorted_final
