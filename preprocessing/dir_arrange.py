
"""
Arranges all MRI and PET files from ADNI format to more convinient format. Creates /MRI and /PET in each subject ID directory. Removes rest of redundant folders
"""

import os
import shutil

DATASET_DIR = "/home/vikasu/Vikas/adni/mri-pet"

SUBCATEGORY_DIR = ["AD" , "CN" , "EMCI" , "LMCI" , "SMC"]

def get_subj_path(walk_dir, path):
    walk_dir_len = len(walk_dir)
    return path[:walk_dir_len+11]


file_list = []

for sub_dir in SUBCATEGORY_DIR:

    walk_dir = DATASET_DIR + "/" + sub_dir + "/ADNI"

    files = []

    for r, d, f in os.walk(walk_dir):
        for file_ in f:
            if '.img' in file_ or '.hdr' in file_:
                # move file to subj root folder
                path_to_file = os.path.join(r, file_)
                files.append([path_to_file, file_])

    c = 0
    for f in files:
        mri_path = os.path.join(get_subj_path(walk_dir, f[0]), "MRI")
        pet_path = os.path.join(get_subj_path(walk_dir, f[0]), "PET")

        if(not(os.path.exists(mri_path)) or not(os.path.exists(pet_path))):
            os.mkdir(mri_path)
            os.mkdir(pet_path)
        
        if("N3m" in f[0]):
            shutil.move(f[0], os.path.join(mri_path, f[1]))
        else:
            shutil.move(f[0], os.path.join(pet_path, f[1]))

        c += 1





            