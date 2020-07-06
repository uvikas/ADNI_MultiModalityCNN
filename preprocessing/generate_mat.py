"""
Generate .cpy matrix files for registered MRI and PET images
"""
from nibabel.processing import conform
import nibabel as nib
import os
import subprocess
import time
import numpy as np
import shutil


def get_img(path_to_dir):
    for f in os.listdir(path_to_dir):
        if "img" in f:
            return f



DATASET_DIR = "/home/vikasu/Vikas/adni/mri-pet"

#SUBCATEGORY_DIR = ["AD" , "CN" , "EMCI" , "LMCI" , "SMC"]
SUBCATEGORY_DIR = ["AD", "CN", "EMCI"]

start_time = time.time()

for sub_dir in SUBCATEGORY_DIR:

    data_dir = os.path.join(DATASET_DIR, sub_dir, 'data')
    mri_data_dir = os.path.join(data_dir, 'MRI')
    pet_data_dir = os.path.join(data_dir, 'PET')


    os.mkdir(data_dir)
    os.mkdir(mri_data_dir)
    os.mkdir(pet_data_dir)

    walk_dir = DATASET_DIR + "/" + sub_dir + "/ADNI"

    subjects = os.listdir(walk_dir)

    for subject in subjects:

        print("Saving matrix for", subject, "with", sub_dir)

        path_to_subject = os.path.join(walk_dir, subject)

        path_to_mri = os.path.join(path_to_subject, "MRI")
        path_to_pet = os.path.join(path_to_subject, "PET")
        
        path_to_mri_img = os.path.join(path_to_mri, 'mri_registered_brain.nii.gz')
        path_to_pet_img = os.path.join(path_to_pet, 'pet_registered_brain.nii.gz')

        mri_img = nib.load(path_to_mri_img)
        pet_img = nib.load(path_to_pet_img)

        mri_arr = np.array(mri_img.dataobj)
        pet_arr = np.array(pet_img.dataobj)

        with open(os.path.join(mri_data_dir, sub_dir + '_' + subject + '.npy'), 'wb') as f:
            np.save(f, mri_arr)

        with open(os.path.join(pet_data_dir, sub_dir + '_' + subject + '.npy'), 'wb') as f:
            np.save(f, pet_arr)
        






        














            