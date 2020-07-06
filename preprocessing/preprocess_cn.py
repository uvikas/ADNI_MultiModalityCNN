"""
Skull strip and affine registration script for MRI and PET for AD
"""

from nibabel.processing import conform
import numpy as np
import nilearn
import nibabel as nib
import os
import subprocess
import time

def get_img(path_to_dir):
    for f in os.listdir(path_to_dir):
        if "img" in f:
            return f

def exec_bet(input_path, output_path, frac="0.5"):
    command = ["bet", input_path, output_path, "-R", "-B"]
    subprocess.call(command)

def strip_skull(input_path, output_path):
    print("BET Brain Extraction: Extracting", input_path)
    try:
        exec_bet(input_path, output_path)
    except RuntimeError:
        print("\tFailed on:", input_path)
    
def exec_flirt(input_path, ref_path, output_path, mat_path):
    command = ["flirt", "-in", input_path, "-ref", ref_path, "-out", output_path, "-omat", mat_path]
    subprocess.call(command)

def affine_register(input_path, ref_path, output_path, mat_path):
    print("Registering image", input_path, "to reference", ref_path)
    try:
        exec_flirt(input_path, ref_path, output_path, mat_path)
    except RuntimeError:
        print("\tFailed on:", input_path)



DATASET_DIR = "/home/vikasu/Vikas/adni/mri-pet"

#SUBCATEGORY_DIR = ["AD" , "CN" , "EMCI" , "LMCI" , "SMC"]
SUBCATEGORY_DIR = ["CN"]

start_time = time.time()

for sub_dir in SUBCATEGORY_DIR:

    walk_dir = DATASET_DIR + "/" + sub_dir + "/ADNI"

    subjects = os.listdir(walk_dir)

    for subject in subjects:

        path_to_subject = os.path.join(walk_dir, subject)

        path_to_mri = os.path.join(path_to_subject, "MRI")
        path_to_pet = os.path.join(path_to_subject, "PET")
        
        path_to_mri_img = os.path.join(path_to_mri, get_img(path_to_mri))
        path_to_pet_img = os.path.join(path_to_pet, get_img(path_to_pet))


        # Resample to 256 x 256 x 256 and 1x1x1mm voxel resolution

        mri_img = nib.load(path_to_mri_img)
        if(len(mri_img.header.get_data_shape()) == 4):
            mri_img = nib.funcs.four_to_three(mri_img)[0]
        
        print("Resampling MRI to 256x256x256 at", os.path.join(path_to_mri, 'conformed_mri.nii.gz'))

        conformed_mri = conform(mri_img)

        nib.save(conformed_mri, os.path.join(path_to_mri, 'conformed_mri.nii.gz'))

        # Brain Extraction

        strip_skull(os.path.join(path_to_mri, 'conformed_mri.nii.gz'), os.path.join(path_to_mri, 'mri_brain.nii.gz'))

        # Affine Registration

        affine_register(os.path.join(path_to_mri, 'mri_brain.nii.gz'), '/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz' , os.path.join(path_to_mri, 'aff_mri_registered_brain.nii.gz'), os.path.join(path_to_mri, 'registered_mri.mat'))

        affine_reg_nib = nib.load(os.path.join(path_to_mri, 'aff_mri_registered_brain.nii.gz'))

        affine_reg_nib_conf = nilearn.image.resample_img(affine_reg_nib, target_affine=np.eye(3)*2.)

        nib.save(affine_reg_nib_conf, os.path.join(path_to_mri, 'mri_registered_brain.nii.gz'))

        pet_img = nib.load(path_to_pet_img)
        if(len(pet_img.header.get_data_shape()) == 4):
            pet_img = nib.funcs.four_to_three(pet_img)[0]

        print("Resampling PET to 256x256x256 at", os.path.join(path_to_pet, 'conformed_pet.nii.gz'))

        conformed_pet = conform(pet_img)

        nib.save(conformed_pet, os.path.join(path_to_pet, 'conformed_pet.nii.gz'))
        
        strip_skull(os.path.join(path_to_pet, 'conformed_pet.nii.gz'), os.path.join(path_to_pet, 'pet_brain.nii.gz'))

        affine_register(os.path.join(path_to_pet, 'pet_brain.nii.gz'), os.path.join(path_to_mri, 'aff_mri_registered_brain.nii.gz') , os.path.join(path_to_pet, 'aff_pet_registered_brain.nii.gz'), os.path.join(path_to_pet, 'registered_pet.mat'))

        paffine_reg_nib = nib.load(os.path.join(path_to_pet, 'aff_pet_registered_brain.nii.gz'))

        paffine_reg_nib_conf = nilearn.image.resample_img(paffine_reg_nib, target_affine=np.eye(3)*2.)

        nib.save(paffine_reg_nib_conf, os.path.join(path_to_pet, 'pet_registered_brain.nii.gz'))

        print("--- Running for %s seconds ---" % (time.time() - start_time))









            