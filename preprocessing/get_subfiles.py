"""
Gets path to MRI and PET files after downloading from ADNI
"""

import os
import sys
import csv

DATASET_DIR = "/home/vikasu/Vikas/adni/mri-pet"

SUBCATEGORY_DIR = ["AD" , "CN" , "EMCI" , "LMCI" , "SMC"]

c = 0

for sub_dir in SUBCATEGORY_DIR:

    csv_dir = DATASET_DIR + "/" + sub_dir + "/" + sub_dir + "-MRI-PET_6_19_2020.csv"

    with open(csv_dir, 'rt') as csv_file:
        adni_samples = csv.reader(csv_file)

        for sample in adni_samples:
            print(sample)

            # Path to Dir
            # /home/vikasu/Vikas/adni/mri-pet/{subcategory}/ADNI/{Subject ID}/{Description}/{Acq Date}/{I for PET; S for MRI} + {Image Data ID}

            path_to_mri = DATASET_DIR + "/" + sub_dir + "/ADNI/" + sample[1] + "/MT1__N3m/" + reformat_date(sample[9]) + "/" + "S" + sample[0] + "/ADNI_" + sample[1] + "_MR_MT1__N3m_Br_"
            path_to_pet = DATASET_DIR + "/" + sub_dir + "/ADNI/" + sample[1] + "/Coreg,_Avg,_Standardized_Image_and_Voxel_Size/" + reformat_date(sample[9]) + "/" + "I" + sample[0]

            print(path_to_mri)
            print(path_to_pet)
        
    

        

