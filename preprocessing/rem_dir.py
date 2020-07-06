"""
Remove redundant directories after running dir_arrange.py
"""

import os
import shutil

DATASET_DIR = "/home/vikasu/Vikas/adni/mri-pet"

#SUBCATEGORY_DIR = ["AD" , "CN" , "EMCI" , "LMCI" , "SMC"]
SUBCATEGORY_DIR = [ "CN" , "EMCI" , "LMCI" , "SMC"]


file_list = []

pet = 0
mri = 0

for sub_dir in SUBCATEGORY_DIR:

    walk_dir = DATASET_DIR + "/" + sub_dir + "/ADNI"


    for r, d, f in os.walk(walk_dir):
        for name in d:
            if "Coreg,_Avg,_Standardized_Image_and_Voxel_Size" in name or "N3m" in name:
                shutil.rmtree(os.path.join(r, name))
                print("Removing path", os.path.join(r, name))

                if "Coreg" in name:
                    pet += 1
                else:
                    mri += 1


print("PET:", pet)
print("MRI:", mri)

    





            