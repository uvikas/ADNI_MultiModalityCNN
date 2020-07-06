"""
Creates 27 3x3x3 patches given matrix files
"""

from nibabel.processing import conform
import nibabel as nib
import os
import subprocess
import time
import numpy as np


def get_img(path_to_dir):
    for f in os.listdir(path_to_dir):
        if "img" in f:
            return f



DATASET_DIR = "/home/vikasu/Vikas/adni/mri-pet"

#SUBCATEGORY_DIR = ["AD" , "CN" , "EMCI" , "LMCI" , "SMC"]
SUBCATEGORY_DIR = ["MRI"]

start_time = time.time()

for sub_dir in SUBCATEGORY_DIR:

    walk_dir = DATASET_DIR + "/training/" + sub_dir

    subjects = os.listdir(walk_dir)

    for subject in subjects:

        path_to_subject = os.path.join(walk_dir, subject)

        subj_img = np.load(path_to_subject)

        patch = make_patch(subj_img)

        c = 1
        for p in patch:
            np.save('Patch_' + str(c), p)
            c += 1

        

        














            