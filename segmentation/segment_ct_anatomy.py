# !pip install git+https://github.com/wasserth/TotalSegmentator.git

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm
import os
import pathlib as plb
import sys


# assumes the original petct dataset's nested directories
def find_studies(path_to_data):
    # find all studies
    root = plb.Path(path_to_data)
    patient_dirs = list(root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        study_dirs.extend(sub_dirs)

    return study_dirs


def segment_studies(study_dirs, nii_out_root):
    for study_dir in tqdm(study_dirs):
        
        input_path = str(plb.Path(study_dir/"CT.nii.gz"))
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)

        #nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        #nii_out_path = nii_out_path/study_dir.name
        #os.makedirs(nii_out_path, exist_ok=True)
        
        # write back to same study directory
        output_path = str(plb.Path(study_dir/"CTseg.nii.gz"))
        
        totalsegmentator(input_path, output_path)
  
    

if __name__ == "__main__":
    path_to_data = plb.Path(sys.argv[1])  # path to input nifti's: /media/storage/Joy/datasets/NIFTI/FDG-PET-CT-Lesions/
    
    study_dirs = find_studies(path_to_data)

    segment_studies(study_dirs, path_to_data)
    


# 1 study testing:    
#if __name__ == "__main__":
    
#    input_path = "/media/storage/Joy/datasets/NIFTI/FDG-PET-CT-Lesions/PETCT_0143bab87a/07-17-2005-NA-PET-CT Ganzkoerper  primaer mit KM-33529/CT.nii.gz"
#    output_path = "/media/storage/Joy/datasets/NIFTI/FDG-PET-CT-Lesions/PETCT_0143bab87a/07-17-2005-NA-PET-CT Ganzkoerper  primaer mit KM-33529/CTseg.nii.gz"
    
#    # provide input and output as file paths
#    totalsegmentator(input_path, output_path)
    

