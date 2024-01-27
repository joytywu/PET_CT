import numpy as np
import nibabel as nib
import nilearn.image
import nilearn
from tqdm import tqdm
import os
import pathlib as plb
import sys

            

# upsamples PET to CT resolution
def resample_pet(study_dir, nii_out_path, interpolation='linear', save=True):
    # resample CT to PET and mask resolution -- this gives float64 memory error for some studies
    ct   = nib.load(study_dir/plb.Path('CT.nii.gz'))
    pet  = nib.load(study_dir/plb.Path('SUV.nii.gz'))
    seg  = nib.load(study_dir/plb.Path('SEG.nii.gz'))
    
#     # resampling pet -- interpolation improves image quality but blows up storage size 
#     try:
#         resPT = nilearn.image.resample_to_img(pet, ct, interpolation = 'nearest')
#         nib.save(resPT, nii_out_path/'SUVres.nii.gz')
#     except Exception as e:
#         print(e)
    # Try lowering memory constraints by changing to a different data type and rounding to 3 decimal places
    try:
        new_dtype = np.float32
        resPT = compress_data(nilearn.image.resample_to_img(pet, ct, interpolation = interpolation), new_dtype)
        if save:
            nib.save(resPT, nii_out_path/'SUVres.nii.gz')
    except Exception as e: 
        print(e) 
    
    # resampling segmentation
    try:
        resSeg = nilearn.image.resample_to_img(seg, ct, interpolation = 'nearest')
        if save:
            nib.save(resSeg, nii_out_path/'SEGres.nii.gz')
    except Exception as e:
        print(e)     
        # Try lowering memory constraints by changing to a different data type
        try:
            new_dtype = np.utf8 #SEG are binary masks
            resSeg = compress_data(nilearn.image.resample_to_img(seg, ct, interpolation = 'nearest'), new_dtype)
            if save:
                nib.save(resSeg, nii_out_path/'SEGres.nii.gz')
        except Exception as e: 
            print(e) 
    
    return resPT, resSeg

def compress_data(niimg_like_obj, new_dtype):
    # update data type:
    hd = niimg_like_obj.header
    new_data = np.round(niimg_like_obj.get_fdata(), 3).astype(new_dtype)
    niimg_like_obj.set_data_dtype(new_dtype)
    
    # if nifty1
    if hd['sizeof_hdr'] == 348:
        new_dtype_niimg = nib.Nifti1Image(new_data, niimg_like_obj.affine, header=hd)
    # if nifty2
    elif hd['sizeof_hdr'] == 540:
        new_dtype_niimg = nib.Nifti2Image(new_data, niimg_like_obj.affine, header=hd)
    else:
        raise IOError('Input image header problem')
    
    return new_dtype_niimg


def find_studies(path_to_data):
    # find all studies
    root = plb.Path(path_to_data)
    patient_dirs = list(root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        study_dirs.extend(sub_dirs)

    return study_dirs


def resampling_studies(study_dirs, nii_out_root):
    # batch conversion of all patients
    for study_dir in tqdm(study_dirs):
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)

        nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        nii_out_path = nii_out_path/study_dir.name
        os.makedirs(nii_out_path, exist_ok=True)
        
        resample_pet(study_dir, nii_out_path)
            

if __name__ == "__main__":
    path_to_data = plb.Path(sys.argv[1])  # path to input nifti
    nii_out_root = plb.Path(sys.argv[2])  # path to where resampled nifti will be saved
    
    for keyword in ['train','val','test']:
        study_dirs = find_studies(path_to_data/plb.Path(keyword))

        resampling_studies(study_dirs, nii_out_root/plb.Path(keyword))
    
#Run from command line example:    
#python resampling_pet.py /Users/joywu/Research/gaze_datasets/pilot/ /Users/joywu/Research/gaze_datasets/pilot/
    
