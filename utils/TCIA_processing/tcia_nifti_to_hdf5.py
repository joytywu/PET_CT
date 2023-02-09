# data preparation (conversion of DICOM PET/CT studies to HDF5 format for running automated lesion segmentation)

# run script from command line as follows:
# python tcia_dicom_to_nifti.py /PATH/TO/NIFTI/FDG-PET-CT-Lesions/ /PATH/TO/HDF5/FDG-PET-CT-Lesions.hdf5

import h5py
from tqdm import tqdm
import pathlib as plb
import sys
import os
import nibabel as nib
import numpy as np

def find_studies(path_to_data):
    # find all studies
    dicom_root = plb.Path(path_to_data)
    patient_dirs = list(dicom_root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        #print(sub_dirs)
        study_dirs.extend(sub_dirs)
        
        #dicom_dirs = dicom_dirs.append(dir.glob('*'))
    return study_dirs


def nifti_to_hdf5(nii_file, path_to_h5_file):
    # conversion for a single file
    # creates an hdf5 file for one patient
    # nii_path:         path to a study directory containing all nifti files for a specific study of one patient
    # path_to_h5_file:  path to a single hdf5 file for one patient and study
    data = nib.load(nii_file)
    with h5py.File(path_to_h5_file, 'w') as h5_file:
        h5_file.create_dataset(data.get_fdata())


def nifti_to_hdf5_study(study_path, path_to_h5_file):
    # conversion for a single study
    # creates an hdf5 file for one patient
    # study_path:       path to a study directory containing all nifti files for a specific study of one patient
    # path_to_h5_file:  path to a single hdf5 file for one patient and study

    study_path = plb.Path(study_path)
    patient = study_path.parent.name
    study = study_path.name

    suv = nib.load(str(study_path / 'SUV.nii.gz'))
    ctres = nib.load(str(study_path / 'CTres.nii.gz'))
    ct = nib.load(str(study_path / 'CT.nii.gz'))
    pet = nib.load(str(study_path / 'PET.nii.gz'))
    seg = nib.load(str(study_path / 'SEG.nii.gz'))

    suv = suv.get_fdata()
    ctres = ctres.get_fdata()
    ct = ct.get_fdata()
    pet = pet.get_fdata()
    seg = seg.get_fdata()

    with h5py.File(path_to_h5_file, 'w') as h5_file:
        try:
            h5_file.create_group(patient + '/' + study)
            h5_file.create_dataset(patient + '/' + study + '/suv', data=suv, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/ctres', data=ctres, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/ct', data=ct, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/pet', data=pet, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/seg', data=seg, compression="gzip")
        except:
            h5_pat = h5_file.create_group(patient)
            h5_pat.create_group(study)
            h5_file.create_dataset(patient + '/' + study + '/suv', data=suv, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/ctres', data=ctres, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/ct', data=ct, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/pet', data=pet, compression="gzip")
            h5_file.create_dataset(patient + '/' + study + '/seg', data=seg, compression="gzip")


def convert_nifti_to_hdf5(study_dirs, path_to_h5_data):
    # batch conversion of all patients
    # creates a single hdf5 file for all patients
    # study_dirs:       NiFTI study directories for all patients
    # path_to_h5_data:  path to a single hdf5 file for all patients

    h5_file = h5py.File(path_to_h5_data, 'w')

    for pat_dir in tqdm(study_dirs):

        patient = pat_dir.parent.name
        study   = pat_dir.name

        suv    = nib.load(str(pat_dir/'SUV.nii.gz'))
        ctres  = nib.load(str(pat_dir/'CTres.nii.gz'))
        ct     = nib.load(str(pat_dir/'CT.nii.gz'))
        pet    = nib.load(str(pat_dir/'PET.nii.gz'))
        seg    = nib.load(str(pat_dir/'SEG.nii.gz'))
        
        suv   = suv.get_fdata()
        ctres = ctres.get_fdata()
        ct    = ct.get_fdata()
        pet   = pet.get_fdata()
        seg   = seg.get_fdata()

        try:
            h5_file.create_group(patient+'/'+study)
            h5_file.create_dataset(patient+'/'+study+'/suv', data=suv, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/ctres', data=ctres, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/ct', data=ct, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/pet', data=pet, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/seg', data=seg, compression="gzip")

        except:
            h5_pat = h5_file.create_group(patient)
            h5_pat.create_group(study)
            h5_file.create_dataset(patient+'/'+study+'/suv', data=suv, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/ctres', data=ctres, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/ct', data=ct, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/pet', data=pet, compression="gzip")
            h5_file.create_dataset(patient+'/'+study+'/seg', data=seg, compression="gzip")
        
    h5_file.close()


if __name__ == "__main__":
    path_to_data = sys.argv[1]     # path to converted NiFTI files (see tcia2nifti) from downloaded TCIA DICOM database e.g. '...tcia_nifti/FDG-PET-CT-Lesions/'
    path_to_h5_data = sys.argv[2]  # path to the to be saved HDF5 file, e.g. '...hdf5/FDG-PET-CT-Lesions.hdf5'
    study_dirs = find_studies(path_to_data)
    convert_nifti_to_hdf5(study_dirs, path_to_h5_data)


