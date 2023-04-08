# data preparation (conversion of DICOM PET/CT studies to nifti format for running automated lesion segmentation)

# run script from command line as follows:
# python tcia_dicom_to_nifti.py /PATH/TO/DICOM/FDG-PET-CT-Lesions/ /PATH/TO/NIFTI/FDG-PET-CT-Lesions/

# you can ignore the nilearn warning:
# .../nilearn/image/resampling.py:527: UserWarning: Casting data from int16 to float32 warnings.warn("Casting data from %s to %s" % (data.dtype.name, aux)) 
# or run as python -W ignore tcia_dicom_to_nifti.py /PATH/TO/DICOM/FDG-PET-CT-Lesions/ /PATH/TO/NIFTI/FDG-PET-CT-Lesions/

import pathlib as plb
import tempfile
import os
import dicom2nifti
import nibabel as nib
import numpy as np
import pydicom
import sys
import shutil
import nilearn.image
from tqdm import tqdm


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


def find_unprocessed_studies(path_to_data, nii_out_root):
    study_dirs = find_studies(path_to_data)
    #print(len(study_dirs))
    study_out_dirs = find_studies(nii_out_root)
    #print(len(study_out_dirs))

    processed_pts = []
    for study_out_dir in study_out_dirs:
        # study_out_dir.parent.name not unique enough
        patient = study_out_dir.name
        processed_pts.append(patient)
    #print(len(set(processed_pts)))

    unprocessed_study_dirs = []
    for study_dir in study_dirs:
        patient = study_dir.name
        if patient in processed_pts:
            continue
    #         print("The following patient directory has been processed: ", patient)
        else: 
            unprocessed_study_dirs.append(study_dir)
    #         print("The following patient directory is being processed: ", patient)
    # print(len(unprocessed_study_dirs))
    
    return unprocessed_study_dirs


def identify_modalities(study_dir):
    # identify CT, PET and mask subfolders and return dicitionary of modalities and corresponding paths, also return series ID, output is a dictionary
    study_dir = plb.Path(study_dir)
    sub_dirs = list(study_dir.glob('*'))

    modalities = {}

    for dir in sub_dirs:
        first_file = next(dir.glob('*.dcm'))
        ds = pydicom.dcmread(str(first_file))
        #print(ds)
        modality = ds.Modality
        modalities[modality] = dir
    
    modalities["ID"] = ds.StudyInstanceUID
    return modalities


def dcm2nii_CT(CT_dcm_path, nii_out_path):
    # conversion of CT DICOM (in the CT_dcm_path) to nifti and save in nii_out_path
    with tempfile.TemporaryDirectory() as tmp: #convert CT
        tmp = plb.Path(str(tmp))
        # convert dicom directory to nifti
        # (store results in temp directory)
        dicom2nifti.convert_directory(CT_dcm_path, str(tmp), 
                                      compression=True, reorient=True)
        nii = next(tmp.glob('*nii.gz'))
        # copy niftis to output folder with consistent naming
        shutil.copy(nii, nii_out_path/'CT.nii.gz')


def dcm2nii_PET(PET_dcm_path, nii_out_path):
    # conversion of PET DICOM (in the PET_dcm_path) to nifti (and SUV nifti) and save in nii_out_path
    first_pt_dcm = next(PET_dcm_path.glob('*.dcm'))
    suv_corr_factor = calculate_suv_factor(first_pt_dcm)

    with tempfile.TemporaryDirectory() as tmp: #convert PET
        tmp = plb.Path(str(tmp))
        # convert dicom directory to nifti
        # (store results in temp directory)
        dicom2nifti.convert_directory(PET_dcm_path, str(tmp), 
                                    compression=True, reorient=True)
        nii = next(tmp.glob('*nii.gz'))
        # copy nifti to output folder with consistent naming
        shutil.copy(nii, nii_out_path/'PET.nii.gz')

        # convert pet images to quantitative suv images and save nifti file
        suv_pet_nii = convert_pet(nib.load(nii_out_path/'PET.nii.gz'), suv_factor=suv_corr_factor)
        nib.save(suv_pet_nii, nii_out_path/'SUV.nii.gz')


def conv_time(time_str):
    # function for time conversion in DICOM tag
    return (float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13]))


def calculate_suv_factor(dcm_path):
    # reads a PET dicom file and calculates the SUV conversion factor
    ds = pydicom.dcmread(str(dcm_path))
    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight / act_dose
    return suv_factor


def convert_pet(pet, suv_factor):
    # function for conversion of PET values to SUV (should work on Siemens PET/CT)
    affine = pet.affine
    pet_data = pet.get_fdata()
    pet_suv_data = (pet_data*suv_factor).astype(np.float32)
    pet_suv = nib.Nifti1Image(pet_suv_data, affine)
    return pet_suv


def dcm2nii_mask(mask_dcm_path, nii_out_path):
    # conversion of the mask dicom file to nifti (not directly possible with dicom2nifti)
    mask_dcm = list(mask_dcm_path.glob('*.dcm'))[0]
    mask = pydicom.read_file(str(mask_dcm))
    mask_array = mask.pixel_array
    
    # get mask array to correct orientation (this procedure is dataset specific)
    mask_array = np.transpose(mask_array,(2,1,0) )  
    mask_orientation = mask[0x5200, 0x9229][0].PlaneOrientationSequence[0].ImageOrientationPatient
    if mask_orientation[4] == 1:
        mask_array = np.flip(mask_array, 1 )
    
    # get affine matrix from the corresponding pet             
    pet = nib.load(str(nii_out_path/'PET.nii.gz'))
    pet_affine = pet.affine
    
    # return mask as nifti object
    mask_out = nib.Nifti1Image(mask_array, pet_affine)
    nib.save(mask_out, nii_out_path/'SEG.nii.gz')   
    

def resample_ct(nii_out_path):
    # resample CT to PET and mask resolution -- this gives float64 memory error for some studies
    ct   = nib.load(nii_out_path/'CT.nii.gz')
    pet  = nib.load(nii_out_path/'PET.nii.gz')
    
    try:
        CTres = nilearn.image.resample_to_img(ct, pet, fill_value=-1024)
        nib.save(CTres, nii_out_path/'CTres.nii.gz')
    
    except Exception as e:
        print(e)
        
        # Try lowering storage by changing to a different data type and rounding to 3 decimal places
        try:
            new_dtype = np.float32
            CTres = compress_data(nilearn.image.resample_to_img(ct, pet, fill_value=-1024), new_dtype)
            nib.save(CTres, nii_out_path/'CTres.nii.gz')
        except Exception as e: 
            print(e)
            
            
def resample_pet(nii_out_path):
    # resample CT to PET and mask resolution -- this gives float64 memory error for some studies
    ct   = nib.load(nii_out_path/'CT.nii.gz')
    pet  = nib.load(nii_out_path/'SUV.nii.gz')
    seg  = nib.load(nii_out_path/'SEG.nii.gz')
    
    # resampling pet
    # Try lowering storage by changing to a different data type and rounding to 3 decimal places
    try:
        new_dtype = np.float32
        res = compress_data(nilearn.image.resample_to_img(pet, ct, interpolation = 'linear'), new_dtype)
        nib.save(res, nii_out_path/'SUVres.nii.gz')
        print('Saved SUVres.nii.gz at', str(nii_out_path))
    except Exception as e: 
        print(e) 
    
    # resampling segmentation
    try:
        res = nilearn.image.resample_to_img(seg, ct, interpolation = 'nearest')
        nib.save(res, nii_out_path/'SEGres.nii.gz')
    except Exception as e:
        print(e)
        
        # Try lowering storage by changing to a different data type and rounding to 3 decimal places
        try:
            new_dtype = np.uint8 #SEG are binary masks
            res = compress_data(nilearn.image.resample_to_img(seg, ct, interpolation = 'nearest'), new_dtype)
            nib.save(res, nii_out_path/'SEGres.nii.gz')
        except Exception as e: 
            print(e) 


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
            

def tcia_to_nifti(tcia_path, nii_out_path, modality='CT'):
    # conversion for a single file
    # creates a nifti file for one patient/study
    # tcia_path:        path to a DICOM directory for a specific study of one patient
    # nii_out_path:     path to a directory where nifti file for one patient, study and modality will be stored
    # modality:         modality to be converted CT, PET or mask ('CT', 'PT', 'SEG')
    os.makedirs(nii_out_path, exist_ok=True)
    if modality == 'CT':
        dcm2nii_CT(tcia_path, nii_out_path)
        resample_ct(nii_out_path)
    elif modality == 'PET':
        dcm2nii_PET(tcia_path, nii_out_path)
    elif modality == 'SEG':
        dcm2nii_mask(tcia_path, nii_out_path)


def tcia_to_nifti_study(study_path, nii_out_path):
    # conversion for a single study
    # creates NIfTI files for one patient
    # study_path:       path to a study directory containing all DICOM files for a specific study of one patient
    # nii_out_path:     path to a directory where all nifti files for one patient and study will be stored
    study_path = plb.Path(study_path)
    modalities = identify_modalities(study_path)
    nii_out_path = plb.Path(nii_out_root / study_path.parent.name)
    nii_out_path = nii_out_path/study_path.name
    os.makedirs(nii_out_path, exist_ok=True)

    ct_dir = modalities["CT"]
    dcm2nii_CT(ct_dir, nii_out_path)

    pet_dir = modalities["PT"]
    dcm2nii_PET(pet_dir, nii_out_path)

    seg_dir = modalities["SEG"]
    dcm2nii_mask(seg_dir, nii_out_path)
    
    # down samples ct resolutation to pet
    resample_ct(nii_out_path)
    
    # up samples pet (standardized to suv) resolutation to ct
    resample_pet(nii_out_path)


def convert_tcia_to_nifti(study_dirs,nii_out_root, processes = ['CT', 'PT', 'SEG','resCT','resPT']):
    # batch conversion of all patients
    for study_dir in tqdm(study_dirs):
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)

        modalities = identify_modalities(study_dir)
        nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        nii_out_path = nii_out_path/study_dir.name
        os.makedirs(nii_out_path, exist_ok=True)
        
        if 'CT' in processes:
            ct_dir = modalities["CT"]
            dcm2nii_CT(ct_dir, nii_out_path)
        
        if 'PT' in processes:
            pet_dir = modalities["PT"]
            dcm2nii_PET(pet_dir, nii_out_path)
        
        if 'SEG' in processes:
            seg_dir = modalities["SEG"]
            dcm2nii_mask(seg_dir, nii_out_path)

        # down samples ct resolutation to pet
        if 'resCT' in processes:
            resample_ct(nii_out_path)

        # up samples pet and the associated segmentation (standardized to suv) resolutation to ct
        # will round voxels to 3 decimal places otherwise storage size blows up 3 times.
        if 'resPT' in processes:
            resample_pet(nii_out_path)

        

if __name__ == "__main__":
    path_to_data = plb.Path(sys.argv[1])  # path to downloaded TCIA DICOM database, e.g. '.../FDG-PET-CT-Lesions/'
    nii_out_root = plb.Path(sys.argv[2])  # path to the to be created NiFTI files, e.g. '...tcia_nifti/FDG-PET-CT-Lesions/')
    
    unprocessed_only = False # default should be True
    processes = ['resPT'] # have processed the rest previously 
    
    if unprocessed_only:
        study_dirs = find_unprocessed_studies(path_to_data, nii_out_root)
    else:
        study_dirs = find_studies(path_to_data)
    
    convert_tcia_to_nifti(study_dirs, nii_out_root, processes)
    
#python tcia_dicom_to_nifti_selective.py /media/storage/Joy/datasets/NIFTI/FDG-PET-CT-Lesions/ /media/storage/Joy/datasets/NIFTI/FDG-PET-CT-Lesions/

