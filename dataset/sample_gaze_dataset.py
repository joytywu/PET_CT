import os
import pathlib as plb
import pandas as pd
import sys
from tqdm import tqdm


# file types to be sampled
global seg_fname
global pet_fname
seg_fname = 'SEGres.nii.gz'
pet_fname = 'SUVres.nii.gz'


# Create reproducible train test split from .csv from the Tubingen dataset
def train_val_test_split(csvpath):
    tab = pd.read_csv(csvpath)
    tab['relative_path'] = [get_relative_studypath(fullpath) for fullpath in tab['File Location']]
    # sort alphabetically by subject ID to obtain same split
    all_subjects = sorted(list(set(tab['Subject ID'])))
    train_ids = all_subjects[:720]
    val_ids = all_subjects[720:800]
    test_ids = all_subjects[800:]
    keep = ['Study UID', 'relative_path','diagnosis', 'age', 'sex']
    train_tab = tab[tab['Subject ID'].isin(train_ids)][keep].drop_duplicates().copy()
    val_tab = tab[tab['Subject ID'].isin(val_ids)][keep].drop_duplicates().copy()
    test_tab = tab[tab['Subject ID'].isin(test_ids)][keep].drop_duplicates().copy()
    
    return train_tab, val_tab, test_tab


def keep_positive_cases(dftab, num_studies):
    pos_tab = dftab[dftab['diagnosis'] != 'NEGATIVE'].copy()
    print(pos_tab.shape)
    pos_tab = pos_tab.sample(n=num_studies, random_state=42, ignore_index=True)
    print(pos_tab.shape)
    return pos_tab


def copy_to_new_dir(file_path, out_dir):
    !cp $file_path $out_dir


def sample_studies(data_in_root, table, num_studies, image_out_root):
    
    pos_tab = keep_positive_cases(dftab, num_studies)
    pos_tab.to_csv(os.path.join(image_out_root,'metadata_{}.csv'.format(num_studies)), index=False)
    
    for i, row in tqdm(pos_tab.iterrows(), total=pos_tab.shape[0]):
        
        # nitfi paths
        suv_path = os.path.join(data_in_root, row['relative_path'], pet_fname)
        seg_path = os.path.join(data_in_root, row['relative_path'], seg_fname)
        print('Processing study:', row['relative_path'])
        
        out_dir = os.path.join(image_out_root,row['relative_path'])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        copy_to_new_dir(suv_path, out_dir)
        copy_to_new_dir(seg_path, out_dir)
        print('Transferred study:', row['relative_path'])    
    

if __name__ == "__main__":
    # set paths
    curr = os.getcwd()
    # path to downloaded csv (from: https://wiki.cancerimagingarchive.net/download/attachments/93258287/Clinical%20Metadata%20FDG%20PET_CT%20Lesions.csv?api=v2)
    csvpath = os.path.join(curr,"Clinical_Metadata_FDG_PET_CT_Lesions.csv") 
    print(csvpath)
    
    data_in_root = plb.Path(sys.argv[1])  # path to parent directory for all studies, e.g. '...datasets/NIFTI_MIP/FDG-PET-CT-Lesions/'
    data_out_root = plb.Path(sys.argv[2])  # path to where we want to save the DETR dataset, e.g. '...datasets/DETR/FDG-PET-CT-Lesions/')
    if not os.path.isdir(data_out_root):
        os.makedirs(data_out_root)
    num_studies = int(plb.Path(sys.argv[3]))
    
    # Get train test split for MIP PETCT coco dataset
    train_tab, val_tab, test_tab = train_val_test_split(csvpath)
    
    # This id will be automatically increased as we go
    for keyword, table in zip(["train", "val", "test"],[train_tab,val_tab,test_tab]):
        # write to separate folders for train test val
        image_out_root = os.path.join(data_out_root, keyword)
        if not os.path.isdir(image_out_root):
            os.makedirs(image_out_root)
        
        # Sample studies from each split bucket
        sample_studies(data_in_root, table, num_studies, image_out_root)
        
# python sample_gaze_dataset.py /media/storage/Joy/datasets/NIFTI/FDG-PET-CT-Lesions/ /home/joytywu/Documents/gaze_datasets/ 10
