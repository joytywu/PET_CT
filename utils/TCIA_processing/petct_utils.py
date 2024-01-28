import os
import sys
import glob
import pathlib as plb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import math
import cv2
from scipy import ndimage


# Windows a CT volume or slice
def window_ct_image(image, window_center, window_width):

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    window_image = image.copy()

    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image, img_min, img_max


# Returns L and Window for set key
# These are the windows and level numbers clinically accepted for CT viewing
def select_ct_window(key=None):
    # soft tissue window (W:350–400 L:20–60)
    if key == ord('1'):
        L = 50
        W = 400
    # lung window
    elif key == ord('2'):
        L = -600
        W = 1500 
    # liver window
    elif key == ord('3'):
        L = 30
        W = 150
    # bone window
    elif key == ord('4'):
        L = 400
        W = 1800
    # mediastinum window
    elif key == ord('5'):
        L = 50
        W = 350
    # brain window
    elif key == ord('6'):
        L = 40
        W = 80
    # temporal bone window (W:2800 L:600 or W:4000 L:700)
    elif key == ord('7'):
        L = 600
        W = 2800
    # subdural W:130-300 L:50-100
    elif key == ord('8'):
        L = 200
        W = 80
    # stroke W:8 L:32 or W:40 L:40 
    elif key == ord('9'):
        L = 40
        W = 40
    else:
        L = 50
        W = 400
    return L, W


# basically switch channel 1 and 3 and allows adjusting alpha for transparency
def convert_bgra2rgba(img_bgra,alpha=1):
    b = img_bgra[:,:,0].copy()
    g = img_bgra[:,:,1].copy()
    r = img_bgra[:,:,2].copy()
    a = img_bgra[:,:,3].copy()*alpha # alpha is the transparency factor
    img_rgba = cv2.merge((r,g,b,a))
    return img_rgba


# Load an axial image slice from a nifti 3D volume given slice number
def load_axial_image_slice(image_path, idx):
    # read pre-saved nifti images
    print('Loading images')
    images = nib.load(image_path).get_fdata()
    img = np.rot90(images[:,:,idx]).squeeze().copy()
    return img


# Add segmentation ground truth to image slice
def add_seg_to_image(img, seg):
    img[seg>0]=(0,255,0,1) # last channel is alpha so doesnt really matter
    return img

    
# prep 1 axial PET image slice
def axial_PET_slice(img, suv_max, suv_min=0):
    norm = plt.Normalize(vmin=suv_min, vmax=suv_max)
    cmap = plt.cm.gist_yarg
    img = cmap(norm(img))
    return img
    

# prep 1 axial CT image slice
# ct level and window can be read from the gaze jsons for each data point --?? need to check if saved properly
# default shows soft tissue window
def axial_CT_slice(img, ct_level=50, ct_window=400):
    img, img_min, img_max = window_ct_image(image, window_center, window_width)
    norm_ct = plt.Normalize(vmin=img_min, vmax=img_max)
    img = cmap(norm_ct(img))
    return img
        
        
# prep 1 axial PET/CT fused image slice
def axial_PETCT_slice(pt, ct, alpha = 0.3):
    # PET image prep
    # closest heatmap to "hot iron" colormap that gets used in PACS for PET
    cmap_pt = plt.cm.gist_heat 
    pt = cmap_pt(norm(pt))
    pt = convert_bgra2rgba(pt) 

    # CT image prep
    cmap_ct = plt.cm.gist_gray
    img, img_min, img_max = window_ct_image(image, window_center, window_width)
    norm_ct = plt.Normalize(vmin=img_min, vmax=img_max)
    ct = cmap_ct(norm_ct(ct)) 

    # fused, somehow minus instead of + worked better
    img = cv2.addWeighted(ct, 1-alpha, pt, alpha, 0)
    return img
    

# Show an axial slice image:
def show_axial_image_slice(img, dim=(512,512), display=True, show_gaze=False):
#     if show_gaze and not paused:
#             img = add_gaze(img)  
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.namedWindow("slice", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("slice", dim[0],dim[1])
    while display:
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow('slice', img)
        if key == ord('q'):
            display = False
    cv2.destroyAllWindows()


# From reading a study to showing an axial image
# study_path assumes a nested directory from Tubingen dataset with processed nifties as named below
def read_and_show_study(study_path, idx, modality, dim=(512,512), out_path=None):
    # .nii.gz files expected in the study_path
    seg_path = os.path.join(study_path,'SEGres.nii.gz')
    pt_path = os.path.join(study_path,'SUVres.nii.gz')
    ct_path = os.path.join(study_path,'CT.nii.gz')
    
    # If presenting a PET image with or without seg
    if modality in ['pet','pet_seg']:
        print('loading image')
        img = load_axial_image_slice(pt_path, idx)
        print('loaded image')
        suv_max = 6
        img = axial_PET_slice(img, suv_max)
    # If presenting a CT image with or without seg
    elif modality in ['ct','ct_seg']:
        print('loading image')
        img = load_axial_image_slice(ct_path, idx)
        print('loaded image')
        # see more L and W options with above select_ct_window, but will need to change manually here with current code
        L = 50
        W = 400
        img = axial_CT_slice(img, ct_level=L, ct_window=W) 
    # If presenting a fused PETCT image with or without seg
    elif modality in ['petct', 'petct_seg']:
        print('loading image')
        pt = load_axial_image_slice(pt_path, idx)
        ct = load_axial_image_slice(ct_path, idx)
        print('loaded image')
        img = axial_PETCT_slice(pt, ct, alpha = 0.3) # can change alpha to adjust how 'solid' looking the PET overlay is.
    else:
        return print('Not a valid modality')

    # read segmentation ground truth mask (green) if wants to show segmentation overlaying the image
    if 'seg' in modality:
        print('loading ground truth')
        seg = load_axial_image_slice(seg_path, idx)
        img = add_seg_to_image(img, seg)
        print('loaded ground truth')
    
    # Show final processed image axial slice
    print('Showing an axial slice of modality: ', modality, ' and slice index: ', idx)
    show_axial_image_slice(img, dim=(512,512))
    
    # Save image. Out_path should have a .png or .jpg extension
    if out_path!=None:
        # Need to do it this way for image to look right when saved
        gry = (img<1.1)
        img[gry]*=255
        cv2.imwrite(out_path, img)    
        
    
# Separates out individual tumors
def get_connected_components_3D(seg_data): 
    # input seg_data is the numpy after reading nifti and get_fdata()
    
    #value of 1 means lesion is present
    value = 1
    binary_mask = seg_data == value

    #label and seperate each component off that has the value of 1
    labled_mask, num_features = ndimage.label(binary_mask)

    #assign a unique id to each component
    unique_ids = np.unique(labled_mask)

    #num of components
    print(num_features)

    #seperate out the masks
    separate_seg_masks = []
    for component_id in unique_ids:
        component_mask = labled_mask == component_id
        #print each id of each component
        print(f"Connected Component {component_id}:")
        separate_seg_masks.append(component_mask)
    
    return separate_seg_masks
                           