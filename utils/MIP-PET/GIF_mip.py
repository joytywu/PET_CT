### A COPIER POUR IMPORTER LES FONCTIONS
#import sys
#sys.path
#sys.path.append('/Users/Paul/Documents/soft/Python/my_func/')
#import my_functions_python
#
#
#

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import nilearn
import shutil
import glob
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import imageio
import datetime
import numpy as np
import scipy
from scipy import ndimage
from tqdm import tqdm
import pathlib as plb
import sys
import cv2

#### Create MIP GIF
def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)

    
## Interpolation to account for difference between pixel spacing and slice thickness 
# interpolation options: 'antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'
# 8mm/2.03642 spacing = 3.92846269434 pixels/mm =~ 100dpi
# colormap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def interpolate_show_MIP(i, nda, suv_max, spacing=(1,1), title=None, margin=0, dpi=100, colormap='Greys', OUTPATH=None,show=False):
    ysize = nda.shape[0]
    xsize = nda.shape[1]

    figsize = (1 + margin) * xsize * spacing[0] / dpi, (1 + margin) * ysize * spacing[1] / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    #hide axis
    ax.axis('off')
    
    extent = (0, xsize * spacing[0], 0, ysize * spacing[1])

    #various papers mentions bicubic interpolation...
    t = ax.imshow(
#         nda, extent=extent, interpolation="hamming", cmap="Greys", origin="upper", vmax=suv_max
#         nda, extent=extent, interpolation="bilinear", cmap="Greys", origin="upper", vmax=suv_max
        nda, extent=extent, interpolation="bicubic", cmap=colormap, origin="upper", vmax=suv_max 
    )

    if title:
        plt.title(title)
    if OUTPATH != None:
        fig.savefig(os.path.join(OUTPATH,'MIP'+'%04d' % (i)+'.png'), dpi = dpi)
    if not show:
        plt.close(fig)

    
def create_mipGIF_from_3D(img,nb_image=48,duration=0.1,is_mask=False,borne_max=None):
    ls_mip=[]

    img_data=img.get_fdata()
    
    w = img.header['pixdim'][1] 
    y = img.header['pixdim'][3] 
    spacing = (1, y/w)
    print('Pixel spacing ratio:', spacing)
    
    liver_idx = img_data.shape[-1]//2
    suv_liver = img_data[:,:,liver_idx].squeeze().max()
    print('Liver SUV max', suv_liver)
    
    print('Interpolating')
    img_data+=1e-5
    for angle in tqdm(np.linspace(0,360,nb_image)):
        #ls_slice=[]
        # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
        vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        
        MIP=np.amax(vol_angle,axis=1)
        MIP-=1e-5
        MIP[MIP<1e-5]=0
        MIP=np.flipud(MIP.T)
        ls_mip.append(MIP)
        print('angle:', angle)
    
    try:
        shutil.rmtree('MIP/')
    except:
        pass
    os.mkdir('MIP/')
    
    print('Creating MIP')
    ls_image=[]
    for mip,i in zip(ls_mip,range(len(ls_mip))):
#         fig,ax=plt.subplots()
#         ax.set_axis_off()
        if borne_max is None:
            if is_mask==True:
                borne_max=1
            else:
                borne_max=suv_liver
#         plt.imshow(mip,cmap='Greys',vmax=borne_max)
#         fig.savefig('MIP/MIP'+'%04d' % (i)+'.png')
#         plt.close(fig)
        interpolate_show_MIP(i, mip, borne_max, spacing=spacing)

    filenames=glob.glob('MIP/*.png')

    create_gif(filenames, duration)
    
#     try:
#         shutil.rmtree('test_gif/')
#     except:
#         pass


# Pad to same shape before stack
def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant') #Default is 0


# Makes more sense to return/save the interpolated MIP as nifti so can read original SUV pixels back, which allows SUV adjustment later
def create_mipNIFTI_from_3D(img, nb_image=48):
    ls_mip=[]
    
    
    img_data=img.get_fdata()
    shape = img.get_fdata().shape
    max_dim = max(shape)
    diag = int(np.ceil(np.sqrt(np.square(shape[0])+np.square(shape[1]))))
    max_dim = max(max_dim, diag)
    target_shape = (max_dim,max_dim)
    
    #Modified nifti header saving useful axial slices information
    header=img.header.copy()
    # Can't seem to create new fields but can use existing fields to store other information...
    header = img.header.copy()
    liver_idx = img_data.shape[-1]//2
    suv_liver = img_data[:,:,liver_idx].squeeze().max()
    suv_brain = img_data[:,:,-1].squeeze().max()
#     print('Liver SUV max', suv_liver)
    header['intent_p1'] = suv_liver
    header['intent_p2'] = suv_brain
    header['intent_p3'] = img_data.max()
    # Can't store too many letters...
    header['intent_name'] = b'liver;brain;max'
    
#     print('Interpolating')
    img_data+=1e-5
    for angle in tqdm(np.linspace(0,360,nb_image)):
        #ls_slice=[]
        # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
#         vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        vol_angle = scipy.ndimage.rotate(img_data,angle,order=0)
        
        MIP=np.amax(vol_angle,axis=1)
        MIP-=1e-5
        MIP[MIP<1e-5]=0
        MIP=np.flipud(MIP.T)
        MIP=to_shape(MIP, target_shape)
        ls_mip.append(MIP)
#         print('angle:', angle, MIP.shape)
    
    new_data = np.dstack(ls_mip) #shape [:,:,i]
    mip_nifti = nib.Nifti1Image(new_data, None, header)
    
    return mip_nifti


def rescale_mipNIFTI(img, seg):
    ls_mip=[]
    ls_seg=[]
    
    img_data = img.get_fdata()
    header = img.header.copy()
    nb_image = img_data.shape[2]
    seg_data = seg.get_fdata()
    seg_header = seg.header.copy()

    max_edge = 0
    for idx in range(nb_image):
        mip = img_data[:,:,idx].copy()
        seg_mip = seg_data[:,:,idx].copy()
        
        coords = cv2.findNonZero(mip)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        target_shape = (w, int(header['pixdim'][3]/header['pixdim'][1]*h))
        max_edge = max([max_edge, np.max(target_shape)])
        
        rect = mip[y:y+h, x:x+w]
        seg_rect = seg_mip[y:y+h, x:x+w]
        
        new_slice = cv2.resize(rect, dsize=target_shape, interpolation=cv2.INTER_LINEAR)
        ls_mip.append(new_slice)
        new_seg = cv2.resize(seg_rect, dsize=target_shape, interpolation=cv2.INTER_LINEAR)
        ls_seg.append(new_seg)

    ls_padded = []
    ls_seg_padded = []
    for new_slice, new_seg in zip(ls_mip, ls_seg):
        new_padded = to_shape(new_slice, (max_edge,max_edge))
        ls_padded.append(new_padded)
        seg_padded = to_shape(new_seg, (max_edge,max_edge))
        ls_seg_padded.append(seg_padded)
    
    new_data = np.dstack(ls_padded) #shape [:,:,i]
    mip_nifti = nib.Nifti1Image(new_data, None, header)
    new_seg_data = np.dstack(ls_seg_padded) #shape [:,:,i]
    seg_nifti = nib.Nifti1Image(new_seg_data, None, seg_header)
    
    return mip_nifti, seg_nifti


def find_studies(path_to_data):
    # find all studies
    root = plb.Path(path_to_data)
    patient_dirs = list(root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        #print(sub_dirs)
        study_dirs.extend(sub_dirs)

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


def convert_axial_niis_to_MIP(study_dirs, nii_out_root):
    # batch conversion of all patients
    for study_dir in tqdm(study_dirs):
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)
        
        # Preserving same diretory structure as original tcia dataset
        nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        nii_out_path = nii_out_path/study_dir.name
        os.makedirs(nii_out_path, exist_ok=True) #leaves dir unaltered if already exists
        
        print('Processing SUV.nii.gz', patient)
        img = nib.load(os.path.join(study_dir, 'SUV.nii.gz'))
        mip_nifti = create_mipNIFTI_from_3D(img, nb_image=48) #48 is the number of MIP slices in MIM available to rads
        nib.save(mip_nifti, os.path.join(nii_out_path, 'SUV_MIP.nii.gz'))
        
        print('Processing SEG.nii.gz', patient)
        img = nib.load(os.path.join(study_dir, 'SEG.nii.gz'))
        mip_nifti = create_mipNIFTI_from_3D(img, nb_image=48) #48 is the number of MIP slices in MIM available to rads
        nib.save(mip_nifti, os.path.join(nii_out_path, 'SEG_MIP.nii.gz'))

        
def rescale_all_MIP(study_dirs, nii_out_root):
    # batch rescaling of all patients
    for study_dir in tqdm(study_dirs):
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)
        
        # Preserving same diretory structure as original tcia dataset
        nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        nii_out_path = nii_out_path/study_dir.name
        os.makedirs(nii_out_path, exist_ok=True) #leaves dir unaltered if already exists
        
        print('Processing SUV_MIP.nii.gz and SEG_MIP.nii.gz for:', patient)
        img = nib.load(os.path.join(study_dir, 'SUV_MIP.nii.gz'))
        seg = nib.load(os.path.join(study_dir, 'SEG_MIP.nii.gz'))
        mip_rescale, seg_rescale = rescale_mipNIFTI(img, seg)
        nib.save(mip_rescale, os.path.join(nii_out_path, 'SUV_MIP_rescale.nii.gz'))
        nib.save(seg_rescale, os.path.join(nii_out_path, 'SEG_MIP_rescale.nii.gz'))
    
    
# Process all the SUV.nii.gz to a MIP_SUV.nii.gz
if __name__ == "__main__":
    nii_in_root = plb.Path(sys.argv[1])  # path to parent directory for all studies, e.g. '...datasets/NIFTI/FDG-PET-CT-Lesions/'
    nii_out_root = plb.Path(sys.argv[2])  # path to where we want to MIP nifti files, e.g. '...datasets/NIFTI_MIP/FDG-PET-CT-Lesions/')
    
#     study_dirs = find_studies(nii_in_root)

    unprocessed_only = False
    if unprocessed_only:
        study_dirs = find_unprocessed_studies(nii_in_root, nii_out_root)
    else:
        study_dirs = find_studies(nii_in_root)
    
    # converting axial to MIP (too much padding)
    #convert_axial_niis_to_MIP(study_dirs, nii_out_root)
    
    # Rescaling to remove white space border and to get same aspect ratio per PET affine 
    # Assumes have run prior axial to MIP conversion already. Can write out in same directory, e.g.:
    # python GIF_mip.py /gpfs/fs0/data/stanford_data/petct/NIFTI_MIP/FDG-PET-CT-Lesions/ /gpfs/fs0/data/stanford_data/petct/NIFTI_MIP/FDG-PET-CT-Lesions/
    rescale_all_MIP(study_dirs, nii_out_root)
    
    
    
