from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
# from shapely.geometry import Polygon, MultiPolygon         # (pip install Shapely)
import os
import json
import pathlib as plb
import nibabel as nib
import scipy
import matplotlib.pyplot as plt
import glob
import cv2
from pycocotools import mask as M
from datetime import date
import pandas as pd
import os
import sys
from tqdm import tqdm
import base64


############## This stuff should ideally but out in a config file

# common file names for all the nifti to be processed
global seg_fname
global pet_fname
seg_fname = 'SEG_MIP.nii.gz'
pet_fname = 'SUV_MIP.nii.gz'

# number of frames to be processed from each nifti
num_frames = 48

# info in coco format
today = date.today()
info = {
    "year": 2022,
    "version": 'v1',
    "description": "COCO FDG PET/CT MIP Tumor Detection",
    "contributor": ["University Hospital Tübingen and University Hospital of the LMU, Germany", "Stanford University, USA", "IBM Research, USA"],
    "url": "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287#9325828763a33c8a5d664f64be6158c55afcef63",
    "source_data_citation": "Gatidis S, Kuestner T. (2022) A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions) [Dataset]. The Cancer Imaging Archive. DOI: 10.7937/gkr0-xv29",
    "source_date_created": "2022/06/02",
    "date_rendered_in_COCO": str(today)
}

# License from source
source_license = {
    "id": 1,
    "name": "TCIA Restricted License Agreement",
    "url": "https://wiki.cancerimagingarchive.net/download/attachments/4556915/TCIA%20Restricted%20License%2020220519.pdf?version=1&modificationDate=1652964581655&api=v2"
}

# Label ids of the dataset
category_ids = {
    "background": 0,
    "tumor": 1,
}


############## 

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


################

def read_nii(nii_fname, study_path):
    img = nib.load(os.path.join(study_path, nii_fname))
    data = img.get_fdata()
    header = img.header
    return data, header


# useful for visualization
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
        fig.savefig(os.path.join(OUTPATH,'%04d' % (i)+'.png'), dpi = dpi)
    if not show:
        plt.close(fig)
    
    
# Get connected components per frame. If negative (all 0s mask) frame, will return empty lists.    
def get_connected_componets_per_frame(frame):
    #https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
    masks = []
    areas = []
    bboxes = []
    
    # Some images are negative - normal negative no objects frames
    gray_img = frame.astype("uint8")
    
    # Initialize a new image to
    # store all the output components
    output = np.zeros(gray_img.shape, dtype="uint8")
    
    # get connected components if there are any segmentations
    if gray_img.max() > 0:
        # Applying threshold
        threshold = cv2.threshold(gray_img, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(gray_img,4,cv2.CV_32S)
        (totalLabels, label_ids, values, centroids) = analysis
        
        # The first element is for the original frame
        centroids = centroids[1:]

        # Loop through each component
        for i in range(1, totalLabels): # The first value is just the original image with all components
            area = values[i, cv2.CC_STAT_AREA]
            areas.append(area)

            # Now extract the coordinate points
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            bbox = (x1, y1, w, h)
            bboxes.append(bbox)

    #         # Coordinate of the bounding box -- dont need this for coco
    #         pt1 = (x1, y1)
    #         pt2 = (x1+ w, y1+ h)
    #         (X, Y) = centroid[i]

            # Create a new array to show individual component
            component = np.zeros(gray_img.shape, dtype="uint8")
            componentMask = (label_ids == i).astype("uint8") * 255

            # Apply the mask using the bitwise operator
            component = cv2.bitwise_or(component,componentMask)
            masks.append(component)
            output = cv2.bitwise_or(output, componentMask)
    else:
        # else returns empty elements for annotation
        masks.append(gray_img)
        areas.append(0)
        centroids = []
    
    return masks, areas, bboxes, centroids, output

    
def binary_mask_to_rle(binary_mask):
    binary_mask = np.asfortranarray(binary_mask)
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


# decode rle back to binary mask
def decode_rle(rle):
    compressed_rle = M.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    binary_mask = M.decode(compressed_rle)
    return binary_mask


# Get all the segmentation from each frame/slice of a nifti
def get_annotations_per_frame(frame_idx, frame, image_id, category_id, annotation_id, diagnosis):
    
    masks, areas, bboxes, centroids, output = get_connected_componets_per_frame(frame)
    
    annotations = []
    # if not negative/empty masks
    if len(bboxes) > 0:
        for i, (binary_mask, area, bbox, centroid) in enumerate(zip(masks, areas, bboxes, centroids)):
            # each binary mask is for a single tumor lesion 
            # object detection task will be like identify all individual traffic lights separately from the image
            rle_seg = binary_mask_to_rle(binary_mask)
            annotation = create_annotation_format(rle_seg, area, image_id, frame_idx, bbox, category_id, annotation_id, diagnosis)
            annotations.append(annotation)
            annotation_id = annotation_id + 1 
    # otherwise adds and empty annotation without incrementing the annotation id
    else:
        rle_seg = binary_mask_to_rle(masks[0])
        annotation = create_annotation_format(rle_seg, areas[0], image_id, frame_idx, bboxes, 0, None, diagnosis)
        annotations.append(annotation)
    
    return annotations, annotation_id


# Get all the segmentation from sampled frames of each nifti
def get_annotations_images_per_nifti(data, row, image_id, sampled_frames, category_id, annotation_id):
    # "images" info: w and h same for frames from same nifti
    w, h = data[:,:,0].shape
    images = []
    annotations = []
    
    # get study information
    diagnosis = row['diagnosis']
    age = row['age']
    sex = row['sex']
    
    # get MIP PET images and useful nifti stats from nifti header
    pet_path = os.path.join(data_in_root, row['relative_path'], pet_fname)
    #img_pet = nib.load(pet_path)
    #pet = img_pet.get_fdata().astype("uint8")
    #header = img_pet.header.copy()
    #stats = dict()
    #stats['liver'] = header['intent_p1']
    #stats['brain'] = header['intent_p2'] 
    #stats['suv_max'] = header['intent_p3'] 
    #stats['pixdim'] = header['pixdim']

    # Get annotations for each frame
    for frame_idx in sampled_frames:
        frame = data[:,:,frame_idx].squeeze()
        frame_annotations, annotation_id = get_annotations_per_frame(frame_idx, frame
                                                                     , image_id, category_id, annotation_id, diagnosis)
        annotations.extend(frame_annotations)
        
        # create_image_annotation returns a dictionary for each nifti
        #mip_pet = pet[:,:,frame_idx].copy().squeeze().tolist()
        #mip_pet = base64.encodebytes(img).decode('utf-8')
        image = create_image_annotation(pet_path, frame_idx, w, h, image_id, age, sex)
        images.append(image)
        
        # update image_id at the end of each frame
        image_id = image_id + 1
        
    return annotations, annotation_id, images, image_id


###

def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def create_image_annotation(file_name, frame_idx, width, height, image_id,  age, sex):
    image = {
        "file_name": file_name,
        "frame_idx": frame_idx,
        #"image": pet_mip,
        "width": width,
        "height": height,
        "id": image_id,
        "age": age,
        "sex": sex
        #"nii_stats": nii_stats
    }

    return image


def create_annotation_format(rle_seg, area, image_id, frame_idx, bbox, category_id, annotation_id, diagnosis):
    # COCO format for a RLE segmentation annotation
    annotation = {
        "segmentation": rle_seg,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "frame_idx": frame_idx, 
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id,
        "diagnosis": diagnosis
    }
    
    return annotation


# create coco json format
def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format


def get_relative_studypath(fullpath):
    pathdirs = fullpath.split('/')
    return os.path.join(pathdirs[2],pathdirs[3])


# Create reproducible train test split from .csv from the Tubingen dataset
def train_test_split(csvpath):
    tab = pd.read_csv(csvpath)
    tab['relative_path'] = [get_relative_studypath(fullpath) for fullpath in tab['File Location']]
    # sort alphabetically by subject ID to obtain same split
    all_subjects = sorted(list(set(tab['Subject ID'])))
    train_ids = all_subjects[:800]
    test_ids = all_subjects[800:]
    keep = ['Study UID', 'relative_path','diagnosis', 'age', 'sex']
    train_tab = tab[tab['Subject ID'].isin(train_ids)][keep].drop_duplicates().copy()
    test_tab = tab[tab['Subject ID'].isin(test_ids)][keep].drop_duplicates().copy()
    
    return train_tab, test_tab


# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)    
    
    
# Create dataset from list of nifties
def images_annotations_info(data_in_root, table, num_frames, image_id, annotation_id, keyword, data_out_root):
    category_id = 1 # just tumor for this dataset
    
    # For each nifti study, write out a json in COCO format (otherwise memory issue)
    for i, row in tqdm(table.iterrows(), total=table.shape[0]):
        
        # path to nifti to read
        seg_path = os.path.join(data_in_root, row['relative_path'], seg_fname)
        print('Processing:', seg_path)
        
        # load segmentation MIP image nifti
        img_seg = nib.load(seg_path)
        data = img_seg.get_fdata()
        
        if num_frames > data.shape[-1]:
            raise Exception("Number of frames exceeded number of nifti slices")
        else:
            # Evenly sample num_frames from nifty volume, returns frame indices
            sampled_frames = np.linspace(0, data.shape[-1] - 1, num_frames).astype(int)
            
            # Get the standard COCO JSON format
            coco_format = get_coco_json_format()

            # Get info and license:
            coco_format['info'] = info # one dictionary
            coco_format['licenses'].append(source_license) # a list of dictionaries

            # Create category section
            coco_format["categories"] = create_category_annotation(category_ids) #returns a list of dictionaries

            # Get annotations from num_frames from the nifti --> outputs a list of dictionaries and an updated annotation_id
            coco_format["annotations"], annotation_id, coco_format["images"], image_id = get_annotations_images_per_nifti(data, row
                                                               , image_id, sampled_frames, category_id, annotation_id)
            
            
            # write out a json with annotations and pet MIP images per nifti
            with open(os.path.join(data_out_root, 'annotations', keyword, keyword + "_" + "{}.json".format(i)),"w") as outfile:
                json.dump(coco_format, outfile, cls=NpEncoder)
    
    return image_id, annotation_id
      
    

if __name__ == "__main__":
    # set paths
    csvpath = "/home/joytywu/Documents/Gaze_PET-CT/dataset/Clinical_Metadata_FDG_PET_CT_Lesions.csv" # full path to downloaded csv (from: https://wiki.cancerimagingarchive.net/download/attachments/93258287/Clinical%20Metadata%20FDG%20PET_CT%20Lesions.csv?api=v2)
    data_in_root = plb.Path(sys.argv[1])  # path to parent directory for all studies, e.g. '...datasets/NIFTI_MIP/FDG-PET-CT-Lesions/'
    data_out_root = plb.Path(sys.argv[2])  # path to where we want to save the COCO jsons, e.g. '...datasets/JSONs/FDG-PET-CT-Lesions/')
    if not os.path.isdir(data_out_root):
        os.makedirs(os.path.join(data_out_root,'annotations','train'))
        os.makedirs(os.path.join(data_out_root,'annotations','test'))
    
    # Get train test split for MIP PETCT coco dataset
    train_tab, test_tab = train_test_split(csvpath)
    
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    for keyword, table in zip(["train", "test"],[train_tab,test_tab]):
        
        # Create images and annotations sections per nifty
        image_id, annotation_id = images_annotations_info(data_in_root, table, num_frames, image_id, annotation_id
                                                         , keyword, data_out_root)
        
        print("Number of nifti processed: %s" % (table.shape[0],keyword))
        print("Created %d image frames in folder: %s" % (image_id, keyword))
        print("Created %d annotations in folder: %s" % (annotation_id, keyword))


# run from terminal e.g.:
# python src/create_annotations_petct.py /media/storage/Joy/datasets/NIFTI_MIP/FDG-PET-CT-Lesions/ /media/storage/Joy/datasets/JSONs/FDG-PET-CT-Lesions/