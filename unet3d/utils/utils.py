import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like
from scipy import ndimage

from .extract_coarse_GT import extract_3D_bbox, crop_to_bbox_3D, save_numpy_2_nifti
from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):

    """    
    :param image_files: 
    :param image_shape: 
    :param crop: 
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return: 
    """

    # print("[DEBUG] Inside read_image_files")
    # print("[DEBUG]", image_files)

    label_indices = [label_indices]

    image_list = list()

    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
        # If it is a GT set interpolation to nearest
        # if index in label_indices:
            interpolation = "nearest"
            # print(interpolation, "-->", image_file)
        # otherwise interpolation is linear
        else:
            interpolation = "linear"
            # print(interpolation, "-->", image_file)
        print("Reading: {}".format(image_file))
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list


def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    if os.path.exists(os.path.abspath(in_file)):
        # print("[DEBUG] Yes, the path exists")
        image = nib.load(os.path.abspath(in_file))

        # image = set_background_zero(image, os.path.dirname(in_file))

        image = fix_shape(image) # Removes extra fourth axis if present
        # image = fix_canonical(image) # Converts all image files to RAS orientation
        # print("[DEBUG]", image.shape)

        if crop:
            # print("[DEBUG] 1")
            image = crop_img_to(image, crop, copy=True)
        if image_shape:
            # print("[DEBUG] 2")
            return resize(image, new_shape=image_shape, interpolation=interpolation)
        else:
            # print("[DEBUG] 3")
            return image
    else:
        print("[WARN] File does not exist. Zerofilling: ", os.path.abspath(in_file))
        if image_shape:
            return nib.Nifti1Image(np.zeros(image_shape), affine=None)
        else:
            return nib.Nifti1Image(np.zeros((240,240,155)), affine=None)


def crop_image_to_GT_bbox(image_files, image_shape=None, label_indices=None, crop0_or_mask1 = 0):
    # print("[DEBUG] inside crop_image_to_GT_bbox")
    mask_3d = image_files[label_indices].get_fdata().astype('int32')
    bbox2_3D = extract_3D_bbox(mask_3d)

    image_list = list()

    for index, image_file in enumerate(image_files):
        # If it is a GT set interpolation to nearest
        if index is label_indices:
            is_GT = True
        # otherwise interpolation is linear
        else:
            is_GT = False

        image = crop_to_bbox(image_file, bbox2_3D, image_shape, is_GT, crop0_or_mask1 = crop0_or_mask1)

        image_list.append(image)

    return image_list



def crop_to_bbox(image_file, bbox2_3D, image_shape, is_GT, crop0_or_mask1 = 0):
    affine = image_file.affine
    # Crop
    image_cropped_to_bbox2, image_masked_by_bbox2 = crop_to_bbox_3D(image_file.get_fdata(),
                                                                    bbox2_3D,
                                                                    preserve_tumor_shape_force_square_crop=True,
                                                                    crop_margin=2)


    if crop0_or_mask1 == 0:
        # print("taking crop")
        image_cropped_or_masked = image_cropped_to_bbox2
    else:
        # print("taking mask")
        image_cropped_or_masked = image_masked_by_bbox2
    # print("image_cropped_to_bbox2", image_cropped_to_bbox2.shape)
    # print("image_masked_by_bbox2", image_masked_by_bbox2.shape)



    dsfactor = [v / w for v, w in zip(image_shape, image_cropped_or_masked.shape)]
    print("dsfactor", dsfactor)

    if not is_GT:
        image_cropped_to_bbox2_resized = ndimage.interpolation.zoom(image_cropped_or_masked, zoom=dsfactor)
    else:
        image_cropped_to_bbox2_resized = ndimage.interpolation.zoom(image_cropped_or_masked, zoom=dsfactor, order = 0)

    # fixme: should affine be this?
    return nib.Nifti1Image(image_cropped_to_bbox2_resized, affine=affine)

    # if image_shape:
    #     image = resize(image_cropped_to_bbox2, new_shape=image_shape, interpolation='linear')
    # else:
    #     image = image_cropped_to_bbox2


def check_unique_elements(np_array):
    # Extract the end-points of the 3D bbox from the tumor mask
    unique, counts = np.unique(np_array, return_counts = True)
    return str(dict(zip(unique,counts)))

def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)

def fix_canonical(image):
    file_ort = nib.aff2axcodes(image.affine)
    
    if file_ort != ('R','A','S'):
        print("Converting to canonical (RAS orientation)")
        return nib.as_closest_canonical(image)
    else:
        # print("Image already canonical (RAS orientation)")
        return image

def set_background_zero(image, session_folder):
    brainmask = np.rint(nib.load(os.path.join(session_folder, 'brainmask.nii.gz')).get_fdata())
    return new_img_like(image, brainmask * image.get_fdata(), affine=image.affine)

