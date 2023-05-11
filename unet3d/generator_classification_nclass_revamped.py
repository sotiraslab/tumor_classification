import math
import os
import copy
from random import shuffle
import itertools
import nibabel as nib
import numpy as np
import keras
import pandas as pd
import random
import string

from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y


def get_training_and_validation_generators_classification(df,
        logger,
                                                        config_file,
                                                        data_file_tr, 
                                                        data_file_val,                                                       

                                                        n_labels, 
                                                        training_keys_file, 
                                                        validation_keys_file,
                                                        labels,  
                                                        
                                                        batch_size, 
                                                        overwrite=False,                                                 

                                                        augment=False, 
                                                        augment_flip=True, 
                                                        augment_distortion_factor=0.25, 
                                                        augment_intensity_shift_scale_prms = { 'shift': {'mu': 1.0, 'std':0.25}, 'scale':{'mu': 1.0, 'std': 0.25} },
                                                        patch_shape=None, 
                                                        validation_patch_overlap=0, 
                                                        training_patch_start_offset=None,
                                                        validation_batch_size=None, 
                                                        skip_blank=True, 
                                                        permute=False):
    """
    Extended from get_training_and_validation_generators() for Classification
    """
    global config
    config = config_file
    
    if not validation_batch_size:
        validation_batch_size = batch_size

    # training_list, validation_list = get_validation_split(data_file,
    #                                                       data_split=data_split,
    #                                                       overwrite=overwrite,
    #                                                       training_file=training_keys_file,
    #                                                       validation_file=validation_keys_file)

    # training_list: List of integers: [0, 1, .. upto nb_samples_tr]
    # validation_list: List of integers: [0, 1, .. upto nb_samples_val]
    
    list_of_training_list = list()
    list_of_validation_list = list()

    for i in range(len(data_file_tr)):
        training_list, validation_list = get_training_validation_lists(data_file_tr[i], 
                                                                        data_file_val[i], 
                                                                        training_file=training_keys_file[i], 
                                                                        validation_file=validation_keys_file[i], 
                                                                        overwrite=overwrite)
        list_of_training_list.append(training_list)
        list_of_validation_list.append(validation_list)


    training_generator = data_generator_classification_shuffle(config, df,
        data_file_tr,
                                                        list_of_training_list,
                                                        n_labels,
                                                        labels,

                                                        batch_size=batch_size,
                                                        augment=augment,
                                                        augment_flip=augment_flip,
                                                        augment_distortion_factor=augment_distortion_factor,
                                                        augment_intensity_shift_scale_prms=augment_intensity_shift_scale_prms,
                                                        patch_shape=patch_shape,
                                                        patch_overlap=0,
                                                        patch_start_offset=training_patch_start_offset,
                                                        skip_blank=skip_blank,
                                                        permute=permute)

    validation_generator = data_generator_classification_shuffle(config, df,
        data_file_val,
                                                          list_of_validation_list,
                                                          n_labels,
                                                          labels,          

                                                          batch_size=validation_batch_size,
                                                          patch_shape=patch_shape,
                                                          patch_overlap=validation_patch_overlap,
                                                          skip_blank=skip_blank)

    
    total_tr_cases = [data_file_tr[i].root.data.shape[0] for i in range(len(data_file_tr))]

    total_val_cases = [data_file_val[i].root.data.shape[0] for i in range(len(data_file_val))]

    steps_train = get_number_of_steps(sum(total_tr_cases),batch_size)
    steps_val = get_number_of_steps(sum(total_val_cases),validation_batch_size)

    # If smote is True then frequency of each tumor class is equal to the frequency of maximum occurring tumor class
    if "smote" in config and config["smote"] is True:
        freq_of_maximum_occurring_tumor_class_train = max(total_tr_cases)
        steps_for_max_occurring_tumor_class = get_number_of_steps(freq_of_maximum_occurring_tumor_class_train, batch_size)
        steps_train = steps_for_max_occurring_tumor_class * len(total_tr_cases)
        print("freq_of_maximum_occurring_tumor_class_train", freq_of_maximum_occurring_tumor_class_train)


    logger.info("\n" + "[#TR_STEPS] " + str(steps_train))
    logger.info("[#VAL_STEPS] " + str(steps_val))

    return training_generator, validation_generator, steps_train, steps_val

def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return 1
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_training_validation_lists(data_file_tr, data_file_val, training_file, validation_file, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file_tr: pytables hdf5 data file for training
    :param data_file_val: pytables hdf5 data file for validation
    :param training_file:
    :param validation_file:
    :param overwrite:
    :return:
    """
    
    # print("Creating training and validation lists...")
    nb_samples_tr = data_file_tr.root.data.shape[0] # Number of training data
    nb_samples_val = data_file_val.root.data.shape[0] # Number of validation data

    training_list = list(range(nb_samples_tr)) # List of integers: [0, 1, .. upto nb_samples_tr]
    validation_list = list(range(nb_samples_val)) # List of integers: [0, 1, .. upto nb_samples_val]

            
    pickle_dump(training_list, training_file)
    pickle_dump(validation_list, validation_file)
    return training_list, validation_list


def data_generator_classification_shuffle(config, df,
                                        data_file,
                                        index_list, 
                                        n_labels,                                        
                                        labels, 
                                        
                                        batch_size=1,

                                        augment=False, 
                                        augment_flip=True, 
                                        augment_distortion_factor=0.25, 
                                        augment_intensity_shift_scale_prms={ 'shift': {'mu': 1.0, 'std':0.25}, 'scale':{'mu': 1.0, 'std': 0.25} },
                                        patch_shape=None, 
                                        patch_overlap=0, 
                                        patch_start_offset=None, 
                                        shuffle_index_list=True, 
                                        skip_blank=True, 
                                        permute=False):

    '''
    Extended from data_generator_dev() for Classification function when single data generator has to be used for the model
    In this case, the model first requires inputs from GBM and then METS from the same data_generator
    So data_generator first produces GBM data till all data gets exhausted, then starts yielding METS data
    '''

    # Read the index_list and keep filling up x_list and y_list by data and corresponding groundtruth
    # x_list, y_list have size of batch_size i.e.
    # when x_list, y_list have number of data equal to batch_size, then yield them, make the x_list,y_list empty again and restart

    # print("[DEBUG] inside data_generator")
    orig_index_list = index_list[0]
    # print("orig_index_list",orig_index_list)

    while True:
        x_list = list()
        y_list = list()
        y_list_clsfctn = list()

        index_list = copy.copy(orig_index_list) # List of integers: [0, 1, .. upto nb_samples_tr or nb_samples_val]
        
        if shuffle_index_list:
            shuffle(index_list) # if index_list was [0,1,2,3] before, after shuffling it becomes [3,1,0,2] or some other shuffled version

        # print("list of sessions to yield",index_list)

        while len(index_list) > 0: # while atleast 1 case is available

            index = index_list.pop()

            add_data(config, df,
                x_list,
                    y_list,
                    y_list_clsfctn,
                    data_file[0],
                    int(index), 
                    augment=augment, 
                    augment_flip=augment_flip, 
                    augment_distortion_factor=augment_distortion_factor, 
                    augment_intensity_shift_scale_prms=augment_intensity_shift_scale_prms,
                    patch_shape=patch_shape, 
                    skip_blank=skip_blank, 
                    permute=permute)

            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                if config["seg_classify"] == 's':
                    yield_func = convert_data_per_dataset_segmentation
                else:
                    yield_func = convert_data_per_dataset_classification_segmentation

                yield yield_func(x_list, y_list, y_list_clsfctn, n_labels=n_labels, labels=labels)  # this works as the generator

                x_list = list()
                y_list = list()
                y_list_clsfctn = list() 

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath, output_path):
    nifti_image = nib.load(reference_nifti_filepath)
    new_header = header=nifti_image.header.copy()
    image_affine = nifti_image.affine
    output_nifti = nib.nifti1.Nifti1Image(image_numpy, None, header=new_header)
    nib.save(output_nifti, output_path)

def add_data(config, df, x_list, y_list, y_list_clsfctn, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             augment_intensity_shift_scale_prms={ 'shift': {'mu': 1.0, 'std':0.25}, 'scale':{'mu': 1.0, 'std': 0.25} },
             patch_shape=False, skip_blank=True, permute=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """

    # ~~~~~~~~~~~~~~~~ Read data (specified by index) from hdf5 data_file ~~~~~~~~~~~~~~~~
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
    affine = data_file.root.affine[index]
    # print("[DEBUG] Taking this session --> ",data_file.root.subject_ids[index].decode('utf-8'))

    #
    # random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    # print("[DEBUG] random_string: ", random_string)
    # print(data.shape)
    # print(truth.shape)
    # print(np.rollaxis(data, 0, 4).shape)
    # nib.Nifti1Image(np.rollaxis(data, 0, 4), affine).to_filename('./temp/{}_before_aug_data.nii.gz'.format(random_string))
    # nib.Nifti1Image(np.squeeze(truth), affine).to_filename('./temp/{}_before_aug_truth.nii.gz'.format(random_string))
    #
    # ~~~~~~~~~~~~~~~~ Augment data ~~~~~~~~~~~~~~~~
    if augment:
        # print("[DEBUG] Augmentation performed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, 
                                                        scale_deviation=augment_distortion_factor, 
                                                        intensity_distortion =augment_intensity_shift_scale_prms)


    # ~~~~~~~~~~~~~~~~ Permute data ~~~~~~~~~~~~~~~~
    if permute:
        # print("[DEBUG] Permute axis performed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    # nib.Nifti1Image(np.rollaxis(data, 0, 4), affine).to_filename('./temp/{}_after_aug_data.nii.gz'.format(random_string))

    if config["seg_classify"] == 'c' or config["seg_classify"] == 's_c':
        # The following GT matters only if model has classification mode
        # ~~~~~~~~~~~~~~~~~~~~ Molecular GT ~~~~~~~~~~~~~~~~~~~~
        session_name = data_file.root.subject_ids[index].decode('utf-8')
        truth_clsfctn = df.loc[df['sessions'] == session_name][config['marker_column']].iloc[0]
    else:
        # otherwise fill it with a dummy variable
        truth_clsfctn = 'HGG'
    
    # print("[DEBUG] truth_clsfctn",truth_clsfctn)

    x_list.append(data)
    y_list.append(truth)
    y_list_clsfctn.append(truth_clsfctn)


def get_data_from_file(data_file, index, patch_shape=None):
    x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y



# def convert_data_per_dataset_classification(x_list, y_list, y_list_clsfctn, n_labels, labels):
#     '''
#     Extended from convert_data() for Classification purposes.
#     '''

#     x = np.asarray(x_list) # shape = (batch_size, number_of_channels, image_shape[0], image_shape[1], image_shape[2])
#     y = np.asarray(y_list) # This is the GT for segmentation

#     bs = x.shape[0]

#     y_classification = []

#     for sess in range(bs):
#         molecular_marker = y_list_clsfctn[sess]
#         class_label = config['labels_to_use'].index(molecular_marker)
#         class_label_one_hot = keras.utils.to_categorical([class_label], num_classes=len(config['labels_to_use']))
#         y_classification.append(class_label_one_hot)

#     y_classification = np.squeeze(np.asarray(y_classification), axis = 1)

#     return x, y_classification

def convert_data_per_dataset_segmentation(x_list, y_list, y_list_clsfctn, n_labels, labels):
    '''
    Extended from convert_data() for Segmentation purposes.
    '''
    x = np.asarray(x_list) # This is the data for input.       shape = (batch_size, number_of_channels, image_shape[0], image_shape[1], image_shape[2])
    y = np.asarray(y_list) # This is the GT for segmentation,  shape = (batch_size, 1, image_shape[0], image_shape[1], image_shape[2])

    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        if labels == (0, 1):
            y[y > 0] = 1
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels) 

    return x, y


def convert_data_per_dataset_classification_segmentation(x_list, y_list, y_list_clsfctn, n_labels, labels):
    '''
    Extended from convert_data() for Segmentation purposes.
    '''

    x = np.asarray(x_list) # This is the data for input.       shape = (batch_size, number_of_channels, image_shape[0], image_shape[1], image_shape[2])
    y = np.asarray(y_list) # This is the GT for segmentation,  shape = (batch_size, 1, image_shape[0], image_shape[1], image_shape[2])

    if labels == (0, 1):
        # Prepare segmentation GT: Take OTMulticlass and make it binary by merging all tumor labels to 1
        y[y>0] = 1

    # for i in range(y.shape[0]):
    #     print("[DEBUG] unique before split", y[i].shape, check_unique_elements(y[i]))

    # y.shape after get_multi_class_labels = (batch_size, number_of_seg_classes, image_shape[0], image_shape[1], image_shape[2])
    y_segmentation = get_multi_class_labels(y, n_labels=n_labels, labels=labels)

    # if labels is (0,1,2,4,5) then for last label add the whole tumor map
    if labels == (0, 1, 2, 4, 5):
        y_wt = copy.deepcopy(y)
        y_wt[y_wt > 0] = 1 # (bs, 128, 128, 128) -- multilabel

        y_wt_multiclass = get_multi_class_labels(y_wt, n_labels=2, labels=(0, 1)) # (bs, 2, 128, 128, 128) -- first dim is background, second is foreground i.e. WT

        y_wt_foreground = y_wt_multiclass[:, 1, :, :, :] # (bs, 128, 128, 128) -- extract only WT tumor axis
        y_wt_foreground = np.expand_dims(y_wt_foreground, axis = 1)  # (bs, 1, 128, 128, 128) -- expand axis for concatenation

        y_segmentation = np.concatenate((y_segmentation[:,:-1,:,:,:], y_wt_foreground), axis = 1) # (bs, 5, 128, 128, 128) -- bg, necrotic, edema, enhancing, whole tumor


    # print("[DEBUG] unique after split")
    #
    # for i in range(y.shape[0]):
    #     print("case no.", i)
    #     gt_i = y[i]
    #     for j in range(gt_i.shape[0]):
    #         gt = gt_i[j]
    #         print(gt.shape, check_unique_elements(gt))

    # Prepare classification GT
    bs = x.shape[0]
    y_classification = []

    for sess in range(bs):
        classlabel = y_list_clsfctn[sess]
        print(classlabel)
        class_label = config['labels_to_use'].index(classlabel)
        class_label_one_hot = keras.utils.to_categorical([class_label], num_classes=len(config['labels_to_use']))
        y_classification.append(class_label_one_hot)

    y_classification = np.squeeze(np.asarray(y_classification), axis = 1)

    return (x, {"clsfctn_op": y_classification, "segm_op": y_segmentation})

def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y