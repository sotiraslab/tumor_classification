import copy
import os

import numpy as np
import tables

from .normalize import normalize_data_storage, reslice_image_set


def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    # https://github.com/ellisdg/3DUnetCNN/issues/58#issuecomment-415884226
    # filters = tables.Filters(complevel=0) # uncomment this if above not working
    data_shape = tuple([0, n_channels] + list(image_shape)) # (#Samples = 0,#C,H,W,D)
    truth_shape = tuple([0, 1] + list(image_shape)) # (#Samples = 0,1,H,W,D)
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape, filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4), filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage


def write_image_data_to_file(config, image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage, truth_dtype=np.uint8, crop=True, drop_idx=[], add_flipped_modality = False):
    '''
    :param image_files: list of lists where each sublist corresponds to one subject
    '''
    # print("[DEBUG] Inside write_image_data_to_file")

    idx_drop_ch1, idx_drop_ch2 = drop_idx
    # print("[DEBUG] channel1 dropped from:", idx_drop_ch1)
    # print("[DEBUG] channel2 dropped from:", idx_drop_ch2)

    for idx, set_of_files in enumerate(image_files):
        # set_of_files is the set of modalities + GT for each subject
        print("*************************************************************************************************************")
        # print("[DEBUG] idx, set_of_files", idx, set_of_files)
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        # print("[DEBUG] images", images)

        # if each scan is [R,C,Z] then subject_data is list with (num_of_modalities+1) entries, each of them are list with R entries, each of those R entries is a list of C entries,
        # and each of those C entries is a list of Z entries
        subject_data = [image.get_fdata() for image in images]

        if idx in idx_drop_ch1:
            print("WARN: Channel 1 is missing. Will make zero-filled channel.")
            subject_data[0] = np.zeros(image_shape).tolist() # Min-fill channel1 of subj

        if idx in idx_drop_ch2:
            print("WARN: Channel 2 is missing. Will make zero-filled channel.")
            subject_data[1] = np.zeros(image_shape).tolist() # Min-fill channel2 of subj

        if add_flipped_modality == True:
            T1c_scan = np.asarray(subject_data[0])
            # print("[DEBUG] Flipped modality added")
            subject_data[1] = np.flip(T1c_scan, axis=[0]).tolist()

        # print("[DEBUG] subject data shape", np.asarray(subject_data).shape)
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels, truth_dtype)

    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):

    # print("affine = ", affine)
    # print("[DEBUG] data_storage shape = ", data_storage.shape)
    # print("[DEBUG] append shape = ", np.asarray(subject_data[:n_channels])[np.newaxis].shape)
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])


def write_data_to_file(config,
                       training_data_files,
                       out_file,
                       image_shape,
                       truth_dtype=np.uint8,
                       subject_ids=None,
                       normalize=True,
                       crop=False,
                       drop_idx=([],[]),
                       add_flipped_modality = False):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of lists containing the training data files. The modalities should be listed in
    the same order in each sublist. The last item in each sublist must be the labeled image. 
    Example: [['sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'], 
              ['sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz'],
							              :
							              :
              ['subn-T1.nii.gz', 'subn-T2.nii.gz', 'subn-truth.nii.gz']]

    :param out_file: name of the hdf5 file where data will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer. 
    :return: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0]) - 1

    try:
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(config, training_data_files, data_storage, truth_storage, image_shape, truth_dtype=truth_dtype, n_channels=n_channels,
                            affine_storage=affine_storage, crop=crop, drop_idx=drop_idx, add_flipped_modality=add_flipped_modality)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    
    if normalize:
        normalize_data_storage(data_storage)
    
    hdf5_file.close()
    return out_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
