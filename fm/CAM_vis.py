import glob
import importlib
import os

# For plotting ROC
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
# For CAM
import scipy.ndimage
import tables
from keras.engine import Model

from unet3d.training import load_old_model


def show_slice(slicenum, data):
    """ Function to display single slice """
    fig, axes = plt.subplots(figsize=(10,8),frameon = False)
    pos = axes.imshow(data[:,:,slicenum].T, cmap="jet", origin="lower")
    fig.colorbar(pos, ax=axes)
    plt.axis('off')
    plt.show()


def main(last_conv_output, affine, case_directory, layer_weights, pred):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ i, layer.name, layer.output.shape ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # 0 input_1 (?, 1, 144, 144, 144)
    # 1 conv3d_1 (?, 16, 144, 144, 144)
    # 2 instance_normalization_1 (?, 16, 144, 144, 144)
    # 3 leaky_re_lu_1 (?, 16, 144, 144, 144)
    # 4 conv3d_2 (?, 16, 144, 144, 144)
    # 5 instance_normalization_2 (?, 16, 144, 144, 144)
    # 6 leaky_re_lu_2 (?, 16, 144, 144, 144)
    # 7 spatial_dropout3d_1 (?, 16, 144, 144, 144)
    # 8 conv3d_3 (?, 16, 144, 144, 144)
    # 9 instance_normalization_3 (?, 16, 144, 144, 144)
    # 10 leaky_re_lu_3 (?, 16, 144, 144, 144)
    # 11 add_1 (?, 16, 144, 144, 144)
    # 12 conv3d_4 (?, 32, 72, 72, 72)
    # 13 instance_normalization_4 (?, 32, 72, 72, 72)
    # 14 leaky_re_lu_4 (?, 32, 72, 72, 72)
    # 15 conv3d_5 (?, 32, 72, 72, 72)
    # 16 instance_normalization_5 (?, 32, 72, 72, 72)
    # 17 leaky_re_lu_5 (?, 32, 72, 72, 72)
    # 18 spatial_dropout3d_2 (?, 32, 72, 72, 72)
    # 19 conv3d_6 (?, 32, 72, 72, 72)
    # 20 instance_normalization_6 (?, 32, 72, 72, 72)
    # 21 leaky_re_lu_6 (?, 32, 72, 72, 72)
    # 22 add_2 (?, 32, 72, 72, 72)
    # 23 conv3d_7 (?, 64, 36, 36, 36)
    # 24 instance_normalization_7 (?, 64, 36, 36, 36)
    # 25 leaky_re_lu_7 (?, 64, 36, 36, 36)
    # 26 conv3d_8 (?, 64, 36, 36, 36)
    # 27 instance_normalization_8 (?, 64, 36, 36, 36)
    # 28 leaky_re_lu_8 (?, 64, 36, 36, 36)
    # 29 spatial_dropout3d_3 (?, 64, 36, 36, 36)
    # 30 conv3d_9 (?, 64, 36, 36, 36)
    # 31 instance_normalization_9 (?, 64, 36, 36, 36)
    # 32 leaky_re_lu_9 (?, 64, 36, 36, 36)
    # 33 add_3 (?, 64, 36, 36, 36)
    # 34 conv3d_10 (?, 128, 18, 18, 18)
    # 35 instance_normalization_10 (?, 128, 18, 18, 18)
    # 36 leaky_re_lu_10 (?, 128, 18, 18, 18)
    # 37 conv3d_11 (?, 128, 18, 18, 18)
    # 38 instance_normalization_11 (?, 128, 18, 18, 18)
    # 39 leaky_re_lu_11 (?, 128, 18, 18, 18)
    # 40 spatial_dropout3d_4 (?, 128, 18, 18, 18)
    # 41 conv3d_12 (?, 128, 18, 18, 18)
    # 42 instance_normalization_12 (?, 128, 18, 18, 18)
    # 43 leaky_re_lu_12 (?, 128, 18, 18, 18)
    # 44 add_4 (?, 128, 18, 18, 18)
    # 45 conv3d_13 (?, 256, 9, 9, 9)
    # 46 instance_normalization_13 (?, 256, 9, 9, 9)
    # 47 leaky_re_lu_13 (?, 256, 9, 9, 9)
    # 48 conv3d_14 (?, 256, 9, 9, 9)
    # 49 instance_normalization_14 (?, 256, 9, 9, 9)
    # 50 leaky_re_lu_14 (?, 256, 9, 9, 9)
    # 51 spatial_dropout3d_5 (?, 256, 9, 9, 9)
    # 52 conv3d_15 (?, 256, 9, 9, 9)
    # 53 instance_normalization_15 (?, 256, 9, 9, 9)
    # 54 leaky_re_lu_15 (?, 256, 9, 9, 9)
    # 55 add_5 (?, 256, 9, 9, 9)
    # 56 global_average_pooling3d_1 (?, 256)
    # 57 Dense_softmax (?, 3)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ## Get last_conv_output and save to file
    if not os.path.exists(os.path.join(case_directory, "CAM_original.nii.gz")):

        print("last_conv_output.shape = {}".format(last_conv_output.shape)) # (1, 256, 9, 9, 9)

        # change dimensions of last convolutional output to (256, 9, 9, 9)
        last_conv_output = np.squeeze(last_conv_output)

        # change dimensions of last convolutional output to (9, 9, 9, 256) and saving as last_conv_output.nii.gz
        nib.Nifti1Image(np.rollaxis(last_conv_output, 0, 4), affine).to_filename(os.path.join(case_directory, "last_conv_output.nii.gz"))


        # get CAM layer weights for that particular prediction
        CAM_layer_weights = layer_weights[:, pred].reshape(1, -1) # dim: (1, 256)
        print("[DEBUG] CAM_layer_weights.shape:",CAM_layer_weights.shape)
        np.save(os.path.join(case_directory,'CAM_layer_weights.npy'), CAM_layer_weights)

        # get class activation map for object class that is predicted to be in the image
        CAM = np.dot(CAM_layer_weights, last_conv_output.reshape((256, 8*8*8))).reshape(8,8,8)

        print("CAM.shape = {}".format(CAM.shape))

        nib.Nifti1Image(CAM, affine).to_filename(os.path.join(case_directory, "CAM_original.nii.gz"))

    if not os.path.exists(os.path.join(case_directory, "CAM_upsampled.nii.gz")):

        # bilinear upsampling to resize each filtered image to size of original image
        factor = 128/CAM.shape[1]
        req_conv_output_upsampled = scipy.ndimage.zoom(CAM, (factor, factor, factor), order=3) # dim: (10, 144, 144, 144)
        # print("req_conv_output_upsampled.shape = {}".format(req_conv_output_upsampled.shape))

        # change dimensions of req_conv_output_upsampled to (144, 144, 144, 256) and saving as req_conv_output_upsampled.nii.gz
        nib.Nifti1Image(req_conv_output_upsampled, affine).to_filename(os.path.join(case_directory, "CAM_upsampled.nii.gz"))
            

            