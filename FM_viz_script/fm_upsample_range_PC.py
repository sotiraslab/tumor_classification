import os

import nibabel as nib
import numpy as np
import glob 
import matplotlib
import matplotlib.pyplot as plt

# For CAM
import scipy.ndimage
from pathlib import Path
import time



# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAM_144x144x144  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
list_of_req_conv_output = glob.glob(os.path.abspath('./req_conv_output_4d*44.nii.gz'))
fm_start_idx = 0
fm_end_idx = 9

for req_conv_output_path in list_of_req_conv_output:
	start_time = time.time()
	filename_with_extension = os.path.basename(req_conv_output_path)
	dot_index = filename_with_extension.index('.')
	filename_without_extension = filename_with_extension[:dot_index]
	# print(filename_without_extension)
	filename_to_save = os.path.join(os.path.dirname(req_conv_output_path),"upsampled_fm_{}_to_{}_{}.nii.gz".format(fm_start_idx,fm_end_idx,filename_without_extension))

	print("FM saved at: ", filename_to_save)


	req_conv_output = np.rollaxis(nib.load(req_conv_output_path).get_fdata(), 3, 0) # (256, 9, 9, 9)
	affine = np.load('./affine.npy')
# 
	# print("[INFO] req_conv_output.shape",req_conv_output.shape)

	

	# bilinear upsampling to resize each filtered image to size of original image 
	factor = 144/req_conv_output.shape[1]
	req_conv_output_upsampled = scipy.ndimage.zoom(req_conv_output[fm_start_idx:fm_end_idx], (1, factor, factor, factor), order=3) # dim: (10, 144, 144, 144)
	# print("req_conv_output_upsampled.shape = {}".format(req_conv_output_upsampled.shape))

	# change dimensions of req_conv_output_upsampled to (144, 144, 144, 256) and saving as req_conv_output_upsampled.nii.gz
	nib.Nifti1Image(np.rollaxis(req_conv_output_upsampled, 0, 4), affine).to_filename(filename_to_save)

	# end_time = time.time()

	# print("[INFO] Time taken: {:.2f} minutes".format((end_time-start_time)/60))
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prediction of Validation cases --> simplified ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


