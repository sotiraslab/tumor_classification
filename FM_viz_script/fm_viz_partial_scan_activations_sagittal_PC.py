import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import os 
import sys
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
T1_path = './data_T1c_subtrMeanDivStd.nii.gz'
fm_path = glob.glob('./upsampled_fm_0_to*44.nii.gz')[0]

T1c = nib.load(T1_path).get_fdata()
fm = nib.load(fm_path).get_fdata() # (144, 144, 144, 9)

T1c_fm_concat = np.concatenate((np.expand_dims(T1c,axis=3),fm),axis=3)
print(T1c_fm_concat.shape)

data = T1c

# AXIAL

# num_slices_ax = data.shape[2]

# # Using the condition "if len(np.unique(data[:,:,i]))>1" to remove slices with just background and because background value is not 0
# # slices = [data[:,:,i] for i in range(0,num_slices_ax,5) if len(np.unique(data[:,:,i]))>1]
# slices = [data[:,:,i] for i in range(0,num_slices_ax,5)]

# num_columns = len(slices)
# num_rows = 10

# fig, axes = plt.subplots(num_rows, num_columns, figsize=(30,10))

# for j in range(num_rows):
#     slices = [T1c_fm_concat[:,:,:,j][:,:,m] for m in range(0,num_slices_ax,5)]
#     for i, slice in enumerate(slices):
#         if j == 0:
#             colmap = "gray"
#         else:
#             colmap = "jet"
#         axes[j,i].imshow(slice.T, cmap=colmap, origin="lower")
#         axes[j,i].axis('off')

# # plt.show()
# # plt.close()
# plt.savefig("fm_partial_activations.png")
# plt.axis('off')

# SAG

num_slices_sag = data.shape[0]

# Using the condition "if len(np.unique(data[:,:,i]))>1" to remove slices with just background and because background value is not 0
# slices = [data[:,:,i] for i in range(0,num_slices_sag,5) if len(np.unique(data[:,:,i]))>1]
slices = [data[i,:,:] for i in range(0,num_slices_sag,5)]

num_columns = len(slices)
num_rows = 10

fig, axes = plt.subplots(num_rows, num_columns, figsize=(30,10))

for j in range(num_rows):
    slices = [T1c_fm_concat[:,:,:,j][m,:,:] for m in range(0,num_slices_sag,5)]
    for i, slice in enumerate(slices):
        if j == 0:
            colmap = "gray"
        else:
            colmap = "jet"
        axes[j,i].imshow(slice.T, cmap=colmap, origin="lower")
        axes[j,i].axis('off')

plt.show()
plt.close()
# plt.savefig("fm_partial_activations.png")
# plt.axis('off')