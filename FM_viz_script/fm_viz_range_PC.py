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
# Determine the slice with biggest tumor section from truth.nii.gz
gt = nib.load('./truth.nii.gz').get_fdata()
print(gt.shape)

size_of_tumor_per_slice = []
for slice in range(gt.shape[2]):
    size_of_tumor_per_slice.append(np.count_nonzero(gt[:,:, slice]))

if not any(size_of_tumor_per_slice):
	largest_tumor_slice = gt.shape[2]//2
else:
	largest_tumor_slice = np.argmax(size_of_tumor_per_slice)

# print(largest_tumor_slice)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

T1_path = './data_T1c_subtrMeanDivStd.nii.gz'
fm_path = glob.glob('./upsampled_fm_0_to*44.nii.gz')[0]

T1c = nib.load(T1_path).get_fdata()
fm = nib.load(fm_path).get_fdata() # (144, 144, 144, 9)

# print(fm.shape)
num_of_fms = fm.shape[-1]

num_of_rows = round(math.sqrt(num_of_fms))
# print(num_of_rows)

slicenum = largest_tumor_slice

fig = plt.figure(figsize=(25,25),frameon = False)

for fm_idx in range(num_of_fms):#    
    axes = fig.add_subplot(num_of_rows,num_of_rows, fm_idx+1)
    axes.imshow(T1c[:,:,slicenum].T, cmap="gray", origin="lower")
    pos = axes.imshow(fm[:,:,:,fm_idx][:,:,slicenum].T, cmap="jet", origin="lower", alpha = 0.5)
    axes.axis('off')
    axes.set_title("FM#{}".format(fm_idx), fontsize=10)

plt.savefig("fm_mosaic_layer44.png")
plt.axis('off')