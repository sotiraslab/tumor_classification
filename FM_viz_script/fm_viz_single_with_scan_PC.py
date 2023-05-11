import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import nibabel as nib
import os 
import sys
import glob
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.ndimage


 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Determine the slice with biggest tumor section from truth.nii.gz
gt = nib.load('./truth.nii.gz').get_fdata()
# print(gt.shape)

size_of_tumor_per_slice = []
for slice in range(gt.shape[2]):
    size_of_tumor_per_slice.append(np.count_nonzero(gt[:,:, slice]))

if not any(size_of_tumor_per_slice):
	largest_tumor_slice = gt.shape[2]//2
else:
	largest_tumor_slice = np.argmax(size_of_tumor_per_slice)
# print(largest_tumor_slice)
# show_slice_axial(largest_tumor_slice, gt)
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

# generate the colors for your colormap
color1 = colorConverter.to_rgba('black')
color2 = colorConverter.to_rgba('red')
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init() # create the _lut array, with rgba values

fig, axes = plt.subplots(1,5, figsize=(25,5))
axes[0].imshow(T1c[:,:,slicenum].T, cmap="gray", origin="lower")
axes[0].imshow(gt[:,:,slicenum].T, cmap=cmap2, origin="lower",alpha=0.3)
axes[0].axis('off')
axes[0].set_title("T1c scan", fontsize=25)

for idx in range(1,len(axes)):
	pos = axes[idx].imshow(fm[:,:,:,idx][:,:,slicenum].T, cmap="jet", origin="lower")
	divider = make_axes_locatable(axes[idx])
	cax = divider.append_axes("right", size="5%", pad=0.05)
	fig.colorbar(pos, cax=cax)	
	axes[idx].axis('off')	
	axes[idx].set_title("FM#{}".format(idx), fontsize=25)
	
plt.tight_layout()
plt.savefig("fm_layer44+scan.png")
# plt.show()
plt.close()


