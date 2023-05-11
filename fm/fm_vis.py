import os
import nibabel as nib
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.ndimage
import matplotlib.patches as patches


from matplotlib.colors import colorConverter
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from unet3d.utils.utils import extract_3D_bbox

def show_slice(slicenum, data):
    """ Function to display single slice """
    fig, axes = plt.subplots(figsize=(10,8),frameon = False)
    pos = axes.imshow(data[:,:,slicenum].T, cmap="jet", origin="lower")
    fig.colorbar(pos, ax=axes)
    plt.axis('off')
    # plt.show()


def get_largest_slice(gt):
    size_of_tumor_per_slice = []

    if 4 in np.unique(gt).tolist():
        tum_idx = 4
    else:
        tum_idx = 1

    for slice in range(gt.shape[2]):
        size_of_tumor_per_slice.append(np.count_nonzero(gt[:,:, slice] == tum_idx))

    if not any(size_of_tumor_per_slice):
        largest_tumor_slice = gt.shape[2]//2
    else:
        largest_tumor_slice = np.argmax(size_of_tumor_per_slice)

    return largest_tumor_slice


def fm_viz_single_with_scan(T1c, gt, slicenum, fm, savepath):


    # T1c = test_data[0,0,:,:,:] # assuming the batchsize of test_data is 1 and T1c is the first modality
    # Convert the gt to a binary Whole Tumor map
    gt[gt>0] = 1

    x1, y1, _, x2, y2, _ = extract_3D_bbox(gt)
    

    crop_margin = 0
    # in case coordinates are out of image boundaries
    y1 = np.maximum(y1 - crop_margin, 0)
    y2 = np.minimum(y2 + crop_margin, T1c.shape[0])
    x1 = np.maximum(x1 - crop_margin, 0)
    x2 = np.minimum(x2 + crop_margin, T1c.shape[1])

    # print("x1, y1, x2, y2", x1, y1, x2, y2)

    height = x2 - x1
    width = y2 - y1
    # print("w , h", width, height)
    
    # print(T1c.shape)
    # print(fm.shape)
    # print(gt.shape)

    num_of_fms = fm.shape[-1]

    num_of_rows = round(math.sqrt(num_of_fms))
    # print(num_of_rows)

    fig, axes = plt.subplots(1,5, figsize=(25,5))
    axes[0].imshow(T1c[:,:,slicenum].T, cmap="gray", origin="lower")


    axes[0].axis('off')
    axes[0].set_title("T1c scan", fontsize=25)


    rect = patches.Rectangle((y1,x1),width, height,linewidth=2,edgecolor='r',facecolor='none', ls = '--')
    axes[0].add_patch(rect)


    for idx in range(1,len(axes)):
        pos = axes[idx].imshow(fm[:,:,:,idx+5][:,:,slicenum].T, cmap="jet", origin="lower")

        # colorbar
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pos, cax=cax)  

        axes[idx].axis('off')   
        axes[idx].set_title("FM#{}".format(idx), fontsize=25)
        
        
    plt.tight_layout()
    plt.savefig(savepath)
    # plt.show()
    plt.close()

def fm_viz_range_PC(T1c, slicenum, fm, savepath):

    T1c = np.squeeze(T1c)

    num_of_fms = fm.shape[-1]

    num_of_rows = round(math.sqrt(num_of_fms))

    fig = plt.figure(figsize=(80,80),frameon = False)

    for fm_idx in range(num_of_fms):#    
        axes = fig.add_subplot(num_of_rows,num_of_rows, fm_idx+1)
        # axes.imshow(T1c[:,:,slicenum].T, cmap="gray", origin="lower")
        pos = axes.imshow(fm[:,:,:,fm_idx][:,:,slicenum].T, cmap="jet", origin="lower")
        axes.axis('off')
        axes.set_title("FM#{}".format(fm_idx), fontsize=20)

        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        
        fig.colorbar(pos, cax=cax)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.axis('off')
    plt.close()



def main(affine, case_directory, list_of_fm_idx, list_of_req_conv_output):

        for layer_idx, req_conv_output in zip(list_of_fm_idx,list_of_req_conv_output):

            # Filename for raw fm
            req_conv_output_path = os.path.join(case_directory, "req_conv_output_4d_layer{}.nii.gz".format(layer_idx))

            # Filename for upsampled fm
            fm_start_idx = 0
            fm_end_idx = 15
            filename_with_extension = os.path.basename(req_conv_output_path)
            dot_index = filename_with_extension.index('.')
            filename_without_extension = filename_with_extension[:dot_index]
            filename_to_save = os.path.join(os.path.dirname(req_conv_output_path),"upsampled_fm_{}_to_{}_layer{}.nii.gz".format(fm_start_idx,fm_end_idx,layer_idx))

            # Filename for plots
            filename_fm_viz_single_with_scan = os.path.join(case_directory, "fm_layer{}+scan.png".format(layer_idx))
            filename_fm_viz_range_PC = os.path.join(case_directory, "fm_mosaic_layer{}.png".format(layer_idx))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Generate fm  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if not os.path.exists(req_conv_output_path):
                ## Get req_conv_output and save to file
                print("req_conv_output.shape = {}".format(req_conv_output.shape)) # (1, #c, x, y, z)

                # change dimensions of last convolutional output to (#c, x, y, z)
                req_conv_output = np.squeeze(req_conv_output)

                # change dimensions of last convolutional output to (x, y, z, #c) and saving as req_conv_output_4d_layer.nii.gz

                nib.Nifti1Image(np.rollaxis(req_conv_output, 0, 4), affine).to_filename(req_conv_output_path)


            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Upsampling  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if not os.path.exists(filename_to_save):
                req_conv_output = np.rollaxis(nib.load(req_conv_output_path).get_fdata(), 3, 0) # (256, 9, 9, 9)

                factor = 128/req_conv_output.shape[1]
                req_conv_output_upsampled = scipy.ndimage.zoom(req_conv_output[fm_start_idx:fm_end_idx], (1, factor, factor, factor), order=3) # dim: (10, 128, 128, 128)

                nib.Nifti1Image(np.rollaxis(req_conv_output_upsampled, 0, 4), affine).to_filename(filename_to_save)


            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plot  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            print("plotting for",filename_to_save)
            gt = nib.load(os.path.join(case_directory, "truth.nii.gz")).get_fdata()
            T1c = nib.load(os.path.join(case_directory, "data_T1c_subtrMeanDivStd.nii.gz")).get_fdata()
            req_conv_output_upsampled = nib.load(filename_to_save).get_fdata()
            slicenum = get_largest_slice(gt)

            fm_viz_single_with_scan(T1c, gt, slicenum, req_conv_output_upsampled, filename_fm_viz_single_with_scan)
            fm_viz_range_PC(T1c, slicenum, req_conv_output_upsampled, filename_fm_viz_range_PC)
