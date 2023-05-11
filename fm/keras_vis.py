import glob
import importlib
import os

# For plotting ROC
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tables
from keras import activations
from keras.utils import CustomObjectScope
from keras_contrib.layers import InstanceNormalization
# For keras-vis
from vis.utils import utils
from vis.visualization import visualize_cam
from vis.visualization.saliency import visualize_cam_with_preinitialized_optimizer

from unet3d.training import load_old_model
from unet3d.utils import pickle_load
from fm.fm_vis import get_largest_slice

import matplotlib.patches as patches


from matplotlib.colors import colorConverter
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from unet3d.utils.utils import extract_3D_bbox
import matplotlib.gridspec as gridspec


def plot_gradcam(T1c, gt, fm, savepath, fig_sup_title):


    # T1c = test_data[0,0,:,:,:] # assuming the batchsize of test_data is 1 and T1c is the first modality
    # Convert the gt to a binary Whole Tumor map
    gt[gt>0] = 1

    slicenum = get_largest_slice(gt)


    fig = plt.figure(figsize=(10,5))
    gs1 = gridspec.GridSpec(1,2)
    gs1.update(wspace=0.025) # set the spacing between axes. 

    # scan in axes0
    print("T1c shape", T1c.shape)
    print("fm shape", fm.shape)

    ax0 =  plt.subplot(gs1[0])
    ax1 =  plt.subplot(gs1[1])

    ax0.imshow(T1c[:,:,slicenum].T, cmap="gray", origin="lower")
    ax0.axis('off')
    # ax0.set_title("T1c scan", fontsize=25)

    if np.count_nonzero(gt) > 0:

        x1, y1, _, x2, y2, _ = extract_3D_bbox(gt)
        

        crop_margin = 0
        # in case coordinates are out of image boundaries
        y1 = np.maximum(y1 - crop_margin, 0)
        y2 = np.minimum(y2 + crop_margin, T1c.shape[0])
        x1 = np.maximum(x1 - crop_margin, 0)
        x2 = np.minimum(x2 + crop_margin, T1c.shape[1])

        
        height = x2 - x1
        width = y2 - y1

        rect = patches.Rectangle((y1,x1),width, height,linewidth=2,edgecolor='r',facecolor='none', ls = '--')
        ax0.add_patch(rect)


    ax1.imshow(T1c[:,:,slicenum].T, cmap="gray", origin="lower")
    ax1.imshow(fm[:, :, slicenum].T, cmap="jet", origin="lower", alpha= 0.5)

    # # colorbar
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(pos, cax=cax)  

    ax1.axis('off')   
    # ax1.set_title("gradcam", fontsize=25)
        
    
    fig.suptitle(str("\n".join(fig_sup_title)), fontsize=10)
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


# Source: https://github.com/raghakot/keras-vis
# https://fairyonice.github.io/Grad-CAM-with-keras-vis.html
# https://colab.research.google.com/github/idealo/cnn-exposed/blob/master/notebooks/Attribution.ipynb#scrollTo=U0rHlH4AnQPV

def main(case_directory, affine, model, test_data, pred, opt, fig_sup_title, custom_objects):

    gradcam_savepath = os.path.join(case_directory, "gradcam_conv3d_15.nii.gz")
    plot_savepath = os.path.join(case_directory, os.path.basename(case_directory)+ "_gradcam_conv3d_15.png")

    if not os.path.exists(gradcam_savepath):
                
        with CustomObjectScope(custom_objects): 
            # CAM = visualize_cam(model, 
            #                     layer_idx = -1, 
            #                     filter_indices=pred, 
            #                     seed_input=test_data, 
            #                     penultimate_layer_idx=utils.find_layer_idx(model, 'conv3d_15'),
            #                     backprop_modifier='guided')  

            CAM = visualize_cam_with_preinitialized_optimizer(model.input, seed_input=test_data, opt = opt)

        nib.Nifti1Image(CAM, affine).to_filename(gradcam_savepath)

    
    fm = nib.load(gradcam_savepath).get_fdata()
    T1c = nib.load(os.path.join(case_directory, "data_T1c_subtrMeanDivStd.nii.gz")).get_fdata()
    gt = nib.load(os.path.join(case_directory, "truth.nii.gz")).get_fdata()

    plot_gradcam(T1c, gt, fm, plot_savepath, fig_sup_title)


    
