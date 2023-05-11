import os

import nibabel as nib
import tables
import pandas as pd
import glob 
import importlib

import keras
from keras import backend as K
from keras.engine import Input, Model
from keras import activations

# For deepexplain
from deepexplain.tensorflow import DeepExplain
from keras.utils import to_categorical

from skimage import feature, transform
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import copy

from fm.fm_vis import get_largest_slice
from unet3d.training import load_old_model
from unet3d.utils import pickle_load

def show_slice(slicenum, data):
    """ Function to display single slice """
    fig, axes = plt.subplots(figsize=(10,8),frameon = False)
    pos = axes.imshow(data[:,:,slicenum].T, cmap="jet", origin="lower")
    fig.colorbar(pos, ax=axes)
    plt.axis('off')
    plt.show()


def plot_deepexplain(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    # Source: https://github.com/marcoancona/DeepExplain/blob/master/examples/utils.py

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    # cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi = copy.copy(mpl.cm.get_cmap("Greys_r"))
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max, origin="lower")
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha, origin="lower")
    axis.axis('off')
    return axis

def de_mosaic_plot(slicenum, fm, T1c, savepath):
    # DE MOSAIC: create a mosaic plot
    fig, axes = plt.subplots(5,5, figsize = (50,50))

    for i, ax in enumerate(axes.ravel()):
        slicenum = slicenum-12+i
        plot_deepexplain(fm[:,:,slicenum].T, xi = T1c[:,:,slicenum].T, axis=ax, alpha = 0.3).set_title("Slice = {}".format(slicenum))

        plt.axis('off')

    # Save plot
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def de_standard_plot(slicenum, fm, T1c, gt, savepath):
    # create custom colormap for displaying GT overlay
    color1 = colorConverter.to_rgba('black')
    color2 = colorConverter.to_rgba('red')
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    cmap2._init() # create the _lut array, with rgba values


    fig, axes = plt.subplots(1,3, figsize=(30,10))

    # Plot scan
    axes[0].imshow(T1c[:,:,slicenum].T, cmap="gray", origin="lower")

    # GT overlay
    axes[0].imshow(gt[:,:,slicenum].T, cmap=cmap2, origin="lower",alpha=0.3)
    axes[0].axis('off')
    axes[0].set_title("T1c scan")

    # Plot attributions_gradin
    # Source: https://github.com/marcoancona/DeepExplain/blob/master/examples/mint_cnn_keras.ipynb
    plot_deepexplain(fm[:,:,slicenum].T, axis=axes[1], alpha = 0.3)
    plot_deepexplain(fm[:,:,slicenum].T, xi = T1c[:,:,slicenum].T, axis=axes[2], alpha = 0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


# Source: https://github.com/marcoancona/DeepExplain

def main(model, config, affine, case_directory, test_data, pred, attribution_method = 'grad*input'):

        # attribution_method: Choose from 'grad*input', 'saliency', 'intgrad', 'deeplift', 'elrp',

        # Get saliency map
        # Refer the API documentation for using deepexplain as below
        with DeepExplain(session=K.get_session()) as de:
            # Need to reconstruct the graph in DeepExplain context, using the same weights.
            # With Keras this is very easy:
            # 1. Get the input tensor to the original model
            input_tensor = model.layers[0].input

            # 2. We now target the output of the last dense layer (pre-softmax)
            # To do so, create a new model sharing the same layers until the last dense (index -2)
            # fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
            fModel = Model(inputs=model.inputs, outputs = model.layers[-1].output)
            # print(fModel.layers[-1].activation)
            fModel.layers[-1].activation = activations.linear
            # print(fModel.layers[-1].activation)

            target_tensor = fModel(input_tensor)

            xs = test_data
            ys_1 = to_categorical(pred, num_classes=len(config["labels_to_use"]))
            ys = ys_1.reshape(1,-1)

            if attribution_method not in ['occlusion', 'shapley_sampling']:
                attributions = de.explain(attribution_method, target_tensor, fModel.inputs[0], xs, ys = ys)
            elif attribution_method == 'occlusion':
                attributions = de.explain(attribution_method, target_tensor, fModel.inputs[0], xs, ys=ys, window_shape=(1, 5, 5, 2), step=2)
            elif attribution_method == 'shapley_sampling':
                attributions_sv     = de.explain(attribution_method, target_tensor, fModel.inputs[0], xs, ys=ys, samples=100)

            nib.Nifti1Image(np.squeeze(attributions), affine).to_filename(os.path.join(case_directory, "attributions_{}.nii.gz".format(attribution_method)))

            # DeepExplain PLOT
            # Determine the slice with biggest tumor section from truth.nii.gz
            gt = nib.load(os.path.join(case_directory, "truth.nii.gz")).get_fdata()
            T1c = nib.load(os.path.join(case_directory, "data_T1c_subtrMeanDivStd.nii.gz")).get_fdata()
            fm = nib.load(os.path.join(case_directory, "attributions_{}.nii.gz".format(attribution_method))).get_fdata()

            largest_tumor_slice = get_largest_slice(gt)

            de_standard_plot(largest_tumor_slice, fm, T1c, gt, savepath=os.path.join(case_directory, "fm_de_standard_plot.png"))
            de_mosaic_plot(largest_tumor_slice, fm, T1c, savepath=os.path.join(case_directory, "fm_de_mosaic_plot.png"))
