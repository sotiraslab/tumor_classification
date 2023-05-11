import copy
import os
import glob
import pathlib
import pprint
import random
import shutil
import pandas as pd


random.seed(9001)
import numpy as np

import nibabel as nib
import tables
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
font = {'family' : 'sans-serif'}
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rc('font', **font)

from unet3d.model.model_classification import *



import logging


def show_axial_sagittal_coronal_allmods(filepath, filename, fig_sup_title):
    n_cols = 3  # three views axial, sag, coronal
    n_rows = len(glob.glob(os.path.join(filepath, "*.nii.gz")))  # as many as the num of modalities that session has
    size_of_each_fig = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * size_of_each_fig, n_rows * size_of_each_fig))

    for idx, modality in enumerate(glob.glob(os.path.join(filepath, "*.nii.gz"))):

        print(modality)

        data = np.squeeze(nib.load(modality).get_fdata())

        num_slices_sag = data.shape[0]
        num_slices_cor = data.shape[1]
        num_slices_ax = data.shape[2]

        slices = [data[:, :, num_slices_ax // 2], data[num_slices_sag // 2, :, :], data[:, num_slices_cor // 2, :]]

        for i, slice in enumerate(slices):
            axes[idx, i].imshow(slice.T, cmap="gray", origin="lower")
            axes[idx, i].axis('off')
            axes[idx, i].set_title(modality.split('\\')[-1])

    fig.suptitle(str("\n".join(fig_sup_title)), fontsize=15)

    plt.savefig(filename)
    plt.close()

def show_axial_sagittal_coronal(data_array, filename, fig_sup_title, config):
    n_cols = 3  # three views axial, sag, coronal
    n_rows = len(config["training_modalities"])  # as many as the num of modalities that session has
    size_of_each_fig = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * size_of_each_fig, n_rows * size_of_each_fig))

    for idx, modality in enumerate(config["training_modalities"]):
        data = data_array[0, idx]

        num_slices_sag = data.shape[0]
        num_slices_cor = data.shape[1]
        num_slices_ax = data.shape[2]

        slices = [data[:, :, num_slices_ax // 2], data[num_slices_sag // 2, :, :], data[:, num_slices_cor // 2, :]]

        #  If there is only one modality available
        if n_rows == 1:
            ax_idx = "[i]"
        else:
            ax_idx = "[idx, i]"

        for i, slice in enumerate(slices):
            eval("axes"+ax_idx).imshow(slice.T, cmap="gray", origin="lower")
            eval("axes"+ax_idx).axis('off')
            eval("axes"+ax_idx).set_title(modality.split('\\')[-1])

    fig.suptitle(str("\n".join(fig_sup_title)), fontsize=20)

    plt.savefig(filename)
    plt.close()

def main(fold, exp):
    config_file_name = "config_file_Exp" + exp
    config_file = importlib.import_module('config_files.' + config_file_name)
    set_fold = config_file.set_fold
    global config
    config = config_file.config
    info = config_file.info
    set_fold(fold, exp)

    # Read excel
    df = pd.read_csv(os.path.join(config['basepath'], 'df_filtered.csv'), index_col=0)

    # ***************************************************************************************************************************

    # Create and configure logger
    log_path = os.path.join(config["basepath"], "training_log.txt")
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(filename=log_path,
                        filemode='a',
                        format=LOG_FORMAT,
                        level=logging.DEBUG)

    logger = logging.getLogger(__file__)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    logging.getLogger('matplotlib.font_manager').disabled = True

    logger.info("*************************************************************************************************")
    logger.info("*" * 40 + " [ INSPECT CASES ] " + "*" * 40)
    logger.info("*************************************************************************************************")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # print(axes)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VALIDATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    hdf5_file_list = [config["data_file_tr"], config["data_file_val"], config["data_file_test"]]
    logger.info("hdf5_file_list: " + str(hdf5_file_list))

    for data_file, train_val_test in zip(hdf5_file_list, ["training", "validation", "testing"]):

        data_file = tables.open_file(data_file, "r")

        prediction_dir = os.path.abspath(config["basepath"] + "inspect_data/")

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)


        # for index in validation_keys_file:
        for index, subj_id in enumerate(data_file.root.subject_ids):
            subj_id = subj_id.decode('utf-8')

            test_data = np.asarray([data_file.root.data[index]])

            # ########################################################################################################################
            # # Uncomment following block to generate actual data
            # case_directory = os.path.join(prediction_dir, train_val_test, subj_id)
            # print(case_directory)
            #
            # if not os.path.exists(case_directory):
            #     os.makedirs(case_directory)
            #
            # affine = data_file.root.affine[index]
            #
            # for i, modality in enumerate(config["training_modalities"]):
            #     image = nib.Nifti1Image(test_data[0, i], affine)
            #     image.to_filename(os.path.join(case_directory, "data_{0}.nii.gz".format(modality)))
            # ########################################################################################################################


            plot_save_path = os.path.join(prediction_dir, "{}_{}.png".format(train_val_test, subj_id))

            my_dict = df.loc[df['sessions'] == subj_id].iloc[0].to_dict()

            fig_sup_title = []
            for key, value in my_dict.items():
                fig_sup_title = fig_sup_title + [str(key) + ' : ' + str(value)]

            fig_sup_title = [fig_sup_title[:3]] + [fig_sup_title[3:6]] + [fig_sup_title[6:8]] + [fig_sup_title[8:]]
            fig_sup_title = [str(i) for i in fig_sup_title]
            # print(fig_sup_title)
            # show_axial_sagittal_coronal_allmods(case_directory, plot_save_path, fig_sup_title)
            logger.info("Now plotting: " + str(plot_save_path))
            show_axial_sagittal_coronal(test_data, plot_save_path, fig_sup_title, config)



        data_file.close()

#



