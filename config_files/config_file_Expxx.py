import os
import random
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold

from brats_nclass_testing_half.train_isensee2017 import trim_df_by_dropping_nans, trim_df_based_on_GT, filter_sessions_based_on_availability_in_scratch, \
    filter_sessions_based_on_availability_of_modalities, tumor_zoo_params_dict_list, split_data_into_n_folds, split_val_test, trim_df_based_on_Tumor_modality, check_or_create_tumor_modality, \
    split_data_into_n_folds_from_path

random.seed(9001)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from unet3d.model.model_classification import *

config = dict()
info = dict()

# Set the molecular parameter for this experiment
config["tumor_type"] = list(tumor_zoo_params_dict_list.keys())
config['labels_to_use'] = config["tumor_type"]
config['marker_column'] = 'Tumor_type'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config['labels'] = (0, 1, 2, 4)
config['n_labels'] = len(config['labels'])
config['patch_shape'] = None
config['path_to_data'] = '/scratch/satrajit/data/'

config['excel_path'] = "/home/satrajit/tumor_classification_revamped/tumor_classification.xlsx"

config["all_modalities"] = ["T1c_subtrMeanDivStd", "T2_subtrMeanDivStd", "Flair_subtrMeanDivStd"]
config["bbox_crop_or_mask"] = None # None or 0 for crop, 1 for mask (code for 'mask' is not validated)
config['crop_to_foreground'] = False
config['normalize_data_using_cohort_mean_and_std'] = True

config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["truth"] = ["OTMultiClass"]
config["nb_channels"] = len(config["training_modalities"])
config["truth_channel"] = config["nb_channels"]

config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))  # (1,128,128,128)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Choose the model type: s / s_c/ c
config["seg_classify"] = "s"
# Choose model from:
# 1. isensee2017_classification_segmentation
# 2. Resnet3DBuilder.build_resnet_"x", x = 18, 34, 50, 101, 152
config["model"] = (isensee2017_classification_segmentation,)
config['loss_function_seg'] = weighted_dice_coefficient_loss
config['use_attention_gate'] = False
# this specifies if fm from downsampling path is passed through sq-ex before feeding into attention block as input. this is only valid when
# use_attention_gate = True
config['SE_before_concat'] = False

config["n_base_filters"] = 16
config["network_depth"] = 5
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config["initial_learning_rate"] = 0.0005  # 0.001, 0.0005, 0.00025
config['gap_dropout_rate'] = None
config['loss_function'] ='categorical_crossentropy'

# Choose following from: None, regularizers.l1_l2(l1=1e-5, l2=1e-4), regularizers.l2(1e-5), regularizers.l1(1e-5)
config['regularizer'] = None
config['sqex'] = False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config["batch_size"] = 3
config["validation_batch_size"] = 3
config["n_epochs"] = 200  # cutoff the training after this many epochs
config["patience"] = 20  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 100  # training will be stopped after this many epochs without the validation loss improving
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Augmentation parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For volume data: specify one or more of [0,1,2] eg: [0], [0,1], [1,2], [0,1,2] etc
config["flip"] = [0, 1, 2]  # augments the data by randomly flipping an axis during training

config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion

# eg: { 'shift': {'mu': 0.0, 'std':0.25}, 'scale':{'mu': 1.0, 'std': 0.1} }
# Trial1: { 'shift': 'None', 'scale':{'mu': 1.0, 'std': 0.1} }
# config["shift_scale_intensity"] = { 'shift': None, 'scale':{'mu': 1.0, 'std': 0.1} }
config["shift_scale_intensity"] = None

config["augment"] = config["flip"] or config["distort"] or config["shift_scale_intensity"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = False  # if True, then patches without any target will be skipped
config["add_flipped_modality"] = False # If True, then every session is once added in normal form, and once by mirroring along vertical axis
config["smote"] = False # Synthetic minority oversampling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ File paths ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

config["overwrite"] = False  # If True, will overwrite previous files. If False, will use previously written files.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seq dropout parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

config["seq_dropout_fraction_tr"] = None  # Manual, None or [0,1]
# config["ids_drop_tr_gbm_ch1"] = [] # list(range(104,157))
# config["ids_drop_tr_gbm_ch2"] = [] # list(range(157,208))

config["seq_dropout_fraction_val"] = None  # Manual, None or [0,1]
# config["ids_drop_val_gbm_ch1"] = [] # list(range(26,39))
# config["ids_drop_val_gbm_ch2"] = [] # list(range(39,51))





# Split data into 5 folds from excel
def set_fold(fold, exp):
    config["fold"] = fold

    # Setting the basepath of the folder inside which everything will be stored: molecular/results/<experiment>/<fold>/
    config["basepath"] = "/scratch/satrajit/tumor_classification_experiments/Exp" + exp + "/" + "fold" + fold + "/"

    # Split data into n folds
    config['training_sessions'], config['validation_sessions'], config['testing_sessions'] = split_data_into_n_folds_from_path(config, info, 
                                                                                                                    data_dir = '_canonical_BRATS2019_HGG_WholeTumor')

    info['training_sessions'] = [os.path.basename(sess) for sess in config['training_sessions']]
    info['validation_sessions'] = [os.path.basename(sess) for sess in config['validation_sessions']]
    info['testing_sessions'] = [os.path.basename(sess) for sess in config['testing_sessions']]
    
    # ***************************************************************************************************************************

    # Path to which training/validation/test hdf5 files will be written to
    config["data_file_tr"] = os.path.abspath(config["basepath"] + "fold{}_data_tr.h5".format(fold))
    config["data_file_val"] = os.path.abspath(config["basepath"] + "fold{}_data_val.h5".format(fold))
    config["data_file_test"] = os.path.abspath(config["basepath"] + "fold{}_data_test.h5".format(fold))

    # Path to which pickle files containing training/validation/test indices will be written to
    config["training_file"] = os.path.abspath(config["basepath"] + "training_ids.pkl")
    config["validation_file"] = os.path.abspath(config["basepath"] + "validation_ids.pkl")
    config["testing_file"] = os.path.abspath(config["basepath"] + "testing_ids.pkl")

    config["model_file"] = os.path.abspath(config["basepath"] + "modelClassifier_ep{epoch:03d}_dice_{val_dice_coef_multilabel:.4f}_vloss{val_loss:.4f}.h5")
    
    # Path to which log file will be written
    config["log_file"] = os.path.abspath(config["basepath"] + "training_classification.log")

    # Path to which tensorboard log will be written
    config["tensorboard_log"] = os.path.abspath(config["basepath"] + "tensorboard_log")

    df = pd.read_excel(config['excel_path'])

    return df



