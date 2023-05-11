import os
import random
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold

from brats_nclass_testing_half.train_isensee2017 import trim_df_by_dropping_nans, trim_df_based_on_GT, filter_sessions_based_on_availability_in_scratch, \
    filter_sessions_based_on_availability_of_modalities, tumor_zoo_params_dict_list, split_data_into_n_folds, split_val_test, trim_df_based_on_Tumor_modality, check_or_create_tumor_modality, \
    split_data_into_n_folds_from_excel, trim_df_based_on_presence_in_scratch_and_modality, split_data_with_interal_testing_from_excel

random.seed(9001)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from unet3d.model.model_classification import *

config = dict()
info = dict()

# Set the molecular parameter for this experiment
config["tumor_type"] = [i['class_tumor'] for i in tumor_zoo_params_dict_list]

for i, tumor_dict in enumerate(tumor_zoo_params_dict_list):
    config[i] = tumor_dict

config['labels_to_use'] = config["tumor_type"]
config['marker_column'] = 'Tumor_type'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config['labels'] = (0, 1)
config['n_labels'] = len(config['labels'])
config['patch_shape'] = None
config['path_to_data'] = '/sample_data/tumor_classification_revamped/'

config['excel_path'] = "/tumor_classification_revamped/tumor_classification.xlsx"

config["all_modalities"] = ["T1c_subtrMeanDivStd"]
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
config["seg_classify"] = "c"
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
config["network_depth"] = 5
config['gap_dropout_rate'] = None
config['loss_function'] ='categorical_crossentropy'

# Choose following from: None, regularizers.l1_l2(l1=1e-5, l2=1e-4), regularizers.l2(1e-5), regularizers.l1(1e-5)
config['regularizer'] = regularizers.l2(1e-5)
config['sqex'] = False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
config["batch_size"] = 1
config["validation_batch_size"] = 5
config["n_epochs"] = 200  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
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
config["simulate_partial_acq_prob"] = 1
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
    config["basepath"] = "/tumor_classification_revamped/results/Exp" + exp + "/" + "fold" + fold + "/"

    # Read excel
    df = pd.read_excel(config['excel_path'])

    # Trim excel by removing entries containing Nans
    df = trim_df_by_dropping_nans(df, config)

    info['sessions_before_modality_omission'] = df['sessions'].tolist()

    # [Conditional] Trim cases based on various requirements
    df = trim_df_based_on_presence_in_scratch_and_modality(df, config)

    info['sessions_after_modality_omission'] = df['sessions'].tolist()
    info['sessions_omitted_in_modality_omission'] = list(set(info['sessions_before_modality_omission']).difference(set(info['sessions_after_modality_omission'])))

    df_after_dropping_nans = copy.deepcopy(df)



    # [Conditional] Trim cases based on requirement of OTMultiClass and T1c
    # df = trim_df_based_on_GT(df, config, exclude_cases_with_partial_GT = True)
    # df = trim_df_based_on_Tumor_modality(df, config)

    # Any additional conditions to filter data will come here
    df = df[~df.fold.str.contains("new")]
    # End

    # Trim cases based on availability of all modalities including GT
    sessions_abspath_exists = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in df.iterrows()]

    if 'Tumor_modality' in config["all_modalities"]:
        check_or_create_tumor_modality(sessions_abspath_exists)
    


    # Split data into n folds
    foldx_sessions = split_data_into_n_folds(config, info, df, sessions_abspath_exists)

    # Set val_test fold and split it into val + test
    foldx_sessions_val_test = split_val_test(config, info, df, foldx_sessions, fold)


    config['validation_sessions'], config['testing_sessions'] = foldx_sessions_val_test
    config['training_sessions'] = [i for sublist in foldx_sessions for i in sublist]

    info['training_sessions'] = [os.path.basename(sess) for sess in config['training_sessions']]
    y_train = df.loc[df.sessions.isin(info['training_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    y_train_unique, y_train_count = np.unique(y_train, return_counts=True)
    info['class_distribution_training'] = str(dict(zip(y_train_unique, y_train_count)))

    info['validation_sessions'] = [os.path.basename(sess) for sess in config['validation_sessions']]
    y_val = df.loc[df.sessions.isin(info['validation_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    y_val_unique, y_val_count = np.unique(y_val, return_counts=True)
    info['class_distribution_validation'] = str(dict(zip(y_val_unique, y_val_count)))

    info['testing_sessions'] = [os.path.basename(sess) for sess in config['testing_sessions']]
    y_test = df.loc[df.sessions.isin(info['testing_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    y_test_unique, y_test_count = np.unique(y_test, return_counts=True)
    info['class_distribution_testing'] = str(dict(zip(y_test_unique, y_test_count)))

    # # *************************************************** Set dataset for external testing *************************************************

    # # Any additional conditions to filter data will come here
    # df_ext = df_after_dropping_nans[(df_after_dropping_nans.fold.str.contains("new"))]
    # # End

    # info['ext_sessions_before_modality_omission'] = df_ext['sessions'].tolist()

    # # [Conditional] Trim cases based on requirement of OTMultiClass and T1c
    # # df_ext = trim_df_based_on_GT(df_ext, config)
    # # df_ext = trim_df_based_on_Tumor_modality(df_ext, config)

    # # Trim cases based on availability in scratch directory
    # sessions_abspath_exists_bef_modality_check = filter_sessions_based_on_availability_in_scratch(df_ext, config)


    # if 'Tumor_modality' in config["all_modalities"]:
    #     check_or_create_tumor_modality(sessions_abspath_exists_bef_modality_check)

    # # Trim cases based on availability of all modalities including GT
    # sessions_abspath_exists = filter_sessions_based_on_availability_of_modalities(sessions_abspath_exists_bef_modality_check, config)

    # info['ext_sessions_after_modality_omission'] = [os.path.basename(i) for i in sessions_abspath_exists]
    # info['ext_sessions_omitted_in_modality_omission'] = \
    #     list(set(info['ext_sessions_before_modality_omission']).difference(set(info['ext_sessions_after_modality_omission'])))


    # config['ext_sessions'] = sessions_abspath_exists

    # info['ext_sessions'] = [os.path.basename(sess) for sess in config['ext_sessions']]
    # y_test = df_after_dropping_nans.loc[df_after_dropping_nans.sessions.isin(info['ext_sessions'])][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
    # y_test_unique, y_test_count = np.unique(y_test, return_counts=True)
    # info['class_distribution_ext'] = str(dict(zip(y_test_unique, y_test_count)))

    
    # ***************************************************************************************************************************
    for i, tumor in enumerate(config["tumor_type"]):
        config[i]["data_file_tr"] = os.path.abspath(config["basepath"]+tumor+"_data_tr.h5")
        config[i]["data_file_val"] = os.path.abspath(config["basepath"]+tumor+"_data_val.h5")
        config[i]["training_file"] = os.path.abspath(config["basepath"]+tumor+"_training_ids.pkl")

        config[i]["validation_file"] = os.path.abspath(config["basepath"]+tumor+"_validation_ids.pkl")
        config[i]["data_file_test"] = os.path.abspath(config["basepath"]+tumor+"_data_test.h5")
        config[i]["testing_file"] = os.path.abspath(config["basepath"]+tumor+"_testing_ids.pkl")
        
    # Path to which external test hdf5 files will be written to
    config["data_file_ext"] = os.path.abspath(config["basepath"] + "Exp{}_fold{}_data_ext.h5".format(exp,fold))
    # Path to which pickle files containing external test indices will be written to
    config["ext_file"] = os.path.abspath(config["basepath"] + "ext_ids.pkl")

    # Path to which model file will be written to
    if config["seg_classify"] == 'c':
        config["model_file"] = os.path.abspath(config["basepath"] + "modelClassifier_ep{epoch:03d}_vacc{val_acc:.4f}_vloss{val_loss:.4f}.h5")
    elif config["seg_classify"] == 's':
        config["model_file"] = os.path.abspath(config["basepath"] + "modelClassifier_ep{epoch:03d}_dice_{val_dice_coefficient:.4f}_vloss{val_loss:.4f}.h5")
    elif config["seg_classify"] == 's_c':
        config["model_file"] = os.path.abspath(config["basepath"] + "modelClassifier_ep{epoch:03d}_vacc{val_clsfctn_op_acc:.4f}_vloss{val_clsfctn_op_loss:.4f}.h5")

    # Path to which log file will be written
    config["log_file"] = os.path.abspath(config["basepath"] + "training_classification.log")

    # Path to which tensorboard log will be written
    config["tensorboard_log"] = os.path.abspath(config["basepath"] + "tensorboard_log")

    return df_after_dropping_nans



