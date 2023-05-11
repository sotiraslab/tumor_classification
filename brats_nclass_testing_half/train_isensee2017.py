import copy
import os
import glob
import pathlib
import pprint
import random
import shutil
import string

import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold

from brats_nclass_testing_half import inspect_data
from unet3d.utils.extract_coarse_GT import save_numpy_2_nifti
from unet3d.utils.utils import check_unique_elements

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

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator_classification_nclass import get_training_and_validation_generators_classification
from unet3d.model.model_classification import *
from unet3d.training import train_model_clsfctn, load_old_model
from unet3d.model.isensee2017 import isensee2017_seg_model

import logging

# **************************************************** Tumor ZOO **************************************************

# tumor_zoo_params_dict_list = {'HGG': {'labels': (0, 1, 2, 4)},
#                               'LGG': {'labels': (0, 1, 2, 4)},
#                               'METS': {'labels': (0, 1)},
#                               'PITADE': {'labels': (0, 1)},
#                               'ACSCHW': {'labels': (0, 1)},
#                               'HEALTHY': {'labels': (0,)},
#                               'MENINGIOMA': {'labels': (0, 1)}}

tumor_zoo_params_dict_list = [{'class_tumor': 'HGG',
                            'labels': (0, 1, 2, 4),
                            'n_labels': 4  },
                              {'class_tumor': 'LGG',
                            'labels': (0, 1, 2, 4),
                            'n_labels': 4  },
                              {'class_tumor': 'METS',
                              'labels': (0, 1),
                              'n_labels': 2  },
                              {'class_tumor': 'PITADE',
                              'labels': (0, 1),
                              'n_labels': 2  },
                              {'class_tumor': 'ACSCHW' ,
                              'labels': (0, 1),
                              'n_labels': 2},
                              {'class_tumor': 'HEALTHY',
                              'labels': (0,),
                              'n_labels': 1},
                              {'class_tumor': 'MENINGIOMA' ,
                              'labels': (0, 1),
                              'n_labels': 2}]

def plot_crosstabs(df, fold, marker_column, savepath):
    crosstab_vars = ["scratch_path", "Tumor_type"]
    fig, axes = plt.subplots(1,len(crosstab_vars), figsize = (10*len(crosstab_vars),10))

    for ax, variable in zip(axes.ravel(), crosstab_vars):
        sns.heatmap(pd.crosstab(df[variable], df[marker_column]), ax = ax, square = True, annot_kws={"size": 30}, cmap = 'YlOrBr', annot=True, cbar=False, fmt="d",
            linewidths=1, linecolor='black')
        ax.yaxis.label.set_visible(False)
        ax.set_title(variable, fontsize = '25')
        
        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(savepath + 'fold{}_info_crosstabs.png'.format(fold))
    plt.close()

def plot_info(info, fold, savepath):

    # barplot : # sessions before vs after modality omission
    ax = plt.subplot()
    xlabels = ['before', 'after']
    ylabels = [len(info['sessions_before_modality_omission']), len(info['sessions_after_modality_omission'])]
        
    ax.bar(xlabels,ylabels)

    rects = ax.patches
    #
    # # Make some labels.
    # labels = ["label%d" % i for i in range(len(rects))]

    for rect, label in zip(rects, ylabels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(savepath + 'fold{}_info_bef_after_mod_omit.png'.format(fold))
    plt.close()

    # # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    # def func(pct, allvals):
    #     absolute = int(pct/100.*np.sum(allvals))
    #     return "{:.1f}%\n({:d})".format(pct, absolute)

    # labels = ["fold" + str(i) for i in range(1,6)]
    # sizes = [len(fold) for fold in info['sessions_per_fold']]

    # fig1, ax1 = plt.subplots(figsize=(10, 10))
    # ax1.pie(sizes, labels=labels, shadow=True, autopct=lambda pct: func(pct, sizes), startangle=90, textprops=dict(size=40))
    # # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # plt.tight_layout()
    # plt.savefig(savepath + 'info_split_per_fold.png')
    # plt.close()

    # Stacked bar plot
    class_distribution_keys = [i for i in list(info.keys()) if 'class_distribution' in i]

    D_list = []
    class_keys = list(eval(info[class_distribution_keys[0]]).keys())

    # print(class_distribution_keys)

    for class_distribution_key in class_distribution_keys:
        D = eval(info[class_distribution_key])
        D_list.append(list(D.values()))

    C = np.array(D_list).T.tolist()

    # Source: https://stackoverflow.com/questions/41296313/stacked-bar-chart-with-centered-labels
    print(C)
    print(class_keys)
    df = pd.DataFrame(dict(zip(class_keys, C)))

    ax = df.plot(stacked=True, kind='bar', figsize=(12, 8), rot='horizontal')

    # .patches is everything inside of the chart
    for rect in ax.patches:
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        
        # The height of the bar is the data value and can be used as the label
        label_text = int(height)  # f'{height:.2f}' to format decimal values
           
        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2

        # plot only when height is greater than specified value
        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=20, color='w')
        
    ax.legend(loc='best', fontsize=30)
    ax.set_ylabel("Count", fontsize=20)
    ax.set_xlabel("Class", fontsize=20)

    xticklabelslist = [i.split('_')[-1] for i in class_distribution_keys]

    ax.set_xticklabels(xticklabelslist, fontsize=15)
    
    plt.tight_layout()
    plt.savefig(savepath + 'fold{}_info_class_dist_per_fold.png'.format(fold))
    plt.close()


def sequence_dropout(files, subject_ids, config, train0val1):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sequence dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    drop_fraction = config["seq_dropout_fraction_tr"] if train0val1 == 0 else config["seq_dropout_fraction_val"]
    number_of_sessions = len(files) # Let's say this is 50
    number_of_sessions_to_drop_sequence = int(number_of_sessions*drop_fraction) # this is 25
    
    list_of_session_ids = list(range(number_of_sessions)) # [0, 1, 2, ...49]
    list_of_session_ids_to_drop_sequence = random.sample(list_of_session_ids, number_of_sessions_to_drop_sequence) # randomly chosen 25 elements from list_of_session_ids
    list_of_session_ids_to_drop_ch1 = random.sample(list_of_session_ids_to_drop_sequence, number_of_sessions_to_drop_sequence//2) # from list_of_session_ids_to_drop_sequence, choose half of it to drop channel 1
    list_of_session_ids_to_drop_ch2 = list(set(list_of_session_ids_to_drop_sequence).difference(list_of_session_ids_to_drop_ch1)) # ... and then choose the rest to drop channel 2

    return (list_of_session_ids_to_drop_ch1, list_of_session_ids_to_drop_ch2)

def create_training_validation_testing_files(logger, config, df, path_to_sessions, manual_label = None):
    training_files = list()
    subject_ids_tr = list()

    for subject_dir in path_to_sessions:
        subject_ids_tr.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["training_modalities"] + config["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_files.append(tuple(subject_files))

    training_files = [list(i) for i in training_files] # converting to list of lists from list of tuples

    logger.info("[SUBJECT_IDS] " + str(len(subject_ids_tr)) + " " + str(subject_ids_tr))

    if not manual_label:
        session_labels = np.array([df[df['sessions'] == i]['Tumor_type'].iloc[0] for i in subject_ids_tr])
    else:
        session_labels = [manual_label] * len(subject_ids_tr)

    assert len(session_labels) == len(subject_ids_tr)


    # Currently we are not dropping any modalities
    drop_idx = ([],[])

    # if config["seq_dropout_fraction_tr"] is "Manual":
    #     drop_idx_tr = (config["ids_drop_tr_gbm_ch1"],config["ids_drop_tr_gbm_ch2"])
    # elif config["seq_dropout_fraction_tr"] is None:
    #     drop_idx_tr = ([],[])
    # else:
    #     drop_idx_tr = sequence_dropout(training_files, subject_ids_tr, 0)
    #
    # if len(drop_idx_tr[0])+len(drop_idx_tr[1]) > 0:
    #     print("Total number of sessions to drop sequence", len(drop_idx_tr[0])+len(drop_idx_tr[1]))
    #     print("Channel1 will be dropped from following GBM sessions: ")
    #     for idx in drop_idx_tr[0]:
    #         print(subject_ids_tr[idx])
    #
    #     print("Channel2 will be dropped from following GBM sessions: ")
    #     for idx in drop_idx_tr[1]:
    #         print(subject_ids_tr[idx])

    return training_files, session_labels, subject_ids_tr, drop_idx

def trim_df_by_dropping_nans(df, config):
    # Prepare the dataframe by reading from excel file
    

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    df = df[df.scratch_path.notna()]  # There are some cases for which we do not have imaging data (i.e. NA in sessions column) - drop them

    return df

def trim_df_based_on_GT(df, config, exclude_cases_with_partial_GT = True):

    if "OTMultiClass" in config["all_modalities"]:
        if exclude_cases_with_partial_GT:
            # keep a check that if "OTMulticlass" is included then only take cases with GT, filter from excel
            df = df[df['seg_groundtruth'] == 'Yes']
        else:
            df = df[df['seg_groundtruth'].isin(['Yes', 'partial'])]

    if "bbox_crop_or_mask" not in config:
        config["bbox_crop_or_mask"] = None

    if config["bbox_crop_or_mask"] is not None:
        df = df[df['seg_groundtruth'] == 'Yes']

    return df

def trim_df_based_on_Tumor_modality(df, config):

    # keep a check that if "Tumor_modality" is included then only take cases with GT & T1c, filter from excel
    if "Tumor_modality" in config["all_modalities"]:
        df = df[(df['seg_groundtruth'] == 'Yes') & (df['T1c'] == 'Yes')]

    return df

def check_or_create_tumor_modality(sessions):
    print("[DEBUG] inside check_or_create_tumor_modality")
    # print(*sessions, sep="\n")
    for session in sessions:
        print(session)
        sess_list_of_modalities = glob.glob(os.path.join(session, "*.nii.gz"))
        # print("sess_list_of_modalities",sess_list_of_modalities)
        is_tumor_modality_boolean_list = ["Tumor_modality" in s for s in sess_list_of_modalities]

        if any(is_tumor_modality_boolean_list):
            print("Tumor modality already exists")
        else:
            print("[DISCLAIMER] Tumor_modality did not exist. Creating it by masking OTMulticlass on T1c_subtrMeanDivStd.nii.gz of this subject: ")

            brainmask = np.rint(nib.load(os.path.join(session, 'brainmask.nii.gz')).get_fdata())
            print("{BRAINMASK}", check_unique_elements(brainmask))

            t1c = nib.load(os.path.join(session, 'T1c_subtrMeanDivStd.nii.gz')).get_fdata()
            print("{t1c}", np.count_nonzero(t1c))

            GT = nib.load(os.path.join(session, 'OTMultiClass.nii.gz')).get_fdata().astype('int32')
            print("{GT}", check_unique_elements(GT))
            print("{GT}", np.count_nonzero(GT))

            GT[GT > 0] = 1
            pseudo = GT * t1c
            # pseudo = pseudo * brainmask
            print("{pseudo}", pseudo[0,0,0])

            save_numpy_2_nifti(pseudo,os.path.join(session, 'T1c_subtrMeanDivStd.nii.gz'),os.path.join(session, 'Tumor_modality.nii.gz'))




def filter_sessions_based_on_availability_in_scratch(df, config):
    # List of all sessions of the worksheet : ['abspath/to/session1', 'abspath/to/session2', ..., 'abspath/to/sessionn']
    sessions_abspath_all = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in df.iterrows()]

    # True/False if session folder exists/doesnt in datapath : [True, False, ..., True]
    session_exists_logical = [os.path.isdir(i) for i in sessions_abspath_all]

    # Subset of sessions_abspath_all, containing only those sessions that exist
    sessions_abspath_exists_bef_modality_check = np.array(sessions_abspath_all)[np.array(session_exists_logical)].tolist()

    return sessions_abspath_exists_bef_modality_check

def trim_df_based_on_presence_in_scratch_and_modality(df, config):
    
    all_modalities = copy.deepcopy(config["all_modalities"])
    if "Tumor_modality" not in all_modalities:
        pass
    else:
        all_modalities.remove("Tumor_modality")

    # List of all sessions of the worksheet : ['abspath/to/session1', 'abspath/to/session2', ..., 'abspath/to/sessionn']
    sessions_abspath_all = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in df.iterrows()]

    # True/False if session folder exists/doesnt in datapath : [True, False, ..., True]
    session_exists_logical = [os.path.isdir(i) for i in sessions_abspath_all]

    # Subset of sessions_abspath_all, containing only those sessions that exist
    sessions_abspath_exists_bef_modality_check = np.array(sessions_abspath_all)[np.array(session_exists_logical)].tolist()


    session_exists_modality_exists_logical_sublist = [[os.path.exists(os.path.join(i, j + ".nii.gz")) for j in all_modalities] for i in sessions_abspath_exists_bef_modality_check]

    # For each session, this gives True if all req modalities exist for that session
    session_exists_modality_exists_logical = [all(i) for i in session_exists_modality_exists_logical_sublist]

    # Use session_exists_modality_exists_logical indices to filter sessions_abspath_exists_bef_modality_check
    # This is the final list of sessions to be used
    sessions_abspath_exists = np.array(sessions_abspath_exists_bef_modality_check)[np.array(session_exists_modality_exists_logical)].tolist()

    df = df[df['sessions'].isin(os.path.basename(i) for i in (sessions_abspath_exists))]

    return df

def filter_sessions_based_on_availability_of_modalities(sessions_abspath_exists_bef_modality_check, config):
    # Further filter based on if session contains all required modalities
    # This list contains sublists containing True/false for each modality for each session
    all_modalities = copy.deepcopy(config["all_modalities"])
    if "Tumor_modality" not in all_modalities:
        pass
    else:
        all_modalities.remove("Tumor_modality")

    session_exists_modality_exists_logical_sublist = [[os.path.exists(os.path.join(i, j + ".nii.gz")) for j in all_modalities] for i in sessions_abspath_exists_bef_modality_check]

    # For each session, this gives True if all req modalities exist for that session
    session_exists_modality_exists_logical = [all(i) for i in session_exists_modality_exists_logical_sublist]

    # Use session_exists_modality_exists_logical indices to filter sessions_abspath_exists_bef_modality_check
    # This is the final list of sessions to be used
    sessions_abspath_exists = np.array(sessions_abspath_exists_bef_modality_check)[np.array(session_exists_modality_exists_logical)].tolist()

    return sessions_abspath_exists

def split_data_into_n_folds(config, info, df, sessions_abspath_exists, n_fold=5):
    sessions_abspath_exists_basename = [os.path.basename(i) for i in sessions_abspath_exists]
    y = df.loc[df.sessions.isin(sessions_abspath_exists_basename)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions

    y_unique, y_count = np.unique(y, return_counts=True)
    info['class_distribution_overall'] = str(dict(zip(y_unique, y_count)))

    # Stratified k-fold sampling
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)  # define sampler

    train_index_list_5folds = []
    val_test_index_list_5folds = []

    for train_index, val_test_index in skf.split(sessions_abspath_exists, y):
        train_index_list_5folds.append(train_index)
        val_test_index_list_5folds.append(val_test_index)

    foldx_sessions = [np.array(sessions_abspath_exists)[idx].tolist() for idx in val_test_index_list_5folds]

    info['sessions_per_fold'] = [[os.path.basename(i) for i in j] for j in foldx_sessions]

    for i, foldi in zip(["fold" + str(i) for i in range(1, n_fold + 1)], foldx_sessions):
        sessname = [os.path.basename(sess) for sess in foldi]
        y_fold = df.loc[df.sessions.isin(sessname)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions
        y_fold_unique, y_fold_count = np.unique(y_fold, return_counts=True)
        info['class_distribution_' + i] = str(dict(zip(y_fold_unique, y_fold_count)))

    return foldx_sessions


def split_data_into_n_folds_from_excel(config, info, df, sessions_abspath_exists):
    sessions_abspath_exists_basename = [os.path.basename(i) for i in sessions_abspath_exists]
    y = df.loc[df.sessions.isin(sessions_abspath_exists_basename)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions

    y_unique, y_count = np.unique(y, return_counts=True)
    info['class_distribution_overall'] = str(dict(zip(y_unique, y_count)))

    training_df = df[~df['fold'].str.contains(config['fold'])]
    val_df = df[df['fold'].str.contains(config['fold'] + '_val')]
    test_df = df[df['fold'].str.contains(config['fold'] + '_test')]

    training_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in training_df.iterrows()]
    val_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in val_df.iterrows()]
    test_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in test_df.iterrows()]

    return training_session_abs_path, val_session_abs_path, test_session_abs_path


def split_data_with_interal_testing_from_excel(config, info, df, sessions_abspath_exists):
    sessions_abspath_exists_basename = [os.path.basename(i) for i in sessions_abspath_exists]
    y = df.loc[df.sessions.isin(sessions_abspath_exists_basename)][config['marker_column']].tolist()  # list of all molecular status corresponding to sessions

    y_unique, y_count = np.unique(y, return_counts=True)
    info['class_distribution_overall'] = str(dict(zip(y_unique, y_count)))

    training_df = df[~df['fold'].str.contains(config['fold']) & ~df['fold'].str.contains('_test')]
    val_df = df[df['fold'].str.contains(config['fold'] + '_val')]
    test_df = df[df['fold'].str.contains('_test')]

    training_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in training_df.iterrows()]
    val_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in val_df.iterrows()]
    test_session_abs_path = [os.path.abspath(os.path.join(config["path_to_data"], row['scratch_path'], row['sessions'])) for index, row in test_df.iterrows()]

    return training_session_abs_path, val_session_abs_path, test_session_abs_path

def split_data_into_n_folds_from_path(config, info, data_dir): 

    val_fold = "fold" + config['fold'] + "/"
    train_fold = "fold[!" + config['fold'] + "]/"

    data_dir_abspath = os.path.join(config["path_to_data"], data_dir)

    training_session_abs_path = glob.glob(os.path.join(data_dir_abspath, train_fold, "*"))
    val_session_abs_path = glob.glob(os.path.join(data_dir_abspath, val_fold, "*"))
    test_session_abs_path = glob.glob(os.path.join(data_dir_abspath, val_fold, "*"))

    return training_session_abs_path, val_session_abs_path, test_session_abs_path

def split_val_test(config, info, df, foldx_sessions, fold):
    config['validation_testing_sessions'] = foldx_sessions.pop(int(fold) - 1)
    info['validation_testing_sessions'] = [os.path.basename(sess) for sess in config['validation_testing_sessions']]

    y_val_test = df.loc[df.sessions.isin(info['validation_testing_sessions'])][config['marker_column']].tolist()

    val_index_list_5folds = []
    test_index_list_5folds = []

    # sampler for splitting the validation data into validation and testing
    skf_val_test = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for val_index, test_index in skf_val_test.split(config['validation_testing_sessions'], y_val_test):
        val_index_list_5folds.append(val_index)
        test_index_list_5folds.append(test_index)

    foldx_sessions_val_test = [np.array(config['validation_testing_sessions'])[idx].tolist() for idx in test_index_list_5folds]

    return foldx_sessions_val_test

def main(fold, exp, debugmode):
    config_file_name="config_file_Exp"+exp

    # The file gets executed upon import, as expected.
    config_file = importlib.import_module('config_files.'+config_file_name)

    # Then you can use the module like normal
    set_fold = config_file.set_fold
    config = config_file.config
    info = config_file.info

    overwrite=config["overwrite"]   
    df = set_fold(fold, exp)


    # Had to put this check as certain keys are introduced in later experiments and this is
    # required to support old experiments    

    if "crop_to_foreground" not in config:
        config["crop_to_foreground"] = False
    if "normalize_data_using_cohort_mean_and_std" not in config:
        config['normalize_data_using_cohort_mean_and_std'] = True
    if "segm_output_activation" not in config:
        config['segm_output_activation'] = 'sigmoid'
    if "simulate_partial_acq_prob" not in config:
        config["simulate_partial_acq_prob"] = None

    if "loss_weights" not in config:
        config['loss_weights'] = 'None'

    # Create the basepath folder if it does not already exist
    if not os.path.exists(config["basepath"]):
        pathlib.Path(config["basepath"]).mkdir(parents=True, exist_ok=True)

    # # todo: fix issue: ValueError: The number of FixedLocator locations (7), usually from a call to set_ticks, does not match the number of ticklabels (10). at 
    # # ax.set_xticklabels(xticklabelslist, fontsize=15)
    # if debugmode is None:
    # plot_info(info, fold, config["basepath"])

    df.to_csv(os.path.join(config["basepath"], 'df_filtered.csv'))


    plot_crosstabs(df, fold, config['marker_column'], config["basepath"])
    

    with open(os.path.join(config["basepath"], 'fold{}_info.pickle'.format(fold)), 'wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # snapshot current code
    if not os.path.exists(os.path.join(config["basepath"],'code_snapshot/')):
        pathlib.Path(os.path.join(config["basepath"],'code_snapshot/')).mkdir(parents=True, exist_ok=True)

    shutil.copy2(__file__, os.path.abspath(os.path.join(config["basepath"],'code_snapshot')))
    shutil.copy2(os.path.join(Path(os.path.dirname(__file__)).parent, 'config_files', config_file_name + ".py"), os.path.abspath(os.path.join(config["basepath"],'code_snapshot')))

    # Create and configure logger
    log_path = os.path.join(config["basepath"], "training_log.txt")
    LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format=LOG_FORMAT,
                        level=logging.DEBUG)

    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)


    logger.info("***************************************************************************************************************")
    logger.info("*"*50 + " [ EXPERIMENT #{} ]".format(exp) + " [ FOLD #{} ]".format(fold) + "*"*50)
    logger.info("***************************************************************************************************************")

    logger.info("~" * 60 + " [CONFIG] " + "~" * 60)

    for line in pprint.pformat(config).split('\n'):
        logger.info(line)

    logger.info("~" * 60 + " [INFO] " + "~" * 60)

    for line in pprint.pformat(info).split('\n'):
        logger.info(line)


    try: 
        logger.info("\n" +"[INFO] Total #sessions after first round of pruning df: " + str(len(info['sessions_before_modality_omission']))) 
    except: pass

    try: 
        logger.info("[INFO] Total #sessions used: " + str(len(info['sessions_after_modality_omission']))) 
    except: pass

    try: 
        logger.info("[INFO] Overall class distribution: " + str(info['class_distribution_overall'])) 
    except: pass

    try: 
        logger.info("[INFO] Total #sessions (training): "+ str(len(info['training_sessions']))) 
    except: pass

    try: 
        logger.info("[INFO] Overall class distribution (training): "+ str(info['class_distribution_training'])) 
    except: pass

    try: 
        logger.info("[INFO] Total #sessions (validation): "+ str(len(info['validation_sessions']))) 
    except: pass

    try: 
        logger.info("[INFO] Overall class distribution (validation): "+ str(info['class_distribution_validation'])) 
    except: pass

    try: 
        logger.info("[INFO] Total #sessions (testing): "+ str(len(info['testing_sessions']))) 
    except: pass

    try: 
        logger.info("[INFO] Overall class distribution (testing): "+ str(info['class_distribution_testing'])) 
    except: pass

    try: 
        logger.info("[INFO] Total #sessions (external): "+ str(len(info['ext_sessions']))) 
    except: pass

    try: 
        logger.info("[INFO] Overall class distribution (external): "+ str(info['class_distribution_ext'])) 
    except: pass

    
    # convert input images into an hdf5 file
    data_file_opened_tr_list = list()
    data_file_opened_val_list = list()


    logger.info("\n" + "=" * 30 + " [TRAINING FILES] " + "=" * 30)
    training_files, training_labels, subject_ids_tr, drop_idx_tr = create_training_validation_testing_files(logger, config, df, path_to_sessions=config["training_sessions"])

    logger.info("\n" + "=" * 30 + " [VALIDATION FILES] " + "=" * 30)
    validation_files, validation_labels, subject_ids_val, drop_idx_val = create_training_validation_testing_files(logger, config, df, path_to_sessions=config["validation_sessions"])

    logger.info("\n" + "=" * 30 + " [TEST FILES] " + "=" * 30)
    test_files, test_labels, subject_ids_test, _ = create_training_validation_testing_files(logger, config, df, path_to_sessions=config["testing_sessions"])

    for i, tumor in enumerate(config["tumor_type"]): 

        logger.info("\n" + "~" * 60 + "  " + tumor + "  " + "~" * 60)
        

        if overwrite or not os.path.exists(config[i]["data_file_tr"]):
            print("\n","="*30,tumor,": [TRAINING] write_data_to_file","="*30,"\n")
            write_data_to_file(config, np.array(training_files)[training_labels == tumor].tolist(),
                               config[i]["data_file_tr"],
                               image_shape=config["image_shape"],
                               subject_ids=np.array(subject_ids_tr)[training_labels == tumor].tolist(),
                               drop_idx=drop_idx_tr,
                               normalize=config['normalize_data_using_cohort_mean_and_std'],
                               add_flipped_modality = config["add_flipped_modality"] )

        if overwrite or not os.path.exists(config[i]["data_file_val"]):
            print("\n","="*30,tumor,": [VALIDATION] write_data_to_file","="*30,"\n")
            write_data_to_file(config, np.array(validation_files)[validation_labels == tumor].tolist(),
                               config[i]["data_file_val"],
                               image_shape=config["image_shape"],
                               subject_ids=np.array(subject_ids_val)[validation_labels == tumor].tolist(),
                               drop_idx=drop_idx_val,
                               normalize=config['normalize_data_using_cohort_mean_and_std'],
                               add_flipped_modality = config["add_flipped_modality"] )

        if overwrite or not os.path.exists(config[i]["data_file_test"]):
            print("\n","="*30,tumor,": [TESTING] write_data_to_file","="*30,"\n")
            write_data_to_file(config, np.array(test_files)[test_labels == tumor].tolist(),
                               config[i]["data_file_test"],
                               image_shape=config["image_shape"],
                               subject_ids=np.array(subject_ids_test)[test_labels == tumor].tolist(),
                               normalize=config['normalize_data_using_cohort_mean_and_std'],
                               add_flipped_modality = config["add_flipped_modality"] )

        # # Inspect data
        # if overwrite or not os.path.exists(os.path.join(config['basepath'], 'inspect_data')):
        #     inspect_data.main(fold,exp)
        # else: pass

        data_file_opened_tr = open_data_file(config[i]["data_file_tr"])
        data_file_opened_val = open_data_file(config[i]["data_file_val"])
        data_file_opened_test = open_data_file(config[i]["data_file_test"])

        logger.info(tumor + ": Number of training sessions: " + str(data_file_opened_tr.root.data.shape))
        logger.info(tumor + ": Number of validation sessions: " + str(data_file_opened_val.root.data.shape))
        logger.info(tumor + ": Number of testing sessions: " + str(data_file_opened_test.root.data.shape))

        data_file_opened_tr_list.append(data_file_opened_tr)
        data_file_opened_val_list.append(data_file_opened_val)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Calling Generators ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # get single training and testing generators for all classes
    # get_training_and_validation_generators_detection
    # get_training_and_validation_generators_classification
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators_classification(
        df, logger,
        config,
        data_file_opened_tr_list,
        data_file_opened_val_list,
        
        training_keys_file=[config[i]["training_file"] for i in range(len(config["tumor_type"]))],
        validation_keys_file=[config[i]["validation_file"] for i in range(len(config["tumor_type"]))],
        n_labels=[config[i]["n_labels"] for i in range(len(config["tumor_type"]))],
        labels=[config[i]["labels"] for i in range(len(config["tumor_type"]))],

        batch_size=config["batch_size"],
        overwrite=overwrite,
        
        patch_shape=None,
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        permute=config["permute"],
        simulate_partial_acq_prob = config["simulate_partial_acq_prob"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"],
        augment_intensity_shift_scale_prms = config["shift_scale_intensity"])

    if debugmode is not None:
        # Sanity of training generator
        for i in range(n_train_steps):
            # print("***************",i)
            x,y = next(train_generator) # yields one batch of data
            # print("[DEBUG] x.shape = ",x.shape)
            # if config["seg_classify"] == 'c':
            #     print("[DEBUG] y_classification -->", *y)
            # elif config["seg_classify"] == 's':
            #     print("[DEBUG] y_segmentation", y.shape)
            # elif config["seg_classify"] == 's_c':
            #     print("[DEBUG] y_classification", *y['clsfctn_op'])
            #     print("[DEBUG] y_segmentation", y['segm_op'].shape)

            # random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            # print(random_string)
            # data = x[0]
            # # truth = y['segm_op'][0]
            # nib.Nifti1Image(np.rollaxis(data, 0, 4), None).to_filename('./temp/{}_after_aug_data.nii.gz'.format(random_string))
            # nib.Nifti1Image(np.rollaxis(truth, 0, 4), None).to_filename('./temp/{}_after_aug_truth.nii.gz'.format(random_string))


        # # Sanity of validation generator
        # for i in range(4):
        #     print("***************", i)
        #     x, y = next(validation_generator)  # yields one batch of data
        #     print("[DEBUG] x.shape = ", x.shape)
        #     print("[DEBUG] y -->", *y)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute classweights ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if config["seg_classify"] == 'c' or config["seg_classify"] == 's_c':
    #     from sklearn.utils import class_weight
    #
    #     subject_ids = [i.decode('utf-8') for i in data_file_opened_tr.root.subject_ids]
    #     y_train = df.loc[df['sessions'].isin(subject_ids)][config['marker_column']].tolist()
    #     unique, counts = np.unique(y_train, return_counts=True)
    #     logger.info("[INFO] [CLASS_DISTRIBUTION_TRAINING]" + str(dict(zip(unique,counts))))
    #
    #     class_weights_clsfctn = class_weight.compute_class_weight('balanced', config['labels_to_use'], y_train)
    #
    #     final_class_weights = class_weights_clsfctn
    #
    #     nb_classes = len(list(set(y_train)))
    #
    #     logger.info("\n" + "[CLASS_WEIGHTS] " + str(dict(zip(config['labels_to_use'], final_class_weights))))
    #
    # else:
    #
    #     final_class_weights = None
    #     nb_classes = 2 # dummy (not used because this loop is for only segmentation)

    from sklearn.utils import class_weight

    # # For generic classification of all classes
    y_train_per_type = [[idx]*data_file_opened_tr.root.data.shape[0] for idx, data_file_opened_tr in enumerate(data_file_opened_tr_list)]
    y_train_per_type_collapsed = [i for sublist in y_train_per_type for i in sublist]
    #
    final_class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_per_type_collapsed), y_train_per_type_collapsed)

    logger.info("\n" + "[CLASS_WEIGHTS] " + str(dict(zip(config['tumor_type'], final_class_weights))))

    nb_classes = len(list(set(y_train_per_type_collapsed)))


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Compile model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if not overwrite and len(glob.glob(os.path.abspath(config["basepath"] + "modelClassifier*.h5"))) > 0:

        # Find last epoch till which model was trained
        modelnames = glob.glob(os.path.abspath(config["basepath"] + "modelClassifier*.h5"))
        epochs_done = [int(i.split('/')[-1].split('_')[1][2:]) for i in modelnames]
        latest_epoch = sorted(epochs_done)[-1]
        latest_epoch_formatted = "ep{:03d}".format(latest_epoch)
        start_from_epoch = latest_epoch
        print("[INFO] Latest epoch:", latest_epoch)

        Path(config["basepath"] + "temp_models/").mkdir(exist_ok=True)

        for m in modelnames:
            # print(m)
            # print(os.path.basename(m))
            if latest_epoch_formatted in m:
                latestModel = m
            else:
                pass
                # os.rename(m, config["basepath"] + "temp_models/" + os.path.basename(m))

        logger.info("[MODEL] Loading existing model from {} and resuming training..".format(latestModel))

        model_clsfctn = load_old_model(latestModel, config['n_labels'])

        # Accordingly make changes in the training log. Only keep lines till 'latest_epoch' and restart from there
        training_df = pd.read_csv(glob.glob(os.path.abspath(config["basepath"] + "training_classification.log"))[0]).set_index('epoch')
        training_df.truncate(before=0, after=start_from_epoch-1).to_csv(os.path.abspath(config["basepath"] + "training_classification.log"))

    else:
        # instantiate new model
        # Had to put this check as config["model"] is a newly introduced parameter and is not defined in previous config files
        try:
            modelname = config["model"]
        except KeyError:
            modelname = (isensee2017_classification_segmentation,)

        func = modelname[0]

        logger.info("[MODEL] Creating new model using {}".format(str(func)))

        # Model (i)
        if func == isensee2017_classification_segmentation:

            model_s_c, model_s, model_c = func(input_shape = config["input_shape"],
                                                nb_classes = nb_classes,
                                                depth = config["network_depth"],
                                                n_base_filters=config["n_base_filters"],
                                                gap_dropout_rate=config['gap_dropout_rate'],
                                                n_labels = config['n_labels'],
                                                initial_learning_rate=config["initial_learning_rate"],
                                                loss_function_seg = config['loss_function_seg'],
                                                loss_function_clsfctn = config['loss_function'],
                                                loss_weights = config['loss_weights'],
                                                regularizer=config['regularizer'])
            if config["seg_classify"] == 'c':
                model_clsfctn = model_c
            elif config["seg_classify"] == 's':
                model_clsfctn = model_s
            elif config["seg_classify"] == 's_c':
                model_clsfctn = model_s_c
            else: raise

        else:
            model_clsfctn = func(input_shape=config["input_shape"], n_labels = config['n_labels'])

        start_from_epoch = 0

    # if debugmode is not None:

    # Source: https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    model_clsfctn.summary(print_fn=lambda x: logger.info(x), line_length=150)

    if config['regularizer']:

        logger.info("[REGULARIZER] Model has regularizer applied in following layers:")

        for i in range(len(model_clsfctn.layers)):
            layer = model_clsfctn.layers[i]
            if 'kernel_regularizer' in layer.__dict__.keys():
                logger.info(str(i) + " " + str(layer.name) + " " + str(layer.output.shape) + " " +  str(layer.name) + " " +  str(layer.__dict__["kernel_regularizer"].__dict__))

    if debugmode is None:

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Starting training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if "seg_classify" in config:
            seg_classify_mode = config['seg_classify']
        else:
            seg_classify_mode = 'c'

        train_model_clsfctn(config, logger,
                            model=model_clsfctn,
                            model_file=(config["model_file"], config["model_file"]),
                            log_file=config["log_file"],
                            tensorboard_log = config["tensorboard_log"],
                            model_save_path = config["basepath"],

                            training_generator=train_generator,
                            validation_generator=validation_generator,
                            training_steps_per_epoch=n_train_steps,
                            validation_steps_per_epoch=n_validation_steps,

                            initial_learning_rate=config["initial_learning_rate"],
                            learning_rate_drop=config["learning_rate_drop"],
                            learning_rate_patience=config["patience"],
                            early_stopping_patience=config["early_stop"],
                            n_epochs=config["n_epochs"],
                            classweights=final_class_weights,
                            start_from_epoch=start_from_epoch,
                            seg_classify_mode = seg_classify_mode)

    for data_file in data_file_opened_tr_list:
        data_file.close()

    for data_file_val in data_file_opened_val_list:
        data_file_val.close()



