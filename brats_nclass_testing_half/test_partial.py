import copy
import os
import glob
import pathlib
import pprint
import random
import shutil
import pandas as pd
import pickle
import logging

from brats_nclass_testing_half.train_isensee2017 import trim_df_by_dropping_nans, trim_df_based_on_GT, trim_df_based_on_Tumor_modality, filter_sessions_based_on_availability_in_scratch, \
    check_or_create_tumor_modality, filter_sessions_based_on_availability_of_modalities, create_training_validation_testing_files
from brats_nclass_testing_half import evaluate, plot_results

from unet3d.prediction_roc_nclass_testing import run_validation_cases_classification
from unet3d.data import write_data_to_file, open_data_file
from unet3d.utils import pickle_dump, pickle_load

from sklearn.model_selection import StratifiedKFold

random.seed(9001)
import numpy as np

import nibabel as nib
import tables
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'sans-serif'}
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rc('font', **font)

import seaborn as sns
import pickle 

from unet3d.training import load_old_model

def get_pickle_lists(data_file, pickle_file_path):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file for training    
    :param pickle_file_path:
    """
    print("Creating pickle lists...")
    nb_samples_tr = data_file.root.data.shape[0] # Number of training data
    training_list = list(range(nb_samples_tr)) # List of integers: [0, 1, .. upto nb_samples_tr]            
    pickle_dump(training_list, pickle_file_path)


def main(fold, exp):
    config_file_name="config_file_Exp"+exp
    config_file = importlib.import_module('config_files.'+config_file_name)
    set_fold = config_file.set_fold
    global config
    config = config_file.config
    info = config_file.info
    _ = set_fold(fold, exp)

    # Read excel
    df = pd.read_csv(os.path.join(config["basepath"], 'df_filtered.csv'))


    # Create and configure logger
    log_path = os.path.join(config["basepath"], "training_log.txt")
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(filename=log_path,
                        filemode='a',
                        format=LOG_FORMAT,
                        level=logging.INFO)

    logger = logging.getLogger(__file__)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    logging.getLogger('matplotlib.font_manager').disabled = True

    logger.info("*************************************************************************************************")
    logger.info("*" * 40 + " [ PREDICTION - EXTERNAL TESTING CASES ] " + "*" * 40)
    logger.info("*************************************************************************************************")

    model_file = glob.glob(os.path.abspath(config["basepath"]+"modelClassifier_ep*.h5"))[0]
    logger.info("[INFO] Loading model from: {}".format(model_file))
    model = load_old_model(model_file, config['n_labels'])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EXTERNAL TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logger.info("\n" + "=" * 30 + " [TESTING PARTIAL FILES] " + "=" * 30)
    
    sessions = glob.glob('/scratch/satrajit/data/_partial_MIRRIR_METS_SINGLE_MULTIPLE/M*')
    # sessions = glob.glob('/scratch/satrajit/data/_partial_M19004_HB/*')
    data_file = "/scratch/satrajit/tumor_classification_experiments/Exp" + exp + "/" + "data_partial_METS.h5"

    ext_test_files, ext_labels, subject_ids_ext_test, drop_idx_ext = create_training_validation_testing_files(logger, config, df, path_to_sessions=sessions, manual_label = 'METS')


    if config["overwrite"] or not os.path.exists(data_file):
        print("\n","="*30,": [TESTING PARTIAL] write_data_to_file","="*30,"\n")
        write_data_to_file(config, ext_test_files,
                           data_file,
                           image_shape=config["image_shape"],
                           subject_ids=subject_ids_ext_test,
                           drop_idx=drop_idx_ext,
                           normalize=config['normalize_data_using_cohort_mean_and_std'],
                           add_flipped_modality = config["add_flipped_modality"])


    hdf5_file_list = [data_file]
    logger.info("hdf5_file_list: " + str(hdf5_file_list)) 

    run_validation_cases_classification(df, logger,
                                        config,
                                        model=model,
                                         training_modalities=config["training_modalities"],
                                         hdf5_file= hdf5_file_list,
                                         output_label_map=True,
                                         val_or_test="partial_METS",
                                         manual_label = 'METS')    

    plot_results.main(fold, exp, cohort_suffix = "partial_METS")
