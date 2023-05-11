import glob
import logging
import os
import random

import pandas as pd

from unet3d.data import open_data_file
from unet3d.prediction_roc_nclass_testing import run_validation_cases_classification
from unet3d.utils import pickle_dump
from unet3d.training import load_old_model
from brats_nclass_testing_half import evaluate, plot_results

random.seed(9001)

import importlib
import matplotlib


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
    set_fold(fold, exp)

    # Read excel
    df = pd.read_csv(os.path.join(config['basepath'], 'df_filtered.csv'))

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
    logger.info("*" * 40 + " [ PREDICTION - VALIDATION + TESTING CASES ] " + "*" * 40)
    logger.info("*************************************************************************************************")

        
    model_file = glob.glob(os.path.abspath(config["basepath"]+"modelClassifier_ep*.h5"))[0]
    logger.info("[INFO] Loading model from: {}".format(model_file))
    model = load_old_model(model_file, config['n_labels'])


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VALIDATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    hdf5_file_list = [config[i]["data_file_val"] for i in range(len(config["tumor_type"]))]

    logger.info("hdf5_file_list: " + str(hdf5_file_list))

    run_validation_cases_classification(df, logger,
                                        config,
                                         model=model,
                                         training_modalities=config["training_modalities"],
                                         hdf5_file= hdf5_file_list,
                                         output_label_map=True,
                                         val_or_test="val")    


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    hdf5_file_list = [config[i]["data_file_test"] for i in range(len(config["tumor_type"]))]

    logger.info("hdf5_file_list: " + str(hdf5_file_list)) 


    run_validation_cases_classification(df, logger,
                                        config,
                                         model=model,
                                         training_modalities=config["training_modalities"],
                                         hdf5_file= hdf5_file_list,
                                         output_label_map=True,
                                         val_or_test="test")    

    if config["seg_classify"] == 'c':

        df_val = pd.read_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_val.csv", index_col=0)
        df_test = pd.read_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_test.csv", index_col=0)

        df_val_test = pd.concat([df_val, df_test])
        df_val_test.to_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_val_test.csv")

        plot_results.main(fold, exp)


    elif config["seg_classify"] == 's' or config["seg_classify"] == 's_c':
        evaluate.main(fold, exp)

