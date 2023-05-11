import os

import nibabel as nib
import numpy as np
import tables
import keras 
import pandas as pd

# For plotting ROC
from scipy import interp
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'sans-serif'}
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rc('font', **font)

from itertools import cycle
from sklearn.metrics import roc_curve, auc

# For plotting Precision-Recall
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For Confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from .training import load_old_model
from .utils import pickle_load
from .utils.utils import check_unique_elements
from .utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from .augment import permute_data, generate_permutation_keys, reverse_permute_data

def patch_wise_prediction(model, data, overlap=0, batch_size=1, permute=False):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    predictions = list()
    indices = compute_patch_indices(data.shape[-3:], patch_size=patch_shape, overlap=overlap)
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data[0], patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = predict(model, np.asarray(batch), permute=permute)
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(model, open_data_file.root.data[index], open_data_file.root.affine)


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return nib.Nifti1Image(data, affine)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images


def run_validation_case_classification(df, logger, val_or_test, df_comps, truth_list, pred_list, data_index, output_dir, model, data_file, training_modalities,
                        output_label_map=False, threshold=0.5, labels = None, overlap=16, permute=False, manual_label = None ):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is 
    considered a positive result and will be assigned a label.  
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
        
    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    session_name = data_file.root.subject_ids[data_index].decode('utf-8')
    
    # print(df)

    try:
        truth_clsfctn = df.loc[df['sessions'] == session_name][config['marker_column']].iloc[0]
    except:
        truth_clsfctn = manual_label
    # print("[DEBUG] truth_clsfctn", truth_clsfctn)

    class_label = config['labels_to_use'].index(truth_clsfctn)
    test_truth = keras.utils.to_categorical([class_label], num_classes=len(config['labels_to_use']))
    truth_list.append(test_truth)
        
    prediction = model.predict(test_data, verbose=0)

    # This condition is for classify+seg models when there are two predictions, assuming that the first prediction is the classification output
    if len(prediction) == 2:
        prediction_seg = prediction[1]
        prediction = prediction[0]
        # print("[DEBUG] prediction_seg.shape", prediction_seg.shape)
        # print("UNIQUE:", check_unique_elements(np.rint(prediction_seg)))

        # The following code is for saving predicted segmentations in case of classification + segmentation models

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        affine = data_file.root.affine[data_index]
        test_data = np.asarray([data_file.root.data[data_index]])

        for i, modality in enumerate(training_modalities):
            image = nib.Nifti1Image(test_data[0, i], affine)
            image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

        test_truth_seg = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
        test_truth_seg.to_filename(os.path.join(output_dir, "truth.nii.gz"))

        prediction_image = prediction_to_image(prediction_seg, affine, label_map=False, threshold=threshold, labels=labels)

        if isinstance(prediction_image, list):
            for i, image in enumerate(prediction_image):
                image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
        else:
            prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))
    
    pred_list.append(prediction)
    
    # https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
    prediction_round = np.zeros_like(prediction)
    prediction_round[np.arange(len(prediction)), np.argmax(prediction)] = 1


    if np.array_equal(prediction_round, test_truth):
        verdict = "correct"
    else:
        verdict = "wrong"

    logger.info(' {} \t --> truth = {}, prediction = {}, verdict = {}, confidence = {},'.format(os.path.basename(output_dir),truth_clsfctn,config["labels_to_use"][np.argmax(prediction)], verdict, np.amax(prediction)))

    confidence = np.max(prediction)
    rows, subject_ids = df_comps
    # print(["fold" + config["fold"], truth_clsfctn, test_truth,np.around(prediction,2),prediction_round,verdict])
    rows.append([session_name, "fold" + config["fold"], val_or_test, truth_clsfctn, test_truth, np.around(prediction,2), verdict, confidence])
    # dasda
    subject_ids.append(os.path.basename(output_dir))  
        

    return rows, truth_list, pred_list  



def run_validation_cases_classification(df, logger, config_file, model, training_modalities, hdf5_file,
                         output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False, val_or_test="val", manual_label = None):
    
    global config
    config = config_file
    

    header = ("sessions", "fold", "val_or_test", "Type", "Truth", "Prediction", "Verdict", "Confidence")
    rows = list()
    subject_ids = list()
    truth_list_per_type = []
    pred_list_per_type = []

    # ###############################
    
    data_files = hdf5_file

    logger.info("\n" + "~"*60 + "Predictions" + "~"*60 + "\n")

    truth_list = []
    pred_list = []    

    for data_file in data_files:
        data_file = tables.open_file(data_file, "r")        
        
        count = 0

        for index, subj_id in enumerate(data_file.root.subject_ids):
            count += 1
            
            case_directory = subj_id.decode('utf-8')

            # print("Now doing:", case_directory)
            if config["seg_classify"] == 's_c' or config["seg_classify"] == 'c':
                rows, truth_list, pred_list = run_validation_case_classification(df, logger, val_or_test,
                                                                                (rows,subject_ids),truth_list, pred_list, 
                                                                                data_index=index, 
                                                                                output_dir=os.path.join(config["basepath"], "predictions", case_directory),
                                                                                model=model, 
                                                                                data_file=data_file,
                                                                                training_modalities=training_modalities, 
                                                                                output_label_map=True, 
                                                                                threshold=threshold, 
                                                                                labels = config['labels'], 
                                                                                overlap=overlap, 
                                                                                permute=permute,
                                                                                manual_label = manual_label)

        data_file.close()

    if config["seg_classify"] == 's_c' or config["seg_classify"] == 'c':
        truth_list = np.squeeze(np.asarray(truth_list))
        pred_list = np.squeeze(np.asarray(pred_list))
        truth_list_per_type.append(truth_list)
        pred_list_per_type.append(pred_list)



        df2 = pd.DataFrame.from_records(rows, columns=header)
        df_merged = pd.merge(df, df2, on="sessions")
        df_merged.to_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_" + val_or_test +  ".csv")

        truth_list_collapsed = [i for sublist in truth_list_per_type for i in sublist]
        pred_list_collapsed = [i for sublist in pred_list_per_type for i in sublist]
        np.save(config["basepath"] + "fold" + config["fold"] + "_" + val_or_test + '_truth.npy', truth_list_collapsed) 
        np.save(config["basepath"] + "fold" + config["fold"] + "_" + val_or_test +  '_pred.npy', pred_list_collapsed)


def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)


def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)[np.newaxis]
        predictions.append(reverse_permute_data(model.predict(temp_data)[0], permutation_key))
    return np.mean(predictions, axis=0)
