import os

import nibabel as nib
import numpy as np
import tables
import keras 
import pandas as pd

# For plotting ROC
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# For plotting Precision-Recall
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For Confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

from .training import load_old_model
from .utils import pickle_load
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


def run_validation_case_classification(df_comps, truth_list, pred_list, data_index, output_dir, model, data_file, tumortype):
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
    # print("[DEBUG] run_validation_case_classification() --> tumortype", tumortype)

    # .csv file
    rows, subject_ids = df_comps

    # Data
    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])

    # Prediction
    prediction = model.predict(test_data, verbose=0)
    pred_list.append(prediction)

    # https://stackoverflow.com/questions/20295046/numpy-change-max-in-each-row-to-1-all-other-numbers-to-0
    prediction_round = np.zeros_like(prediction)
    prediction_round[np.arange(len(prediction)), np.argmax(prediction)] = 1

    if tumortype is not "Unknown":
        class_label = config["tumor_type"].index(tumortype)
        test_truth = keras.utils.to_categorical([class_label], num_classes=len(config["tumor_type"]))
        truth_list.append(test_truth)

        if np.argmax(prediction) == class_label:
            verdict = "correct"
        else:
            verdict = "wrong"

        print('truth = {},\t prediction = {},pred_type = {}, verdict = {}'.format(tumortype,np.around(prediction,2),config["tumor_type"][np.argmax(prediction)],verdict))
 
    else:
        print('\t\t\t prediction = {}'.format(np.around(prediction,2)))
        test_truth = "NA"
        truth_list.append(test_truth)
        verdict = "NA"

    rows.append([tumortype, test_truth, np.around(prediction,2), prediction_round, verdict])
    subject_ids.append(os.path.basename(output_dir))    

    return rows, subject_ids, truth_list, pred_list     




def run_validation_cases_classification(config_file, validation_keys_file_list, model_file, training_modalities, labels, hdf5_file,
                         output_label_map=False, output_dir=".", threshold=0.5, overlap=16, permute=False, val_or_test="val"):
    
    global config
    config = config_file
    
    model = load_old_model(model_file)

    header = ("Type", "Truth", "Actual prediction", "Thresholded prediction", "Verdict")
    rows = list()
    subject_ids = list()


    truth_list_per_type = []
    pred_list_per_type = []

    for i, tumor in enumerate(labels):
        validation_keys_file = pickle_load(validation_keys_file_list[i])
        data_file = hdf5_file[i] 

        print("\n","~"*60,"Predictions [TEST]","~"*60,"\n")
        data_file = tables.open_file(data_file, "r")

        truth_list = []
        pred_list = []    
        
        for index in validation_keys_file:
            if 'subject_ids' in data_file.root:
                case_directory = data_file.root.subject_ids[index].decode('utf-8')

            print("Session: ", case_directory, end=" ")
            rows, subject_ids, truth_list, pred_list = run_validation_case_classification((rows,subject_ids),truth_list, pred_list, 
                                                                                        data_index=index, 
                                                                                        output_dir=case_directory, 
                                                                                        model=model, 
                                                                                        data_file=data_file,
                                                                                        tumortype=tumor)

        truth_list = np.squeeze(np.asarray(truth_list))
        pred_list = np.squeeze(np.asarray(pred_list))
        truth_list_per_type.append(truth_list)
        pred_list_per_type.append(pred_list)
        np.save(config["basepath"] + 'TEST_truth.npy', truth_list) 
        np.save(config["basepath"] + 'TEST_pred.npy', pred_list)

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv(config["basepath"] + "prediction_scores_" + val_or_test +  ".csv")

    truth_list_collapsed = [i for sublist in truth_list_per_type for i in sublist]
    pred_list_collapsed = [i for sublist in pred_list_per_type for i in sublist]
    
    plot_cm_gen_report(np.asarray(truth_list_collapsed), np.asarray(pred_list_collapsed))

def plot_cm_gen_report(y_test, y_score):

    # y_test = np.concatenate((truth_GBM, truth_METS, truth_MENINGIOMA, truth_PITADE, truth_ACSCHW), axis=0)
    # y_score = np.concatenate((pred_GBM, pred_METS, pred_MENINGIOMA, pred_PITADE, pred_ACSCHW), axis=0)
    
    # Convert predicted and GT labels into 1d array
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_score, axis=1)

    type_truth = np.unique(y_true)[0]
    
    unique, counts = np.unique(y_pred, return_counts=True)
    cm = dict(zip(unique, counts))

    print(("="*30+" Classification Report "+"="*30+"\n"))
    for key in cm:
        print("Truth: {}, Classified as {}: {}/{}, {:.2f}%".format(config["tumor_type"][np.unique(y_true)[0]],config["tumor_type"][key],cm[key],y_true.shape[0],(cm[key]/y_true.shape[0])*100))  