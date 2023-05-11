import glob
import importlib
import os

import nibabel as nib
import numpy as np
import tables
# SCORECAM
from keras_contrib.layers import InstanceNormalization

from fm.gradcamutils import ScoreCam
from unet3d.training import load_old_model


def main(fold, exp):
    config_file_name="config_file_Exp"+exp

    # The file gets executed upon import, as expected.
    config_file = importlib.import_module('config_files.'+config_file_name)

    # # Then you can use the module like normal
    set_fold = config_file.set_fold
    global config
    config = config_file.config

    overwrite=config["overwrite"]   
    set_fold(fold, exp)


    model_file = glob.glob(os.path.abspath(config["basepath"]+"modelClassifier*.h5"))[0]
    print("[INFO] Loading model from {}".format(model_file))


    model = load_old_model(model_file)

    custom_objects = {'InstanceNormalization': InstanceNormalization}

    
    
    ############## Set data
    prediction_dir = os.path.abspath(config["basepath"]+"fm/")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    tumortypes = ['HGG']

    for type_of_tumor in tumortypes:

        tumor_id = config["tumor_type"].index(type_of_tumor)
        

        validation_keys_file = config[tumor_id]["validation_file"]      
        labels = config[tumor_id]["labels"]
        datafile = config[tumor_id]["data_file_val"]

        print("[INFO] validation_keys_file_list", validation_keys_file)
        print("[INFO] labels_list", labels)
        print("[INFO] hdf5_file_list", datafile)

        
        validation_indices = [2] # pickle_load(validation_keys_file)
        data_file = tables.open_file(datafile, "r")

        session_counter = 0
        
        for index in validation_indices:
            session_counter += 1
            case_directory = os.path.join(prediction_dir, '_'+type_of_tumor+'_'+data_file.root.subject_ids[index].decode('utf-8'))  
            print("Session: ", case_directory)

            if not os.path.exists(case_directory):
                os.makedirs(case_directory)

            affine = data_file.root.affine[index]
            np.save(os.path.join(case_directory, "affine.npy"),affine)

            test_data = np.asarray([data_file.root.data[index]])

            for i, modality in enumerate(config["training_modalities"]):
                image = nib.Nifti1Image(test_data[0, i], affine)
                image.to_filename(os.path.join(case_directory, "data_{0}.nii.gz".format(modality)))

            test_truth = nib.Nifti1Image(data_file.root.truth[index][0], affine)
            test_truth.to_filename(os.path.join(case_directory, "truth.nii.gz"))

            layer_name = 'conv3d_15'

            prediction = model.predict(test_data)
            pred = np.argmax(prediction)
            print("prediction", np.around(prediction,2))
            print("Predicted class: ",pred)

            CAM=ScoreCam(model,test_data,layer_name)
            
            print("[INFO] CAM.shape",CAM.shape)

            nib.Nifti1Image(CAM, affine).to_filename(os.path.join(case_directory, "scoreCAM.nii.gz"))


            if session_counter == 1:
                break

        data_file.close()