import glob
import importlib
import os
import copy
import nibabel as nib
import numpy as np
import tables
from keras.engine import Model
from keras_contrib.layers import InstanceNormalization
from keras.utils import CustomObjectScope
from keras import activations

import pandas as pd
# from keras.utils.vis_utils import plot_model

from fm import deepexplain_vis, fm_vis, keras_vis
from unet3d.training import load_old_model
from unet3d.utils import pickle_load

# For keras-vis
from vis.utils import utils
from vis.visualization.saliency import visualize_cam_init_optimizer


def main(fold, exp):
    config_file_name = "config_file_Exp" + exp

    # The file gets executed upon import, as expected.
    config_file = importlib.import_module('config_files.' + config_file_name)

    # # Then you can use the module like normal
    set_fold = config_file.set_fold
    global config
    config = config_file.config

    overwrite = config["overwrite"]
    set_fold(fold, exp)

    model_file = glob.glob(os.path.abspath(config["basepath"] + "modelClassifier*.h5"))[0]
    print("[INFO] Loading model from {}".format(model_file))

    model = load_old_model(model_file, config['n_labels'])

    df = pd.read_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_val_test.csv", index_col=0)

    # plot_model(model, to_file=os.path.join(config["basepath"], 'model.png'), show_shapes=True, show_layer_names=True)

    # https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

    # summarize feature map shapes
    for i in range(len(model.layers)):
        layer = model.layers[i]
        print(i, layer.name, layer.output.shape)

    
    # 2. model for FMs

    # list_of_fm_idx = [i for i in range(len(model.layers)) if 'add' in model.layers[i].name]
    list_of_fm_idx = [11, 44]

    if len(list_of_fm_idx) > 5:
        list_of_fm_idx = list_of_fm_idx[:5]

    # # For channel with attention following fms are of importance
    # To check how the attention coefficients are affecting: plot following:
    # x -> layer44 Name: add_4 (None, 128, 16, 16, 16)
    # g -> layer55 Name: add_5 (None, 256, 8, 8, 8)
    # alpha -> layer66 Name: lambda (None, 128, 16, 16, 16)
    # alpha*x (before concat) -> layer68 Name: multiply_1 (None, 128, 16, 16, 16)
    # list_of_fm_idx += [66, 68]

    print("[INFO] FMs will be saved from following layer outputs: ", list_of_fm_idx)
    outputs = [layer.output for idx, layer in enumerate(model.layers) if idx in list_of_fm_idx]

    
    # extract wanted output
    model_fm = Model(inputs=model.inputs, outputs=outputs)

    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3. model for CAM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # https://github.com/alexisbcook/ResNetCAM-keras
    # # https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/

    # # (i) CAM_layer_idx is the fm after global average pooling
    # CAM_layer_idx = model.layers[56] # [i for i in model.layers if 'global_average_pooling' in i.name][0]

    # print(CAM_layer_idx.__dict__)

    # # (ii) This is the layer that is before the GAP layer (add_5 in our case)
    # last_conv_layer_idx = [i for i in model.layers if 'add_5' in i.name][0]

    # print("[INFO] CAM Layer --> ", CAM_layer_idx.name, CAM_layer_idx.output.shape)
    # print("[INFO] last_conv_layer --> ", last_conv_layer_idx.name, last_conv_layer_idx.output.shape)

    # layer_weights = np.squeeze(CAM_layer_idx.get_weights()[0])
    # # print("[INFO] CAM layer_weights.shape:", layer_weights.shape)

    

    # # extract wanted output along with the prediction (i.e. the output of last layer model.layers[-1].output)
    # model_CAM = Model(inputs=model.inputs, outputs=(last_conv_layer_idx.output, model.layers[-1].output))
    # model_CAM.summary(line_length=150)    


    prediction_dir = os.path.abspath(config["basepath"] + "fm/")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    tumor_type = config["tumor_type"]

    # gradcam
    custom_objects  = {'InstanceNormalization': InstanceNormalization}

    
    # Swap softmax with linear
    model.layers[-1].activation = activations.linear

    with CustomObjectScope(custom_objects): 
        model = utils.apply_modifications(model)

    opt_list = []

    for tumor in config["tumor_type"]:

        tumor_idx = config["tumor_type"].index(tumor)

        with CustomObjectScope(custom_objects): 
            # conv3d_15, conv3d_12, conv3d_9
            opt = visualize_cam_init_optimizer(model, layer_idx = -1, filter_indices = tumor_idx, penultimate_layer_idx=utils.find_layer_idx(model, 'conv3d_15'), backprop_modifier = 'guided')

        opt_list.append(opt)


    # for tumor in tumor_type:

    #     tumor_idx = config["tumor_type"].index(tumor)
    
    #     for test0_val1 in [0, 1]:
    #     # for test0_val1 in [0]:
        
    #         datafile = config[tumor_idx]["data_file_val"] if test0_val1 else config[tumor_idx]["data_file_test"]
            
    #         print("[INFO] hdf5_file_list", datafile)         
            
    #         data_file = tables.open_file(datafile, "r")

    #         session_counter = 0
            
    #         # for index in [5]: 
    #         for index in range(len(data_file.root.subject_ids)):
    #             session_counter += 1
    #             session_name = data_file.root.subject_ids[index].decode('utf-8')
    #             case_directory = os.path.join(prediction_dir, tumor + '_test_'+session_name) if test0_val1 == 0 else os.path.join(prediction_dir, tumor + '_val_'+session_name)
    #             print("Session: ", case_directory)

    #             if not os.path.exists(case_directory):
    #                 os.makedirs(case_directory)

    #             affine = data_file.root.affine[index]
    #             np.save(os.path.join(case_directory, "affine.npy"),affine)

    #             test_data = np.asarray([data_file.root.data[index]])

    #             for i, modality in enumerate(config["training_modalities"]):
    #                 image = nib.Nifti1Image(test_data[0, i], affine)
    #                 image.to_filename(os.path.join(case_directory, "data_{0}.nii.gz".format(modality)))

    #             test_truth = nib.Nifti1Image(data_file.root.truth[index][0], affine)
    #             test_truth.to_filename(os.path.join(case_directory, "truth.nii.gz"))

    #             # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Session sup_title ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #             truth_clsfctn = df.loc[df['sessions'] == session_name]['Verdict'].iloc[0]

    #             my_dict = df.loc[df['sessions'] == session_name].iloc[0].to_dict()

    #             fig_sup_title = []
    #             for key, value in my_dict.items():
    #                 fig_sup_title = fig_sup_title + [str(key) + ' : ' + str(value)]

    #             fig_sup_title = [fig_sup_title[2:5]] + [fig_sup_title[19:]]
    #             fig_sup_title = [str(i) for i in fig_sup_title]
    #             print(fig_sup_title)
    #             # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gradcam ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #             prediction = model.predict(test_data)
    #             print(prediction)
    #             pred = np.argmax(prediction)
    #             print(pred)

    #             keras_vis.main(case_directory, affine, model, test_data, pred, opt_list[pred], fig_sup_title, custom_objects  = {'InstanceNormalization': InstanceNormalization})

    #             # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #             # # for fm
    #             # list_of_req_conv_output = model_fm.predict(test_data)

    #             # fm_vis.main(affine, case_directory, list_of_fm_idx, list_of_req_conv_output)


    #             # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DeepExplain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #             # # for de
    #             # prediction = model.predict(test_data)
    #             # pred = np.argmax(prediction)

    #             # print("prediction", np.around(prediction,2))
    #             # print("Predicted class: ",pred)

    #             # deepexplain_vis.main(model, config, affine, case_directory, test_data, pred)

    #             # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #             # last_conv_output, prediction = model_CAM.predict(test_data)
                
    #             # CAM_vis.main(last_conv_output, affine, case_directory, layer_weights, pred)


    #         data_file.close()

    df = pd.read_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_partial_METS.csv", index_col=0)

    for tumor in ['METS']:

        tumor_idx = config["tumor_type"].index(tumor)    
                
        datafile = "/scratch/satrajit/tumor_classification_experiments/Exp" + exp + "/" + "data_partial_METS.h5"
        
        print("[INFO] hdf5_file_list", datafile)         
        
        data_file = tables.open_file(datafile, "r")

        session_counter = 0
        
        # for index in [5]: 
        for index in range(len(data_file.root.subject_ids)):
            session_counter += 1
            session_name = data_file.root.subject_ids[index].decode('utf-8')
            case_directory = os.path.join(prediction_dir, tumor + '_partial_'+session_name) 
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

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Session sup_title ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            truth_clsfctn = df.loc[df['sessions'] == session_name]['Verdict'].iloc[0]

            my_dict = df.loc[df['sessions'] == session_name].iloc[0].to_dict()

            fig_sup_title = []
            for key, value in my_dict.items():
                fig_sup_title = fig_sup_title + [str(key) + ' : ' + str(value)]

            fig_sup_title = [fig_sup_title[2:5]] + [fig_sup_title[19:]]
            fig_sup_title = [str(i) for i in fig_sup_title]
            print(fig_sup_title)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gradcam ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            prediction = model.predict(test_data)
            print(prediction)
            pred = np.argmax(prediction)
            print(pred)

            keras_vis.main(case_directory, affine, model, test_data, pred, opt_list[pred], fig_sup_title, custom_objects  = {'InstanceNormalization': InstanceNormalization})


        data_file.close()