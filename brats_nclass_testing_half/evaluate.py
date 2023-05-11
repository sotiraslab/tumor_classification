import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import importlib
import logging
import copy

import matplotlib.patches as patches


from matplotlib.colors import colorConverter
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


from fm.fm_vis import show_slice, get_largest_slice
from unet3d.utils.extract_coarse_GT import extract_3D_bbox
from unet3d.utils.utils import check_unique_elements

def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def extract_bbox_plot_coords(T1c, gt_multiclass):

    gt = copy.deepcopy(gt_multiclass)

    gt[gt>0] = 1

    x1, y1, _, x2, y2, _ = extract_3D_bbox(gt)
    

    crop_margin = 0
    # in case coordinates are out of image boundaries
    y1 = np.maximum(y1 - crop_margin, 0)
    y2 = np.minimum(y2 + crop_margin, T1c.shape[0])
    x1 = np.maximum(x1 - crop_margin, 0)
    x2 = np.minimum(x2 + crop_margin, T1c.shape[1])

    # print("x1, y1, x2, y2", x1, y1, x2, y2)

    height = x2 - x1
    width = y2 - y1

    return (y1, x1), width, height

def gt_pred_viz_with_scan(T1c, truth, fm, savepath):

    slicenum = get_largest_slice(truth)
    

    fig, axes = plt.subplots(1,3, figsize=(15,5))
    slice_id = "[:,:,slicenum].T"
    axes[0].imshow(eval("T1c" + slice_id), cmap="gray", origin="lower")
    axes[0].set_title("Scan", fontsize=15)
    
    (y1, x1), width, height = extract_bbox_plot_coords(T1c, truth)

    axes[0].add_patch(patches.Rectangle((y1, x1), width, height,linewidth=2,edgecolor='r',facecolor='none', ls = '--'))

    
    
    axes[1].imshow(eval("truth" + slice_id), cmap="jet", origin="lower")
    axes[1].set_title("GT", fontsize=15)
    axes[2].imshow(eval("fm" + slice_id), cmap="jet", origin="lower")
    axes[2].set_title("Pred", fontsize=15)


    for ax in axes:
        ax.axis('off')
        
        
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()



def main(fold, exp):
    print("[INFO] Experiment#", exp) 
    config_file_name="config_file_Exp"+exp
    config_file = importlib.import_module('config_files.'+config_file_name)
    set_fold = config_file.set_fold
    global config
    config = config_file.config
    set_fold(fold, exp)

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
    logger.info("*" * 40 + " [ EVALUATE SEGMENTATION ] " + "*" * 40)
    logger.info("*************************************************************************************************")


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ for GBM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    savepath = os.path.join(config["basepath"], "fold" + config["fold"] + "_" + "segmentation_scores.csv")
    plot_savepath = os.path.join(config["basepath"], "fold" + config["fold"] + "_" + "segmentation_scores_boxplot.png")

    if config["labels"] == (0, 1):
        whole_tumor = True
    else:
        whole_tumor = False

    if not whole_tumor:
        header = ("WholeTumor", "TumorCore", "EnhancingTumor")
        masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    else:
        header = ("WholeTumor",)
        masking_functions = (get_whole_tumor_mask, )

    
    rows = list()
    subject_ids = list()
    for case_folder in glob.glob(config["basepath"] + "predictions/*"):
        if not os.path.isdir(case_folder):
            continue
        subj_id = os.path.basename(case_folder)
        subject_ids.append(subj_id)

        truth = nib.load(os.path.join(case_folder, "truth.nii.gz")).get_fdata()
        prediction = nib.load(os.path.join(case_folder, "prediction.nii.gz")).get_fdata()   

        try:
            T1c = nib.load(os.path.join(case_folder, "data_T1c_subtrMeanDivStd.nii.gz")).get_fdata()
        except:
            T1c = nib.load(os.path.join(case_folder, "data_Flair_subtrMeanDivStd.nii.gz")).get_fdata()

        if whole_tumor:
            truth = get_whole_tumor_mask(truth)

        # Visualize
        gt_pred_viz_with_scan(T1c, truth, prediction, os.path.join(config["basepath"], "predictions", "{}.png".format(subj_id)))

        # Dice scores
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])
        

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    print("[INFO] Now saving csv of Dice scores for GBM validation data at:",savepath)
    df.to_csv(savepath)

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    print("[INFO] Saving Boxplot of Dice scores for GBM validation data at: ", plot_savepath)
    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.grid()
    plt.savefig(plot_savepath)
    plt.close()

   

if __name__ == "__main__":
    main()
