import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import logging 

def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


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
    logger.info("*" * 40 + " [ PLOT LOSS + CHOOSE BEST MODEL ] " + "*" * 40)
    logger.info("*************************************************************************************************")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get the loss and accuracy values ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    logfile_path = config["basepath"] + "training*.log"
    savefile_path_subplots = config["basepath"] + "fold" + config["fold"] + "_" + "all_vars_graph.png"
    savefile_path = config["basepath"] + "fold" + config["fold"] + "_" + "loss_acc_graph.png"

    training_log = glob.glob(os.path.abspath(logfile_path))[0]    
    logger.info("Reading loss vs epoch from:" + training_log)
    logger.info("Saving loss vs epoch plot for training at:" + savefile_path)
    logger.info("Saving all vars vs epoch plot for training at:" + savefile_path_subplots)

    training_df = pd.read_csv(training_log).set_index('epoch')

    # Subplot of all tracked metrics vs epoch - as subplots
    split_idx = [idx for idx,i in enumerate(training_df.columns.tolist()) if 'val' in i][0]
    df_tr, df_val = np.split(training_df, [split_idx], axis=1)
    subplot_h_w = 2
    n_subplots = len(df_tr.columns.tolist())
    cmap = 'rainbow'
    ax = df_tr.plot.line(subplots=True, layout = (3,-1), grid = True,  figsize=(n_subplots*subplot_h_w, n_subplots*subplot_h_w), sharey = False, lw=3, style = '--', title = df_tr.columns.tolist(), cmap=cmap)
    df_val.plot.line(subplots=True, ax = ax.ravel()[:n_subplots], grid = True, sharey = False, lw = 3, cmap=cmap)
    plt.suptitle('Experiment {} - fold {}'.format(exp,fold), fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  #[left, bottom, right, top]
    plt.savefig(savefile_path_subplots)   
    plt.close()


    if config["seg_classify"] == 'c':
        # For classification
        training_loss = training_df['loss']
        validation_loss = training_df['val_loss']
        training_accuracy = training_df['acc']
        validation_accuracy = training_df['val_acc']
    elif config["seg_classify"] == 's_c':
        # For classification + segmentation
        training_loss = training_df['clsfctn_op_loss']
        validation_loss = training_df['val_clsfctn_op_loss']
        training_accuracy = training_df['clsfctn_op_acc']
        validation_accuracy = training_df['val_clsfctn_op_acc']
    elif config["seg_classify"] == 's':
        # For segmentation
        training_loss = training_df['loss']
        validation_loss = training_df['val_loss']
        training_accuracy = training_df['dice_coef_multilabel']
        validation_accuracy = training_df['val_dice_coef_multilabel']

        # training_accuracy = training_df['dice_coefficient']
        # validation_accuracy = training_df['val_dice_coefficient']

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Determined best epoch ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Find Classification model with best validation acc and loss, keep that file and delete everything else

    # Rationale: Epoch with best validation accuracy and validation loss might not have any overlap. So the following rationale is followed
    # to determine the best epoch:

    # 1. The array of validation accuracy and loss across all epochs
    val_acc = np.array(validation_accuracy)
    val_loss = np.array(validation_loss)

    # 2. Best validation acc might have occurred at multiple epochs. So determine the indices of ALL those epochs.
    best_val_acc_eps = np.argwhere(np.isclose(val_acc, np.amax(val_acc))).flatten()

    # 3. Now once those epochs are determined, determine what the val losses were for those epochs only
    best_val_acc_eps_val_loss = val_loss[best_val_acc_eps]

    # 4. Of those losses, determine the index of the min val loss
    best_val_acc_loss_ep = np.argmin(best_val_acc_eps_val_loss)

    # 5. Finally, check out of all best_val_acc_eps, which ep does that correspond to. 
    best_epoch_int = best_val_acc_eps[best_val_acc_loss_ep] + 1

    best_epoch = "{:03d}".format(best_epoch_int)
    
    best_epoch_str = 'ep' + str(best_epoch)
    
    print("[INFO] Best epoch Classification: {}".format(best_epoch_str))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loss plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fig, axes = plt.subplots(2,1, figsize=(10,10))

    
    axes[0].plot(training_loss.values, linewidth=2, label='training loss')
    axes[0].plot(validation_loss.values, linewidth=2,  label='validation loss')

    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_xlim((0, len(training_df.index)))
    axes[0].legend(loc='upper right')

    axes[0].axvline(x=best_epoch_int-1,alpha=0.5,color='k',linestyle='--', label='Best epoch (highest val_acc w/ min val_loss)')

    
    # First change the ticks to [0, 100, 200 ..]
    axes[0].xaxis.set_ticks((np.arange(0, len(training_df.index)+1, 50)))

    # Then append the array of new ticks [ReduceLR epochs + max_val_acc epoch] to existing ticks
    x_ticks = np.concatenate((axes[0].get_xticks(), np.asarray([best_epoch_int-1])))

    # Set the xticks
    axes[0].set_xticks((x_ticks))

    # https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend
    handles, labels = axes[0].get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
    axes[0].legend(newHandles, newLabels, loc="best")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accuracy plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    axes[1].plot(training_accuracy.values, linewidth=2,  label='training accuracy')
    axes[1].plot(validation_accuracy.values, linewidth=2,  label='validation accuracy')

    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_xlim((0, len(training_df.index)))
    axes[1].legend(loc='lower right')


    # axes[1].axhline(y=max(val_acc),alpha=0.5,color='k',linestyle='--')
    axes[1].axvline(x=best_epoch_int-1,alpha=0.5,color='k',linestyle='--', label='Best epoch (highest val_acc w/ min val_loss)')

    # First change the ticks to [0, 100, 200 ..]
    axes[1].xaxis.set_ticks((np.arange(0, len(training_df.index)+1, 50)))

    # Then append the array of new ticks [ReduceLR epochs + max_val_acc epoch] to existing ticks
    x_ticks = np.concatenate((axes[1].get_xticks(), np.asarray([best_epoch_int-1])))

    # Set the xticks
    axes[1].set_xticks((x_ticks))

    # https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend
    handles, labels = axes[1].get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

    axes[1].legend(newHandles, newLabels, loc='best')


    plt.suptitle('Experiment {} - fold {}'.format(exp,fold), fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  #[left, bottom, right, top]
    plt.savefig(savefile_path)   
    plt.close()

    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Keep only best model and move rest to temp_models/ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    modelPath = config["basepath"] + "modelClassifier*.h5"
    modelnames = glob.glob(os.path.abspath(modelPath))
    
    Path(config["basepath"] + "temp_models/").mkdir(exist_ok=True)
    
    for m in modelnames:
        # print(m)
        # print(os.path.basename(m))
        if best_epoch_str in m:
            bestModel = m
        else:
            if "final" in m:
                pass
            else:
                os.rename(m, config["basepath"] + "temp_models/" + os.path.basename(m))

    logger.info("Best Classification model: {}".format(bestModel))


if __name__ == "__main__":
    main()

