import os
import logging
import os
import random
from itertools import cycle

import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix

random.seed(9001)
import numpy as np

import importlib
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'sans-serif'}
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rc('font', **font)


import seaborn as sns


def plot_roc(config, y_test, y_score, cohort_suffix, axis=plt):
    # Source: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
    #         https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    # y_test = np.concatenate((truth_GBM, truth_METS, truth_MENINGIOMA, truth_PITADE, truth_ACSCHW), axis=0)
    # y_score = np.concatenate((pred_GBM, pred_METS, pred_MENINGIOMA, pred_PITADE, pred_ACSCHW), axis=0)

    tumor_type = config["tumor_type"]
    # tumor_type = ['tumor', 'healthy']

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(tumor_type)):
        # print(i)
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        # unique, counts = np.unique(y_test[:, i],return_counts=True)
        # print(str(dict(zip(unique,counts))))
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    # plt.figure(1)
    # fig = plt.figure(figsize=(8, 8))

    colors = cycle(['red', 'green', 'darkorange', 'darkviolet', 'dimgray', 'dodgerblue', 'gold'])
    for i, color, name in zip(range(len(tumor_type)), colors, tumor_type):
        axis.plot(fpr[i], tpr[i], color=color, lw=4, label='{0} (AUC = {1:0.3f})'.format(name, roc_auc[i]))

    axis.plot([0, 1], [0, 1], 'k--', lw=4)
    axis.set_xlim([0, 1.0])
    axis.set_ylim([0, 1.05])
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_title('ROC [{}]'.format(cohort_suffix), fontsize=20)
    axis.legend(loc="best", prop=dict(size=18))

    # plt.tight_layout()
    # plt.savefig(config["basepath"] + "fold" + config["fold"] + "_" + 'ROC_' + cohort_suffix + '.png')
    # pickle.dump(axis, open(config["basepath"] + "fold" + config["fold"] + "_" + 'ROC_' + cohort_suffix + '.pkl', 'wb'))
    # plt.close()

    return axis
    # # Zoom in view of the upper left corner.
    # plt.figure(2)
    # plt.figure(figsize=(8, 8))

    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1.0)

    # for i, color, name in zip(range(len(tumor_type)), colors, tumor_type):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2, label='{0} (AUC = {1:0.3f})'.format(name, roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Zoomed ROC [{}]'.format(cohort_suffix))
    # plt.legend(loc="lower right")
    # plt.savefig(config["basepath"] + 'ROC_zoomed_' + cohort_suffix + '.png')
    # plt.close()


def plot_precision_recall(config, y_test, y_score, cohort_suffix, axis=plt):
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-precision-recall-curve-for-each-class-and-iso-f1-curves

    # y_test = np.concatenate((truth_GBM, truth_METS, truth_MENINGIOMA, truth_PITADE, truth_ACSCHW), axis=0)
    # y_score = np.concatenate((pred_GBM, pred_METS, pred_MENINGIOMA, pred_PITADE, pred_ACSCHW), axis=0)

    tumor_type = config["tumor_type"]
    # tumor_type = ['tumor', 'healthy']

    # Calculate precision, recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(tumor_type)):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # setup plot details
    colors = cycle(['red', 'green', 'darkorange', 'darkviolet', 'dimgray', 'dodgerblue', 'gold'])

    # axis.figure(figsize=(8, 8))

    lines = []
    labels = []

    # # Plot the iso-f1 lines
    # f_scores = np.linspace(0.2, 0.8, num=4)
    # lines = []
    # labels = []
    # for f_score in f_scores:
    #     x = np.linspace(0.01, 1)
    #     y = f_score * x / (2 * x - f_score)
    #     l, = axis.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    #     axis.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    # lines.append(l)
    # labels.append('iso-f1 curves')

    # Plot the precision-recall for each class
    for i, color, name in zip(range(len(tumor_type)), colors, tumor_type):
        l, = axis.plot(recall[i], precision[i], color=color, lw=4)
        lines.append(l)
        labels.append('{0}, (AUC = {1:0.3f})'.format(name, average_precision[i]))

    # fig = axis.gcf()
    # fig.subplots_adjust(bottom=0.25)
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.05])
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.set_title('Precision-Recall curve [{}]'.format(cohort_suffix), fontsize=20)
    axis.legend(lines, labels, loc="best", prop=dict(size=18))

    # axis.tight_layout()
    # axis.savefig(config["basepath"] + "fold" + config["fold"] + "_" + 'PR_' + cohort_suffix + '.png')
    # axis.close()

    return axis


def gen_report(config, y_test, y_score, cohort_suffix):
    # y_test = np.concatenate((truth_GBM, truth_METS, truth_MENINGIOMA, truth_PITADE, truth_ACSCHW), axis=0)
    # y_score = np.concatenate((pred_GBM, pred_METS, pred_MENINGIOMA, pred_PITADE, pred_ACSCHW), axis=0)

    tumor_type = config["tumor_type"]
    # tumor_type = ['tumor', 'healthy']

    # Convert predicted and GT labels into 1d array
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_score, axis=1)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    print(classification_report(y_true, y_pred, target_names=tumor_type, labels = tumor_type))

    # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class

    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # The diagonal entries are the accuracies of each class
    acc_per_class = cm.diagonal()

    for class_name, acc in zip(tumor_type, acc_per_class.tolist()):
        print("Accuracy of {}: {:.2f}%".format(class_name, acc * 100))

    # Writing to file
    with open(os.path.join(config["basepath"], "fold" + config["fold"] + "_" + "classification_report.log"), "a") as file1:
        # Writing data to a file
        file1.write("=" * 30 + " Classification Report [" + cohort_suffix + "]" + "=" * 30 + "\n")
        file1.write(classification_report(y_true, y_pred, target_names=tumor_type))
        file1.write("=" * 30 + " Accuracy per class " + "=" * 30 + "\n")
        for class_name, acc in zip(tumor_type, acc_per_class.tolist()):
            file1.write("Accuracy of {}: {:.2f}% \n".format(class_name, acc * 100))


def plot_cm(config, y_true, y_pred, cohort_suffix, axis=plt):
    if cohort_suffix == 'val':
        cmap = 'Blues'
    elif cohort_suffix == 'test':
        cmap = 'Reds'
    else:
        cmap = 'Greens'

    y_true = np.asarray(config["labels_to_use"])[np.argmax(y_true, axis=1)]
    y_pred = np.asarray(config["labels_to_use"])[np.argmax(y_pred, axis=1)]

    cm = confusion_matrix(y_true, y_pred, labels=config["labels_to_use"])
    # print(cm)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    # print(cm[5,:])
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            s = cm_sum[i]

            if i == j:
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

    cm_perc = pd.DataFrame(cm_perc, index=config["labels_to_use"], columns=config["labels_to_use"])
    cm_perc.index.name = 'True label'
    cm_perc.columns.name = 'Predicted label'
    # fig, ax = plt.subplots(figsize=(10,10))

    axis.set_title('Confusion matrix [{}]'.format(cohort_suffix), fontsize='25')
    axis.xaxis.label.set_size(25)
    axis.yaxis.label.set_size(25)
    # todo: annot_kws={"size": 15}
    hmap = sns.heatmap(cm_perc, cmap=cmap, annot=annot, square=True, fmt='', ax=axis, annot_kws={"size": 20}, cbar_kws={"shrink": 0.75}, linewidths=0.1, vmax=100, linecolor='gray')

    hmap.set_xticklabels(config["labels_to_use"], fontsize=20)
    hmap.set_yticklabels(config["labels_to_use"], fontsize=20)

    # use matplotlib.colorbar.Colorbar object
    cbar = hmap.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    for _, spine in hmap.spines.items():
        spine.set_visible(True)

    # plt.tight_layout()
    # plt.savefig(config["basepath"] + "fold" + config["fold"] + "_" + 'CM_' + cohort_suffix + '.png')
    # plt.close()

    return axis

def plot_cm_with_absent_labels(config, y_true, y_pred, cohort_suffix, axis=plt):

    if cohort_suffix == 'val':
        cmap = 'Blues'
    elif cohort_suffix == 'test':
        cmap = 'Reds'
    else:
        cmap = 'Greens'
    
    y_true = np.asarray(config["labels_to_use"])[np.argmax(y_true, axis=1)]
    y_pred = np.asarray(config["labels_to_use"])[np.argmax(y_pred, axis=1)]

    # List of labels which are present in either true or predicted classes
    labels_to_use = [i for i in config['labels_to_use'] if i in list(set(np.unique(y_true).tolist() + np.unique(y_pred).tolist()))]

    # List of labels which are present in only true classes
    classes_present_true = [tuple([idx,i]) for idx, i in enumerate(labels_to_use) if i in list(set(np.unique(y_true).tolist()))]

    # List of labels which are present in only pred classes
    classes_present_pred = [tuple([idx,i]) for idx, i in enumerate(labels_to_use) if i in list(set(np.unique(y_pred).tolist()))]

    cm = confusion_matrix(y_true, y_pred, labels=labels_to_use)

    # Remove all non-zero rows from cm (i.e. rows of classes that are not present in true class)
    cm = cm[~np.all(cm == 0, axis=1)] 

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            s = cm_sum[i]

            if c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

    cm_perc = pd.DataFrame(cm_perc, index=classes_present_true, columns=classes_present_pred)
    cm_perc.index.name = 'True label'
    cm_perc.columns.name = 'Predicted label'

    axis.set_title('Confusion matrix [{}]'.format(cohort_suffix), fontsize='25')
    axis.xaxis.label.set_size(25)
    axis.yaxis.label.set_size(25)
    hmap = sns.heatmap(cm_perc, cmap=cmap, annot=annot, square=True, fmt='', ax=axis, annot_kws={"size": 20}, cbar_kws={"shrink": 0.75}, linewidths=0.1, vmax=100, vmin =0, linecolor='gray')

    hmap.set_xticklabels(labels_to_use, fontsize=20)
    hmap.set_yticklabels(labels_to_use, fontsize=20)

    # use matplotlib.colorbar.Colorbar object
    cbar = hmap.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    for _, spine in hmap.spines.items():
        spine.set_visible(True)

    return axis

def main(fold, exp, cohort_suffix = 'test'):
    config_file_name="config_file_Exp"+exp
    config_file = importlib.import_module('config_files.'+config_file_name)
    set_fold = config_file.set_fold
    global config
    config = config_file.config
    set_fold(fold, exp)

    # ***************************************************************************************************************************


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
    logger.info("*" * 40 + " [ PLOT RESULTS - VALIDATION + TESTING CASES ] " + "*" * 40)
    logger.info("*************************************************************************************************")

    if config["seg_classify"] == 's_c' or config["seg_classify"] == 'c':
        fig, axes = plt.subplots(1, 3, figsize=(33,11))

        ax_ROC, ax_PR, ax_CM = axes

        truth_list_collapsed = np.load(os.path.join(config["basepath"], "fold" + str(config["fold"]) + "_" + cohort_suffix + '_truth.npy'))
        pred_list_collapsed = np.load(os.path.join(config["basepath"], "fold" + str(config["fold"]) + "_" + cohort_suffix + '_pred.npy'))

        print(truth_list_collapsed)
        print(pred_list_collapsed)

        # logger.debug("Plotting ROC..")
        # plot_roc(config, np.asarray(truth_list_collapsed), np.asarray(pred_list_collapsed), cohort_suffix, ax_ROC)

        # # Plot Precision-Recall for this fold
        # logger.debug("Plotting PR..")
        # plot_precision_recall(config, np.asarray(truth_list_collapsed), np.asarray(pred_list_collapsed), cohort_suffix, ax_PR)

        # Plot Confusion Matrix and generate classification report for this fold
        logger.debug("Plotting CM..")
        plot_cm_with_absent_labels(config, np.asarray(truth_list_collapsed), np.asarray(pred_list_collapsed), cohort_suffix, ax_CM)


        # Save plot
        plt.suptitle('Experiment {} - fold {}'.format(exp, fold), fontsize=24)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # [left, bottom, right, top]
        plt.savefig(config["basepath"] + "fold" + str(config["fold"]) + "_" + 'ROC_PR_CM_' + cohort_suffix +'.png')
        plt.close()

        # gen_report(config, np.asarray(truth_list_collapsed), np.asarray(pred_list_collapsed), cohort_suffix)


        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Crosstab results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # df = pd.read_csv(config["basepath"] + "fold" + config["fold"] + "_" + "prediction_scores_" + cohort_suffix + ".csv", index_col=0)

        # cont1 = pd.crosstab(index =  [df["cohort_suffix"], df["Tumor_type"]], columns = df['Verdict'])
        # cont2 = pd.crosstab(index =  [df["cohort_suffix"], df["Slicethickness"]], columns = df['Verdict'])

        # fig, axes = plt.subplots(1,2, figsize=(10,10))
        # sns.heatmap(cont1, ax = axes[0], square = True, annot_kws={"size": 20}, cbar_kws={"shrink": 0.75}, cmap = 'YlOrBr', annot=True, cbar=True, fmt="d")
        # sns.heatmap(cont2, ax = axes[1], square = True, annot_kws={"size": 20}, cbar_kws={"shrink": 0.75}, cmap = 'YlOrBr', annot=True, cbar=True, fmt="d")
        # # Save plot
        # plt.yticks(rotation=0)
        # plt.suptitle('Experiment {} - fold {}'.format(exp, fold), fontsize=24)
        # plt.tight_layout(rect=[0, 0, 1, 0.98])  # [left, bottom, right, top]
        # plt.savefig(config["basepath"] + "fold" + config["fold"] + "_" + 'results_crosstabs' + cohort_suffix +'.png')
        # plt.close()
    else:
        logger.info("The trained model does not perform classification. No classification ROC/PR/CM plots to generate. Exiting....")