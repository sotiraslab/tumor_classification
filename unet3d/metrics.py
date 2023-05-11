from functools import partial
import tensorflow as tf
from keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# https://github.com/keras-team/keras/issues/9395#issuecomment-379276452
def dice_coef_multilabel(y_true, y_pred, numLabels):
    '''
    This simply calculates the dice score for each individual label, and then sums them together, and includes the background. The best dice score you will ever get is equal to numLables*-1.0. When monitoring I always keep in mind that the dice for the background is almost always near 1.0.
    '''
    dice=0

    for index in range(numLabels):
        dice += dice_coefficient(y_true[:,index,:,:,:], y_pred[:,index,:,:,:])

    return dice

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred, axis=axis) + smooth/2)/(K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1-weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

# Define our custom loss function
# https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

# # https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7
# def precision(y_true, y_pred):  
#     """Precision metric.    
#     Only computes a batch-wise average of precision.    
#     Computes the precision, a metric for multi-label classification of  
#     how many selected items are relevant.   
#     """ 
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  
#     precision = true_positives / (predicted_positives + K.epsilon())    
#     return precision    


# def recall(y_true, y_pred): 
#     """Recall metric.   
#     Only computes a batch-wise average of recall.   
#     Computes the recall, a metric for multi-label classification of 
#     how many relevant items are selected.   
#     """ 
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
#     recall = true_positives / (possible_positives + K.epsilon())    
#     return recall
