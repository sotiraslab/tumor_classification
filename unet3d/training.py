import math
import pickle
import os 
from functools import partial

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback, TensorBoard
from keras.models import load_model
from keras.activations import softmax
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.engine import Model
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import random
import time
import sys
from datetime import datetime

from unet3d.metrics import *

K.common.set_image_dim_ordering('th')

# How to run Keras on multiple cores?
# https://stackoverflow.com/questions/41588383/how-to-run-keras-on-multiple-cores
session_conf = tf.ConfigProto(intra_op_parallelism_threads=24, inter_op_parallelism_threads=24, allow_soft_placement=True)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, logging_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def load_old_model(model_file, n_labels):
    # print("Loading pre-trained model")

    metrics_seg = partial(dice_coef_multilabel, numLabels=n_labels)
    metrics_seg.__setattr__('__name__', 'dice_coef_multilabel')


    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                      'dice_coef_multilabel': metrics_seg,
                      'focal_loss': focal_loss,
                      'softmax': softmax, 
                      'label_0_dice_coef': get_label_dice_coefficient_function(0),
                      'label_1_dice_coef': get_label_dice_coefficient_function(1),
                      'label_2_dice_coef': get_label_dice_coefficient_function(2),
                      'label_3_dice_coef': get_label_dice_coefficient_function(3),
                      'label_4_dice_coef': get_label_dice_coefficient_function(4)}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error

# Source: https://keras.io/guides/writing_your_own_callbacks/
class LossPrintingCallback(Callback):
    def __init__(self, tumortype, file1, parentEpoch):
        super(LossPrintingCallback, self).__init__()
        self.parentEpoch = parentEpoch
        self.logfile = file1
        self.tumortype = tumortype

    # def on_train_batch_end(self, batch, logs=None):
    #     print("For training batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    # def on_test_batch_end(self, batch, logs=None):
    #     print("For validation batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(self.parentEpoch, keys))
        # print("The average training loss for epoch {} is {:7.4f} and validation loss is {:7.4f}.".format( self.parentEpoch, logs["loss"], logs["val_loss"] ) )
        logfile = open(self.logfile, "a")  # append mode 
        line = str(self.parentEpoch) + "," + str(logs["loss"]) + "," + str(logs["dice_coefficient"]) + "," + str(logs["val_loss"]) + "," + str(logs["val_dice_coefficient"]) + "\n"
        logfile.writelines(line) 
        logfile.close() 
        if self.parentEpoch > 100:
            self.model.save("model{}_ep{}_vloss{:.4f}.h5".format(self.tumortype, self.parentEpoch, logs["val_loss"]))

# Source: https://keras.io/guides/writing_your_own_callbacks/
class LastEpochModelSaverCallback(Callback):
    def __init__(self, modelpath, number_of_epochs):
        super(LastEpochModelSaverCallback, self).__init__()
        self.modelPath = modelpath
        self.numberOfEpochs = number_of_epochs

    def on_epoch_end(self, epoch, logs=None):
        # print(list(logs.keys()))
        if epoch+1 == self.numberOfEpochs:
            print("[DEBUG] Model saved at: {}modelClassifier_ep{:02d}_vacc{:.4f}.h5".format(self.modelPath, self.numberOfEpochs, logs["val_acc"]))
            self.model.save("{}modelClassifier_final_ep{:02d}_vacc{:.4f}.h5".format(self.modelPath, self.numberOfEpochs, logs["val_acc"]))
        else:
            pass

# Source: https://keras.io/guides/writing_your_own_callbacks/
class LossPrintingCallbackClassification(Callback):
    def __init__(self, validation_generator, parentEpoch, lfile):
        super(LossPrintingCallbackClassification, self).__init__()
        self.val_gen = validation_generator
        self.parentEpoch = parentEpoch
        self.logfile = lfile

    # def on_train_batch_end(self, batch, logs=None):
    #     print("For training batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    # def on_test_batch_end(self, batch, logs=None):
    #     print("For validation batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        # print("\n...Training: end of batch {}; got log keys: {}".format(batch, keys))
        # print("\n...self.model.output: ",self.model.output)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        # Source: https://stackoverflow.com/questions/47676248/accessing-validation-data-within-a-custom-callback --> xx
        # Source: https://github.com/keras-team/keras/issues/10472#issuecomment-472543538
        # 5.4.1 For each validation batch
        # for batch_index in range(0, 47): # range(0, number_of_val_steps)
        #     x,y = next(self.val_gen)
        #     temp_predict = (np.asarray(self.model.predict(x)))
        #     print("[DEBUG] Data#{}, Truth = {}, Predicted = {}".format(batch_index, y, temp_predict))

        # print("End epoch {} of training; got log keys: {}".format(self.parentEpoch, keys))  # ['val_loss', 'val_acc', 'loss', 'acc']
        logfile = open(self.logfile, "a")  # append mode 
        line = str(self.parentEpoch) + "," + str(logs["loss"]) + "," + str(logs["acc"]) + "," + str(logs["val_loss"]) + "," + str(logs["val_acc"]) + "\n"
        logfile.writelines(line) 
        logfile.close() 
        if self.parentEpoch >= 0:
            self.model.save("model_ep{}_vloss{:.4f}.h5".format(self.parentEpoch, logs["val_loss"]))

# https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0
class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, modelpath, monitor='val_acc', baseline=1):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.modelPath = modelpath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc == self.baseline:
                print('[TerminateOnBaseline] Epoch %d: Reached validation accuracy = 1, terminating training' % (epoch))
                self.model.save("{}modelClassifier_ep{:02d}_vacc{:.4f}.h5".format(self.modelPath, epoch+1, acc))
                self.model.stop_training = True

# https://github.com/keras-team/keras/issues/3358#issuecomment-312531958
# https://github.com/keras-team/keras/issues/3358#issuecomment-422826820
# https://github.com/keras-team/keras/issues/3358#issuecomment-533047896

class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


# Source: https://groups.google.com/g/keras-users/c/IVAarPXhv9A/m/0uEGNWa1EwAJ?pli=1
class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn


    def on_epoch_end(self, epoch, logs={}):
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)


def train_model(model, model_file, log_file, training_generator, validation_generator, training_steps_per_epoch, validation_steps_per_epoch,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param training_steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps_per_epoch: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=training_steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps_per_epoch,
                        callbacks=get_callbacks(model_file,
                                                log_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience),
                        # Following lines introduced to deal with problem of training stopping at Epoch 1/300
                        # If following lines are not there, then validation_data and early_stopping_patience needs to be commented out above
                        # Solution found at: https://github.com/ellisdg/3DUnetCNN/issues/58#issuecomment-435336522
                        use_multiprocessing=True)
                        # workers=0 # uncomment this if training not working


def train_model_dev1(model, model_file, log_file, training_generator, validation_generator, training_steps_per_epoch, validation_steps_per_epoch,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param training_steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps_per_epoch: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    print("[DEBUG] Training using train_model_dev1()")

    model_whole, model1, model2, train_GBM, test_GBM, train_METS, test_METS = model

    w0 = model_whole.get_weights()

    train_loss_over_epochs_GBM = []
    train_loss_over_epochs_METS = []
    val_loss_over_epochs_GBM = []
    val_loss_over_epochs_METS = []

    # print("[DEBUG] Trainable layers of whole model")
    # for l in model_whole.layers:        
    #     print(l.name, l.trainable)

    # print("[DEBUG] Trainable layers of GBM model")
    # for l in model1.layers:
    #     print(l.name, l.trainable)

    # print("[DEBUG] Trainable layers of METS model")
    # for l in model2.layers:
    #     print(l.name, l.trainable)

    # Motivation: https://towardsdatascience.com/keras-custom-training-loop-59ce779d60fb

    for epoch in range(n_epochs):
        print('Epoch %s:' % epoch)

        # Storing training losses for computing mean over all batches for a single epoch
        losses_train_GBM = []
        losses_train_METS = []

        # Batch loop for training
        for idx in range(training_steps_per_epoch):

            # print("[DEBUG] Training step: ",idx)
            x,y,tumortype = next(training_generator) # at every training_step one batch of training data is generated
            # print("[DEBUG] x.shape, y.shape, tumortype",x.shape,y.shape, tumortype)

            if tumortype == 'GBM':
                sample = x
                target = y

                # To tensors, input of 
                # K.function must be tensors
                sample = K.constant(sample)
                target = K.constant(target)

                # Running the train graph
                loss_train_GBM = train_GBM([sample, target])
                losses_train_GBM.append(loss_train_GBM[0])

                print("[DEBUG] Tumortype was:", tumortype, "=> Weight updates are as following: ", end=" ")
                w1 = model_whole.get_weights()
                print([np.allclose(x, y) for x, y in zip(w0, w1)])

            elif tumortype == 'METS':
                sample = x
                target = y

                # To tensors, input of 
                # K.function must be tensors
                sample = K.constant(sample)
                target = K.constant(target)

                # Running the train graph
                loss_train_METS = train_METS([sample, target])
                
                # Compute loss mean
                losses_train_METS.append(loss_train_METS[0])

                print("[DEBUG] Tumortype was:", tumortype, "=> Weight updates are as following: ", end=" ")
                w2 = model_whole.get_weights()
                print([np.allclose(x, y) for x, y in zip(w1, w2)])
                # assert w1 == w2
                w0 = w2



        print("losses_train_GBM",losses_train_GBM)
        print("losses_train_METS",losses_train_METS)
        loss_train_mean_GBM = np.mean(losses_train_GBM)
        loss_train_mean_METS = np.mean(losses_train_METS)
        train_loss_over_epochs_GBM.append(loss_train_mean_GBM)
        train_loss_over_epochs_METS.append(loss_train_mean_METS)
        print('[INFO] Train Loss GBM: ', loss_train_mean_GBM)
        print('[INFO] Train Loss METS:', loss_train_mean_METS)

        # Storing testing losses for computing mean over all batches for a single epoch
        losses_test_GBM = []
        losses_test_METS = []

        # Batch loop for training
        for idx in range(validation_steps_per_epoch):

            # print("[DEBUG] Validation step: ",idx)
            x_test,y_test,tumortype = next(validation_generator) # at every training_step one batch of training data is generated
            # print("[DEBUG] x_test.shape, y_test.shape, tumortype",x_test.shape,y_test.shape, tumortype)

            # K.function must be tensors
            sample_test = K.constant(x_test)
            target_test = K.constant(y_test)

            if tumortype == 'GBM':

                # Running the train graph
                loss_test_GBM = test_GBM([sample_test, target_test])
                losses_test_GBM.append(loss_test_GBM[0])

            elif tumortype == 'METS':

                # Running the train graph
                loss_test_METS = test_METS([sample_test, target_test])               
                losses_test_METS.append(loss_test_METS[0])

        print("losses_test_GBM",losses_test_GBM)
        print("losses_test_METS",losses_test_METS)
        loss_test_mean_GBM = np.mean(losses_test_GBM)
        loss_test_mean_METS = np.mean(losses_test_METS)
        val_loss_over_epochs_GBM.append(loss_test_mean_GBM)
        val_loss_over_epochs_METS.append(loss_test_mean_METS)
        print('[INFO] Test Loss GBM: ' , loss_test_mean_GBM)
        print('[INFO] Test Loss METS: ', loss_test_mean_METS)

   
        print(" ######################## End of Epoch:", epoch, "=> Reporting losses over all previous epochs ########################")
        print("train_loss_over_epochs_GBM",train_loss_over_epochs_GBM) 
        print("train_loss_over_epochs_METS",train_loss_over_epochs_METS) 
        print("val_loss_over_epochs_GBM",val_loss_over_epochs_GBM)
        print("val_loss_over_epochs_METS",val_loss_over_epochs_METS) 



def train_model_dev2(model, model_file, log_file, training_generator, validation_generator, training_steps_per_epoch, validation_steps_per_epoch,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    """
    Uses model.train_on_batch, model.test_on_batch
    """
    print("[DEBUG] Training using train_model_dev2()")
    
    model_whole, model1, model2, train_GBM, test_GBM, train_METS, test_METS = model

    w0 = model_whole.get_weights()

    train_loss_over_epochs_GBM = []
    train_loss_over_epochs_METS = []
    val_loss_over_epochs_GBM = []
    val_loss_over_epochs_METS = []

    # print("[DEBUG] Trainable layers of whole model")
    # for l in model_whole.layers:        
    #     print(l.name, l.trainable)

    # print("[DEBUG] Trainable layers of GBM model")
    # for l in model1.layers:
    #     print(l.name, l.trainable)

    # print("[DEBUG] Trainable layers of METS model")
    # for l in model2.layers:
    #     print(l.name, l.trainable)

    # Motivation: https://towardsdatascience.com/keras-custom-training-loop-59ce779d60fb

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        print('Epoch %s:' % epoch)

        # Storing training losses for computing mean over all batches for a single epoch
        losses_train_GBM = []
        losses_train_METS = []

        # Batch loop for training
        for idx in range(training_steps_per_epoch):

            # print("[DEBUG] Training step: ",idx)
            x,y,tumortype = next(training_generator) # at every training_step one batch of training data is generated
            # print("[DEBUG] x.shape, y.shape, tumortype",x.shape,y.shape, tumortype)

            if tumortype == 'GBM':               

                loss_train_GBM = model1.train_on_batch( x=x, y=y)

                losses_train_GBM.append(loss_train_GBM)

                print("[DEBUG] Tumortype was:", tumortype, "=> Weight updates are as following: ", end=" ")
                w1 = model_whole.get_weights()
                print([np.allclose(x, y) for x, y in zip(w0, w1)])

            elif tumortype == 'METS':
                
                # Running the train graph
                loss_train_METS = model2.train_on_batch( x=x, y=y)
                
                losses_train_METS.append(loss_train_METS)

                print("[DEBUG] Tumortype was:", tumortype, "=> Weight updates are as following: ", end=" ")
                w2 = model_whole.get_weights()
                print([np.allclose(x, y) for x, y in zip(w1, w2)])
                # assert w1 == w2
                w0 = w2


        print("losses_train_GBM",losses_train_GBM)
        print("losses_train_METS",losses_train_METS)
        loss_train_mean_GBM = np.mean(losses_train_GBM)
        loss_train_mean_METS = np.mean(losses_train_METS)
        train_loss_over_epochs_GBM.append(loss_train_mean_GBM)
        train_loss_over_epochs_METS.append(loss_train_mean_METS)
        print('[INFO] Train Loss GBM: ', loss_train_mean_GBM)
        print('[INFO] Train Loss METS:', loss_train_mean_METS)

        # Storing testing losses for computing mean over all batches for a single epoch
        losses_test_GBM = []
        losses_test_METS = []

        # Batch loop for training
        for idx in range(validation_steps_per_epoch):

            # print("[DEBUG] Validation step: ",idx)
            x_test,y_test,tumortype = next(validation_generator) # at every training_step one batch of training data is generated
            # print("[DEBUG] x_test.shape, y_test.shape, tumortype",x_test.shape,y_test.shape, tumortype)

            if tumortype == 'GBM':
                # Running the train graph
                loss_test_GBM = model1.test_on_batch( x=x_test, y=y_test)
                losses_test_GBM.append(loss_test_GBM)

            elif tumortype == 'METS':
                # Running the train graph
                loss_test_METS = model2.test_on_batch( x=x_test, y=y_test)
                losses_test_METS.append(loss_test_METS)

        print("losses_test_GBM",losses_test_GBM)
        print("losses_test_METS",losses_test_METS)
        loss_test_mean_GBM = np.mean(losses_test_GBM)
        loss_test_mean_METS = np.mean(losses_test_METS)
        val_loss_over_epochs_GBM.append(loss_test_mean_GBM)
        val_loss_over_epochs_METS.append(loss_test_mean_METS)
        print('[INFO] Test Loss GBM: ' , loss_test_mean_GBM)
        print('[INFO] Test Loss METS: ', loss_test_mean_METS)

        epoch_end_time = time.time()
   
        print(" ######################## End of Epoch:", epoch, "=> Reporting losses over all previous epochs ########################")
        print("train_loss_over_epochs_GBM",train_loss_over_epochs_GBM) 
        print("train_loss_over_epochs_METS",train_loss_over_epochs_METS) 
        print("val_loss_over_epochs_GBM",val_loss_over_epochs_GBM)
        print("val_loss_over_epochs_METS",val_loss_over_epochs_METS) 
        print("Epoch took:", epoch_end_time-epoch_start_time, " seconds")
    
            
def train_model_dev3(model, model_file, log_file, training_generator, validation_generator, training_steps_per_epoch, validation_steps_per_epoch,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None):
    """
    Uses model.fit_generator
    """
    print("[DEBUG] Training using train_model_dev3()")

    training_generator_GBM,training_generator_METS = training_generator
    validation_generator_GBM,validation_generator_METS = validation_generator
    training_steps_per_epoch_GBM,training_steps_per_epoch_METS = training_steps_per_epoch
    validation_steps_per_epoch_GBM,validation_steps_per_epoch_METS = validation_steps_per_epoch

    print("Number of training steps for GBM: ", training_steps_per_epoch_GBM)
    print("Number of validation steps for GBM: ", validation_steps_per_epoch_GBM)
    print("Number of training steps for METS: ", training_steps_per_epoch_METS)
    print("Number of validation steps for METS: ", validation_steps_per_epoch_METS)
    
    model_whole, model1, model2 = model

    # w0 = model_whole.get_weights()

    logfile_GBM, logfile_METS  = log_file
    file1 = open(logfile_GBM, "w") 
    file2 = open(logfile_METS, "w") 
    L = ["epoch,training_loss,training_dice,val_loss,val_dice\n"] 
    file1.writelines(L) 
    file2.writelines(L) 
    file1.close() 
    file2.close() 

    for parent_epoch in range(n_epochs):
        print('PARENT Epoch %s:' % parent_epoch)
        epoch_start_time = time.time()

        # Get batch of GBM data and fit_gen for GBM
        # model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, 
        # class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
        model1.fit_generator(generator=training_generator_GBM,
                        steps_per_epoch=training_steps_per_epoch_GBM,
                        epochs=1,
                        validation_data=validation_generator_GBM,
                        validation_steps=validation_steps_per_epoch_GBM,
                        callbacks=[
                                    # ModelCheckpoint(filepath='model_GBM.{val_loss:.2f}.h5'), 
                                   # CSVLogger('log_GBM.log', append=True),
                                   LossPrintingCallback("GBM", logfile_GBM, parent_epoch)],
                        use_multiprocessing=True)
                    
        # # Sanity of weight
        # print("[DEBUG] Tumortype was: GBM => Weight updates are as following: ", end=" ")
        # w1 = model_whole.get_weights()
        # print([np.allclose(x, y) for x, y in zip(w0, w1)])

        # Get batch of METS data and fit_gen for METS
        model2.fit_generator(generator=training_generator_METS,
                        steps_per_epoch=training_steps_per_epoch_METS,
                        epochs=1,
                        validation_data=validation_generator_METS,
                        validation_steps=validation_steps_per_epoch_METS,
                        callbacks=[
                        # ModelCheckpoint(filepath='model_METS.{val_loss:.2f}.h5'), 
                                    # CSVLogger('log_METS.log', append=True),
                                    LossPrintingCallback("METS", logfile_METS, parent_epoch)],
                        use_multiprocessing=True)


        # # Sanity of weight
        # print("[DEBUG] Tumortype was: METS => Weight updates are as following: ", end=" ")
        # w2 = model_whole.get_weights()
        # print([np.allclose(x, y) for x, y in zip(w1, w2)])
        # # assert w1 == w2
        # w0 = w2

        epoch_end_time = time.time()
        print("PARENT Epoch took:", epoch_end_time-epoch_start_time, " seconds")



def train_model_clsfctn(config, logger, model, model_file, log_file, tensorboard_log, model_save_path, training_generator, validation_generator,
                        training_steps_per_epoch, validation_steps_per_epoch, 
                        initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                        learning_rate_patience=20, early_stopping_patience=None, classweights=None, start_from_epoch = 0, seg_classify_mode = False):
    """
    Uses model.fit_generator
    """
    if seg_classify_mode == 'c':
        logger.info("[TRAINING] Training using train_model_clsfctn(): ONLY CLASSIFICATION MODE ")
        history = model.fit_generator(generator=training_generator,
                            steps_per_epoch= training_steps_per_epoch,
                            epochs=n_epochs,
                            initial_epoch=start_from_epoch,
                            validation_data=validation_generator,
                            validation_steps=validation_steps_per_epoch,
                            callbacks=[ModelCheckpoint(model_file[0], monitor = 'val_loss', save_best_only=False),
                                       LastEpochModelSaverCallback(model_save_path, n_epochs),
                                       CSVLogger(log_file, append=True),
                                       EarlyStopping(monitor="val_loss", patience=early_stopping_patience, verbose=1, restore_best_weights=False),
                                       TensorBoard(log_dir = tensorboard_log),
                                       # TerminateOnBaseline(model_save_path),
                                       ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
                                       LoggingCallback(logger.info)],
                            class_weight = classweights,
                            use_multiprocessing=True)

    elif seg_classify_mode == 's':
        logger.info("[TRAINING] Training using train_model_clsfctn(): ONLY SEGMENTATION MODE ")
        history = model.fit_generator(generator=training_generator,
                                    steps_per_epoch=training_steps_per_epoch,
                                    epochs=n_epochs,
                                    initial_epoch=start_from_epoch,
                                    validation_data=validation_generator,
                                    validation_steps=validation_steps_per_epoch,
                                    callbacks=[ModelCheckpoint(model_file[0], save_best_only=False),
                                               CSVLogger(log_file, append=True),
                                               EarlyStopping(patience=early_stopping_patience, verbose=1, restore_best_weights=False),
                                               ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
                                               LoggingCallback(logger.info)],
                                    # class_weight= None,
                                    use_multiprocessing=True)

    elif seg_classify_mode == 's_c':
        logger.info("[TRAINING] Training using train_model_clsfctn(): SEGMENTATION + CLASSIFICATION MODE ")
        history = model.fit_generator(generator=training_generator,
                            steps_per_epoch=training_steps_per_epoch,
                            epochs=n_epochs,
                            initial_epoch=start_from_epoch,
                            validation_data=validation_generator,
                            validation_steps=validation_steps_per_epoch,
                            callbacks=[ModelCheckpoint(model_file[0], monitor='val_clsfctn_op_loss', save_best_only=False),
                                       # ModelCheckpoint(model_file[1], monitor = 'val_clsfctn_op_acc', save_best_only=True),
                                       CSVLogger(log_file, append=True),
                                       EarlyStopping(monitor="val_clsfctn_op_loss", patience=early_stopping_patience, verbose=1, restore_best_weights=False),
                                       TensorBoard(log_dir=tensorboard_log),
                                       ReduceLROnPlateau(monitor='val_clsfctn_op_loss', factor=0.5, patience=10, verbose=1),
                                       LoggingCallback(logger.info)],
                            class_weight={"clsfctn_op": classweights, "segm_op": None},
                            use_multiprocessing=True)

    with open(os.path.join(config["basepath"], 'model_history'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)