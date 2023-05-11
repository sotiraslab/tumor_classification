from functools import partial

from keras import backend as K
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPooling3D, BatchNormalization, PReLU, Deconvolution3D
from keras.layers import GlobalAveragePooling3D, Reshape, Flatten, Dense
from keras.engine import Model
from keras.optimizers import Adam

from keras.layers.merge import concatenate
from ..metrics import *

import numpy as np

K.set_image_data_format("channels_first")


def isensee2017_seg_model(input_shape=(4, 128, 128, 128),
                          n_base_filters=16,
                          depth=5,
                          dropout_rate=0.3,

                          n_segmentation_levels=3,
                          n_labels=4,
                          optimizer=Adam,
                          initial_learning_rate=5e-4,

                          loss_function=weighted_dice_coefficient_loss,
                          activation_name="sigmoid",

                          include_label_wise_dice_coefficients=True):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    for level_number in range(depth):
        n_level_filters = (2 ** level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])  # number of summation_layers = depth
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)

    metrics = partial(dice_coef_multilabel, numLabels=n_labels)
    metrics.__setattr__('__name__', 'dice_coef_multilabel')

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
    return model

# def isensee2017_model_joint(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
#                       n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
#                       loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid",
#                       include_label_wise_dice_coefficients=False, metrics=dice_coefficient):
#     """
#     This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
#     https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

#     This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
#     Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


#     :param input_shape:
#     :param n_base_filters:
#     :param depth:
#     :param dropout_rate:
#     :param n_segmentation_levels:
#     :param n_labels:
#     :param optimizer:
#     :param initial_learning_rate:
#     :param loss_function:
#     :param activation_name:
#     :return:
#     """
#     inputs = Input(input_shape)

#     current_layer = inputs
#     level_output_layers = list()
#     level_filters = list()

#     for level_number in range(depth):
#         n_level_filters = (2**level_number) * n_base_filters
#         level_filters.append(n_level_filters)

#         if current_layer is inputs:
#             in_conv = create_convolution_block_isensee(current_layer, n_level_filters)
#         else:
#             in_conv = create_convolution_block_isensee(current_layer, n_level_filters, strides=(2, 2, 2))

#         context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

#         summation_layer = Add()([in_conv, context_output_layer]) # number of summation_layers = depth
#         level_output_layers.append(summation_layer)
#         current_layer = summation_layer

#     segmentation_layers = list()
#     segmentation_layers_METS = list()
#     for level_number in range(depth - 2, -1, -1):
#         up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
#         concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
#         localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
#         current_layer = localization_output
#         if level_number < n_segmentation_levels:
#             segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1), name="seg_GBM"+str(level_number))(current_layer))
#             segmentation_layers_METS.insert(0, Conv3D(2, (1, 1, 1), name="seg_METS"+str(level_number))(current_layer))

#     output_layer = None
#     for level_number in reversed(range(n_segmentation_levels)):
#         segmentation_layer = segmentation_layers[level_number]
#         segmentation_layer_METS = segmentation_layers_METS[level_number]
#         if output_layer is None:
#             output_layer = segmentation_layer
#             output_layer_METS = segmentation_layer_METS
#         else:
#             output_layer = Add()([output_layer, segmentation_layer])
#             output_layer_METS = Add()([output_layer_METS, segmentation_layer_METS])

#         if level_number > 0:
#             output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)
#             output_layer_METS = UpSampling3D(size=(2, 2, 2))(output_layer_METS)

#     activation_block = Activation(activation_name)(output_layer)
#     activation_block_METS = Activation(activation_name)(output_layer_METS)

#     model = Model(inputs=inputs, outputs=[activation_block, activation_block_METS]) # entire model
#     model1 = Model(inputs=inputs, outputs=activation_block) # for GBM
#     model2 = Model(inputs=inputs, outputs=activation_block_METS) # for METS
 
#     if not isinstance(metrics, list):
#         metrics = [metrics]

#     if include_label_wise_dice_coefficients and n_labels > 1:
#         label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
#         if metrics:
#             metrics = metrics + label_wise_dice_metrics
#         else:
#             metrics = label_wise_dice_metrics

#     model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
#     model1.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
#     model2.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function, metrics=metrics)
  
#     return model, model1, model2
    

# def unet_model_3d_dev(input_shape, pool_size=(2, 2, 2), n_labels=4, initial_learning_rate=0.00001, deconvolution=False,
#                   depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
#                   batch_normalization=False, activation_name="sigmoid"):
#     """
#     Builds the 3D UNet Keras model.f
#     :param metrics: List metrics to be calculated during model training (default is dice coefficient).
#     :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
#     coefficient for each label as metric.
#     :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
#     layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
#     to train the model.
#     :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
#     layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
#     :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
#     divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
#     :param pool_size: Pool size for the max pooling operations.
#     :param n_labels: Number of binary labels that the model is learning.
#     :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
#     :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
#     increases the amount memory required during training.
#     :return: Untrained 3D UNet Model
#     """
#     inputs = Input(input_shape)
#     current_layer = inputs
#     levels = list()

#     # add levels with max pooling
#     for layer_depth in range(depth):
#         layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth), batch_normalization=batch_normalization)
#         layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2, batch_normalization=batch_normalization)
        
#         if layer_depth < depth - 1:
          
#           current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
#           levels.append([layer1, layer2, current_layer])
#         else:
          
#           current_layer = layer2
#           levels.append([layer1, layer2])

#     # add levels with up-convolution or up-sampling
#     for layer_depth in range(depth-2, 0, -1):
#         up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, n_filters=current_layer._keras_shape[1])(current_layer)
#         concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
#         current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=concat, batch_normalization=batch_normalization)
#         current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=current_layer, batch_normalization=batch_normalization)


#     # add levels with up-convolution or up-sampling
    
#     layer_depth = 0
#     up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, n_filters=current_layer._keras_shape[1])(current_layer)
#     concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)

#     # Branching out for GBM/METS starts from here for joint segmentation: Only defining layers with no input
#     custom_layer1_GBM = Conv3D(levels[layer_depth][1]._keras_shape[1], (3, 3, 3), padding='same', strides=(1, 1, 1), name="first_layer_GBM"+str(layer_depth))
#     custom_act1_GBM = Activation('relu',name="custom_act1_GBM")        
#     custom_layer2_GBM = Conv3D(levels[layer_depth][1]._keras_shape[1], (3, 3, 3), padding='same', strides=(1, 1, 1), name="second_layer_GBM"+str(layer_depth))
#     custom_act2_GBM = Activation('relu',name="custom_act2_GBM")

#     custom_layer1_METS = Conv3D(levels[layer_depth][1]._keras_shape[1], (3, 3, 3), padding='same', strides=(1, 1, 1), name="first_layer_METS"+str(layer_depth))
#     custom_act1_METS = Activation('relu',name="custom_act1_METS")        
#     custom_layer2_METS = Conv3D(levels[layer_depth][1]._keras_shape[1], (3, 3, 3), padding='same', strides=(1, 1, 1), name="second_layer_METS"+str(layer_depth))
#     custom_act2_METS = Activation('relu',name="custom_act2_METS")

#     final_convolution_GBM = Conv3D(n_labels, (1, 1, 1), name="GBM_final_conv")
#     final_convolution_METS = Conv3D(2, (1, 1, 1), name="METS_final_conv")

#     act_GBM = Activation(activation_name,name="GBM_act")
#     act_METS = Activation(activation_name,name="METS_act")

#     outputs=act_GBM(final_convolution_GBM(custom_act2_GBM(custom_layer2_GBM(custom_act1_GBM(custom_layer1_GBM(concat))))))
#     outputs_METS=act_METS(final_convolution_METS(custom_act2_METS(custom_layer2_METS(custom_act1_METS(custom_layer1_METS(concat))))))

#     # model1 and model2 will be required during prediction when we will have single input and need single output
#     model1 = Model(inputs=inputs, outputs=outputs)
#     model2 = Model(inputs=inputs, outputs=outputs_METS)
#     model = Model(inputs = inputs , outputs = [outputs, outputs_METS])

#     # Need to compile the models for Keras APIs like train_on_batch()
#     model.compile(optimizer=Adam(lr=initial_learning_rate), loss=weighted_dice_coefficient_loss)
#     model1.compile(optimizer=Adam(lr=initial_learning_rate), loss=weighted_dice_coefficient_loss)
#     model2.compile(optimizer=Adam(lr=initial_learning_rate), loss=weighted_dice_coefficient_loss)

#     # if not isinstance(metrics, list):
#     #     metrics = [metrics]

#     # if include_label_wise_dice_coefficients and n_labels > 1:
#     #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
#     #     if metrics:
#     #         metrics = metrics + label_wise_dice_metrics
#     #     else:
#     #         metrics = label_wise_dice_metrics

#     # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
#     # model1.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
#     # model2.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)

#     # model.summary()

#     ##########################################################################
#     y_pred_GBM = model1(inputs)
#     y_true_GBM = Input(shape=(4, 144,144,144))
#     optimizer = Adam(lr=5e-4)
#     loss_GBM = weighted_dice_coefficient_loss(y_true_GBM, y_pred_GBM)
#     updates_op_GBM = optimizer.get_updates(params=model1.trainable_weights, loss=loss_GBM)
#     train_GBM = K.function( inputs=[inputs, y_true_GBM], outputs=[loss_GBM], updates=updates_op_GBM)
#     test_GBM = K.function(  inputs=[inputs, y_true_GBM], outputs=[loss_GBM])
#     ##########################################################################
#     y_pred_METS = model2(inputs)
#     y_true_METS = Input(shape=(2, 144,144,144))
#     optimizer = Adam(lr=5e-4)
#     loss_METS = weighted_dice_coefficient_loss(y_true_METS, y_pred_METS)
#     updates_op_METS = optimizer.get_updates(params=model2.trainable_weights, loss=loss_METS)
#     train_METS = K.function( inputs=[inputs, y_true_METS], outputs=[loss_METS], updates=updates_op_METS)
#     test_METS = K.function(  inputs=[inputs, y_true_METS], outputs=[loss_METS])
#     ##########################################################################

#     return model, model1, model2, train_GBM, test_GBM, train_METS, test_METS


# def unet_model_3d_classification(input_shape, pool_size=(2, 2, 2), n_labels=4, initial_learning_rate=0.00001, deconvolution=False,
#                   depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
#                   batch_normalization=False, activation_name="sigmoid"):
#     """
#     Builds the 3D UNet Keras model.f
#     :param metrics: List metrics to be calculated during model training (default is dice coefficient).
#     :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
#     coefficient for each label as metric.
#     :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
#     layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
#     to train the model.
#     :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
#     layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
#     :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
#     divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
#     :param pool_size: Pool size for the max pooling operations.
#     :param n_labels: Number of binary labels that the model is learning.
#     :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
#     :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
#     increases the amount memory required during training.
#     :return: Untrained 3D UNet Model
#     """
#     inputs = Input(input_shape)
#     current_layer = inputs
#     levels = list()

    
#     # add levels with max pooling
#     for layer_depth in range(depth):
#         layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth), batch_normalization=batch_normalization)
#         layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2, batch_normalization=batch_normalization)
        
#         if layer_depth < depth - 1:
            
#             current_layer = MaxPooling3D(pool_size=pool_size,data_format='channels_first')(layer2)
#             levels.append([layer1, layer2, current_layer])
#         else:
            
#             current_layer = layer2
#             levels.append([layer1, layer2])

#     # # Motivation: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8363710
#     # nb_classes = 2
#     # clsfctn_op = current_layer    
#     # clsfctn_op = GlobalAveragePooling3D(name="GAP")(clsfctn_op)
#     # print(K.int_shape(clsfctn_op)[1])
#     # clsfctn_op = Reshape(target_shape=(K.int_shape(clsfctn_op)[1],1,1,1))(clsfctn_op)
#     # # https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
#     # clsfctn_op = Conv3D(nb_classes, (1,1,1), padding="valid", strides=(1,1,1), name="aggregator", activation="softmax")(clsfctn_op)
#     # clsfctn_op = Reshape(target_shape=(2,1))(clsfctn_op)


#     # Motivation: https://www.pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/
#     nb_classes = 2
#     i = Flatten()((current_layer))
#     h = Dense(100, activation='relu')(i)
#     clsfctn_op = Dense(nb_classes, activation='softmax')(h)

#     # # add levels with up-convolution or up-sampling
#     # for layer_depth in range(depth-2, -1, -1):
#     #     up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, n_filters=current_layer._keras_shape[1])(current_layer)
#     #     concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
#     #     current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=concat, batch_normalization=batch_normalization)
#     #     current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=current_layer, batch_normalization=batch_normalization)

#     # final_convolution = Conv3D(n_labels, (1, 1, 1), data_format="channels_first")(current_layer)
#     # act = Activation(activation_name)(final_convolution)

#     # model_clsfctn = Model(inputs=inputs, outputs=clsfctn_op)
#     # model = Model(inputs=inputs, outputs=act)

#     # if not isinstance(metrics, list):
#     #     metrics = [metrics]

#     # if include_label_wise_dice_coefficients and n_labels > 1:
#     #     label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
#     #     if metrics:
#     #         metrics = metrics + label_wise_dice_metrics
#     #     else:
#     #         metrics = label_wise_dice_metrics

#     # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
#     model_clsfctn.compile(optimizer=Adam(lr=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

#     # return model, model_clsfctn
#     return model_clsfctn

# ################################################################################## Following modules are for Isensee ##################################################################################

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, data_format="channels_first")(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


create_convolution_block_isensee = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block_isensee(input_layer, n_filters)
    convolution2 = create_convolution_block_isensee(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block_isensee(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block_isensee(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block_isensee(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size, strides=strides, data_format = 'channels_first')
    else:
        return UpSampling3D(size=pool_size)