from functools import partial

from keras import backend as K
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPooling3D, BatchNormalization, PReLU, Deconvolution3D, Conv3DTranspose, Lambda, add, concatenate
from keras.layers import GlobalAveragePooling3D, Reshape, Flatten, Dense, AveragePooling3D, Dropout, Softmax
from keras.layers import Permute, multiply
from keras.engine import Model
from keras.optimizers import Adam, SGD
from keras.activations import softmax
from keras import regularizers

from ..metrics import *


K.set_image_data_format("channels_first")


###########################################################################################################################################
# ############################################################ Building blocks ############################################################
###########################################################################################################################################

def create_localization_module(input_layer, n_filters, regularizer = None):
    convolution1 = create_convolution_block_isensee(input_layer, n_filters, regularizer = regularizer)
    convolution2 = create_convolution_block_isensee(convolution1, n_filters, kernel=(1, 1, 1), regularizer = regularizer)
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2), regularizer = None):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block_isensee(up_sample, n_filters, regularizer = regularizer)
    return convolution

def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first", regularizer = None, sqex = False):
    convolution1 = create_convolution_block_isensee(input_layer=input_layer, n_filters=n_level_filters, regularizer = regularizer)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block_isensee(input_layer=dropout, n_filters=n_level_filters, regularizer = regularizer)

    if not sqex:
        return convolution2
    else:
        return squeeze_excite_block(convolution2)


def create_convolution_block(input_layer, n_filters, name=None, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, regularizer=None):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, name=name, kernel_regularizer=regularizer)(input_layer)
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

def squeeze_excite_block(tensor, ratio=16):
    # Source: https://github.com/titu1994/keras-squeeze-excite-network
    # Modified by me for 3D

    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    # Squeeze: GAP
    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)

    # Excite stage 1: bottleneck with reduction ratio = ratio
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)

    # Excite stage 2
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    # print(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((4, 1, 2, 3))(se)

    #
    x = multiply([init, se])
    return x


def gating_signal(input, batch_norm=False, regularizer = None):
    # Source: code modified on top of https://github.com/MoleImg/Attention_UNet/blob/master/AttSEResUNet.py

    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    out_size = K.int_shape(input)[1] // 2
    x = Conv3D(filters=out_size, kernel_size=(1, 1, 1), padding='same', kernel_regularizer= regularizer)(input)
    if batch_norm: x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def attention_block(x, gating, regularizer=None):
    # Source: code modified on top of https://github.com/MoleImg/Attention_UNet/blob/master/AttSEResUNet.py

    """
    self gated attention, attention mechanism on spatial dimension
    :param x: input feature map
    :param gating: gate signal, feature map from the lower layer
    :param inter_shape: intermedium channle numer
    :param name: name of attention layer, for output
    :return: attention weighted on spatial dimension feature map
    """

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    inter_shape = shape_x[1]//2

    theta_x = Conv3D(filters = inter_shape, kernel_size = 2, strides=2, padding='same', kernel_regularizer=regularizer)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    # print("shape_theta_x", shape_theta_x)

    phi_g = Conv3D(filters = inter_shape, kernel_size = 1, padding='same', kernel_regularizer=regularizer)(gating)
    # print("phi_g", phi_g)

    upsample_g = Conv3DTranspose(filters = inter_shape,
                                 kernel_size = 3,
                                 strides=(shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3], shape_theta_x[4] // shape_g[4]),
                                 padding='same',
                                 kernel_regularizer=regularizer)(phi_g)  # 16
    # print("upsample_g", upsample_g)
    # upsample_g = UpSampling2D(size=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
    #                                  data_format="channels_last")(phi_g)

    concat_xg = add([upsample_g, theta_x])

    act_xg = Activation('relu')(concat_xg)

    # print("act_xg", act_xg)

    psi = Conv3D(filters = 1, kernel_size = 1, padding='same', kernel_regularizer=regularizer)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)

    # print("psi", shape_sigmoid)

    upsample_psi = UpSampling3D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3], shape_x[4] // shape_sigmoid[4]))(sigmoid_xg)  # 32

    # print("upsample_psi", upsample_psi)

    upsample_psi = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=1), arguments={'repnum': shape_x[1]})(upsample_psi)

    # print("upsample_psi_expanded", upsample_psi)


    y = multiply([upsample_psi, x])

    # print("y", y)

    result = Conv3D(filters = shape_x[1], kernel_size = 1, padding='same', kernel_regularizer=regularizer)(y)
    result_bn = BatchNormalization()(result)

    # print("result_bn", result_bn)
    return result_bn

###########################################################################################################################################

def isensee2017_classification_segmentation(input_shape,
                                            nb_classes = 2,
                                            depth = 5,
                                            n_base_filters=16,
                                            context_dropout_rate=0.3,
                                            gap_dropout_rate=0.4,
                                            n_segmentation_levels=3,
                                            n_labels=4,
                                            optimizer=Adam,
                                            initial_learning_rate=0.0005,
                                            loss_function_seg=weighted_dice_coefficient_loss,
                                            loss_function_clsfctn='categorical_crossentropy',
                                            loss_weights = "None",
                                            activation_name="sigmoid",
                                            regularizer=None,
                                            sqex=False,
                                            SE_before_concat = False,
                                            use_attention_gate = False):

    print("[MODEL] Training using isensee2017_classification_segmentation")
    # print(n_labels)

    # Metrics for classification

    metrics_clsfctn = 'accuracy'

    if not isinstance(metrics_clsfctn, list):
        metrics_clsfctn = [metrics_clsfctn]

    # Metrics for segmentation
    # if 0 in
    metrics_seg = partial(dice_coef_multilabel, numLabels=n_labels)
    metrics_seg.__setattr__('__name__', 'dice_coef_multilabel')

    if not isinstance(metrics_seg, list):
        metrics_seg = [metrics_seg]

    label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
    metrics_seg = metrics_seg + label_wise_dice_metrics

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Encoding path ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    for level_number in range(depth):
        n_level_filters = (2 ** level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters, regularizer=regularizer)
        else:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters, strides=(2, 2, 2), regularizer=regularizer)

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=context_dropout_rate, regularizer=regularizer, sqex=sqex)

        summation_layer = Add()([in_conv, context_output_layer])  # number of summation_layers = depth
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # Define classification model
    clsfctn_op_GAP = GlobalAveragePooling3D()(current_layer)

    if gap_dropout_rate:
        clsfctn_op_GAP = Dropout(rate=gap_dropout_rate)(clsfctn_op_GAP)

    clsfctn_Dense = Dense(nb_classes, name="Dense_without_softmax", kernel_regularizer=regularizer)(clsfctn_op_GAP)
    clsfctn_op = Activation('softmax', name="clsfctn_op")(clsfctn_Dense)

    model_clsfctn = Model(inputs=inputs, outputs=clsfctn_op)
    model_clsfctn.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function_clsfctn, metrics=metrics_clsfctn)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Decoding path ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number], regularizer=regularizer)

        if use_attention_gate:
            # gating_signal = current_layer (more #features, smaller dim) - comes from upsampling/encoding path
            # input_to_attention = level_output_layers[level_number] (less #features, bigger dim) - comes from downsampling/decoding path
            # attention_op = attention_gate(input_to_attention, gating_signal)

            input_to_attention = level_output_layers[level_number] # This comes from the decoding path

            if SE_before_concat:
                input_to_attention = squeeze_excite_block(input_to_attention)

            gating = gating_signal(current_layer, regularizer=regularizer)
            attention_op = attention_block(input_to_attention, gating, regularizer=regularizer)
            concatenation_layer = concatenate([attention_op, up_sampling], axis=1)

        else:
            concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)


        localization_output = create_localization_module(concatenation_layer, level_filters[level_number], regularizer=regularizer)
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1), kernel_regularizer=regularizer)(current_layer))
            # todo: should there be an activation after the conv3D block here? compare with  https://github.com/MIC-DKFZ/BraTS2017/blob/master/network_architecture.py

    # Accumulate deep supervision
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    if activation_name == 'sigmoid':
        activation_block = Activation(activation_name, name="segm_op")(output_layer)
    elif activation_name == 'softmax':
        # fixme: model doesnot converge with softmax activation
        # fixme: consult https://github.com/ellisdg/3DUnetCNN/issues?q=is%3Aissue+softmax
        # https://github.com/ellisdg/3DUnetCNN/issues/94#issuecomment-397208546
        # maybe replace following with keras.layers.Softmax(axis=1, name = "")(output_layer)
        activation_block = Softmax(axis=1, name = "segm_op")(output_layer)
        # activation_block = Lambda(lambda x: softmax(x, axis=1), name="segm_op")(output_layer)

    # Define segmentation model
    model_seg = Model(inputs=inputs, outputs=activation_block)

    model_seg.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function_seg, metrics=metrics_seg)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Define combined model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model_combined = Model(inputs=inputs, outputs=[clsfctn_op, activation_block])
    model_combined.compile(optimizer=optimizer(lr=initial_learning_rate),
                           loss={"clsfctn_op": loss_function_clsfctn , "segm_op": loss_function_seg},
                           loss_weights=eval(loss_weights),
                           metrics={"clsfctn_op": metrics_clsfctn, "segm_op": metrics_seg})

    return model_combined, model_seg, model_clsfctn


def isensee2017_classification(input_shape,
                               nb_classes,
                               depth=5,
                               n_base_filters=16,
                               context_dropout_rate=0.3,
                               gap_dropout_rate=0.4,
                               optimizer=Adam,
                               initial_learning_rate=0.0005,
                               loss_function_clsfctn='categorical_crossentropy',
                               regularizer=None,
                               sqex=False):
    print("[MODEL] Training using isensee2017_classification")

    # Metrics for classification
    metrics_clsfctn = 'accuracy'

    if not isinstance(metrics_clsfctn, list):
        metrics_clsfctn = [metrics_clsfctn]

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    for level_number in range(depth):
        n_level_filters = (2 ** level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters, regularizer=regularizer)
        else:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters, strides=(2, 2, 2), regularizer=regularizer)

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=context_dropout_rate, regularizer=regularizer, sqex=sqex)

        summation_layer = Add()([in_conv, context_output_layer])  # number of summation_layers = depth
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # Define classification model
    clsfctn_op_GAP = GlobalAveragePooling3D()(current_layer)

    if gap_dropout_rate:
        clsfctn_op_GAP = Dropout(rate=gap_dropout_rate)(clsfctn_op_GAP)

    clsfctn_Dense = Dense(nb_classes, name="Dense_without_softmax", kernel_regularizer=regularizer)(clsfctn_op_GAP)
    clsfctn_op = Activation('softmax', name="clsfctn_op")(clsfctn_Dense)

    model_clsfctn = Model(inputs=inputs, outputs=clsfctn_op)
    model_clsfctn.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function_clsfctn, metrics=metrics_clsfctn)

    return model_clsfctn