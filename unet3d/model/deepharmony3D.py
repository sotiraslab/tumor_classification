from functools import partial

from keras import backend as K
from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, MaxPooling3D, BatchNormalization, PReLU, Deconvolution3D, Conv3DTranspose, Lambda
from keras.layers import GlobalAveragePooling3D, Reshape, Flatten, Dense, AveragePooling3D, Dropout, Softmax
from keras.layers import Permute, multiply
from keras.engine import Model
from keras.optimizers import Adam, SGD
from keras.activations import softmax
from keras import regularizers
from .unet import create_convolution_block, concatenate


from ..metrics import *


K.set_image_data_format("channels_first")

'''
Motivation: 
[1] B. E. Dewey et al., “DeepHarmony: A deep learning approach to contrast harmonization across scanner changes,” Magn. Reson. Imaging, vol. 64, pp. 160–170, Dec. 2019.

* Main differences from vanilla U-net:
1. addition of final concatenation step between input and final feature map
2. Use of strided conv (and deconv) instead of maxpooling (and nearest neighbour upsampling)
    -- Read paper for more details
    
[!!] Implementation below not tested
'''

###########################################################################################################################################
# ############################################################ Building blocks ############################################################
###########################################################################################################################################

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

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size, strides=strides, data_format = 'channels_first')
    else:
        return UpSampling3D(size=pool_size)


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

###########################################################################################################################################

def isensee2017_classification_deepharmony(input_shape, nb_classes, n_base_filters=16, depth=5, dropout_rate=0.3,
                                          n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                                          loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid",
                                          include_label_wise_dice_coefficients=True, metrics=dice_coefficient):



    print("[MODEL] Training using isensee2017_classification_deepharmony")

    inputs = Input(input_shape)
    

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block_isensee(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer]) # number of summation_layers = depth
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    clsfctn_op_GAP = GlobalAveragePooling3D()(current_layer)
    clsfctn_op = Dense(nb_classes, activation='softmax', name="Dense_softmax")(clsfctn_op_GAP)

    
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(input_shape[0], (1, 1, 1))(current_layer))

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

    # https://keras.io/guides/functional_api/#models-with-multiple-inputs-and-outputs

    model_cls = Model(inputs=inputs, outputs=clsfctn_op)
    model_cls.compile(optimizer=optimizer(lr=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    model_DH = Model(inputs=inputs, outputs=activation_block)
    model_DH.compile(optimizer=optimizer(lr=initial_learning_rate), loss='mean_absolute_error', metrics=['mse'])

    model_combined = Model(inputs=inputs, outputs=[clsfctn_op,activation_block])
    model_combined.compile(optimizer=optimizer(lr=initial_learning_rate), loss=['categorical_crossentropy','mean_absolute_error'], metrics=['mse'])
    # model_combined.summary()
    return model_DH


def unet_deepharmony(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="relu"):

    print("[MODEL] Training using unet_deepharmony")

    '''
    Ref1: https://link.springer.com/content/pdf/10.1007%2F978-3-319-67389-9_34.pdf
    Ref2: DeepHarmony: A deep learning approach to contrast harmonization across scanner changes, Blake E. Dewey et al.

    Disclaimer: U-Net network used as-is and changes from Ref1/2 have NOT been included. Only changes are in the loss and accuracy metrics.
    '''

    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth), batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2, batch_normalization=batch_normalization)
        
        if layer_depth < depth - 1:
            
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=current_layer, batch_normalization=batch_normalization)

    final_convolution = Conv3D(input_shape[0], (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss='mean_absolute_error', metrics=['mse'])

    # plot_model(model, to_file='model.png')

    return model


