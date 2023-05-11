import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.layers import GlobalAveragePooling3D, Reshape, Flatten, Lambda, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.layers import  PReLU, ReLU, LeakyReLU 

from unet3d.metrics import *

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

# fixme: The unet architecture is not correct

def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    
    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth), batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2, batch_normalization=batch_normalization)
        
        if layer_depth < depth - 1:
            
            current_layer = MaxPooling3D(pool_size=pool_size,data_format='channels_first')(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution, n_filters=current_layer._keras_shape[1])(current_layer)
        # fixme: between upsampling and concatenate there should be a convolution
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1], input_layer=current_layer, batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1), data_format="channels_first")(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)

    return model


def create_convolution_block(input_layer, n_filters, name=None, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, name=name)(input_layer)
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

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size, strides=strides, data_format = 'channels_first')
    else:
        return UpSampling3D(size=pool_size)

##########################################################################################################################################################

# Correct unet-architecture

def unet_model_3d(input_shape = (1, 144, 144, 144), nb_classes = 3, n_labels = 2, initial_learning_rate=5e-4):

    '''
    Correct unet 3d model
    Check if there are 23 conv2D in the network [Quote: `` In total the network has 23 convolutional layers. ``, original Unet paper]
    Following implementation of unet taken from: https://github.com/zhixuhao/unet
    '''
    inputs = Input(input_shape)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Encoding path ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    custom_conv = partial(Conv3D, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    custom_maxpool = partial(MaxPooling3D, pool_size=2, strides=2)

    # depth 1
    conv1 = custom_conv(64,3)(inputs)
    conv1 = custom_conv(64,3)(conv1)
    pool1 = custom_maxpool()(conv1)

    # depth 2
    conv2 = custom_conv(128, 3)(pool1)
    conv2 = custom_conv(128, 3)(conv2)
    pool2 = custom_maxpool()(conv2)

    # # depth3
    conv3 = custom_conv(256, 3)(pool2)
    conv3 = custom_conv(256, 3)(conv3)
    pool3 = custom_maxpool()(conv3)

    # depth 4
    conv4 = custom_conv(512, 3)(pool3)
    conv4 = custom_conv(512, 3)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = custom_maxpool()(drop4)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bottleneck ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    conv5 = custom_conv(1024, 3)(pool4)
    conv5 = custom_conv(1024, 3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Decoding path ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # How about using Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same') instead of custom_conv(512, 2)(UpSampling3D(size = 2)) ?

    # depth 4
    up6 = custom_conv(512, 2)(UpSampling3D(size = 2)(drop5))
    merge6 = concatenate([drop4,up6], axis = 1)
    conv6 = custom_conv(512, 3)(merge6)
    conv6 = custom_conv(512, 3)(conv6)

    # depth 3
    up7 = custom_conv(256, 2)(UpSampling3D(size = 2)(conv6))
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = custom_conv(256, 3)(merge7)
    conv7 = custom_conv(256, 3)(conv7)

    # depth 2
    up8 = custom_conv(128, 2)(UpSampling3D(size = 2)(conv7))
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = custom_conv(128, 3)(merge8)
    conv8 = custom_conv(128, 3)(conv8)

    # depth 1
    up9 = custom_conv(64, 2)(UpSampling3D(size = 2)(conv8))
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = custom_conv(64, 3)(merge9)
    conv9 = custom_conv(64, 3)(conv9)

    # final layer [Quote: ``At the final layer a 1x1 convolution is used to map each 64- component feature vector to the desired number of classes.``]
    if n_labels == 2:
        conv10 = Conv3D(n_labels, 1, activation='sigmoid', name="segm_op")(conv9)
    else:
        # if there are mroe than 2 segmentation classes, then use softmax
        conv10 = Conv3D(n_labels, 1, activation='softmax', name="segm_op")(conv9)

    model_seg = Model(inputs = inputs, outputs = conv10)

    return model_seg




