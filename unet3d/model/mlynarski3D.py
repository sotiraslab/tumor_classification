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

'''
Motivation: 
Mlynarski, P., Delingette, H., Criminisi, A. and Ayache, N., “Deep learning with mixed supervision for brain tumor segmentation,” J. Med. Imaging 6(03), 1 (2019).
https://github.com/PawelMlynarski/segmentation_mixed_supervision

* Original network is 2d (uses input image size of 300 x 300 and 2 class labels) and extends vanilla 2d u-net, following are the changes:
1. additional image-level classification branch connected at the end of decoding path
2. classification branch takes as input the second to last convolutional layer of U-Net (bs x 64 x 101 x 101)
3. composed of 1 mean-pooling, 1 convolutional layer, and 7 fully connected layers. 
4. (bs x 64 x 101 x 101) --> mean-pooling (kernels size = 8 × 8, stride = 8 × 8). --> (bs x 64 x 13 x 13)
5.  (bs x 64 x 13 x 13) --> Conv2d(n = 32, k_size = 3 × 3, ReLu)  --> (bs x 32 x 11 x 11)
6. (bs x 32 x 11 x 11) --> Flatten --> (bs x 3872) --Dense(300, 250, 200, 150, 100, 50, #classes)---> (bs x #classes)
7. In the dense layers, take (bs x 300) fm and concat with (bs x 100) before feeding to Dense(50)

'''

###########################################################################################################################################
def mlynarski_on_unet_3d(input_shape,
                         nb_classes=2,
                         n_labels=4,
                         optimizer=Adam,
                         initial_learning_rate=0.0005,
                         loss_function_seg=weighted_dice_coefficient_loss,
                         loss_function_clsfctn='categorical_crossentropy',
                         activation_name="sigmoid"):
    '''
    Vanilla Unet architecture with modifications according to: 'Deep Learning with Mixed Supervision for Brain Tumor Segmentation', Pawel Mlynarski et al.
    Check if there are 23 conv2D in the network [Quote: `` In total the network has 23 convolutional layers. ``, original Unet paper]
    Following implementation of unet taken from: https://github.com/zhixuhao/unet with modifications in layer conv9
    '''
    inputs = Input(input_shape)

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
    custom_conv = partial(Conv3D, activation='relu', padding='same', kernel_initializer='he_normal')
    custom_maxpool = partial(MaxPooling3D, pool_size=2, strides=2)

    # depth 1
    conv1 = custom_conv(64, 3)(inputs)
    conv1 = custom_conv(64, 3)(conv1)
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
    up6 = custom_conv(512, 2)(UpSampling3D(size=2)(drop5))
    merge6 = concatenate([drop4, up6], axis=1)
    conv6 = custom_conv(512, 3)(merge6)
    conv6 = custom_conv(512, 3)(conv6)

    # depth 3
    up7 = custom_conv(256, 2)(UpSampling3D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=1)
    conv7 = custom_conv(256, 3)(merge7)
    conv7 = custom_conv(256, 3)(conv7)

    # depth 2
    up8 = custom_conv(128, 2)(UpSampling3D(size=2)(conv7))
    merge8 = concatenate([conv2, up8], axis=1)
    conv8 = custom_conv(128, 3)(merge8)
    conv8 = custom_conv(128, 3)(conv8)

    # depth 1
    up9 = custom_conv(64, 2)(UpSampling3D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=1)
    conv9 = custom_conv(64, 3)(merge9)
    conv9 = custom_conv(64, 3)(conv9)

    # final layer [Quote: ``At the final layer a 1x1 convolution is used to map each 64- component feature vector to the desired number of classes.``]
    conv10 = Conv3D(n_labels, 1)(conv9)

    if activation_name == 'sigmoid':
        activation_block = Activation(activation_name, name="segm_op")(conv10)
    elif activation_name == 'softmax':
        # fixme: model doesnot converge with softmax activation
        # fixme: consult https://github.com/ellisdg/3DUnetCNN/issues?q=is%3Aissue+softmax
        # https://github.com/ellisdg/3DUnetCNN/issues/94#issuecomment-397208546
        # maybe replace following with keras.layers.Softmax(axis=1, name = "")(output_layer)
        activation_block = Softmax(axis=1, name="segm_op")(conv10)
        # activation_block = Lambda(lambda x: softmax(x, axis=1), name="segm_op")(output_layer)

    model_seg = Model(inputs=inputs, outputs=activation_block)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification branch, ref: Mlynarski ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # (i) original mlynarski implementation : too many parameters due to dense block
    # mean_pool = AveragePooling3D(pool_size=8, strides=8, padding="valid")(conv9)
    # clsfctn_op = Conv3D(32, 3, activation="relu")(mean_pool)
    # clsfctn_op = Flatten()(clsfctn_op)
    # clsfctn_op_d1 = Dense(300, activation='relu')(clsfctn_op)
    # clsfctn_op_d2 = Dense(250, activation='relu')(clsfctn_op_d1)
    # clsfctn_op_d3 = Dense(200, activation='relu')(clsfctn_op_d2)
    # clsfctn_op_d4 = Dense(150, activation='relu')(clsfctn_op_d3)
    # clsfctn_op_d5 = Dense(100, activation='relu')(clsfctn_op_d4)
    # clsfctn_op_concat = concatenate([clsfctn_op_d1, clsfctn_op_d5], axis=1)
    # clsfctn_op_d6 = Dense(50, activation='relu')(clsfctn_op_concat)
    # clsfctn_op = Dense(nb_classes, activation='softmax', name="clsfctn_op")(clsfctn_op_d6)

    # (ii) alternative
    mean_pool = AveragePooling3D(pool_size=8, strides=8, padding="valid")(conv9)
    clsfctn_op = Conv3D(32, 3, activation="relu")(mean_pool)
    clsfctn_op = Conv3D(16, 3, activation="relu")(clsfctn_op)  # extra
    clsfctn_op = Conv3D(8, 3, activation="relu")(clsfctn_op)  # extra
    clsfctn_op = Flatten()(clsfctn_op)
    clsfctn_op_d1 = Dense(300, activation='relu')(clsfctn_op)
    clsfctn_op_d2 = Dense(250, activation='relu')(clsfctn_op_d1)
    clsfctn_op_d3 = Dense(200, activation='relu')(clsfctn_op_d2)
    clsfctn_op_d4 = Dense(150, activation='relu')(clsfctn_op_d3)
    clsfctn_op_d5 = Dense(100, activation='relu')(clsfctn_op_d4)
    clsfctn_op_concat = concatenate([clsfctn_op_d1, clsfctn_op_d5], axis=1)
    clsfctn_op_d6 = Dense(50, activation='relu')(clsfctn_op_concat)
    clsfctn_op = Dense(nb_classes, activation='softmax', name="clsfctn_op")(clsfctn_op_d6)

    model_clsfctn = Model(inputs=inputs, outputs=clsfctn_op)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Define combined model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model_combined = Model(inputs=inputs, outputs=[clsfctn_op, activation_block])
    model_combined.compile(optimizer=optimizer(lr=initial_learning_rate),
                           loss={"clsfctn_op": loss_function_clsfctn, "segm_op": loss_function_seg},
                           metrics={"clsfctn_op": metrics_clsfctn, "segm_op": metrics_seg})

    return model_combined, model_seg, model_clsfctn