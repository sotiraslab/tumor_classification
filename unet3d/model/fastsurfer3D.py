from keras.engine import Model
from keras.layers import GlobalAveragePooling3D, Dense, concatenate
from keras.layers import Input, LeakyReLU, Activation, Conv3D, MaxPooling3D, BatchNormalization, Lambda
from keras.optimizers import Adam

from keras import backend as K
K.set_image_data_format("channels_first")

from ..metrics import *

'''
Motivation: 
L. Henschel, S. Conjeti, S. Estrada, K. Diers, B. Fischl, and M. Reuter, “FastSurfer - A fast and accurate deep learning based neuroimaging pipeline,” Neuroimage, vol. 219, Oct. 2020.
https://github.com/Deep-MI/FastSurfer/blob/c5e96778a237c7e21b1c525b8cf8cf1fc95cfdf2/FastSurferCNN/models/networks.py

* Modified for 3d volumes with following changes compared to actual network:

1.   LeakyRELU used instead of PreLU (to reduce number of parameters)
2.   In (1,1,1) Conv blocks, padding = 'same' used instead of valid (not for any particular reason)
3.   (3,3,3) conv used instead of (5,5,5) (not for any particular reason)
4.   Original 2D Network is extended to 3D
5.   num_filters = 32 instead of 64 (Out-of-memory)
6.   Using BN as first layer of CompetitiveEncoderBlockInput even though inputs are z-scored? (need to change)
'''

def CompetitiveEncoderBlockInput(input_layer, n_filters, kernel=(3, 3, 3), padding='same', strides=(1, 1, 1)):
    """
        Encoder Block = CompetitiveDenseBlockInput + Max Pooling
        * CompetitiveDenseBlockInput's computational Graph
        in -> BN -> {Conv -> BN -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} -> {Conv -> BN} -> CDB_out

        * Max Pooling
        CDB_out -> {MaxPool3D} -> out
    """

    # Input batch normalization
    layer0_bn = BatchNormalization(axis=1)(input_layer)

    # Convolution block1
    layer1_conv = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer0_bn)
    layer1_bn = BatchNormalization(axis=1)(layer1_conv)
    layer1_activation = LeakyReLU()(layer1_bn)  # Should be PreLU

    # Convolution block2 [with First Maxout]
    layer2_conv = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer1_activation)
    layer2_bn = BatchNormalization(axis=1)(layer2_conv)

    layer1_bn_ed = Lambda(lambda x: K.expand_dims(x, -1), name='expand_d1')(layer1_bn)  # Add Singleton Dimension along 6th
    layer2_bn_ed = Lambda(lambda x: K.expand_dims(x, -1), name='expand_d2')(layer2_bn)  # Add Singleton Dimension along 6th
    layer_1_2_bn_ed_concat = concatenate([layer1_bn_ed, layer2_bn_ed], axis=5)  # Concatenating along the 6th dimension
    layer2_maxout = Lambda(lambda x: K.max(x, axis=5), name='maxout1')(layer_1_2_bn_ed_concat)  # Taking max along the 6th dimension

    layer2_activation = LeakyReLU()(layer2_maxout)  # Should be PreLU

    # Convolution block 3
    layer3_conv = Conv3D(n_filters, (1, 1, 1), padding=padding, strides=strides)(layer2_activation)  # Original network has "valid" padding
    layer3_bn = BatchNormalization(axis=1)(layer3_conv)

    # MaxPool
    layer4_maxpool = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(layer3_bn)

    return layer4_maxpool


def CompetitiveDenseBlock(input_layer, n_filters, kernel=(3, 3, 3), padding='same', strides=(1, 1, 1)):
    """
        * CompetitiveDenseBlock's computational Graph
        {in (Conv - BN - Maxpool from prev. block) -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} x 2 -> {Conv -> BN} -> CDB_out
        end with batch-normed output to allow maxout across skip-connections
    """

    # Activation from pooled input
    layer0_activation = LeakyReLU()(input_layer)  # Should be PreLU

    # Convolution block1 [with First Maxout]
    layer1_conv = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer0_activation)
    layer1_bn = BatchNormalization(axis=1)(layer1_conv)

    layer0_bn_ed = Lambda(lambda x: K.expand_dims(x, -1))(input_layer)  # Add Singleton Dimension along 6th
    layer1_bn_ed = Lambda(lambda x: K.expand_dims(x, -1))(layer1_bn)  # Add Singleton Dimension along 6th
    layer_0_1_bn_ed_concat = concatenate([layer0_bn_ed, layer1_bn_ed], axis=5)  # Concatenating along the 6th dimension
    layer1_maxout = Lambda(lambda x: K.max(x, axis=5))(layer_0_1_bn_ed_concat)  # Taking max along the 6th dimension

    layer1_activation = LeakyReLU()(layer1_maxout)  # Should be PreLU

    # Convolution block2 [with Second Maxout]
    layer2_conv = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer1_activation)
    layer2_bn = BatchNormalization(axis=1)(layer2_conv)

    layer1_maxout_ed = Lambda(lambda x: K.expand_dims(x, -1))(layer1_maxout)  # Add Singleton Dimension along 6th
    layer2_bn_ed = Lambda(lambda x: K.expand_dims(x, -1))(layer2_bn)  # Add Singleton Dimension along 6th
    layer_1_2_bn_ed_concat = concatenate([layer1_maxout_ed, layer2_bn_ed], axis=5)  # Concatenating along the 6th dimension
    layer2_maxout = Lambda(lambda x: K.max(x, axis=5))(layer_1_2_bn_ed_concat)  # Taking max along the 6th dimension

    layer2_activation = LeakyReLU()(layer2_maxout)  # Should be PreLU

    # Convolution block 3
    layer3_conv = Conv3D(n_filters, (1, 1, 1), padding=padding, strides=strides)(layer2_activation)  # Original network has "valid" padding
    layer3_bn = BatchNormalization(axis=1)(layer3_conv)

    return layer3_bn


def CompetitiveEncoderBlock(input_layer, n_filters, kernel=(3, 3, 3), padding='same', strides=(1, 1, 1)):
    """
        Encoder Block = CompetitiveDenseBlock + Max Pooling
        * CompetitiveDenseBlock's computational Graph
        {in (Conv - BN - Maxpool from prev. block) -> PReLU} -> {Conv -> BN -> Maxout -> PReLU} x 2 -> {Conv -> BN} -> CDB_out
        end with batch-normed output to allow maxout across skip-connections

        * Max Pooling
        CDB_out -> {MaxPool3D} -> out
    """

    # Activation from pooled input
    layer0_activation = LeakyReLU()(input_layer)  # Should be PreLU

    # Convolution block1 [with First Maxout]
    layer1_conv = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer0_activation)
    layer1_bn = BatchNormalization(axis=1)(layer1_conv)

    layer0_bn_ed = Lambda(lambda x: K.expand_dims(x, -1))(input_layer)  # Add Singleton Dimension along 6th
    layer1_bn_ed = Lambda(lambda x: K.expand_dims(x, -1))(layer1_bn)  # Add Singleton Dimension along 6th
    layer_0_1_bn_ed_concat = concatenate([layer0_bn_ed, layer1_bn_ed], axis=5)  # Concatenating along the 6th dimension
    layer1_maxout = Lambda(lambda x: K.max(x, axis=5))(layer_0_1_bn_ed_concat)  # Taking max along the 6th dimension

    layer1_activation = LeakyReLU()(layer1_maxout)  # Should be PreLU

    # Convolution block2 [with Second Maxout]
    layer2_conv = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer1_activation)
    layer2_bn = BatchNormalization(axis=1)(layer2_conv)

    layer1_maxout_ed = Lambda(lambda x: K.expand_dims(x, -1))(layer1_maxout)  # Add Singleton Dimension along 6th
    layer2_bn_ed = Lambda(lambda x: K.expand_dims(x, -1))(layer2_bn)  # Add Singleton Dimension along 6th
    layer_1_2_bn_ed_concat = concatenate([layer1_maxout_ed, layer2_bn_ed], axis=5)  # Concatenating along the 6th dimension
    layer2_maxout = Lambda(lambda x: K.max(x, axis=5))(layer_1_2_bn_ed_concat)  # Taking max along the 6th dimension

    layer2_activation = LeakyReLU()(layer2_maxout)  # Should be PreLU

    # Convolution block 3
    layer3_conv = Conv3D(n_filters, (1, 1, 1), padding=padding, strides=strides)(layer2_activation)  # Original network has "valid" padding
    layer3_bn = BatchNormalization(axis=1)(layer3_conv)

    # MaxPool
    layer4_maxpool = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(layer3_bn)

    return layer4_maxpool


def fastsurfer_classification(input_shape, nb_classes, optimizer=Adam, initial_learning_rate=0.0005, loss_function='categorical_crossentropy'):

    print("[MODEL] Training using fastsurfer_classification")

    inputs = Input(input_shape)

    # Encoding / Descending Arm
    num_filters = 32
    encoder_output1 = CompetitiveEncoderBlockInput(inputs, num_filters)
    encoder_output2 = CompetitiveEncoderBlock(encoder_output1, num_filters)
    encoder_output3 = CompetitiveEncoderBlock(encoder_output2, num_filters)
    encoder_output4 = CompetitiveEncoderBlock(encoder_output3, num_filters)
    bottleneck = CompetitiveDenseBlock(encoder_output4, num_filters)
        
    clsfctn_op_GAP = GlobalAveragePooling3D()(bottleneck)
    clsfctn_Dense = Dense(nb_classes, name="Dense_without_softmax")(clsfctn_op_GAP)
    clsfctn_op = Activation('softmax')(clsfctn_Dense)

    model = Model(inputs=inputs, outputs=clsfctn_op)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
