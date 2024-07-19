from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Activation, add, \
    LSTM, Layer, Dense, BatchNormalization, Activation, GlobalAveragePooling1D, Multiply, Add
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


def MLSU_Net(length, n_channel=1):
    def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

        kernel = 3

        x = Conv1D(filters, kernel, padding=padding)(x)
        x = BatchNormalization()(x)

        if (activation == None):
            return x

        x = Activation(activation, name=name)(x)
        return x

    def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):

        x = UpSampling1D(size=2)(x)
        x = BatchNormalization()(x)

        return x

    def MultiResBlock(U, inp, alpha=2.5):
        """
        MultiRes Block

        Arguments:
            U {int} -- Number of filters in a corresponding UNet stage
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        """

        W = alpha * U

        shortcut = inp

        shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) +
                             int(W * 0.5), 1, 1, activation=None, padding='same')

        conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3,
                            activation='relu', padding='same')

        conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3,
                            activation='relu', padding='same')

        conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3,
                            activation='relu', padding='same')

        out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
        out = BatchNormalization()(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        return out

    def ResPath(filters, length, inp):
        """
        ResPath

        Arguments:
            filters {int} -- [description]
            length {int} -- length of ResPath
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        """
        shortcut = Conv1D(filters, 1, padding='same')(inp)
        out = conv2d_bn(inp, filters, 1, 1, activation='relu', padding='same')
        out = conv2d_bn(out, filters, 1, 1, activation=None, padding='same')
        out = LSTM(filters, return_sequences=True)(out)

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)

        for i in range(length - 1):
            shortcut = out
            shortcut = conv2d_bn(shortcut, filters, 1, 1,
                                 activation=None, padding='same')

            out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)

        return out

    inputs = Input((length, n_channel))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling1D(pool_size=2)(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32 * 2, pool1)
    pool2 = MaxPooling1D(pool_size=2)(mresblock2)
    mresblock2 = ResPath(32 * 2, 3, mresblock2)

    mresblock3 = MultiResBlock(32 * 4, pool2)
    pool3 = MaxPooling1D(pool_size=2)(mresblock3)
    mresblock3 = ResPath(32 * 4, 2, mresblock3)

    mresblock4 = MultiResBlock(32 * 8, pool3)
    pool4 = MaxPooling1D(pool_size=2)(mresblock4)
    mresblock4 = ResPath(32 * 8, 1, mresblock4)

    mresblock5 = MultiResBlock(32 * 16, pool4)

    level4 = Conv1D(1, 1, name="level4")(mresblock5)

    up6 = concatenate([UpSampling1D(size=2)(mresblock5), SKAttention(32 * 8)(mresblock4)], axis=-1)
    mresblock6 = MultiResBlock(32 * 8, up6)

    level3 = Conv1D(1, 1, name="level3")(mresblock6)

    up7 = concatenate([UpSampling1D(size=2)(mresblock6), SKAttention(32 * 4)(mresblock3)], axis=-1)
    mresblock7 = MultiResBlock(32 * 4, up7)

    level2 = Conv1D(1, 1, name="level2")(mresblock7)

    up8 = concatenate([UpSampling1D(size=2)(mresblock7), SKAttention(32 * 2)(mresblock2)], axis=-1)
    mresblock8 = MultiResBlock(32 * 2, up8)

    level1 = Conv1D(1, 1, name="level1")(mresblock8)

    up9 = concatenate([UpSampling1D(size=2)(mresblock8), SKAttention(32)(mresblock1)], axis=-1)
    mresblock9 = MultiResBlock(32, up9)

    out = Conv1D(1, 1, name="out")(mresblock9)

    model = Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])

    return model



class SKAttention(Layer):
    def __init__(self, units, reduction_ratio=4, **kwargs):
        super(SKAttention, self).__init__(**kwargs)
        self.units = units
        self.reduction_ratio = reduction_ratio

        self.spatial_conv1 = Dense(units, activation='relu')
        self.spatial_conv2 = Dense(units)
        self.spatial_act = Activation('sigmoid')

        self.channel_fc1 = Dense(units // reduction_ratio, activation='relu')
        self.channel_fc2 = Dense(units, activation='sigmoid')
        self.channel_avg_pool = GlobalAveragePooling1D()

    def call(self, inputs):
        # Spatial Attention
        spatial_attention = self.spatial_conv1(inputs)
        spatial_attention = self.spatial_conv2(spatial_attention)
        spatial_attention = self.spatial_act(spatial_attention)

        # Channel Attention
        channel_attention = self.channel_avg_pool(inputs)
        channel_attention = self.channel_fc1(channel_attention)
        channel_attention = self.channel_fc2(channel_attention)

        # Combine Spatial and Channel Attention
        output = Multiply()([inputs, spatial_attention])
        output = Add()([output, Multiply()([inputs, channel_attention])])

        return output