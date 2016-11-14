
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, BatchNormalization
from keras.layers import MaxPooling2D, Activation, Flatten, Dropout
from keras.regularizers import l2


class CNN_B(object):
    """ Custom Architecture based on VGGNet """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'CNN_B'
        # Weight initialization type
        self.init = model_params['init']
        # Dropout parameters
        self.use_dropout = model_params['use_dropout']
        self.dropout_param = model_params['dropout_param']
        # L2 regularization factor for linear weights
        self.reg_factor = model_params['reg_factor']
        # Use batchnorm ?
        self.use_batchnorm = model_params['use_batchnorm']
        # Feature dimension
        self.input_dim = input_dim

    def define(self, verbose=0):

        input_image = Input(shape=self.input_dim, name='input_image')

        # Conv-Relu-MaxPool Layer
        conv11 = Convolution2D(8, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(input_image)
        if self.use_batchnorm:
            conv11 = BatchNormalization(mode=2, axis=1)(conv11)
        conv11 = Activation('relu')(conv11)
        conv11 = MaxPooling2D(pool_size=(2, 2))(conv11)

        # Conv-Relu-MaxPool Layer
        conv21 = Convolution2D(16, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv11)
        if self.use_batchnorm:
            conv21 = BatchNormalization(mode=2, axis=1)(conv21)
        conv21 = Activation('relu')(conv21)
        conv21 = MaxPooling2D(pool_size=(2, 2))(conv21)

        # Conv-Relu Layer
        conv31 = Convolution2D(16, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv21)
        if self.use_batchnorm:
            conv31 = BatchNormalization(mode=2, axis=1)(conv31)
        conv31 = Activation('relu')(conv31)

        # Conv-Relu Layer
        conv32 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv31)
        if self.use_batchnorm:
            conv32 = BatchNormalization(mode=2, axis=1)(conv32)
        conv32 = Activation('relu')(conv32)

        # Conv-Relu-MaxPool Layer
        conv33 = Convolution2D(32, 1, 1, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv32)
        if self.use_batchnorm:
            conv33 = BatchNormalization(mode=2, axis=1)(conv33)
        conv33 = Activation('relu')(conv33)
        conv33 = MaxPooling2D(pool_size=(2, 2))(conv33)

        # Conv-Relu Layer
        conv41 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv33)
        if self.use_batchnorm:
            conv41 = BatchNormalization(mode=2, axis=1)(conv41)
        conv41 = Activation('relu')(conv41)

        # Conv-Relu Layer
        conv42 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv41)
        if self.use_batchnorm:
            conv42 = BatchNormalization(mode=2, axis=1)(conv42)
        conv42 = Activation('relu')(conv42)

        # Conv-Relu-MaxPool Layer
        conv43 = Convolution2D(64, 1, 1, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv42)
        if self.use_batchnorm:
            conv43 = BatchNormalization(mode=2, axis=1)(conv43)
        conv43 = Activation('relu')(conv43)
        conv43 = MaxPooling2D(pool_size=(2, 2))(conv43)

        conv_features = Flatten()(conv43)

        # Affine-Relu-[Dropout] Layer
        dense1 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features)
        if self.use_batchnorm:
            dense1 = BatchNormalization(mode=1)(dense1)
        dense1 = Activation('relu')(dense1)
        if self.use_dropout:
            dense1 = Dropout(self.dropout_param)(dense1)

        digit1_pred = Dense(10, activation='softmax', name='digit1')(dense1)
        digit2_pred = Dense(10, activation='softmax', name='digit2')(dense1)
        digit3_pred = Dense(10, activation='softmax', name='digit3')(dense1)
        digit4_pred = Dense(10, activation='softmax', name='digit4')(dense1)
        digit5_pred = Dense(10, activation='softmax', name='digit5')(dense1)
        length_pred = Dense(7, activation='softmax', name='length')(dense1)

        self.model = Model(input=input_image, output=[length_pred, digit1_pred,
                                                      digit2_pred, digit3_pred,
                                                      digit4_pred, digit5_pred])

        if verbose == 1:
            self.model.summary()

class VGGNetMod_2(object):
    """ Custom Architecture based on VGGNet """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'VGGNetMod_2'
        # Weight initialization type
        self.init = model_params['init']
        # Dropout parameters
        self.use_dropout = model_params['use_dropout']
        self.dropout_param = model_params['dropout_param']
        # L2 regularization factor for linear weights
        self.reg_factor = model_params['reg_factor']
        # Use batchnorm ?
        self.use_batchnorm = model_params['use_batchnorm']
        # Feature dimension
        self.input_dim = input_dim

    def define(self, verbose=0):

        input_image = Input(shape=self.input_dim, name='input_image')

        # Conv-Relu-MaxPool Layer
        conv11 = Convolution2D(16, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(input_image)
        if self.use_batchnorm:
            conv11 = BatchNormalization(mode=2, axis=1)(conv11)
        conv11 = Activation('relu')(conv11)
        conv11 = MaxPooling2D(pool_size=(2, 2))(conv11)

        # Conv-Relu-MaxPool Layer
        conv21 = Convolution2D(16, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv11)
        if self.use_batchnorm:
            conv21 = BatchNormalization(mode=2, axis=1)(conv21)
        conv21 = Activation('relu')(conv21)
        conv21 = MaxPooling2D(pool_size=(2, 2))(conv21)

        # Conv-Relu Layer
        conv31 = Convolution2D(16, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv21)
        if self.use_batchnorm:
            conv31 = BatchNormalization(mode=2, axis=1)(conv31)
        conv31 = Activation('relu')(conv31)

        # Conv-Relu Layer
        conv32 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv31)
        if self.use_batchnorm:
            conv32 = BatchNormalization(mode=2, axis=1)(conv32)
        conv32 = Activation('relu')(conv32)

        # Conv-Relu-MaxPool Layer
        conv33 = Convolution2D(32, 1, 1, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv32)
        if self.use_batchnorm:
            conv33 = BatchNormalization(mode=2, axis=1)(conv33)
        conv33 = Activation('relu')(conv33)
        conv33 = MaxPooling2D(pool_size=(2, 2))(conv33)

        # Conv-Relu Layer
        conv41 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv33)
        if self.use_batchnorm:
            conv41 = BatchNormalization(mode=2, axis=1)(conv41)
        conv41 = Activation('relu')(conv41)

        # Conv-Relu Layer
        conv42 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv41)
        if self.use_batchnorm:
            conv42 = BatchNormalization(mode=2, axis=1)(conv42)
        conv42 = Activation('relu')(conv42)

        # Conv-Relu-MaxPool Layer
        conv43 = Convolution2D(64, 1, 1, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv42)
        if self.use_batchnorm:
            conv43 = BatchNormalization(mode=2, axis=1)(conv43)
        conv43 = Activation('relu')(conv43)
        conv43 = MaxPooling2D(pool_size=(2, 2))(conv43)

        conv_features = Flatten()(conv43)

        # Affine-Relu-[Dropout] Layer
        dense1 = Dense(1024,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features)
        if self.use_batchnorm:
            dense1 = BatchNormalization(mode=1)(dense1)
        dense1 = Activation('relu')(dense1)
        if self.use_dropout:
            dense1 = Dropout(self.dropout_param)(dense1)


        # Affine-Relu-[Dropout] Layer
        dense2 = Dense(1024,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(dense1)
        if self.use_batchnorm:
            dense2 = BatchNormalization(mode=1)(dense2)
        dense2 = Activation('relu')(dense2)
        if self.use_dropout:
            dense2 = Dropout(self.dropout_param)(dense2)

        digit1_pred = Dense(10, activation='softmax', name='digit1')(dense2)
        digit2_pred = Dense(10, activation='softmax', name='digit2')(dense2)
        digit3_pred = Dense(10, activation='softmax', name='digit3')(dense2)
        digit4_pred = Dense(10, activation='softmax', name='digit4')(dense2)
        digit5_pred = Dense(10, activation='softmax', name='digit5')(dense2)
        length_pred = Dense(7, activation='softmax', name='length')(dense2)

        self.model = Model(input=input_image, output=[length_pred, digit1_pred,
                                                      digit2_pred, digit3_pred,
                                                      digit4_pred, digit5_pred])

        if verbose == 1:
            self.model.summary()


class DigitConvNet(object):
    """ Custom Architecture based on VGGNet """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'DigitConvNet'
        # Weight initialization type
        self.init = model_params['init']
        # Dropout parameters
        self.use_dropout = model_params['use_dropout']
        self.dropout_param = model_params['dropout_param']
        # L2 regularization factor for linear weights
        self.reg_factor = model_params['reg_factor']
        # Use batchnorm ?
        self.use_batchnorm = model_params['use_batchnorm']
        # Feature dimension
        self.input_dim = input_dim

    def define(self, verbose=0):

        input_image = Input(shape=self.input_dim, name='input_image')

        # Conv-Relu-MaxPool Layer
        conv11 = Convolution2D(8, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(input_image)
        if self.use_batchnorm:
            conv11 = BatchNormalization(mode=2, axis=1)(conv11)
        conv11 = Activation('relu')(conv11)
        conv11 = MaxPooling2D(pool_size=(2, 2))(conv11)

        # Conv-Relu-MaxPool Layer
        conv21 = Convolution2D(16, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv11)
        if self.use_batchnorm:
            conv21 = BatchNormalization(mode=2, axis=1)(conv21)
        conv21 = Activation('relu')(conv21)
        conv21 = MaxPooling2D(pool_size=(2, 2))(conv21)

        # Conv-Relu Layer
        conv31 = Convolution2D(16, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv21)
        if self.use_batchnorm:
            conv31 = BatchNormalization(mode=2, axis=1)(conv31)
        conv31 = Activation('relu')(conv31)

        # Conv-Relu Layer
        conv32 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv31)
        if self.use_batchnorm:
            conv32 = BatchNormalization(mode=2, axis=1)(conv32)
        conv32 = Activation('relu')(conv32)

        # Conv-Relu-MaxPool Layer
        conv33 = Convolution2D(32, 1, 1, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv32)
        if self.use_batchnorm:
            conv33 = BatchNormalization(mode=2, axis=1)(conv33)
        conv33 = Activation('relu')(conv33)
        conv33 = MaxPooling2D(pool_size=(2, 2))(conv33)

        # Conv-Relu Layer
        conv41 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv33)
        if self.use_batchnorm:
            conv41 = BatchNormalization(mode=2, axis=1)(conv41)
        conv41 = Activation('relu')(conv41)

        # Conv-Relu Layer
        conv42 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv41)
        if self.use_batchnorm:
            conv42 = BatchNormalization(mode=2, axis=1)(conv42)
        conv42 = Activation('relu')(conv42)

        # Conv-Relu-MaxPool Layer
        conv43 = Convolution2D(64, 1, 1, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv42)
        if self.use_batchnorm:
            conv43 = BatchNormalization(mode=2, axis=1)(conv43)
        conv43 = Activation('relu')(conv43)
        conv43 = MaxPooling2D(pool_size=(2, 2))(conv43)


        # Conv-Relu Layer
        conv51 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv43)
        if self.use_batchnorm:
            conv51 = BatchNormalization(mode=2, axis=1)(conv51)
        conv51 = Activation('relu')(conv51)

        # Conv-Relu Layer
        conv52 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv43)
        if self.use_batchnorm:
            conv52 = BatchNormalization(mode=2, axis=1)(conv52)
        conv52 = Activation('relu')(conv52)

        # Conv-Relu Layer
        conv53 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv43)
        if self.use_batchnorm:
            conv53 = BatchNormalization(mode=2, axis=1)(conv53)
        conv53 = Activation('relu')(conv53)

        # Conv-Relu Layer
        conv54 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv43)
        if self.use_batchnorm:
            conv54 = BatchNormalization(mode=2, axis=1)(conv54)
        conv54 = Activation('relu')(conv54)

        # Conv-Relu Layer
        conv55 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv43)
        if self.use_batchnorm:
            conv55 = BatchNormalization(mode=2, axis=1)(conv55)
        conv55 = Activation('relu')(conv55)

        # Conv-Relu Layer
        conv56 = Convolution2D(64, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv43)
        if self.use_batchnorm:
            conv56 = BatchNormalization(mode=2, axis=1)(conv56)
        conv56 = Activation('relu')(conv56)


        conv_features_1 = Flatten()(conv51)
        conv_features_2 = Flatten()(conv52)
        conv_features_3 = Flatten()(conv53)
        conv_features_4 = Flatten()(conv54)
        conv_features_5 = Flatten()(conv55)
        conv_features_6 = Flatten()(conv56)

        # Affine-Relu-[Dropout] Layer
        dense1 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features_1)
        if self.use_batchnorm:
            dense1 = BatchNormalization(mode=1)(dense1)
        dense1 = Activation('relu')(dense1)
        if self.use_dropout:
            dense1 = Dropout(self.dropout_param)(dense1)

        # Affine-Relu-[Dropout] Layer
        dense2 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features_2)
        if self.use_batchnorm:
            dense2 = BatchNormalization(mode=1)(dense2)
        dense2 = Activation('relu')(dense2)
        if self.use_dropout:
            dense2 = Dropout(self.dropout_param)(dense2)

        # Affine-Relu-[Dropout] Layer
        dense3 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features_3)
        if self.use_batchnorm:
            dense3 = BatchNormalization(mode=1)(dense3)
        dense3 = Activation('relu')(dense3)
        if self.use_dropout:
            dense3 = Dropout(self.dropout_param)(dense3)

        # Affine-Relu-[Dropout] Layer
        dense4 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features_4)
        if self.use_batchnorm:
            dense4 = BatchNormalization(mode=1)(dense4)
        dense4 = Activation('relu')(dense4)
        if self.use_dropout:
            dense4 = Dropout(self.dropout_param)(dense4)

        # Affine-Relu-[Dropout] Layer
        dense5 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features_5)
        if self.use_batchnorm:
            dense5 = BatchNormalization(mode=1)(dense5)
        dense5 = Activation('relu')(dense5)
        if self.use_dropout:
            dense5 = Dropout(self.dropout_param)(dense5)

        # Affine-Relu-[Dropout] Layer
        dense6 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features_6)
        if self.use_batchnorm:
            dense6 = BatchNormalization(mode=1)(dense6)
        dense6 = Activation('relu')(dense6)
        if self.use_dropout:
            dense6 = Dropout(self.dropout_param)(dense6)

        digit1_pred = Dense(10, activation='softmax', name='digit1')(dense1)
        digit2_pred = Dense(10, activation='softmax', name='digit2')(dense2)
        digit3_pred = Dense(10, activation='softmax', name='digit3')(dense3)
        digit4_pred = Dense(10, activation='softmax', name='digit4')(dense4)
        digit5_pred = Dense(10, activation='softmax', name='digit5')(dense5)
        length_pred = Dense(7, activation='softmax', name='length')(dense6)

        self.model = Model(input=input_image, output=[length_pred, digit1_pred,
                                                      digit2_pred, digit3_pred,
                                                      digit4_pred, digit5_pred])

        if verbose == 1:
            self.model.summary()



class LeNet5Mod(object):
    """ Modified LeNet-5 Architecture """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'LeNet5Mod'
        # Weight initialization type
        self.init = model_params['init']
        # Dropout parameters
        self.use_dropout = model_params['use_dropout']
        self.dropout_param = model_params['dropout_param']
        # L2 regularization factor for linear weights
        self.reg_factor = model_params['reg_factor']
        # Use batchnorm ?
        self.use_batchnorm = model_params['use_batchnorm']
        # Feature dimension
        self.input_dim = input_dim

    def define(self, verbose=0):

        input_image = Input(shape=self.input_dim, name='input_image')

        # Conv-Relu-MaxPool Layer
        conv11 = Convolution2D(6, 5, 5, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(input_image)
        if self.use_batchnorm:
            conv11 = BatchNormalization(mode=2, axis=1)(conv11)
        conv11 = Activation('relu')(conv11)
        conv11 = MaxPooling2D(pool_size=(2, 2))(conv11)

        # Conv-Relu-MaxPool Layer
        conv21 = Convolution2D(16, 5, 5, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv11)
        if self.use_batchnorm:
            conv21 = BatchNormalization(mode=2, axis=1)(conv21)
        conv21 = Activation('relu')(conv21)
        conv21 = MaxPooling2D(pool_size=(2, 2))(conv21)

        # Conv-Relu Layer
        conv31 = Convolution2D(120, 6, 6, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv21)
        if self.use_batchnorm:
            conv31 = BatchNormalization(mode=2, axis=1)(conv31)
        conv31 = Activation('relu')(conv31)
        conv31 = MaxPooling2D(pool_size=(2, 2))(conv31)

        conv_features = Flatten()(conv31)

        # Affine-Relu-[Dropout] Layer
        dense1 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features)
        if self.use_batchnorm:
            dense1 = BatchNormalization(mode=1)(dense1)
        dense1 = Activation('relu')(dense1)
        if self.use_dropout:
            dense1 = Dropout(self.dropout_param)(dense1)

        # Affine-Relu Layer
        dense2 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(dense1)
        if self.use_batchnorm:
            dense2 = BatchNormalization(mode=1)(dense2)
        dense2 = Activation('relu')(dense2)

        digit1_pred = Dense(10, activation='softmax', name='digit1')(dense2)
        digit2_pred = Dense(10, activation='softmax', name='digit2')(dense2)
        digit3_pred = Dense(10, activation='softmax', name='digit3')(dense2)
        digit4_pred = Dense(10, activation='softmax', name='digit4')(dense2)
        digit5_pred = Dense(10, activation='softmax', name='digit5')(dense2)
        length_pred = Dense(7, activation='softmax', name='length')(dense2)

        self.model = Model(input=input_image, output=[length_pred, digit1_pred,
                                                      digit2_pred, digit3_pred,
                                                      digit4_pred, digit5_pred])

        if verbose == 1:
            self.model.summary()


class SermanetNet(object):
    """ 16x5x5 - 512x7x7 - 20 - 10 ConvNet used in
        - P. Sermanet, S. Chintala, and Y. LeCun. Convolutional Neural Networks
          Applied to House Numbers Digit Classification. 2012.
    """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'SermanetNet'
        # Weight initialization type
        self.init = model_params['init']
        # Dropout parameters
        self.use_dropout = model_params['use_dropout']
        self.dropout_param = model_params['dropout_param']
        # L2 regularization factor for linear weights
        self.reg_factor = model_params['reg_factor']
        # Use batchnorm ?
        self.use_batchnorm = model_params['use_batchnorm']
        # Feature dimension
        self.input_dim = input_dim

    def define(self, verbose=0):

        input_image = Input(shape=self.input_dim, name='input_image')


        # Conv-Relu-MaxPool Layer
        conv11 = Convolution2D(16, 5, 5, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(input_image)
        if self.use_batchnorm:
            conv11 = BatchNormalization(mode=2, axis=1)(conv11)
        conv11 = Activation('relu')(conv11)
        conv11 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                              border_mode='same')(conv11)


        # Conv-Relu-MaxPool Layer
        conv21 = Convolution2D(512, 7, 7, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv11)
        if self.use_batchnorm:
            conv21 = BatchNormalization(mode=2, axis=1)(conv21)
        conv21 = Activation('relu')(conv21)
        conv21 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                              border_mode='same')(conv21)

        conv_features = Flatten()(conv21)

        # Affine-Relu-[Dropout] Layer
        dense1 = Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv_features)
        if self.use_batchnorm:
            dense1 = BatchNormalization(mode=1)(dense1)
        dense1 = Activation('relu')(dense1)
        if self.use_dropout:
            dense1 = Dropout(self.dropout_param)(dense1)

        digit1_pred = Dense(10, activation='softmax', name='digit1')(dense1)
        digit2_pred = Dense(10, activation='softmax', name='digit2')(dense1)
        digit3_pred = Dense(10, activation='softmax', name='digit3')(dense1)
        digit4_pred = Dense(10, activation='softmax', name='digit4')(dense1)
        digit5_pred = Dense(10, activation='softmax', name='digit5')(dense1)
        length_pred = Dense(7, activation='softmax', name='length')(dense1)

        self.model = Model(input=input_image, output=[length_pred, digit1_pred,
                                                      digit2_pred, digit3_pred,
                                                      digit4_pred, digit5_pred])
        if verbose == 1:
            self.model.summary()