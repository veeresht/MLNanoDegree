
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
