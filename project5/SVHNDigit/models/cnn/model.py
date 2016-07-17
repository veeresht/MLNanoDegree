from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


class CNN_1(object):

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'CNN1'
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

        self.model = Sequential()

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(64, 3, 3, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(64, 3, 3, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-[Dropout] Layer
        self.model.add(Convolution2D(64, 3, 3, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        self.model.add(Flatten())

        # Affine-Relu-[Dropout] Layer
        self.model.add(Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        # Affine-Relu Layer
        self.model.add(Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))

        # Affine-Softmax Layer
        self.model.add(Dense(10,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        self.model.add(Activation('softmax'))
        if verbose == 1:
            self.model.summary()

    def save(self):

        json_string = self.model.to_json()
        open('./architecture.json', 'w').write(json_string)
        self.model.save_weights('./weights.h5')


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

        self.model = Sequential()

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(6, 5, 5, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(16, 5, 5, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(120, 6, 6, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        # Affine-Relu-[Dropout] Layer
        self.model.add(Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        # Affine-Relu Layer
        self.model.add(Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=1))
        self.model.add(Activation('relu'))

        # Affine-Softmax Layer
        self.model.add(Dense(10,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        self.model.add(Activation('softmax'))
        if verbose == 1:
            self.model.summary()

    def save(self):

        json_string = self.model.to_json()
        open('./architecture.json', 'w').write(json_string)
        self.model.save_weights('./weights.h5')


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

        self.model = Sequential()

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(8, 3, 3, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(16, 3, 3, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu Layer
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu Layer
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(32, 1, 1, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu Layer
        self.model.add(Convolution2D(64, 3, 3, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu Layer
        self.model.add(Convolution2D(64, 3, 3, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(64, 1, 1, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        # Affine-Relu-[Dropout] Layer
        self.model.add(Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        # Affine-Relu Layer
        self.model.add(Dense(128,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=1))
        self.model.add(Activation('relu'))

        # Affine-Softmax Layer
        self.model.add(Dense(10,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        self.model.add(Activation('softmax'))
        if verbose == 1:
            self.model.summary()

    def save(self):

        json_string = self.model.to_json()
        open('./architecture.json', 'w').write(json_string)
        self.model.save_weights('./weights.h5')


class GoogleSVHNNet(object):
    """ Google's Architecture for SVHN Digit Sequence Identification
        Incomplete.
    """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'GoogleSVHNNet'
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

        self.model = Sequential()

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(48, 5, 5, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(16, 6, 6, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(120, 6, 6, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())

        # Affine-Relu-[Dropout] Layer
        self.model.add(Dense(64,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        # Affine-Relu Layer
        self.model.add(Dense(64,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))

        # Affine-Softmax Layer
        self.model.add(Dense(10,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        self.model.add(Activation('softmax'))
        if verbose == 1:
            self.model.summary()

    def save(self):

        json_string = self.model.to_json()
        open('./architecture.json', 'w').write(json_string)
        self.model.save_weights('./weights.h5')


class HintonNet1(object):
    """ 64-64-64 5x5 ConvNet used in
        - G.E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever,
          and R. R. Salakhutdinov. Improving neural networks by
          preventing co-adaptation of feature detectors.
          arXiv:1207.0580, 2012.
        - M. D. Zeiler and R. Fergus. Stochastic Pooling for Regularization of
          Deep Convolutional Neural Networks. ICLR, 2013.
    """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'HintonNet1'
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

        self.model = Sequential()

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(64, 5, 5, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                    border_mode='same'))

        # Conv-Relu-AvgPool Layer
        self.model.add(Convolution2D(64, 5, 5, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                                        border_mode='same'))

        # Conv-Relu-AvgPool Layer
        self.model.add(Convolution2D(64, 5, 5, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2),
                                        border_mode='same'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        self.model.add(Flatten())

        # Affine-Softmax Layer
        self.model.add(Dense(10,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        self.model.add(Activation('softmax'))
        if verbose == 1:
            self.model.summary()

    def save(self):

        json_string = self.model.to_json()
        open('./architecture.json', 'w').write(json_string)
        self.model.save_weights('./weights.h5')


class SermanetNet(object):
    """ 16x5x5 - 512x7x7 - 20 - 10 ConvNet used in
        - P. Sermanet, S. Chintala, and Y. LeCun. Convolutional Neural Networks
          Applied to House Numbers Digit Classification. 2012.
    """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'HintonNet1'
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

        self.model = Sequential()

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(16, 5, 5, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        # if self.use_batchnorm:
        #    self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                    border_mode='same'))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(512, 7, 7, border_mode='same',
                                     input_shape=self.input_dim,
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                    border_mode='same'))

        self.model.add(Flatten())

        # Affine-Softmax Layer
        self.model.add(Dense(20,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        # Affine-Softmax Layer
        self.model.add(Dense(10,
                       W_regularizer=l2(self.reg_factor),
                       init=self.init))
        self.model.add(Activation('softmax'))
        if verbose == 1:
            self.model.summary()

    def save(self):

        json_string = self.model.to_json()
        open('./architecture.json', 'w').write(json_string)
        self.model.save_weights('./weights.h5')