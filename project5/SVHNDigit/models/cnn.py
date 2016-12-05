from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, merge
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


class VGGNetMod_1(object):
    """ Custom Architecture based on VGGNet """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'VGGNetMod_1'
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
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(1, 1),
                                    border_mode='same'))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(16, 3, 3, border_mode='same',
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(1, 1),
                                    border_mode='same'))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(16, 3, 3, border_mode='same',
                                     W_regularizer=l2(self.reg_factor),
                                     init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(1, 1),
                                    border_mode='same'))

        # Conv-Relu-MaxPool Layer
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
                       W_regularizer=l2(self.reg_factor),
                       init=self.init, subsample=(1, 1)))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    border_mode='same'))

        # Conv-Relu-MaxPool Layer
        # self.model.add(Convolution2D(32, 1, 1, border_mode='same',
        #                W_regularizer=l2(self.reg_factor),
        #                init=self.init, subsample=(1, 1)))
        # if self.use_batchnorm:
        #     self.model.add(BatchNormalization(mode=0, axis=1))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2),
        #                             strides=(1, 1),
        #                             border_mode='same'))

        # Conv-Relu Layer
        self.model.add(Convolution2D(32, 3, 3, border_mode='same',
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
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    border_mode='same'))


        # Conv-Relu-MaxPool Layer
        # self.model.add(Convolution2D(64, 1, 1, border_mode='same',
        #                              W_regularizer=l2(self.reg_factor),
        #                              init=self.init, subsample=(1, 1)))
        # if self.use_batchnorm:
        #     self.model.add(BatchNormalization(mode=0, axis=1))
        # self.model.add(Activation('relu'))

        #self.model.add(MaxPooling2D(pool_size=(2, 2)))

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


class InceptionNet(object):
    """ Custom convnet based on GoogLeNet type of architecture
        using Inception-v1 modules """

    def __init__(self, model_params, input_dim):

        # Name of the model
        self.name = 'InceptionNet'
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
        conv11 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(input_image)
        if self.use_batchnorm:
            conv11 = BatchNormalization(mode=0, axis=1)(conv11)
        conv11 = Activation('relu')(conv11)
        conv11 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                              border_mode='same')(conv11)
        conv11 = MaxPooling2D(pool_size=(2, 2))(conv11)

        # Conv-Relu-MaxPool Layer
        conv21 = Convolution2D(32, 3, 3, border_mode='same',
                               W_regularizer=l2(self.reg_factor),
                               init=self.init, subsample=(1, 1))(conv11)
        if self.use_batchnorm:
            conv21 = BatchNormalization(mode=0, axis=1)(conv21)
        conv21 = Activation('relu')(conv21)
        conv21 = MaxPooling2D(pool_size=(2, 2))(conv21)

        #print conv21.get_shape()

        # Inception-v1 Layer
        conv1x1 = Convolution2D(32, 1, 1, border_mode='same',
                                W_regularizer=l2(self.reg_factor),
                                init=self.init, subsample=(1, 1))(conv21)
        if self.use_batchnorm:
            conv1x1 = BatchNormalization(mode=0, axis=1)(conv1x1)
        conv1x1 = Activation('relu')(conv1x1)
        #conv1x1 = MaxPooling2D(pool_size=(2, 2))(conv1x1)

        conv1x1dbl = Convolution2D(32, 1, 1, border_mode='same',
                                   W_regularizer=l2(self.reg_factor),
                                   init=self.init, subsample=(1, 1))(conv21)
        if self.use_batchnorm:
            conv1x1dbl = BatchNormalization(mode=0, axis=1)(conv1x1dbl)
        conv1x1dbl = Activation('relu')(conv1x1dbl)

        conv3x3dbl = Convolution2D(32, 3, 3, border_mode='same',
                                   W_regularizer=l2(self.reg_factor),
                                   init=self.init, subsample=(1, 1))(conv1x1dbl)
        if self.use_batchnorm:
            conv3x3dbl = BatchNormalization(mode=0, axis=1)(conv3x3dbl)
        conv3x3dbl = Activation('relu')(conv3x3dbl)

        conv1x1dbl2 = Convolution2D(32, 1, 1, border_mode='same',
                                    W_regularizer=l2(self.reg_factor),
                                    init=self.init, subsample=(1, 1))(conv21)
        if self.use_batchnorm:
            conv1x1dbl2 = BatchNormalization(mode=0, axis=1)(conv1x1dbl2)
        conv1x1dbl2 = Activation('relu')(conv1x1dbl2)

        conv5x5dbl = Convolution2D(32, 5, 5, border_mode='same',
                                   W_regularizer=l2(self.reg_factor),
                                   init=self.init, subsample=(1, 1))(conv1x1dbl2)
        if self.use_batchnorm:
            conv5x5dbl = BatchNormalization(mode=0, axis=1)(conv5x5dbl)
        conv5x5dbl = Activation('relu')(conv5x5dbl)

        branch_pool3x3dbl = MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                                         border_mode='same')(conv21)
        conv1x1dbl3 = Convolution2D(32, 1, 1, border_mode='same',
                                    W_regularizer=l2(self.reg_factor),
                                    init=self.init, subsample=(1, 1))(branch_pool3x3dbl)
        if self.use_batchnorm:
            conv1x1dbl3 = BatchNormalization(mode=0, axis=1)(conv1x1dbl3)
        conv1x1dbl3 = Activation('relu')(conv1x1dbl3)

        inception1 = merge([conv1x1, conv3x3dbl, conv5x5dbl, conv1x1dbl3],
                           mode='concat', concat_axis=1)

        # Inception-v1 Layer
        # conv1x1 = Convolution2D(32, 1, 1, border_mode='same',
        #                         W_regularizer=l2(self.reg_factor),
        #                         init=self.init, subsample=(1, 1))(inception1)
        # if self.use_batchnorm:
        #     conv1x1 = BatchNormalization(mode=0, axis=1)(conv1x1)
        # conv1x1 = Activation('relu')(conv1x1)

        # conv1x1dbl = Convolution2D(32, 1, 1, border_mode='same',
        #                            W_regularizer=l2(self.reg_factor),
        #                            init=self.init, subsample=(1, 1))(inception1)
        # if self.use_batchnorm:
        #     conv1x1dbl = BatchNormalization(mode=0, axis=1)(conv1x1dbl)
        # conv1x1dbl = Activation('relu')(conv1x1dbl)

        # conv3x3dbl = Convolution2D(32, 3, 3, border_mode='same',
        #                            W_regularizer=l2(self.reg_factor),
        #                            init=self.init, subsample=(1, 1))(conv1x1dbl)
        # if self.use_batchnorm:
        #     conv3x3dbl = BatchNormalization(mode=0, axis=1)(conv3x3dbl)
        # conv3x3dbl = Activation('relu')(conv3x3dbl)

        # conv1x1dbl2 = Convolution2D(32, 1, 1, border_mode='same',
        #                             W_regularizer=l2(self.reg_factor),
        #                             init=self.init, subsample=(1, 1))(inception1)
        # if self.use_batchnorm:
        #     conv1x1dbl2 = BatchNormalization(mode=0, axis=1)(conv1x1dbl2)
        # conv1x1dbl2 = Activation('relu')(conv1x1dbl2)

        # conv5x5dbl = Convolution2D(32, 5, 5, border_mode='same',
        #                            W_regularizer=l2(self.reg_factor),
        #                            init=self.init, subsample=(1, 1))(conv1x1dbl2)
        # if self.use_batchnorm:
        #     conv5x5dbl = BatchNormalization(mode=0, axis=1)(conv5x5dbl)
        # conv5x5dbl = Activation('relu')(conv5x5dbl)

        # branch_pool3x3dbl = MaxPooling2D(pool_size=(3, 3))(inception1)
        # conv1x1dbl3 = Convolution2D(32, 1, 1, border_mode='same',
        #                             W_regularizer=l2(self.reg_factor),
        #                             init=self.init, subsample=(1, 1))(branch_pool3x3dbl)
        # if self.use_batchnorm:
        #     conv1x1dbl3 = BatchNormalization(mode=0, axis=1)(conv1x1dbl3)
        # conv1x1dbl3 = Activation('relu')(conv1x1dbl3)

        # inception2 = merge([conv1x1, conv3x3dbl, conv5x5dbl, conv1x1dbl3],
        #                    mode='concat', concat_axis=1)

        avgpool = AveragePooling2D(pool_size=(2, 2))(inception1)
        conv1x1out = Convolution2D(32, 1, 1, border_mode='same',
                                   W_regularizer=l2(self.reg_factor),
                                   init=self.init, subsample=(1, 1))(avgpool)
        if self.use_batchnorm:
            conv1x1out = BatchNormalization(mode=0, axis=1)(conv1x1out)
        conv1x1out = Activation('relu')(conv1x1out)

        conv1x1out = Flatten()(conv1x1out)

        # Affine-Relu-[Dropout] Layer
        dense1 = Dense(512, W_regularizer=l2(self.reg_factor),
                       init=self.init)(conv1x1out)
        if self.use_batchnorm:
            dense1 = BatchNormalization(mode=1)(dense1)
        dense1 = Activation('relu')(dense1)
        if self.use_dropout:
            dense1 = Dropout(self.dropout_param)(dense1)

        # Affine-Relu Layer
        dense2 = Dense(1024, W_regularizer=l2(self.reg_factor),
                       init=self.init)(dense1)
        if self.use_batchnorm:
            dense2 = BatchNormalization(mode=1)(dense2)
        dense2 = Activation('relu')(dense2)

        # Affine-Softmax Layer
        prediction = Dense(10, W_regularizer=l2(self.reg_factor),
                           init=self.init)(dense2)
        prediction = Activation('softmax')(prediction)

        self.model = Model(input=input_image, output=prediction)

        if verbose == 1:
            self.model.summary()

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