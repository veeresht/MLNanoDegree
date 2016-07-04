from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


class FCNN(object):

    def __init__(self, model_params, input_dim):

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

        # Layer 1
        self.model.add(Dense(64,
                             input_dim=self.input_dim,
                             W_regularizer=l2(self.reg_factor),
                             init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        # Layer 2
        self.model.add(Dense(128,
                             W_regularizer=l2(self.reg_factor),
                             init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))

        # Layer 3
        self.model.add(Dense(256,
                             W_regularizer=l2(self.reg_factor),
                             init=self.init))
        if self.use_batchnorm:
            self.model.add(BatchNormalization(mode=0, axis=1))
        self.model.add(Activation('relu'))
        if self.use_dropout:
            self.model.add(Dropout(self.dropout_param))
        self.model.add(Dense(10,
                             W_regularizer=l2(self.reg_factor),
                             init=self.init))
        self.model.add(Activation('softmax'))

        if verbose == 1:
            self.model.summary()

        return self.model

    def save(self):
        json_string = self.model.to_json()
        open('./architecture.json', 'w').write(json_string)
        self.model.save_weights('./weights.h5')
